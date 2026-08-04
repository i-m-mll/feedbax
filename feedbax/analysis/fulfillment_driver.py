"""The fulfillment driver: one plan's closure, materialized once.

A plan (:mod:`feedbax.analysis.fulfillment_plan`) says what must exist. This
module makes it exist: it proves the closure's external boundary before anything
runs, lowers each plan node into the feedbax node request its compiled schema
executes, and walks the closure in dependency order, reusing every node whose
receipt admits and executing only the ones that are missing. Nothing here decides
*what* a node is; orchestration decides only whether it runs.

## What this module owns, and what it does not

feedbax owns receipt admission, the executor adapters, and custody semantics.
Every reuse-or-execute decision is
:func:`feedbax.analysis.fulfillment_adapters.fulfill_node`, every admission is
the uniform per-kind validator behind
:func:`~feedbax.analysis.fulfillment_adapters.admit_node`, and rebuild and repair
are :mod:`feedbax.analysis.fulfillment_custody`'s. Deriving the plan is
:mod:`feedbax.analysis.fulfillment_derivation`'s and lowering one node is
:mod:`feedbax.analysis.fulfillment_lowering`'s. This module reimplements none of
them.

What it owns is the *walk*: the order nodes are reached in, which admitted
receipt fills each declared input, and the refusals that stop the walk. There is
no caller-supplied lowering, payload preparation, or omission applier: those
existed only while the engine that produced these plans lived downstream, and
every one of them is now Feedbax code reached directly.

## The external boundary

Some producers are receipts this runner may consume but never make — a compiled
training run matrix is exactly that, launched through its own orchestration
entrypoint. Derivation marks such a node by setting
:attr:`~.fulfillment_plan.PlanNode.boundary`, and :func:`preflight` refuses the
whole closure before any node of any branch executes, naming each boundary node,
the consumers that name it with their role paths, and the subtree its receipt
would unblock.

## Omissions are honored only under a rule this build owns

An input the closure carries as ``not_applicable`` on a ``compiler_rule`` basis
is honored on the strength of the rule it quotes. The plan kernel deliberately
does not know what a rule *name* means, so :func:`preflight` proves it here,
against the closed table in :mod:`feedbax.contracts.applicability_rules`. A rule
this build cannot state certifies nothing, and the closure is refused rather than
executed with an input neither bound nor proved inapplicable.

## The plan is a derived record; the locks are the authority

A plan is durable and travels apart from the locks it came from, so every
authenticating fact it carries is a copy. :func:`preflight` re-derives each
node's edges from its own compile lock and refuses on any disagreement
(:class:`PlanLockDisagreementError`). The node content hash cannot cover this:
it pins the compiled *document*, while edges come from the *lock*, so an intact
hash sits happily beside an edge whose byte profile was removed.

## Missing receipts are refusals, and the lock says which bytes count

An input the plan carries as an already-produced external receipt is resolved at
its canonical ``(kind, id)`` location within whichever authority holds it, never
through the manifest index, which is derived acceleration. An external receipt
that is absent from every declared authority is a structured refusal
(:class:`MissingExternalReceiptError`), never an inapplicability: a missing
receipt means not-yet-produced, wrong-root, or corrupt, and each of those is its
own outcome.

## One reference, one authority

A run declares the authorities it may read from: the receipt root always, plus
whatever retained manifest stores and immutable artifact providers the caller
bound by name. They are searched together and they are not ranked. A reference
held by two of them is :class:`AmbiguousExternalReceiptError` rather than a tie
to be broken, because each authority is a separate custody domain the caller
declared deliberately, and preferring one would silently pick a domain nobody
chose. The refusal happens during resolution, before any node executes.

## The staged context is per node, and it is on the request

A node's parents are located exactly once, by the resolution that bound them,
and the resulting :class:`~feedbax.analysis.execution_context.StagedExecutionContext`
is carried on the node request rather than assembled during the walk. Ordinary
fulfillment, cached admission, rebuild-as-verification, and repair all lower each
node again, so the request is the only place a per-node context survives all four.
A run that declared no staged bindings lowers no context at all, which leaves the
cold-start path exactly as it was.

Addressing a receipt is not authenticating it. When the compile lock quoted a
byte profile — an ``authenticated_receipt`` reference — the bytes found at that
location are checked against the quote, and disagreement is
:class:`ExternalReceiptAuthenticationError`. The lock is the sole authority for
which bytes a reference names: nothing here blesses a manifest on kind, id, and
status alone, because those three are satisfied equally by a rerun that produced
different results at the same address.

Execution is strictly serial and deterministic in order, as everything on the
feedbax fulfillment surface is.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from feedbax.analysis.fulfillment import (
    AdmittedManifestBytes,
    FulfillmentAdmissionError,
    FulfillmentReceipt,
)
from feedbax.analysis.fulfillment_adapters import (
    AnalysisBundleNodeRequest,
    EvaluationMatrixNodeRequest,
    FulfillmentEnvironment,
    FulfillmentRun,
    NodeFulfillment,
    NodeRequest,
    admit_node,
    fulfill_node,
    receipt_exact_parent_entry,
    receipt_parent_ref,
)
from feedbax.analysis.fulfillment_custody import (
    RebuildRun,
    RepairResult,
    rebuild_nodes,
    repair_node,
)
from feedbax.analysis.fulfillment_derivation import (
    CompiledEnvelope,
    CompiledOutputIndex,
    ExternalReceiptRecord,
    lock_edge_declarations,
    require_external_record,
)
from feedbax.analysis.execution_context import (
    EMPTY_STAGED_EXECUTION_CONTEXT,
    StagedExecutionContext,
    StagedParentExecutionLocation,
    with_staged_repo_root,
    with_staged_resolved_parents,
)
from feedbax.analysis.fulfillment_lowering import lower_compiled_node
from feedbax.analysis.manifest_inputs import (
    canonical_staged_manifest_locator,
    is_staged_manifest_kind,
)
from feedbax.analysis.fulfillment_plan import (
    FulfillmentPlan,
    LogicalKey,
    PlanEdge,
    PlanNode,
)
from feedbax.analysis.exact_parents import StagedExactParentEntry
from feedbax.contracts.applicability_rules import (
    StructuralApplicabilityRuleMismatchError,
    UnknownStructuralApplicabilityRuleError,
    certify_structural_applicability,
)
from feedbax.contracts.manifest import (
    AnyManifest,
    ParentRef,
    authenticated_manifest_ref_metadata,
    authenticated_manifest_ref_profile,
    canonical_manifest_path,
    load_manifest_bytes,
)


class FulfillmentDriverError(RuntimeError):
    """Base class for the structured refusals this driver raises."""


@dataclass(frozen=True)
class BoundaryNode:
    """One boundary node the closure names, and what its absence blocks."""

    key: LogicalKey
    source_ref: str
    boundary: str
    named_by: tuple[tuple[str, tuple[str, ...]], ...]
    unblocks: tuple[str, ...]

    def describe(self) -> str:
        naming = ", ".join(f"{consumer} via {list(role)}" for consumer, role in self.named_by)
        return f"{self.key.text} ({self.source_ref}, boundary {self.boundary!r}) named by {naming}"

    def record(self) -> dict[str, Any]:
        return {
            **self.key.record(),
            "source_ref": self.source_ref,
            "boundary": self.boundary,
            "named_by": [
                {"consumer": consumer, "role_path": list(role)} for consumer, role in self.named_by
            ],
            "unblocks": list(self.unblocks),
        }


class ExternalBoundaryError(FulfillmentDriverError):
    """The closure needs a receipt this runner may consume but never produce.

    This is a repository-state failure rather than an authoring failure: no edit
    to a declaration produces the missing receipt. It is raised by
    :func:`preflight`, before any node of any branch of the closure executes.
    """

    def __init__(self, target: LogicalKey, nodes: Sequence[BoundaryNode]) -> None:
        self.target = target
        self.nodes = tuple(nodes)
        listing = "; ".join(node.describe() for node in self.nodes)
        super().__init__(
            f"fulfilling {target.text} requires {len(self.nodes)} receipt(s) this runner cannot "
            f"produce or resolve: {listing}. A boundary node is never executed by fulfillment: "
            "produce it through its own entrypoint, then name the produced receipt in the "
            "declaration that consumes it, so the reference is an authenticated receipt rather "
            "than a pending one."
        )

    def record(self) -> dict[str, Any]:
        """Return the structured refusal, deterministic in every field."""
        return {
            "target": self.target.text,
            "boundary_nodes": [node.record() for node in self.nodes],
        }


class MissingExternalReceiptError(FulfillmentDriverError):
    """An input the plan carries as an already-produced receipt is in no authority.

    The plan states that some previous run produced this reference; none of the
    authorities the run declared holds it. That is not-yet-copied, wrong-root, or
    deleted — never evidence that the input does not apply.
    """

    def __init__(
        self,
        kind: str,
        manifest_id: str,
        path: Path,
        *,
        consumer: LogicalKey | None = None,
        role_path: Sequence[str] = (),
        searched: Sequence[str] = (),
    ) -> None:
        self.kind = kind
        self.manifest_id = manifest_id
        self.path = path
        self.consumer = consumer
        self.role_path = tuple(role_path)
        self.searched = tuple(searched)
        origin = (
            f"{consumer.text} declares input {list(self.role_path)} as the already-produced "
            if consumer is not None
            else "the plan names the already-produced "
        )
        # One authority is the receipt root alone, and its canonical location is
        # the whole answer. Several authorities means a declared staged surface,
        # and naming only one of them would describe a search that did not happen.
        where = (
            f"canonical location {path}"
            if len(self.searched) < 2
            else f"canonical location under any declared authority ({', '.join(self.searched)})"
        )
        super().__init__(
            f"{origin}{kind} {manifest_id!r}, but no completed receipt is stored at its "
            f"{where}. A missing receipt is not-yet-produced, wrong-root, or "
            "deleted; it never means the input does not apply."
        )


class AmbiguousExternalReceiptError(FulfillmentDriverError):
    """One already-produced receipt is held by more than one declared authority.

    A reference names one artifact, so it must resolve to one authority. Two
    authorities holding a completed manifest of the same kind and id is not a
    tie to be broken: there is no precedence among a receipt root, a retained
    manifest store, and an immutable artifact provider, because each is a
    separate custody domain the caller declared deliberately. Preferring one
    would silently pick a custody domain nobody chose, and the two could differ
    in exactly the bytes an authenticated reference exists to pin.

    This refuses during resolution, before any node of the closure executes.
    """

    def __init__(
        self,
        kind: str,
        manifest_id: str,
        authorities: Sequence[str],
        *,
        consumer: LogicalKey | None = None,
        role_path: Sequence[str] = (),
    ) -> None:
        self.kind = kind
        self.manifest_id = manifest_id
        self.authorities = tuple(authorities)
        self.consumer = consumer
        self.role_path = tuple(role_path)
        origin = (
            f"{consumer.text} declares input {list(self.role_path)} as the already-produced "
            if consumer is not None
            else "the plan names the already-produced "
        )
        super().__init__(
            f"{origin}{kind} {manifest_id!r}, which is held by "
            f"{len(self.authorities)} declared authorities: {', '.join(self.authorities)}. "
            "A reference names one artifact and must resolve through one authority; there is "
            "no precedence among the receipt root, retained manifest stores, and artifact "
            "providers, so bind exactly one of them for this reference."
        )

    def record_detail(self) -> dict[str, Any]:
        """Return the structured refusal, deterministic in every field."""
        return {
            "consumer": None if self.consumer is None else self.consumer.text,
            "role_path": list(self.role_path),
            "manifest_kind": self.kind,
            "manifest_id": self.manifest_id,
            "authorities": list(self.authorities),
        }


class AmbiguousNodeReceiptError(FulfillmentDriverError):
    """One closure node did not resolve to the single receipt its consumers bind."""


class PlanDocumentDriftError(FulfillmentDriverError):
    """A node's declaration no longer hashes to what the plan pinned.

    Plan construction and lowering read the same declarations, so this can only
    mean they moved between them. Refusing names the node rather than letting a
    closure execute one document while its plan described another.
    """

    def __init__(self, key: LogicalKey, source_ref: str, expected: str, observed: str) -> None:
        self.key = key
        self.source_ref = source_ref
        self.expected = expected
        self.observed = observed
        super().__init__(
            f"{source_ref} now hashes to {observed} but the plan pinned {expected} for node "
            f"{key.text}; the declarations changed between plan construction and lowering"
        )


class UnpinnedPlanNodeError(FulfillmentDriverError):
    """A plan node names a compiled document without pinning its content hash.

    The pin is what makes the drift check a check. A node carrying none does not
    make the closure safer to run than one whose pin still matches; it makes the
    comparison unperformable, and an unperformable comparison that is skipped is
    indistinguishable from one that passed. Derivation always records the pin, so
    a node without one is a plan that was hand-written or stripped.
    """

    def __init__(self, key: LogicalKey, source_ref: str) -> None:
        self.key = key
        self.source_ref = source_ref
        super().__init__(
            f"the fulfillment plan carries node {key.text} ({source_ref}) with no "
            "content_hash, so the compiled document it executes cannot be proved to be the "
            "one the plan described. Re-derive the plan from the compiled outputs it is "
            "meant to describe; a missing pin is a refusal, never a skipped check."
        )

    def record(self) -> dict[str, Any]:
        """Return the structured refusal, deterministic in every field."""
        return {**self.key.record(), "source_ref": self.source_ref}


class PlanNodeDisagreementError(FulfillmentDriverError):
    """A plan node states facts about itself that its compile lock does not.

    The content hash proves which *document* a node executes. It proves nothing
    about the rest of what the node says it is, and the rest is not decoration:
    the logical key is the address every receipt, edge, and dependency is
    resolved through; the schema id is what the node is lowered by; the boundary
    is what makes a node one this runner must never execute; the execution
    identity is the compile's own statement of what it built.

    Every one of those was copied out of a compile lock when the plan was
    derived, and a copy is authority only until somebody checks it. So they are
    checked here, on the same footing as the plan's edges, and against the same
    authority: the lock is the sole source of what a node is, and the plan is a
    derived record of it. A node that carries an intact document pin beside a
    substituted key, kind, boundary, or execution identity is a plan describing
    a node the compile never emitted.
    """

    def __init__(
        self, key: LogicalKey, source_ref: str, differences: Sequence[str]
    ) -> None:
        self.key = key
        self.source_ref = source_ref
        self.differences = tuple(differences)
        listing = "; ".join(self.differences)
        super().__init__(
            f"the fulfillment plan describes node {key.text} ({source_ref}) as something its "
            f"compile lock does not determine: {listing}. The compile lock is the sole "
            "authority for what a node is; a plan is a derived record of it. Re-derive the "
            "plan from the compiled outputs it is meant to describe."
        )

    def record(self) -> dict[str, Any]:
        """Return the structured refusal, deterministic in every field."""
        return {
            **self.key.record(),
            "source_ref": self.source_ref,
            "differences": list(self.differences),
        }


@dataclass(frozen=True)
class ClosureNode:
    """One node of a preflighted closure, with the compiled output it executes.

    ``compiled`` is the lock/document pair the plan node was derived from, read
    back and verified against the hash the plan pinned.
    """

    key: LogicalKey
    plan_node: PlanNode
    compiled: CompiledEnvelope
    order: int

    @property
    def layer(self) -> str:
        return self.key.layer

    @property
    def kind(self) -> str:
        """The compiled document's schema identity."""
        return self.plan_node.kind

    @property
    def document(self) -> Mapping[str, Any]:
        """The compiled document this node executes."""
        return self.compiled.document

    @property
    def source_ref(self) -> str:
        return self.plan_node.source_ref


@dataclass(frozen=True)
class FulfillmentClosure:
    """A preflighted plan: boundary proved, applicability decided.

    The closure holds no receipt and no manifest id. A dependent node's identity
    is mintable only once its parents' receipts bind, so ids are resolved while
    walking, not here.
    """

    plan: FulfillmentPlan
    nodes: tuple[ClosureNode, ...]

    @property
    def target(self) -> LogicalKey:
        return self.plan.target

    @property
    def order(self) -> tuple[str, ...]:
        """Return the one deterministic execution order this closure has."""
        return tuple(node.key.text for node in self.nodes)

    def node(self, key: LogicalKey) -> ClosureNode:
        for node in self.nodes:
            if node.key == key:
                return node
        raise KeyError(key.text)


@dataclass(frozen=True)
class NodeBinding:
    """Everything the lowering needs to bind one node's declared inputs.

    The lowering is handed the node, the plan, the required input edges in the
    plan's canonical order, and the receipts of the producers already walked. The
    two resolution helpers are the only sanctioned ways to turn an edge into an
    authenticated reference: :meth:`parent_ref` for a normal input and
    :meth:`exact_parent_entry` for one that also records where it was executed
    from.

    A lowering may ask for both — a report binds its parents *and* the locations
    it executes them from — so one external edge is resolved once per binding and
    both answers come from that resolution. Resolving twice would read the same
    file twice, and two reads of one path are two files as far as anything
    outside this process is concerned: the second one is what the exact-parent
    locator would describe, and no comparison between them can be trusted to
    catch it, because both sides would then be describing the replacement.
    """

    closure: FulfillmentClosure
    environment: FulfillmentEnvironment
    receipts: Mapping[LogicalKey, FulfillmentReceipt] = field(default_factory=dict)
    #: Per-edge external resolutions, memoized for this binding's lifetime. One
    #: binding lowers one node, and an edge is addressed by
    #: ``(consumer, role_path)``, which the plan kernel proves is unique.
    _resolved: dict[tuple[LogicalKey, tuple[str, ...]], "ResolvedExternalReceipt"] = field(
        default_factory=dict, repr=False, compare=False
    )

    @property
    def plan(self) -> FulfillmentPlan:
        return self.closure.plan

    def required_edges(self, key: LogicalKey) -> tuple[PlanEdge, ...]:
        """Return one node's required inputs, in the plan's canonical order."""
        return self.plan.required_edges(key)

    def producer_receipt(self, edge: PlanEdge) -> FulfillmentReceipt | None:
        """Return the admitted receipt one edge's in-closure producer wrote.

        A producer that did not resolve to a single receipt — an expanded matrix,
        a driven bundle — refuses here rather than earlier, because a
        multi-receipt node no consumer names is a legitimate closure member and
        only a consumer reaching for its receipt makes the ambiguity real.
        """
        if edge.producer is None:
            return None
        receipt = self.receipts.get(edge.producer)
        if receipt is None:
            raise AmbiguousNodeReceiptError(
                f"{edge.consumer.text} declares input {list(edge.role_path)} as the product of "
                f"{edge.producer.text}, which resolved to no single receipt; a consumer binds "
                "one authenticated reference per declared input, and this version does not "
                "choose among the receipts of a multi-receipt node"
            )
        return receipt

    def parent_ref(self, edge: PlanEdge, *, role: str) -> ParentRef:
        """Return the authenticated reference one required edge binds.

        An edge with a producer in this closure binds that producer's admitted
        receipt. An edge carrying an external record binds the receipt stored at
        its canonical location; the record's manifest kind, id, and quoted byte
        profile all come from the typed lock reference the edge was derived from,
        so nothing is guessed, nothing is supplied by a caller, and a quoted
        profile that the stored bytes do not match refuses rather than binds.
        """
        receipt = self.producer_receipt(edge)
        if receipt is not None:
            return receipt_parent_ref(receipt, role=role)
        return self.resolved_external(edge).parent_ref(role)

    def exact_parent_entry(self, edge: PlanEdge, *, role: str) -> StagedExactParentEntry:
        """Bind one required edge to the location within its authority it executes from.

        The locator is relative to the authority that actually holds the bytes,
        which for a receipt produced in this closure and for a reference resolved
        beneath the receipt root is the receipt root, and for a reference
        resolved through a retained store or an artifact provider is that root.
        Making it relative to the receipt root regardless would describe a file
        that is not there.
        """
        receipt = self.producer_receipt(edge)
        if receipt is not None:
            return receipt_exact_parent_entry(receipt, role=role)
        resolved = self.resolved_external(edge)
        return StagedExactParentEntry(
            parent=resolved.parent_ref(role),
            execution_uri=resolved.execution_uri,
        )

    def parent_location(self, edge: PlanEdge, *, role: str) -> StagedParentExecutionLocation:
        """Bind one required edge to the staged execution location it resolves at.

        This is :meth:`exact_parent_entry` plus the authority the locator is
        relative to, which is what a staged execution context needs and a
        ``StagedExactParents`` document does not carry: the document travels with
        a report and states locations within one root, while the context states
        which root each parent's bytes live beneath.
        """
        receipt = self.producer_receipt(edge)
        if receipt is not None:
            entry = receipt_exact_parent_entry(receipt, role=role)
            return StagedParentExecutionLocation(
                parent=entry.parent,
                root=Path(self.environment.root),
                execution_uri=entry.execution_uri,
            )
        return self.resolved_external(edge).execution_location(role)

    def authenticated_parent_location(
        self, parent: ParentRef
    ) -> StagedParentExecutionLocation:
        """Locate one already-authenticated parent that no plan edge bound.

        A row-expanded figure's per-row parents come from produced custody rather
        than from a plan edge, and they are complete authenticated references, so
        they are located by the same exactly-one authority search every external
        edge takes. Nothing about that search is relaxed here: a parent held by
        two declared authorities refuses, and a parent held by none refuses.
        """
        profile = authenticated_manifest_ref_profile(parent)
        if profile is None:
            raise MissingExternalReceiptError(
                parent.kind,
                parent.id,
                canonical_manifest_path(
                    parent.kind, parent.id, root=Path(self.environment.root)
                ),
            )
        digest, size_bytes = profile
        authorities = declared_manifest_authorities(
            parent.kind,
            parent.id,
            root=Path(self.environment.root),
            manifest_sha256=digest,
            size_bytes=size_bytes,
            execution_context=self.environment.execution_context,
        )
        authority, _raw, _manifest = require_single_manifest_authority(
            parent.kind,
            parent.id,
            authorities,
            missing_path=canonical_manifest_path(
                parent.kind, parent.id, root=Path(self.environment.root)
            ),
        )
        return StagedParentExecutionLocation(
            parent=parent,
            root=authority.root,
            execution_uri=authority.execution_uri,
            artifact_provider=authority.artifact_provider,
        )

    def node_execution_context(
        self,
        locations: Sequence[StagedParentExecutionLocation],
    ) -> StagedExecutionContext | None:
        """Return the staged context one node executes under, or ``None``.

        ``None`` is the statement that this run declared no staged bindings at
        all, and it is the whole of the cold-start case: every parent lives at
        its canonical location beneath the receipt root, figure input resolution
        takes the root authority, and nothing downstream sees a context. That
        path is left byte-identical rather than routed through an empty one.

        Otherwise the node's context is the run's base context — the opened
        providers, retained manifest stores, and custody roots the caller
        declared — augmented with *exactly this node's* parents: where each
        one's bytes were found, and the provider aliases that follow from the
        descriptor's own declarations. It is per node because that is the
        smallest correct unit: a node binds the parents it binds, and a global
        context would either be missing a node's parents or carrying another
        node's.
        """
        base = self.environment.execution_context
        if base is EMPTY_STAGED_EXECUTION_CONTEXT:
            return None
        return with_staged_resolved_parents(
            with_staged_repo_root(base, self.environment.repo_root), locations
        )

    def resolved_external(self, edge: PlanEdge) -> "ResolvedExternalReceipt":
        """Return the single resolution one external edge has under this binding.

        The first ask reads and authenticates the receipt; every later ask for
        the same edge is answered from that resolution rather than from the
        filesystem, so a lowering that binds a parent, its execution location,
        and the staged context it executes under binds three descriptions of one
        read.
        """
        cache_key = (edge.consumer, edge.role_path)
        resolved = self._resolved.get(cache_key)
        if resolved is None:
            resolved = resolve_external_receipt(
                require_external_record(edge.external, field=self._edge_field(edge)),
                root=self.environment.root,
                consumer=edge.consumer,
                role_path=edge.role_path,
                execution_context=self.environment.execution_context,
            )
            self._resolved[cache_key] = resolved
        return resolved

    @staticmethod
    def _edge_field(edge: PlanEdge) -> str:
        return f"{edge.consumer.text} input {list(edge.role_path)}"


class ExternalReceiptAuthenticationError(FulfillmentDriverError):
    """The receipt at the canonical location is not the bytes the lock quoted.

    A lock that authenticated a receipt named the exact bytes it read. Kind, id,
    and status address *where* a receipt is and *whether* it completed; none of
    them says the bytes are the same ones. So a completed manifest of the right
    kind and id whose bytes disagree with the quote is a different artifact
    occupying the same address, and binding it would let a recompiled,
    re-executed, or substituted run be consumed as the one the compile
    authenticated.
    """

    def __init__(
        self,
        record: ExternalReceiptRecord,
        path: Path,
        *,
        found_sha256: str,
        found_size: int,
        consumer: LogicalKey | None = None,
        role_path: Sequence[str] = (),
    ) -> None:
        self.record = record
        self.path = path
        self.found_sha256 = found_sha256
        self.found_size = found_size
        self.consumer = consumer
        self.role_path = tuple(role_path)
        origin = (
            f"{consumer.text} declares input {list(self.role_path)} as the authenticated "
            if consumer is not None
            else "the plan quotes the authenticated "
        )
        super().__init__(
            f"{origin}{record.manifest_kind} {record.manifest_id!r} with "
            f"manifest_sha256={record.manifest_sha256} size_bytes={record.size_bytes}, but "
            f"{path} holds manifest_sha256={found_sha256} size_bytes={found_size}. The compile "
            "lock is the sole authority for which bytes that reference names; a manifest is "
            "never blessed on kind, id, and status alone. Recompile the closure against the "
            "receipt that is actually there, or restore the quoted receipt."
        )

    def record_detail(self) -> dict[str, Any]:
        """Return the structured refusal, deterministic in every field."""
        return {
            "consumer": None if self.consumer is None else self.consumer.text,
            "role_path": list(self.role_path),
            "manifest_kind": self.record.manifest_kind,
            "manifest_id": self.record.manifest_id,
            "path": str(self.path),
            "lock_manifest_sha256": self.record.manifest_sha256,
            "lock_size_bytes": self.record.size_bytes,
            "found_manifest_sha256": self.found_sha256,
            "found_size_bytes": self.found_size,
        }


@dataclass(frozen=True)
class ManifestAuthority:
    """One declared place a already-produced manifest could be held.

    An authority is a *root plus a location within it*, never a bare root: the
    receipt root and a retained manifest store address a manifest by the
    canonical layout, while an immutable artifact provider addresses it by
    content, and those are different locations for the same artifact.

    Attributes:
        name: How this authority is named in a refusal, so a diagnostic says
            which declaration was searched rather than only which path was.
        root: The runtime root the bytes are read beneath.
        execution_uri: The root-relative POSIX location within it.
        artifact_provider: The runtime provider name, when the authority *is* a
            provider; ``None`` for a plain retained root.
    """

    name: str
    root: Path
    execution_uri: str
    artifact_provider: str | None = None

    @property
    def path(self) -> Path:
        return self.root.joinpath(*self.execution_uri.split("/"))


@dataclass(frozen=True)
class ResolvedExternalReceipt:
    """One external receipt resolved from exactly one read of its bytes.

    Everything downstream is derived from :attr:`raw`. The manifest is parsed
    from those bytes rather than loaded again from the path, and the byte profile
    is measured on them, so there is no second read for the file to change
    between: the bytes that were authenticated are the bytes that are used.

    Attributes:
        record: The external edge record this was resolved for.
        manifest: The manifest parsed from :attr:`raw`.
        path: Where those bytes were read from, for diagnostics and execution
            locators.
        raw: The single read.
        manifest_sha256: The digest this reference binds. When the lock quoted a
            profile this is the *lock's* digest, verified equal to the digest of
            :attr:`raw`; otherwise it is the digest of :attr:`raw`, because a
            locator quoted nothing to prefer over what was read.
        size_bytes: The size this reference binds, on the same rule.
        authority: The single declared authority the bytes came from.
    """

    record: ExternalReceiptRecord
    manifest: AnyManifest
    path: Path
    raw: bytes
    manifest_sha256: str
    size_bytes: int
    authority: ManifestAuthority

    @property
    def root(self) -> Path:
        """The runtime root this receipt's bytes were read beneath."""
        return self.authority.root

    @property
    def execution_uri(self) -> str:
        """Where within that root the bytes are, as an execution locator."""
        return self.authority.execution_uri

    @property
    def artifact_provider(self) -> str | None:
        """The runtime provider name, when a provider held these bytes."""
        return self.authority.artifact_provider

    def execution_location(self, role: str) -> "StagedParentExecutionLocation":
        """Bind this resolution as the staged location its parent executes from."""
        return StagedParentExecutionLocation(
            parent=self.parent_ref(role),
            root=self.authority.root,
            execution_uri=self.authority.execution_uri,
            artifact_provider=self.authority.artifact_provider,
        )

    def parent_ref(self, role: str) -> ParentRef:
        """Return the authenticated parent this resolution binds, under *role*.

        The metadata is minted from the profile this resolution already settled,
        never by hashing the file again. Re-reading to mint a digest is what
        would let bytes swapped after authentication be blessed with their own
        digest, which is exactly the profile a consumer would then record as
        proof.
        """
        return ParentRef(
            kind=self.manifest.kind,
            id=self.manifest.id,
            role=role,
            uri=None,
            metadata=authenticated_manifest_ref_metadata(self.manifest_sha256, self.size_bytes),
        )


def declared_manifest_authorities(
    kind: str,
    manifest_id: str,
    *,
    root: Path | str,
    manifest_sha256: str | None = None,
    size_bytes: int | None = None,
    execution_context: StagedExecutionContext | None = None,
) -> tuple[ManifestAuthority, ...]:
    """Return every declared authority that could hold one already-produced manifest.

    Three kinds of authority are searched, and they are not ranked. The receipt
    root is one candidate and keeps its ordinary meaning — output and admission
    custody — rather than becoming an inferred provider root. Each retained
    manifest store is a candidate at the same canonical layout position, for
    every manifest kind admitted as a staged input. Each immutable artifact
    provider is a candidate at the content address the reference's own quoted
    digest names, which is why an unauthenticated reference reaches no provider:
    a locator states no digest, and a provider has no other way to be addressed.

    The list is deterministic — receipt root, then manifest stores in name order,
    then providers in name order — so a refusal names the same search twice.
    """
    root_path = Path(root)
    canonical = canonical_manifest_path(kind, manifest_id, root=root_path)
    authorities = [
        ManifestAuthority(
            name="receipt root",
            root=root_path,
            execution_uri=canonical.relative_to(root_path).as_posix(),
        )
    ]
    if execution_context is None:
        return tuple(authorities)
    if is_staged_manifest_kind(kind):
        relative = canonical_staged_manifest_locator(kind, manifest_id)
        for name in sorted(execution_context.manifest_roots):
            authorities.append(
                ManifestAuthority(
                    name=f"manifest root {name!r}",
                    root=execution_context.manifest_root(name),
                    execution_uri=relative,
                )
            )
    if manifest_sha256 is not None:
        artifact_id = f"artifact://sha256/{manifest_sha256}"
        for name in sorted(execution_context.opened_artifact_providers):
            provider = execution_context.artifact_provider(name)
            try:
                provider_relative = provider.canonical_relative_path(
                    artifact_id, size_bytes=size_bytes
                )
            except FileNotFoundError:
                continue
            authorities.append(
                ManifestAuthority(
                    name=f"artifact provider {name!r}",
                    root=Path(provider.root),
                    execution_uri=provider_relative.as_posix(),
                    artifact_provider=name,
                )
            )
    return tuple(authorities)


def _read_completed_manifest(
    authority: ManifestAuthority, *, kind: str, manifest_id: str
) -> tuple[bytes, AnyManifest] | None:
    """Return one authority's completed manifest bytes, or ``None`` if it holds none.

    Absent, unreadable, unparseable, mis-addressed, and incomplete are one answer
    here — this authority does not hold the receipt — because each of them is a
    reason the reference is not resolvable through it, and none of them is a
    reason to prefer or refuse another authority that does hold it.
    """
    try:
        raw = authority.path.read_bytes()
    except OSError:
        return None
    try:
        manifest = load_manifest_bytes(raw)
    except (ValueError, TypeError):
        return None
    if manifest.kind != kind or manifest.id != manifest_id or manifest.status != "completed":
        return None
    return raw, manifest


def require_single_manifest_authority(
    kind: str,
    manifest_id: str,
    authorities: Sequence[ManifestAuthority],
    *,
    consumer: LogicalKey | None = None,
    role_path: Sequence[str] = (),
    missing_path: Path,
) -> tuple[ManifestAuthority, bytes, AnyManifest]:
    """Return the one declared authority holding a receipt, or refuse.

    Zero is :class:`MissingExternalReceiptError` and more than one is
    :class:`AmbiguousExternalReceiptError`. Both refuse before the resolution is
    used for anything, so no node executes against a reference whose custody
    domain was undecided.
    """
    found = [
        (authority, *read)
        for authority in authorities
        if (read := _read_completed_manifest(authority, kind=kind, manifest_id=manifest_id))
        is not None
    ]
    if not found:
        raise MissingExternalReceiptError(
            kind,
            manifest_id,
            missing_path,
            consumer=consumer,
            role_path=role_path,
            searched=tuple(authority.name for authority in authorities),
        )
    if len(found) > 1:
        raise AmbiguousExternalReceiptError(
            kind,
            manifest_id,
            tuple(authority.name for authority, _raw, _manifest in found),
            consumer=consumer,
            role_path=role_path,
        )
    return found[0]


def resolve_external_receipt(
    record: ExternalReceiptRecord,
    *,
    root: Path | str,
    consumer: LogicalKey | None = None,
    role_path: Sequence[str] = (),
    execution_context: StagedExecutionContext | None = None,
) -> ResolvedExternalReceipt:
    """Authenticate one already-produced receipt from the single authority holding it.

    The manifest index is derived acceleration and is never consulted: a receipt
    lives where feedbax's ``(kind, id, root)`` derivation says it lives within
    whichever authority holds it, so an absent or stale index entry cannot change
    what this resolves.

    Resolution proves three separate things. That exactly one declared authority
    holds a completed receipt of the named kind and id is *addressing*: none is
    :class:`MissingExternalReceiptError` and several is
    :class:`AmbiguousExternalReceiptError`. That its bytes are the ones the lock
    quoted is *authentication*, and only a lock that quoted a byte profile can
    assert it; when one did, disagreement refuses rather than binding whatever
    now occupies the address.

    Every proof is made against **one** read per authority, and the resolution
    carries the read that decided. Authenticating a file and then reopening it to
    describe what was authenticated is two different files as far as anything
    outside this process is concerned, and the second one is the one that would
    be recorded.
    """
    kind, manifest_id = record.locator
    root_path = Path(root)
    authorities = declared_manifest_authorities(
        kind,
        manifest_id,
        root=root_path,
        manifest_sha256=record.manifest_sha256,
        size_bytes=record.size_bytes,
        execution_context=execution_context,
    )
    authority, raw, manifest = require_single_manifest_authority(
        kind,
        manifest_id,
        authorities,
        consumer=consumer,
        role_path=role_path,
        missing_path=canonical_manifest_path(kind, manifest_id, root=root_path),
    )
    path = authority.path
    found_sha256 = hashlib.sha256(raw).hexdigest()
    found_size = len(raw)
    if record.is_authenticated:
        if found_sha256 != record.manifest_sha256 or found_size != record.size_bytes:
            raise ExternalReceiptAuthenticationError(
                record,
                path,
                found_sha256=found_sha256,
                found_size=found_size,
                consumer=consumer,
                role_path=role_path,
            )
        # Equal by the check above, and taken from the lock deliberately: the
        # profile a consumer records is the one the lock stated, not one this
        # process measured and happens to believe.
        bound_sha256 = str(record.manifest_sha256)
        bound_size = int(record.size_bytes or 0)
    else:
        bound_sha256, bound_size = found_sha256, found_size
    return ResolvedExternalReceipt(
        record=record,
        manifest=manifest,
        path=path,
        raw=raw,
        manifest_sha256=bound_sha256,
        size_bytes=bound_size,
        authority=authority,
    )


def external_parent_ref(
    record: ExternalReceiptRecord,
    *,
    role: str,
    root: Path | str,
    consumer: LogicalKey | None = None,
    role_path: Sequence[str] = (),
    execution_context: StagedExecutionContext | None = None,
) -> tuple[ParentRef, Path]:
    """Return the authenticated parent one external receipt record names."""
    resolved = resolve_external_receipt(
        record,
        root=root,
        consumer=consumer,
        role_path=role_path,
        execution_context=execution_context,
    )
    return resolved.parent_ref(role), resolved.path


class UncertifiedApplicabilityError(FulfillmentDriverError):
    """The closure omits an input under a structural rule that does not certify it.

    A ``compiler_rule`` omission is honored on the strength of the rule it
    quotes. A rule id this build cannot state certifies nothing, and neither does
    a rule it can state quoted on a consumer or role the rule proves nothing
    about. Either way the omission is refused rather than executed around: the
    input is neither bound nor proved inapplicable, and running the node would
    silently produce an artifact missing an input nobody decided about.
    """

    def __init__(self, target: LogicalKey, failures: Sequence[tuple[PlanEdge, str]]) -> None:
        self.target = target
        self.failures = tuple(failures)
        listing = "; ".join(
            f"{edge.consumer.text} input {list(edge.role_path)}: {detail}"
            for edge, detail in self.failures
        )
        super().__init__(
            f"fulfilling {target.text} would honor {len(self.failures)} uncertified "
            f"applicability decision(s): {listing}"
        )

    def record(self) -> dict[str, Any]:
        """Return the structured refusal, deterministic in every field."""
        return {
            "target": self.target.text,
            "uncertified": [
                {
                    "consumer": edge.consumer.text,
                    "role_path": list(edge.role_path),
                    "rule": edge.rule,
                    "detail": detail,
                }
                for edge, detail in self.failures
            ],
        }


def require_certified_applicability(plan: FulfillmentPlan) -> None:
    """Refuse a plan that omits an input under a rule that does not decide it.

    The plan kernel deliberately does not know what a rule *name* means, so the
    proof that a quoted rule is one of Feedbax's closed structural rules — and
    that it decides the input it was quoted on — is made here, where the closure
    is about to be executed. The edge carries everything the predicate needs: the
    consumer's artifact layer, the role path, and the reason the decision states.

    Ownership alone is not certification. The rule table's sole entry is about a
    per-row figure input slot, so a lock quoting it over an evaluation, report,
    or analysis prerequisite would otherwise drop that prerequisite from the
    closure under a rule that proves nothing about it.

    Every uncertified decision in the closure is collected before any is raised,
    so one refusal names the whole problem rather than the first instance of it.
    """
    failures: list[tuple[PlanEdge, str]] = []
    for edge in sorted(plan.certified_omissions(), key=lambda item: item.sort_key):
        try:
            certify_structural_applicability(
                edge.rule,
                consumer_layer=edge.consumer.layer,
                role_path=edge.role_path,
                reason=edge.reason,
                ref=f"{edge.consumer.text} input {list(edge.role_path)}",
            )
        except (
            UnknownStructuralApplicabilityRuleError,
            StructuralApplicabilityRuleMismatchError,
        ) as exc:
            failures.append((edge, str(exc)))
    if failures:
        raise UncertifiedApplicabilityError(plan.target, failures)


def require_no_external_boundary(plan: FulfillmentPlan) -> None:
    """Refuse a plan that needs a receipt only another entrypoint can produce."""
    boundary_nodes = plan.boundary_nodes()
    if not boundary_nodes:
        return
    consumers = plan.consumers()
    boundary = [
        BoundaryNode(
            key=node.key,
            source_ref=node.source_ref,
            boundary=node.boundary or "",
            named_by=tuple(
                (edge.consumer.text, edge.role_path)
                for edge in sorted(consumers.get(node.key, ()), key=lambda item: item.sort_key)
            ),
            unblocks=plan.descendants(node.key),
        )
        for node in boundary_nodes
    ]
    raise ExternalBoundaryError(plan.target, boundary)


class PlanLockDisagreementError(FulfillmentDriverError):
    """A plan's declared inputs are not the ones its compile lock determines.

    A :class:`~feedbax.analysis.fulfillment_plan.FulfillmentPlan` is a durable
    document. It is derived from compile locks, but it travels apart from them
    and can be edited, regenerated against different locks, or hand-written. That
    makes every authenticating fact it carries a *copy*, and a copy is authority
    only for as long as nobody checks it.

    So the plan is reconciled against the locks before the closure runs. The lock
    is the sole authority; the plan is a cache of it. Any disagreement refuses —
    a stripped byte profile, a substituted manifest id, an input added or
    dropped, a decision restated on a different basis — because each of those is
    a way of saying the node's inputs are something the lock never said.
    """

    def __init__(self, key: LogicalKey, source_ref: str, differences: Sequence[str]) -> None:
        self.key = key
        self.source_ref = source_ref
        self.differences = tuple(differences)
        listing = "; ".join(self.differences)
        super().__init__(
            f"the fulfillment plan declares inputs for {key.text} that {source_ref}'s compile "
            f"lock does not determine: {listing}. The compile lock is the sole authority for "
            "what a node's inputs are and how they are authenticated; a plan is a derived "
            "record of it. Re-derive the plan from the compiled outputs it is meant to describe."
        )

    def record(self) -> dict[str, Any]:
        """Return the structured refusal, deterministic in every field."""
        return {
            **self.key.record(),
            "source_ref": self.source_ref,
            "differences": list(self.differences),
        }


def _edge_facts(edge: PlanEdge) -> dict[str, Any]:
    """Return the authority-bearing facts one edge states, for exact comparison."""
    return {
        "role_path": list(edge.role_path),
        "status": edge.status,
        "basis": edge.basis,
        "reason": edge.reason,
        "rule": edge.rule,
        "producer": None if edge.producer is None else edge.producer.text,
        "external": None if edge.external is None else dict(edge.external),
    }


def _duplicate_role_differences(
    edges: Sequence[PlanEdge], *, stated_by: str
) -> list[str]:
    """Return one difference per role path *edges* states more than once.

    Comparison keys an edge by its role path, so a duplicate would collapse into
    whichever copy is keyed last and the others would never be compared with
    anything. That is not a comparison that can fail, so the duplicate is named
    before any keying happens rather than being reconciled away.
    """
    counts: dict[tuple[str, ...], int] = {}
    for edge in edges:
        counts[edge.role_path] = counts.get(edge.role_path, 0) + 1
    return [
        f"input {list(role_path)} is stated {count} times by {stated_by}; a role path "
        "addresses exactly one input, and reconciliation compares one edge per role"
        for role_path, count in sorted(counts.items())
        if count > 1
    ]


def require_plan_matches_locks(plan: FulfillmentPlan, index: CompiledOutputIndex) -> None:
    """Refuse a plan whose edges are not the edges its locks determine.

    Every node's inputs are re-derived from its own compile lock and compared,
    field for field, with what the plan document carries. Nothing is repaired and
    nothing is preferred: a plan that disagrees with a lock is refused, because
    silently taking either side would make one of the two documents decorative.

    Both sides are proved to state at most one edge per role path *before* they
    are keyed by role path. The plan kernel already refuses a duplicate and a
    lock already refuses two references at one role, so this is defense in depth
    over the one shape that would otherwise be dropped silently by the
    comparison itself rather than reported by it.
    """
    for plan_node in plan.nodes:
        compiled = index.require(plan_node.source_ref)
        lock_edges: list[PlanEdge] = []
        for declaration in lock_edge_declarations(compiled):
            producer: LogicalKey | None = None
            if declaration.producer_ref is not None:
                producer = index.require(declaration.producer_ref).key
            lock_edges.append(declaration.edge(plan_node.key, producer))
        plan_edges = plan.input_edges(plan_node.key)
        duplicates = _duplicate_role_differences(
            lock_edges, stated_by="the lock"
        ) + _duplicate_role_differences(plan_edges, stated_by="the plan")
        if duplicates:
            raise PlanLockDisagreementError(
                plan_node.key, plan_node.source_ref, duplicates
            )
        expected: dict[tuple[str, ...], PlanEdge] = {
            edge.role_path: edge for edge in lock_edges
        }
        declared = {edge.role_path: edge for edge in plan_edges}
        differences: list[str] = []
        for role_path in sorted(set(expected) | set(declared)):
            lock_edge = expected.get(role_path)
            plan_edge = declared.get(role_path)
            if lock_edge is None:
                differences.append(
                    f"input {list(role_path)} is declared by the plan but the lock states no "
                    "reference at that role"
                )
                continue
            if plan_edge is None:
                differences.append(
                    f"input {list(role_path)} is stated by the lock but the plan declares no "
                    "edge for it"
                )
                continue
            lock_facts = _edge_facts(lock_edge)
            plan_facts = _edge_facts(plan_edge)
            for name in sorted(lock_facts):
                if lock_facts[name] != plan_facts[name]:
                    differences.append(
                        f"input {list(role_path)} {name}: the lock states "
                        f"{lock_facts[name]!r} and the plan declares {plan_facts[name]!r}"
                    )
        if differences:
            raise PlanLockDisagreementError(
                plan_node.key, plan_node.source_ref, differences
            )


def _node_facts_from_lock(compiled: CompiledEnvelope) -> dict[str, Any]:
    """Return the node facts one compiled output determines, for exact comparison."""
    return {
        "key": compiled.key.text,
        "kind": compiled.schema_id,
        "content_hash": compiled.content_hash,
        "execution_identity": compiled.execution_identity,
        "boundary": compiled.kind.boundary,
    }


def _node_facts_from_plan(plan_node: PlanNode) -> dict[str, Any]:
    """Return the same node facts as the plan document carries them."""
    return {
        "key": plan_node.key.text,
        "kind": plan_node.kind,
        "content_hash": plan_node.content_hash,
        "execution_identity": plan_node.execution_identity,
        "boundary": plan_node.boundary,
    }


def require_plan_node_matches_lock(
    plan_node: PlanNode, compiled: CompiledEnvelope
) -> None:
    """Refuse a plan node whose own facts are not the ones its lock determines.

    This is the node-side counterpart of
    :func:`require_plan_matches_locks`, which does the same for edges. Both
    exist for one reason: the plan is a cache of the locks, and every
    authenticating fact it carries is a copy. ``content_hash`` is compared
    separately by :func:`preflight`, before this, because a moved document is
    the coarser fact and explains the differences that follow from it.
    """
    expected = _node_facts_from_lock(compiled)
    declared = _node_facts_from_plan(plan_node)
    differences = [
        f"{name}: the lock determines {expected[name]!r} and the plan declares "
        f"{declared[name]!r}"
        for name in sorted(expected)
        if expected[name] != declared[name]
    ]
    if differences:
        raise PlanNodeDisagreementError(plan_node.key, plan_node.source_ref, differences)


def preflight(plan: FulfillmentPlan, index: CompiledOutputIndex) -> FulfillmentClosure:
    """Bind one plan's closure to its compiled outputs and prove its boundary.

    Nothing executes here and nothing is written. The boundary is proved first,
    because a closure that cannot run at all is not made runnable by anything a
    later check finds.

    Every node is re-read from *index* and re-hashed. Plan derivation and
    fulfillment read the same compiled outputs, so a hash that no longer matches
    means the outputs moved between them, and the node is named rather than
    executed under a plan that described something else.

    The node hash proves the compiled *document* is the one the plan described.
    It says nothing about the rest of what the plan asserts. The node's own
    facts — its logical key, the schema id it is lowered by, its execution
    identity, whether it is a boundary — are copied out of the compile lock at
    derivation, so they are reconciled back against it here. Its edges are
    derived from the lock rather than from the document, so they are reconciled
    separately: otherwise a plan could carry an intact document hash beside an
    input the lock never stated, or beside an authenticated reference with its
    byte profile removed. The document check runs first, because a moved document
    is the coarser fact and explains the node and edge differences that follow
    from it.

    Raises:
        ExternalBoundaryError: The closure names a boundary node.
        UncertifiedApplicabilityError: An omission quotes a structural rule this
            build does not own, so nothing certified it.
        PlanDocumentDriftError: A compiled document no longer hashes to its pin.
        UnpinnedPlanNodeError: A node carries no content hash, so the document it
            executes cannot be proved to be the one the plan described.
        PlanNodeDisagreementError: A node's own facts — its logical key, schema
            id, execution identity, or boundary — are not the ones its compile
            lock determines.
        PlanLockDisagreementError: A node's declared inputs are not the ones its
            compile lock determines.
    """
    require_no_external_boundary(plan)
    require_certified_applicability(plan)
    nodes: list[ClosureNode] = []
    for order, plan_node in enumerate(plan.nodes):
        compiled = index.require(plan_node.source_ref)
        if plan_node.content_hash is None:
            raise UnpinnedPlanNodeError(plan_node.key, plan_node.source_ref)
        if compiled.content_hash != plan_node.content_hash:
            raise PlanDocumentDriftError(
                plan_node.key,
                plan_node.source_ref,
                plan_node.content_hash,
                compiled.content_hash,
            )
        require_plan_node_matches_lock(plan_node, compiled)
        nodes.append(
            ClosureNode(
                key=plan_node.key, plan_node=plan_node, compiled=compiled, order=order
            )
        )
    require_plan_matches_locks(plan, index)
    return FulfillmentClosure(plan=plan, nodes=tuple(nodes))


def fulfill_closure(
    closure: FulfillmentClosure,
    *,
    environment: FulfillmentEnvironment,
) -> FulfillmentRun:
    """Walk one preflighted closure, reusing or executing each node in order.

    Each node is reused or executed exactly once, strictly serially, in the
    plan's dependency order, and its receipt binds the nodes that declare it as
    an input. Re-invoking re-runs only what is missing, so a second walk over a
    fulfilled closure executes nothing, and an interrupted walk resumes at the
    node boundary it stopped at.
    """
    results: list[NodeFulfillment] = []
    receipts: dict[LogicalKey, FulfillmentReceipt] = {}
    for node in closure.nodes:
        request = _lowered(node, closure=closure, receipts=receipts, environment=environment)
        result = fulfill_node(request, environment=environment)
        results.append(result)
        receipt = _producer_receipt(result)
        if receipt is not None:
            receipts[node.key] = receipt
    return FulfillmentRun(results=tuple(results))


def closure_requests(
    closure: FulfillmentClosure,
    *,
    environment: FulfillmentEnvironment,
    stop_at: LogicalKey | None = None,
) -> tuple[NodeRequest, ...]:
    """Resolve every node request of an already-fulfilled closure, in order.

    Each node's request is derived from its parents' *admitted* receipts, so
    resolution walks the same dependency order fulfillment did and admits every
    node it passes. A receipt that exists but fails admission stops the walk with
    that named failure: an operation over a closure whose custody is broken must
    refuse before it touches anything, not verify around the break.

    Args:
        stop_at: Return this node's request without admitting it. Repair is the
            one operation whose subject is expected to fail admission.
    """
    requests: list[NodeRequest] = []
    receipts: dict[LogicalKey, FulfillmentReceipt] = {}
    for node in closure.nodes:
        request = _lowered(node, closure=closure, receipts=receipts, environment=environment)
        requests.append(request)
        if stop_at is not None and node.key == stop_at:
            return tuple(requests)
        receipts[node.key] = _admitted_receipt(node, request, environment=environment)
    if stop_at is not None:
        raise KeyError(stop_at.text)
    return tuple(requests)


def rebuild_closure(
    closure: FulfillmentClosure,
    *,
    environment: FulfillmentEnvironment,
) -> RebuildRun:
    """Verify one fulfilled closure by rebuilding every node into shadow custody.

    This is feedbax's rebuild-as-verification over the whole closure: serial,
    deterministic in order, authoritative receipts never written to, and every
    node verified before any drift is raised so the report is complete. Altered
    stored bytes are an admission failure and refuse before any rebuild happens.

    Raises:
        FulfillmentDriftError: Any node rebuilt to a different output projection.
        FulfillmentAdmissionError: Any node's stored receipt fails admission.
    """
    requests = closure_requests(closure, environment=environment)
    return rebuild_nodes(requests, environment=environment)


def repair_closure_node(
    closure: FulfillmentClosure,
    node_key: LogicalKey,
    *,
    environment: FulfillmentEnvironment,
    metadata: Mapping[str, Any] | None = None,
) -> RepairResult:
    """Repair one node of a closure whose receipt exists but fails admission.

    Every node upstream of the subject is admitted first, because a repair
    candidate is only meaningful when the inputs it binds are the authenticated
    ones. The repair itself — quarantine, shadow execution, complete
    revalidation, promotion, durable repair record — is feedbax's.
    """
    requests = closure_requests(closure, environment=environment, stop_at=node_key)
    return repair_node(requests[-1], environment=environment, metadata=dict(metadata or {}))


def truncated_closure(closure: FulfillmentClosure, count: int) -> FulfillmentClosure:
    """Return the same closure holding only its first *count* nodes.

    A walk that stops early is the shape an interrupted run leaves behind, and
    resuming is just walking the whole closure again. This exists so that shape
    can be stated without rebuilding a plan.
    """
    return replace(closure, nodes=closure.nodes[:count])


def _lowered(
    node: ClosureNode,
    *,
    closure: FulfillmentClosure,
    receipts: Mapping[LogicalKey, FulfillmentReceipt],
    environment: FulfillmentEnvironment,
) -> NodeRequest:
    binding = NodeBinding(closure=closure, environment=environment, receipts=dict(receipts))
    request = lower_compiled_node(node, binding=binding)
    if request.node_key != node.key.text:
        raise AmbiguousNodeReceiptError(
            f"the lowering for {node.key.text} returned a request keyed {request.node_key!r}; a "
            "node request carries the logical key of the node it fulfills, so a walk's results "
            "address the plan they came from"
        )
    return request


def _producer_receipt(result: NodeFulfillment) -> FulfillmentReceipt | None:
    """Return the one receipt a node's consumers may bind, if it resolved to one.

    A node that produced several — an expanded matrix, a driven analysis bundle —
    binds nothing, and that is not an error by itself: such a node is fulfilled
    like any other, and only a consumer that reaches for its single receipt makes
    the ambiguity real. :meth:`NodeBinding.producer_receipt` is where that
    refusal happens, with the consumer and the role it declared both named.
    """
    if len(result.receipts) == 1:
        return result.receipts[0]
    return None


def _admitted_receipt(
    node: ClosureNode,
    request: NodeRequest,
    *,
    environment: FulfillmentEnvironment,
) -> FulfillmentReceipt:
    """Admit one node's stored receipt, or refuse with its named failure."""
    if isinstance(request, EvaluationMatrixNodeRequest):
        raise AmbiguousNodeReceiptError(
            f"node {node.key.text} is an evaluation matrix; resolving a single receipt for it "
            "would mean choosing among its rows, which this version does not do"
        )
    if isinstance(request, AnalysisBundleNodeRequest):
        raise AmbiguousNodeReceiptError(
            f"node {node.key.text} is an analysis bundle; resolving a single receipt for it "
            "would mean choosing among its stage products, which this version does not do"
        )
    captured = AdmittedManifestBytes()
    outcome = admit_node(request, environment=environment, capture=captured)
    if not outcome.admitted or outcome.manifest_path is None:
        raise FulfillmentAdmissionError(outcome)
    return captured.receipt(node_kind=request.node_kind, root=Path(environment.root))
