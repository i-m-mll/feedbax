"""The fulfillment driver: one plan's closure, materialized once.

A plan (:mod:`feedbax.analysis.fulfillment_plan`) says what must exist. This
module makes it exist: it proves the closure's external boundary before anything
runs, lowers each plan node into the feedbax node request its kind executes, and
walks the closure in dependency order, reusing every node whose receipt admits
and executing only the ones that are missing. Nothing here decides *what* a node
is; orchestration decides only whether it runs.

## What this module owns, and what it does not

feedbax owns receipt admission, the executor adapters, and custody semantics.
Every reuse-or-execute decision is
:func:`feedbax.analysis.fulfillment_adapters.fulfill_node`, every admission is
the uniform per-kind validator behind
:func:`~feedbax.analysis.fulfillment_adapters.admit_node`, and rebuild and repair
are :mod:`feedbax.analysis.fulfillment_custody`'s. This module reimplements none
of them.

What it owns is the *walk*: the order nodes are reached in, which admitted
receipt fills each declared input, and the refusals that stop the walk. What it
does not own is the *lowering* — which node request a project's node kind
compiles into. That reaches it as the :class:`NodeRequestLowering` callable the
caller supplies, so no project vocabulary is ever switched on here.

## The external boundary

Some producers are receipts this runner may consume but never make: a run only
another entrypoint launches, an acquisition that happens outside the process.
The lowerer marks such a node by setting :attr:`~.fulfillment_plan.PlanNode.boundary`,
and :func:`preflight` refuses the whole closure before any node of any branch
executes, naming each boundary node, the consumers that name it with their role
paths, and the subtree its receipt would unblock. Which nodes are boundaries is
the project's fact; that a boundary refuses everything is this module's rule.

## Missing receipts are refusals

An input the plan carries as an already-produced external receipt is resolved at
its canonical ``(kind, id, root)`` location, never through the manifest index,
which is derived acceleration. An external receipt that is absent is a
structured refusal (:class:`MissingExternalReceiptError`), never an
inapplicability: a missing receipt means not-yet-produced, wrong-root, or
corrupt, and each of those is its own outcome.

Execution is strictly serial and deterministic in order, as everything on the
feedbax fulfillment surface is.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Protocol

from feedbax.analysis.fulfillment import (
    FulfillmentAdmissionError,
    FulfillmentReceipt,
)
from feedbax.analysis.fulfillment_adapters import (
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
from feedbax.analysis.fulfillment_plan import (
    FulfillmentPlan,
    LogicalKey,
    OmissionApplier,
    OmissionRecord,
    PlanEdge,
    PlanNode,
    apply_certified_omissions,
    require_no_certified_omissions,
)
from feedbax.analysis.exact_parents import StagedExactParentEntry
from feedbax.analysis.manifest_inputs import authenticated_manifest_ref
from feedbax.contracts.manifest import (
    AnyManifest,
    ParentRef,
    canonical_manifest_path,
    load_manifest,
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
    """An input the plan carries as an already-produced receipt is not in the root.

    The plan states that some previous run produced this reference; the receipt
    root does not hold it at its canonical location. That is not-yet-copied,
    wrong-root, or deleted — never evidence that the input does not apply.
    """

    def __init__(
        self,
        kind: str,
        manifest_id: str,
        path: Path,
        *,
        consumer: LogicalKey | None = None,
        role_path: Sequence[str] = (),
    ) -> None:
        self.kind = kind
        self.manifest_id = manifest_id
        self.path = path
        self.consumer = consumer
        self.role_path = tuple(role_path)
        origin = (
            f"{consumer.text} declares input {list(self.role_path)} as the already-produced "
            if consumer is not None
            else "the plan names the already-produced "
        )
        super().__init__(
            f"{origin}{kind} {manifest_id!r}, but no completed receipt is stored at its "
            f"canonical location {path}. A missing receipt is not-yet-produced, wrong-root, or "
            "deleted; it never means the input does not apply."
        )


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


@dataclass(frozen=True)
class ClosureNode:
    """One executable node of a preflighted closure, with its prepared payload.

    ``payload`` is whatever the project's :class:`PreparePayload` returned for
    this node — typically its compiled document. Nothing here reads it; the
    lowering callable does.
    """

    key: LogicalKey
    plan_node: PlanNode
    payload: Any
    order: int

    @property
    def layer(self) -> str:
        return self.key.layer

    @property
    def kind(self) -> str:
        return self.plan_node.kind

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
    omissions: tuple[OmissionRecord, ...] = ()
    omit_certified: bool = False

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


class PreparePayload(Protocol):
    """Returns the per-node document a lowering will be given.

    This is where a project re-reads or re-derives what its plan node pinned. It
    is called once per node, in plan order, and never for a closure that has
    already refused.
    """

    def __call__(self, node: PlanNode) -> Any:  # pragma: no cover - protocol
        ...


@dataclass(frozen=True)
class NodeBinding:
    """Everything a lowering needs to bind one node's declared inputs.

    A lowering is handed the node, the plan, the required input edges in the
    plan's canonical order, and the receipts of the producers already walked. The
    two resolution helpers are the only sanctioned ways to turn an edge into an
    authenticated reference: :meth:`parent_ref` for a normal input and
    :meth:`exact_parent_entry` for one that also records where it was executed
    from.
    """

    closure: FulfillmentClosure
    environment: FulfillmentEnvironment
    receipts: Mapping[LogicalKey, FulfillmentReceipt] = field(default_factory=dict)

    @property
    def plan(self) -> FulfillmentPlan:
        return self.closure.plan

    def required_edges(self, key: LogicalKey) -> tuple[PlanEdge, ...]:
        """Return one node's required inputs, in the plan's canonical order."""
        return self.plan.required_edges(key)

    def producer_receipt(self, edge: PlanEdge) -> FulfillmentReceipt | None:
        """Return the admitted receipt one edge's in-closure producer wrote."""
        if edge.producer is None:
            return None
        return self.receipts[edge.producer]

    def parent_ref(
        self,
        edge: PlanEdge,
        *,
        role: str,
        kind: str | None = None,
        manifest_id: str | None = None,
    ) -> ParentRef:
        """Return the authenticated reference one required edge binds.

        An edge with a producer in this closure binds that producer's admitted
        receipt. An edge carrying an external record binds the receipt stored at
        its canonical location; the caller supplies the ``kind`` and
        ``manifest_id`` its own external vocabulary names, because what an
        external record means is the project's fact.
        """
        receipt = self.producer_receipt(edge)
        if receipt is not None:
            return receipt_parent_ref(receipt, role=role)
        return external_parent_ref(
            _require_external(edge, kind, manifest_id),
            role=role,
            root=self.environment.root,
            consumer=edge.consumer,
            role_path=edge.role_path,
        )[0]

    def exact_parent_entry(
        self,
        edge: PlanEdge,
        *,
        role: str,
        kind: str | None = None,
        manifest_id: str | None = None,
    ) -> StagedExactParentEntry:
        """Bind one required edge to the root-relative location it executes from."""
        receipt = self.producer_receipt(edge)
        if receipt is not None:
            return receipt_exact_parent_entry(receipt, role=role)
        parent, path = external_parent_ref(
            _require_external(edge, kind, manifest_id),
            role=role,
            root=self.environment.root,
            consumer=edge.consumer,
            role_path=edge.role_path,
        )
        return StagedExactParentEntry(
            parent=parent,
            execution_uri=path.relative_to(Path(self.environment.root)).as_posix(),
        )


class NodeRequestLowering(Protocol):
    """Lowers one closure node into the feedbax node request its kind executes.

    This is the driver's project-facing seam. The kernel never switches on a
    project's node kind, family, or layer; the lowering does, and returns one of
    feedbax's node requests. It is called once per node per walk, in dependency
    order, with the receipts of everything upstream already admitted.
    """

    def __call__(
        self, node: ClosureNode, *, binding: NodeBinding
    ) -> NodeRequest:  # pragma: no cover - protocol
        ...


def _require_external(
    edge: PlanEdge, kind: str | None, manifest_id: str | None
) -> tuple[str, str]:
    if kind is None or manifest_id is None:
        raise AmbiguousNodeReceiptError(
            f"{edge.consumer.text} declares input {list(edge.role_path)} as an external record "
            f"({edge.external!r}); binding it needs the manifest kind and id that record names, "
            "which only the lowering project can read"
        )
    return kind, manifest_id


def resolve_external_receipt(
    kind: str,
    manifest_id: str,
    *,
    root: Path | str,
    consumer: LogicalKey | None = None,
    role_path: Sequence[str] = (),
) -> tuple[AnyManifest, Path]:
    """Authenticate one already-produced receipt from its canonical location.

    The manifest index is derived acceleration and is never consulted: a receipt
    lives where feedbax's ``(kind, id, root)`` derivation says it lives, so an
    absent or stale index entry cannot change what this resolves.
    """
    path = canonical_manifest_path(kind, manifest_id, root=Path(root))
    if not path.is_file():
        raise MissingExternalReceiptError(
            kind, manifest_id, path, consumer=consumer, role_path=role_path
        )
    manifest = load_manifest(path)
    if manifest.kind != kind or manifest.id != manifest_id or manifest.status != "completed":
        raise MissingExternalReceiptError(
            kind, manifest_id, path, consumer=consumer, role_path=role_path
        )
    return manifest, path


def external_parent_ref(
    external: tuple[str, str],
    *,
    role: str,
    root: Path | str,
    consumer: LogicalKey | None = None,
    role_path: Sequence[str] = (),
) -> tuple[ParentRef, Path]:
    """Return the authenticated parent one ``(kind, manifest_id)`` pair names."""
    kind, manifest_id = external
    manifest, path = resolve_external_receipt(
        kind, manifest_id, root=root, consumer=consumer, role_path=role_path
    )
    return authenticated_manifest_ref(manifest, path, role), path


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


def preflight(
    plan: FulfillmentPlan,
    *,
    prepare: PreparePayload | Callable[[PlanNode], Any],
    content_hash: Callable[[Any], str] | None = None,
    omit_certified: bool = False,
    apply_omission: OmissionApplier | None = None,
) -> FulfillmentClosure:
    """Prepare one plan's closure and prove its external boundary.

    Nothing executes here and nothing is written. The order of the two refusals
    is fixed: the boundary is proved first, because a closure that cannot run at
    all is not made runnable by an applicability decision.

    Args:
        plan: The closure to preflight.
        prepare: Returns each node's payload, called once per node in plan order.
        content_hash: Hashes a prepared payload in the same domain the plan's
            ``content_hash`` was minted in. When given, a node whose payload no
            longer hashes to what the plan pinned refuses.
        omit_certified: Materialize the plan's certified ``not_applicable``
            decisions into the payloads that declare them, before any identity is
            minted from them. Without it, a plan carrying any such decision
            refuses with every certified decision named.
        apply_omission: How a payload says a decision was materialized. Required
            when *omit_certified* is set; it only ever sees certified edges.

    Raises:
        ExternalBoundaryError: The closure names a boundary node.
        CertifiedOmissionsPendingError: A node still declares an input the plan
            certified as not applicable, and *omit_certified* is off.
        PlanDocumentDriftError: A payload no longer hashes to what was pinned.
    """
    require_no_external_boundary(plan)
    if not omit_certified:
        require_no_certified_omissions(plan)
    elif apply_omission is None:
        raise ValueError(
            "omit_certified materializes certified decisions into a payload, so it needs an "
            "apply_omission that states how this project's document says one"
        )

    nodes: list[ClosureNode] = []
    materialized: list[OmissionRecord] = []
    for order, plan_node in enumerate(plan.nodes):
        payload = prepare(plan_node)
        if content_hash is not None and plan_node.content_hash is not None:
            observed = content_hash(payload)
            if observed != plan_node.content_hash:
                raise PlanDocumentDriftError(
                    plan_node.key, plan_node.source_ref, plan_node.content_hash, observed
                )
        if omit_certified and apply_omission is not None:
            payload, records = apply_certified_omissions(
                payload, plan, consumer=plan_node.key, apply=apply_omission
            )
            materialized.extend(records)
        nodes.append(
            ClosureNode(key=plan_node.key, plan_node=plan_node, payload=payload, order=order)
        )
    return FulfillmentClosure(
        plan=plan,
        nodes=tuple(nodes),
        omissions=tuple(materialized),
        omit_certified=omit_certified,
    )


def fulfill_closure(
    closure: FulfillmentClosure,
    *,
    environment: FulfillmentEnvironment,
    lower: NodeRequestLowering,
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
        request = _lowered(node, closure=closure, receipts=receipts, environment=environment,
                           lower=lower)
        result = fulfill_node(request, environment=environment)
        results.append(result)
        receipts[node.key] = _single_receipt(node, result)
    return FulfillmentRun(results=tuple(results))


def closure_requests(
    closure: FulfillmentClosure,
    *,
    environment: FulfillmentEnvironment,
    lower: NodeRequestLowering,
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
        request = _lowered(node, closure=closure, receipts=receipts, environment=environment,
                           lower=lower)
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
    lower: NodeRequestLowering,
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
    requests = closure_requests(closure, environment=environment, lower=lower)
    return rebuild_nodes(requests, environment=environment)


def repair_closure_node(
    closure: FulfillmentClosure,
    node_key: LogicalKey,
    *,
    environment: FulfillmentEnvironment,
    lower: NodeRequestLowering,
    metadata: Mapping[str, Any] | None = None,
) -> RepairResult:
    """Repair one node of a closure whose receipt exists but fails admission.

    Every node upstream of the subject is admitted first, because a repair
    candidate is only meaningful when the inputs it binds are the authenticated
    ones. The repair itself — quarantine, shadow execution, complete
    revalidation, promotion, durable repair record — is feedbax's.
    """
    requests = closure_requests(
        closure, environment=environment, lower=lower, stop_at=node_key
    )
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
    lower: NodeRequestLowering,
) -> NodeRequest:
    binding = NodeBinding(closure=closure, environment=environment, receipts=dict(receipts))
    request = lower(node, binding=binding)
    if request.node_key != node.key.text:
        raise AmbiguousNodeReceiptError(
            f"the lowering for {node.key.text} returned a request keyed {request.node_key!r}; a "
            "node request carries the logical key of the node it fulfills, so a walk's results "
            "address the plan they came from"
        )
    return request


def _single_receipt(node: ClosureNode, result: NodeFulfillment) -> FulfillmentReceipt:
    """Return the one receipt a node's consumers may bind, or refuse the ambiguity."""
    if len(result.receipts) == 1:
        return result.receipts[0]
    raise AmbiguousNodeReceiptError(
        f"node {node.key.text} produced {len(result.receipts)} receipts; a consumer binds one "
        f"authenticated reference per declared input, and this version does not choose among "
        f"the receipts of a multi-receipt {node.kind!r} node"
    )


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
    outcome = admit_node(request, environment=environment)
    if not outcome.admitted or outcome.manifest_path is None:
        raise FulfillmentAdmissionError(outcome)
    path = Path(outcome.manifest_path)
    return FulfillmentReceipt(
        node_kind=request.node_kind,
        manifest=load_manifest(path),
        path=path,
        root=Path(environment.root),
    )
