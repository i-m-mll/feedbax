"""Deriving a fulfillment plan from compiled experiment outputs.

One compile emits two files: the compiled document, and the
:mod:`~feedbax.contracts.experiment_compile_lock` that pins what the compile read
and decided. This module turns a directory of such pairs into a
:class:`~feedbax.analysis.fulfillment_plan.FulfillmentPlan`, mechanically.

## Nothing is inferred, and nothing is asked

A plan edge is decided by the *type* of a lock reference and the *type* of its
consumer binding, both of which are closed unions Feedbax owns. There is no
edge-builder callback, no per-project dispatch, and no reading of
``resolved_deltas``: a compile that wants something downstream says so as a typed
reference, or the plan does not carry it.

The five reference kinds land in exactly three places:

* :class:`~feedbax.contracts.experiment_compile_lock.ContentPinReference` is not
  an edge at all — it records bytes the compile *read* — and is filtered out by
  :func:`~feedbax.contracts.experiment_compile_lock.compile_lock_plan_edges`
  before this module sees it;
* :class:`~feedbax.contracts.experiment_compile_lock.PlannedProductReference` is
  an edge to another node *of this closure*, resolved through the upstream
  envelope's own compiled output. Both pre-run pins the reference carries — the
  upstream envelope's hash and the content hash of the document it compiles into
  — are checked against that output, so a stale reference refuses instead of
  binding the wrong document;
* the two receipt kinds are edges to something already produced outside the
  closure, carried on the edge as an ``external`` record, and
  :class:`~feedbax.contracts.experiment_compile_lock.NotApplicableReference` is a
  decision that binds nothing.

## Layers are the compiled document's own identity

A node's layer is read from the compiled document's ``schema_id`` through
:data:`COMPILED_PRODUCT_KINDS`, never from a label the authored envelope carries.
Both sides of that table are Feedbax-owned, so it is Feedbax code: a project
names no layer here, and a ``schema_id`` the table does not enumerate refuses
with the supported set named.

## Training is a boundary

A compiled training run matrix is never executed by artifact fulfillment. It is
launched through its own orchestration entrypoint, and fulfillment consumes the
receipts that launch produced. It is therefore derived as a *boundary* node
(:data:`TRAINING_MATRIX_BOUNDARY`), which
:func:`~feedbax.analysis.fulfillment_driver.preflight` refuses before any node of
any branch runs. Once training has run, the consuming envelope quotes an
authenticated receipt instead of a planned product, and the closure no longer
names a boundary.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from feedbax.analysis.fulfillment_plan import (
    EdgeDeclaration,
    FulfillmentPlan,
    LogicalKey,
    NodeDeclaration,
    PlanNode,
    expand_fulfillment_plan,
)
from feedbax.contracts.analysis_bundle_composition import ANALYSIS_BUNDLE_SPEC_SCHEMA_ID
from feedbax.contracts.authored_canonical import canonical_sha256
from feedbax.contracts.experiment_compile_lock import (
    AuthenticatedReceiptReference,
    NotApplicableReference,
    PlannedProductReference,
    ReceiptLocatorReference,
    compile_lock_plan_edges,
    load_compile_lock,
)
from feedbax.contracts.figures import (
    FIGURE_COMPOSITION_SPEC_SCHEMA_ID,
    FIGURE_SPEC_SCHEMA_ID,
)
from feedbax.contracts.manifest import (
    ANALYSIS_RUN_SPEC_SCHEMA_ID,
    EVALUATION_RUN_MATRIX_SPEC_SCHEMA_ID,
    EVALUATION_RUN_SPEC_SCHEMA_ID,
    REPORT_SPEC_SCHEMA_ID,
)
from feedbax.contracts.run_matrix import TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID

#: The filename suffix one compile writes its lock under. It is the emitter's
#: own naming, restated here so a reader can find locks without a compiler.
COMPILE_LOCK_SUFFIX = ".compile-lock.json"

#: The boundary name a compiled training run matrix is derived under. It is
#: Feedbax's own spec identity, not a project's word for "training".
TRAINING_MATRIX_BOUNDARY = TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID


@dataclass(frozen=True)
class CompiledProductKind:
    """What one compiled ``schema_id`` is, for planning purposes.

    Attributes:
        schema_id: The compiled document's durable schema identity.
        layer: The Feedbax artifact layer it belongs to. This is the ``layer``
            half of every :class:`~feedbax.analysis.fulfillment_plan.LogicalKey`
            derived for it.
        boundary: When set, the node is never executed by artifact fulfillment
            and is derived as a boundary carrying this name.
    """

    schema_id: str
    layer: str
    boundary: str | None = None

    @property
    def executable(self) -> bool:
        """Whether artifact fulfillment may execute a node of this kind."""
        return self.boundary is None


#: Compiled ``schema_id`` to the layer it belongs to. Both sides are
#: Feedbax-owned, so this table is Feedbax code; a project contributes no row and
#: an unlisted schema id refuses rather than being lowered by resemblance.
COMPILED_PRODUCT_KINDS: Mapping[str, CompiledProductKind] = {
    kind.schema_id: kind
    for kind in (
        CompiledProductKind(
            TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID, "training", boundary=TRAINING_MATRIX_BOUNDARY
        ),
        CompiledProductKind(EVALUATION_RUN_SPEC_SCHEMA_ID, "evaluation"),
        CompiledProductKind(EVALUATION_RUN_MATRIX_SPEC_SCHEMA_ID, "evaluation"),
        CompiledProductKind(ANALYSIS_RUN_SPEC_SCHEMA_ID, "analysis"),
        CompiledProductKind(ANALYSIS_BUNDLE_SPEC_SCHEMA_ID, "analysis"),
        CompiledProductKind(FIGURE_SPEC_SCHEMA_ID, "figure"),
        CompiledProductKind(FIGURE_COMPOSITION_SPEC_SCHEMA_ID, "figure"),
        CompiledProductKind(REPORT_SPEC_SCHEMA_ID, "report"),
    )
}

#: Every layer a compiled product can belong to, in the union's declared order.
EXPERIMENT_LAYERS: tuple[str, ...] = ("training", "evaluation", "analysis", "figure", "report")


class FulfillmentDerivationError(RuntimeError):
    """Base class for the structured refusals plan derivation raises."""


class UnsupportedCompiledProductError(FulfillmentDerivationError):
    """A compiled document declares a schema this build cannot plan against."""

    def __init__(self, schema_id: Any, *, ref: str) -> None:
        self.schema_id = schema_id
        self.ref = ref
        super().__init__(
            f"{ref} compiles to schema_id {schema_id!r}, which is not a Feedbax artifact "
            f"layer product; supported={sorted(COMPILED_PRODUCT_KINDS)}. A compiled document "
            "is planned by its own declared identity, never by resemblance or by a label the "
            "authored envelope carries."
        )


class CompiledOutputError(FulfillmentDerivationError):
    """A compiled output is absent, unreadable, or disagrees with its lock."""


class UnresolvedPlannedProductError(FulfillmentDerivationError):
    """A lock names an upstream product no compiled output in the directory holds."""

    def __init__(self, reference: PlannedProductReference, *, consumer_ref: str) -> None:
        self.reference = reference
        self.consumer_ref = consumer_ref
        super().__init__(
            f"{consumer_ref} references the product of {reference.envelope_ref!r} at role "
            f"{reference.role_path!r}, but no compile lock in this directory was emitted for "
            "that envelope. Compile the upstream envelope into the same output directory; a "
            "planned product is an edge to a document this closure carries."
        )


class DuplicateReferenceRoleError(FulfillmentDerivationError):
    """One lock states two references for the same role path."""

    def __init__(self, role_path: str, *, ref: str) -> None:
        self.role_path = role_path
        self.ref = ref
        super().__init__(
            f"{ref} states two references at role {role_path!r}; a role addresses exactly one "
            "input, and a plan cannot bind a role that means two things"
        )


@dataclass(frozen=True)
class CompiledEnvelope:
    """One compile's two emitted files, read back and cross-checked.

    The document is verified against the hash its own lock pinned at read time,
    so every consumer of this record is reading the document the lock describes.
    """

    lock: Mapping[str, Any]
    document: Mapping[str, Any]
    lock_path: Path
    document_path: Path

    @property
    def name(self) -> str:
        """The compiled output's name."""
        return str(self.lock["name"])

    @property
    def envelope_ref(self) -> str:
        """The repo-relative path of the envelope this was compiled from."""
        return str(self.lock["envelope"]["ref"])

    @property
    def envelope_hash(self) -> str:
        """The authored envelope's own content hash."""
        return str(self.lock["envelope"]["envelope_hash"])

    @property
    def family(self) -> str:
        """The output family the compiled document belongs to."""
        return str(self.lock["compiled_document"]["family"])

    @property
    def content_hash(self) -> str:
        """The compiled document's pinned content hash."""
        return str(self.lock["compiled_document"]["content_hash"])

    @property
    def execution_identity(self) -> str:
        """The execution identity this compile minted."""
        return str(self.lock["execution_identity"]["sha256"])

    @property
    def schema_id(self) -> str:
        """The compiled document's own durable schema identity."""
        declared = self.document.get("schema_id")
        if not isinstance(declared, str) or not declared:
            raise CompiledOutputError(
                f"{self.document_path} declares no schema_id; a compiled document is planned "
                "and lowered by its own durable identity, which it must state"
            )
        return declared

    @property
    def kind(self) -> CompiledProductKind:
        """The layer this compiled document belongs to, or a refusal."""
        found = COMPILED_PRODUCT_KINDS.get(self.schema_id)
        if found is None:
            raise UnsupportedCompiledProductError(self.schema_id, ref=str(self.document_path))
        return found

    @property
    def key(self) -> LogicalKey:
        """The logical address of the plan node this compile produces."""
        return LogicalKey(self.kind.layer, self.name)

    def plan_edge_references(self) -> tuple[Any, ...]:
        """Return this lock's references that are plan edges, typed and in order.

        Content pins are filtered out by kind. Two references claiming one role
        refuse here, so every later pairing of a role to a reference is exact.
        """
        references = compile_lock_plan_edges(self.lock, field=str(self.lock_path))
        seen: set[str] = set()
        for reference in references:
            role_path = str(reference.role_path)
            if role_path in seen:
                raise DuplicateReferenceRoleError(role_path, ref=str(self.lock_path))
            seen.add(role_path)
        return references


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CompiledOutputError(f"cannot read compiled output {path}: {exc}") from exc


def read_compiled_envelope(lock_path: Path) -> CompiledEnvelope:
    """Read one emitted lock and the compiled document it pins.

    The document's location follows from the lock's own ``name`` and ``family``,
    which is how the compile emitted it. Its bytes are re-hashed and compared
    with the pin, so a document edited after its compile refuses here rather than
    executing under a plan that described something else.
    """
    lock = load_compile_lock(_read_json(lock_path), field=str(lock_path))
    document_path = lock_path.parent / f"{lock['name']}.{lock['compiled_document']['family']}.json"
    if not document_path.is_file():
        raise CompiledOutputError(
            f"{lock_path} pins the compiled document {document_path.name}, which is not in "
            f"{lock_path.parent}; a lock and the document it pins are emitted together"
        )
    document = _read_json(document_path)
    if not isinstance(document, Mapping):
        raise CompiledOutputError(f"{document_path} is not a JSON object")
    observed = canonical_sha256(document)
    expected = str(lock["compiled_document"]["content_hash"])
    if observed != expected:
        raise CompiledOutputError(
            f"{document_path} hashes to {observed} but {lock_path.name} pinned {expected}; the "
            "compiled document changed after the compile that locked it"
        )
    return CompiledEnvelope(
        lock=lock, document=document, lock_path=lock_path, document_path=document_path
    )


@dataclass(frozen=True)
class CompiledOutputIndex:
    """Every compiled output in one directory, addressed as a compile names them."""

    directory: Path
    envelopes: tuple[CompiledEnvelope, ...]

    def by_envelope_ref(self) -> dict[str, CompiledEnvelope]:
        """Return the outputs keyed by the envelope each was compiled from."""
        return {compiled.envelope_ref: compiled for compiled in self.envelopes}

    def require(self, envelope_ref: str) -> CompiledEnvelope:
        """Return the output compiled from one envelope, or refuse."""
        found = self.by_envelope_ref().get(envelope_ref)
        if found is None:
            raise CompiledOutputError(
                f"no compile lock in {self.directory} was emitted for envelope "
                f"{envelope_ref!r}; known envelopes: "
                f"{sorted(self.by_envelope_ref())}"
            )
        return found

    def resolve_target(self, target: str) -> CompiledEnvelope:
        """Return the output one target names, by envelope ref or compiled name.

        Both spellings address exactly one output. A name that matches two
        outputs refuses rather than picking one.
        """
        direct = self.by_envelope_ref().get(target)
        if direct is not None:
            return direct
        named = [compiled for compiled in self.envelopes if compiled.name == target]
        if len(named) == 1:
            return named[0]
        if len(named) > 1:
            raise CompiledOutputError(
                f"{target!r} names {len(named)} compiled outputs in {self.directory}: "
                f"{sorted(compiled.envelope_ref for compiled in named)}; address the one you "
                "mean by its envelope path"
            )
        raise CompiledOutputError(
            f"{target!r} names neither an envelope nor a compiled output in {self.directory}; "
            f"known names: {sorted(compiled.name for compiled in self.envelopes)}"
        )


def read_compiled_outputs(directory: Path | str) -> CompiledOutputIndex:
    """Read every compiled lock/document pair in one output directory.

    Reading is exhaustive and eager: every lock in the directory is admitted and
    every document it pins is verified, so a plan derived from this index cannot
    be surprised later by a lock the walk never opened.
    """
    root = Path(directory)
    if not root.is_dir():
        raise CompiledOutputError(f"{root} is not a directory of compiled experiment outputs")
    locks = sorted(root.glob(f"*{COMPILE_LOCK_SUFFIX}"))
    envelopes = tuple(read_compiled_envelope(path) for path in locks)
    refs: dict[str, Path] = {}
    for compiled in envelopes:
        previous = refs.get(compiled.envelope_ref)
        if previous is not None:
            raise CompiledOutputError(
                f"{previous.name} and {compiled.lock_path.name} were both emitted for envelope "
                f"{compiled.envelope_ref!r}; one envelope compiles to one output"
            )
        refs[compiled.envelope_ref] = compiled.lock_path
    return CompiledOutputIndex(directory=root, envelopes=envelopes)


def _role_path_parts(role_path: str) -> tuple[str, ...]:
    return tuple(role_path.split("."))


def _external_record(
    reference: ReceiptLocatorReference | AuthenticatedReceiptReference,
) -> dict[str, Any]:
    """Return the edge record one already-produced receipt reference carries.

    Both receipt kinds resolve the same way — at the receipt's canonical
    ``(kind, id, root)`` location — so both carry the same two addressing fields.
    An authenticated reference additionally carries the byte profile the compile
    quoted, which the driver checks against the bytes it resolves.
    """
    record: dict[str, Any] = {
        "manifest_kind": reference.manifest_kind,
        "manifest_id": reference.manifest_id,
    }
    if isinstance(reference, AuthenticatedReceiptReference):
        record["manifest_sha256"] = reference.manifest_sha256
        record["size_bytes"] = reference.size_bytes
        if reference.execution_uri is not None:
            record["execution_uri"] = reference.execution_uri
    return record


def _edge_declaration(reference: Any, *, consumer: CompiledEnvelope) -> EdgeDeclaration:
    """Return the one edge a typed reference determines. No other input decides it."""
    role_path = _role_path_parts(str(reference.role_path))
    if isinstance(reference, PlannedProductReference):
        return EdgeDeclaration(
            role_path=role_path, status="required", basis="authored",
            producer_ref=reference.envelope_ref,
        )
    if isinstance(reference, (ReceiptLocatorReference, AuthenticatedReceiptReference)):
        return EdgeDeclaration(
            role_path=role_path, status="required", basis="authored",
            external=_external_record(reference),
        )
    if isinstance(reference, NotApplicableReference):
        return EdgeDeclaration(
            role_path=role_path,
            status="not_applicable",
            basis=reference.basis,
            reason=reference.reason,
            rule=reference.rule_id,
        )
    raise UnsupportedCompiledProductError(
        type(reference).__name__, ref=str(consumer.lock_path)
    )


def lock_edge_declarations(compiled: CompiledEnvelope) -> tuple[EdgeDeclaration, ...]:
    """Return the edges one compile lock, by itself, determines for its node.

    This is the same derivation :func:`derive_fulfillment_plan` performs, exposed
    so a *later* stage can re-derive it and compare. A plan is a durable document
    and travels apart from the locks it was derived from; re-deriving from the
    lock is the only way to prove that what a plan says a node's inputs are is
    still what the lock says they are.
    """
    return tuple(
        _edge_declaration(reference, consumer=compiled)
        for reference in compiled.plan_edge_references()
    )


def _check_planned_product(
    reference: PlannedProductReference,
    *,
    consumer: CompiledEnvelope,
    upstream: CompiledEnvelope,
) -> None:
    """Prove the upstream output is the one the reference pinned before the run.

    A planned product reference states three facts that exist pre-run: the
    upstream envelope's hash, the content hash of the document it compiles into,
    and that document's schema identity. All three are checked, so a reference
    that has gone stale refuses instead of binding whatever now sits at the ref.
    """
    mismatches: list[str] = []
    if upstream.envelope_hash != reference.envelope_hash:
        mismatches.append(
            f"envelope_hash pinned {reference.envelope_hash} but {upstream.lock_path.name} "
            f"records {upstream.envelope_hash}"
        )
    if upstream.content_hash != reference.compiled_content_hash:
        mismatches.append(
            f"compiled_content_hash pinned {reference.compiled_content_hash} but "
            f"{upstream.document_path.name} hashes to {upstream.content_hash}"
        )
    if upstream.schema_id != reference.product_schema_id:
        mismatches.append(
            f"product_schema_id pinned {reference.product_schema_id!r} but "
            f"{upstream.document_path.name} declares {upstream.schema_id!r}"
        )
    if upstream.name != reference.product_name:
        mismatches.append(
            f"product_name pinned {reference.product_name!r} but the upstream compiled output "
            f"is named {upstream.name!r}"
        )
    if mismatches:
        raise CompiledOutputError(
            f"{consumer.lock_path.name} references {reference.envelope_ref!r} at role "
            f"{reference.role_path!r} with pins that no longer hold: {'; '.join(mismatches)}. "
            "Recompile the closure so every reference pins what it actually reaches."
        )


def derive_fulfillment_plan(
    index: CompiledOutputIndex,
    *,
    target: str,
    origin: Mapping[str, Any] | None = None,
) -> FulfillmentPlan:
    """Derive one target's fulfillment plan from the compiled outputs it reaches.

    Every node is one compiled output, addressed by its layer and its compiled
    name. Every edge is one typed lock reference. Nothing else contributes: no
    project callable participates, and no field outside the closed reference
    union is read.

    Args:
        index: The compiled outputs the closure is drawn from.
        target: The envelope path or compiled name of the artifact to fulfill.
        origin: Extra provenance to record on the plan, merged under the
            compiler identity this derivation always records.
    """
    root = index.resolve_target(target)

    def expand(source_ref: str) -> NodeDeclaration:
        compiled = index.require(source_ref)
        kind = compiled.kind
        edges: list[EdgeDeclaration] = []
        for reference in compiled.plan_edge_references():
            if isinstance(reference, PlannedProductReference):
                upstream = index.by_envelope_ref().get(reference.envelope_ref)
                if upstream is None:
                    raise UnresolvedPlannedProductError(reference, consumer_ref=source_ref)
                _check_planned_product(reference, consumer=compiled, upstream=upstream)
            edges.append(_edge_declaration(reference, consumer=compiled))
        return NodeDeclaration(
            node=PlanNode(
                key=compiled.key,
                source_ref=source_ref,
                kind=compiled.schema_id,
                content_hash=compiled.content_hash,
                execution_identity=compiled.execution_identity,
                boundary=kind.boundary,
                metadata={
                    "family": compiled.family,
                    "envelope_hash": compiled.envelope_hash,
                    "compiled_name": compiled.name,
                },
            ),
            edges=tuple(edges),
        )

    return expand_fulfillment_plan(
        root.envelope_ref,
        expand=expand,
        origin={
            "target_envelope_ref": root.envelope_ref,
            "compiler_contract": dict(root.lock["compiler_contract"]),
            "compiler_implementation": dict(root.lock["compiler_implementation"]),
            **dict(origin or {}),
        },
    )


@dataclass(frozen=True)
class ExternalReceiptRecord:
    """What one external edge states about the receipt it binds.

    The two addressing fields say *where* the receipt is; the byte profile, when
    the lock quoted one, says *which bytes* are allowed to be there. Both travel
    together because they are one statement: a lock that authenticated a receipt
    named bytes, and resolving the canonical location without checking them would
    turn an authenticated quote back into a bare locator.
    """

    manifest_kind: str
    manifest_id: str
    manifest_sha256: str | None = None
    size_bytes: int | None = None

    @property
    def is_authenticated(self) -> bool:
        """Whether the lock quoted the byte profile the resolved bytes must have."""
        return self.manifest_sha256 is not None

    @property
    def locator(self) -> tuple[str, str]:
        """The canonical ``(kind, id)`` location this record addresses."""
        return self.manifest_kind, self.manifest_id


def require_external_record(
    external: Mapping[str, Any] | None, *, field: str
) -> ExternalReceiptRecord:
    """Return the receipt one external edge record addresses, or refuse.

    The record's shape is Feedbax's, derived from a Feedbax reference kind, so
    reading it is Feedbax's too. A record missing either addressing field is a
    derivation bug, not something a caller supplies by hand, and so is a
    half-stated byte profile: the lock's own reference model states the digest and
    the size together or neither, so a record with one of them describes a
    reference kind that does not exist.
    """
    if external is None:
        raise CompiledOutputError(
            f"{field} binds neither a producer in this closure nor an already-produced receipt"
        )
    kind = external.get("manifest_kind")
    manifest_id = external.get("manifest_id")
    if not isinstance(kind, str) or not isinstance(manifest_id, str):
        raise CompiledOutputError(
            f"{field} carries the external record {dict(external)!r}, which names no manifest "
            "kind and id; every receipt reference kind states both"
        )
    digest = external.get("manifest_sha256")
    size = external.get("size_bytes")
    if (digest is None) != (size is None):
        raise CompiledOutputError(
            f"{field} carries the external record {dict(external)!r}, which states half a byte "
            "profile; an authenticated receipt reference states manifest_sha256 and size_bytes "
            "together, and a locator states neither"
        )
    if digest is not None and (not isinstance(digest, str) or not isinstance(size, int)):
        raise CompiledOutputError(
            f"{field} carries the external record {dict(external)!r}, whose byte profile is not "
            "a sha256 string and an integer size"
        )
    return ExternalReceiptRecord(
        manifest_kind=kind,
        manifest_id=manifest_id,
        manifest_sha256=digest,
        size_bytes=size,
    )


__all__ = [
    "COMPILED_PRODUCT_KINDS",
    "COMPILE_LOCK_SUFFIX",
    "EXPERIMENT_LAYERS",
    "TRAINING_MATRIX_BOUNDARY",
    "CompiledEnvelope",
    "CompiledOutputError",
    "CompiledOutputIndex",
    "CompiledProductKind",
    "DuplicateReferenceRoleError",
    "ExternalReceiptRecord",
    "FulfillmentDerivationError",
    "UnresolvedPlannedProductError",
    "UnsupportedCompiledProductError",
    "derive_fulfillment_plan",
    "lock_edge_declarations",
    "read_compiled_envelope",
    "read_compiled_outputs",
    "require_external_record",
]
