"""The compile lock: everything knowable before a run is allocated.

The custody boundary has exactly two artifacts and neither is hand-authored:

* the **compile lock**, emitted at preflight, pins what the compiler read and
  decided — the envelope, its content-pinned parent lineage, the resolved
  deltas, the compiled document's hash, and the compiler provenance;
* the **run receipt**, which the manifest layer already owns, records what only
  running can produce — allocation, run and transaction ids, output hashes, and
  checkpoints.

Execution and resume consume both. A lock is never composed with another lock and
never gains a post-allocation fact. If one is ever needed here, the boundary is
drawn wrong and the design, not this module, is what must change.
:func:`check_plan_receipt_boundary` is the single machine-checkable statement of
that rule, and it runs on every lock the engine builds.

## Compile plans and execution receipts

A cross-layer reference resolves in two phases. At compile, the lock pins the
upstream **envelope** and the content hash of the **compiled document** it
produces: everything that exists before anything runs. An **authenticated**
reference — one carrying a manifest digest and size — can only be produced by a
run. The compiler may *quote* a receipt a previous run already wrote, and may
never *author* one. A compiled document that still needs a post-run input is
therefore a plan, not an executable spec, until its receipt exists.

## Compiler provenance is two facts, not one

A lock records provenance in two separate blocks because they answer different
questions and change on different schedules:

* :class:`CompilerContract` is the **logical** contract the compiled output
  conforms to — the global ``compiler_contract_id`` and
  ``compiler_contract_version`` of the one envelope compiler. It changes when the
  meaning of a compiled document changes, and it is what a consumer checks
  before trusting the output's shape.
* :class:`CompilerImplementation` is the **physical** provenance of the code that
  ran — the package that hosts the compiler, its version, and the versions of the
  packages it was built against. It changes whenever anything is released, and it
  is what an investigator uses to attribute a byte-level difference.

Collapsing the two into one string forces a logical version bump for every
release, or hides a release behind an unchanged logical version. Keeping them
apart lets a recompilation that changes derived bytes be attributable without
implying that the contract moved.

## References are a closed typed union

Everything one compile resolved about a *different* document is recorded in
``references``, and every entry is one member of :data:`CompileLockReference`.
The union is closed because the downstream plan is derived from it mechanically:
a free-form mapping would make "is this an edge?" a question about string keys
rather than about the record's own kind. The six kinds answer six genuinely
different questions:

* :class:`ContentPinReference` — bytes this compile *read*. It is a compile-time
  input and is never a plan edge; nothing runs because of it.
* :class:`GovernedParentReference` — immutable external parent bytes this
  compile authenticated, with their complete semantic and artifact identity.
  It is compile-time input and never a plan edge.
* :class:`PlannedProductReference` — a product another envelope will compile to,
  pinned by the upstream envelope and by the content hash of what it compiles
  into. This is the ordinary pre-run edge.
* :class:`ReceiptLocatorReference` — a manifest named by kind and id with **no**
  digest. The compiler knows *which* receipt is wanted before that receipt
  exists; pretending otherwise would either fabricate a digest or force the
  reference out of the lock entirely.
* :class:`AuthenticatedReceiptReference` — a manifest a previous run really
  wrote, quoted with its digest and size. The compiler may quote one and may
  never author one.
* :class:`NotApplicableReference` — a role that is deliberately unfilled, on an
  authored basis or under a versioned Feedbax structural rule. Silence and
  not-applicability are different facts and are recorded differently.

Every kind except the content pin also states *who consumes it*, as one member
of :data:`CompileLockConsumerBinding`. The consumer vocabulary is Feedbax-owned
and closed: an evaluation subject, an analysis alias/role input, a figure runtime
input authority, an exact report parent, or a checkpoint initialization.

## Row provenance is a refinement of the parent pin, not a reference

``references`` answers "what did this compile resolve about some *other*
document?", and every answer there is either a compile-time input or a plan edge.
A row derived from a parent row is neither: the document it comes from is already
the one named in ``base`` and walked in ``lineage``, and nothing runs because of
it. What is missing from those two blocks is only *which row inside the parent*
each derived row descends from, so that is what ``row_provenance`` states, as
:class:`RowProvenanceReference` records. Keeping it out of the closed union is
deliberate: the plan lane derives edges from ``references`` by kind, and a sixth
consumer-less kind would make "is this an edge?" a question again.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field as dataclass_field
from importlib.metadata import PackageNotFoundError, version as package_version
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import Field, TypeAdapter, ValidationError, model_validator

from feedbax.contracts.authored_canonical import (
    CANONICAL_PIN_ALGORITHM,
    canonical_sha256,
)
from feedbax.contracts.checkpoint_initialization import CheckpointInitializationMode
from feedbax.contracts.experiment_envelope import (
    ExperimentEnvelopeRejection,
    ExperimentEnvelopeRejectionCategory,
)
from feedbax.contracts.experiment_envelope_dialect import (
    EXPERIMENT_ENVELOPE_COMPILER_CONTRACT_ID,
)
from feedbax.contracts.figure_roles import FigureRoleBindingContract
from feedbax.contracts.base import StrictModel
from feedbax.contracts.run_composition import AuthoredIntentParent, ResolvedOutputParent

EXPERIMENT_COMPILE_LOCK_SCHEMA_ID = "feedbax.spec.experiment_compile_lock"
EXPERIMENT_COMPILE_LOCK_SCHEMA_VERSION_V1 = f"{EXPERIMENT_COMPILE_LOCK_SCHEMA_ID}.v1"
EXPERIMENT_COMPILE_LOCK_SCHEMA_VERSION_V2 = f"{EXPERIMENT_COMPILE_LOCK_SCHEMA_ID}.v2"
EXPERIMENT_COMPILE_LOCK_SCHEMA_VERSION_V3 = f"{EXPERIMENT_COMPILE_LOCK_SCHEMA_ID}.v3"
EXPERIMENT_COMPILE_LOCK_SCHEMA_VERSION_V4 = f"{EXPERIMENT_COMPILE_LOCK_SCHEMA_ID}.v4"
EXPERIMENT_COMPILE_LOCK_SCHEMA_VERSION = EXPERIMENT_COMPILE_LOCK_SCHEMA_VERSION_V4

#: The only lock versions read. Enumerated, never inferred. v1 remains readable
#: as exactly the grammar it names: a lock recorded before figure runtime input
#: bindings could carry a typed artifact contract still describes a real compile,
#: and the nodes it binds still execute — a root figure input bound by a v1
#: reference is refused at lowering, by name, rather than by version here.
EXPERIMENT_COMPILE_LOCK_SUPPORTED_SCHEMA_VERSIONS: tuple[str, ...] = (
    EXPERIMENT_COMPILE_LOCK_SCHEMA_VERSION_V1,
    EXPERIMENT_COMPILE_LOCK_SCHEMA_VERSION_V2,
    EXPERIMENT_COMPILE_LOCK_SCHEMA_VERSION_V4,
)

#: Versions this loader migrates, mapped to their explicit target. v1 and v2
#: remain readable as exactly their own grammars. Only v3 was emitted by the
#: current Feedbax encoder without an execution-identity algorithm pin, so only
#: provenance-attributable v3 locks have a migration edge.
EXPERIMENT_COMPILE_LOCK_MIGRATION_TABLE: dict[str, str] = {
    EXPERIMENT_COMPILE_LOCK_SCHEMA_VERSION_V3: EXPERIMENT_COMPILE_LOCK_SCHEMA_VERSION_V4,
}

#: Facts that only exist after a run is allocated. They belong to the run
#: receipt; a lock must never grow a key from this set, at the top level or
#: inside its identity contributions.
RUN_RECEIPT_ONLY_FACTS = frozenset(
    {
        "run_id",
        "run_ids",
        "transaction_id",
        "allocation",
        "provider_allocation",
        "checkpoint_root",
        "checkpoints",
        "output_hashes",
        "orchestration_root",
        "keep_alive",
    }
)


class CompileLockError(ValueError):
    """Raised when a compile lock cannot be built or read."""


class PlanReceiptBoundaryError(AssertionError):
    """Raised when a compile lock has absorbed a post-allocation fact."""


def check_plan_receipt_boundary(lock: Mapping[str, Any]) -> None:
    """Refuse a compile lock that has absorbed a post-allocation fact.

    Both the lock's own keys and the keys of its identity contributions are
    checked, so a receipt fact cannot enter through the extension slot that a
    project uses to widen execution identity.
    """
    conflicts = set(RUN_RECEIPT_ONLY_FACTS) & set(lock)
    contributions = lock.get("identity_contributions")
    if isinstance(contributions, Mapping):
        conflicts |= set(RUN_RECEIPT_ONLY_FACTS) & set(contributions)
    if conflicts:
        raise PlanReceiptBoundaryError(
            f"compile lock absorbed post-allocation facts {sorted(conflicts)}; "
            "these belong to the run receipt, not to a compile plan"
        )


def _installed_version(name: str) -> str | None:
    try:
        return package_version(name)
    except PackageNotFoundError:
        return None


# -- the closed typed reference union -------------------------------------

_DIGEST_RE = re.compile(r"[0-9a-f]{64}")
_VERSIONED_RULE_RE = re.compile(r".+\.v[0-9]+$")


def _require_nonempty(value: str, name: str) -> str:
    if not value.strip():
        raise ValueError(f"{name} must be nonempty")
    return value


def _require_role_path(value: str, name: str) -> str:
    _require_nonempty(value, name)
    if any(not part for part in value.split(".")):
        raise ValueError(f"{name} is not a dotted role path: {value!r}")
    return value


def _require_digest(value: str, name: str) -> str:
    if not _DIGEST_RE.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase sha256 digest")
    return value


class EvaluationSubjectBinding(StrictModel):
    """The referenced product is the subject an evaluation evaluates."""

    consumer: Literal["evaluation_subject"] = "evaluation_subject"
    subject_id: str

    @model_validator(mode="after")
    def _validate(self) -> "EvaluationSubjectBinding":
        _require_nonempty(self.subject_id, "evaluation_subject subject_id")
        return self


class AnalysisInputBinding(StrictModel):
    """The referenced product fills one analysis input, addressed by alias and role."""

    consumer: Literal["analysis_input"] = "analysis_input"
    alias: str
    role: str

    @model_validator(mode="after")
    def _validate(self) -> "AnalysisInputBinding":
        _require_nonempty(self.alias, "analysis_input alias")
        _require_nonempty(self.role, "analysis_input role")
        return self


class AnalysisReceiptSetBinding(StrictModel):
    """The referenced product fills one analysis input with every receipt it produced."""

    consumer: Literal["analysis_receipt_set"] = "analysis_receipt_set"
    alias: str
    role: str

    @model_validator(mode="after")
    def _validate(self) -> "AnalysisReceiptSetBinding":
        _require_nonempty(self.alias, "analysis_receipt_set alias")
        _require_nonempty(self.role, "analysis_receipt_set role")
        return self


class FigureRuntimeInputBinding(StrictModel):
    """The referenced product satisfies one figure runtime input authority.

    ``input_role`` is the role a
    :class:`~feedbax.contracts.figures.FigureInputRoleAuthority` addresses its
    single exact parent by; this binding is the compile-time statement of the
    same role.

    ``contract`` is the closed artifact contract the envelope authored for that
    role: which artifact of the bound manifest is read, from which provider, at
    which media type, and under which decoded payload identity and name. It is
    the durable half of the fix for a root figure that recorded its inputs and
    then rendered with no authority over them at all — the lowering builds the
    runtime :class:`~feedbax.contracts.figures.FigureInputRoleAuthority` from
    this and from nothing else. It is optional because a lock recorded at
    ``feedbax.spec.experiment_compile_lock.v1`` predates it; a v1 lock stating one
    is refused rather than read as a v2.
    """

    consumer: Literal["figure_runtime_input"] = "figure_runtime_input"
    input_role: str
    contract: FigureRoleBindingContract | None = None

    @model_validator(mode="after")
    def _validate(self) -> "FigureRuntimeInputBinding":
        _require_nonempty(self.input_role, "figure_runtime_input input_role")
        if self.contract is not None and self.contract.input_role != self.input_role:
            raise ValueError(
                "figure_runtime_input contract names input_role "
                f"{self.contract.input_role!r} while the binding addresses "
                f"{self.input_role!r}; one binding states one role"
            )
        return self


class ReportParentBinding(StrictModel):
    """The referenced product is one exact parent of a report.

    ``parent_kind`` states the referenced product kind and ``parent_id`` is the
    authored report-input role under which its authenticated receipt is bound.
    The receipt supplies the real parent kind and manifest id at fulfillment;
    the compiler must never substitute the product name for this consumer role.
    """

    consumer: Literal["report_parent"] = "report_parent"
    parent_kind: str
    parent_id: str

    @model_validator(mode="after")
    def _validate(self) -> "ReportParentBinding":
        _require_nonempty(self.parent_kind, "report_parent parent_kind")
        _require_nonempty(self.parent_id, "report_parent input role")
        return self


class CheckpointInitializationBinding(StrictModel):
    """The referenced product initializes or continues one training row."""

    consumer: Literal["checkpoint_initialization"] = "checkpoint_initialization"
    mode: CheckpointInitializationMode
    row_id: str

    @model_validator(mode="after")
    def _validate(self) -> "CheckpointInitializationBinding":
        _require_nonempty(self.row_id, "checkpoint_initialization row_id")
        return self


#: Who consumes a reference. Feedbax-owned and closed: a project names a role
#: string *inside* one of these bindings and never adds a consumer kind.
CompileLockConsumerBinding: TypeAlias = Annotated[
    EvaluationSubjectBinding
    | AnalysisInputBinding
    | AnalysisReceiptSetBinding
    | FigureRuntimeInputBinding
    | ReportParentBinding
    | CheckpointInitializationBinding,
    Field(discriminator="consumer"),
]


class ContentPinReference(StrictModel):
    """Bytes this compile read, pinned. A compile-time input, never a plan edge.

    A content pin says "these exact bytes were consulted". Nothing has to run
    because of it, so plan derivation skips it by kind rather than by inspecting
    its fields. It therefore states no consumer: nothing downstream is waiting
    on it.
    """

    kind: Literal["content_pin"] = "content_pin"
    ref: str
    content_hash: str
    pin_algorithm: str = CANONICAL_PIN_ALGORITHM

    @model_validator(mode="after")
    def _validate(self) -> "ContentPinReference":
        _require_nonempty(self.ref, "content_pin ref")
        _require_digest(self.content_hash, "content_pin content_hash")
        if self.pin_algorithm != CANONICAL_PIN_ALGORITHM:
            raise ValueError(f"content_pin pin_algorithm must be {CANONICAL_PIN_ALGORITHM!r}")
        return self


class GovernedParentReference(StrictModel):
    """Exact governed parent artifact bytes consumed during compilation."""

    kind: Literal["governed_parent"] = "governed_parent"
    parent: AuthoredIntentParent | ResolvedOutputParent
    role: str
    artifact_id: str
    artifact_sha256: str
    schema_id: str
    schema_version: str

    @model_validator(mode="after")
    def _validate(self) -> "GovernedParentReference":
        _require_nonempty(self.role, "governed_parent role")
        _require_nonempty(self.artifact_id, "governed_parent artifact_id")
        _require_digest(self.artifact_sha256, "governed_parent artifact_sha256")
        _require_nonempty(self.schema_id, "governed_parent schema_id")
        _require_nonempty(self.schema_version, "governed_parent schema_version")
        if not self.schema_version.startswith(f"{self.schema_id}."):
            raise ValueError(
                f"governed_parent schema_version {self.schema_version!r} does not extend "
                f"schema id {self.schema_id!r}"
            )
        return self


class PlannedProductReference(StrictModel):
    """A product another envelope compiles to, pinned before anything runs.

    Both facts that exist pre-run are recorded: the upstream envelope's own hash,
    and the content hash of the document it compiles into. The expected output
    schema is stated so a consumer can check the shape it is planning against
    without opening the upstream file.
    """

    kind: Literal["planned_product"] = "planned_product"
    envelope_ref: str
    envelope_hash: str
    product_name: str
    product_schema_id: str
    product_schema_version: str
    compiled_content_hash: str
    role_path: str
    consumer: CompileLockConsumerBinding

    @model_validator(mode="after")
    def _validate(self) -> "PlannedProductReference":
        _require_nonempty(self.envelope_ref, "planned_product envelope_ref")
        _require_digest(self.envelope_hash, "planned_product envelope_hash")
        _require_nonempty(self.product_name, "planned_product product_name")
        _require_nonempty(self.product_schema_id, "planned_product product_schema_id")
        if not self.product_schema_version.startswith(f"{self.product_schema_id}."):
            raise ValueError(
                f"planned_product product_schema_version {self.product_schema_version!r} "
                f"does not extend schema id {self.product_schema_id!r}"
            )
        _require_digest(self.compiled_content_hash, "planned_product compiled_content_hash")
        _require_role_path(self.role_path, "planned_product role_path")
        return self


class ReceiptLocatorReference(StrictModel):
    """A manifest named by kind and id, with no digest, because none exists yet.

    The real corpus needs this kind: a figure or analysis names the receipt it
    wants long before that receipt is written. Naming it without a digest is the
    honest record. Authenticating it is
    :class:`AuthenticatedReceiptReference`'s job, and only a run can supply the
    facts that promotion needs.
    """

    kind: Literal["receipt_locator"] = "receipt_locator"
    manifest_kind: str
    manifest_id: str
    role_path: str
    consumer: CompileLockConsumerBinding

    @model_validator(mode="after")
    def _validate(self) -> "ReceiptLocatorReference":
        _require_nonempty(self.manifest_kind, "receipt_locator manifest_kind")
        _require_nonempty(self.manifest_id, "receipt_locator manifest_id")
        _require_role_path(self.role_path, "receipt_locator role_path")
        return self


class AuthenticatedReceiptReference(StrictModel):
    """A manifest a previous run really wrote, quoted with its byte profile."""

    kind: Literal["authenticated_receipt"] = "authenticated_receipt"
    manifest_kind: str
    manifest_id: str
    manifest_sha256: str
    size_bytes: int = Field(ge=0)
    role_path: str
    consumer: CompileLockConsumerBinding
    execution_uri: str | None = None

    @model_validator(mode="after")
    def _validate(self) -> "AuthenticatedReceiptReference":
        _require_nonempty(self.manifest_kind, "authenticated_receipt manifest_kind")
        _require_nonempty(self.manifest_id, "authenticated_receipt manifest_id")
        _require_digest(self.manifest_sha256, "authenticated_receipt manifest_sha256")
        _require_role_path(self.role_path, "authenticated_receipt role_path")
        if self.execution_uri is not None:
            _require_nonempty(self.execution_uri, "authenticated_receipt execution_uri")
        return self


class NotApplicableReference(StrictModel):
    """A role deliberately left unfilled, and the basis for leaving it so.

    ``authored`` means a human stated it in the envelope; ``compiler_rule`` means
    a versioned Feedbax structural rule decided it. No project callback decides
    applicability, so there is no third basis.
    """

    kind: Literal["not_applicable"] = "not_applicable"
    role_path: str
    basis: Literal["authored", "compiler_rule"]
    reason: str
    rule_id: str | None = None

    @model_validator(mode="after")
    def _validate(self) -> "NotApplicableReference":
        _require_role_path(self.role_path, "not_applicable role_path")
        _require_nonempty(self.reason, "not_applicable reason")
        if self.basis == "compiler_rule":
            if self.rule_id is None:
                raise ValueError(
                    "not_applicable basis 'compiler_rule' must name the rule that decided it"
                )
            if not _VERSIONED_RULE_RE.fullmatch(self.rule_id):
                raise ValueError(
                    f"not_applicable rule_id {self.rule_id!r} must end with a version "
                    "segment such as '.v1'"
                )
        elif self.rule_id is not None:
            raise ValueError(
                "not_applicable basis 'authored' states no rule id; a human decided it"
            )
        return self


#: The only reference records a compile lock stores. Closed by construction.
CompileLockReference: TypeAlias = Annotated[
    ContentPinReference
    | GovernedParentReference
    | PlannedProductReference
    | ReceiptLocatorReference
    | AuthenticatedReceiptReference
    | NotApplicableReference,
    Field(discriminator="kind"),
]

_REFERENCE_ADAPTER: TypeAdapter[Any] = TypeAdapter(CompileLockReference)

_REFERENCE_MEMBERS = (
    ContentPinReference,
    GovernedParentReference,
    PlannedProductReference,
    ReceiptLocatorReference,
    AuthenticatedReceiptReference,
    NotApplicableReference,
)

#: Every reference kind, in the order the union declares them.
COMPILE_LOCK_REFERENCE_KINDS: tuple[str, ...] = (
    "content_pin",
    "governed_parent",
    "planned_product",
    "receipt_locator",
    "authenticated_receipt",
    "not_applicable",
)

#: Reference kinds that are plan edges. Compile-time inputs are not edges.
COMPILE_LOCK_PLAN_EDGE_KINDS: frozenset[str] = frozenset(
    {"planned_product", "receipt_locator", "authenticated_receipt", "not_applicable"}
)


def parse_compile_lock_reference(value: Any, *, field: str) -> Any:
    """Validate one reference into its union member, failing closed.

    Accepts either an already-typed member or the mapping form a tracked lock
    holds. An unknown kind, a missing consumer binding, or a stray key is a
    rejection here rather than a shape surprise in the plan lane.
    """
    if isinstance(value, _REFERENCE_MEMBERS):
        return value
    try:
        return _REFERENCE_ADAPTER.validate_python(value)
    except ValidationError as exc:
        raise ExperimentEnvelopeRejection(
            ExperimentEnvelopeRejectionCategory.INVALID_VALUE,
            f"not a compile-lock reference; kinds={list(COMPILE_LOCK_REFERENCE_KINDS)}: {exc}",
            field=field,
        ) from exc


def compile_lock_reference_record(value: Any, *, field: str) -> dict[str, Any]:
    """Return one validated reference as the mapping a lock stores."""
    return parse_compile_lock_reference(value, field=field).model_dump(
        mode="json", exclude_none=True
    )


# -- row provenance --------------------------------------------------------


class RowProvenanceReference(StrictModel):
    """One compiled row, and the parent row it was derived from.

    A layer that derives a row states the row it inherits *from*; the compiled
    document keeps the result but not the derivation, and ``base`` pins the parent
    document as a whole rather than the row inside it. This record is the missing
    half: the derived row's own id, the key it named in the parent, and the pinned
    bytes that key was resolved against.

    ``source_ref`` and ``source_content_hash`` are the same parent the lock's
    ``base`` block names, restated per row so a reader holding one row's
    provenance needs nothing else to check it.

    Attributes:
        row_id: The derived row's id in the compiled document.
        source_row_key: The parent row key the derivation named.
        source_ref: The resolved parent document the key was looked up in.
        source_content_hash: The canonical hash of that parent's bytes.
        pin_algorithm: The hash domain ``source_content_hash`` is stated in.
    """

    row_id: str
    source_row_key: str
    source_ref: str
    source_content_hash: str
    pin_algorithm: str = CANONICAL_PIN_ALGORITHM

    @model_validator(mode="after")
    def _validate(self) -> "RowProvenanceReference":
        _require_nonempty(self.row_id, "row_provenance row_id")
        _require_nonempty(self.source_row_key, "row_provenance source_row_key")
        _require_nonempty(self.source_ref, "row_provenance source_ref")
        _require_digest(self.source_content_hash, "row_provenance source_content_hash")
        if self.pin_algorithm != CANONICAL_PIN_ALGORITHM:
            raise ValueError(f"row_provenance pin_algorithm must be {CANONICAL_PIN_ALGORITHM!r}")
        return self


_ROW_PROVENANCE_ADAPTER: TypeAdapter[Any] = TypeAdapter(RowProvenanceReference)


def parse_row_provenance_reference(value: Any, *, field: str) -> RowProvenanceReference:
    """Validate one row provenance record, failing closed.

    Accepts either an already-typed record or the mapping form a tracked lock
    holds, for the same reason the reference union does: a lock edited into a
    shape nothing can read is caught at the boundary rather than downstream.
    """
    if isinstance(value, RowProvenanceReference):
        return value
    try:
        return _ROW_PROVENANCE_ADAPTER.validate_python(value)
    except ValidationError as exc:
        raise ExperimentEnvelopeRejection(
            ExperimentEnvelopeRejectionCategory.INVALID_VALUE,
            f"not a compile-lock row provenance record: {exc}",
            field=field,
        ) from exc


def row_provenance_record(value: Any, *, field: str) -> dict[str, Any]:
    """Return one validated row provenance record as the mapping a lock stores."""
    return parse_row_provenance_reference(value, field=field).model_dump(
        mode="json", exclude_none=True
    )


@dataclass(frozen=True)
class CompilerContract:
    """The logical contract a compiled document conforms to.

    There is one dialect and one compiler, so this contract is global rather
    than per-project: see
    :data:`~feedbax.contracts.experiment_envelope_dialect.EXPERIMENT_ENVELOPE_COMPILER_CONTRACT_VERSION`.
    A project cannot state a contract of its own, which is what makes two
    projects' compiled documents mean the same thing.
    """

    contract_id: str
    contract_version: str

    def __post_init__(self) -> None:
        if not self.contract_id.strip() or not self.contract_version.strip():
            raise CompileLockError("compiler contract must declare a nonempty id and version")
        if not self.contract_version.startswith(f"{self.contract_id}."):
            raise CompileLockError(
                f"compiler contract version {self.contract_version!r} does not extend "
                f"contract id {self.contract_id!r}"
            )

    def record(self) -> dict[str, str]:
        """Return this contract as its lock block."""
        return {
            "contract_id": self.contract_id,
            "contract_version": self.contract_version,
        }


@dataclass(frozen=True)
class CompilerImplementation:
    """The physical provenance of the code unit that produced a lock."""

    code_unit: str
    packages: tuple[str, ...] = ("feedbax",)

    def __post_init__(self) -> None:
        if not self.code_unit.strip():
            raise CompileLockError("compiler implementation must name a code unit")

    def record(self) -> dict[str, Any]:
        """Return this implementation's lock block, resolving installed versions.

        A package that is not installed records ``None`` rather than being
        omitted, so the absence is a stated fact instead of a silent gap.
        """
        return {
            "code_unit": self.code_unit,
            "package_versions": {
                name: _installed_version(name) for name in sorted(set(self.packages))
            },
        }


@dataclass(frozen=True)
class CompileLockInputs:
    """Everything the engine needs to assemble one compile lock.

    Attributes:
        envelope_ref: Repo-relative path of the authored envelope.
        envelope_document: The parsed authored envelope, hashed as read.
        envelope_schema: The envelope's declared schema string.
        name: The compiled output's name.
        family: The output family the compiled document belongs to.
        compiled_document: The document this compile produced.
        contract: The logical compiler contract, from the project declaration.
        implementation: The physical provenance of the running compiler.
        base: The resolved parent's pin record, or ``None`` for a root document.
        lineage_pins: The ordered content-pinned lineage behind the parent.
        resolved_deltas: What the compiler resolved, keyed by project concern.
        references: Cross-document references the compile resolved, each one a
            member of the closed :data:`CompileLockReference` union.
        row_provenance: One :class:`RowProvenanceReference` per compiled row that
            was derived from a row of the resolved parent.
        assertions: The inherited preconditions the compile checked.
        identity_contributions: Extra compile-time facts that widen execution
            identity, in the order the project states them. Two envelopes that
            differ only in a contribution are two different executions.
        issue: Optional tracking reference for the change that authored this.
    """

    envelope_ref: str
    envelope_document: Mapping[str, Any]
    envelope_schema: str
    name: str
    family: str
    compiled_document: Any
    contract: CompilerContract
    implementation: CompilerImplementation
    base: Mapping[str, Any] | None = None
    lineage_pins: Sequence[Mapping[str, Any]] = ()
    resolved_deltas: Mapping[str, Any] = dataclass_field(default_factory=dict)
    references: Sequence[Any] = ()
    row_provenance: Sequence[Any] = ()
    assertions: Sequence[Mapping[str, Any]] = ()
    identity_contributions: Mapping[str, Any] = dataclass_field(default_factory=dict)
    issue: str | None = None


def build_compile_lock(inputs: CompileLockInputs) -> dict[str, Any]:
    """Assemble the immutable compile lock for one compiled envelope.

    Execution identity is the canonical hash of an ordered preimage: the compiled
    document's content hash, plus each declared identity contribution hashed in
    its own right. ``inputs`` records what went in, so a consumer can see which
    facts an identity was built from without re-deriving them.
    """
    document_hash = canonical_sha256(inputs.compiled_document)
    contributions = dict(inputs.identity_contributions or {})
    preimage: dict[str, Any] = {"compiled_document": document_hash}
    identity_inputs = ["compiled_document.content_hash"]
    for key in sorted(contributions):
        preimage[key] = canonical_sha256(contributions[key])
        identity_inputs.append(f"identity_contributions.{key}")

    lock: dict[str, Any] = {
        "schema_id": EXPERIMENT_COMPILE_LOCK_SCHEMA_ID,
        "schema_version": EXPERIMENT_COMPILE_LOCK_SCHEMA_VERSION,
        "name": inputs.name,
        "envelope": {
            "schema": inputs.envelope_schema,
            "ref": inputs.envelope_ref,
            "envelope_hash": canonical_sha256(inputs.envelope_document),
            "pin_algorithm": CANONICAL_PIN_ALGORITHM,
        },
        "base": dict(inputs.base) if inputs.base is not None else None,
        "lineage": [dict(pin) for pin in inputs.lineage_pins],
        "row_provenance": [
            row_provenance_record(record, field=f"row_provenance[{index}]")
            for index, record in enumerate(inputs.row_provenance)
        ],
        "resolved_deltas": dict(inputs.resolved_deltas or {}),
        "references": [
            compile_lock_reference_record(reference, field=f"references[{index}]")
            for index, reference in enumerate(inputs.references)
        ],
        "assertions": [dict(assertion) for assertion in inputs.assertions],
        "compiled_document": {
            "family": inputs.family,
            "content_hash": document_hash,
            "pin_algorithm": CANONICAL_PIN_ALGORITHM,
        },
        "compiler_contract": inputs.contract.record(),
        "compiler_implementation": inputs.implementation.record(),
        "execution_identity": {
            "sha256": canonical_sha256(preimage),
            "inputs": identity_inputs,
            "pin_algorithm": CANONICAL_PIN_ALGORITHM,
        },
    }
    if inputs.issue is not None:
        lock["issue"] = inputs.issue
    if contributions:
        lock["identity_contributions"] = contributions
    check_plan_receipt_boundary(lock)
    return lock


#: The blocks every v1 lock states. ``base`` is stated as ``null`` for a root
#: document rather than omitted, so absence is a malformed lock, not a root one.
REQUIRED_COMPILE_LOCK_KEYS: tuple[str, ...] = (
    "schema_id",
    "schema_version",
    "name",
    "envelope",
    "base",
    "lineage",
    "row_provenance",
    "resolved_deltas",
    "references",
    "assertions",
    "compiled_document",
    "compiler_contract",
    "compiler_implementation",
    "execution_identity",
)

#: The blocks a v1 lock states only when it has something to say: the tracking
#: reference the envelope authored, and the contributions that widened execution
#: identity. Nothing else may appear at the top level.
OPTIONAL_COMPILE_LOCK_KEYS: tuple[str, ...] = ("issue", "identity_contributions")

_IDENTITY_DOCUMENT_INPUT = "compiled_document.content_hash"


def _lock_reject(
    field: str,
    message: str,
    category: ExperimentEnvelopeRejectionCategory = (
        ExperimentEnvelopeRejectionCategory.INVALID_VALUE
    ),
) -> Any:
    raise ExperimentEnvelopeRejection(category, message, field=field)


def _lock_mapping(value: Any, field: str, what: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _lock_reject(field, f"{what} is an object; found {type(value).__name__}")
    return value


def _lock_sequence(value: Any, field: str, what: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        _lock_reject(field, f"{what} is a list; found {type(value).__name__}")
    return value


def _lock_text(value: Any, field: str, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _lock_reject(field, f"{what} is a nonempty string; found {value!r}")
    return value


def _lock_digest(value: Any, field: str, what: str) -> str:
    _lock_text(value, field, what)
    if not _DIGEST_RE.fullmatch(value):
        _lock_reject(field, f"{what} is a lowercase sha256 digest; found {value!r}")
    return value


def _lock_keys(value: Mapping[str, Any], expected: Sequence[str], field: str, what: str) -> None:
    missing = sorted(set(expected) - set(value))
    if missing:
        _lock_reject(
            field,
            f"{what} states {missing!r}",
            ExperimentEnvelopeRejectionCategory.MISSING_FIELD,
        )
    unknown = sorted(set(value) - set(expected))
    if unknown:
        _lock_reject(
            field,
            f"{what} states nothing outside {sorted(expected)!r}; found {unknown!r}",
            ExperimentEnvelopeRejectionCategory.UNKNOWN_FIELD,
        )


def _lock_pin(value: Any, field: str, what: str, *, extra: Sequence[str] = ()) -> None:
    """Validate one content-pin record: a ref, its digest, and the hash domain."""
    record = _lock_mapping(value, field, what)
    _lock_keys(record, ("ref", "content_hash", "pin_algorithm", *extra), field, what)
    for name in ("ref", *extra):
        _lock_text(record[name], f"{field}.{name}", f"{what} {name}")
    _lock_digest(record["content_hash"], f"{field}.content_hash", f"{what} content_hash")
    if record["pin_algorithm"] != CANONICAL_PIN_ALGORITHM:
        _lock_reject(
            f"{field}.pin_algorithm",
            f"{what} pin_algorithm is {CANONICAL_PIN_ALGORITHM!r}; found "
            f"{record['pin_algorithm']!r}",
        )


def _validate_lock_envelope(lock: Mapping[str, Any], field: str) -> None:
    what = "a compile lock's envelope block"
    record = _lock_mapping(lock["envelope"], f"{field}#envelope", what)
    _lock_keys(
        record, ("schema", "ref", "envelope_hash", "pin_algorithm"), f"{field}#envelope", what
    )
    _lock_text(record["schema"], f"{field}#envelope.schema", f"{what} schema")
    _lock_text(record["ref"], f"{field}#envelope.ref", f"{what} ref")
    _lock_digest(
        record["envelope_hash"], f"{field}#envelope.envelope_hash", f"{what} envelope_hash"
    )
    if record["pin_algorithm"] != CANONICAL_PIN_ALGORITHM:
        _lock_reject(
            f"{field}#envelope.pin_algorithm",
            f"{what} pin_algorithm is {CANONICAL_PIN_ALGORITHM!r}; found "
            f"{record['pin_algorithm']!r}",
        )


def _validate_lock_lineage(lock: Mapping[str, Any], field: str) -> None:
    if lock["base"] is not None:
        _lock_pin(
            lock["base"],
            f"{field}#base",
            "a compile lock's resolved parent pin",
            extra=("kind",),
        )
    pins = _lock_sequence(lock["lineage"], f"{field}#lineage", "a compile lock's lineage")
    for index, pin in enumerate(pins):
        _lock_pin(pin, f"{field}#lineage[{index}]", "a compile lock's lineage pin")


def _validate_lock_resolved_deltas(lock: Mapping[str, Any], field: str) -> None:
    """Validate the layers the compile resolved, keyed by their own layer ids."""
    what = "a compile lock's resolved deltas"
    deltas = _lock_mapping(lock["resolved_deltas"], f"{field}#resolved_deltas", what)
    for key, value in deltas.items():
        locator = f"{field}#resolved_deltas.{key}"
        _lock_text(key, locator, f"{what} key")
        record = _lock_mapping(value, locator, "a resolved delta")
        optional_schema_fields = tuple(
            name for name in ("schema_id", "schema_version") if name in record
        )
        _lock_keys(
            record,
            (
                "layer_id",
                "patches",
                "acknowledges_ancestor_paths",
                *optional_schema_fields,
            ),
            locator,
            "a resolved delta",
        )
        if len(optional_schema_fields) == 1:
            missing = ({"schema_id", "schema_version"} - set(optional_schema_fields)).pop()
            _lock_reject(
                locator,
                f"a resolved delta states {missing!r} with its schema boundary",
                ExperimentEnvelopeRejectionCategory.MISSING_FIELD,
            )
        for name in optional_schema_fields:
            _lock_text(record[name], f"{locator}.{name}", f"a resolved delta {name}")
        if record["layer_id"] != key:
            _lock_reject(
                f"{locator}.layer_id",
                f"a resolved delta is keyed by its own layer id; {key!r} holds "
                f"{record['layer_id']!r}",
            )
        patches = _lock_sequence(record["patches"], f"{locator}.patches", "a delta's patches")
        for index, patch in enumerate(patches):
            entry = _lock_mapping(patch, f"{locator}.patches[{index}]", "a delta patch")
            for name in ("path", "op"):
                if name not in entry:
                    _lock_reject(
                        f"{locator}.patches[{index}]",
                        f"a delta patch states {name!r}",
                        ExperimentEnvelopeRejectionCategory.MISSING_FIELD,
                    )
                _lock_text(
                    entry[name], f"{locator}.patches[{index}].{name}", f"a delta patch {name}"
                )
        acknowledged = _lock_sequence(
            record["acknowledges_ancestor_paths"],
            f"{locator}.acknowledges_ancestor_paths",
            "a delta's acknowledged ancestor paths",
        )
        for index, path in enumerate(acknowledged):
            _lock_text(
                path,
                f"{locator}.acknowledges_ancestor_paths[{index}]",
                "an acknowledged ancestor path",
            )


def _validate_lock_assertions(lock: Mapping[str, Any], field: str) -> None:
    what = "a compile lock's assertions"
    records = _lock_sequence(lock["assertions"], f"{field}#assertions", what)
    for index, value in enumerate(records):
        locator = f"{field}#assertions[{index}]"
        record = _lock_mapping(value, locator, "a checked assertion")
        _lock_keys(
            record, ("path", "expected", "actual", "owner_ref"), locator, "a checked assertion"
        )
        _lock_text(record["path"], f"{locator}.path", "a checked assertion path")
        _lock_text(record["owner_ref"], f"{locator}.owner_ref", "a checked assertion owner_ref")


def _validate_lock_provenance(lock: Mapping[str, Any], field: str) -> None:
    """Validate the compiled document's identity and the compiler that produced it."""
    what = "a compile lock's compiled document block"
    compiled = _lock_mapping(lock["compiled_document"], f"{field}#compiled_document", what)
    _lock_keys(
        compiled,
        ("family", "content_hash", "pin_algorithm"),
        f"{field}#compiled_document",
        what,
    )
    _lock_text(compiled["family"], f"{field}#compiled_document.family", f"{what} family")
    _lock_digest(
        compiled["content_hash"],
        f"{field}#compiled_document.content_hash",
        f"{what} content_hash",
    )
    if compiled["pin_algorithm"] != CANONICAL_PIN_ALGORITHM:
        _lock_reject(
            f"{field}#compiled_document.pin_algorithm",
            f"{what} pin_algorithm is {CANONICAL_PIN_ALGORITHM!r}; found "
            f"{compiled['pin_algorithm']!r}",
        )

    contract = _lock_mapping(
        lock["compiler_contract"], f"{field}#compiler_contract", "a compiler contract"
    )
    _lock_keys(
        contract,
        ("contract_id", "contract_version"),
        f"{field}#compiler_contract",
        "a compiler contract",
    )
    contract_id = _lock_text(
        contract["contract_id"], f"{field}#compiler_contract.contract_id", "a contract id"
    )
    contract_version = _lock_text(
        contract["contract_version"],
        f"{field}#compiler_contract.contract_version",
        "a contract version",
    )
    if not contract_version.startswith(f"{contract_id}."):
        _lock_reject(
            f"{field}#compiler_contract.contract_version",
            f"contract version {contract_version!r} does not extend contract id {contract_id!r}",
        )

    implementation = _lock_mapping(
        lock["compiler_implementation"],
        f"{field}#compiler_implementation",
        "a compiler implementation",
    )
    _lock_keys(
        implementation,
        ("code_unit", "package_versions"),
        f"{field}#compiler_implementation",
        "a compiler implementation",
    )
    _lock_text(
        implementation["code_unit"],
        f"{field}#compiler_implementation.code_unit",
        "a compiler code unit",
    )
    packages = _lock_mapping(
        implementation["package_versions"],
        f"{field}#compiler_implementation.package_versions",
        "compiler package versions",
    )
    for name, installed in packages.items():
        locator = f"{field}#compiler_implementation.package_versions.{name}"
        _lock_text(name, locator, "a compiler package name")
        # ``None`` is the stated fact that a named package is not installed.
        if installed is not None:
            _lock_text(installed, locator, "a compiler package version")


def _validate_lock_execution_identity(
    lock: Mapping[str, Any],
    field: str,
    *,
    pin_required: bool,
) -> None:
    """Re-derive execution identity from the lock's own facts.

    Everything the writer hashed is in the document the reader is holding, so an
    identity that does not reproduce is not a fact this lock can support. The
    input list is derived the same way, which is what makes a contribution that
    was added or dropped after emission visible rather than silent.
    """
    what = "a compile lock's execution identity"
    identity = _lock_mapping(lock["execution_identity"], f"{field}#execution_identity", what)
    keys = ("sha256", "inputs", "pin_algorithm") if pin_required else ("sha256", "inputs")
    _lock_keys(identity, keys, f"{field}#execution_identity", what)
    _lock_digest(identity["sha256"], f"{field}#execution_identity.sha256", f"{what} sha256")
    if pin_required and identity["pin_algorithm"] != CANONICAL_PIN_ALGORITHM:
        _lock_reject(
            f"{field}#execution_identity.pin_algorithm",
            f"{what} pin_algorithm is {CANONICAL_PIN_ALGORITHM!r}; found "
            f"{identity['pin_algorithm']!r}",
        )
    inputs = _lock_sequence(
        identity["inputs"], f"{field}#execution_identity.inputs", f"{what} inputs"
    )
    contributions = lock.get("identity_contributions", {})
    expected_inputs = [
        _IDENTITY_DOCUMENT_INPUT,
        *(f"identity_contributions.{key}" for key in sorted(contributions)),
    ]
    if list(inputs) != expected_inputs:
        _lock_reject(
            f"{field}#execution_identity.inputs",
            f"{what} names the facts it was built from; expected {expected_inputs!r}, "
            f"found {list(inputs)!r}",
        )
    preimage: dict[str, Any] = {"compiled_document": lock["compiled_document"]["content_hash"]}
    for key in sorted(contributions):
        preimage[key] = canonical_sha256(contributions[key])
    expected_sha256 = canonical_sha256(preimage)
    if identity["sha256"] != expected_sha256:
        _lock_reject(
            f"{field}#execution_identity.sha256",
            f"{what} does not re-derive from this lock's own facts; expected "
            f"{expected_sha256!r}, found {identity['sha256']!r}",
        )


def _validate_compile_lock_body(
    lock: Mapping[str, Any],
    field: str,
    *,
    pin_required: bool,
) -> None:
    """Validate the whole document, not only the blocks a reader happens to use.

    A lock is the compile-side half of the custody boundary: every consumer that
    trusts it trusts all of it. Validating identity and references while leaving
    the parent pin, the resolved deltas, the compiler provenance, and the
    execution identity unchecked would mean a lock edited anywhere else loads
    cleanly and is discovered downstream, or not at all.
    """
    missing = sorted(set(REQUIRED_COMPILE_LOCK_KEYS) - set(lock))
    if missing:
        _lock_reject(
            field,
            f"a compile lock states {missing!r}",
            ExperimentEnvelopeRejectionCategory.MISSING_FIELD,
        )
    known = {*REQUIRED_COMPILE_LOCK_KEYS, *OPTIONAL_COMPILE_LOCK_KEYS}
    unknown = sorted(set(lock) - known)
    if unknown:
        _lock_reject(
            field,
            f"a compile lock states nothing outside {sorted(known)!r}; found {unknown!r}",
            ExperimentEnvelopeRejectionCategory.UNKNOWN_FIELD,
        )
    _lock_text(lock["name"], f"{field}#name", "a compile lock's name")
    if "issue" in lock:
        _lock_text(lock["issue"], f"{field}#issue", "a compile lock's issue")
    if "identity_contributions" in lock:
        contributions = _lock_mapping(
            lock["identity_contributions"],
            f"{field}#identity_contributions",
            "a compile lock's identity contributions",
        )
        if not contributions:
            _lock_reject(
                f"{field}#identity_contributions",
                "a compile lock states identity contributions only when it has some; an "
                "empty block is omitted",
            )
        for key in contributions:
            _lock_text(
                key,
                f"{field}#identity_contributions",
                "an identity contribution key",
            )
    _validate_lock_envelope(lock, field)
    _validate_lock_lineage(lock, field)
    _validate_lock_resolved_deltas(lock, field)
    _validate_lock_assertions(lock, field)
    _validate_lock_provenance(lock, field)
    _validate_lock_execution_identity(lock, field, pin_required=pin_required)


def _refuse_v1_figure_input_contract(references: Sequence[Any], *, field: str) -> None:
    """Refuse a v1 lock that states a v2 figure runtime input contract.

    A version names one grammar. The typed artifact contract on a figure runtime
    input binding is v2 grammar, and a v1 document carrying one is a v2 document
    wearing the wrong version — accepting it as a wider v1 would make "v1" the
    name of two grammars, and a reader that then ignored the contract would
    render the figure with no authority over the artifacts it names.
    """
    for index, reference in enumerate(references):
        if not isinstance(reference, Mapping):
            continue
        consumer = reference.get("consumer")
        if not isinstance(consumer, Mapping):
            continue
        if consumer.get("consumer") != "figure_runtime_input":
            continue
        if consumer.get("contract") is None:
            continue
        raise ExperimentEnvelopeRejection(
            ExperimentEnvelopeRejectionCategory.UNSUPPORTED_SCHEMA_VERSION,
            "a figure runtime input binding's typed artifact contract is "
            f"{EXPERIMENT_COMPILE_LOCK_SCHEMA_VERSION_V2!r} grammar, and this lock declares "
            f"{EXPERIMENT_COMPILE_LOCK_SCHEMA_VERSION_V1!r}. A version names exactly one "
            "grammar, so it is refused here rather than accepted as a wider v1",
            field=f"{field}#references[{index}]#consumer.contract",
        )


def _refuse_pre_v3_analysis_receipt_set(references: Sequence[Any], *, field: str) -> None:
    """Refuse a v1/v2 lock edited to carry the v3 receipt-set discriminator."""
    for index, reference in enumerate(references):
        if not isinstance(reference, Mapping):
            continue
        consumer = reference.get("consumer")
        if isinstance(consumer, Mapping) and consumer.get("consumer") == "analysis_receipt_set":
            raise ExperimentEnvelopeRejection(
                ExperimentEnvelopeRejectionCategory.UNSUPPORTED_SCHEMA_VERSION,
                "an analysis receipt-set binding is "
                f"{EXPERIMENT_COMPILE_LOCK_SCHEMA_VERSION_V3!r} grammar, but this lock "
                "declares an earlier version; a version names exactly one grammar",
                field=f"{field}#references[{index}]#consumer.consumer",
            )


def _refuse_pre_v4_governed_parent(references: Sequence[Any], *, field: str) -> None:
    """Refuse an older lock edited to carry the v4 governed-parent input."""
    for index, reference in enumerate(references):
        if isinstance(reference, Mapping) and reference.get("kind") == "governed_parent":
            raise ExperimentEnvelopeRejection(
                ExperimentEnvelopeRejectionCategory.UNSUPPORTED_SCHEMA_VERSION,
                "a governed compile-time parent reference is "
                f"{EXPERIMENT_COMPILE_LOCK_SCHEMA_VERSION_V4!r} grammar, but this lock "
                "declares an earlier version; a version names exactly one grammar",
                field=f"{field}#references[{index}]#kind",
            )


def _validate_compile_lock_payload(
    lock: Mapping[str, Any],
    *,
    field: str,
    version: str,
) -> None:
    check_plan_receipt_boundary(lock)
    references = lock.get("references", [])
    if not isinstance(references, Sequence) or isinstance(references, (str, bytes)):
        raise ExperimentEnvelopeRejection(
            ExperimentEnvelopeRejectionCategory.INVALID_VALUE,
            "a compile lock's references are a list of typed reference records",
            field=f"{field}#references",
        )
    for index, reference in enumerate(references):
        parse_compile_lock_reference(reference, field=f"{field}#references[{index}]")
    if version == EXPERIMENT_COMPILE_LOCK_SCHEMA_VERSION_V1:
        _refuse_v1_figure_input_contract(references, field=field)
    if version in (
        EXPERIMENT_COMPILE_LOCK_SCHEMA_VERSION_V1,
        EXPERIMENT_COMPILE_LOCK_SCHEMA_VERSION_V2,
    ):
        _refuse_pre_v3_analysis_receipt_set(references, field=field)
    if version in (
        EXPERIMENT_COMPILE_LOCK_SCHEMA_VERSION_V1,
        EXPERIMENT_COMPILE_LOCK_SCHEMA_VERSION_V2,
        EXPERIMENT_COMPILE_LOCK_SCHEMA_VERSION_V3,
    ):
        _refuse_pre_v4_governed_parent(references, field=field)
    provenance = lock.get("row_provenance", [])
    if not isinstance(provenance, Sequence) or isinstance(provenance, (str, bytes)):
        raise ExperimentEnvelopeRejection(
            ExperimentEnvelopeRejectionCategory.INVALID_VALUE,
            "a compile lock's row provenance is a list of typed row records",
            field=f"{field}#row_provenance",
        )
    for index, record in enumerate(provenance):
        parse_row_provenance_reference(record, field=f"{field}#row_provenance[{index}]")
    _validate_compile_lock_body(
        lock,
        field,
        pin_required=version == EXPERIMENT_COMPILE_LOCK_SCHEMA_VERSION_V4,
    )


def migrate_compile_lock_v3_to_v4(
    document: Mapping[str, Any],
    *,
    field: str = "ExperimentCompileLock migration input",
) -> dict[str, Any]:
    """Pin a validated v3 lock attributed to the built-in Feedbax compiler.

    The migration never searches for digest-shaped objects. Its input must be one
    exact v3 compile lock whose own producer record names the built-in Feedbax
    entrypoint and a concrete Feedbax package version.
    """
    lock = dict(document)
    if lock.get("schema_id") != EXPERIMENT_COMPILE_LOCK_SCHEMA_ID:
        _lock_reject(
            f"{field}#schema_id",
            "only a Feedbax experiment compile lock can receive an execution-identity pin",
            ExperimentEnvelopeRejectionCategory.UNSUPPORTED_SCHEMA_VERSION,
        )
    if lock.get("schema_version") != EXPERIMENT_COMPILE_LOCK_SCHEMA_VERSION_V3:
        _lock_reject(
            f"{field}#schema_version",
            "the execution-identity pin migration accepts exactly "
            f"{EXPERIMENT_COMPILE_LOCK_SCHEMA_VERSION_V3!r}; found "
            f"{lock.get('schema_version')!r}",
            ExperimentEnvelopeRejectionCategory.UNSUPPORTED_SCHEMA_VERSION,
        )
    _validate_compile_lock_payload(
        lock,
        field=field,
        version=EXPERIMENT_COMPILE_LOCK_SCHEMA_VERSION_V3,
    )
    contract = lock["compiler_contract"]
    implementation = lock["compiler_implementation"]
    package_versions = implementation["package_versions"]
    feedbax_version = package_versions.get("feedbax")
    if (
        contract["contract_id"] != EXPERIMENT_ENVELOPE_COMPILER_CONTRACT_ID
        or implementation["code_unit"] != "feedbax.envelope.entrypoint"
        or set(package_versions) != {"feedbax"}
        or not isinstance(feedbax_version, str)
        or not feedbax_version.strip()
    ):
        _lock_reject(
            f"{field}#compiler_implementation",
            "the v3 execution-identity pin can be asserted only for a lock with "
            "attributable built-in Feedbax compiler provenance; downstream-authored "
            "or unattributed documents must remain unpinned",
        )
    migrated = dict(lock)
    migrated["schema_version"] = EXPERIMENT_COMPILE_LOCK_SCHEMA_VERSION_V4
    migrated["execution_identity"] = {
        **lock["execution_identity"],
        "pin_algorithm": CANONICAL_PIN_ALGORITHM,
    }
    return migrated


def load_compile_lock(document: Any, *, field: str) -> dict[str, Any]:
    """Read one compile lock, failing closed on an unsupported version.

    A version absent from both the supported set and the migration table has no
    path forward and is refused with both named. The plan/receipt boundary is
    re-checked on read, so a lock that was edited into carrying a receipt fact is
    caught by the reader as well as by the writer. Every reference is re-validated
    against the closed union for the same reason: a lock whose references were
    edited into a shape the plan lane cannot read is caught here, not there.

    The rest of the v1 document is validated to the same standard — the envelope
    pin, the parent and its lineage, the resolved deltas, the checked assertions,
    the compiled document's identity, both halves of compiler provenance, and an
    execution identity re-derived from the lock's own facts. A reader that
    validated only what it happened to touch would let a lock edited anywhere
    else load cleanly.
    """
    if not isinstance(document, Mapping):
        raise ExperimentEnvelopeRejection(
            ExperimentEnvelopeRejectionCategory.INVALID_VALUE,
            "a compile lock is a JSON object",
            field=field,
        )
    found_id = document.get("schema_id")
    if found_id != EXPERIMENT_COMPILE_LOCK_SCHEMA_ID:
        raise ExperimentEnvelopeRejection(
            ExperimentEnvelopeRejectionCategory.UNSUPPORTED_SCHEMA_VERSION,
            f"expected schema_id {EXPERIMENT_COMPILE_LOCK_SCHEMA_ID!r}, found {found_id!r}",
            field=f"{field}#schema_id",
        )
    version = document.get("schema_version")
    if (
        version not in EXPERIMENT_COMPILE_LOCK_SUPPORTED_SCHEMA_VERSIONS
        and version not in EXPERIMENT_COMPILE_LOCK_MIGRATION_TABLE
    ):
        raise ExperimentEnvelopeRejection(
            ExperimentEnvelopeRejectionCategory.UNSUPPORTED_SCHEMA_VERSION,
            f"unsupported schema_version {version!r}; "
            f"supported={list(EXPERIMENT_COMPILE_LOCK_SUPPORTED_SCHEMA_VERSIONS)}; "
            f"migration table={EXPERIMENT_COMPILE_LOCK_MIGRATION_TABLE!r}; "
            "migration_intentionally_absent=yes",
            field=f"{field}#schema_version",
        )
    lock = dict(document)
    if version == EXPERIMENT_COMPILE_LOCK_SCHEMA_VERSION_V3:
        lock = migrate_compile_lock_v3_to_v4(lock, field=field)
        version = EXPERIMENT_COMPILE_LOCK_SCHEMA_VERSION_V4
    _validate_compile_lock_payload(lock, field=field, version=version)
    return lock


def compile_lock_plan_edges(lock: Mapping[str, Any], *, field: str) -> tuple[Any, ...]:
    """Return the lock's references that are plan edges, typed and in order.

    Compile-time inputs are filtered out by kind, so the
    plan lane never has to decide what an edge is by looking at fields.
    """
    return tuple(
        parsed
        for index, reference in enumerate(lock.get("references", []))
        if (
            parsed := parse_compile_lock_reference(reference, field=f"{field}#references[{index}]")
        ).kind
        in COMPILE_LOCK_PLAN_EDGE_KINDS
    )


__all__ = [
    "COMPILE_LOCK_PLAN_EDGE_KINDS",
    "COMPILE_LOCK_REFERENCE_KINDS",
    "OPTIONAL_COMPILE_LOCK_KEYS",
    "REQUIRED_COMPILE_LOCK_KEYS",
    "EXPERIMENT_COMPILE_LOCK_MIGRATION_TABLE",
    "EXPERIMENT_COMPILE_LOCK_SCHEMA_ID",
    "EXPERIMENT_COMPILE_LOCK_SCHEMA_VERSION",
    "EXPERIMENT_COMPILE_LOCK_SCHEMA_VERSION_V1",
    "EXPERIMENT_COMPILE_LOCK_SCHEMA_VERSION_V2",
    "EXPERIMENT_COMPILE_LOCK_SCHEMA_VERSION_V3",
    "EXPERIMENT_COMPILE_LOCK_SCHEMA_VERSION_V4",
    "GovernedParentReference",
    "EXPERIMENT_COMPILE_LOCK_SUPPORTED_SCHEMA_VERSIONS",
    "RUN_RECEIPT_ONLY_FACTS",
    "AnalysisInputBinding",
    "AnalysisReceiptSetBinding",
    "AuthenticatedReceiptReference",
    "CheckpointInitializationBinding",
    "CompileLockConsumerBinding",
    "CompileLockError",
    "CompileLockInputs",
    "CompileLockReference",
    "CompilerContract",
    "CompilerImplementation",
    "ContentPinReference",
    "EvaluationSubjectBinding",
    "FigureRuntimeInputBinding",
    "NotApplicableReference",
    "PlanReceiptBoundaryError",
    "PlannedProductReference",
    "ReceiptLocatorReference",
    "ReportParentBinding",
    "RowProvenanceReference",
    "build_compile_lock",
    "check_plan_receipt_boundary",
    "compile_lock_plan_edges",
    "compile_lock_reference_record",
    "load_compile_lock",
    "migrate_compile_lock_v3_to_v4",
    "parse_compile_lock_reference",
    "parse_row_provenance_reference",
    "row_provenance_record",
]
