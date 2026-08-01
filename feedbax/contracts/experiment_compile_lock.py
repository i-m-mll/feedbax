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
rather than about the record's own kind. The five kinds answer five genuinely
different questions:

* :class:`ContentPinReference` — bytes this compile *read*. It is a compile-time
  input and is never a plan edge; nothing runs because of it.
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
from feedbax.contracts.manifest import StrictModel

EXPERIMENT_COMPILE_LOCK_SCHEMA_ID = "feedbax.spec.experiment_compile_lock"
EXPERIMENT_COMPILE_LOCK_SCHEMA_VERSION_V1 = f"{EXPERIMENT_COMPILE_LOCK_SCHEMA_ID}.v1"
EXPERIMENT_COMPILE_LOCK_SCHEMA_VERSION = EXPERIMENT_COMPILE_LOCK_SCHEMA_VERSION_V1

#: The only lock versions read. Enumerated, never inferred.
EXPERIMENT_COMPILE_LOCK_SUPPORTED_SCHEMA_VERSIONS: tuple[str, ...] = (
    EXPERIMENT_COMPILE_LOCK_SCHEMA_VERSION_V1,
)

#: Versions this loader accepts, mapped to the version they migrate to. Empty at
#: v1: no Feedbax-owned lock predates it. A project-owned lock family that
#: migrates into this one registers its edge here and in ``default_spec_registry``
#: in one change, which is the slot the downstream conversion lane fills.
EXPERIMENT_COMPILE_LOCK_MIGRATION_TABLE: dict[str, str] = {}

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


class FigureRuntimeInputBinding(StrictModel):
    """The referenced product satisfies one figure runtime input authority.

    ``input_role`` is the role a
    :class:`~feedbax.contracts.figures.FigureInputRoleAuthority` addresses its
    single exact parent by; this binding is the compile-time statement of the
    same role.
    """

    consumer: Literal["figure_runtime_input"] = "figure_runtime_input"
    input_role: str

    @model_validator(mode="after")
    def _validate(self) -> "FigureRuntimeInputBinding":
        _require_nonempty(self.input_role, "figure_runtime_input input_role")
        return self


class ReportParentBinding(StrictModel):
    """The referenced product is one exact parent of a report.

    ``parent_kind`` and ``parent_id`` are the two fields a
    :class:`~feedbax.contracts.manifest.ParentRef` identifies a parent by, stated
    at compile time before the parent exists.
    """

    consumer: Literal["report_parent"] = "report_parent"
    parent_kind: str
    parent_id: str

    @model_validator(mode="after")
    def _validate(self) -> "ReportParentBinding":
        _require_nonempty(self.parent_kind, "report_parent parent_kind")
        _require_nonempty(self.parent_id, "report_parent parent_id")
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
            raise ValueError(
                f"content_pin pin_algorithm must be {CANONICAL_PIN_ALGORITHM!r}"
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
    | PlannedProductReference
    | ReceiptLocatorReference
    | AuthenticatedReceiptReference
    | NotApplicableReference,
    Field(discriminator="kind"),
]

_REFERENCE_ADAPTER: TypeAdapter[Any] = TypeAdapter(CompileLockReference)

_REFERENCE_MEMBERS = (
    ContentPinReference,
    PlannedProductReference,
    ReceiptLocatorReference,
    AuthenticatedReceiptReference,
    NotApplicableReference,
)

#: Every reference kind, in the order the union declares them.
COMPILE_LOCK_REFERENCE_KINDS: tuple[str, ...] = (
    "content_pin",
    "planned_product",
    "receipt_locator",
    "authenticated_receipt",
    "not_applicable",
)

#: Reference kinds that are plan edges. A content pin is an input, not an edge.
COMPILE_LOCK_PLAN_EDGE_KINDS: frozenset[str] = frozenset(
    kind for kind in COMPILE_LOCK_REFERENCE_KINDS if kind != "content_pin"
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
            raise CompileLockError(
                "compiler contract must declare a nonempty id and version"
            )
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
        },
    }
    if inputs.issue is not None:
        lock["issue"] = inputs.issue
    if contributions:
        lock["identity_contributions"] = contributions
    check_plan_receipt_boundary(lock)
    return lock


def load_compile_lock(document: Any, *, field: str) -> dict[str, Any]:
    """Read one compile lock, failing closed on an unsupported version.

    A version absent from both the supported set and the migration table has no
    path forward and is refused with both named. The plan/receipt boundary is
    re-checked on read, so a lock that was edited into carrying a receipt fact is
    caught by the reader as well as by the writer. Every reference is re-validated
    against the closed union for the same reason: a lock whose references were
    edited into a shape the plan lane cannot read is caught here, not there.
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
    if version not in EXPERIMENT_COMPILE_LOCK_SUPPORTED_SCHEMA_VERSIONS:
        raise ExperimentEnvelopeRejection(
            ExperimentEnvelopeRejectionCategory.UNSUPPORTED_SCHEMA_VERSION,
            f"unsupported schema_version {version!r}; "
            f"supported={list(EXPERIMENT_COMPILE_LOCK_SUPPORTED_SCHEMA_VERSIONS)}; "
            f"migration table={EXPERIMENT_COMPILE_LOCK_MIGRATION_TABLE!r}; "
            "migration_intentionally_absent=yes",
            field=f"{field}#schema_version",
        )
    lock = dict(document)
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
    return lock


def compile_lock_plan_edges(lock: Mapping[str, Any], *, field: str) -> tuple[Any, ...]:
    """Return the lock's references that are plan edges, typed and in order.

    Content pins are compile-time inputs and are filtered out by kind, so the
    plan lane never has to decide what an edge is by looking at fields.
    """
    return tuple(
        parsed
        for index, reference in enumerate(lock.get("references", []))
        if (parsed := parse_compile_lock_reference(reference, field=f"{field}#references[{index}]"))
        .kind
        in COMPILE_LOCK_PLAN_EDGE_KINDS
    )


__all__ = [
    "COMPILE_LOCK_PLAN_EDGE_KINDS",
    "COMPILE_LOCK_REFERENCE_KINDS",
    "EXPERIMENT_COMPILE_LOCK_MIGRATION_TABLE",
    "EXPERIMENT_COMPILE_LOCK_SCHEMA_ID",
    "EXPERIMENT_COMPILE_LOCK_SCHEMA_VERSION",
    "EXPERIMENT_COMPILE_LOCK_SCHEMA_VERSION_V1",
    "EXPERIMENT_COMPILE_LOCK_SUPPORTED_SCHEMA_VERSIONS",
    "RUN_RECEIPT_ONLY_FACTS",
    "AnalysisInputBinding",
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
    "build_compile_lock",
    "check_plan_receipt_boundary",
    "compile_lock_plan_edges",
    "compile_lock_reference_record",
    "load_compile_lock",
    "parse_compile_lock_reference",
]
