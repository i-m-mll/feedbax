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
  conforms to — the ``compiler_contract_id`` and ``compiler_contract_version``
  the project's extension declaration states. It changes when the meaning of a
  compiled document changes, and it is what a consumer checks before trusting
  the output's shape.
* :class:`CompilerImplementation` is the **physical** provenance of the code that
  ran — the package that hosts the compiler, its version, and the versions of the
  packages it was built against. It changes whenever anything is released, and it
  is what an investigator uses to attribute a byte-level difference.

Collapsing the two into one string forces a logical version bump for every
release, or hides a release behind an unchanged logical version. Keeping them
apart lets a recompilation that changes derived bytes be attributable without
implying that the contract moved.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version as package_version
from typing import Any

from feedbax.contracts.authored_canonical import (
    CANONICAL_PIN_ALGORITHM,
    canonical_sha256,
)
from feedbax.contracts.experiment_envelope import (
    ExperimentEnvelopeRejection,
    ExperimentEnvelopeRejectionCategory,
)
from feedbax.contracts.project_extension import ProjectExtensionDeclaration

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


@dataclass(frozen=True)
class CompilerContract:
    """The logical contract a compiled document conforms to.

    Both fields come from the project's extension declaration; the engine never
    invents, defaults, or infers them.
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

    @classmethod
    def from_declaration(
        cls, declaration: ProjectExtensionDeclaration
    ) -> "CompilerContract":
        """Read the logical contract the project declares it compiles against."""
        return cls(
            contract_id=declaration.compiler_contract_id,
            contract_version=declaration.compiler_contract_version,
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
        references: Cross-document references the compile resolved.
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
    resolved_deltas: Mapping[str, Any] = ()
    references: Sequence[Mapping[str, Any]] = ()
    assertions: Sequence[Mapping[str, Any]] = ()
    identity_contributions: Mapping[str, Any] = ()
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
        "references": [dict(reference) for reference in inputs.references],
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
    caught by the reader as well as by the writer.
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
    return lock


__all__ = [
    "EXPERIMENT_COMPILE_LOCK_MIGRATION_TABLE",
    "EXPERIMENT_COMPILE_LOCK_SCHEMA_ID",
    "EXPERIMENT_COMPILE_LOCK_SCHEMA_VERSION",
    "EXPERIMENT_COMPILE_LOCK_SCHEMA_VERSION_V1",
    "EXPERIMENT_COMPILE_LOCK_SUPPORTED_SCHEMA_VERSIONS",
    "RUN_RECEIPT_ONLY_FACTS",
    "CompileLockError",
    "CompileLockInputs",
    "CompilerContract",
    "CompilerImplementation",
    "PlanReceiptBoundaryError",
    "build_compile_lock",
    "check_plan_receipt_boundary",
    "load_compile_lock",
]
