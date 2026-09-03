"""Authored-experiment envelope dispatch for the one Feedbax dialect.

Feedbax owns exactly one authoring entrypoint,
``python -m feedbax preflight-experiment-envelope <envelope>``, and exactly one
authored envelope dialect, :data:`EXPERIMENT_ENVELOPE_SCHEMA_VERSION`. There is
no compiler seam: no project registers a compiler, no plugin claims a schema
string, and no callable can be injected between an authored envelope and the
document it compiles to. Dispatch is direct.

The dispatcher's whole job is: read the envelope, read its declared schema,
refuse it unless that schema is the one built-in dialect, compile it, and turn
the outcome into the documented exit codes.

* exit 0 — accepted; the compiler wrote its declared outputs
* exit 2 — the envelope was rejected; stderr names the offending field, the
  rejection category, and the correct home for the content
* exit 1 — infrastructure failure, distinguishable from a rejection

Rejection categories are a closed set. A failure that does not name a category
from this set is describing a different kind of failure and must say so through
a new schema version rather than inventing a category string.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator

from feedbax.contracts.base import StrictModel
from feedbax.contracts.project_experiment import ProjectExperimentDeclaration
from feedbax.contracts.run_composition import AuthoredIntentParent, ResolvedOutputParent
from feedbax.contracts.run_matrix import TrainingRowParentProvenance

EXPERIMENT_ENVELOPE_COMPILE_RESULT_SCHEMA_ID = "feedbax.spec.experiment_envelope_compile_result"
EXPERIMENT_ENVELOPE_COMPILE_RESULT_SCHEMA_VERSION = (
    "feedbax.spec.experiment_envelope_compile_result.v1"
)

#: The envelope field naming the dialect the envelope is authored in.
ENVELOPE_SCHEMA_FIELD = "schema"


class ExperimentEnvelopeRejectionCategory(StrEnum):
    """Closed set of authored-envelope rejection categories.

    This is the one rejection vocabulary the authoring surface speaks. The
    engine kernel and the dialect compiler name a category from this set; a
    failure that needs a category not listed here is describing a different
    kind of failure and must say so through a new schema version rather than
    inventing a category string of its own.
    """

    UNKNOWN_FIELD = "unknown-field"
    MISSING_FIELD = "missing-field"
    INVALID_VALUE = "invalid-value"
    DUPLICATE_KEY = "duplicate-key"
    NONCANONICAL_FORMAT = "noncanonical-format"
    ECHOED_INHERITED_VALUE = "echoed-inherited-value"
    DERIVED_VALUE_AUTHORED = "derived-value-authored"
    BUDGET_EXCEEDED = "budget-exceeded"
    ASSERTION_FAILED = "assertion-failed"
    ILLEGAL_ASSERTION_PATH = "illegal-assertion-path"
    UNRESOLVED_ROW_KEY = "unresolved-row-key"
    EMPTY_SELECTION = "empty-selection"
    UNSUPPORTED_SCHEMA_VERSION = "unsupported-schema-version"
    UNRESOLVED_BASE = "unresolved-base"
    CROSS_FAMILY_BASE = "cross-family-base"
    RETIRED_BASE_FAMILY = "retired-base-family"
    UNRESOLVED_UPSTREAM_REFERENCE = "unresolved-upstream-reference"
    CO_CREATED_PROTECTED_DOCUMENT = "co-created-protected-document"
    MISSING_PARENT_AUTHORITY = "missing-parent-authority"
    AMBIGUOUS_PARENT_AUTHORITY = "ambiguous-parent-authority"
    UNDECLARED_PARENT_AUTHORITY = "undeclared-parent-authority"
    PARENT_SEMANTIC_DRIFT = "parent-semantic-drift"
    PARENT_BYTE_DRIFT = "parent-byte-drift"


class ExperimentEnvelopeRejection(ValueError):
    """An authored envelope was rejected; the author can act on this alone."""

    def __init__(
        self,
        category: ExperimentEnvelopeRejectionCategory,
        message: str,
        *,
        field: str | None = None,
        correct_home: str | None = None,
    ) -> None:
        super().__init__(message)
        self.category = ExperimentEnvelopeRejectionCategory(category)
        self.field = field
        self.correct_home = correct_home

    def render(self) -> str:
        """Return the one-line diagnostic written to stderr."""
        parts = [f"category={self.category.value}"]
        if self.field is not None:
            parts.append(f"field={self.field}")
        parts.append(str(self))
        if self.correct_home is not None:
            parts.append(f"correct home: {self.correct_home}")
        return "envelope rejected: " + "; ".join(parts)


class PendingProductCustodyError(RuntimeError):
    """The envelope is well-formed but the data it names has not been produced.

    This is a repository-state failure, not an authoring failure: no edit to the
    envelope fixes it, so the entrypoint reports it as infrastructure (exit 1)
    and never as a rejection. It is the explicit counterpart of
    :class:`ExperimentEnvelopeRejection`, and the reason the two exit codes stay
    distinguishable when a compile plan outruns the runs that feed it.
    """

    def __init__(self, roles: tuple[str, ...], custody_ref: str) -> None:
        self.roles = tuple(roles)
        self.custody_ref = custody_ref
        super().__init__(
            f"the data this envelope names has not been produced: {sorted(self.roles)} still "
            f"await a run receipt. The receipt layer writes {custody_ref}; until it exists, "
            "compilation resolves the reference but cannot emit a document naming unproduced "
            "data."
        )


class ExperimentEnvelopeCompilerError(ValueError):
    """Raised when envelope dispatch itself cannot proceed."""


@dataclass(frozen=True)
class ExperimentEnvelopeParentAuthority:
    """Exact declared composition-parent bytes available during compilation.

    The caller owns artifact acquisition. Feedbax receives the immutable bytes
    together with the typed semantic parent and the existing governed-parent
    provenance record, then authenticates all three before using the payload.
    """

    provenance: TrainingRowParentProvenance
    parent: AuthoredIntentParent | ResolvedOutputParent
    payload_bytes: bytes

    def __init__(
        self,
        *,
        provenance: TrainingRowParentProvenance,
        parent: AuthoredIntentParent | ResolvedOutputParent,
        payload_bytes: bytes,
    ) -> None:
        object.__setattr__(self, "provenance", provenance.model_copy(deep=True))
        object.__setattr__(self, "parent", parent.model_copy(deep=True))
        object.__setattr__(self, "payload_bytes", bytes(payload_bytes))


@dataclass(frozen=True)
class ExperimentEnvelopeCompileRequest:
    """Everything the compiler is given, and nothing more.

    ``project_declaration`` is the data declaration of the project whose envelope
    directory holds this envelope. The caller resolves it by directory before
    dispatch, so the compiler never reads a project identity out of the envelope.
    """

    envelope: Mapping[str, Any]
    envelope_path: Path
    repo_root: Path
    out_dir: Path
    project_declaration: ProjectExperimentDeclaration | None = None
    parent_authorities: tuple[ExperimentEnvelopeParentAuthority, ...] = ()


class ExperimentEnvelopeCompileResult(StrictModel):
    """The versioned outcome the dialect compiler returns to the dispatcher."""

    schema_id: str = EXPERIMENT_ENVELOPE_COMPILE_RESULT_SCHEMA_ID
    schema_version: str = EXPERIMENT_ENVELOPE_COMPILE_RESULT_SCHEMA_VERSION
    envelope_schema: str
    name: str
    family: str
    compile_lock_path: str
    document_path: str
    extra_outputs: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_result(self) -> "ExperimentEnvelopeCompileResult":
        if self.schema_id != EXPERIMENT_ENVELOPE_COMPILE_RESULT_SCHEMA_ID:
            raise ValueError(
                f"unsupported ExperimentEnvelopeCompileResult schema_id: {self.schema_id!r}"
            )
        if self.schema_version != EXPERIMENT_ENVELOPE_COMPILE_RESULT_SCHEMA_VERSION:
            raise ValueError(
                "unsupported ExperimentEnvelopeCompileResult schema_version: "
                f"{self.schema_version!r}"
            )
        for name in ("envelope_schema", "name", "family", "compile_lock_path", "document_path"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"ExperimentEnvelopeCompileResult {name} must be nonempty")
        return self

    @property
    def outputs(self) -> tuple[str, ...]:
        """Return every declared output path in stable order."""
        return (self.compile_lock_path, self.document_path, *self.extra_outputs)


def envelope_schema_of(envelope: Mapping[str, Any]) -> str:
    """Return the declared envelope schema or reject the document."""
    if not isinstance(envelope, Mapping):
        raise ExperimentEnvelopeCompilerError("authored envelope must be a JSON object")
    value = envelope.get(ENVELOPE_SCHEMA_FIELD)
    if not isinstance(value, str) or not value.strip():
        raise ExperimentEnvelopeRejection(
            ExperimentEnvelopeRejectionCategory.UNKNOWN_FIELD,
            f"authored envelope declares no {ENVELOPE_SCHEMA_FIELD!r} string, so the dialect it "
            "is written in is unstated",
            field=ENVELOPE_SCHEMA_FIELD,
            correct_home="the envelope's first line, naming its authored schema",
        )
    return value


def require_builtin_envelope_schema(schema: str) -> None:
    """Refuse any authored schema outside the one built-in dialect's versions.

    There is exactly one dialect, so an envelope declaring anything else is an
    authoring error the author can fix by naming a supported version. No
    fallback, no inference, and no second compiler exists to try instead. The
    dialect's supported versions are enumerated, never inferred, and each is
    compiled as the grammar it names rather than being widened into the current
    one.
    """
    # Local import: the dialect module imports this one for its rejection
    # vocabulary, so the supported-schema constants can only be read here.
    from feedbax.contracts.experiment_envelope_dialect import (
        EXPERIMENT_ENVELOPE_SCHEMA_VERSION,
        EXPERIMENT_ENVELOPE_SUPPORTED_SCHEMA_VERSIONS,
    )

    if schema not in EXPERIMENT_ENVELOPE_SUPPORTED_SCHEMA_VERSIONS:
        raise ExperimentEnvelopeRejection(
            ExperimentEnvelopeRejectionCategory.UNSUPPORTED_SCHEMA_VERSION,
            f"authored envelope declares schema {schema!r}, but Feedbax compiles exactly one "
            f"envelope dialect: supported="
            f"{list(EXPERIMENT_ENVELOPE_SUPPORTED_SCHEMA_VERSIONS)}, "
            f"current={EXPERIMENT_ENVELOPE_SCHEMA_VERSION!r}",
            field=ENVELOPE_SCHEMA_FIELD,
            correct_home=(
                f"the envelope's {ENVELOPE_SCHEMA_FIELD!r} field, set to "
                f"{EXPERIMENT_ENVELOPE_SCHEMA_VERSION!r}"
            ),
        )


def dispatch_experiment_envelope(
    envelope: Mapping[str, Any],
    *,
    envelope_path: Path,
    repo_root: Path,
    out_dir: Path,
    project_declaration: ProjectExperimentDeclaration | None = None,
    parent_authorities: Sequence[ExperimentEnvelopeParentAuthority] = (),
) -> ExperimentEnvelopeCompileResult:
    """Compile one authored envelope with the single built-in dialect compiler."""
    schema = envelope_schema_of(envelope)
    require_builtin_envelope_schema(schema)
    # Local import: the built-in compiler is an implementation module that
    # imports this contract, so resolving it at module scope would be a cycle.
    from feedbax.envelope.entrypoint import compile_experiment_envelope

    result = compile_experiment_envelope(
        ExperimentEnvelopeCompileRequest(
            envelope=envelope,
            envelope_path=envelope_path,
            repo_root=repo_root,
            out_dir=out_dir,
            project_declaration=project_declaration,
            parent_authorities=tuple(parent_authorities),
        )
    )
    if result.envelope_schema != schema:
        raise ExperimentEnvelopeCompilerError(
            f"the built-in compiler reported envelope schema {result.envelope_schema!r} for a "
            f"{schema!r} document"
        )
    return result


def missing_outputs(result: ExperimentEnvelopeCompileResult, out_dir: Path) -> Sequence[str]:
    """Return declared outputs the compiler did not actually write."""
    return tuple(
        path
        for path in result.outputs
        if not (Path(path) if Path(path).is_absolute() else out_dir / path).is_file()
    )


__all__ = [
    "ENVELOPE_SCHEMA_FIELD",
    "EXPERIMENT_ENVELOPE_COMPILE_RESULT_SCHEMA_ID",
    "EXPERIMENT_ENVELOPE_COMPILE_RESULT_SCHEMA_VERSION",
    "ExperimentEnvelopeCompileRequest",
    "ExperimentEnvelopeParentAuthority",
    "ExperimentEnvelopeCompileResult",
    "ExperimentEnvelopeCompilerError",
    "ExperimentEnvelopeRejection",
    "ExperimentEnvelopeRejectionCategory",
    "PendingProductCustodyError",
    "dispatch_experiment_envelope",
    "envelope_schema_of",
    "missing_outputs",
    "require_builtin_envelope_schema",
]
