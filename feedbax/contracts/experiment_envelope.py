"""Generic authored-experiment envelope dispatch.

Feedbax owns exactly one authoring entrypoint,
``python -m feedbax preflight-experiment-envelope <envelope>``, and no
knowledge of any downstream envelope dialect. A downstream project registers a
compiler that claims one envelope schema string through the ordinary plugin
bootstrap, with its registry injected like every other family; there is no
import-time discovery and no module-path convention.

The dispatcher's whole job is: read the envelope, read its declared schema,
find the one compiler claiming that schema, hand it the request, and turn the
outcome into the documented exit codes.

* exit 0 — accepted; the compiler wrote its declared outputs
* exit 2 — the envelope was rejected; stderr names the offending field, the
  rejection category, and the correct home for the content
* exit 1 — infrastructure failure, distinguishable from a rejection

Rejection categories are a closed set. A compiler that needs a category not
listed here is describing a different kind of failure and must say so through a
new schema version rather than inventing a category string.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

from pydantic import Field, model_validator

from feedbax.contracts.manifest import StrictModel
from feedbax.contracts.project_experiment import ProjectExperimentDeclaration
from feedbax.registry_errors import RegistryCollisionError

EXPERIMENT_ENVELOPE_COMPILE_RESULT_SCHEMA_ID = "feedbax.spec.experiment_envelope_compile_result"
EXPERIMENT_ENVELOPE_COMPILE_RESULT_SCHEMA_VERSION = (
    "feedbax.spec.experiment_envelope_compile_result.v1"
)

#: The envelope field naming the dialect a downstream compiler claims.
ENVELOPE_SCHEMA_FIELD = "schema"


class ExperimentEnvelopeRejectionCategory(StrEnum):
    """Closed set of authored-envelope rejection categories.

    This is the one rejection vocabulary the authoring surface speaks. The
    engine kernel and every downstream compiler name a category from this set;
    a compiler that needs a category not listed here is describing a different
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


class ExperimentEnvelopeCompilerCollisionError(
    ExperimentEnvelopeCompilerError,
    RegistryCollisionError,
):
    """Raised when two compilers claim the same envelope schema."""


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


class ExperimentEnvelopeCompileResult(StrictModel):
    """The versioned outcome a downstream compiler returns to the dispatcher."""

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


@runtime_checkable
class ExperimentEnvelopeCompiler(Protocol):
    """One downstream compiler for one envelope schema."""

    def __call__(
        self, request: ExperimentEnvelopeCompileRequest
    ) -> ExperimentEnvelopeCompileResult:
        """Compile one authored envelope or raise ExperimentEnvelopeRejection."""
        ...


@dataclass(frozen=True)
class ExperimentEnvelopeCompilerRegistration:
    """One compiler bound to the exact envelope schema string it claims."""

    envelope_schema: str
    owner: str
    compile: Callable[[ExperimentEnvelopeCompileRequest], ExperimentEnvelopeCompileResult]

    def __post_init__(self) -> None:
        if not self.envelope_schema.strip():
            raise ExperimentEnvelopeCompilerError(
                "experiment envelope compiler must claim a nonempty envelope schema"
            )
        if not self.owner.strip():
            raise ExperimentEnvelopeCompilerError(
                "experiment envelope compiler must declare an owner"
            )
        if not callable(self.compile):
            raise ExperimentEnvelopeCompilerError(
                "experiment envelope compiler must supply a callable"
            )


class ExperimentEnvelopeCompilerRegistry:
    """Injected registry resolving one compiler per envelope schema."""

    def __init__(self) -> None:
        self._sealed = False
        self._registrations: dict[str, ExperimentEnvelopeCompilerRegistration] = {}

    def register(self, registration: ExperimentEnvelopeCompilerRegistration) -> None:
        """Register one compiler; a duplicate schema claim fails closed."""
        if self._sealed:
            raise RuntimeError("experiment envelope compiler registry is sealed")
        if not isinstance(registration, ExperimentEnvelopeCompilerRegistration):
            raise TypeError("registration must be an ExperimentEnvelopeCompilerRegistration")
        existing = self._registrations.get(registration.envelope_schema)
        if existing is not None:
            raise ExperimentEnvelopeCompilerCollisionError(
                f"envelope schema {registration.envelope_schema!r} is already claimed by "
                f"{existing.owner!r}"
            )
        self._registrations[registration.envelope_schema] = registration

    def get(self, envelope_schema: str) -> ExperimentEnvelopeCompilerRegistration:
        """Return the single compiler claiming *envelope_schema* or fail closed."""
        registration = self._registrations.get(envelope_schema)
        if registration is None:
            known = ", ".join(sorted(self._registrations)) or "none"
            raise ExperimentEnvelopeCompilerError(
                f"no registered compiler claims envelope schema {envelope_schema!r}; "
                f"registered schemas: {known}"
            )
        return registration

    def available_keys(self) -> tuple[str, ...]:
        """Return every claimed envelope schema, in stable order."""
        return tuple(sorted(self._registrations))

    def seal(self) -> None:
        """Prevent further registration after bootstrap completes."""
        self._sealed = True


def envelope_schema_of(envelope: Mapping[str, Any]) -> str:
    """Return the declared envelope schema or reject the document."""
    if not isinstance(envelope, Mapping):
        raise ExperimentEnvelopeCompilerError("authored envelope must be a JSON object")
    value = envelope.get(ENVELOPE_SCHEMA_FIELD)
    if not isinstance(value, str) or not value.strip():
        raise ExperimentEnvelopeRejection(
            ExperimentEnvelopeRejectionCategory.UNKNOWN_FIELD,
            f"authored envelope declares no {ENVELOPE_SCHEMA_FIELD!r} string, so no compiler "
            "can claim it",
            field=ENVELOPE_SCHEMA_FIELD,
            correct_home="the envelope's first line, naming its authored schema",
        )
    return value


def dispatch_experiment_envelope(
    envelope: Mapping[str, Any],
    registry: ExperimentEnvelopeCompilerRegistry,
    *,
    envelope_path: Path,
    repo_root: Path,
    out_dir: Path,
    project_declaration: ProjectExperimentDeclaration | None = None,
) -> ExperimentEnvelopeCompileResult:
    """Route one authored envelope to the compiler claiming its schema."""
    schema = envelope_schema_of(envelope)
    registration = registry.get(schema)
    result = registration.compile(
        ExperimentEnvelopeCompileRequest(
            envelope=envelope,
            envelope_path=envelope_path,
            repo_root=repo_root,
            out_dir=out_dir,
            project_declaration=project_declaration,
        )
    )
    if not isinstance(result, ExperimentEnvelopeCompileResult):
        raise ExperimentEnvelopeCompilerError(
            f"compiler {registration.owner!r} returned {type(result).__name__}, not an "
            "ExperimentEnvelopeCompileResult"
        )
    if result.envelope_schema != schema:
        raise ExperimentEnvelopeCompilerError(
            f"compiler {registration.owner!r} reported envelope schema "
            f"{result.envelope_schema!r} for a {schema!r} document"
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
    "ExperimentEnvelopeCompileResult",
    "ExperimentEnvelopeCompiler",
    "ExperimentEnvelopeCompilerCollisionError",
    "ExperimentEnvelopeCompilerError",
    "ExperimentEnvelopeCompilerRegistration",
    "ExperimentEnvelopeCompilerRegistry",
    "ExperimentEnvelopeRejection",
    "ExperimentEnvelopeRejectionCategory",
    "PendingProductCustodyError",
    "dispatch_experiment_envelope",
    "envelope_schema_of",
    "missing_outputs",
]
