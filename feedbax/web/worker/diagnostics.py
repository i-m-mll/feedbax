"""Structured worker diagnostics for Studio training transport."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError as PydanticValidationError

from feedbax.contracts.domain import DomainDiagnostic
from feedbax.studio.schema import SchemaValidationIssue, schema_issues_to_domain_diagnostics


class GraphCompilationError(ValueError):
    """Worker preflight error carrying structured graph diagnostics."""

    def __init__(self, message: str, diagnostics: list[DomainDiagnostic]) -> None:
        details = "; ".join(
            f"{diagnostic.code}: {diagnostic.message}" for diagnostic in diagnostics
        )
        super().__init__(f"{message}: {details}" if details else message)
        self.diagnostics = diagnostics


def graph_compilation_error(
    message: str,
    issues: list[SchemaValidationIssue],
) -> GraphCompilationError:
    """Build a graph compilation error from schema validation issues."""
    return GraphCompilationError(message, schema_issues_to_domain_diagnostics(issues))


def object_required_diagnostic(name: str) -> DomainDiagnostic:
    """Return a diagnostic for a missing or malformed worker request object."""
    return DomainDiagnostic(
        severity="error",
        code="worker.invalid_request",
        message=f"Training worker requires {name} to be an object",
        location={"path": f"/{name}"},
        details={"field": name},
    )


def model_validation_diagnostics(
    prefix: str,
    exc: PydanticValidationError,
) -> list[DomainDiagnostic]:
    """Convert Pydantic validation errors into structured diagnostics."""
    return [
        DomainDiagnostic(
            severity="error",
            code="worker.schema_error",
            message=str(error.get("msg", "Invalid value")),
            location={"path": f"/{prefix}/" + "/".join(str(part) for part in error.get("loc", ()))},
            details={"field": prefix},
        )
        for error in exc.errors()
    ]


def exception_diagnostic(
    exc: Exception,
    *,
    code: str,
    message: str | None = None,
    details: dict[str, Any] | None = None,
) -> DomainDiagnostic:
    """Return one diagnostic for a worker exception without including a traceback."""
    headline = f"{type(exc).__name__}: {exc}"
    return DomainDiagnostic(
        severity="error",
        code=code,
        message=message or headline,
        details={"exception_type": type(exc).__name__, **(details or {})},
    )
