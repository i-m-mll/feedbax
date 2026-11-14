"""Structured diagnostics for retained Studio graph preparation."""

from __future__ import annotations

from typing import Any

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
