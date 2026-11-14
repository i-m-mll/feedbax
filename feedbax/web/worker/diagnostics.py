"""Structured worker diagnostics for Studio training transport."""

from __future__ import annotations

from pydantic import ValidationError as PydanticValidationError

from feedbax.contracts.domain import DomainDiagnostic


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
