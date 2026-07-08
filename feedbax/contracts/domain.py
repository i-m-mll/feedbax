"""Contracts for graph-domain metadata served to Studio.

Domain compile reports are derived authoring cache records. Loaders accept only
``feedbax.spec.domain_compile_report.v1``; unknown or older report versions are
dropped from project caches so the corresponding node returns to
``never_compiled`` rather than trusting stale derived data.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DOMAIN_REGISTRY_PAYLOAD_SCHEMA_ID = "feedbax.spec.domain"
DOMAIN_REGISTRY_PAYLOAD_SCHEMA_VERSION = "feedbax.spec.domain.v1"
DOMAIN_COMPILE_REPORT_SCHEMA_ID = "feedbax.spec.domain_compile_report"
DOMAIN_COMPILE_REPORT_SCHEMA_VERSION = "feedbax.spec.domain_compile_report.v1"

CAUSAL_DOMAIN_ID = "feedbax.domain.causal"
ACAUSAL_DOMAIN_ID = "feedbax.domain.acausal"
MECHANICS_DOMAIN_ID = "feedbax.domain.mechanics"
PENZAI_DOMAIN_ID = "feedbax.domain.penzai"
DomainCompileStatus = Literal["never_compiled", "stale", "compiling", "ok", "ok_with_warnings", "error"]
DomainReportStatus = Literal["ok", "ok_with_warnings", "error"]


def validate_domain_id(domain_id: str) -> str:
    """Validate a domain registry identifier."""
    if not domain_id.startswith("feedbax.domain."):
        raise ValueError(
            "Domain id must use the feedbax.domain.* namespace: "
            f"domain_id={domain_id!r}"
        )
    suffix = domain_id.removeprefix("feedbax.domain.")
    if not suffix or any(part == "" for part in suffix.split(".")):
        raise ValueError(f"Domain id must include a non-empty suffix: domain_id={domain_id!r}")
    return domain_id


class EditorCapability(BaseModel):
    """Studio editor capability for a domain."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["canvas", "tree", "inspector", "none"]
    editable: bool


class DomainTheme(BaseModel):
    """Frontend theme tokens for a domain."""

    model_config = ConfigDict(extra="forbid")

    color: str
    icon: str
    edge_style: Literal["directed", "undirected"]


class DomainMeta(BaseModel):
    """Registry metadata for one graph domain."""

    model_config = ConfigDict(extra="forbid")

    id: str
    display_name: str
    interior_schema_id: str | None
    edge_semantics: Literal["directed", "undirected"]
    allows_multi_edge_per_port: bool
    nestable_domains: list[str]
    editor: EditorCapability
    theme: DomainTheme
    compiler_id: str | None

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        return validate_domain_id(value)

    @field_validator("nestable_domains")
    @classmethod
    def _validate_nestable_domains(cls, value: list[str]) -> list[str]:
        return [validate_domain_id(domain_id) for domain_id in value]


class DomainRegistryPayload(BaseModel):
    """Versioned payload for ``GET /api/domains``."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal[DOMAIN_REGISTRY_PAYLOAD_SCHEMA_ID] = DOMAIN_REGISTRY_PAYLOAD_SCHEMA_ID
    schema_version: Literal[DOMAIN_REGISTRY_PAYLOAD_SCHEMA_VERSION] = (
        DOMAIN_REGISTRY_PAYLOAD_SCHEMA_VERSION
    )
    domains: list[DomainMeta]


class DomainDiagnostic(BaseModel):
    """Structured diagnostic emitted by a domain authoring compiler."""

    model_config = ConfigDict(extra="forbid")

    severity: Literal["error", "warning", "info"]
    code: str
    message: str
    node_ids: List[str] = Field(default_factory=list)
    ports: List[Tuple[str, str]] = Field(default_factory=list)
    variables: List[str] = Field(default_factory=list)
    counts: Optional[Dict[str, int]] = None


class DomainCompileReport(BaseModel):
    """Authoring-time structural compile report for one domain interior."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal[DOMAIN_COMPILE_REPORT_SCHEMA_ID] = DOMAIN_COMPILE_REPORT_SCHEMA_ID
    schema_version: Literal[DOMAIN_COMPILE_REPORT_SCHEMA_VERSION] = (
        DOMAIN_COMPILE_REPORT_SCHEMA_VERSION
    )
    status: DomainReportStatus
    interior_content_hash: str
    diagnostics: List[DomainDiagnostic] = Field(default_factory=list)
    derived_interface: Optional[Dict[str, Any]] = None
    summary: Dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_status_matches_diagnostics(self) -> "DomainCompileReport":
        has_error = any(diagnostic.severity == "error" for diagnostic in self.diagnostics)
        has_warning = any(diagnostic.severity == "warning" for diagnostic in self.diagnostics)
        if self.status == "ok" and has_error:
            raise ValueError("DomainCompileReport status 'ok' cannot include error diagnostics")
        if self.status == "ok" and has_warning:
            raise ValueError("DomainCompileReport status 'ok' cannot include warning diagnostics")
        if self.status == "ok_with_warnings" and has_error:
            raise ValueError(
                "DomainCompileReport status 'ok_with_warnings' cannot include error diagnostics"
            )
        if self.status == "error" and not has_error:
            raise ValueError("DomainCompileReport status 'error' requires an error diagnostic")
        return self


def report_status_for_diagnostics(diagnostics: list[DomainDiagnostic]) -> DomainReportStatus:
    """Derive the persisted report status from diagnostic severities."""

    if any(diagnostic.severity == "error" for diagnostic in diagnostics):
        return "error"
    if any(diagnostic.severity == "warning" for diagnostic in diagnostics):
        return "ok_with_warnings"
    return "ok"


def derive_compile_status(
    report: DomainCompileReport | None,
    *,
    current_interior_hash: str,
) -> DomainCompileStatus:
    """Return the visible authoring status for a cached compile report."""

    if report is None:
        return "never_compiled"
    if report.interior_content_hash != current_interior_hash:
        return "stale"
    return report.status
