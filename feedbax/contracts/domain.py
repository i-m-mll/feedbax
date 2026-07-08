"""Contracts for graph-domain metadata served to Studio."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from feedbax.contracts.acausal import ACAUSAL_GRAPH_SCHEMA_ID  # noqa: F401


DOMAIN_REGISTRY_PAYLOAD_SCHEMA_ID = "feedbax.spec.domain"
DOMAIN_REGISTRY_PAYLOAD_SCHEMA_VERSION = "feedbax.spec.domain.v1"
DOMAIN_DIAGNOSTIC_SCHEMA_ID = "feedbax.diagnostic.domain"
DOMAIN_DIAGNOSTIC_SCHEMA_VERSION = "feedbax.diagnostic.domain.v1"

CAUSAL_DOMAIN_ID = "feedbax.domain.causal"
ACAUSAL_DOMAIN_ID = "feedbax.domain.acausal"
MECHANICS_DOMAIN_ID = "feedbax.domain.mechanics"
PENZAI_DOMAIN_ID = "feedbax.domain.penzai"

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
    """Structured diagnostic emitted by Studio domain validation and compilation."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal[DOMAIN_DIAGNOSTIC_SCHEMA_ID] = DOMAIN_DIAGNOSTIC_SCHEMA_ID
    schema_version: Literal[DOMAIN_DIAGNOSTIC_SCHEMA_VERSION] = DOMAIN_DIAGNOSTIC_SCHEMA_VERSION
    severity: Literal["error", "warning", "info"] = "error"
    code: str
    message: str
    node_ids: list[str] = Field(default_factory=list)
    location: dict[str, Any] | None = None
    details: dict[str, Any] = Field(default_factory=dict)
