"""Versioned durable form for neutral scientific declarations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from feedbax.contracts.authored_canonical import canonical_bytes
from feedbax.contracts.scientific_compiler_schema import (
    DECLARATION_DOCUMENT_SCHEMA_ID,
    DECLARATION_DOCUMENT_SCHEMA_VERSION,
)
from feedbax.contracts.strict_json import strict_json_loads

from .core import Declaration, DeclarationCompositionError


class DeclarationDocument(BaseModel):
    """Serializable neutral declaration facts with an explicit protocol identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal[DECLARATION_DOCUMENT_SCHEMA_ID] = DECLARATION_DOCUMENT_SCHEMA_ID
    schema_version: str = DECLARATION_DOCUMENT_SCHEMA_VERSION
    kind: str = Field(min_length=1)
    type_id: str = Field(min_length=1)
    payload_schema_id: str = Field(min_length=1)
    payload_schema_version: str = Field(min_length=1)
    capabilities: tuple[str, ...]
    runtime_protocol_id: str = Field(min_length=1)
    owner: str = Field(min_length=1)

    @field_validator("schema_version")
    @classmethod
    def reject_unsupported_version(cls, value: str) -> str:
        if value != DECLARATION_DOCUMENT_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported DeclarationDocument schema version: "
                f"source_version={value!r}; current_version="
                f"{DECLARATION_DOCUMENT_SCHEMA_VERSION!r}; "
                "migration_intentionally_absent=yes"
            )
        return value

    @field_validator("capabilities")
    @classmethod
    def require_canonical_capabilities(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or any(not capability for capability in value):
            raise ValueError("DeclarationDocument capabilities must be non-empty strings")
        if value != tuple(sorted(set(value))):
            raise ValueError(
                "DeclarationDocument capabilities must be sorted and unique"
            )
        return value


def serialize_declaration(
    declaration: Declaration,
    *,
    runtime_protocol_id: str,
) -> bytes:
    """Serialize one declaration without inferring its durable protocol identity."""
    document = DeclarationDocument(
        kind=declaration.kind,
        type_id=declaration.type_id,
        payload_schema_id=declaration.schema_id,
        payload_schema_version=declaration.schema_version,
        capabilities=tuple(sorted(declaration.capabilities)),
        runtime_protocol_id=runtime_protocol_id,
        owner=declaration.owner,
    )
    return canonical_bytes(document.model_dump(mode="json"))


def load_declaration(
    raw: bytes,
    *,
    runtime_protocols: Mapping[str, type[object]],
) -> Declaration:
    """Load one current declaration document through an explicit protocol registry."""
    try:
        payload = strict_json_loads(raw)
        document = DeclarationDocument.model_validate(payload)
    except (TypeError, ValueError) as exc:
        raise DeclarationCompositionError(f"invalid declaration document: {exc}") from exc
    try:
        runtime_protocol = runtime_protocols[document.runtime_protocol_id]
    except KeyError as exc:
        raise DeclarationCompositionError(
            "declaration document names unknown runtime protocol "
            f"{document.runtime_protocol_id!r}"
        ) from exc
    if not isinstance(runtime_protocol, type):
        raise DeclarationCompositionError(
            f"runtime protocol {document.runtime_protocol_id!r} must resolve to a type"
        )
    return Declaration(
        kind=document.kind,
        type_id=document.type_id,
        schema_id=document.payload_schema_id,
        schema_version=document.payload_schema_version,
        capabilities=frozenset(document.capabilities),
        runtime_protocol=runtime_protocol,
        owner=document.owner,
    )


__all__ = [
    "DECLARATION_DOCUMENT_SCHEMA_ID",
    "DECLARATION_DOCUMENT_SCHEMA_VERSION",
    "DeclarationDocument",
    "load_declaration",
    "serialize_declaration",
]
