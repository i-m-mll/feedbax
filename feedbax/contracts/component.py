"""Pydantic models for component definitions."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from feedbax.contracts.domain import CAUSAL_DOMAIN_ID
from feedbax.contracts.graph import GraphSpec, GraphUIState, ParamSchema, ParamValue
from feedbax.contracts.representation import RepresentationSpec


COMPONENT_DEFINITION_SCHEMA_ID = "feedbax.spec.component_definition"
COMPONENT_DEFINITION_SCHEMA_VERSION_V1 = "feedbax.spec.component_definition.v1"
COMPONENT_DEFINITION_SCHEMA_VERSION = "feedbax.spec.component_definition.v2"
COMPONENT_DEFINITION_PORT_KIND_MIGRATION_ID = "component-definition-v1-to-v2-port-kind"


def _migrate_port_type_payload(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    if "kind" in payload:
        return payload
    return {**payload, "kind": "signal"}


def migrate_component_definition_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate legacy component definitions to the conserving-port schema."""

    schema_version = payload.get("schema_version")
    if schema_version not in {None, COMPONENT_DEFINITION_SCHEMA_VERSION_V1}:
        return dict(payload)

    migrated = dict(payload)
    migrated.setdefault("schema_id", COMPONENT_DEFINITION_SCHEMA_ID)
    migrated["schema_version"] = COMPONENT_DEFINITION_SCHEMA_VERSION
    port_types = migrated.get("port_types")
    if isinstance(port_types, dict):
        migrated["port_types"] = {
            **port_types,
            "inputs": {
                key: _migrate_port_type_payload(value)
                for key, value in dict(port_types.get("inputs") or {}).items()
            },
            "outputs": {
                key: _migrate_port_type_payload(value)
                for key, value in dict(port_types.get("outputs") or {}).items()
            },
        }
    return migrated


class PortType(BaseModel):
    """Type information for a port."""

    model_config = ConfigDict(extra="forbid")

    dtype: str = "scalar"
    shape: Optional[List[int]] = None
    rank: Optional[int] = None
    kind: Literal["signal", "conserving"] = "signal"
    physical_domain: Optional[str] = None
    across_vars: Optional[List[str]] = None
    through_var: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_signal_port(cls, data: Any) -> Any:
        return _migrate_port_type_payload(data)


class PortTypeSpec(BaseModel):
    """Port type specifications for a component."""

    model_config = ConfigDict(extra="forbid")

    inputs: Dict[str, PortType] = Field(default_factory=dict)
    outputs: Dict[str, PortType] = Field(default_factory=dict)


class ComponentIdentity(BaseModel):
    """Stable ownership and provenance for a component type."""

    type_id: str
    owner: Optional[str] = None
    provenance: Optional[str] = None
    provenance_kind: Literal["feedbax", "package", "file", "local", "unknown"] = "unknown"
    package: Optional[str] = None
    import_path: Optional[str] = None
    stable: bool = False


class ComponentMigrationInfo(BaseModel):
    """Discoverable component ID or parameter-schema migration edge."""

    migration_id: str
    owner: str
    source_type: str
    target_type: str
    source_param_schema_version: Optional[str] = None
    target_param_schema_version: Optional[str] = None
    description: str = ""


class ComponentDefinition(BaseModel):
    """Definition of a component type available in the library."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal[COMPONENT_DEFINITION_SCHEMA_ID] = COMPONENT_DEFINITION_SCHEMA_ID
    schema_version: Literal[COMPONENT_DEFINITION_SCHEMA_VERSION] = (
        COMPONENT_DEFINITION_SCHEMA_VERSION
    )
    name: str
    category: str
    description: str
    param_schema: List[ParamSchema] = Field(default_factory=list)
    input_ports: List[str] = Field(default_factory=list)
    output_ports: List[str] = Field(default_factory=list)
    icon: str = "box"
    default_params: Dict[str, ParamValue] = Field(default_factory=dict)
    port_types: Optional[PortTypeSpec] = None
    domain: str = CAUSAL_DOMAIN_ID
    interior_domain: Optional[str] = None
    is_composite: bool = False
    template_graph: Optional[GraphSpec] = None
    template_ui_state: Optional[GraphUIState] = None
    template_id: Optional[str] = None
    template_kind: Optional[str] = None
    provenance: Optional[str] = None
    identity: Optional[ComponentIdentity] = None
    owner: Optional[str] = None
    param_schema_version: str = "1"
    supported_param_schema_versions: List[str] = Field(default_factory=lambda: ["1"])
    migrations: List[ComponentMigrationInfo] = Field(default_factory=list)
    trainable_by_default: bool = False
    representation: Optional[RepresentationSpec] = None

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_definition(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        return migrate_component_definition_payload(data)
