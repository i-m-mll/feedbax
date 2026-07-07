"""Pydantic models for component definitions."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from feedbax.contracts.graph import GraphSpec, GraphUIState, ParamSchema, ParamValue
from feedbax.contracts.representation import RepresentationSpec


class PortType(BaseModel):
    """Type information for a port."""

    dtype: str
    shape: Optional[List[int]] = None
    rank: Optional[int] = None


class PortTypeSpec(BaseModel):
    """Port type specifications for a component."""

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

    name: str
    category: str
    description: str
    param_schema: List[ParamSchema] = Field(default_factory=list)
    input_ports: List[str] = Field(default_factory=list)
    output_ports: List[str] = Field(default_factory=list)
    icon: str = "box"
    default_params: Dict[str, ParamValue] = Field(default_factory=dict)
    port_types: Optional[PortTypeSpec] = None
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
