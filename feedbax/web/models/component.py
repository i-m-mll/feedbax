"""Pydantic models for component definitions."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from feedbax.web.models.graph import ParamSchema, ParamValue


class PortType(BaseModel):
    """Type information for a port."""

    dtype: str
    shape: Optional[List[int]] = None
    rank: Optional[int] = None


class PortTypeSpec(BaseModel):
    """Port type specifications for a component."""

    inputs: Dict[str, PortType] = Field(default_factory=dict)
    outputs: Dict[str, PortType] = Field(default_factory=dict)


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
