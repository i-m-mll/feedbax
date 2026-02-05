"""Pydantic models for graph specifications."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field


# Use Any for nested param values to avoid recursive type issues
ParamValue = Union[int, float, str, bool, None, List[Any], Dict[str, Any]]


class ParamSchema(BaseModel):
    """Schema for a component parameter."""

    name: str
    type: Literal["int", "float", "bool", "str", "enum", "array", "object", "bounds2d"]
    default: Optional[ParamValue] = None
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    options: Optional[List[str]] = None
    description: Optional[str] = None
    required: bool = False
    nested_schema: Optional[List["ParamSchema"]] = None


class ComponentSpec(BaseModel):
    """Specification for a component instance in a graph."""

    type: str
    params: Dict[str, ParamValue] = Field(default_factory=dict)
    input_ports: List[str] = Field(default_factory=list)
    output_ports: List[str] = Field(default_factory=list)


class WireSpec(BaseModel):
    """Specification for a wire connecting two ports."""

    source_node: str
    source_port: str
    target_node: str
    target_port: str


class UserPortSpec(BaseModel):
    """User-defined ports for a subgraph."""

    inputs: List[str] = Field(default_factory=list)
    outputs: List[str] = Field(default_factory=list)


class TapTransform(BaseModel):
    """Transform applied by a tap."""

    type: str
    params: Dict[str, ParamValue] = Field(default_factory=dict)


class TapSpec(BaseModel):
    """Specification for a tap (probe or intervention point)."""

    id: str
    type: Literal["probe", "intervention"]
    position: Dict[str, Any]
    paths: Dict[str, str] = Field(default_factory=dict)
    transform: Optional[TapTransform] = None


class BarnacleSpec(BaseModel):
    """Specification for a barnacle (attached probe/intervention)."""

    id: str
    kind: Literal["probe", "intervention"]
    timing: Literal["input", "output"]
    label: str
    read_paths: List[str] = Field(default_factory=list)
    write_paths: List[str] = Field(default_factory=list)
    transform: str = ""


class GraphMetadata(BaseModel):
    """Metadata for a graph."""

    name: str
    description: Optional[str] = None
    created_at: str
    updated_at: str
    version: str = "1.0.0"
    author: Optional[str] = None
    tags: Optional[List[str]] = None


class GraphSpec(BaseModel):
    """Complete specification for a computation graph."""

    nodes: Dict[str, ComponentSpec] = Field(default_factory=dict)
    wires: List[WireSpec] = Field(default_factory=list)
    input_ports: List[str] = Field(default_factory=list)
    output_ports: List[str] = Field(default_factory=list)
    input_bindings: Dict[str, Tuple[str, str]] = Field(default_factory=dict)
    output_bindings: Dict[str, Tuple[str, str]] = Field(default_factory=dict)
    subgraphs: Optional[Dict[str, "GraphSpec"]] = None
    barnacles: Optional[Dict[str, List[BarnacleSpec]]] = None
    user_ports: Optional[Dict[str, UserPortSpec]] = None
    taps: Optional[List[TapSpec]] = None
    metadata: Optional[GraphMetadata] = None


class EdgeRoutingPoint(BaseModel):
    """A point in edge routing."""

    x: float
    y: float


class EdgeRouting(BaseModel):
    """Routing information for an edge."""

    style: Literal["bezier", "elbow"] = "bezier"
    points: List[EdgeRoutingPoint] = Field(default_factory=list)


class EdgeUIState(BaseModel):
    """UI state for an edge."""

    routing: EdgeRouting


class NodeUIState(BaseModel):
    """UI state for a node."""

    position: Dict[str, float]
    collapsed: bool = False
    selected: bool = False
    size: Optional[Dict[str, float]] = None


class TapUIState(BaseModel):
    """UI state for a tap."""

    position: Dict[str, float]
    selected: Optional[bool] = None


class GraphUIState(BaseModel):
    """UI state for the entire graph."""

    viewport: Dict[str, float] = Field(default_factory=lambda: {"x": 0, "y": 0, "zoom": 1})
    node_states: Dict[str, NodeUIState] = Field(default_factory=dict)
    edge_states: Optional[Dict[str, EdgeUIState]] = None
    subgraph_states: Optional[Dict[str, "GraphUIState"]] = None
    tap_states: Optional[Dict[str, TapUIState]] = None


class GraphProject(BaseModel):
    """A complete graph project with metadata and UI state."""

    metadata: GraphMetadata
    graph: GraphSpec
    ui_state: Optional[GraphUIState] = None


class ValidationError(BaseModel):
    """A validation error."""

    type: str
    message: str
    location: Optional[Dict[str, str]] = None


class ValidationWarning(BaseModel):
    """A validation warning."""

    type: str
    message: str
    location: Optional[Dict[str, str]] = None


class ValidationResult(BaseModel):
    """Result of graph validation."""

    valid: bool
    errors: List[ValidationError] = Field(default_factory=list)
    warnings: List[ValidationWarning] = Field(default_factory=list)
    cycles: List[List[str]] = Field(default_factory=list)


# Enable forward references
GraphSpec.model_rebuild()
GraphUIState.model_rebuild()
ParamSchema.model_rebuild()
