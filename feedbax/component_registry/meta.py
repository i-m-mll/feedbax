from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from feedbax.contracts.component import ComponentIdentity, ComponentMigrationInfo, PortTypeSpec
from feedbax.contracts.graph import GraphSpec, GraphUIState, NodeUIState, ParamSchema
from feedbax.contracts.representation import RepresentationSpec

if TYPE_CHECKING:
    from feedbax.contracts.graph import ParamValue
    from feedbax.runtime.graph import Component


ComponentBuilder = Callable[[Mapping[str, Any]], "Component"]
OutputPrototypeFn = Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]]


class MissingPrototypeInput(ValueError):
    """Raised when an output prototype needs an input prototype not yet known."""


@dataclass
class ComponentMeta:
    name: str
    category: str
    description: str
    param_schema: List[ParamSchema]
    input_ports: List[str]
    output_ports: List[str]
    icon: str = "box"
    port_types: Optional[PortTypeSpec] = None
    is_composite: bool = False
    template_graph: Optional[GraphSpec] = None
    template_ui_state: Optional[GraphUIState] = None
    template_id: Optional[str] = None
    template_kind: Optional[str] = None
    builder: Optional[ComponentBuilder] = None
    output_prototype_fn: Optional[OutputPrototypeFn] = None
    provenance: Optional[str] = None
    identity: Optional[ComponentIdentity] = None
    owner: Optional[str] = None
    param_schema_version: str = "1"
    supported_param_schema_versions: List[str] = field(default_factory=list)
    migrations: List[ComponentMigrationInfo] = field(default_factory=list)
    trainable_by_default: bool = False
    representation: Optional[RepresentationSpec] = None

    def __setattr__(self, name: str, value: Any) -> None:
        if (
            name == "output_prototype_fn"
            and hasattr(self, "output_prototype_fn")
            and getattr(self, "provenance", None) == "feedbax"
            and value is not getattr(self, "output_prototype_fn")
        ):
            raise AttributeError(
                "Feedbax-owned ComponentMeta.output_prototype_fn is immutable; "
                "register output prototypes in feedbax component metadata"
            )
        super().__setattr__(name, value)

    def __post_init__(self) -> None:
        if not self.supported_param_schema_versions:
            self.supported_param_schema_versions = [self.param_schema_version]
        elif self.param_schema_version not in self.supported_param_schema_versions:
            self.supported_param_schema_versions = [
                *self.supported_param_schema_versions,
                self.param_schema_version,
            ]

    @property
    def default_params(self) -> Dict[str, ParamValue]:
        return {schema.name: schema.default for schema in self.param_schema}


def _template_ui_state(
    positions: Dict[str, tuple[float, float]],
    *,
    subgraph_states: Optional[Dict[str, GraphUIState]] = None,
) -> GraphUIState:
    return GraphUIState(
        node_states={
            node_id: NodeUIState(position={"x": x, "y": y}) for node_id, (x, y) in positions.items()
        },
        subgraph_states=subgraph_states,
    )
