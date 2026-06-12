from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from feedbax.contracts.component import PortTypeSpec
from feedbax.contracts.graph import GraphSpec, GraphUIState, NodeUIState, ParamSchema

if TYPE_CHECKING:
    from feedbax.contracts.graph import ParamValue
    from feedbax.graph import Component


ComponentBuilder = Callable[[Mapping[str, Any]], "Component"]


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
    provenance: Optional[str] = None

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
