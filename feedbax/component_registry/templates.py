from __future__ import annotations

from typing import Protocol

from feedbax.contracts.component import PortType, PortTypeSpec
from feedbax.contracts.graphs.templates import (
    BUILTIN_GRAPH_TEMPLATES,
    recurrent_controller_template_graph,
    simple_feedback_template_graph,
)

from .meta import ComponentMeta, _template_ui_state


class _Registry(Protocol):
    def register(self, meta: ComponentMeta) -> None: ...


def register_builtin_graph_templates(registry: _Registry) -> None:
    metadata = {template.id: template for template in BUILTIN_GRAPH_TEMPLATES}

    network_meta = metadata["feedbax.templates.recurrent_controller"]
    network_graph = recurrent_controller_template_graph(
        input_size=6,
        hidden_size=100,
        out_size=2,
        cell_type="GRU",
        out_nonlinearity="tanh",
    )
    registry.register(
        ComponentMeta(
            name=network_meta.name,
            category=network_meta.category,
            description=network_meta.description,
            param_schema=[],
            input_ports=list(network_graph.input_ports),
            output_ports=list(network_graph.output_ports),
            icon='CircuitBoard',
            is_composite=True,
            port_types=PortTypeSpec(
                inputs={
                    'input': PortType(dtype='vector'),
                    'feedback': PortType(dtype='vector'),
                },
                outputs={
                    'output': PortType(dtype='vector'),
                    'hidden': PortType(dtype='vector'),
                },
            ),
            template_graph=network_graph,
            template_ui_state=_template_ui_state(
                {
                    'input_mux': (80, 160),
                    'cell': (280, 160),
                    'readout': (500, 160),
                }
            ),
            template_id=network_meta.id,
            template_kind=network_meta.kind,
            trainable_by_default=True,
        )
    )

    feedback_meta = metadata["feedbax.templates.simple_feedback"]
    feedback_graph = simple_feedback_template_graph()
    registry.register(
        ComponentMeta(
            name=feedback_meta.name,
            category=feedback_meta.category,
            description=feedback_meta.description,
            param_schema=[],
            input_ports=list(feedback_graph.input_ports),
            output_ports=list(feedback_graph.output_ports),
            icon='Radar',
            is_composite=True,
            port_types=PortTypeSpec(
                inputs={'input': PortType(dtype='vector')},
                outputs={'effector': PortType(dtype='state')},
            ),
            template_graph=feedback_graph,
            template_ui_state=_template_ui_state(
                {
                    'feedback': (120, 260),
                    'feedback_ravel': (120, 110),
                    'input_mux': (340, 180),
                    'cell': (520, 180),
                    'readout': (700, 180),
                    'efferent': (880, 180),
                    'mechanics': (1060, 180),
                },
            ),
            template_id=feedback_meta.id,
            template_kind=feedback_meta.kind,
            trainable_by_default=True,
        )
    )
