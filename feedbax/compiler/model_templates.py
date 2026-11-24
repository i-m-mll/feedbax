"""Materialize a registered causal graph as an editable Model composite."""

from __future__ import annotations

from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.graph import GraphSpec, WireSpec
from feedbax.contracts.model_address import ComponentPortEnd, InputBoundaryEnd, OutputBoundaryEnd
from feedbax.contracts.model_document import (
    CompositeComponent, GraphInterior, LeafComponent, ModelPorts, ModelWire, SignalInput, SignalOutput, SignalType,
)


def instantiate_model_template(name: str, registry: ComponentRegistry, period: float) -> CompositeComponent:
    declaration = registry.get(name)
    if declaration is None:
        raise ValueError(f"Unknown component definition {name!r}")
    graph = declaration.template_graph
    if not isinstance(graph, GraphSpec):
        raise ValueError(f"{name} does not declare a causal graph template")
    return _composite(declaration.name.rsplit(".", 1)[-1], graph, period)


def _component_end(node: str, port: str) -> ComponentPortEnd:
    return ComponentPortEnd(owner="component", component=node, port=port)


def _composite(name: str, graph: GraphSpec, period: float) -> CompositeComponent:
    if graph.step_seconds not in (None, period):
        raise ValueError(f"The template {name} requires its own step length")
    unsupported = [
        field for field in (
            "additive_channel_adapters", "derived_dimensions", "barnacles", "user_ports",
            "taps", "retained_observables", "parameter_constraints",
        )
        if getattr(graph, field)
    ]
    if unsupported:
        raise ValueError(f"The template {name} has features Model composites cannot store: {', '.join(unsupported)}")
    if any(wire.temporality != "instant" or wire.recurrent_initializer is not None for wire in graph.wires):
        raise ValueError(f"The template {name} has recurrent wires that cannot be placed as Model wires")
    if set(graph.input_ports) != graph.input_bindings.keys() or set(graph.output_ports) != graph.output_bindings.keys():
        raise ValueError(f"The template {name} does not bind every boundary port")
    subgraphs = graph.subgraphs or {}
    if subgraphs.keys() - graph.nodes.keys():
        raise ValueError(f"The template {name} has an interior without a component")

    components = {}
    for node_id, node in graph.nodes.items():
        if node_id in subgraphs:
            child = subgraphs[node_id]
            if node.type != "feedbax.Subgraph" or node.params or not isinstance(child, GraphSpec):
                raise ValueError(f"The template {name} has an unsupported interior at {node_id}")
            components[node_id] = _composite(node_id, child, period)
            continue
        fields = node.model_dump(mode="json")["params"]
        for key in ("dt", "control_dt"):
            if key in fields and fields.pop(key) != period:
                raise ValueError(f"The template {name} component {node_id} requires its own step length")
        components[node_id] = LeafComponent(
            kind="leaf", name=node_id, type=node.type, revision=node.revision, fields=fields,
        )

    wires = {
        wire.authored_id(): ModelWire(
            source=_component_end(wire.source_node, wire.source_port),
            target=_component_end(wire.target_node, wire.target_port),
        )
        for wire in graph.wires
    }
    if len(wires) != len(graph.wires):
        raise ValueError(f"The template {name} repeats a wire")
    for port, (node, target) in graph.input_bindings.items():
        key = WireSpec(source_node="$inputs", source_port=port, target_node=node, target_port=target).authored_id()
        wires[key] = ModelWire(source=InputBoundaryEnd(owner="inputs", port=port), target=_component_end(node, target))
    for port, (node, source) in graph.output_bindings.items():
        key = WireSpec(source_node=node, source_port=source, target_node="$outputs", target_port=port).authored_id()
        wires[key] = ModelWire(source=_component_end(node, source), target=OutputBoundaryEnd(owner="outputs", port=port))
    return CompositeComponent(
        kind="composite", name=name,
        ports=ModelPorts(
            inputs=tuple(SignalInput(id=port, name=port, type=SignalType()) for port in graph.input_ports),
            outputs=tuple(SignalOutput(id=port, name=port, type=SignalType()) for port in graph.output_ports),
        ),
        interior=GraphInterior(components=components, wires=wires),
    )
