"""Materialize a registered causal graph as an editable Model composite."""

from __future__ import annotations

from feedbax.compiler.model_document import lower_model_document
from feedbax.component_registry import ComponentRegistry
from feedbax.component_registry.declarations import DeclaredComponent
from feedbax.contracts.graph import GraphSpec, WireSpec
from feedbax.contracts.model_address import ComponentPortEnd, InputBoundaryEnd, OutputBoundaryEnd
from feedbax.contracts.model_document import (
    CompositeComponent, DelayAttachment, GraphInterior, LeafComponent, ModelDocument, ModelPorts, ModelTime,
    ModelWire, SignalInput, SignalOutput, SignalType,
)


def instantiate_model_template(name: str, registry: ComponentRegistry, period: float) -> CompositeComponent:
    declaration = registry.get(name)
    if declaration is None:
        raise ValueError(f"Unknown component definition {name!r}")
    return _template_component(declaration, registry, period)


def template_placement_reason(declaration: DeclaredComponent, registry: ComponentRegistry) -> str | None:
    """An intrinsic reason a template cannot be placed, independent of the receiving Model."""
    if not isinstance(declaration.template_graph, GraphSpec) and declaration.name != "feedbax.AcausalSystem":
        return None
    try:
        _template_component(declaration, registry, None)
    except ValueError as error:
        return str(error)
    return None


def _template_component(
    declaration: DeclaredComponent, registry: ComponentRegistry, period: float | None,
) -> CompositeComponent:
    graph = declaration.template_graph
    if not isinstance(graph, GraphSpec):
        if declaration.name == "feedbax.AcausalSystem":
            raise ValueError("Equations templates require an authored physical domain")
        raise ValueError(f"{declaration.name} does not declare a causal graph template")
    return _composite(declaration.name.rsplit(".", 1)[-1], graph, registry, period)


def _component_end(node: str, port: str) -> ComponentPortEnd:
    return ComponentPortEnd(owner="component", component=node, port=port)


def _composite(name: str, graph: GraphSpec, registry: ComponentRegistry, period: float | None) -> CompositeComponent:
    if period is not None and graph.step_seconds not in (None, period):
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
            components[node_id] = _composite(node_id, child, registry, period)
            continue
        fields = node.model_dump(mode="json")["params"]
        for key in ("dt", "control_dt"):
            if key in fields:
                step_length = fields.pop(key)
                if period is not None and step_length != period:
                    raise ValueError(f"The template {name} component {node_id} requires its own step length")
        components[node_id] = LeafComponent(
            kind="leaf", name=node_id, type=node.type, revision=node.revision, fields=fields,
        )

    wires = {}
    recurrent_shapes = {}
    for wire in graph.wires:
        key = wire.authored_id()
        attachments = ()
        if wire.temporality == "recurrent":
            reset = wire.recurrent_initializer or {}
            shape = reset.get("shape")
            if reset.get("seed_channel_history"):
                raise ValueError(f"The template {name} initializes channel history from state, which Model Delay cannot store")
            if (
                reset.get("kind") != "zeros" or reset.get("dtype", "float32") != "float32"
                or not isinstance(shape, (list, tuple)) or any(type(size) is not int or size <= 0 for size in shape)
            ):
                raise ValueError(f"The template {name} has a recurrent reset that Model Delay cannot store")
            recurrent_shapes[key] = tuple(shape)
            attachments = (DelayAttachment(kind="delay", id="delay", delay=ModelTime(value=1, unit="st")),)
        wires[key] = ModelWire(
            source=_component_end(wire.source_node, wire.source_port),
            target=_component_end(wire.target_node, wire.target_port),
            attachments=attachments,
        )
    for port, (node, target) in graph.input_bindings.items():
        key = WireSpec(source_node="$inputs", source_port=port, target_node=node, target_port=target).authored_id()
        wires[key] = ModelWire(source=InputBoundaryEnd(owner="inputs", port=port), target=_component_end(node, target))
    for port, (node, source) in graph.output_bindings.items():
        key = WireSpec(source_node=node, source_port=source, target_node="$outputs", target_port=port).authored_id()
        wires[key] = ModelWire(source=_component_end(node, source), target=OutputBoundaryEnd(owner="outputs", port=port))
    if len(wires) != len(graph.wires) + len(graph.input_bindings) + len(graph.output_bindings):
        raise ValueError(f"The template {name} repeats a wire")
    component = CompositeComponent(
        kind="composite", name=name,
        ports=ModelPorts(
            inputs=tuple(SignalInput(id=port, name=port, type=SignalType()) for port in graph.input_ports),
            outputs=tuple(SignalOutput(id=port, name=port, type=SignalType()) for port in graph.output_ports),
        ),
        interior=GraphInterior(components=components, wires=wires),
    )

    if recurrent_shapes:
        # Shape qualification uses a delay in whole steps, independent of the receiving Model's period.
        validation_period = period if period is not None else graph.step_seconds or 1.0
        model = ModelDocument(step_seconds=validation_period, ports=component.ports, interior=component.interior)
        lowered = lower_model_document(model, registry).graph
        resolved = {
            wire.id: tuple(wire.recurrent_initializer["shape"])
            for wire in lowered.wires
            if wire.id in recurrent_shapes and wire.recurrent_initializer is not None
        }
        if resolved != recurrent_shapes:
            raise ValueError(f"The template {name} has a recurrent reset shape that Model Delay cannot preserve")
    return component
