"""Lower the stored model tree into the shared graph compiler's input."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256

from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.graph import ComponentSpec, GraphSpec, StudioTaskBinding, StudioTaskBindingSpec, WireSpec
from feedbax.contracts.model_address import FieldElement, ModelAddress
from feedbax.contracts.model_delay import ModelDelayCarry
from feedbax.contracts.model_document import (
    DelayAttachment, GraphInterior, InterventionAttachment, ModelDocument, ModelPorts,
    PhysicalInterior, ProbeAttachment, SignalInput,
)
from feedbax.contracts.project_inputs import ProjectName
from feedbax.contracts.project import json_pointer_parts
from feedbax.runtime.model_values import MODEL_OWNERS
from feedbax.runtime.task_data_roles import task_data_role
from feedbax.studio.model_ports import ModelRegistryPorts
from feedbax.studio.protocol import task_binding_id


class ModelCompilationProblem(ValueError):
    def __init__(self, address: ModelAddress, detail: str):
        super().__init__(detail)
        self.address = address


def _generated(role: str, *parts: str) -> str:
    return "_model_" + role + "_" + sha256(canonical_json_v2_bytes(list(parts))).hexdigest()


@dataclass
class _Lower:
    document: str
    model: ModelDocument
    components: ComponentRegistry
    ports: ModelRegistryPorts
    origins: dict[tuple[str, ...], ModelAddress] = field(default_factory=dict)

    def address(self, path, kind, **fields) -> ModelAddress:
        return ModelAddress.model_validate({
            "document": self.document, "composites": path, "element": {"kind": kind, **fields},
        })

    def problem(self, path, kind, detail, **fields) -> ModelCompilationProblem:
        return ModelCompilationProblem(self.address(path, kind, **fields), detail)

    def remember(self, path, derived, element):
        tokens = tuple(token for parent in path for token in ("subgraphs", parent))
        self.origins[("graph", *tokens, *derived)] = self.address(path, **element)

    def graph(
        self, interior: GraphInterior | PhysicalInterior, ports: ModelPorts, path: tuple[str, ...],
        *, available: frozenset[str] = frozenset(),
    ) -> GraphSpec:
        if isinstance(interior, PhysicalInterior):
            if interior.kind == "mechanics" and ports.physical:
                port = ports.physical[0]
                raise self.problem(
                    path, "port", "Mechanics physical interfaces are unavailable for this execution",
                    end={"owner": "boundary", "port": port.id},
                )
            raise self.problem(path, "graph", "this physical interior has no qualified model-document lowering")
        nodes, subgraphs = {}, {}
        input_bindings, output_bindings = {}, {}
        wires = []
        input_sources = {}
        output_names = {port.id: port.id if path else port.name for port in ports.outputs}

        def add_generated(key, type_id, params, *, origin, inputs=("input",)):
            if key in interior.components or key in nodes:
                raise self.problem(path, "component", "component ID collides with a generated occurrence", id=key)
            nodes[key] = ComponentSpec(
                type=type_id, params=params, input_ports=list(inputs), output_ports=["output"],
            )
            self.remember(path, ("nodes", key), origin)

        for key, component in interior.components.items():
            self.remember(path, ("nodes", key), {"kind": "component", "id": key})
            if component.kind == "composite":
                if component.substeps != 1:
                    raise self.problem(path, "component", "substeps require qualified mechanics lowering", id=key)
                child = self.graph(component.interior, component.ports, (*path, key))
                if component.attachments.sources or component.attachments.sinks:
                    raise self.problem(path, "component", "composite port attachments need owner lowering", id=key)
                subgraphs[key] = child
                nodes[key] = ComponentSpec(
                    type="feedbax.Subgraph", params={},
                    input_ports=child.input_ports, output_ports=child.output_ports,
                )
            else:
                definition = self.components.get(component.type)
                if definition is None or definition.is_composite:
                    raise self.problem(path, "component", f"unknown leaf type {component.type!r}", id=key)
                params = dict(component.fields)
                if "dt" in params or "control_dt" in params:
                    raise self.problem(path, "component", "component periods are supplied by the model root", id=key)
                if declaration := MODEL_OWNERS.get(component.type):
                    if declaration.clock:
                        params[declaration.clock] = self.model.step_seconds
                policy = definition.dynamic_port_policy
                if policy and policy.count_param in component.port_counts:
                    params[policy.count_param] = component.port_counts[policy.count_param]
                layout = self.components.dynamic_port_layout(component.type, params)
                nodes[key] = ComponentSpec(
                    type=component.type, revision=component.revision, params=params,
                    input_ports=list(layout.input_ports if layout else definition.input_ports),
                    output_ports=list(layout.output_ports if layout else definition.output_ports),
                )

        def emit(source, target, key, origin, *, reset=None):
            self.remember(path, ("wires", str(len(wires))), origin)
            wires.append(WireSpec(
                id=key, source_node=source[0], source_port=source[1],
                target_node=target[0], target_port=target[1],
                temporality="recurrent" if reset is not None else "instant",
                recurrent_initializer=reset,
            ))

        def transform(source, chain, key, at, *, finish=None):
            pending_reset = None
            for attachment in chain:
                origin = {"kind": "attachment", "parent": at, "id": attachment.id}
                if isinstance(attachment, ProbeAttachment):
                    raise self.problem(path, "attachment", "probe observation is unavailable in this milestone", parent=at, id=attachment.id)
                if isinstance(attachment, DelayAttachment):
                    length = attachment.delay.value
                    if attachment.delay.unit == "s":
                        length /= self.model.step_seconds
                    if round(length) != 1 or attachment.initial != 0 or pending_reset is not None:
                        raise self.problem(path, "attachment", "only a one-step zero-initialized delay is qualified", parent=at, id=attachment.id)
                    shape, dtype = source_type
                    if shape is None:
                        raise self.problem(path, "attachment", "delay has no resolved reset shape", parent=at, id=attachment.id)
                    if dtype != "float32":
                        raise self.problem(path, "attachment", "delay requires a float32 reset", parent=at, id=attachment.id)
                    pending_reset = ModelDelayCarry(
                        shape=tuple(shape), dtype=dtype, model_delay_id=attachment.id, model_wire_id=key,
                    ).model_dump(mode="json")
                    relay = _generated("delay", key, attachment.id)
                    add_generated(relay, "feedbax.Input", {}, origin=origin)
                    emit(source, (relay, "input"), _generated("delay_input", key, attachment.id), origin)
                    source = (relay, "output")
                    continue
                if isinstance(attachment, InterventionAttachment):
                    if attachment.window is not None:
                        raise self.problem(path, "attachment", "intervention window lowering is unavailable", parent=at, id=attachment.id)
                    parameters = attachment.fields
                    match attachment.operation.lower():
                        case "scale" if type(parameters.get("factor")) in (int, float):
                            type_id, params = "feedbax.Scale", {"scale": parameters["factor"], "fixed": True}
                        case "clamp" if all(type(parameters.get(name)) in (int, float) for name in ("low", "high")):
                            if parameters["low"] > parameters["high"]:
                                raise self.problem(path, "attachment", "clamp bounds are reversed", parent=at, id=attachment.id)
                            type_id, params = "feedbax.Saturation", {"min_val": parameters["low"], "max_val": parameters["high"]}
                        case _:
                            raise self.problem(path, "attachment", "unrecognized intervention or fields", parent=at, id=attachment.id)
                    segment = _generated("attachment", key, attachment.id)
                    add_generated(segment, type_id, params, origin=origin)
                    emit(source, (segment, "input"), _generated("segment", key, attachment.id), origin, reset=pending_reset)
                    pending_reset = None
                    source = (segment, "output")
            if finish is not None:
                origin = (
                    {"kind": "attachment", "parent": at, "id": pending_reset["model_delay_id"]}
                    if pending_reset is not None else at
                )
                emit(source, finish, key, origin, reset=pending_reset)
            return source, pending_reset

        source_type = (None, "float32")
        for port in ports.inputs:
            external = port.id if path else port.name
            relay = _generated("input", port.id)
            origin = {"kind": "port", "end": {"owner": "inputs", "port": port.id}}
            default = not path and external not in available and port.default is not None
            if default:
                add_generated(relay, "feedbax.Constant", {"value": port.default.value}, origin=origin, inputs=())
            else:
                add_generated(relay, "feedbax.Input", {}, origin=origin)
                input_bindings[external] = (relay, "input")
            source, _ = transform(
                (relay, "output"), interior.attachments.sources.get(port.id, ()), relay,
                origin,
            )
            input_sources[port.id] = source

        component_sources = {}
        for key, component in interior.components.items():
            for port, chain in component.attachments.sources.items():
                source, _ = transform(
                    (key, port), chain, _generated("port", key, port),
                    {"kind": "port", "end": {"owner": "component", "component": key,
                                             "port": port, "direction": "output"}},
                )
                component_sources[key, port] = source

        for key, wire in sorted(interior.wires.items()):
            a, b = wire.source, wire.target
            if a.owner == "inputs":
                source = input_sources.get(a.port)
                input_port = next((port for port in ports.inputs if port.id == a.port), None)
                shape = input_port.type.shape if input_port else None
                source_type = (list(shape) if shape is not None and all(isinstance(n, int) for n in shape) else None, "float32")
            else:
                source = component_sources.get((a.component, a.port), (a.component, a.port))
                if a.component not in nodes or a.port not in nodes[a.component].output_ports:
                    source = None
                node = nodes.get(a.component)
                component = interior.components.get(a.component)
                declaration = self.ports.port(component.model_dump(mode="json"), "output", a.port) if component and component.kind == "leaf" else None
                shape = declaration.type.shape if declaration and not isinstance(declaration.type, str) else None
                if node and node.type == "feedbax.GRU" and a.port in {"hidden", "output"}:
                    shape = (node.params["hidden_size"],)
                source_type = (list(shape) if shape is not None and all(isinstance(n, int) for n in shape) else None,
                               node.params.get("dtype", "float32") if node else "float32")
            if source is None:
                raise self.problem(path, "wire", "source port does not exist", id=key)
            at = {"kind": "wire", "id": key}
            if b.owner == "outputs":
                external = output_names.get(b.port)
                if external is None:
                    raise self.problem(path, "wire", "boundary output port does not exist", id=key)
                if external in output_bindings:
                    raise self.problem(path, "wire", "boundary output already has a driver", id=key)
                output, reset = transform(source, wire.attachments, key, at)
                if reset is not None:
                    relay = _generated("output", b.port)
                    add_generated(relay, "feedbax.Input", {}, origin=at)
                    emit(output, (relay, "input"), key, at, reset=reset)
                    output = (relay, "output")
                output_bindings[external] = output
            else:
                target = nodes.get(b.component)
                if target is None or b.port not in target.input_ports:
                    raise self.problem(path, "wire", "target port does not exist", id=key)
                if any(w.target_node == b.component and w.target_port == b.port for w in wires):
                    raise self.problem(path, "wire", "target port already has a driver", id=key)
                transform(source, wire.attachments, key, at, finish=(b.component, b.port))
        return GraphSpec(
            nodes=nodes, wires=wires, input_ports=list(input_bindings), output_ports=list(output_names.values()),
            input_bindings=input_bindings, output_bindings=output_bindings, subgraphs=subgraphs or None,
        )

    def root_inputs(self, binding: StudioTaskBindingSpec, graph: GraphSpec) -> StudioTaskBindingSpec:
        signals = {item.id: item for item in binding.exposed_data if item.bindable}
        result = []
        for port in self.model.ports.inputs:
            if port.name not in graph.input_bindings:
                continue
            node, port_id = graph.input_bindings[port.name]
            item = signals.get(port.name)
            if item is None:
                raise self.problem(
                    (), "port", f"task has no compatible signal named {port.name!r}",
                    end={"owner": "inputs", "port": port.id},
                )
            if not self._compatible_task_signal(port, item):
                raise self.problem(
                    (), "port", f"task signal {port.name!r} does not match the root input type",
                    end={"owner": "inputs", "port": port.id},
                )
            self.origins[("task_binding_spec", "bindings", str(len(result)))] = self.address(
                (), "port", end={"owner": "inputs", "port": port.id},
            )
            result.append(StudioTaskBinding(
                id=task_binding_id(item.id, node, port_id), source_data_id=item.id,
                target_node_id=node, target_port=port_id, role=task_data_role(item),
            ))
        return binding.model_copy(update={"bindings": result})

    @staticmethod
    def _compatible_task_signal(port: SignalInput, item) -> bool:
        expected = item.expected_shape
        shape = expected[1:] if expected and expected[0] == "time" else expected
        return not (
            port.type.unit is not None and item.units is not None and port.type.unit != item.units
            or port.type.shape is not None and shape is not None and (
                len(port.type.shape) != len(shape)
                or any(type(a) is int and type(b) is int and a != b for a, b in zip(port.type.shape, shape))
            )
        )


@dataclass(frozen=True)
class LoweredModelDocument:
    graph: GraphSpec
    task: StudioTaskBindingSpec | None
    origins: dict[tuple[str, ...], ModelAddress]

    def source_address(self, path: str) -> ModelAddress | None:
        parts = json_pointer_parts(path)
        for stop in range(len(parts), 0, -1):
            if address := self.origins.get(parts[:stop]):
                if address.element.kind == "component" and parts[stop:stop + 1] == ("params",):
                    return address.model_copy(update={"element": FieldElement(
                        kind="field", parent=address.element, path=("fields", *parts[stop + 1:]),
                    )})
                return address
        return None


def lower_model_document(
    document: ProjectName, model: ModelDocument, components: ComponentRegistry,
    task: StudioTaskBindingSpec | None = None,
) -> LoweredModelDocument:
    """Project one authored model through the graph compiler's registered declarations."""
    lower = _Lower(document, model, components, ModelRegistryPorts(components))
    if model.world is not None:
        raise lower.problem((), "field", "model world requires qualified mechanics lowering",
                            parent={"kind": "graph"}, path=["world"])
    names = frozenset(item.id for item in task.exposed_data if item.bindable) if task else frozenset()
    graph = lower.graph(model.interior, model.ports, (), available=names)
    binding = lower.root_inputs(task, graph) if task is not None else None
    return LoweredModelDocument(graph, binding, lower.origins)
