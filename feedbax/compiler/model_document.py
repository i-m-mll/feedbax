"""Lower the stored model tree into the shared graph compiler's input."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from math import isfinite
from collections.abc import Mapping

import numpy as np

from feedbax.component_registry import ComponentRegistry
from feedbax.compiler.resolution import GraphParameterResolution, resolve_graph_parameters
from feedbax.compiler.prototypes import prototypes_from_task_bindings
from feedbax.contracts.acausal import AcausalConnectionSpec, AcausalGraphSpec, PlanarDynamicsSpec, SolverConfigSpec
from feedbax.contracts.acausal_interface import (
    PLANAR_STATE_ONLY_QUANTITIES, derive_acausal_interface, is_acausal_type,
)
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.expression_units import UNITS
from feedbax.contracts.expression_values import Level, LiteralValue, NameValue, ScopedValue, ValueBase, ValueType
from feedbax.contracts.graph import ComponentSpec, GraphSpec, StudioTaskBinding, StudioTaskBindingSpec, WireSpec
from feedbax.contracts.model_address import (
    ComponentPortEnd, FieldElement, ModelAddress, PhysicalEnd, PortElement, SignalSource,
)
from feedbax.contracts.model_delay import ModelDelayCarry
from feedbax.contracts.model_document import (
    DelayAttachment, GraphInterior, InterventionAttachment, ModelDocument, ModelPorts,
    PhysicalInterior, ProbeAttachment, SignalInput,
)
from feedbax.contracts.project_inputs import ProjectName
from feedbax.contracts.project import json_pointer_parts
from feedbax.runtime.task_data_roles import task_data_role
from feedbax.runtime.model_reset import ModelResetPlan, resolve_model_reset
from feedbax.runtime._signal_schedule import SignalScheduleError, resolve_signal_schedule
from feedbax.studio.protocol import task_binding_id


class ModelCompilationProblem(ValueError):
    def __init__(self, address: ModelAddress, detail: str):
        super().__init__(detail)
        self.address = address


def model_tau_seconds(tau: object, period: float) -> float:
    time = LiteralValue.model_validate(tau) if isinstance(tau, dict) else LiteralValue(value=tau, unit="st")
    if time.unit not in {"st", "s"} or type(time.value) not in {float, int} or time.value <= 0:
        raise ValueError("Tau requires a positive value in seconds or steps")
    value = time.value * (period if time.unit == "st" else 1)
    if not isfinite(value) or value <= 0:
        raise ValueError("Tau does not resolve to a positive finite duration")
    return value


def _generated(role: str, *parts: str) -> str:
    return "_model_" + role + "_" + sha256(canonical_json_v2_bytes(list(parts))).hexdigest()


@dataclass(frozen=True)
class _Signal:
    shape: tuple[int, ...] | None = None
    dtype: str = "float32"
    units: str | None = None


@dataclass
class _Lower:
    document: str
    model: ModelDocument
    components: ComponentRegistry
    origins: dict[tuple[str, ...], ModelAddress] = field(default_factory=dict)
    links: list[ResolvedModelLink] = field(default_factory=list)
    plugins: set[str] = field(default_factory=set)

    def address(self, scope, kind, **fields) -> ModelAddress:
        return ModelAddress.model_validate({
            "document": self.document, "composites": scope, "element": {"kind": kind, **fields},
        })

    def problem(self, scope, kind, detail, **fields) -> ModelCompilationProblem:
        return ModelCompilationProblem(self.address(scope, kind, **fields), detail)

    def remember(self, path, derived, element):
        tokens = tuple(token for parent in path for token in ("subgraphs", parent))
        self.origins[("graph", *tokens, *derived)] = self.address(path, **element)

    def source_signal(
        self, interior: GraphInterior | PhysicalInterior, ports: ModelPorts, end: SignalSource,
    ) -> _Signal:
        def shape_of(shape):
            return tuple(shape) if shape is not None and all(type(size) is int for size in shape) else None

        if end.owner == "inputs":
            port = next((port for port in ports.inputs if port.id == end.port), None)
            return _Signal(shape_of(port.type.shape), units=port.type.unit) if port else _Signal()

        component = interior.components.get(end.component)
        if component is None:
            return _Signal()
        if component.kind == "composite":
            output = next((port for port in component.ports.outputs if port.id == end.port), None)
            if output is None:
                return _Signal()
            source = next(
                (wire.source for wire in component.interior.wires.values()
                 if wire.target.owner == "outputs" and wire.target.port == end.port),
                None,
            )
            declared = shape_of(output.type.shape)
            upstream = (
                self.source_signal(component.interior, component.ports, source)
                if source is not None else _Signal()
            )
            return _Signal(
                upstream.shape if declared is None else declared,
                upstream.dtype, output.type.unit or upstream.units,
            )
        definition = self.components.get(component.type)
        types = definition.port_types if definition is not None else None
        port_type = types.outputs.get(end.port) if types is not None else None
        shape = shape_of(port_type.shape) if port_type is not None else None
        units = None
        if component.type == "feedbax.GRU" and end.port in {"hidden", "output"}:
            size = component.fields.get("hidden_size")
            if size is None:
                size = definition.params.model.model_fields["hidden_size"].default
            shape = shape_of((size,))
            units = "1"
        if isinstance(interior, PhysicalInterior) and interior.physical_domain == "planar_multibody":
            if is_acausal_type(component.type, "SensorOutput"):
                quantity = component.fields.get("quantity", "position")
                if quantity == "position":
                    return _Signal((2,), units="m")
                if quantity == "velocity":
                    return _Signal((2,), units="m/s")
        return _Signal(shape, component.fields.get("dtype", "float32"), units)

    def has_reset_output(self, node: ComponentSpec, child, port: str) -> bool:
        if isinstance(child, AcausalGraphSpec):
            return self.state_only_output(node, child, port)
        if isinstance(child, GraphSpec):
            source = child.output_bindings.get(port)
            return source is not None and self.has_reset_output(
                child.nodes[source[0]], (child.subgraphs or {}).get(source[0]), source[1],
            )
        definition = self.components.get(node.type)
        return (
            definition is not None
            and port in definition.compiler.reset_outputs
            and node.params.get("stateful", True) is True
        )

    def delay_reset_source(
        self, reference: NameValue, path: tuple[str, ...], interior: GraphInterior, ports: ModelPorts,
        signal: _Signal, nodes: Mapping[str, ComponentSpec], subgraphs: Mapping[str, GraphSpec | AcausalGraphSpec],
    ) -> ComponentPortEnd:
        address = ModelAddress.from_text(reference.reference)
        if address.document != self.document or address.composites != path:
            raise ValueError("delay Initial must read a reset output of the same graph")
        element = address.element
        if (
            not isinstance(element, PortElement)
            or element.end.owner != "component"
            or element.end.direction != "output"
        ):
            raise ValueError("delay Initial must read an output port at reset")
        end = ComponentPortEnd(component=element.end.component, port=element.end.port, owner="component")
        component = nodes.get(end.component)
        if component is None:
            raise ValueError("delay Initial output owner no longer exists")
        if not self.has_reset_output(component, subgraphs.get(end.component), end.port):
            raise ValueError("delay Initial refers to an output without a qualified numeric reset value")
        if self.source_signal(interior, ports, end) != signal:
            raise ValueError("delay Initial reset output type differs from the delayed signal")
        return end

    def physical(
        self, interior: PhysicalInterior, ports: ModelPorts, path: tuple[str, ...], *, substeps: int = 1,
    ) -> AcausalGraphSpec:
        mechanics = interior.kind == "mechanics"
        if mechanics:
            if ports.physical:
                raise self.problem(path, "port", "Mechanics conserving interfaces require coupled systems",
                                   end={"owner": "boundary", "port": ports.physical[0].id})
            if interior.physical_domain != "planar_multibody":
                raise self.problem(path, "graph", "This Mechanics interior requires an authored spatial asset")
        elif substeps != 1:
            raise self.problem(path, "graph", "The scalar Equations solver supports one step per model step")
        nodes: dict[str, ComponentSpec] = {}
        children: dict[str, AcausalGraphSpec] = {}
        for name, component in interior.components.items():
            self.remember(path, ("nodes", name), {"kind": "component", "id": name})
            if component.kind == "composite":
                if (
                    mechanics or not isinstance(component.interior, PhysicalInterior)
                    or component.interior.physical_domain != interior.physical_domain
                    or component.substeps != 1
                    or component.ports.inputs or component.ports.outputs
                ):
                    raise self.problem(path, "component", "nested equations require the same physical domain and conserving ports", id=name)
                children[name] = self.physical(component.interior, component.ports, (*path, name))
                nodes[name] = ComponentSpec(type="feedbax.AcausalSystem")
            else:
                definition = self.components.get(component.type)
                domains = {"feedbax.domain.acausal", "feedbax.domain.mechanics"} if mechanics else {"feedbax.domain.acausal"}
                if definition is None or definition.domain not in domains:
                    raise self.problem(path, "component", "component is unavailable in this physical interior", id=name)
                self.plugins.add(definition.owner)
                if is_acausal_type(component.type, "BoundaryPort"):
                    raise self.problem(path, "component", "declare conserving interfaces on the composite port list", id=name)
                fields = dict(component.fields)
                nodes[name] = ComponentSpec(type=component.type, revision=component.revision, params=fields)
            if component.attachments.sources or component.attachments.sinks:
                raise self.problem(path, "component", "equations port attachments require signal lowering", id=name)

        boundary = {}
        for index, port in enumerate(ports.physical):
            name = _generated("physical", port.id)
            end = {"owner": "boundary", "port": port.id}
            if port.type != interior.physical_domain or name in nodes:
                raise self.problem(path, "port", "conserving interface differs from the equations domain", end=end)
            nodes[name] = ComponentSpec(type="feedbax.BoundaryPort", params={
                "port_name": port.id, "order": index,
            })
            self.remember(path, ("nodes", name), {"kind": "port", "end": end})
            boundary[port.id] = name

        def endpoint(end: PhysicalEnd, key: str) -> tuple[str, str]:
            if end.owner == "boundary":
                if end.port not in boundary:
                    raise self.problem(path, "physical_connection", "unknown conserving boundary port", id=key)
                return boundary[end.port], "flange"
            return end.component, end.port

        connections = []
        for key, connection in sorted(interior.connections.items()):
            self.remember(path, ("connections", str(len(connections))), {"kind": "physical_connection", "id": key})
            try:
                connections.append(AcausalConnectionSpec.model_validate({
                    "a": endpoint(connection.a, key), "b": endpoint(connection.b, key),
                }))
            except ModelCompilationProblem:
                raise
            except ValueError as error:
                raise self.problem(path, "physical_connection", str(error), id=key) from error

        if interior.attachments.sources or interior.attachments.sinks:
            raise self.problem(path, "graph", "equations boundary attachments require signal lowering")
        inputs = {port.id: index for index, port in enumerate(ports.inputs)}
        outputs = {port.id: index for index, port in enumerate(ports.outputs)}
        adapters: set[str] = set()
        routed_ports: set[tuple[str, str]] = set()
        for key, wire in sorted(interior.wires.items()):
            source, target = wire.source, wire.target
            if source.owner == "inputs" and target.owner == "component" and target.port == "u":
                node_id, port_id, kind, order = target.component, source.port, "ActuationInput", inputs
            elif target.owner == "outputs" and source.owner == "component" and source.port == "y":
                node_id, port_id, kind, order = source.component, target.port, "SensorOutput", outputs
            else:
                raise self.problem(path, "wire", "equations signals connect boundary ports to solver adapters", id=key)
            node = nodes.get(node_id)
            if (
                wire.attachments or node is None or not is_acausal_type(node.type, kind)
                or port_id not in order or node_id in adapters or (kind, port_id) in routed_ports
            ):
                raise self.problem(path, "wire", "invalid or decorated equations signal adapter route", id=key)
            if (
                mechanics and kind == "SensorOutput"
                and node.params.get("quantity", "position") not in PLANAR_STATE_ONLY_QUANTITIES
            ):
                raise self.problem(path, "wire", "This sensor requires within-step Mechanics evaluation", id=key)
            adapters.add(node_id)
            routed_ports.add((kind, port_id))
            nodes[node_id] = node.model_copy(update={
                "params": node.params | {"port_name": port_id, "order": order[port_id]},
            })
        for name, node in nodes.items():
            if any(is_acausal_type(node.type, kind) for kind in ("ActuationInput", "SensorOutput")) and name not in adapters:
                raise self.problem(path, "component", "solver adapter requires a boundary wire", id=name)
        if mechanics:
            self.planar_frames(nodes, connections, path)
        dynamics = None
        if mechanics and self.model.world is not None:
            x, y, z = self.model.world.gravity
            if z:
                raise self.problem((), "field", "planar gravity must lie in the model plane",
                                   parent={"kind": "graph"}, path=["world", "gravity"])
            dynamics = PlanarDynamicsSpec(gravity=(x, y))
        return AcausalGraphSpec(
            physical_domain=interior.physical_domain, nodes=nodes, connections=connections,
            subgraphs=children or None,
            solver=SolverConfigSpec(
                solver_type="euler", dt=self.model.step_seconds,
                planar_substeps=substeps if mechanics else None,
            ),
            planar_dynamics=dynamics,
        )

    def planar_frames(self, nodes, connections, path) -> None:
        """Project existing conserving frame connections to the serial-chain compiler."""
        destinations = {
            "Anchor": {"world": "world_frame", "frame": "frame"},
            "RevoluteJoint": {"parent": "parent_frame", "child": "child_frame"},
        }
        for connection in connections:
            for endpoint, peer in ((connection.a, connection.b), (connection.b, connection.a)):
                name, port = endpoint
                node = nodes.get(name)
                if node is None:
                    continue
                for role, fields in destinations.items():
                    if not is_acausal_type(node.type, role) or port not in fields:
                        continue
                    field_name = fields[port]
                    reference = ".".join(peer)
                    if node.params.get(field_name, reference) != reference:
                        raise self.problem(path, "component", f"{field_name} disagrees with the frame connection", id=name)
                    nodes[name] = node.model_copy(update={"params": node.params | {field_name: reference}})

    def graph(
        self, interior: GraphInterior, ports: ModelPorts, path: tuple[str, ...],
        *, available: frozenset[str] = frozenset(),
    ) -> GraphSpec:
        nodes, subgraphs = {}, {}
        input_bindings, output_bindings = {}, {}
        wires = []
        input_sources = {}
        sampled_outputs = {}
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
                scope = (*path, key)
                if not isinstance(component.interior, PhysicalInterior) and component.substeps != 1:
                    raise self.problem(path, "component", "the composite solver supports one step per model step", id=key)
                if component.attachments.sinks:
                    raise self.problem(path, "component", "sink probes require observation lowering", id=key)
                if isinstance(component.interior, PhysicalInterior):
                    from feedbax.compiler.acausal_compiler import compile_acausal_authoring_report

                    child = self.physical(component.interior, component.ports, scope, substeps=component.substeps)
                    report = compile_acausal_authoring_report(
                        child, node_path=list(scope), component_registry=self.components,
                    )
                    if issue := next((item for item in report.diagnostics if item.severity == "error"), None):
                        detail = issue.message
                        if issue.counts:
                            detail += " (" + ", ".join(f"{name}={value}" for name, value in issue.counts.items()) + ")"
                        raise self.problem(scope, "graph", detail)
                    interface = derive_acausal_interface(child)
                    for direction, declared, actual in (
                        ("inputs", component.ports.inputs, interface.input_ports),
                        ("outputs", component.ports.outputs, interface.output_ports),
                    ):
                        for port in declared:
                            if port.id not in actual:
                                raise self.problem(scope, "port", "equations port requires a solver adapter route",
                                                   end={"owner": direction, "port": port.id})
                    type_id, input_ports, output_ports = "feedbax.AcausalSystem", list(interface.input_ports), list(interface.output_ports)
                else:
                    child = self.graph(component.interior, component.ports, scope)
                    type_id, input_ports, output_ports = "feedbax.Subgraph", child.input_ports, child.output_ports
                subgraphs[key] = child
                nodes[key] = ComponentSpec(
                    type=type_id, params={}, input_ports=input_ports, output_ports=output_ports,
                )
            else:
                definition = self.components.get(component.type)
                if definition is None or definition.is_composite:
                    raise self.problem(path, "component", f"unknown leaf type {component.type!r}", id=key)
                self.plugins.add(definition.owner)
                params = dict(component.fields)
                if "dt" in params or "control_dt" in params:
                    raise self.problem(path, "component", "component periods are supplied by the model root", id=key)
                for period in ("dt", "control_dt"):
                    if period in definition.params.model.model_fields:
                        params[period] = self.model.step_seconds
                if component.type == "feedbax.VanillaRNN":
                    tau = params.get("tau", definition.params.model.model_fields["tau"].default)
                    try:
                        params["tau"] = model_tau_seconds(tau, self.model.step_seconds)
                    except ValueError as error:
                        raise self.problem(
                            path, "field", str(error), parent={"kind": "component", "id": key}, path=("fields", "tau"),
                        ) from error
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

        def state_sample(source, origin):
            if source not in sampled_outputs:
                node, port = source
                relay = _generated("state_sample", node, port)
                add_generated(relay, "feedbax.Input", {}, origin=origin)
                emit(source, (relay, "input"), _generated("state_sample_wire", node, port), origin,
                     reset={"kind": "state_output", "model_state_sample": True,
                            "source_node": node, "source_port": port})
                sampled_outputs[source] = relay, "output"
            return sampled_outputs[source]

        def transform(source, chain, key, at, *, finish=None, signal: _Signal | None = None):
            pending_reset = None
            for attachment in chain:
                origin = {"kind": "attachment", "parent": at, "id": attachment.id}
                if isinstance(attachment, ProbeAttachment):
                    raise self.problem(path, "attachment", "probe observation is unavailable in this milestone", parent=at, id=attachment.id)
                if isinstance(attachment, DelayAttachment):
                    length = attachment.delay.value
                    if attachment.delay.unit == "s":
                        length /= self.model.step_seconds
                    if not isfinite(length):
                        raise self.problem(path, "attachment", "delay step count is not finite", parent=at, id=attachment.id)
                    steps = round(length)
                    if steps < 1 or attachment.delay.unit == "st" and steps != length:
                        raise self.problem(path, "attachment", "delay must resolve to at least one whole step", parent=at, id=attachment.id)
                    if signal is None or signal.shape is None:
                        raise self.problem(path, "attachment", "delay has no resolved reset shape", parent=at, id=attachment.id)
                    if signal.dtype != "float32":
                        raise self.problem(path, "attachment", "delay requires a float32 reset", parent=at, id=attachment.id)
                    initial = attachment.initial
                    reset_output = None
                    if isinstance(initial, dict):
                        try:
                            reference = NameValue.model_validate(initial)
                            reset_output = self.delay_reset_source(
                                reference, path, interior, ports, signal, nodes, subgraphs,
                            )
                        except ValueError as error:
                            raise self.problem(path, "attachment", str(error), parent=at, id=attachment.id) from error
                        unit = UNITS.get(signal.units or "")
                        if unit is None or any(size <= 0 for size in signal.shape):
                            raise self.problem(
                                path, "attachment", "delay Initial requires a resolved numeric value type",
                                parent=at, id=attachment.id,
                            )
                        self.links.append(ResolvedModelLink(
                            address=self.address(path, "field", parent=origin, path=("initial",)),
                            value=reference,
                            type=ValueType(base=ValueBase.REAL, shape=signal.shape, dimensions=unit.dimensions),
                        ))
                        initial = 0
                    if type(initial) not in (int, float) or not -3.4028234663852886e38 <= initial <= 3.4028234663852886e38:
                        raise self.problem(path, "attachment", "delay Initial requires a qualified numeric value", parent=at, id=attachment.id)
                    reset = ModelDelayCarry(
                        shape=signal.shape, dtype=signal.dtype, units=signal.units,
                        steps=steps, initial=initial, reset_output=reset_output,
                        model_delay_id=attachment.id, model_wire_id=key,
                    ).model_dump(mode="json")
                    relay = _generated("delay", key, attachment.id)
                    add_generated(relay, "feedbax.Input", {}, origin=origin)
                    emit(source, (relay, "input"), _generated("delay_input", key, attachment.id), origin, reset=pending_reset)
                    source = (relay, "output")
                    pending_reset = reset
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

        for port in ports.inputs:
            external = port.id if path else port.name
            relay = _generated("input", port.id)
            origin = {"kind": "port", "end": {"owner": "inputs", "port": port.id}}
            default = not path and external not in available and port.default is not None
            if default:
                value = port.default.value
                params = {"value": value}
                if isinstance(value, list):
                    try:
                        dtype = np.asarray(value).dtype.kind
                    except ValueError as error:
                        raise self.problem(path, "port", "A default requires a rectangular numeric array",
                                           end={"owner": "inputs", "port": port.id}) from error
                    if dtype not in {"b", "i", "u", "f"}:
                        raise self.problem(path, "port", "A default requires a numeric sample",
                                           end={"owner": "inputs", "port": port.id})
                    params["dtype"] = {"b": "bool", "i": "int32", "u": "int32", "f": "float32"}[dtype]
                add_generated(relay, "feedbax.Constant", params, origin=origin, inputs=())
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
                origin = {"kind": "port", "end": {"owner": "component", "component": key,
                                                   "port": port, "direction": "output"}}
                source = (key, port)
                if self.state_only_output(nodes[key], subgraphs.get(key), port):
                    source = state_sample(source, origin)
                source, _ = transform(
                    source, chain, _generated("port", key, port), origin,
                )
                component_sources[key, port] = source

        for key, wire in sorted(interior.wires.items()):
            a, b = wire.source, wire.target
            if a.owner == "inputs":
                source = input_sources.get(a.port)
            else:
                source = component_sources.get((a.component, a.port), (a.component, a.port))
                if a.component not in nodes or a.port not in nodes[a.component].output_ports:
                    source = None
            if source is None:
                raise self.problem(path, "wire", "source port does not exist", id=key)
            at = {"kind": "wire", "id": key}
            if (
                a.owner == "component" and (a.component, a.port) not in component_sources
                and (b.owner != "outputs" or wire.attachments)
                and self.state_only_output(nodes[a.component], subgraphs.get(a.component), a.port)
            ):
                source = state_sample(source, at)
            signal = (
                self.source_signal(interior, ports, a)
                if any(isinstance(item, DelayAttachment) for item in wire.attachments) else None
            )
            if b.owner == "outputs":
                external = output_names.get(b.port)
                if external is None:
                    raise self.problem(path, "wire", "boundary output port does not exist", id=key)
                if external in output_bindings:
                    raise self.problem(path, "wire", "boundary output already has a driver", id=key)
                output, reset = transform(source, wire.attachments, key, at, signal=signal)
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
                transform(source, wire.attachments, key, at, finish=(b.component, b.port), signal=signal)
        return GraphSpec(
            step_seconds=self.model.step_seconds if not path else None,
            nodes=nodes, wires=wires, input_ports=list(input_bindings), output_ports=list(output_names.values()),
            input_bindings=input_bindings, output_bindings=output_bindings, subgraphs=subgraphs or None,
        )

    def state_only_output(self, node: ComponentSpec, child, port: str) -> bool:
        if isinstance(child, AcausalGraphSpec):
            return child.physical_domain == "planar_multibody" and any(
                is_acausal_type(sensor.type, "SensorOutput")
                and sensor.params.get("port_name") == port
                and sensor.params.get("quantity", "position") in PLANAR_STATE_ONLY_QUANTITIES
                for sensor in child.nodes.values()
            )
        if isinstance(child, GraphSpec):
            if port not in child.output_bindings:
                return False
            name, output = child.output_bindings[port]
            return self.state_only_output(child.nodes[name], (child.subgraphs or {}).get(name), output)
        definition = self.components.get(node.type)
        return definition is not None and port in definition.compiler.state_only_outputs

    def reject_within_step_paths(self, graph: GraphSpec) -> None:
        from feedbax.contracts.model_occurrence import graph_leaf_specs

        def interior(scope):
            current = graph
            for name in scope:
                current = current.subgraphs[name]
            return current

        leaves = dict(graph_leaf_specs(graph))
        mechanics, instant = set(), set()
        for scope, node in leaves.items():
            child = (interior(scope[:-1]).subgraphs or {}).get(scope[-1])
            declaration = self.components.get(node.type)
            if (
                isinstance(child, AcausalGraphSpec) and child.physical_domain == "planar_multibody"
                or declaration is not None and declaration.compiler.state_only_outputs
            ):
                mechanics.add(scope)
            elif declaration is not None and declaration.compiler.instantaneous:
                instant.add(scope)
        if not mechanics:
            return
        try:
            schedule = resolve_signal_schedule(graph)
        except SignalScheduleError as error:
            address = self.origins.get(json_pointer_parts(error.authored_path))
            if address is not None:
                raise ModelCompilationProblem(address, str(error)) from error
            raise
        outgoing = {}
        samples = []
        for route in schedule.routes:
            if route.temporality == "instant":
                outgoing.setdefault(route.source.node, []).append(route)
            else:
                wire = interior(route.wire.owner).wires[route.wire.index]
                if (wire.recurrent_initializer or {}).get("model_state_sample"):
                    samples.append(route)
        for sample in samples:
            pending, seen = [sample.target.node], set()
            while pending:
                scope = pending.pop()
                if scope in seen:
                    continue
                seen.add(scope)
                for route in outgoing.get(scope, ()):
                    target = route.target.node
                    if target in mechanics:
                        address = self.origins[json_pointer_parts(route.wire.authored_path)]
                        raise ModelCompilationProblem(
                            address, "This path between mechanisms requires within-step evaluation. "
                            "A Hold or a delay makes it a sampled signal.",
                        )
                    if target in instant:
                        pending.append(target)

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
    links: tuple[ResolvedModelLink, ...]
    plugins: tuple[str, ...]

    def interface(
        self, document: ProjectName, model: ModelDocument, components: ComponentRegistry,
        *, input_prototypes: Mapping[tuple[str, str], object] | None = None,
    ) -> ResolvedModelInterface:
        """Project resolved sample types for task and objective pairing."""
        prototypes = (
            prototypes_from_task_bindings(self.task) if self.task is not None else {}
        ) | dict(input_prototypes or {})
        resolution = resolve_graph_parameters(self.graph, prototypes, components)
        return ResolvedModelInterface(
            model.step_seconds,
            _scoped_ports(document, model, resolution, "inputs"),
            _scoped_ports(document, model, resolution, "outputs"),
            self.links,
            self.plugins,
            resolve_model_reset(document, resolution, components),
        )

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


@dataclass(frozen=True)
class ResolvedModelInterface:
    period_seconds: float
    inputs: tuple[ResolvedModelInput, ...]
    outputs: tuple[ScopedValue, ...]
    links: tuple[ResolvedModelLink, ...]
    plugins: tuple[str, ...]
    reset: ModelResetPlan = ModelResetPlan()


class ResolvedModelInput(ScopedValue):
    required: bool

    @property
    def address(self) -> ModelAddress:
        return ModelAddress.from_text(self.id)


@dataclass(frozen=True)
class ResolvedModelLink:
    address: ModelAddress
    value: NameValue
    type: ValueType


def _scoped_ports(
    document: ProjectName, model: ModelDocument, resolved: GraphParameterResolution, direction: str,
) -> tuple[ScopedValue, ...]:
    results = []
    for port in getattr(model.ports, direction):
        address = ModelAddress.model_validate({
            "document": document, "composites": [],
            "element": {"kind": "port", "end": {"owner": direction, "port": port.id}},
        })
        if direction == "inputs":
            target = resolved.graph.input_bindings.get(port.name)
            proto = resolved.node_inputs.get(target)
            if proto is None and port.default is not None:
                proto = resolved.node_outputs.get((_generated("input", port.id), "output"))
        else:
            proto = resolved.graph_outputs.get(port.name)
        shape = getattr(proto, "shape", None)
        dtype = getattr(getattr(proto, "dtype", None), "kind", None)
        if shape is None or dtype not in {"b", "i", "u", "f"}:
            raise ModelCompilationProblem(address, f"Cannot resolve the numeric sample type of {port.name!r}")
        unit = UNITS.get(port.type.unit or "")
        if unit is None:
            raise ModelCompilationProblem(address, f"Unknown expression unit {port.type.unit!r}")
        base = ValueBase.BOOLEAN if dtype == "b" else ValueBase.REAL if dtype == "f" else ValueBase.INTEGER
        if any(size <= 0 for size in shape) or (
            port.type.shape is not None and (
                len(shape) != len(port.type.shape) or any(
                    type(declared) is int and actual != declared
                    for actual, declared in zip(shape, port.type.shape)
                )
            )
        ):
            raise ModelCompilationProblem(address, f"Resolved shape of {port.name!r} conflicts with its port")
        value = ScopedValue(
            id=address.to_text(), name=port.name, level=Level.STEP,
            type=ValueType(base=base, shape=tuple(shape), dimensions=unit.dimensions),
        )
        results.append(
            ResolvedModelInput(**value.model_dump(), required=port.default is None)
            if isinstance(port, SignalInput) else value
        )
    return tuple(results)


def lower_model_document(
    document: ProjectName, model: ModelDocument, components: ComponentRegistry,
    task: StudioTaskBindingSpec | None = None,
    *, supplied_inputs: frozenset[str] = frozenset(),
) -> LoweredModelDocument:
    """Project one authored model through the graph compiler's registered declarations."""
    lower = _Lower(document, model, components)
    names = frozenset(item.id for item in task.exposed_data if item.bindable) if task else frozenset()
    graph = lower.graph(model.interior, model.ports, (), available=names | supplied_inputs)
    lower.reject_within_step_paths(graph)
    binding = lower.root_inputs(task, graph) if task is not None else None
    return LoweredModelDocument(graph, binding, lower.origins, tuple(lower.links), tuple(sorted(lower.plugins)))


def lower_planar_scene_interior(
    document: ProjectName, model: ModelDocument, path: tuple[str, ...], components: ComponentRegistry,
) -> AcausalGraphSpec:
    """Lower the selected mechanics occurrence without resolving the surrounding model."""
    interior = model.interior
    for index, name in enumerate(path):
        component = interior.components.get(name)
        if component is None or component.kind != "composite":
            break
        child = component.interior
        if isinstance(child, PhysicalInterior):
            if child.physical_domain == "planar_multibody" and index == len(path) - 1:
                return _Lower(document, model, components).physical(
                    child, component.ports, path, substeps=component.substeps,
                )
            break
        interior = child
    raise ValueError(f"{list(path)!r} must identify a planar multibody interior.")
