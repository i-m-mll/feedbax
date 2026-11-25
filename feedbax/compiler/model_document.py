"""Lower the stored model tree into the shared graph compiler's input."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from hashlib import sha256
from math import isfinite
from collections.abc import Iterator, Mapping

import jax.numpy as jnp
import numpy as np
from jax import ShapeDtypeStruct

from feedbax.builtins.mechanics.linear_state_space import LinearStateSpace, linear_state_space_output_prototype
from feedbax.component_registry import ComponentRegistry
from feedbax.compiler.resolution import GraphParameterResolution, resolve_graph_parameters
from feedbax.compiler.model_signals import (
    ModelSignalTable, ResolvedSignalLeaf, boundary_signal_port, resolve_model_signals, task_data_value_type,
)
from feedbax.compiler.prototypes import prototypes_from_task_bindings
from feedbax.compiler.model_edit import model_graph_problems
from feedbax.compiler.model_ports import BOUNDARY_PORT_PLACEMENT_REASON, ModelRegistryPorts, WIRED_VECTOR_INPUT_TYPES
from feedbax.contracts.acausal import AcausalConnectionSpec, AcausalGraphSpec, PlanarDynamicsSpec, SolverConfigSpec
from feedbax.contracts.acausal_interface import (
    PLANAR_STATE_ONLY_QUANTITIES, derive_acausal_interface, is_acausal_type,
)
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.expression_units import UNITS
from feedbax.contracts.expression_values import Level, LiteralValue, NameValue, ScopedValue, UnresolvedValue, ValueBase, ValueType
from feedbax.contracts.graph import (
    ComponentSpec, DerivedDimensionRuleSpec, GraphSpec, StudioTaskBinding, StudioTaskBindingSpec, WireSpec,
)
from feedbax.contracts.model_address import (
    AttachmentElement, ComponentPortAddress, ComponentPortEnd, FieldElement, InputBoundaryEnd, ModelAddress,
    PhysicalEnd, PortElement, SignalSource, SignalTarget,
)
from feedbax.contracts.model_delay import ModelDelayCarry
from feedbax.contracts.model_edit import ModelGraphProblem
from feedbax.contracts.model_document import (
    DelayAttachment, DelayInitialTask, DelayInitialValue, DelayInitialWire, GraphInterior, HoldAttachment, InterventionAttachment,
    ModelDocument, ModelPorts,
    PhysicalInterior, ProbeAttachment, SignalInput, SignalType,
)
from feedbax.contracts.project import json_pointer_parts
from feedbax.contracts.time_units import TimeValue, length_steps, time_in_seconds
from feedbax.mechanics.mechanics import Mechanics
from feedbax.runtime.task_data_roles import task_data_role
from feedbax.compiler.model_reset import ModelResetPlan, resolve_model_reset
from feedbax.runtime._signal_schedule import SignalScheduleError, resolve_signal_schedule
from feedbax.contracts.graph import task_binding_id


class ModelCompilationProblem(ValueError):
    def __init__(
        self, address: ModelAddress, detail: str, *, related: tuple[ModelGraphProblem, ...] = (),
    ):
        super().__init__(detail)
        self.address = address
        self.problems = (ModelGraphProblem(address=address, reason=detail), *related)


def _signal_dtype(base: ValueBase | str | None) -> np.dtype:
    if base in ("bool", ValueBase.BOOLEAN):
        return np.dtype(np.bool_)
    if base in ("int", "int32", ValueBase.INTEGER):
        return np.dtype(np.int32)
    return jnp.result_type(float)


def _model_float_dtype(dtype: str | None = None) -> str:
    """Resolve the authored Model's standard real precision for this process."""
    return jnp.result_type(float).name if dtype in (None, "float32", "float64") else dtype


def declared_model_input_prototypes(model: ModelDocument) -> dict[str, ShapeDtypeStruct]:
    prototypes = {}
    for port in model.ports.inputs:
        shape = port.type.shape
        if shape is None or any(type(size) is not int or size <= 0 for size in shape):
            continue
        prototypes[port.name] = ShapeDtypeStruct(shape, _signal_dtype(port.type.base))
    return prototypes


def model_input_prototypes(
    graph: GraphSpec, model: ModelDocument, task_inputs: Mapping[str, ValueType],
) -> dict[tuple[str, str], ShapeDtypeStruct]:
    supplied = declared_model_input_prototypes(model)
    supplied.update({port: ShapeDtypeStruct(value.shape, _signal_dtype(value.base)) for port, value in task_inputs.items()})
    return {target: supplied[port] for port, target in graph.input_bindings.items() if port in supplied}


def inspect_model_signals(
    model: ModelDocument, components: ComponentRegistry, task_inputs: Mapping[str, ValueType] | None = None,
) -> ModelSignalTable:
    """Resolve authored signal facts for Studio reads even when the whole Model cannot run."""
    supplied = task_inputs or {}
    try:
        lowered = lower_model_document(model, components, supplied_inputs=frozenset(supplied))
    except ValueError as error:
        table = resolve_model_signals(model, None, components, task_inputs=supplied)
        diagnostics = error.problems if isinstance(error, ModelCompilationProblem) else (
            ModelGraphProblem(address=ModelAddress.at_stored_path(()), reason=str(error)),
        )
        problems = {(problem.address.to_text(), problem.reason): problem for problem in (*diagnostics, *table.problems)}
        return replace(table, problems=tuple(problems.values()))
    prototypes = model_input_prototypes(lowered.graph, model, supplied)
    resolution = resolve_graph_parameters(lowered.graph, prototypes, components, allow_incomplete=True)
    return resolve_model_signals(model, resolution, components, task_inputs=supplied)


def model_tau_seconds(tau: object, period: float) -> float:
    time = LiteralValue.model_validate(tau) if isinstance(tau, dict) else LiteralValue(value=tau, unit="st")
    if time.unit not in {"st", "s"} or type(time.value) not in {float, int} or time.value <= 0:
        raise ValueError("Tau requires a positive value in seconds or steps")
    value = time_in_seconds(TimeValue(value=time.value, unit=time.unit), period)
    if value <= 0:
        raise ValueError("Tau does not resolve to a positive finite duration")
    return value


def _generated(role: str, *parts: str) -> str:
    return "_model_" + role + "_" + sha256(canonical_json_v2_bytes(list(parts))).hexdigest()


def _named_physical_elements(
    interior: PhysicalInterior, ids: tuple[str, ...] = (), names: tuple[str, ...] = (),
) -> Iterator[tuple[str, str]]:
    for key, component in interior.components.items():
        path, title = (*ids, key), (*names, component.name)
        yield ".".join(path), ".".join(title)
        if component.kind == "composite" and isinstance(component.interior, PhysicalInterior):
            yield from _named_physical_elements(component.interior, path, title)


@dataclass(frozen=True)
class _Signal:
    shape: tuple[int, ...] | None = None
    dtype: str = field(default_factory=_model_float_dtype)
    units: str | None = None


@dataclass
class _Lower:
    model: ModelDocument
    components: ComponentRegistry
    origins: dict[tuple[str, ...], ModelAddress] = field(default_factory=dict)
    holds: set[tuple[str, ...]] = field(default_factory=set)
    plugins: set[str] = field(default_factory=set)

    def address(self, scope, kind, **fields) -> ModelAddress:
        return ModelAddress.model_validate({
            "composites": scope, "element": {"kind": kind, **fields},
        })

    def problem(self, scope, kind, detail, **fields) -> ModelCompilationProblem:
        return ModelCompilationProblem(self.address(scope, kind, **fields), detail)

    @property
    def period(self) -> float:
        return self.numeric_field(self.model.step_seconds, (), {"kind": "graph"}, ("step_seconds",))

    def numeric_field[T](self, value: T | UnresolvedValue, scope, owner, path: tuple[str, ...]) -> T:
        if isinstance(value, UnresolvedValue) or isinstance(value, dict) and value.get("kind") == "unresolved":
            raise self.problem(scope, "field", "This Model field contains unresolved text", parent=owner, path=path)
        return value

    def parameters(self, component, scope, name: str):
        owner = {"kind": "component", "id": name}
        return {
            field: self.numeric_field(value, scope, owner, ("fields", field))
            for field, value in component.fields.items()
        }

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
            return _Signal(shape_of(port.type.shape), _signal_dtype(port.type.base).name, port.type.unit) if port else _Signal()

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
        if end.port == "state" and definition is not None and issubclass(definition.params.runtime_type, LinearStateSpace):
            prototype = linear_state_space_output_prototype(definition.params.default_params | component.fields, {})["state"]
            return _Signal(prototype.shape, prototype.dtype.name, "1")
        if isinstance(interior, PhysicalInterior) and interior.physical_domain == "planar_multibody":
            if is_acausal_type(component.type, "SensorOutput"):
                quantity = component.fields.get("quantity", "position")
                if quantity == "position":
                    return _Signal((2,), units="m")
                if quantity == "velocity":
                    return _Signal((2,), units="m/s")
        return _Signal(shape, _model_float_dtype(component.fields.get("dtype")), units)

    def destination_shape(
        self, interior: GraphInterior, ports: ModelPorts, end: SignalTarget,
    ) -> tuple[int, ...] | None:
        """Read the input contract when the wire's source does not state a reset shape."""
        if end.owner == "outputs":
            declared = next((port.type for port in ports.outputs if port.id == end.port), None)
        elif (component := interior.components.get(end.component)) is None:
            return None
        elif component.kind == "composite":
            declared = next((port.type for port in component.ports.inputs if port.id == end.port), None)
        else:
            port = ModelRegistryPorts(self.components).port(component.model_dump(mode="json"), "input", end.port)
            declared = port.type if port is not None else None
        shape = declared.shape if isinstance(declared, SignalType) else None
        return tuple(shape) if shape is not None and all(type(size) is int for size in shape) else None

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
        self, end: SignalSource, interior: GraphInterior, ports: ModelPorts, signal: _Signal,
        nodes: Mapping[str, ComponentSpec], subgraphs: Mapping[str, GraphSpec | AcausalGraphSpec],
        input_sources: Mapping[str, tuple[str, str]],
    ) -> tuple[ComponentPortEnd, bool]:
        supplied = self.source_signal(interior, ports, end)
        if supplied.units is not None and signal.units is not None and supplied.units != signal.units:
            raise ValueError("Delay Initial units differ from the delayed signal.")
        if isinstance(end, InputBoundaryEnd):
            if end.port not in input_sources:
                raise ValueError("Delay Initial input no longer exists.")
            component, port = input_sources[end.port]
            return ComponentPortEnd(owner="component", component=component, port=port), False
        component = nodes.get(end.component)
        if component is None or end.port not in component.output_ports:
            raise ValueError("Delay Initial output no longer exists.")
        return end, self.has_reset_output(component, subgraphs.get(end.component), end.port)

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
                    raise self.problem(path, "component", BOUNDARY_PORT_PLACEMENT_REASON, id=name)
                fields = self.parameters(component, path, name)
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
                solver_type="euler", dt=self.period,
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
        derived_dimensions = []
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
                substeps = self.numeric_field(component.substeps, path, {"kind": "component", "id": key}, ("substeps",))
                if not isinstance(component.interior, PhysicalInterior) and substeps != 1:
                    raise self.problem(path, "component", "the composite solver supports one step per model step", id=key)
                if component.attachments.sinks:
                    raise self.problem(path, "component", "sink probes require observation lowering", id=key)
                if isinstance(component.interior, PhysicalInterior):
                    from feedbax.compiler.acausal_compiler import compile_acausal_authoring_report

                    child = self.physical(component.interior, component.ports, scope, substeps=substeps)
                    report = compile_acausal_authoring_report(
                        child, node_path=list(scope), component_registry=self.components,
                        element_names=dict(_named_physical_elements(component.interior)),
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
                assert definition is not None and not definition.is_composite
                self.plugins.add(definition.owner)
                params = self.parameters(component, path, key)
                if "dt" in params or "control_dt" in params:
                    raise self.problem(path, "component", "component periods are supplied by the model root", id=key)
                for period in ("dt", "control_dt"):
                    if period in definition.params.model.model_fields:
                        params[period] = self.period
                for field_name in ("dtype", "input_dtype"):
                    if declaration := definition.params.model.model_fields.get(field_name):
                        dtype = params.get(field_name, declaration.default)
                        if dtype in ("float32", "float64") and dtype != (resolved_dtype := _model_float_dtype(dtype)):
                            params[field_name] = resolved_dtype
                if component.type == "feedbax.VanillaRNN":
                    tau = params.get("tau", definition.params.model.model_fields["tau"].default)
                    try:
                        params["tau"] = model_tau_seconds(tau, self.period)
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

        def intervention_operand(value, field, key, origin):
            if isinstance(value, dict) and value.get("kind") == "name":
                reference = NameValue.model_validate(value).reference
                try:
                    address = ModelAddress.from_text(reference)
                except ValueError as error:
                    raise ValueError("An intervention Link needs a paired Task variable or a local Model output.") from error
                if address.composites != path or not isinstance(address.element, PortElement):
                    raise ValueError("An intervention Link must read an output in the same graph.")
                end = address.element.end
                if isinstance(end, InputBoundaryEnd) and any(port.id == end.port for port in ports.inputs):
                    return _generated("input", end.port), "output"
                if isinstance(end, ComponentPortAddress) and end.direction == "output":
                    node = nodes.get(end.component)
                    if node is not None and end.port in node.output_ports:
                        source = end.component, end.port
                        if self.state_only_output(node, subgraphs.get(end.component), end.port):
                            source = state_sample(source, origin)
                        return source
                raise ValueError("The linked Model output does not exist.")
            try:
                array = np.asarray(value)
            except ValueError as error:
                raise ValueError("Enter a number or a rectangular numeric array.") from error
            if array.dtype.kind not in {"f", "i", "u"} or not np.isfinite(array).all():
                raise ValueError("Enter a finite number or a numeric array.")
            node = _generated("intervention_constant", key, origin["id"], field)
            add_generated(node, "feedbax.Constant", {"value": value, "dtype": "float32"}, origin=origin, inputs=())
            return node, "output"

        def transform(source, chain, key, at, *, finish=None, signal: _Signal | None = None):
            pending_reset = None
            for attachment in chain:
                origin = {"kind": "attachment", "parent": at, "id": attachment.id}
                if isinstance(attachment, ProbeAttachment):
                    raise self.problem(path, "attachment", "probe observation is unavailable in this milestone", parent=at, id=attachment.id)
                if isinstance(attachment, HoldAttachment):
                    relay = _generated("hold", key, attachment.id)
                    add_generated(relay, "feedbax.Input", {}, origin=origin)
                    self.holds.add((*path, relay))
                    emit(source, (relay, "input"), _generated("hold_input", key, attachment.id), origin, reset=pending_reset)
                    source, pending_reset = (relay, "output"), None
                    continue
                if isinstance(attachment, DelayAttachment):
                    delay = self.numeric_field(attachment.delay, path, origin, ("delay",))
                    try:
                        steps = length_steps(delay, self.period)
                    except ValueError as error:
                        raise self.problem(path, "field", str(error), parent=origin, path=("delay",)) from error
                    signal = signal or _Signal()
                    initial, reset_output, reset_evaluate, reset_value = 0, None, None, None
                    match attachment.initial_source:
                        case DelayInitialValue(value=value):
                            initial = self.numeric_field(value, path, origin, ("initial_source", "value"))
                        case DelayInitialWire(source=end) if end is not None:
                            try:
                                endpoint, state_only = self.delay_reset_source(
                                    end, interior, ports, signal, nodes, subgraphs, input_sources,
                                )
                            except ValueError as error:
                                raise self.problem(path, "attachment", str(error), parent=at, id=attachment.id) from error
                            if state_only:
                                reset_output = endpoint
                            else:
                                reset_evaluate = endpoint
                        case DelayInitialTask(variable=variable):
                            reset_value = variable
                        case _:
                            raise self.problem(
                                path, "attachment", "Connect an output or Model input to Delay Initial.",
                                parent=at, id=attachment.id,
                            )
                    limit = float(np.finfo(signal.dtype).max) if signal.dtype in ("float32", "float64") else float("inf")
                    if type(initial) not in (int, float) or not isfinite(initial) or not -limit <= initial <= limit:
                        raise self.problem(path, "attachment", "delay Initial requires a qualified numeric value", parent=at, id=attachment.id)
                    dtype = signal.dtype if signal.dtype in ("float32", "float64") else None
                    reset = ModelDelayCarry(
                        shape=signal.shape if dtype is not None else None,
                        dtype=dtype or _model_float_dtype(), units=signal.units,
                        steps=steps, initial=initial, reset_output=reset_output,
                        reset_evaluate=reset_evaluate, reset_value=reset_value,
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
                    parameters = {
                        name: self.numeric_field(value, path, origin, ("fields", name))
                        for name, value in attachment.fields.items()
                    }
                    operation = attachment.operation.lower()
                    operands = {}
                    input_port = "input"
                    match operation:
                        case "scale" if type(parameters.get("factor")) in (int, float):
                            type_id, params = "feedbax.Scale", {"scale": parameters["factor"], "fixed": True}
                        case "add" if isinstance(parameters.get("value"), list) and all(
                            type(value) in (int, float) and isfinite(value) for value in parameters["value"]
                        ):
                            type_id, params = "feedbax.ConstantInput", {"arrays": parameters["value"], "active": True}
                        case "clamp" if all(type(parameters.get(name)) in (int, float) for name in ("low", "high")):
                            if parameters["low"] > parameters["high"]:
                                raise self.problem(path, "attachment", "clamp bounds are reversed", parent=at, id=attachment.id)
                            type_id, params = "feedbax.Saturation", {"min_val": parameters["low"], "max_val": parameters["high"]}
                        case "add":
                            type_id, params, input_port = "feedbax.Sum", {}, "a"
                            operands = {"value": "b"}
                        case "scale":
                            type_id, params, input_port = "feedbax.Multiply", {}, "a"
                            operands = {"factor": "b"}
                        case "clamp":
                            type_id, params = "feedbax.Clamp", {}
                            operands = {"low": "low", "high": "high"}
                        case _:
                            raise self.problem(path, "attachment", "unrecognized intervention or fields", parent=at, id=attachment.id)
                    segment = _generated("attachment", key, attachment.id)
                    add_generated(segment, type_id, params, origin=origin, inputs=(input_port, *operands.values()))
                    for field, port in operands.items():
                        try:
                            operand = intervention_operand(parameters.get(field), field, key, origin)
                        except ValueError as error:
                            raise self.problem(path, "field", str(error), parent=origin, path=("fields", field)) from error
                        emit(operand, (segment, port), _generated("intervention_operand", key, attachment.id, field), origin)
                    emit(source, (segment, input_port), _generated("segment", key, attachment.id), origin, reset=pending_reset)
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
                    params["dtype"] = {"b": "bool", "i": "int32", "u": "int32", "f": _model_float_dtype()}[dtype]
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
            if signal is not None and signal.shape is None:
                signal = replace(signal, shape=self.destination_shape(interior, ports, b))
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
                    origin = {"kind": "attachment", "parent": at, "id": reset["model_delay_id"]}
                    emit(output, (relay, "input"), key, origin, reset=reset)
                    output = (relay, "output")
                output_bindings[external] = output
            else:
                target = nodes.get(b.component)
                if target is None or b.port not in target.input_ports:
                    raise self.problem(path, "wire", "target port does not exist", id=key)
                if any(w.target_node == b.component and w.target_port == b.port for w in wires):
                    raise self.problem(path, "wire", "target port already has a driver", id=key)
                transform(source, wire.attachments, key, at, finish=(b.component, b.port), signal=signal)
                if b.port == "input" and target.type in WIRED_VECTOR_INPUT_TYPES:
                    if "input_size" in target.params:
                        raise self.problem(
                            path, "field", "The input wire determines this width",
                            parent={"kind": "component", "id": b.component}, path=("fields", "input_size"),
                        )
                    derived_dimensions.append(DerivedDimensionRuleSpec(node=b.component, param="input_size", port=b.port))
        return GraphSpec(
            step_seconds=self.period if not path else None,
            nodes=nodes, wires=wires, input_ports=list(input_bindings), output_ports=list(output_names.values()),
            derived_dimensions=derived_dimensions,
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
                and issubclass(declaration.params.runtime_type, Mechanics)
            ):
                mechanics.add(scope)
            elif scope not in self.holds and declaration is not None and declaration.compiler.instantaneous:
                instant.add(scope)
        try:
            schedule = resolve_signal_schedule(graph)
        except SignalScheduleError as error:
            if error.cycle:
                addresses = {}
                for wire in error.cycle:
                    address = self.origins[json_pointer_parts(wire.authored_path)]
                    if isinstance(address.element, AttachmentElement):
                        address = address.model_copy(update={"element": address.element.parent})
                    addresses[address.to_text()] = address
                first, *related = addresses.values()
                reason = "Add a Delay to a wire in this same-step loop."
                raise ModelCompilationProblem(
                    first, reason, related=tuple(ModelGraphProblem(address=place, reason=reason) for place in related),
                ) from error
            address = self.origins.get(json_pointer_parts(error.authored_path))
            if address is not None:
                raise ModelCompilationProblem(address, str(error)) from error
            raise
        if not mechanics:
            return
        outgoing = {}
        samples = []
        for route in schedule.routes:
            if route.temporality == "instant":
                outgoing.setdefault(route.source.node, []).append(route)
            elif route.source.node in mechanics:
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
        signals = {item.label: item for item in binding.exposed_data if item.bindable}
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
            self.origins[("task_binding_spec", "bindings", str(len(result)))] = self.address(
                (), "port", end={"owner": "inputs", "port": port.id},
            )
            result.append(StudioTaskBinding(
                id=task_binding_id(item.id, node, port_id), source_data_id=item.id,
                target_node_id=node, target_port=port_id, role=task_data_role(item),
            ))
        return binding.model_copy(update={"bindings": result})

@dataclass(frozen=True)
class LoweredModelDocument:
    graph: GraphSpec
    task: StudioTaskBindingSpec | None
    origins: dict[tuple[str, ...], ModelAddress]
    plugins: tuple[str, ...]

    def interface(
        self, model: ModelDocument, components: ComponentRegistry,
        *, input_prototypes: Mapping[tuple[str, str], object] | None = None,
        task_inputs: Mapping[str, ValueType] | None = None,
    ) -> ResolvedModelInterface:
        """Project resolved sample types for task and objective pairing."""
        prototypes = (
            prototypes_from_task_bindings(self.task) if self.task is not None else {}
        ) | dict(input_prototypes or {})
        resolution = resolve_graph_parameters(self.graph, prototypes, components)
        legacy_inputs = {
            item.label: value for item in self.task.exposed_data
            if item.bindable and (value := task_data_value_type(item)) is not None
        } if self.task is not None else {}
        table = resolve_model_signals(model, resolution, components, task_inputs=legacy_inputs | (task_inputs or {}))
        for port in model.ports.inputs:
            incoming = legacy_inputs.get(port.name)
            if incoming is not None and (leaf := table.at(boundary_signal_port((), "inputs", port.id))) is not None:
                if not leaf.accepts(incoming):
                    raise ModelCompilationProblem(leaf.address, f"Task signal {port.name!r} does not fit the Model input")
        return ResolvedModelInterface(
            _Lower(model, components).period,
            _scoped_ports(model, table, "inputs"),
            _scoped_ports(model, table, "outputs"),
            tuple(_resolved_delay_links(resolution)),
            self.plugins,
            resolve_model_reset(resolution, components),
            table,
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
    signals: ModelSignalTable = field(default_factory=ModelSignalTable)

    @property
    def readable_leaves(self) -> tuple[ResolvedSignalLeaf, ...]:
        return self.signals.readable_leaves


class ResolvedModelInput(ScopedValue):
    required: bool

    @property
    def root_id(self) -> str:
        end = self.address.element
        assert isinstance(end, PortElement)
        return end.end.port

    @property
    def address(self) -> ModelAddress:
        return ModelAddress.from_text(self.id)


@dataclass(frozen=True)
class ResolvedModelLink:
    address: ModelAddress
    variable: str
    type: ValueType


def _resolved_delay_links(
    resolution: GraphParameterResolution, scope: tuple[str, ...] = (),
) -> Iterator[ResolvedModelLink]:
    for wire in resolution.graph.wires:
        initializer = wire.recurrent_initializer or {}
        if "model_delay_id" not in initializer:
            continue
        delay = ModelDelayCarry.model_validate(initializer)
        if delay.reset_value is None:
            continue
        address = ModelAddress.model_validate({
            "composites": scope,
            "element": {"kind": "field", "path": ["initial_source"], "parent": {
                "kind": "attachment", "parent": {"kind": "wire", "id": delay.model_wire_id},
                "id": delay.model_delay_id,
            }},
        })
        layout = resolution.delay_layouts[wire.target_node, wire.target_port]
        leaf = layout.leaves[0]
        unit = UNITS.get(delay.units or "1")
        if len(layout.leaves) != 1 or leaf.path or any(not size for size in leaf.shape) or unit is None:
            raise ModelCompilationProblem(address, "Delay Initial needs a Task variable with the delayed signal's numeric type.")
        base = {"b": ValueBase.BOOLEAN, "i": ValueBase.INTEGER, "u": ValueBase.INTEGER, "f": ValueBase.REAL}
        yield ResolvedModelLink(address, delay.reset_value, ValueType(
            base=base[leaf.dtype.kind], shape=leaf.shape, dimensions=unit.dimensions,
        ))
    for name, child in resolution.subgraphs.items():
        yield from _resolved_delay_links(child, (*scope, name))


def _scoped_ports(
    model: ModelDocument, table: ModelSignalTable, direction: str,
) -> tuple[ScopedValue, ...]:
    results = []
    for port in getattr(model.ports, direction):
        address = boundary_signal_port((), direction, port.id)
        if isinstance(port, SignalInput):
            leaf = table.at(address)
        else:
            leaves = table.at_port(address)
            if any(item.path and item.type is not None for item in leaves):
                continue
            leaf = table.at(address)
        if leaf is None or leaf.type is None:
            raise ModelCompilationProblem(address, f"Cannot resolve the numeric sample type of {port.name!r}")
        value = ScopedValue(id=address.to_text(), name=port.name, level=Level.STEP, type=leaf.type)
        results.append(
            ResolvedModelInput(**value.model_dump(), required=port.default is None)
            if isinstance(port, SignalInput) else value
        )
    return tuple(results)


def lower_model_document(
    model: ModelDocument, components: ComponentRegistry,
    task: StudioTaskBindingSpec | None = None,
    *, supplied_inputs: frozenset[str] = frozenset(),
) -> LoweredModelDocument:
    """Project one authored model through the graph compiler's registered declarations."""
    ports = ModelRegistryPorts(components)
    problems = tuple(model_graph_problems(model, ports=ports.port, component_problem=ports.registration_problem))
    if problems:
        first, *related = problems
        raise ModelCompilationProblem(first.address, first.reason, related=tuple(related))
    lower = _Lower(model, components)
    names = frozenset(item.label for item in task.exposed_data if item.bindable) if task else frozenset()
    graph = lower.graph(model.interior, model.ports, (), available=names | supplied_inputs)
    lower.reject_within_step_paths(graph)
    binding = lower.root_inputs(task, graph) if task is not None else None
    return LoweredModelDocument(graph, binding, lower.origins, tuple(sorted(lower.plugins)))


def lower_planar_scene_interior(
    model: ModelDocument, path: tuple[str, ...], components: ComponentRegistry,
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
                lower = _Lower(model, components)
                substeps = lower.numeric_field(
                    component.substeps, path[:-1], {"kind": "component", "id": name}, ("substeps",),
                )
                return lower.physical(child, component.ports, path, substeps=substeps)
            break
        interior = child
    raise ValueError(f"{list(path)!r} must identify a planar multibody interior.")
