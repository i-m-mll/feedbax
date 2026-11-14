"""Schema validation for Graph compiler connections and task bindings."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from feedbax.contracts._causal_inputs import input_driver_conflicts
from feedbax.contracts.value_schema import SchemaOrigin, ValueSchema
from feedbax.component_registry import format_missing_interior_message, required_interior_domain
from feedbax.studio.protocol import (
    GRAPH_BINDABLE_TASK_DATA_ROLES,
    PROTOCOL_TASK_DATA_KINDS,
    TASK_DATA_ROLES,
    is_bindable_task_data,
    task_binding_id,
    task_data_role,
    task_data_surface,
    task_data_uses_protocol_path,
)
from feedbax.compiler.prototypes import (
    DerivedDimensionConflict,
    DerivedDimensionError,
    prototypes_from_task_bindings,
)
from feedbax.compiler.resolution import GraphParameterResolution, resolve_graph_parameters
from feedbax.runtime._signal_schedule import (
    SignalScheduleError,
    needs_signal_schedule,
    resolve_signal_schedule,
)
from feedbax.contracts.component import (
    DynamicPortPolicyError,
    PortType,
    derive_dynamic_port_layout,
)
from feedbax.contracts.graph import GraphSpec, StudioTaskBindingSpec


class StudioSchemaModel(BaseModel):
    """Base model for static Studio schema records."""

    model_config = ConfigDict(extra="forbid")


class PortSchema(StudioSchemaModel):
    """Provider-owned graph port schema record."""

    id: str
    label: str
    node_id: Optional[str] = None
    component_type: Optional[str] = None
    port: str
    direction: Literal["input", "output"]
    value_schema: ValueSchema
    bound_task_data_id: Optional[str] = None
    origin: SchemaOrigin = "unknown"
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskDataSchema(StudioSchemaModel):
    """Provider-owned task data schema record."""

    id: str
    label: str
    kind: str
    role: str
    path: str
    bindable: bool = False
    value_schema: ValueSchema
    origin: SchemaOrigin = "unknown"
    metadata: dict[str, Any] = Field(default_factory=dict)


class SchemaValidationIssue(StudioSchemaModel):
    """Validation issue found while checking graph connections or task bindings."""

    type: str
    message: str
    severity: Literal["error", "warning", "info"] = "error"
    location: Optional[dict[str, str]] = None


def validate_graph_connection_schema(
    graph: GraphSpec,
    base_path: str = "",
    task_binding_spec: Optional[StudioTaskBindingSpec] = None,
    *,
    component_registry: Any,
    _resolution: GraphParameterResolution | None = None,
) -> list[SchemaValidationIssue]:
    """Validate graph wires against provider-owned port schema records."""

    resolution = _resolution or resolve_graph_parameters(
        graph,
        prototypes_from_task_bindings(task_binding_spec) if task_binding_spec else {},
        component_registry,
        allow_incomplete=True,
    )
    graph = resolution.graph
    issues = _validate_local_connections(
        resolution, base_path, task_binding_spec, component_registry=component_registry,
    )
    issues.extend(_parameter_resolution_issues(resolution, base_path))
    if needs_signal_schedule(graph):
        try:
            resolve_signal_schedule(graph)
        except SignalScheduleError as error:
            issues.append(SchemaValidationIssue(
                type=error.issue_type,
                message=str(error),
                location={"path": base_path + error.authored_path.removeprefix("/graph")},
            ))
    else:
        issues.extend(_instant_cycle_issues(graph, base_path))
    return issues


def _validate_local_connections(
    resolution: GraphParameterResolution,
    base_path: str,
    task_binding_spec: StudioTaskBindingSpec | None,
    *,
    component_registry: Any,
) -> list[SchemaValidationIssue]:
    graph = resolution.graph
    ports = _enumerate_graph_ports(
        graph, component_registry=component_registry, resolution=resolution
    )
    by_node_port_direction = {
        (port.node_id, port.port, port.direction): port
        for port in ports
        if port.node_id is not None
    }
    by_node_port = {(port.node_id, port.port): port for port in ports if port.node_id is not None}
    issues: list[SchemaValidationIssue] = _missing_subgraph_issues(
        graph, base_path, component_registry=component_registry
    )
    issues.extend(
        SchemaValidationIssue(
            type="graph_input_occupied",
            message=str(conflict),
            location={"path": conflict.authored_path, "occupied_by": conflict.occupied_by},
        )
        for conflict in input_driver_conflicts(graph.wires, graph.input_bindings, base_path)
    )
    issues.extend(
        _dynamic_port_arity_issues(
            graph,
            task_binding_spec,
            base_path,
            component_registry=component_registry,
        )
    )
    for index, wire in enumerate(graph.wires):
        wire_path = f"{base_path}/wires/{index}"
        source = by_node_port_direction.get((wire.source_node, wire.source_port, "output"))
        target = by_node_port_direction.get((wire.target_node, wire.target_port, "input"))
        source_any_direction = by_node_port.get((wire.source_node, wire.source_port))
        target_any_direction = by_node_port.get((wire.target_node, wire.target_port))

        if source is None:
            if source_any_direction is not None:
                issues.append(
                    SchemaValidationIssue(
                        type="wrong_source_port_direction",
                        message=(
                            f"Wire source {wire.source_node}.{wire.source_port} is "
                            "not an output port"
                        ),
                        location={"path": f"{wire_path}/source_port"},
                    )
                )
            elif wire.source_node in graph.nodes:
                issues.append(
                    SchemaValidationIssue(
                        type="unknown_source_port",
                        message=(
                            f"Wire source port {wire.source_node}.{wire.source_port} does not exist"
                        ),
                        location={"path": f"{wire_path}/source_port"},
                    )
                )
            else:
                issues.append(
                    SchemaValidationIssue(
                        type="unknown_source_node",
                        message=f"Wire source node {wire.source_node!r} does not exist",
                        location={"path": f"{wire_path}/source_node"},
                    )
                )

        if target is None:
            if target_any_direction is not None:
                issues.append(
                    SchemaValidationIssue(
                        type="wrong_target_port_direction",
                        message=(
                            f"Wire target {wire.target_node}.{wire.target_port} is "
                            "not an input port"
                        ),
                        location={"path": f"{wire_path}/target_port"},
                    )
                )
            elif wire.target_node in graph.nodes:
                issues.append(
                    SchemaValidationIssue(
                        type="unknown_target_port",
                        message=(
                            f"Wire target port {wire.target_node}.{wire.target_port} does not exist"
                        ),
                        location={"path": f"{wire_path}/target_port"},
                    )
                )
            else:
                issues.append(
                    SchemaValidationIssue(
                        type="unknown_target_node",
                        message=f"Wire target node {wire.target_node!r} does not exist",
                        location={"path": f"{wire_path}/target_node"},
                    )
                )

        if source is None or target is None:
            continue
        if wire.temporality == "recurrent" and wire.recurrent_initializer is None:
            issues.append(
                SchemaValidationIssue(
                    type="recurrent_initializer_missing",
                    message=(
                        f"Recurrent wire {wire.source_node}.{wire.source_port} -> "
                        f"{wire.target_node}.{wire.target_port} needs an initial value"
                    ),
                    location={"path": f"{wire_path}/recurrent_initializer"},
                )
            )
        issues.extend(
            _value_schema_compatibility_issues(
                source.value_schema,
                target.value_schema,
                path=wire_path,
                source_label=source.label,
                target_label=target.label,
                issue_prefix="graph_wire",
            )
        )

    issues.extend(_mux_connected_input_issues(graph, task_binding_spec, base_path))
    for name, child_resolution in resolution.subgraphs.items():
        token = name.replace("~", "~0").replace("/", "~1")
        issues.extend(_validate_local_connections(
            child_resolution, f"{base_path}/subgraphs/{token}", None,
            component_registry=component_registry,
        ))
    return issues


def _missing_subgraph_issues(
    graph: GraphSpec, base_path: str, *, component_registry: Any
) -> list[SchemaValidationIssue]:
    registry = component_registry
    subgraphs = graph.subgraphs or {}
    issues: list[SchemaValidationIssue] = []
    for node_id, node in graph.nodes.items():
        required_domain = required_interior_domain(node.type, registry)
        if required_domain is None or node_id in subgraphs:
            continue
        if node.type == "feedbax.Network":
            message = (
                f"Network node {node_id!r} has no subgraph. Open it in Studio to "
                "generate the internal architecture, then save again."
            )
        else:
            message = format_missing_interior_message(
                node_name=node_id,
                node_type=node.type,
                domain_id=required_domain,
            )
        issues.append(
            SchemaValidationIssue(
                type="missing_subgraph",
                message=message,
                location={"path": f"{base_path}/nodes/{node_id}", "node": node_id},
            )
        )
    return issues


def _parameter_resolution_issues(
    resolution: GraphParameterResolution,
    base_path: str,
) -> list[SchemaValidationIssue]:
    issues = []
    for error in resolution.errors:
        cause = error.cause
        location = {"path": base_path + error.authored_path.removeprefix("/graph")}
        kind = "component_parameters_unresolved"
        if isinstance(cause, DerivedDimensionError):
            kind = (
                "derived_dimension_conflict"
                if isinstance(cause, DerivedDimensionConflict)
                else "derived_dimension_unresolved"
            )
            graph_path = error.authored_path.rsplit("/", 2)[0]
            graph = resolution.graph
            for name in graph_path.split("/")[3::2]:
                graph = graph.subgraphs[name.replace("~1", "/").replace("~0", "~")]
            index = next(
                (i for i, rule in enumerate(graph.derived_dimensions) if rule == cause.rule), 0
            )
            location["path"] = (
                base_path + graph_path.removeprefix("/graph") + f"/derived_dimensions/{index}"
            )
            location.update({"node": cause.rule.node, "param": cause.rule.param})
            if isinstance(cause, DerivedDimensionConflict):
                location.update({"declared": str(cause.declared), "derived": str(cause.derived)})
        issues.append(SchemaValidationIssue(type=kind, message=str(error), location=location))
    return issues


def _mux_connected_input_issues(
    graph: GraphSpec,
    task_binding_spec: Optional[StudioTaskBindingSpec],
    base_path: str,
) -> list[SchemaValidationIssue]:
    issues: list[SchemaValidationIssue] = []
    graph_input_ports = {
        (node_id, port)
        for node_id, port in (tuple(binding) for binding in graph.input_bindings.values())
    }
    wired_ports = {(wire.target_node, wire.target_port) for wire in graph.wires}
    task_bound_ports = (
        {(binding.target_node_id, binding.target_port) for binding in task_binding_spec.bindings}
        if task_binding_spec is not None
        else set()
    )
    occupied_ports = graph_input_ports | wired_ports | task_bound_ports

    for node_id, node in graph.nodes.items():
        if node.type != "feedbax.Mux":
            continue
        connected_inputs = {port for port in node.input_ports if (node_id, port) in occupied_ports}
        if len(connected_inputs) >= 2:
            continue
        issues.append(
            SchemaValidationIssue(
                type="mux_needs_two_connected_inputs",
                message=f"Mux {node_id!r} needs at least two connected inputs",
                location={"path": f"{base_path}/nodes/{node_id}"},
            )
        )
    return issues


def _instant_cycle_issues(graph: GraphSpec, base_path: str) -> list[SchemaValidationIssue]:
    adjacency: dict[str, list[tuple[str, int]]] = {node_id: [] for node_id in graph.nodes}
    for index, wire in enumerate(graph.wires):
        if wire.temporality == "recurrent":
            continue
        if wire.source_node not in graph.nodes or wire.target_node not in graph.nodes:
            continue
        adjacency.setdefault(wire.source_node, []).append((wire.target_node, index))

    visited: set[str] = set()
    active: set[str] = set()
    path_edges: list[int] = []

    def visit(node_id: str) -> Optional[list[int]]:
        visited.add(node_id)
        active.add(node_id)
        for target_id, wire_index in adjacency.get(node_id, []):
            if target_id not in visited:
                path_edges.append(wire_index)
                found = visit(target_id)
                if found is not None:
                    return found
                path_edges.pop()
            elif target_id in active:
                return [*path_edges, wire_index]
        active.remove(node_id)
        return None

    for node_id in graph.nodes:
        if node_id in visited:
            continue
        cycle = visit(node_id)
        if cycle is not None:
            first_index = cycle[0]
            return [
                SchemaValidationIssue(
                    type="instant_cycle",
                    message="Instant wires contain a same-step cycle; mark one cycle edge recurrent",
                    location={
                        "path": f"{base_path}/wires/{first_index}",
                        "cycle_wires": ",".join(str(index) for index in cycle),
                    },
                )
            ]
    return []


def _dynamic_port_arity_issues(
    graph: GraphSpec,
    task_binding_spec: Optional[StudioTaskBindingSpec],
    base_path: str,
    *,
    component_registry: Any,
) -> list[SchemaValidationIssue]:
    issues: list[SchemaValidationIssue] = []
    for node_id, node in graph.nodes.items():
        meta = component_registry.get(node.type)
        if meta is None or meta.dynamic_port_policy is None:
            continue
        try:
            expected = derive_dynamic_port_layout(meta.dynamic_port_policy, node.params)
        except DynamicPortPolicyError as exc:
            issues.append(
                SchemaValidationIssue(
                    type="dynamic_port_policy_invalid",
                    message=f"{node.type} node {node_id!r}: {exc}",
                    location={"path": f"{base_path}/nodes/{node_id}/params"},
                )
            )
            continue
        assert expected is not None
        if (
            tuple(node.input_ports) != expected.input_ports
            or tuple(node.output_ports) != expected.output_ports
        ):
            count_param = meta.dynamic_port_policy.count_param
            issues.append(
                SchemaValidationIssue(
                    type="dynamic_port_arity_mismatch",
                    message=(
                        f"{node.type} node {node_id!r} declares input_ports "
                        f"{node.input_ports!r} and output_ports {node.output_ports!r}, but "
                        f"{count_param}={node.params.get(count_param, meta.default_params.get(count_param))!r} "
                        f"requires input_ports {list(expected.input_ports)!r} and "
                        f"output_ports {list(expected.output_ports)!r}"
                    ),
                    location={
                        "path": f"{base_path}/nodes/{node_id}/{meta.dynamic_port_policy.direction}_ports"
                    },
                )
            )
        if meta.dynamic_port_policy.direction == "input":
            referenced = {
                port
                for target_node, port in graph.input_bindings.values()
                if target_node == node_id
            }
            referenced.update(
                wire.target_port for wire in graph.wires if wire.target_node == node_id
            )
            if task_binding_spec is not None:
                referenced.update(
                    binding.target_port
                    for binding in task_binding_spec.bindings
                    if binding.target_node_id == node_id
                )
            allowed = expected.input_ports
        else:
            referenced = {
                port
                for source_node, port in graph.output_bindings.values()
                if source_node == node_id
            }
            referenced.update(
                wire.source_port for wire in graph.wires if wire.source_node == node_id
            )
            allowed = expected.output_ports
        for port in sorted(set(referenced).difference(allowed)):
            issues.append(
                SchemaValidationIssue(
                    type="dynamic_port_arity_mismatch",
                    message=(
                        f"{node.type} node {node_id!r} has a connection to {port!r} "
                        f"outside its policy-derived {meta.dynamic_port_policy.direction} "
                        f"namespace {list(allowed)!r}"
                    ),
                    location={
                        "path": (
                            f"{base_path}/nodes/{node_id}/"
                            f"{meta.dynamic_port_policy.direction}_ports"
                        )
                    },
                )
            )
    return issues


def _enumerate_graph_ports(
    graph: GraphSpec,
    task_binding_spec: Optional[StudioTaskBindingSpec] = None,
    *,
    component_registry: Any,
    resolution: GraphParameterResolution | None = None,
) -> list[PortSchema]:
    registry = component_registry
    ports: list[PortSchema] = []
    for node_id, node in graph.nodes.items():
        meta = registry.get(node.type)
        subgraph = graph.subgraphs.get(node_id) if graph.subgraphs else None
        input_ports = list(node.input_ports or (meta.input_ports if meta is not None else []))
        output_ports = list(node.output_ports or (meta.output_ports if meta is not None else []))
        for port_name in input_ports:
            port_type = _component_input_port_type(meta, node.params, port_name)
            ports.append(
                _port_schema(
                    node_id=node_id,
                    component_type=node.type,
                    node_params=node.params,
                    port=port_name,
                    direction="input",
                    port_type=port_type,
                    subgraph=subgraph,
                    component_registry=component_registry,
                )
            )
        for port_name in output_ports:
            port_type = _component_output_port_type(meta, node.params, port_name)
            ports.append(
                _port_schema(
                    node_id=node_id,
                    component_type=node.type,
                    node_params=node.params,
                    port=port_name,
                    direction="output",
                    port_type=port_type,
                    subgraph=subgraph,
                    component_registry=component_registry,
                )
            )
    if resolution is not None:
        for port in ports:
            key = (port.node_id, port.port)
            prototype = (
                resolution.node_inputs if port.direction == "input" else resolution.node_outputs
            ).get(key)
            shape = getattr(prototype, "shape", None)
            if shape is not None:
                port.value_schema.shape = list(shape)
                port.value_schema.rank = len(shape)
                port.value_schema.origin = "inferred_static"
                port.value_schema.metadata["dimension_source"] = "compiler_prototype"
                port.origin = "inferred_static"
    for port_name in graph.input_ports:
        ports.append(
            _graph_port_schema(
                port=port_name,
                direction="input",
                bound=node_input_binding(graph, port_name),
                ports=ports,
            )
        )
    for port_name in graph.output_ports:
        ports.append(
            _graph_port_schema(
                port=port_name,
                direction="output",
                bound=node_output_binding(graph, port_name),
                ports=ports,
            )
        )
    _apply_mux_output_shapes(ports, graph, task_binding_spec)
    return ports


def _component_input_port_type(
    meta: Any, params: dict[str, Any], port_name: str
) -> Optional[PortType]:
    if meta is None:
        return None
    port_type = meta.port_types.inputs.get(port_name) if meta.port_types is not None else None
    if port_type is None and meta.dynamic_port_policy is not None:
        effective_params = {**meta.default_params, **params}
        try:
            layout = derive_dynamic_port_layout(meta.dynamic_port_policy, effective_params)
        except DynamicPortPolicyError:
            return None
        if meta.dynamic_port_policy.direction == "input" and port_name in layout.input_ports:
            port_type = meta.dynamic_port_policy.dynamic_port_type
    return port_type


def _component_output_port_type(
    meta: Any, params: dict[str, Any], port_name: str
) -> Optional[PortType]:
    if meta is None:
        return None
    port_type = meta.port_types.outputs.get(port_name) if meta.port_types is not None else None
    if port_type is None and meta.dynamic_port_policy is not None:
        effective_params = {**meta.default_params, **params}
        try:
            layout = derive_dynamic_port_layout(meta.dynamic_port_policy, effective_params)
        except DynamicPortPolicyError:
            return None
        if meta.dynamic_port_policy.direction == "output" and port_name in layout.output_ports:
            port_type = meta.dynamic_port_policy.dynamic_port_type
    return port_type


def _port_schema(
    *,
    node_id: str,
    component_type: str,
    node_params: dict[str, Any],
    port: str,
    direction: Literal["input", "output"],
    port_type: Optional[PortType],
    subgraph: Optional[GraphSpec] = None,
    component_registry: Any,
) -> PortSchema:
    port_id = f"port:{node_id}.{port}:{direction}"
    dtype = port_type.dtype if port_type is not None else None
    origin: SchemaOrigin = "declared" if port_type is not None else "unknown"
    shape = list(port_type.shape) if port_type is not None and port_type.shape else None
    rank = port_type.rank if port_type is not None else None
    metadata: dict[str, Any] = {"component_type": component_type}
    inferred = _component_port_dimension(
        component_type=component_type,
        node_params=node_params,
        port=port,
        direction=direction,
    )
    if inferred is not None:
        shape = inferred["shape"]
        rank = len(inferred["shape"])
        origin = "inferred_static"
        metadata.update(inferred["metadata"])
    boundary = _subgraph_boundary_value_schema(
        subgraph, port, direction, component_registry=component_registry
    )
    if shape is None and boundary is not None and boundary.shape is not None:
        shape = boundary.shape
        rank = boundary.rank if boundary.rank is not None else len(boundary.shape)
        origin = "inferred_static"
        metadata["dimension_source"] = "subgraph_boundary"
        metadata["boundary_port"] = port
    if dtype is None and boundary is not None:
        dtype = boundary.dtype
    return PortSchema(
        id=port_id,
        label=f"{node_id}.{port}",
        node_id=node_id,
        component_type=component_type,
        port=port,
        direction=direction,
        value_schema=ValueSchema(
            id=f"value:{port_id}",
            label=f"{node_id}.{port}",
            kind="graph_port",
            dtype=dtype,
            shape=shape,
            rank=rank,
            origin=origin,
            metadata=metadata,
        ),
        origin=origin,
    )


def _component_port_dimension(
    *,
    component_type: str,
    node_params: dict[str, Any],
    port: str,
    direction: Literal["input", "output"],
) -> Optional[dict[str, Any]]:
    dimension_param: Optional[str] = None
    dimension: Optional[int] = None
    if component_type == "feedbax.Network":
        if direction == "output" and port == "output":
            dimension_param = "out_size"
        elif direction == "output" and port == "hidden":
            dimension_param = "hidden_size"
    elif component_type in {"feedbax.Linear", "feedbax.MLP"}:
        if direction == "input" and port == "input":
            dimension_param = "input_size"
        elif direction == "output" and port == "output":
            dimension_param = "output_size"
    elif component_type in {"feedbax.GRU", "feedbax.LSTM"}:
        if direction == "input" and port == "input":
            dimension_param = "input_size"
        elif port in {"output", "hidden", "cell"}:
            dimension_param = "hidden_size"
    elif component_type in {"feedbax.TwoLinkArm", "feedbax.PointMass", "feedbax.Mechanics"} and port == "force":
        dimension = 2
    elif component_type == "feedbax.Arm6MuscleRigidTendon":
        if port == "excitation":
            dimension = 6
        elif port in {"angles", "angular_velocities", "torques"}:
            dimension = 2
        elif port in {"forces", "activations"}:
            dimension = 6
    elif component_type == "feedbax.PointMass8MuscleRelu":
        if port == "excitation":
            n_pairs = _positive_int_param(node_params, "n_pairs")
            dimension = n_pairs * 2 if n_pairs is not None else None
        elif port == "force_2d":
            dimension = 2
        elif port in {"forces", "activations"}:
            n_pairs = _positive_int_param(node_params, "n_pairs")
            dimension = n_pairs * 2 if n_pairs is not None else None
    elif component_type == "feedbax.Demux":
        sizes = node_params.get("sizes")
        if isinstance(sizes, (list, tuple)) and all(
            isinstance(size, int) and size > 0 for size in sizes
        ):
            if direction == "input" and port == "input":
                dimension = sum(sizes)
            elif (
                direction == "output"
                and port.startswith("out_")
                and port.removeprefix("out_").isdigit()
            ):
                index = int(port.removeprefix("out_"))
                if 0 <= index < len(sizes):
                    dimension = sizes[index]

    if dimension_param is not None:
        dimension = _positive_int_param(node_params, dimension_param)
    if dimension is None:
        return None
    return {
        "shape": [dimension],
        "metadata": {
            "dimension_source": "component_param" if dimension_param else "component_contract",
            "dimension_param": dimension_param,
        },
    }


def _positive_int_param(params: dict[str, Any], key: str) -> Optional[int]:
    value = params.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    return None


def _subgraph_boundary_value_schema(
    subgraph: Any,
    port: str,
    direction: Literal["input", "output"],
    *,
    component_registry: Any,
) -> Optional[ValueSchema]:
    if subgraph is None:
        return None
    if not isinstance(subgraph, GraphSpec):
        return ValueSchema(
            id=f"value:acausal-boundary:{port}:{direction}",
            label=port,
            kind="signal",
            dtype=None,
            shape=None,
            rank=None,
            origin="inferred_static",
            metadata={"dimension_source": "acausal_boundary"},
        )
    binding = (
        subgraph.input_bindings.get(port)
        if direction == "input"
        else subgraph.output_bindings.get(port)
    )
    if binding is None:
        return None
    node_id, bound_port = binding
    for schema in _enumerate_graph_ports(subgraph, component_registry=component_registry):
        if (
            schema.node_id == node_id
            and schema.port == bound_port
            and schema.direction == direction
        ):
            return schema.value_schema
    return None


def _apply_mux_output_shapes(
    ports: list[PortSchema],
    graph: GraphSpec,
    task_binding_spec: Optional[StudioTaskBindingSpec],
) -> None:
    task_data_schemas = (
        {
            item.id.removeprefix("task_data:"): item.value_schema
            for item in _enumerate_task_data(task_binding_spec)
        }
        if task_binding_spec is not None
        else {}
    )
    output_ports = {
        (port.node_id, port.port): port
        for port in ports
        if port.direction == "output" and port.node_id is not None
    }
    input_sources: dict[tuple[str, str], ValueSchema] = {}
    for wire in graph.wires:
        source = output_ports.get((wire.source_node, wire.source_port))
        if source is not None:
            input_sources[(wire.target_node, wire.target_port)] = source.value_schema
    for binding in task_binding_spec.bindings if task_binding_spec is not None else []:
        source = task_data_schemas.get(binding.source_data_id)
        if source is not None:
            input_sources[(binding.target_node_id, binding.target_port)] = source

    for port in ports:
        if port.component_type != "feedbax.Mux" or port.direction != "output" or port.port != "output":
            continue
        node = graph.nodes.get(port.node_id or "")
        if node is None:
            continue
        input_shapes = [
            _compatibility_shape(input_sources[(port.node_id, input_port)])
            for input_port in node.input_ports
            if (port.node_id, input_port) in input_sources
        ]
        if not input_shapes or any(shape is None or len(shape) != 1 for shape in input_shapes):
            continue
        dimensions = [shape[0] for shape in input_shapes if shape is not None]
        if not all(isinstance(dim, int) and dim > 0 for dim in dimensions):
            continue
        width = sum(dimensions)
        metadata = dict(port.value_schema.metadata)
        metadata.update(
            {
                "dimension_source": "mux_concat_inputs",
                "input_shapes": input_shapes,
            }
        )
        port.value_schema = port.value_schema.model_copy(
            update={
                "shape": [width],
                "rank": 1,
                "origin": "inferred_static",
                "metadata": metadata,
            }
        )
        port.origin = "inferred_static"


def _graph_port_schema(
    *,
    port: str,
    direction: Literal["input", "output"],
    bound: Optional[tuple[str, str]] = None,
    ports: list[PortSchema] | None = None,
) -> PortSchema:
    port_id = f"port:graph.{port}:{direction}"
    metadata: dict[str, Any] = {}
    if bound is not None:
        metadata["binding"] = {"node_id": bound[0], "port": bound[1]}
    bound_schema = None
    if bound is not None and ports is not None:
        bound_schema = next(
            (
                schema
                for schema in ports
                if schema.node_id == bound[0]
                and schema.port == bound[1]
                and schema.direction == direction
            ),
            None,
        )
    value_metadata = dict(metadata)
    if bound_schema is not None:
        value_metadata["dimension_source"] = "graph_port_binding"
    return PortSchema(
        id=port_id,
        label=f"graph.{port}",
        port=port,
        direction=direction,
        value_schema=ValueSchema(
            id=f"value:{port_id}",
            label=f"graph.{port}",
            kind="graph_port",
            dtype=bound_schema.value_schema.dtype if bound_schema is not None else None,
            shape=bound_schema.value_schema.shape if bound_schema is not None else None,
            rank=bound_schema.value_schema.rank if bound_schema is not None else None,
            origin="inferred_static",
            metadata=value_metadata,
        ),
        origin="inferred_static",
        metadata=metadata,
    )


def node_input_binding(graph: GraphSpec, port_name: str) -> Optional[tuple[str, str]]:
    binding = graph.input_bindings.get(port_name)
    return tuple(binding) if binding is not None else None


def node_output_binding(graph: GraphSpec, port_name: str) -> Optional[tuple[str, str]]:
    binding = graph.output_bindings.get(port_name)
    return tuple(binding) if binding is not None else None


def _enumerate_task_data(task_binding_spec: StudioTaskBindingSpec) -> list[TaskDataSchema]:
    task_data: list[TaskDataSchema] = []
    for data in task_binding_spec.exposed_data:
        value = data.value_spec
        role = task_data_role(data)
        surface = task_data_surface(data)
        shape = data.expected_shape or (value.shape if value is not None else None)
        sample_shape = _task_data_sample_shape(shape)
        metadata = {
            **data.metadata,
            "task_data_role": role,
            "task_data_surface": surface,
        }
        if sample_shape is not None:
            metadata["sample_shape"] = sample_shape
            metadata["sample_rank"] = len(sample_shape)
            if shape is not None and sample_shape != shape:
                metadata["time_axis"] = 0
                metadata["view"] = "trajectory"
        task_data.append(
            TaskDataSchema(
                id=f"task_data:{data.id}",
                label=data.label,
                kind=data.kind,
                role=role,
                path=data.path,
                bindable=is_bindable_task_data(data),
                value_schema=ValueSchema(
                    id=f"value:task_data:{data.id}",
                    label=data.label,
                    kind="task_data",
                    dtype=data.dtype or (value.dtype if value is not None else None),
                    shape=shape,
                    rank=len(shape) if shape is not None else None,
                    units=data.units or (value.units if value is not None else None),
                    frame=data.frame or (value.frame if value is not None else None),
                    origin="declared",
                    metadata={
                        **metadata,
                        "task_data_path": data.path,
                        "value_spec": value.model_dump(mode="json", exclude_none=True)
                        if value is not None
                        else None,
                    },
                ),
                origin="declared",
                metadata=metadata,
            )
        )
    return task_data


def _task_data_sample_shape(shape: Optional[list[Any]]) -> Optional[list[Any]]:
    if shape is None:
        return None
    if len(shape) > 0 and shape[0] == "time":
        return list(shape[1:])
    return list(shape)


def validate_task_binding_schema(
    task_binding_spec: StudioTaskBindingSpec,
    graph: Optional[GraphSpec],
    base_path: str,
    *,
    component_registry: Any,
) -> list[SchemaValidationIssue]:
    """Validate task-data bindings against task-data and graph port schemas."""

    issues: list[SchemaValidationIssue] = []
    seen_data: set[str] = set()
    seen_bindings: set[str] = set()
    data_by_id = {}
    for index, data in enumerate(task_binding_spec.exposed_data):
        data_path = f"{base_path}/exposed_data/{index}"
        if data.id in seen_data:
            issues.append(
                SchemaValidationIssue(
                    type="duplicate_task_data",
                    message=f"Task data {data.id!r} is declared more than once",
                    location={"path": f"{data_path}/id"},
                )
            )
        seen_data.add(data.id)
        data_by_id[data.id] = data
        issues.extend(_task_data_role_issues(data, data_path))

    if graph is None:
        if task_binding_spec.bindings:
            issues.append(
                SchemaValidationIssue(
                    type="task_bindings_without_graph",
                    message="Task bindings cannot be checked because the scenario has no graph",
                    location={"path": f"{base_path}/bindings"},
                )
            )
        return issues

    occupied = {(wire.target_node, wire.target_port) for wire in graph.wires}
    task_data_schemas = {
        item.id.removeprefix("task_data:"): item for item in _enumerate_task_data(task_binding_spec)
    }
    port_schemas = {
        (port.node_id, port.port): port
        for port in _enumerate_graph_ports(
            graph, task_binding_spec, component_registry=component_registry
        )
        if port.direction == "input" and port.node_id is not None
    }
    binding_targets: set[tuple[str, str]] = set()
    for index, binding in enumerate(task_binding_spec.bindings):
        binding_path = f"{base_path}/bindings/{index}"
        expected_id = task_binding_id(binding.source_data_id, binding.target_node_id, binding.target_port)
        if binding.id in seen_bindings:
            issues.append(
                SchemaValidationIssue(
                    type="duplicate_task_binding",
                    message=f"Task binding {binding.id!r} is declared more than once",
                    location={"path": f"{binding_path}/id"},
                )
            )
        seen_bindings.add(binding.id)
        if binding.id != expected_id:
            issues.append(
                SchemaValidationIssue(
                    type="task_binding_id_mismatch",
                    message=f"Task binding id {binding.id!r} does not match {expected_id!r}",
                    location={"path": f"{binding_path}/id"},
                )
            )
        data = data_by_id.get(binding.source_data_id)
        if data is None:
            issues.append(
                SchemaValidationIssue(
                    type="unknown_task_data",
                    message=f"Task binding source {binding.source_data_id!r} is not declared",
                    location={"path": f"{binding_path}/source_data_id"},
                )
            )
        elif not is_bindable_task_data(data):
            role = task_data_role(data)
            issues.append(
                SchemaValidationIssue(
                    type="task_data_not_bindable",
                    message=(
                        f"Task data {binding.source_data_id!r} has protocol role "
                        f"{role!r} and is not bindable to graph inputs"
                    ),
                    location={"path": f"{binding_path}/source_data_id"},
                )
            )
        else:
            role = task_data_role(data)
            if binding.role != role:
                issues.append(
                    SchemaValidationIssue(
                        type="task_binding_role_mismatch",
                        message=(
                            f"Task binding role {binding.role!r} does not match source "
                            f"task data role {role!r}"
                        ),
                        location={"path": f"{binding_path}/role"},
                    )
                )

        target_node = graph.nodes.get(binding.target_node_id)
        if target_node is None:
            issues.append(
                SchemaValidationIssue(
                    type="unknown_task_binding_target_node",
                    message=f"Task binding target node {binding.target_node_id!r} does not exist",
                    location={"path": f"{binding_path}/target_node_id"},
                )
            )
            continue
        if binding.target_port not in target_node.input_ports:
            issues.append(
                SchemaValidationIssue(
                    type="unknown_task_binding_target_port",
                    message=(
                        f"Task binding target "
                        f"{binding.target_node_id}.{binding.target_port} does not exist"
                    ),
                    location={"path": f"{binding_path}/target_port"},
                )
            )
        else:
            data_schema = task_data_schemas.get(binding.source_data_id)
            port_schema = port_schemas.get((binding.target_node_id, binding.target_port))
            if data_schema is not None and port_schema is not None:
                issues.extend(
                    _value_schema_compatibility_issues(
                        data_schema.value_schema,
                        port_schema.value_schema,
                        path=binding_path,
                        source_label=data_schema.label,
                        target_label=port_schema.label,
                        issue_prefix="task_binding",
                    )
                )

        target = (binding.target_node_id, binding.target_port)
        if target in occupied or target in binding_targets:
            issues.append(
                SchemaValidationIssue(
                    type="task_binding_target_occupied",
                    message=(
                        f"Task binding target "
                        f"{binding.target_node_id}.{binding.target_port} is already occupied"
                    ),
                    location={"path": binding_path},
                )
            )
        binding_targets.add(target)
    return issues


def _task_data_role_issues(
    data: Any,
    data_path: str,
) -> list[SchemaValidationIssue]:
    role = task_data_role(data)
    issues: list[SchemaValidationIssue] = []
    if role not in TASK_DATA_ROLES:
        issues.append(
            SchemaValidationIssue(
                type="unknown_task_data_role",
                message=f"Task data {data.id!r} declares unknown role {role!r}",
                location={"path": f"{data_path}/role"},
            )
        )
        return issues

    if data.bindable and role not in GRAPH_BINDABLE_TASK_DATA_ROLES:
        issues.append(
            SchemaValidationIssue(
                type="task_data_bindable_role_mismatch",
                message=(
                    f"Task data {data.id!r} has protocol role {role!r}; only "
                    "model_input or graph_input Task Data "
                    "may be marked bindable"
                ),
                location={"path": f"{data_path}/bindable"},
            )
        )
    if not data.bindable and role in GRAPH_BINDABLE_TASK_DATA_ROLES:
        issues.append(
            SchemaValidationIssue(
                type="task_data_graph_role_not_bindable",
                message=f"Task data {data.id!r} has graph-facing role {role!r} but is not bindable",
                location={"path": f"{data_path}/bindable"},
            )
        )
    if (
        data.bindable
        and (data.kind in PROTOCOL_TASK_DATA_KINDS or task_data_uses_protocol_path(data))
    ):
        issues.append(
            SchemaValidationIssue(
                type="task_data_protocol_path_bindable",
                message=(
                    f"Task data {data.id!r} points at protocol-owned task data; "
                    "targets, initial state, and eval/trial controls "
                    "must stay out of graph input bindings"
                ),
                location={"path": f"{data_path}/path"},
            )
        )
    return issues


def _value_schema_compatibility_issues(
    source: ValueSchema,
    target: ValueSchema,
    *,
    path: str,
    source_label: str,
    target_label: str,
    issue_prefix: str,
) -> list[SchemaValidationIssue]:
    issues: list[SchemaValidationIssue] = []
    known_constraint = False

    source_dtype = source.dtype
    target_dtype = target.dtype
    if source_dtype is not None or target_dtype is not None:
        known_constraint = True
    if (
        source_dtype
        and target_dtype
        and source_dtype != "any"
        and target_dtype != "any"
        and not _dtypes_compatible(source, target)
    ):
        issues.append(
            SchemaValidationIssue(
                type=f"{issue_prefix}_dtype_mismatch",
                message=(
                    f"{source_label} has dtype {source_dtype!r}, but "
                    f"{target_label} expects {target_dtype!r}"
                ),
                location={"path": path},
            )
        )

    source_shape = _compatibility_shape(source)
    target_shape = _compatibility_shape(target)
    source_rank = len(source_shape) if source_shape is not None else source.rank
    target_rank = len(target_shape) if target_shape is not None else target.rank

    if source_rank is not None or target_rank is not None:
        known_constraint = True
    if source_rank is not None and target_rank is not None and source_rank != target_rank:
        issues.append(
            SchemaValidationIssue(
                type=f"{issue_prefix}_rank_mismatch",
                message=(
                    f"{source_label} has rank {source_rank}, but "
                    f"{target_label} expects rank {target_rank}"
                ),
                location={"path": path},
            )
        )

    if source_shape is not None or target_shape is not None:
        known_constraint = True
    if source_shape is not None and target_shape is not None:
        if len(source_shape) != len(target_shape):
            issues.append(
                SchemaValidationIssue(
                    type=f"{issue_prefix}_shape_mismatch",
                    message=(
                        f"{source_label} has shape {source_shape!r}, but "
                        f"{target_label} expects {target_shape!r}"
                    ),
                    location={"path": path},
                )
            )
        else:
            for source_dim, target_dim in zip(source_shape, target_shape):
                if _shape_dim_matches(source_dim, target_dim):
                    continue
                issues.append(
                    SchemaValidationIssue(
                        type=f"{issue_prefix}_shape_mismatch",
                        message=(
                            f"{source_label} has shape {source_shape!r}, but "
                            f"{target_label} expects {target_shape!r}"
                        ),
                        location={"path": path},
                    )
                )
                break

    if not known_constraint or source.origin == "unknown" or target.origin == "unknown":
        issues.append(
            SchemaValidationIssue(
                type=f"{issue_prefix}_unknown_schema",
                message=(
                    f"Compatibility for {source_label} -> {target_label} cannot "
                    "be fully checked from static schema data"
                ),
                severity="warning",
                location={"path": path},
            )
        )
    return issues


def _shape_dim_matches(source_dim: Any, target_dim: Any) -> bool:
    if source_dim in (None, "any", "*", -1) or target_dim in (None, "any", "*", -1):
        return True
    return source_dim == target_dim


def _compatibility_shape(schema: ValueSchema) -> Optional[list[Any]]:
    sample_shape = schema.metadata.get("sample_shape")
    if isinstance(sample_shape, list):
        return sample_shape
    return schema.shape


def _dtypes_compatible(source: ValueSchema, target: ValueSchema) -> bool:
    source_dtype = _normalize_dtype(source.dtype)
    target_dtype = _normalize_dtype(target.dtype)
    if source_dtype is None or target_dtype is None:
        return True
    if source_dtype == target_dtype:
        return True
    if source_dtype == "any" or target_dtype == "any":
        return True
    if _is_concrete_numeric_dtype(source_dtype) and _semantic_numeric_compatible(
        source, target_dtype
    ):
        return True
    if _is_concrete_numeric_dtype(target_dtype) and _semantic_numeric_compatible(
        target, source_dtype
    ):
        return True
    if source_dtype == "state" and target_dtype == "vector":
        return True
    if source_dtype == "vector" and target_dtype == "state":
        return True
    return False


def _normalize_dtype(dtype: Optional[str]) -> Optional[str]:
    return dtype.strip().lower() if dtype else None


def _is_concrete_numeric_dtype(dtype: str) -> bool:
    return dtype == "number" or dtype.startswith(("float", "int", "uint", "complex"))


def _semantic_numeric_compatible(schema: ValueSchema, semantic_dtype: str) -> bool:
    if semantic_dtype == "scalar":
        return _schema_is_scalar(schema)
    if semantic_dtype in {"vector", "matrix", "tensor"}:
        return not _schema_is_scalar(schema)
    return semantic_dtype == "state"


def _schema_is_scalar(schema: ValueSchema) -> bool:
    shape = _compatibility_shape(schema)
    return schema.rank == 0 or shape == []
