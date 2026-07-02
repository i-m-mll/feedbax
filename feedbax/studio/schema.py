"""Static provider-owned schema enumeration for Studio workspaces."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError as PydanticValidationError

from feedbax.contracts.value_schema import SchemaOrigin, ValueSchema
from feedbax.contracts.migrations import migrate_studio_workspace_spec
from feedbax.contracts.manifest import SCHEMA_VERSION, utc_now
from feedbax.studio.protocol import (
    GRAPH_BINDABLE_TASK_DATA_ROLES,
    PROTOCOL_TASK_DATA_KINDS,
    TASK_DATA_ROLES,
    is_bindable_task_data,
    task_data_role,
    task_data_surface,
    task_data_uses_protocol_path,
)
from feedbax.contracts.graphs.normalization import (
    normalize_graph_for_studio_authoring,
    normalize_task_binding_spec_for_studio_authoring,
)
from feedbax.contracts.graphs.prototypes import (
    DerivedDimensionConflict,
    DerivedDimensionError,
    normalize_derived_dimensions,
    prototypes_from_task_bindings,
)
from feedbax.runtime.task_bindings import COMPONENT_PARAMETER_ROLE, task_data_temporality
from feedbax.contracts.component import PortType
from feedbax.contracts.graph import GraphSpec, StudioTaskBindingSpec, StudioWorkspaceSpec


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


class SelectorTargetSchema(StudioSchemaModel):
    """Selectable provider target for objectives, probes, and Studio editors."""

    id: str
    label: str
    kind: Literal[
        "port",
        "edge",
        "graph_output",
        "recurrent_carry",
        "task_data",
        "objective",
        "probe",
        "state_path",
        "state_hint",
        "sample_leaf",
    ]
    selector: str
    value_schema: ValueSchema
    origin: SchemaOrigin = "unknown"
    source: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SchemaValidationIssue(StudioSchemaModel):
    """Validation issue raised while enumerating static Studio schemas."""

    type: str
    message: str
    severity: Literal["error", "warning", "info"] = "error"
    location: Optional[dict[str, str]] = None


class StudioSchemaRegistry(StudioSchemaModel):
    """Static Studio schema registry emitted by the Feedbax provider."""

    kind: Literal["studio_schema_registry"] = "studio_schema_registry"
    schema_version: str = SCHEMA_VERSION
    generated_at: datetime = Field(default_factory=utc_now)
    workspace_id: Optional[str] = None
    scenario_id: Optional[str] = None
    ports: list[PortSchema] = Field(default_factory=list)
    task_data: list[TaskDataSchema] = Field(default_factory=list)
    selector_targets: list[SelectorTargetSchema] = Field(default_factory=list)
    issues: list[SchemaValidationIssue] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RuntimeIntrospectionOptions(StudioSchemaModel):
    """Explicit opt-in options for bounded runtime/sample schema enrichment."""

    enabled: bool = False
    max_targets: int = Field(default=64, ge=1, le=256)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RuntimeSampleLeafSchema(StudioSchemaModel):
    """Schema record returned by an optional provider runtime introspector."""

    path: str
    label: Optional[str] = None
    selector: Optional[str] = None
    kind: str = "sample_leaf"
    dtype: Optional[str] = None
    shape: Optional[list[Any]] = None
    rank: Optional[int] = None
    units: Optional[str] = None
    frame: Optional[str] = None
    value: Any = None
    source: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RuntimeIntrospectionResult(StudioSchemaModel):
    """Result returned by a bounded runtime/sample introspection hook."""

    sample_leaves: list[RuntimeSampleLeafSchema] = Field(default_factory=list)
    issues: list[SchemaValidationIssue] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class StudioSchemaEnumerationRequest(StudioSchemaModel):
    """HTTP request for static Studio schema enumeration."""

    workspace: StudioWorkspaceSpec | dict[str, Any]
    scenario_id: Optional[str] = None
    runtime_introspection: RuntimeIntrospectionOptions | bool | None = None


RuntimeSchemaIntrospector = Callable[
    [StudioWorkspaceSpec, str, RuntimeIntrospectionOptions],
    RuntimeIntrospectionResult | dict[str, Any],
]


def enumerate_studio_schema_registry(
    workspace: StudioWorkspaceSpec | dict[str, Any],
    scenario_id: Optional[str] = None,
    *,
    runtime_introspection: RuntimeIntrospectionOptions | bool | dict[str, Any] | None = None,
    runtime_introspector: Optional[RuntimeSchemaIntrospector] = None,
) -> StudioSchemaRegistry:
    """Enumerate Studio schemas, optionally enriched by an explicit runtime hook.

    The default path is static-only and never compiles or runs JAX code. Runtime
    sample enrichment is opt-in and delegated to a caller-supplied hook so the
    provider contract can expose richer schemas without making basic validation
    depend on full execution.
    """

    try:
        workspace_spec = (
            workspace
            if isinstance(workspace, StudioWorkspaceSpec)
            else StudioWorkspaceSpec.model_validate(
                migrate_studio_workspace_spec(workspace).payload
            )
        )
    except PydanticValidationError as exc:
        return StudioSchemaRegistry(
            scenario_id=scenario_id,
            issues=[
                SchemaValidationIssue(
                    type="workspace_schema_error",
                    message=str(error.get("msg", "Invalid workspace value")),
                    location={"path": "/" + "/".join(str(part) for part in error.get("loc", ()))},
                )
                for error in exc.errors()
            ],
        )
    except ValueError as exc:
        return StudioSchemaRegistry(
            scenario_id=scenario_id,
            issues=[
                SchemaValidationIssue(
                    type="workspace_schema_version_error",
                    message=str(exc),
                    location={"path": "/workspace/schema_version"},
                )
            ],
        )

    selected_scenario_id = scenario_id or _active_scenario_id(workspace_spec)
    issues = _workspace_reference_issues(workspace_spec)
    registry = StudioSchemaRegistry(
        workspace_id=workspace_spec.id,
        scenario_id=selected_scenario_id,
        issues=issues,
        metadata={
            "enumerated_by": "feedbax.studio.schema",
            "requires_jax_execution": False,
            "runtime_introspection": _runtime_introspection_metadata(
                runtime_introspection,
                status="not_requested",
            ),
        },
    )

    if selected_scenario_id is None:
        registry.issues.append(
            SchemaValidationIssue(
                type="missing_scenario_id",
                message="No scenario_id was supplied and no active stage scenario is available",
                location={"path": "/scenario_id"},
            )
        )
        return registry

    scenario = workspace_spec.scenarios.get(selected_scenario_id)
    if scenario is None:
        registry.issues.append(
            SchemaValidationIssue(
                type="missing_scenario",
                message=f"Scenario {selected_scenario_id!r} does not exist in the workspace",
                location={"path": f"/scenarios/{selected_scenario_id}"},
            )
        )
        return registry

    schema_graph: Optional[GraphSpec] = None
    task_binding_spec = scenario.task_binding_spec
    if scenario.graph is None:
        registry.issues.append(
            SchemaValidationIssue(
                type="missing_graph",
                message=f"Scenario {scenario.id!r} does not have a graph",
                location={"path": f"/scenarios/{scenario.id}/graph"},
            )
        )
    else:
        authoring_graph = normalize_graph_for_studio_authoring(scenario.graph)
        task_binding_spec = normalize_task_binding_spec_for_studio_authoring(
            task_binding_spec,
            authoring_graph,
        )
        schema_graph = _normalize_dynamic_graph_ports(
            authoring_graph,
            task_binding_spec,
        )
        schema_graph = _normalize_derived_dimensions_for_schema(
            schema_graph,
            task_binding_spec,
        )
        registry.ports = _enumerate_graph_ports(schema_graph, task_binding_spec)
        registry.selector_targets.extend(_port_selector_targets(registry.ports))
        registry.selector_targets.extend(_graph_structural_selector_targets(schema_graph))
        registry.selector_targets.extend(_graph_probe_selector_targets(schema_graph))
        registry.issues.extend(
            validate_graph_connection_schema(
                schema_graph,
                f"/scenarios/{scenario.id}/graph",
                task_binding_spec,
            )
        )

    if task_binding_spec is None:
        registry.issues.append(
            SchemaValidationIssue(
                type="missing_task_binding_spec",
                message=f"Scenario {scenario.id!r} does not have a task_binding_spec",
                severity="warning",
                location={"path": f"/scenarios/{scenario.id}/task_binding_spec"},
            )
        )
    else:
        registry.task_data = _enumerate_task_data(task_binding_spec)
        registry.selector_targets.extend(_task_data_selector_targets(registry.task_data))
        registry.issues.extend(
            validate_task_binding_schema(
                task_binding_spec,
                schema_graph,
                f"/scenarios/{scenario.id}/task_binding_spec",
            )
        )
        _mark_bound_ports(registry.ports, task_binding_spec)

    registry.selector_targets.extend(_objective_selector_targets(scenario.objective_spec))
    registry.selector_targets.extend(_explicit_probe_selector_targets(scenario.probe_specs))
    _apply_runtime_introspection(
        registry,
        workspace_spec,
        scenario.id,
        runtime_introspection,
        runtime_introspector,
    )
    registry.selector_targets.extend(_known_state_hint_targets())
    registry.selector_targets = _dedupe_selector_targets(registry.selector_targets)
    if scenario.graph is not None:
        registry.issues.extend(
            validate_intervention_schema(
                scenario.graph,
                registry.selector_targets,
                f"/scenarios/{scenario.id}/graph",
            )
        )
    return registry


def validate_graph_connection_schema(
    graph: GraphSpec,
    base_path: str = "",
    task_binding_spec: Optional[StudioTaskBindingSpec] = None,
) -> list[SchemaValidationIssue]:
    """Validate graph wires against provider-owned port schema records."""

    ports = _enumerate_graph_ports(graph)
    by_node_port_direction = {
        (port.node_id, port.port, port.direction): port
        for port in ports
        if port.node_id is not None
    }
    by_node_port = {(port.node_id, port.port): port for port in ports if port.node_id is not None}
    occupied: dict[tuple[str, str], int] = {}
    for binding in graph.input_bindings.values():
        occupied[(binding[0], binding[1])] = -1

    issues: list[SchemaValidationIssue] = []
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

        target_key = (wire.target_node, wire.target_port)
        previous = occupied.get(target_key)
        if previous is not None:
            location = (
                f"{base_path}/input_bindings" if previous < 0 else f"{base_path}/wires/{previous}"
            )
            issues.append(
                SchemaValidationIssue(
                    type="graph_input_occupied",
                    message=f"Input {wire.target_node}.{wire.target_port} is already occupied",
                    location={"path": wire_path, "occupied_by": location},
                )
            )
        occupied[target_key] = index

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
    issues.extend(_derived_dimension_issues(graph, task_binding_spec, base_path))
    issues.extend(_instant_cycle_issues(graph, base_path))
    return issues


def _normalize_derived_dimensions_for_schema(
    graph: GraphSpec,
    task_binding_spec: Optional[StudioTaskBindingSpec],
) -> GraphSpec:
    if not graph.derived_dimensions and not graph.subgraphs:
        return graph
    try:
        return normalize_derived_dimensions(
            graph,
            prototypes_from_task_bindings(task_binding_spec) if task_binding_spec else {},
        )
    except DerivedDimensionError:
        return graph


def _derived_dimension_issues(
    graph: GraphSpec,
    task_binding_spec: Optional[StudioTaskBindingSpec],
    base_path: str,
) -> list[SchemaValidationIssue]:
    if not graph.derived_dimensions and not graph.subgraphs:
        return []
    try:
        normalize_derived_dimensions(
            graph,
            prototypes_from_task_bindings(task_binding_spec) if task_binding_spec else {},
        )
    except DerivedDimensionConflict as exc:
        return [
            SchemaValidationIssue(
                type="derived_dimension_conflict",
                message=str(exc),
                location={
                    "path": f"{base_path}/derived_dimensions/{_derived_dimension_index(graph, exc)}",
                    "node": exc.rule.node,
                    "param": exc.rule.param,
                    "declared": str(exc.declared),
                    "derived": str(exc.derived),
                },
            )
        ]
    except DerivedDimensionError as exc:
        return [
            SchemaValidationIssue(
                type="derived_dimension_unresolved",
                message=str(exc),
                location={
                    "path": f"{base_path}/derived_dimensions/{_derived_dimension_index(graph, exc)}",
                    "node": exc.rule.node,
                    "param": exc.rule.param,
                },
            )
        ]
    return []


def _derived_dimension_index(graph: GraphSpec, exc: DerivedDimensionError) -> int:
    for index, rule in enumerate(graph.derived_dimensions):
        if rule == exc.rule:
            return index
    return 0


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
        {
            (binding.target_node_id, binding.target_port)
            for binding in task_binding_spec.bindings
        }
        if task_binding_spec is not None
        else set()
    )
    occupied_ports = graph_input_ports | wired_ports | task_bound_ports

    for node_id, node in graph.nodes.items():
        if node.type != "Mux":
            continue
        connected_inputs = {
            port
            for port in node.input_ports
            if (node_id, port) in occupied_ports
        }
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


def _runtime_introspection_options(
    value: RuntimeIntrospectionOptions | bool | dict[str, Any] | None,
) -> RuntimeIntrospectionOptions:
    if isinstance(value, RuntimeIntrospectionOptions):
        return value
    if isinstance(value, bool):
        return RuntimeIntrospectionOptions(enabled=value)
    if isinstance(value, dict):
        return RuntimeIntrospectionOptions.model_validate(value)
    return RuntimeIntrospectionOptions(enabled=False)


def _runtime_introspection_metadata(
    value: RuntimeIntrospectionOptions | bool | dict[str, Any] | None,
    *,
    status: str,
) -> dict[str, Any]:
    options = _runtime_introspection_options(value)
    return {
        "enabled": options.enabled,
        "status": status if options.enabled else "not_requested",
        "max_targets": options.max_targets,
    }


def _apply_runtime_introspection(
    registry: StudioSchemaRegistry,
    workspace: StudioWorkspaceSpec,
    scenario_id: str,
    requested: RuntimeIntrospectionOptions | bool | dict[str, Any] | None,
    runtime_introspector: Optional[RuntimeSchemaIntrospector],
) -> None:
    options = _runtime_introspection_options(requested)
    if not options.enabled:
        registry.metadata["runtime_introspection"] = _runtime_introspection_metadata(
            options,
            status="not_requested",
        )
        return

    if runtime_introspector is None:
        registry.metadata["runtime_introspection"] = _runtime_introspection_metadata(
            options,
            status="unavailable",
        )
        registry.issues.append(
            SchemaValidationIssue(
                type="runtime_introspection_unavailable",
                message=(
                    "Runtime sample introspection was requested, but no provider "
                    "introspector is configured"
                ),
                severity="warning",
                location={"path": "/runtime_introspection"},
            )
        )
        return

    try:
        result = RuntimeIntrospectionResult.model_validate(
            runtime_introspector(workspace, scenario_id, options)
        )
    except Exception as exc:
        registry.metadata["runtime_introspection"] = _runtime_introspection_metadata(
            options,
            status="failed",
        )
        registry.issues.append(
            SchemaValidationIssue(
                type="runtime_introspection_failed",
                message=f"Runtime sample introspection failed: {exc}",
                severity="warning",
                location={"path": "/runtime_introspection"},
            )
        )
        return

    leaves = result.sample_leaves[: options.max_targets]
    registry.selector_targets.extend(_runtime_sample_selector_targets(leaves))
    registry.issues.extend(result.issues)
    registry.metadata["runtime_introspection"] = {
        **_runtime_introspection_metadata(options, status="completed"),
        "target_count": len(leaves),
        "truncated": len(result.sample_leaves) > len(leaves),
        **result.metadata,
    }


def _active_scenario_id(workspace: StudioWorkspaceSpec) -> Optional[str]:
    if workspace.active_stage_id is not None:
        active = next(
            (stage for stage in workspace.stages if stage.id == workspace.active_stage_id),
            None,
        )
        if active is not None and active.scenario_id is not None:
            return active.scenario_id
    first_stage = next((stage for stage in workspace.stages if stage.scenario_id), None)
    return first_stage.scenario_id if first_stage is not None else None


def _workspace_reference_issues(workspace: StudioWorkspaceSpec) -> list[SchemaValidationIssue]:
    issues: list[SchemaValidationIssue] = []
    for index, stage in enumerate(workspace.stages):
        if stage.scenario_id is None:
            continue
        if stage.scenario_id not in workspace.scenarios:
            issues.append(
                SchemaValidationIssue(
                    type="stage_missing_scenario",
                    message=(
                        f"Stage {stage.id!r} references missing scenario {stage.scenario_id!r}"
                    ),
                    location={"path": f"/stages/{index}/scenario_id"},
                )
            )
    return issues


def _mux_input_index(port_name: str) -> Optional[int]:
    if not port_name.startswith("in_"):
        return None
    suffix = port_name.removeprefix("in_")
    if not suffix.isdigit():
        return None
    return int(suffix)


def _normalize_dynamic_graph_ports(
    graph: GraphSpec,
    task_binding_spec: Optional[StudioTaskBindingSpec] = None,
) -> GraphSpec:
    changed = False
    nodes = dict(graph.nodes)
    for node_id, node in graph.nodes.items():
        if node.type == "Demux":
            sizes = node.params.get("sizes")
            if isinstance(sizes, list) and all(isinstance(size, int) and size > 0 for size in sizes):
                output_ports = [f"out_{index}" for index in range(len(sizes))]
                if node.input_ports == ["input"] and node.output_ports == output_ports:
                    continue
                nodes[node_id] = node.model_copy(
                    update={
                        "input_ports": node.input_ports or ["input"],
                        "output_ports": output_ports,
                    }
                )
                changed = True
            continue
        if node.type != "Mux":
            continue
        max_index = -1
        for wire in graph.wires:
            if wire.target_node != node_id:
                continue
            index = _mux_input_index(wire.target_port)
            if index is not None:
                max_index = max(max_index, index)
        for binding in task_binding_spec.bindings if task_binding_spec is not None else []:
            if binding.target_node_id != node_id:
                continue
            index = _mux_input_index(binding.target_port)
            if index is not None:
                max_index = max(max_index, index)
        required_count = max(2, max_index + 1)
        input_ports = [f"in_{index}" for index in range(required_count)]
        if node.input_ports == input_ports and node.params.get("n_inputs") == required_count:
            continue
        params = dict(node.params)
        params["n_inputs"] = required_count
        nodes[node_id] = node.model_copy(
            update={
                "params": params,
                "input_ports": input_ports,
                "output_ports": node.output_ports or ["output"],
            }
        )
        changed = True
    return graph.model_copy(update={"nodes": nodes}) if changed else graph


def _enumerate_graph_ports(
    graph: GraphSpec,
    task_binding_spec: Optional[StudioTaskBindingSpec] = None,
) -> list[PortSchema]:
    from feedbax.component_registry import ComponentRegistry

    registry = ComponentRegistry()
    ports: list[PortSchema] = []
    for node_id, node in graph.nodes.items():
        meta = registry.get(node.type)
        subgraph = graph.subgraphs.get(node_id) if graph.subgraphs else None
        input_ports = list(node.input_ports or (meta.input_ports if meta is not None else []))
        output_ports = list(node.output_ports or (meta.output_ports if meta is not None else []))
        for port_name in input_ports:
            port_type = _component_input_port_type(meta, node.type, port_name)
            ports.append(
                _port_schema(
                    node_id=node_id,
                    component_type=node.type,
                    node_params=node.params,
                    port=port_name,
                    direction="input",
                    port_type=port_type,
                    subgraph=subgraph,
                )
            )
        for port_name in output_ports:
            port_type = _component_output_port_type(meta, node.type, port_name)
            ports.append(
                _port_schema(
                    node_id=node_id,
                    component_type=node.type,
                    node_params=node.params,
                    port=port_name,
                    direction="output",
                    port_type=port_type,
                    subgraph=subgraph,
                )
            )
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
    meta: Any, component_type: str, port_name: str
) -> Optional[PortType]:
    if meta is None or meta.port_types is None:
        return None
    port_type = meta.port_types.inputs.get(port_name)
    if (
        port_type is None
        and component_type == "Mux"
        and port_name.startswith("in_")
        and port_name.removeprefix("in_").isdigit()
    ):
        port_type = meta.port_types.inputs.get("in_0") or meta.port_types.inputs.get("in_1")
    return port_type


def _component_output_port_type(
    meta: Any, component_type: str, port_name: str
) -> Optional[PortType]:
    if meta is None or meta.port_types is None:
        return None
    port_type = meta.port_types.outputs.get(port_name)
    if (
        port_type is None
        and component_type == "Demux"
        and port_name.startswith("out_")
        and port_name.removeprefix("out_").isdigit()
    ):
        port_type = meta.port_types.outputs.get("out_0") or meta.port_types.outputs.get("out_1")
    return port_type


def _value_schema_for_graph_port(
    graph: GraphSpec,
    node_id: str,
    port: str,
    direction: Literal["input", "output"],
) -> ValueSchema:
    from feedbax.component_registry import ComponentRegistry

    node = graph.nodes.get(node_id)
    if node is None:
        return ValueSchema(
            id=f"value:missing:{node_id}.{port}:{direction}",
            label=f"{node_id}.{port}",
            kind="unknown",
            origin="unknown",
        )
    registry = ComponentRegistry()
    meta = registry.get(node.type)
    subgraph = graph.subgraphs.get(node_id) if graph.subgraphs else None
    port_type = (
        _component_input_port_type(meta, node.type, port)
        if direction == "input"
        else _component_output_port_type(meta, node.type, port)
    )
    return _port_schema(
        node_id=node_id,
        component_type=node.type,
        node_params=node.params,
        port=port,
        direction=direction,
        port_type=port_type,
        subgraph=subgraph,
    ).value_schema


def _port_schema(
    *,
    node_id: str,
    component_type: str,
    node_params: dict[str, Any],
    port: str,
    direction: Literal["input", "output"],
    port_type: Optional[PortType],
    subgraph: Optional[GraphSpec] = None,
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
    boundary = _subgraph_boundary_value_schema(subgraph, port, direction)
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
    if component_type == "Network":
        if direction == "output" and port == "output":
            dimension_param = "out_size"
        elif direction == "output" and port == "hidden":
            dimension_param = "hidden_size"
    elif component_type in {"Linear", "MLP"}:
        if direction == "input" and port == "input":
            dimension_param = "input_size"
        elif direction == "output" and port == "output":
            dimension_param = "output_size"
    elif component_type in {"GRU", "LSTM"}:
        if direction == "input" and port == "input":
            dimension_param = "input_size"
        elif port in {"output", "hidden", "cell"}:
            dimension_param = "hidden_size"
    elif component_type in {"TwoLinkArm", "PointMass", "Mechanics"} and port == "force":
        dimension = 2
    elif component_type == "Arm6MuscleRigidTendon":
        if port == "excitation":
            dimension = 6
        elif port in {"angles", "angular_velocities", "torques"}:
            dimension = 2
        elif port in {"forces", "activations"}:
            dimension = 6
    elif component_type == "PointMass8MuscleRelu":
        if port == "excitation":
            n_pairs = _positive_int_param(node_params, "n_pairs")
            dimension = n_pairs * 2 if n_pairs is not None else None
        elif port == "force_2d":
            dimension = 2
        elif port in {"forces", "activations"}:
            n_pairs = _positive_int_param(node_params, "n_pairs")
            dimension = n_pairs * 2 if n_pairs is not None else None
    elif component_type == "FeedbackChannels" and direction == "output" and port == "output":
        dimension = _feedback_channels_width(node_params)
    elif component_type == "Demux":
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


def _feedback_channels_width(params: dict[str, Any]) -> Optional[int]:
    channels = params.get("channels")
    if not isinstance(channels, list):
        return None
    width = 0
    for channel in channels:
        if channel in {"position", "velocity", "force"}:
            width += 2
    return width if width > 0 else None


def _positive_int_param(params: dict[str, Any], key: str) -> Optional[int]:
    value = params.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    return None


def _subgraph_boundary_value_schema(
    subgraph: Optional[GraphSpec],
    port: str,
    direction: Literal["input", "output"],
) -> Optional[ValueSchema]:
    if subgraph is None:
        return None
    binding = (
        subgraph.input_bindings.get(port)
        if direction == "input"
        else subgraph.output_bindings.get(port)
    )
    if binding is None:
        return None
    node_id, bound_port = binding
    for schema in _enumerate_graph_ports(subgraph):
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
        if port.component_type != "Mux" or port.direction != "output" or port.port != "output":
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


def _mark_bound_ports(
    ports: list[PortSchema],
    task_binding_spec: StudioTaskBindingSpec,
) -> None:
    task_data_ids = {data.id: f"task_data:{data.id}" for data in task_binding_spec.exposed_data}
    by_target = {
        (port.node_id, port.port): port
        for port in ports
        if port.direction == "input" and port.node_id is not None
    }
    for binding in task_binding_spec.bindings:
        port = by_target.get((binding.target_node_id, binding.target_port))
        if port is not None:
            port.bound_task_data_id = task_data_ids.get(binding.source_data_id)


def validate_task_binding_schema(
    task_binding_spec: StudioTaskBindingSpec,
    graph: Optional[GraphSpec],
    base_path: str,
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
        for port in _enumerate_graph_ports(graph, task_binding_spec)
        if port.direction == "input" and port.node_id is not None
    }
    binding_targets: set[tuple[str, str]] = set()
    for index, binding in enumerate(task_binding_spec.bindings):
        binding_path = f"{base_path}/bindings/{index}"
        expected_id = (
            f"task:{binding.source_data_id}->{binding.target_node_id}:{binding.target_port}"
        )
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
            if data is not None and task_data_role(data) == COMPONENT_PARAMETER_ROLE:
                issues.extend(
                    _component_parameter_binding_issues(
                        data,
                        target_node,
                        binding,
                        binding_path,
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
                    "model_input, graph_input, or component_parameter Task Data "
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
    if role != COMPONENT_PARAMETER_ROLE and data.bindable and (
        data.kind in PROTOCOL_TASK_DATA_KINDS or task_data_uses_protocol_path(data)
    ):
        issues.append(
            SchemaValidationIssue(
                type="task_data_protocol_path_bindable",
                message=(
                    f"Task data {data.id!r} points at protocol-owned task data; "
                    "targets, initial state, interventions, and eval/trial controls "
                    "must stay out of graph input bindings"
                ),
                location={"path": f"{data_path}/path"},
            )
        )
    return issues


def _component_parameter_binding_issues(
    data: Any,
    target_node: Any,
    binding: Any,
    binding_path: str,
) -> list[SchemaValidationIssue]:
    issues: list[SchemaValidationIssue] = []
    temporality = task_data_temporality(data)
    if temporality == "constant":
        labels = _static_task_parameter_labels(binding.target_node_id, target_node)
        label = _task_parameter_label(binding)
        if not labels:
            issues.append(
                SchemaValidationIssue(
                    type="component_parameter_declaration_unknown",
                    message=(
                        f"Task binding {binding.id!r} is constant within trial, but "
                        f"{binding.target_node_id!r} has no static task-parameter "
                        "declaration visible to Studio validation"
                    ),
                    location={"path": binding_path},
                )
            )
        elif label not in labels:
            issues.append(
                SchemaValidationIssue(
                    type="component_parameter_label_unknown",
                    message=(
                        f"Task binding {binding.id!r} references task-parameter "
                        f"label {label!r}; declared labels are {sorted(labels)!r}"
                    ),
                    location={"path": f"{binding_path}/metadata/task_parameter_label"},
                )
            )
    elif binding.target_port == "params_override":
        labels = _static_task_parameter_labels(binding.target_node_id, target_node)
        if not labels:
            issues.append(
                SchemaValidationIssue(
                    type="component_parameter_declaration_unknown",
                    message=(
                        f"Task binding {binding.id!r} targets params_override, but "
                        f"{binding.target_node_id!r} has no static task-parameter "
                        "declaration visible to Studio validation"
                    ),
                    location={"path": binding_path},
                )
            )
    return issues


def _task_parameter_label(binding: Any) -> str:
    metadata = getattr(binding, "metadata", None)
    if isinstance(metadata, dict):
        label = metadata.get("task_parameter_label")
        if isinstance(label, str) and label:
            return label
    return str(binding.source_data_id)


def _static_task_parameter_labels(node_id: str, node: Any) -> set[str]:
    if "label" not in getattr(node, "params", {}):
        return set()
    label = node.params.get("label") or node_id
    return {str(label)}


def validate_intervention_schema(
    graph: GraphSpec,
    selector_targets: list[SelectorTargetSchema],
    base_path: str = "",
) -> list[SchemaValidationIssue]:
    """Validate typed intervention tap semantics against selector target schemas."""

    issues: list[SchemaValidationIssue] = []
    by_selector = {target.selector: target for target in selector_targets}
    by_id = {target.id: target for target in selector_targets}
    for index, tap in enumerate(graph.taps or []):
        if tap.type != "intervention":
            continue
        path = f"{base_path}/taps/{index}/transform/intervention"
        intervention = tap.transform.intervention if tap.transform is not None else None
        if intervention is None:
            issues.append(
                SchemaValidationIssue(
                    type="intervention_missing_spec",
                    message=f"Intervention tap {tap.id!r} needs a typed target and operation",
                    location={"path": path},
                )
            )
            continue

        operation = intervention.operation
        if operation not in {"clamp", "noise", "constant", "offset", "scale"}:
            issues.append(
                SchemaValidationIssue(
                    type="intervention_unknown_operation",
                    message=(
                        f"Intervention operation {operation!r} is not supported "
                        "by schema validation"
                    ),
                    location={"path": f"{path}/operation"},
                )
            )
            continue

        target_selector = intervention.target_selector
        schema_target_id = None
        if target_selector is not None:
            schema_target_id = target_selector.metadata.get("schema_target_id")
        target = by_id.get(schema_target_id) if isinstance(schema_target_id, str) else None
        if target is None and target_selector is not None:
            target = by_selector.get(target_selector.compact)
        if target is None:
            compact = target_selector.compact if target_selector is not None else None
            issues.append(
                SchemaValidationIssue(
                    type="intervention_unknown_target",
                    message=(
                        f"Intervention target {compact!r} is not in the selector schema registry"
                    ),
                    location={"path": f"{path}/target_selector"},
                )
            )
            continue

        issues.extend(
            _intervention_numeric_schema_issues(
                operation,
                target.value_schema,
                path,
                target.label,
            )
        )

        if operation == "clamp" and not _has_clamp_bound(intervention.bounds):
            issues.append(
                SchemaValidationIssue(
                    type="intervention_missing_bounds",
                    message=f"Clamp intervention for {target.label} needs at least one bound",
                    location={"path": f"{path}/bounds"},
                )
            )
        if operation in {"constant", "offset", "scale"} and intervention.value is None:
            issues.append(
                SchemaValidationIssue(
                    type="intervention_missing_value",
                    message=f"{operation} intervention for {target.label} needs a value spec",
                    location={"path": f"{path}/value"},
                )
            )
        if operation == "noise" and not _has_noise_scale(intervention):
            issues.append(
                SchemaValidationIssue(
                    type="intervention_missing_noise_scale",
                    message=f"Noise intervention for {target.label} needs a scale or std value",
                    location={"path": f"{path}/value"},
                )
            )

        if intervention.value is not None:
            compatible = _intervention_shape_compatible(
                intervention.value.shape,
                target.value_schema.shape,
            )
            if compatible is False:
                issues.append(
                    SchemaValidationIssue(
                        type="intervention_value_shape_mismatch",
                        message=(
                            f"Intervention value shape {intervention.value.shape!r} "
                            f"does not match {target.label} shape "
                            f"{target.value_schema.shape!r}"
                        ),
                        location={"path": f"{path}/value/shape"},
                    )
                )
            elif (
                compatible is None
                and operation in {"constant", "offset"}
                and target.value_schema.shape is not None
            ):
                issues.append(
                    SchemaValidationIssue(
                        type="intervention_value_unknown_shape",
                        message=(
                            f"Intervention value for {target.label} cannot be "
                            "shape-checked from static schema data"
                        ),
                        severity="warning",
                        location={"path": f"{path}/value"},
                    )
                )
    return issues


def _intervention_numeric_schema_issues(
    operation: str,
    schema: ValueSchema,
    path: str,
    label: str,
) -> list[SchemaValidationIssue]:
    numeric = _is_numeric_schema(schema)
    if numeric is False:
        return [
            SchemaValidationIssue(
                type="intervention_target_dtype_mismatch",
                message=(
                    f"{operation} intervention target {label} has non-numeric "
                    f"dtype {schema.dtype!r}"
                ),
                location={"path": path},
            )
        ]
    if numeric is None or schema.origin == "unknown":
        return [
            SchemaValidationIssue(
                type="intervention_target_unknown_schema",
                message=(
                    f"{operation} intervention target {label} cannot be fully "
                    "checked from static schema data"
                ),
                severity="warning",
                location={"path": path},
            )
        ]
    return []


def _is_numeric_schema(schema: ValueSchema) -> Optional[bool]:
    if schema.dtype is None:
        return None
    dtype = schema.dtype.lower()
    if dtype in {
        "float",
        "float16",
        "float32",
        "float64",
        "int",
        "int16",
        "int32",
        "int64",
        "uint",
        "uint16",
        "uint32",
        "uint64",
        "number",
        "scalar",
        "vector",
        "array",
        "tensor",
    }:
        return True
    if "float" in dtype or "int" in dtype:
        return True
    return False


def _has_clamp_bound(bounds: Any) -> bool:
    return bounds is not None and (bounds.min is not None or bounds.max is not None)


def _has_noise_scale(intervention: Any) -> bool:
    parameters = intervention.parameters or {}
    return (
        intervention.value is not None
        or parameters.get("scale") is not None
        or parameters.get("std") is not None
        or parameters.get("noise_std") is not None
    )


def _intervention_shape_compatible(
    source_shape: Optional[list[Any]],
    target_shape: Optional[list[Any]],
) -> Optional[bool]:
    if source_shape is None or target_shape is None:
        return None
    if len(source_shape) != len(target_shape):
        return False
    return all(
        _shape_dim_matches(source_dim, target_dim)
        for source_dim, target_dim in zip(source_shape, target_shape)
    )


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


def _port_selector_targets(ports: list[PortSchema]) -> list[SelectorTargetSchema]:
    targets: list[SelectorTargetSchema] = []
    for port in ports:
        if port.node_id is None:
            continue
        selector = f"port:{port.node_id}.{port.port}"
        targets.append(
            SelectorTargetSchema(
                id=f"selector:{selector}",
                label=port.label,
                kind="port",
                selector=selector,
                value_schema=port.value_schema,
                origin=port.origin,
                source={"port_id": port.id, "direction": port.direction},
                metadata={
                    "retention_target": {
                        "kind": "port",
                        "selector": selector,
                        "node_id": port.node_id,
                        "port": port.port,
                        "timing": port.direction,
                    },
                    "default_retention": {"mode": "trajectory"},
                },
            )
        )
    return targets


def _graph_structural_selector_targets(graph: GraphSpec) -> list[SelectorTargetSchema]:
    targets: list[SelectorTargetSchema] = []
    for output_name, binding in graph.output_bindings.items():
        node_id, port = binding
        selector = f"graph_output:{output_name}"
        port_schema = _value_schema_for_graph_port(graph, node_id, port, "output")
        targets.append(
            SelectorTargetSchema(
                id=f"selector:{selector}",
                label=f"Graph output {output_name}",
                kind="graph_output",
                selector=selector,
                value_schema=port_schema,
                origin=port_schema.origin,
                source={"output_name": output_name, "node_id": node_id, "port": port},
                metadata={
                    "retention_target": {
                        "kind": "graph_output",
                        "selector": selector,
                        "node_id": node_id,
                        "port": port,
                    },
                    "default_retention": {"mode": "trajectory"},
                },
            )
        )
    for index, wire in enumerate(graph.wires):
        edge_selector = (
            f"edge:{wire.source_node}.{wire.source_port}->"
            f"{wire.target_node}.{wire.target_port}"
        )
        source_schema = _value_schema_for_graph_port(
            graph,
            wire.source_node,
            wire.source_port,
            "output",
        )
        kind: Literal["edge", "recurrent_carry"] = (
            "recurrent_carry" if wire.temporality == "recurrent" else "edge"
        )
        targets.append(
            SelectorTargetSchema(
                id=f"selector:{edge_selector}",
                label=(
                    f"{wire.source_node}.{wire.source_port} -> "
                    f"{wire.target_node}.{wire.target_port}"
                ),
                kind=kind,
                selector=edge_selector,
                value_schema=source_schema,
                origin=source_schema.origin,
                source={
                    "wire_index": index,
                    "source_node": wire.source_node,
                    "source_port": wire.source_port,
                    "target_node": wire.target_node,
                    "target_port": wire.target_port,
                    "temporality": wire.temporality,
                },
                metadata={
                    "retention_target": {
                        "kind": kind,
                        "selector": edge_selector,
                        "edge_id": edge_selector,
                        "node_id": wire.source_node,
                        "port": wire.source_port,
                        "timing": "step",
                    },
                    "default_retention": {
                        "mode": "window" if wire.temporality == "recurrent" else "trajectory",
                        "window_size": 1 if wire.temporality == "recurrent" else None,
                    },
                },
            )
        )
    if graph.retained_observables:
        for observable in graph.retained_observables:
            selector = _retained_observable_selector(observable)
            targets.append(_selector_target_from_retained_observable(observable, selector))
    return targets


def _task_data_selector_targets(task_data: list[TaskDataSchema]) -> list[SelectorTargetSchema]:
    return [
        SelectorTargetSchema(
            id=f"selector:task_data:{item.id.removeprefix('task_data:')}",
            label=item.label,
            kind="task_data",
            selector=f"task_data:{item.path}",
            value_schema=item.value_schema,
            origin=item.origin,
            source={"task_data_id": item.id},
        )
        for item in task_data
    ]


def _runtime_sample_selector_targets(
    leaves: list[RuntimeSampleLeafSchema],
) -> list[SelectorTargetSchema]:
    targets: list[SelectorTargetSchema] = []
    for leaf in leaves:
        selector = leaf.selector or _runtime_leaf_selector(leaf.path)
        label = leaf.label or leaf.path
        inferred = _infer_value_schema_from_sample(leaf.value)
        dtype = leaf.dtype or inferred.get("dtype")
        shape = leaf.shape or inferred.get("shape")
        rank = leaf.rank if leaf.rank is not None else inferred.get("rank")
        targets.append(
            SelectorTargetSchema(
                id=f"selector:{selector}",
                label=label,
                kind="sample_leaf",
                selector=selector,
                value_schema=ValueSchema(
                    id=f"value:{selector}",
                    label=label,
                    kind=leaf.kind,
                    dtype=dtype,
                    shape=shape,
                    rank=rank,
                    units=leaf.units,
                    frame=leaf.frame,
                    origin="runtime_sample",
                    metadata={
                        **leaf.metadata,
                        "sample_path": leaf.path,
                        "runtime_introspection": True,
                    },
                ),
                origin="runtime_sample",
                source={
                    "runtime_introspection": True,
                    "path": leaf.path,
                    **leaf.source,
                },
                metadata=leaf.metadata,
            )
        )
    return targets


def _runtime_leaf_selector(path: str) -> str:
    if path.startswith(("path:", "task_data:", "port:", "probe:")):
        return path
    return f"path:{path}"


def _infer_value_schema_from_sample(value: Any) -> dict[str, Any]:
    if value is None:
        return {}

    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None:
        shape_list = [int(dim) if isinstance(dim, int) else dim for dim in list(shape)]
        return {
            "dtype": str(dtype) if dtype is not None else None,
            "shape": shape_list,
            "rank": len(shape_list),
        }

    if isinstance(value, bool):
        return {"dtype": "bool", "shape": [], "rank": 0}
    if isinstance(value, int):
        return {"dtype": "int", "shape": [], "rank": 0}
    if isinstance(value, float):
        return {"dtype": "float", "shape": [], "rank": 0}
    if isinstance(value, str):
        return {"dtype": "string", "shape": [], "rank": 0}
    if isinstance(value, list):
        nested = _infer_list_shape(value)
        return {"dtype": "array", "shape": nested, "rank": len(nested)}
    return {}


def _infer_list_shape(value: list[Any]) -> list[Any]:
    if not value:
        return [0]
    first = value[0]
    if isinstance(first, list):
        return [len(value), *_infer_list_shape(first)]
    return [len(value)]


def _graph_probe_selector_targets(graph: GraphSpec) -> list[SelectorTargetSchema]:
    from feedbax.objectives.service import loss_service

    targets: list[SelectorTargetSchema] = []
    for probe in loss_service.get_available_probes(graph):
        targets.append(
            SelectorTargetSchema(
                id=f"selector:{probe.selector}",
                label=probe.label,
                kind="probe" if probe.selector.startswith("probe:") else "port",
                selector=probe.selector,
                value_schema=ValueSchema(
                    id=f"value:{probe.selector}",
                    label=probe.label,
                    kind="probe",
                    origin="inferred_static",
                    metadata={"node": probe.node, "timing": probe.timing},
                ),
                origin="inferred_static",
                source={
                    "probe_id": probe.id,
                    "node_id": probe.node,
                    "timing": probe.timing,
                    "description": probe.description,
                },
            )
        )
    return targets


def _objective_selector_targets(
    objective_spec: Optional[dict[str, Any]],
) -> list[SelectorTargetSchema]:
    if not objective_spec:
        return []
    targets: list[SelectorTargetSchema] = []
    selectors = _collect_selector_strings(objective_spec)
    for selector in selectors:
        targets.append(
            SelectorTargetSchema(
                id=f"selector:objective:{selector}",
                label=selector,
                kind="objective",
                selector=selector,
                value_schema=ValueSchema(
                    id=f"value:objective:{selector}",
                    label=selector,
                    kind="objective",
                    origin="inferred_static",
                ),
                origin="inferred_static",
                source={"objective_spec": True},
            )
        )
    return targets


def _explicit_probe_selector_targets(
    probe_specs: list[Any],
) -> list[SelectorTargetSchema]:
    targets: list[SelectorTargetSchema] = []
    for index, probe in enumerate(probe_specs):
        probe_dict = (
            probe.model_dump(mode="json", exclude_none=True)
            if hasattr(probe, "model_dump")
            else probe
        )
        if not isinstance(probe_dict, dict):
            continue
        selector = _retained_observable_selector(probe_dict)
        if not isinstance(selector, str) or not selector:
            continue
        target = _selector_target_from_retained_observable(probe_dict, selector)
        target.source["probe_specs_index"] = index
        targets.append(target)
    return targets


def _retained_observable_selector(observable: Any) -> str:
    if hasattr(observable, "model_dump"):
        observable = observable.model_dump(mode="json", exclude_none=True)
    if not isinstance(observable, dict):
        return ""
    target = observable.get("target")
    if isinstance(target, dict) and isinstance(target.get("selector"), str):
        return target["selector"]
    selector = observable.get("selector") or observable.get("id")
    if not isinstance(selector, str):
        return ""
    if ":" not in selector:
        return f"probe:{selector}"
    return selector


def _selector_target_from_retained_observable(
    observable: Any,
    selector: str,
) -> SelectorTargetSchema:
    if hasattr(observable, "model_dump"):
        observable = observable.model_dump(mode="json", exclude_none=True)
    observable = observable if isinstance(observable, dict) else {}
    target = observable.get("target") if isinstance(observable.get("target"), dict) else {}
    value_schema = observable.get("value_schema")
    retention = observable.get("retention") if isinstance(observable.get("retention"), dict) else {}
    label = str(observable.get("label") or selector)
    target_kind = str(target.get("kind") or "")
    selector_kind: Literal["probe", "port", "edge", "graph_output", "recurrent_carry", "state_path", "task_data"]
    if target_kind in {
        "port",
        "edge",
        "graph_output",
        "recurrent_carry",
        "state_path",
        "task_data",
    }:
        selector_kind = target_kind  # type: ignore[assignment]
    elif selector.startswith("port:"):
        selector_kind = "port"
    elif selector.startswith("edge:"):
        selector_kind = "edge"
    elif selector.startswith("graph_output:"):
        selector_kind = "graph_output"
    elif selector.startswith("path:"):
        selector_kind = "state_path"
    elif selector.startswith("task_data:"):
        selector_kind = "task_data"
    else:
        selector_kind = "probe"
    return SelectorTargetSchema(
        id=f"selector:{selector}",
        label=label,
        kind=selector_kind,
        selector=selector,
        value_schema=ValueSchema(
            id=f"value:{selector}",
            label=label,
            kind="retained_observable",
            dtype=value_schema.get("dtype") if isinstance(value_schema, dict) else None,
            shape=value_schema.get("shape") if isinstance(value_schema, dict) else None,
            units=value_schema.get("units") if isinstance(value_schema, dict) else None,
            frame=value_schema.get("frame") if isinstance(value_schema, dict) else None,
            origin="declared",
            metadata={"retention": retention},
        ),
        origin="declared",
        source={
            "retained_observable_id": observable.get("id"),
            "target": target,
        },
        metadata={
            "retention_target": target or {"kind": selector_kind, "selector": selector},
            "retention": retention,
            **{
                key: value
                for key, value in observable.items()
                if key not in {"selector", "target", "retention", "value_schema"}
            },
        },
    )


def _known_state_hint_targets() -> list[SelectorTargetSchema]:
    hints = [
        ("path:states.mechanics.effector.pos", "Effector position", "vector"),
        ("path:states.mechanics.effector.vel", "Effector velocity", "vector"),
        ("path:states.net.hidden", "Network hidden state", "vector"),
        ("path:states.net.output", "Network output", "vector"),
        ("path:states.efferent.output", "Motor command", "vector"),
        ("path:task.validation_trials.targets", "Validation targets", "vector"),
    ]
    return [
        SelectorTargetSchema(
            id=f"selector:{selector}",
            label=label,
            kind="state_hint",
            selector=selector,
            value_schema=ValueSchema(
                id=f"value:{selector}",
                label=label,
                kind="state_hint",
                dtype=dtype,
                origin="curated_fallback",
            ),
            origin="curated_fallback",
            source={"curated": True},
        )
        for selector, label, dtype in hints
    ]


def _collect_selector_strings(value: Any) -> list[str]:
    selectors: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "selector" and isinstance(item, str):
                selectors.append(item)
            else:
                selectors.extend(_collect_selector_strings(item))
    elif isinstance(value, list):
        for item in value:
            selectors.extend(_collect_selector_strings(item))
    return selectors


def _dedupe_selector_targets(
    targets: list[SelectorTargetSchema],
) -> list[SelectorTargetSchema]:
    by_selector: dict[str, SelectorTargetSchema] = {}
    for target in targets:
        by_selector.setdefault(target.selector, target)
    return list(by_selector.values())
