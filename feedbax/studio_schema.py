"""Static provider-owned schema enumeration for Studio workspaces."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError as PydanticValidationError

from feedbax.manifest import SCHEMA_VERSION, utc_now
from feedbax.studio_protocol import (
    GRAPH_BINDABLE_TASK_DATA_ROLES,
    PROTOCOL_TASK_DATA_KINDS,
    TASK_DATA_ROLES,
    is_bindable_task_data,
    task_data_role,
    task_data_surface,
    task_data_uses_protocol_path,
)
from feedbax.web.models.component import PortType
from feedbax.web.models.graph import GraphSpec, StudioTaskBindingSpec, StudioWorkspaceSpec

SchemaOrigin = Literal[
    "declared",
    "inferred_static",
    "runtime_sample",
    "curated_fallback",
    "unknown",
]


class StudioSchemaModel(BaseModel):
    """Base model for static Studio schema records."""

    model_config = ConfigDict(extra="forbid")


class ValueSchema(StudioSchemaModel):
    """Provider-owned value type metadata for selectable Studio surfaces."""

    id: str
    label: str
    kind: str
    dtype: Optional[str] = None
    shape: Optional[list[Any]] = None
    rank: Optional[int] = None
    units: Optional[str] = None
    frame: Optional[str] = None
    origin: SchemaOrigin = "unknown"
    metadata: dict[str, Any] = Field(default_factory=dict)


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
        "task_data",
        "objective",
        "probe",
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

    workspace: StudioWorkspaceSpec
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
            else StudioWorkspaceSpec.model_validate(workspace)
        )
    except PydanticValidationError as exc:
        return StudioSchemaRegistry(
            scenario_id=scenario_id,
            issues=[
                SchemaValidationIssue(
                    type="workspace_schema_error",
                    message=str(error.get("msg", "Invalid workspace value")),
                    location={
                        "path": "/" + "/".join(str(part) for part in error.get("loc", ()))
                    },
                )
                for error in exc.errors()
            ],
        )

    selected_scenario_id = scenario_id or _active_scenario_id(workspace_spec)
    issues = _workspace_reference_issues(workspace_spec)
    registry = StudioSchemaRegistry(
        workspace_id=workspace_spec.id,
        scenario_id=selected_scenario_id,
        issues=issues,
        metadata={
            "enumerated_by": "feedbax.studio_schema",
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

    if scenario.graph is None:
        registry.issues.append(
            SchemaValidationIssue(
                type="missing_graph",
                message=f"Scenario {scenario.id!r} does not have a graph",
                location={"path": f"/scenarios/{scenario.id}/graph"},
            )
        )
    else:
        registry.ports = _enumerate_graph_ports(scenario.graph)
        registry.selector_targets.extend(_port_selector_targets(registry.ports))
        registry.selector_targets.extend(_graph_probe_selector_targets(scenario.graph))
        registry.issues.extend(
            validate_graph_connection_schema(
                scenario.graph,
                f"/scenarios/{scenario.id}/graph",
            )
        )

    if scenario.task_binding_spec is None:
        registry.issues.append(
            SchemaValidationIssue(
                type="missing_task_binding_spec",
                message=f"Scenario {scenario.id!r} does not have a task_binding_spec",
                severity="warning",
                location={"path": f"/scenarios/{scenario.id}/task_binding_spec"},
            )
        )
    else:
        registry.task_data = _enumerate_task_data(scenario.task_binding_spec)
        registry.selector_targets.extend(_task_data_selector_targets(registry.task_data))
        registry.issues.extend(
            validate_task_binding_schema(
                scenario.task_binding_spec,
                scenario.graph,
                f"/scenarios/{scenario.id}/task_binding_spec",
            )
        )
        _mark_bound_ports(registry.ports, scenario.task_binding_spec)

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
) -> list[SchemaValidationIssue]:
    """Validate graph wires against provider-owned port schema records."""

    ports = _enumerate_graph_ports(graph)
    by_node_port_direction = {
        (port.node_id, port.port, port.direction): port
        for port in ports
        if port.node_id is not None
    }
    by_node_port = {
        (port.node_id, port.port): port
        for port in ports
        if port.node_id is not None
    }
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
                            f"Wire source port "
                            f"{wire.source_node}.{wire.source_port} does not exist"
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
                            f"Wire target port "
                            f"{wire.target_node}.{wire.target_port} does not exist"
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

    return issues


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
                        f"Stage {stage.id!r} references missing scenario "
                        f"{stage.scenario_id!r}"
                    ),
                    location={"path": f"/stages/{index}/scenario_id"},
                )
            )
    return issues


def _enumerate_graph_ports(graph: GraphSpec) -> list[PortSchema]:
    from feedbax.web.services.component_registry import ComponentRegistry

    registry = ComponentRegistry()
    ports: list[PortSchema] = []
    for node_id, node in graph.nodes.items():
        meta = registry.get(node.type)
        input_ports = list(node.input_ports or (meta.input_ports if meta is not None else []))
        output_ports = list(node.output_ports or (meta.output_ports if meta is not None else []))
        for port_name in input_ports:
            port_type = meta.port_types.inputs.get(port_name) if meta and meta.port_types else None
            ports.append(
                _port_schema(
                    node_id=node_id,
                    component_type=node.type,
                    port=port_name,
                    direction="input",
                    port_type=port_type,
                )
            )
        for port_name in output_ports:
            port_type = meta.port_types.outputs.get(port_name) if meta and meta.port_types else None
            ports.append(
                _port_schema(
                    node_id=node_id,
                    component_type=node.type,
                    port=port_name,
                    direction="output",
                    port_type=port_type,
                )
            )
    for port_name in graph.input_ports:
        ports.append(
            _graph_port_schema(port=port_name, direction="input", bound=node_input_binding(graph, port_name))
        )
    for port_name in graph.output_ports:
        ports.append(_graph_port_schema(port=port_name, direction="output"))
    return ports


def _port_schema(
    *,
    node_id: str,
    component_type: str,
    port: str,
    direction: Literal["input", "output"],
    port_type: Optional[PortType],
) -> PortSchema:
    port_id = f"port:{node_id}.{port}:{direction}"
    dtype = port_type.dtype if port_type is not None else None
    origin: SchemaOrigin = "declared" if port_type is not None else "unknown"
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
            shape=list(port_type.shape) if port_type is not None and port_type.shape else None,
            rank=port_type.rank if port_type is not None else None,
            origin=origin,
            metadata={"component_type": component_type},
        ),
        origin=origin,
    )


def _graph_port_schema(
    *,
    port: str,
    direction: Literal["input", "output"],
    bound: Optional[tuple[str, str]] = None,
) -> PortSchema:
    port_id = f"port:graph.{port}:{direction}"
    metadata: dict[str, Any] = {}
    if bound is not None:
        metadata["binding"] = {"node_id": bound[0], "port": bound[1]}
    return PortSchema(
        id=port_id,
        label=f"graph.{port}",
        port=port,
        direction=direction,
        value_schema=ValueSchema(
            id=f"value:{port_id}",
            label=f"graph.{port}",
            kind="graph_port",
            origin="inferred_static",
            metadata=metadata,
        ),
        origin="inferred_static",
        metadata=metadata,
    )


def node_input_binding(graph: GraphSpec, port_name: str) -> Optional[tuple[str, str]]:
    binding = graph.input_bindings.get(port_name)
    return tuple(binding) if binding is not None else None


def _enumerate_task_data(task_binding_spec: StudioTaskBindingSpec) -> list[TaskDataSchema]:
    task_data: list[TaskDataSchema] = []
    for data in task_binding_spec.exposed_data:
        value = data.value_spec
        role = task_data_role(data)
        surface = task_data_surface(data)
        metadata = {
            **data.metadata,
            "task_data_role": role,
            "task_data_surface": surface,
        }
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
                    shape=data.expected_shape or (value.shape if value is not None else None),
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


def _mark_bound_ports(
    ports: list[PortSchema],
    task_binding_spec: StudioTaskBindingSpec,
) -> None:
    task_data_ids = {
        data.id: f"task_data:{data.id}" for data in task_binding_spec.exposed_data
    }
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
        for port in _enumerate_graph_ports(graph)
        if port.direction == "input" and port.node_id is not None
    }
    binding_targets: set[tuple[str, str]] = set()
    for index, binding in enumerate(task_binding_spec.bindings):
        binding_path = f"{base_path}/bindings/{index}"
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
                    "model_input/graph_input Task Data may be marked bindable"
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
    if data.bindable and (
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
        target = (
            by_id.get(schema_target_id)
            if isinstance(schema_target_id, str)
            else None
        )
        if target is None and target_selector is not None:
            target = by_selector.get(target_selector.compact)
        if target is None:
            compact = target_selector.compact if target_selector is not None else None
            issues.append(
                SchemaValidationIssue(
                    type="intervention_unknown_target",
                    message=(
                        f"Intervention target {compact!r} is not in the selector "
                        "schema registry"
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
        and source_dtype != target_dtype
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

    if source.rank is not None or target.rank is not None:
        known_constraint = True
    if source.rank is not None and target.rank is not None and source.rank != target.rank:
        issues.append(
            SchemaValidationIssue(
                type=f"{issue_prefix}_rank_mismatch",
                message=(
                    f"{source_label} has rank {source.rank}, but "
                    f"{target_label} expects rank {target.rank}"
                ),
                location={"path": path},
            )
        )

    source_shape = source.shape
    target_shape = target.shape
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
            )
        )
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
    from feedbax.web.services.loss_service import loss_service

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
    probe_specs: list[dict[str, Any]],
) -> list[SelectorTargetSchema]:
    targets: list[SelectorTargetSchema] = []
    for index, probe in enumerate(probe_specs):
        selector = probe.get("selector") or probe.get("id")
        if not isinstance(selector, str) or not selector:
            continue
        if ":" not in selector:
            selector = f"probe:{selector}"
        targets.append(
            SelectorTargetSchema(
                id=f"selector:{selector}",
                label=str(probe.get("label") or selector),
                kind="probe",
                selector=selector,
                value_schema=ValueSchema(
                    id=f"value:{selector}",
                    label=str(probe.get("label") or selector),
                    kind="probe",
                    dtype=probe.get("dtype") if isinstance(probe.get("dtype"), str) else None,
                    origin="declared",
                ),
                origin="declared",
                source={"probe_specs_index": index},
                metadata={key: value for key, value in probe.items() if key != "selector"},
            )
        )
    return targets


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
