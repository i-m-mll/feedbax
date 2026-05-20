"""Static provider-owned schema enumeration for Studio workspaces."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError as PydanticValidationError

from feedbax.manifest import SCHEMA_VERSION, utc_now
from feedbax.web.models.component import PortType
from feedbax.web.models.graph import GraphSpec, StudioTaskBindingSpec, StudioWorkspaceSpec


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
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskDataSchema(StudioSchemaModel):
    """Provider-owned task data schema record."""

    id: str
    label: str
    kind: str
    path: str
    bindable: bool = False
    value_schema: ValueSchema
    metadata: dict[str, Any] = Field(default_factory=dict)


class SelectorTargetSchema(StudioSchemaModel):
    """Selectable provider target for objectives, probes, and Studio editors."""

    id: str
    label: str
    kind: Literal["port", "task_data", "objective", "probe", "state_hint"]
    selector: str
    value_schema: ValueSchema
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


class StudioSchemaEnumerationRequest(StudioSchemaModel):
    """HTTP request for static Studio schema enumeration."""

    workspace: StudioWorkspaceSpec
    scenario_id: Optional[str] = None


def enumerate_studio_schema_registry(
    workspace: StudioWorkspaceSpec | dict[str, Any],
    scenario_id: Optional[str] = None,
) -> StudioSchemaRegistry:
    """Enumerate static Studio schemas without compiling or running JAX code."""

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
            _task_binding_issues(
                scenario.task_binding_spec,
                scenario.graph,
                f"/scenarios/{scenario.id}/task_binding_spec",
            )
        )
        _mark_bound_ports(registry.ports, scenario.task_binding_spec)

    registry.selector_targets.extend(_objective_selector_targets(scenario.objective_spec))
    registry.selector_targets.extend(_explicit_probe_selector_targets(scenario.probe_specs))
    registry.selector_targets.extend(_known_state_hint_targets())
    registry.selector_targets = _dedupe_selector_targets(registry.selector_targets)
    return registry


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
            metadata={"component_type": component_type},
        ),
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
            metadata=metadata,
        ),
        metadata=metadata,
    )


def node_input_binding(graph: GraphSpec, port_name: str) -> Optional[tuple[str, str]]:
    binding = graph.input_bindings.get(port_name)
    return tuple(binding) if binding is not None else None


def _enumerate_task_data(task_binding_spec: StudioTaskBindingSpec) -> list[TaskDataSchema]:
    task_data: list[TaskDataSchema] = []
    for output in task_binding_spec.exposed_outputs:
        value = output.value_spec
        task_data.append(
            TaskDataSchema(
                id=f"task_data:{output.id}",
                label=output.label,
                kind=output.kind,
                path=output.path,
                bindable=output.bindable,
                value_schema=ValueSchema(
                    id=f"value:task_data:{output.id}",
                    label=output.label,
                    kind="task_data",
                    dtype=output.dtype or (value.dtype if value is not None else None),
                    shape=output.expected_shape or (value.shape if value is not None else None),
                    units=output.units or (value.units if value is not None else None),
                    frame=output.frame or (value.frame if value is not None else None),
                    metadata={
                        **output.metadata,
                        "task_data_path": output.path,
                        "value_spec": value.model_dump(mode="json", exclude_none=True)
                        if value is not None
                        else None,
                    },
                ),
                metadata=output.metadata,
            )
        )
    return task_data


def _mark_bound_ports(
    ports: list[PortSchema],
    task_binding_spec: StudioTaskBindingSpec,
) -> None:
    task_data_ids = {
        output.id: f"task_data:{output.id}" for output in task_binding_spec.exposed_outputs
    }
    by_target = {
        (port.node_id, port.port): port
        for port in ports
        if port.direction == "input" and port.node_id is not None
    }
    for binding in task_binding_spec.bindings:
        port = by_target.get((binding.target_node_id, binding.target_port))
        if port is not None:
            port.bound_task_data_id = task_data_ids.get(binding.source_output_id)


def _task_binding_issues(
    task_binding_spec: StudioTaskBindingSpec,
    graph: Optional[GraphSpec],
    base_path: str,
) -> list[SchemaValidationIssue]:
    issues: list[SchemaValidationIssue] = []
    seen_outputs: set[str] = set()
    output_by_id = {}
    for index, output in enumerate(task_binding_spec.exposed_outputs):
        if output.id in seen_outputs:
            issues.append(
                SchemaValidationIssue(
                    type="duplicate_task_data",
                    message=f"Task data {output.id!r} is declared more than once",
                    location={"path": f"{base_path}/exposed_outputs/{index}/id"},
                )
            )
        seen_outputs.add(output.id)
        output_by_id[output.id] = output

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
    binding_targets: set[tuple[str, str]] = set()
    for index, binding in enumerate(task_binding_spec.bindings):
        binding_path = f"{base_path}/bindings/{index}"
        output = output_by_id.get(binding.source_output_id)
        if output is None:
            issues.append(
                SchemaValidationIssue(
                    type="unknown_task_data",
                    message=f"Task binding source {binding.source_output_id!r} is not declared",
                    location={"path": f"{binding_path}/source_output_id"},
                )
            )
        elif not output.bindable:
            issues.append(
                SchemaValidationIssue(
                    type="task_data_not_bindable",
                    message=f"Task data {binding.source_output_id!r} is not bindable",
                    location={"path": f"{binding_path}/source_output_id"},
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
            source={"task_data_id": item.id},
        )
        for item in task_data
    ]


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
                    metadata={"node": probe.node, "timing": probe.timing},
                ),
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
                ),
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
                ),
                source={"probe_specs_index": index},
                metadata={key: value for key, value in probe.items() if key != "selector"},
            )
        )
    return targets


def _known_state_hint_targets() -> list[SelectorTargetSchema]:
    hints = [
        ("path:state.effector.pos", "Effector position", "vector"),
        ("path:state.effector.vel", "Effector velocity", "vector"),
        ("path:state.mechanics.effector.pos", "Mechanics effector position", "vector"),
        ("path:state.mechanics.effector.vel", "Mechanics effector velocity", "vector"),
        ("path:state.network.hidden", "Network hidden state", "vector"),
        ("path:state.controller.hidden", "Controller hidden state", "vector"),
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
            ),
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
