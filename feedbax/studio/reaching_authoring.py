"""Typed Task and Objective choices for the managed reaching authoring path."""

from __future__ import annotations

from pydantic import TypeAdapter, ValidationError

from feedbax.component_registry.registry import ComponentRegistry
from feedbax.contracts.graph import (
    GraphSpec,
    StudioTaskBinding,
    StudioTaskBindingSpec,
    StudioTaskDataSpec,
    StudioValueSpec,
)
from feedbax.contracts.studio_authoring import (
    ReachCoordinate,
    StudioAuthoringOption,
    StudioReachPosition,
    StudioReachTargetAuthoring,
    StudioReachTargetEdit,
)
from feedbax.contracts.value_schema import ValueSchema
from feedbax.contracts.model_values import MODEL_OWNERS
from feedbax.studio.protocol import task_binding_id
from feedbax.studio.schema import validate_task_binding_schema

REACH_TARGET_ID = "reach_target"
_TARGET_PATH = "inputs.effector_target.pos"
_TARGET_ROLE = "model_input"
_POSITION = TypeAdapter(tuple[ReachCoordinate, ReachCoordinate])


def reaching_endpoint(graph: GraphSpec, components: ComponentRegistry) -> ValueSchema | None:
    endpoint = graph.output_bindings.get("effector")
    if endpoint is None or "effector" not in graph.output_ports:
        return None
    node_id, port = endpoint
    node = graph.nodes.get(node_id)
    if node is None or node.type != "feedbax.PointMass" or port != "effector":
        return None
    if "." in node_id or "/" in node_id or not node_id or port not in node.output_ports:
        return None
    component = components.get(node.type)
    owner = MODEL_OWNERS[node.type]
    if (
        component is None or component.revision != owner.revision
        or node.revision not in (None, owner.revision)
    ):
        return None
    position = next(
        (state for state in owner.states or () if state.symbol == owner.scene_position), None,
    )
    if position is None or position.axis != "xy":
        return None
    return ValueSchema(
        id=f"path:{'.'.join((node_id, *position.fields))}",
        label="Endpoint position",
        kind="array",
        shape=[2],
        units=position.units,
        frame=position.frame,
        origin="declared",
    )


def _reach_value(position: tuple[float, float]) -> StudioValueSpec:
    return StudioValueSpec(
        value_form="literal", mode="constant", value=list(position),
        dtype="float32", shape=[2], units="m", frame="world",
    )


def reach_target_recipe(
    graph: GraphSpec,
    spec: StudioTaskBindingSpec,
    components: ComponentRegistry,
    change: StudioReachTargetEdit | None,
) -> tuple[StudioReachTargetAuthoring | None, StudioTaskBindingSpec | None]:
    names = {REACH_TARGET_ID, _TARGET_PATH}
    matching_data = [item for item in spec.exposed_data if names.intersection((item.id, item.path))]
    matching_bindings = [item for item in spec.bindings if item.source_data_id == REACH_TARGET_ID]
    if len(matching_data) > 1 or len(matching_bindings) > 1:
        return None, None
    datum = matching_data[0] if matching_data else None
    previous = matching_bindings[0] if matching_bindings else None
    position = None
    if datum is not None:
        value = datum.value_spec
        if (
            datum.id != REACH_TARGET_ID or datum.path != _TARGET_PATH
            or datum.role != _TARGET_ROLE or not datum.bindable
            or datum.expected_shape != ["time", 2] or datum.dtype != "float32"
            or datum.units != "m" or datum.frame != "world"
            or value is None or value.value_form != "literal" or value.mode != "constant"
            or value.variation.scope != "fixed"
            or value.dtype not in (None, datum.dtype) or value.shape not in (None, [2])
            or value.units not in (None, datum.units) or value.frame not in (None, datum.frame)
        ):
            return None, None
        try:
            x, y = _POSITION.validate_python(value.value)
            position = StudioReachPosition(x=x, y=y)
        except ValidationError:
            return None, None
    elif previous is not None:
        return None, None

    if datum is None:
        datum = StudioTaskDataSpec(
            id=REACH_TARGET_ID, path=_TARGET_PATH, label="Reach target", kind="signal",
            role=_TARGET_ROLE, bindable=True, expected_shape=["time", 2],
            dtype="float32", units="m", frame="world", value_spec=_reach_value((0.0, 0.0)),
        )
        data = [*spec.exposed_data, datum]
    else:
        data = spec.exposed_data
    bindings = [binding for binding in spec.bindings if binding is not previous]
    candidates = {}
    occupied = set(graph.input_bindings.values())
    component = components.get("feedbax.GRU")
    if component is None:
        return None, None
    for node_id, node in graph.nodes.items():
        if (
            node.type != component.name or node.revision not in (None, component.revision)
            or "input" not in node.input_ports or (node_id, "input") in occupied
        ):
            continue
        binding = StudioTaskBinding(
            id=task_binding_id(REACH_TARGET_ID, node_id, "input"),
            source_data_id=REACH_TARGET_ID, target_node_id=node_id, target_port="input",
            role=_TARGET_ROLE, metadata=previous.metadata if previous else {},
        )
        candidate = spec.model_copy(update={"exposed_data": data, "bindings": [*bindings, binding]})
        if any(issue.severity == "error" for issue in validate_task_binding_schema(
            candidate, graph, "/task_binding_spec", component_registry=components,
        )):
            continue
        candidates[binding.id] = binding
    if not candidates:
        return None, None
    selected = previous.id if previous is not None and previous.id in candidates else None
    view = StudioReachTargetAuthoring(
        inputs=[
            StudioAuthoringOption(value=key, label=f"{binding.target_node_id}.{binding.target_port}")
            for key, binding in candidates.items()
        ],
        input=selected, position=position,
    )
    if change is None or (change.input, change.position) == (selected, position):
        return view, None
    if change.input not in candidates:
        raise ValueError("Reach Target requires a declared Model Input")
    assert datum.value_spec is not None
    edited = datum.model_copy(update={
        "value_spec": datum.value_spec.model_copy(update={"value": [change.position.x, change.position.y]}),
    })
    selected_binding = candidates[change.input]
    return view, spec.model_copy(update={
        "exposed_data": [edited if item is datum else item for item in data],
        "bindings": (
            [selected_binding if binding is previous else binding for binding in spec.bindings]
            if previous is not None else [*bindings, selected_binding]
        ),
    })
