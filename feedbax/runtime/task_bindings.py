"""Runtime helpers for Studio task-data to graph bindings."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import equinox as eqx
from equinox.nn import State, StateIndex
import jax.numpy as jnp
import jax.tree as jt

from feedbax.contracts.graph import (
    StudioTaskBinding,
    StudioTaskBindingSpec,
    StudioTaskDataSpec,
)
from feedbax.runtime.graph import Graph
from feedbax.runtime.task_data_roles import task_data_role


COMPONENT_PARAMETER_ROLE = "component_parameter"
TaskBindingTemporality = Literal["constant", "time_varying"]


@dataclass(frozen=True)
class TaskInputPlan:
    """A task-data stream bound to one graph input port."""

    data_id: str
    data_path: str
    graph_input: str
    target_node: str
    target_port: str
    role: str


@dataclass(frozen=True)
class TaskParameterStateInitPlan:
    """A constant-within-trial task value bound to a component StateIndex."""

    data_id: str
    data_path: str
    target_node: str
    target_port: str
    target_label: str
    role: str = COMPONENT_PARAMETER_ROLE


@dataclass(frozen=True)
class TaskBindingExposure:
    """Graph plus executable plans derived from a StudioTaskBindingSpec."""

    graph: Graph
    input_plans: tuple[TaskInputPlan, ...]
    state_init_plans: tuple[TaskParameterStateInitPlan, ...]


def task_data_temporality(data: Any) -> TaskBindingTemporality:
    """Return whether task data is trial-constant or time-varying.

    A leading ``"time"`` dimension is the durable schema signal for a
    time-varying trajectory. Metadata aliases support current Studio surfaces
    that already describe temporal support without changing the schema shape.
    """

    metadata = _metadata(data)
    explicit = metadata.get("temporality") or metadata.get("temporal_support")
    if isinstance(explicit, str):
        if explicit in {
            "trajectory",
            "materialized_trajectory",
            "epoch_masked_signal",
            "time_varying",
            "per_step",
        }:
            return "time_varying"
        if explicit in {"constant", "trial_constant", "initial", "per_trial"}:
            return "constant"

    shape = _field(data, "expected_shape")
    if shape is None:
        value_spec = _field(data, "value_spec")
        shape = _field(value_spec, "shape")
    if isinstance(shape, (list, tuple)) and len(shape) > 0 and shape[0] == "time":
        return "time_varying"
    return "constant"


def expose_task_bindings(
    graph: Graph,
    task_binding_spec: StudioTaskBindingSpec,
) -> TaskBindingExposure:
    """Expose graph inputs and StateIndex init plans from task-data bindings."""

    input_ports = list(graph.input_ports)
    input_bindings = dict(graph.input_bindings)
    input_plans: list[TaskInputPlan] = []
    state_init_plans: list[TaskParameterStateInitPlan] = []
    data_by_id = _task_data_by_id(task_binding_spec)
    state_indices = graph.task_parameter_state_indices()

    for binding in task_binding_spec.bindings:
        data = data_by_id[binding.source_data_id]
        role = binding.role or task_data_role(data)
        target_key = _validate_target(graph, binding)

        if role == COMPONENT_PARAMETER_ROLE and task_data_temporality(data) == "constant":
            label = _task_parameter_label(binding)
            if label not in state_indices:
                raise ValueError(
                    f"Task binding {binding.id!r} targets component parameter "
                    f"{label!r}, but the graph declares {sorted(state_indices)!r}"
                )
            state_init_plans.append(
                TaskParameterStateInitPlan(
                    data_id=binding.source_data_id,
                    data_path=data.path,
                    target_node=binding.target_node_id,
                    target_port=binding.target_port,
                    target_label=label,
                    role=role,
                )
            )
            continue

        graph_input = next(
            (name for name, bound in input_bindings.items() if tuple(bound) == target_key),
            f"task:{binding.source_data_id}->{binding.target_node_id}.{binding.target_port}",
        )
        if graph_input not in input_ports:
            input_ports.append(graph_input)
        input_bindings[graph_input] = target_key
        input_plans.append(
            TaskInputPlan(
                data_id=binding.source_data_id,
                data_path=data.path,
                graph_input=graph_input,
                target_node=binding.target_node_id,
                target_port=binding.target_port,
                role=role,
            )
        )

    exposed_graph = eqx.tree_at(
        lambda g: (g.input_ports, g.input_bindings),
        graph,
        (tuple(input_ports), input_bindings),
    )
    return TaskBindingExposure(
        graph=exposed_graph,
        input_plans=tuple(input_plans),
        state_init_plans=tuple(state_init_plans),
    )


def expose_task_inputs(
    graph: Graph,
    task_binding_spec: StudioTaskBindingSpec,
) -> tuple[Graph, tuple[TaskInputPlan, ...]]:
    """Backward-compatible wrapper returning graph-input task plans only."""

    exposure = expose_task_bindings(graph, task_binding_spec)
    return exposure.graph, exposure.input_plans


def apply_task_parameter_state_inits(
    state: State,
    graph: Graph,
    plans: Iterable[TaskParameterStateInitPlan],
    task_data: Mapping[str, Any],
) -> State:
    """Apply constant component-parameter task values to graph StateIndex slots."""

    indices = graph.task_parameter_state_indices()
    next_state = state
    for plan in plans:
        if plan.target_label not in indices:
            raise ValueError(f"Unknown task-parameter StateIndex label {plan.target_label!r}")
        value = _lookup_task_value(task_data, plan)
        idx = indices[plan.target_label]
        next_state = _set_state_matching_dtypes(next_state, idx, value)
    return next_state


def binding_spec_from_legacy_extra_inputs(
    graph: Graph,
    extra_inputs: Iterable[str],
) -> StudioTaskBindingSpec:
    """Map legacy extra input names, including ``intervene:*``, to task bindings."""

    exposed_data: list[StudioTaskDataSpec] = []
    bindings: list[StudioTaskBinding] = []
    for input_name in extra_inputs:
        if input_name not in graph.input_bindings:
            continue
        target_node, target_port = graph.input_bindings[input_name]
        if input_name.startswith("intervene:"):
            label = input_name.removeprefix("intervene:")
            data_id = label
            role = COMPONENT_PARAMETER_ROLE
            kind = "intervention"
            path = f"intervene.{label}"
            metadata = {
                "legacy_input": input_name,
                "task_parameter_label": label,
                "temporal_support": "trajectory",
            }
        else:
            data_id = input_name
            role = "graph_input"
            kind = "signal"
            path = input_name
            metadata = {"legacy_input": input_name}
        exposed_data.append(
            StudioTaskDataSpec(
                id=data_id,
                label=data_id,
                kind=kind,
                role=role,
                path=path,
                bindable=True,
                metadata=metadata,
            )
        )
        bindings.append(
            StudioTaskBinding(
                id=f"task:{data_id}->{target_node}:{target_port}",
                source_data_id=data_id,
                target_node_id=target_node,
                target_port=target_port,
                role=role,
                metadata=metadata,
            )
        )
    return StudioTaskBindingSpec(exposed_data=exposed_data, bindings=bindings)


def _validate_target(graph: Graph, binding: StudioTaskBinding) -> tuple[str, str]:
    if binding.target_node_id not in graph.nodes:
        raise ValueError(
            f"Task binding {binding.id!r} targets missing node {binding.target_node_id!r}"
        )
    if binding.target_port not in graph.nodes[binding.target_node_id].input_ports:
        raise ValueError(
            f"Task binding {binding.id!r} targets missing port "
            f"{binding.target_node_id}.{binding.target_port}"
        )
    return (binding.target_node_id, binding.target_port)


def _task_data_by_id(spec: StudioTaskBindingSpec) -> dict[str, StudioTaskDataSpec]:
    data = {item.id: item for item in spec.exposed_data}
    missing = [
        binding.source_data_id for binding in spec.bindings if binding.source_data_id not in data
    ]
    if missing:
        raise ValueError(f"Task bindings reference unknown task data ids: {sorted(set(missing))}")
    return data


def _task_parameter_label(binding: StudioTaskBinding) -> str:
    label = binding.metadata.get("task_parameter_label")
    if isinstance(label, str) and label:
        return label
    return binding.source_data_id


def _lookup_task_value(task_data: Mapping[str, Any], plan: TaskParameterStateInitPlan) -> Any:
    for key in (plan.data_id, plan.data_path):
        if key in task_data:
            return task_data[key]
    raise ValueError(
        f"Task data does not include {plan.data_id!r} or path {plan.data_path!r} "
        f"for task-parameter binding {plan.target_label!r}"
    )


def _set_state_matching_dtypes(state: State, idx: StateIndex, new_value: Any) -> State:
    current_value = state.get(idx)
    return state.set(idx, _cast_to_state_dtypes(new_value, current_value))


def _cast_to_state_dtypes(new_value: Any, current_value: Any) -> Any:
    def _cast_leaf(new_leaf: Any, current_leaf: Any) -> Any:
        if hasattr(current_leaf, "dtype"):
            return jnp.asarray(new_leaf, dtype=current_leaf.dtype)
        return new_leaf

    return jt.map(_cast_leaf, new_value, current_value)


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _metadata(value: Any) -> dict[str, Any]:
    metadata = _field(value, "metadata", {})
    return metadata if isinstance(metadata, dict) else {}
