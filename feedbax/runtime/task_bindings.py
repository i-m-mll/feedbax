"""Runtime helpers for Studio task-data to graph bindings."""

from __future__ import annotations

from dataclasses import dataclass, replace

import equinox as eqx

from feedbax.contracts.graph import (
    StudioTaskBinding,
    StudioTaskBindingSpec,
    StudioTaskDataSpec,
)
from feedbax.runtime.graph import Graph
from feedbax.runtime._signal_schedule import ScopedPort
from feedbax.runtime.task_data_roles import task_data_role


@dataclass(frozen=True)
class TaskInputPlan:
    """A task-data stream bound to one graph input port."""

    data_id: str
    data_path: str
    graph_input: str
    target_node: str
    target_port: str
    role: str


def expose_task_inputs(
    graph: Graph,
    task_binding_spec: StudioTaskBindingSpec,
) -> tuple[Graph, tuple[TaskInputPlan, ...]]:
    """Expose graph inputs from task-data bindings."""

    input_ports = list(graph.input_ports)
    input_bindings = dict(graph.input_bindings)
    external_by_target = {(node, port): name for name, (node, port) in input_bindings.items()}
    task_input_owners: dict[str, str] = {}
    input_plans: list[TaskInputPlan] = []
    data_by_id = _task_data_by_id(task_binding_spec)

    for binding in task_binding_spec.bindings:
        data = data_by_id[binding.source_data_id]
        role = binding.role or task_data_role(data)
        target_key = _validate_target(graph, binding)

        graph_input = external_by_target.get(target_key)
        if graph_input is None:
            graph_input = f"task:{binding.source_data_id}->{target_key[0]}.{target_key[1]}"
            if graph_input in input_ports:
                raise ValueError(
                    f"Task binding {binding.id!r} cannot expose {target_key[0]}.{target_key[1]} "
                    f"as external input {graph_input!r}, already declared on the graph"
                )
            input_ports.append(graph_input)
            input_bindings[graph_input] = target_key
            external_by_target[target_key] = graph_input
        if graph_input in task_input_owners:
            raise ValueError(
                f"Input {target_key[0]}.{target_key[1]} receives task data from both "
                f"{task_input_owners[graph_input]!r} and {binding.id!r}"
            )
        task_input_owners[graph_input] = binding.id
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

    if input_bindings == graph.input_bindings:
        return graph, tuple(input_plans)
    exposed_graph = eqx.tree_at(
        lambda g: (g.input_ports, g.input_bindings),
        graph,
        (tuple(input_ports), input_bindings),
    )
    exposed_graph._validate_graph()
    if (schedule := graph._scoped_schedule) is not None:
        aliases = dict(schedule.input_aliases)
        occupied = {target for _, target in schedule.inputs}
        occupied.update(route.target for route in schedule.routes)
        inputs = list(schedule.inputs)
        for name in input_bindings.keys() - graph.input_bindings.keys():
            node, port = input_bindings[name]
            endpoint = ScopedPort((node,), port)
            target = aliases.get(endpoint, endpoint)
            if target in occupied:
                raise ValueError(f"Task input {name!r} would add another driver to {target!r}")
            occupied.add(target)
            inputs.append((name, target))
        object.__setattr__(exposed_graph, "schedule", replace(schedule, inputs=tuple(sorted(inputs))))
    return exposed_graph, tuple(input_plans)


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
