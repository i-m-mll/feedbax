"""Studio authoring normalization for persisted graph specs."""

from __future__ import annotations

from typing import Optional

from feedbax.contracts.acausal_interface import normalize_acausal_interfaces_for_graph
from feedbax.contracts.graph import (
    ComponentSpec,
    GraphProject,
    GraphSpec,
    StudioTaskBindingSpec,
    StudioWorkspaceSpec,
    WireSpec,
)


def _authoring_component_type(component_type: str) -> str:
    if component_type == "FeedbackChannel":
        return "Channel"
    if component_type == "PenzaiSubgraph":
        return "PenzaiAdapter"
    return component_type


def normalize_graph_for_studio_authoring(graph: GraphSpec) -> GraphSpec:
    """Normalize runtime/persisted component names to Studio authoring names."""

    graph = normalize_acausal_interfaces_for_graph(graph)
    nodes: dict[str, ComponentSpec] = {}
    for node_id, node_spec in graph.nodes.items():
        next_type = _authoring_component_type(node_spec.type)
        params = dict(node_spec.params)
        input_ports = list(node_spec.input_ports)
        output_ports = list(node_spec.output_ports)
        nodes[node_id] = node_spec.model_copy(
            update={
                "type": next_type,
                "params": params,
                "input_ports": input_ports,
                "output_ports": output_ports,
            }
        )

    def rename_port(node_name: str, port: str) -> str:
        node = nodes.get(node_name)
        if node and node.type == "Network" and port == "target":
            return "input"
        return port

    wires: list[WireSpec] = []
    for wire in graph.wires:
        normalized_wire = WireSpec(
            source_node=wire.source_node,
            source_port=rename_port(wire.source_node, wire.source_port),
            target_node=wire.target_node,
            target_port=rename_port(wire.target_node, wire.target_port),
            temporality=wire.temporality,
            recurrent_initializer=wire.recurrent_initializer,
        )
        wires.append(normalized_wire)
    input_bindings = {
        ("input" if name == "target" else name): (
            node,
            rename_port(node, port),
        )
        for name, (node, port) in graph.input_bindings.items()
    }
    subgraphs = None
    if graph.subgraphs:
        subgraphs = {}
        for node_id, subgraph in graph.subgraphs.items():
            if isinstance(subgraph, GraphSpec):
                subgraphs[node_id] = normalize_graph_for_studio_authoring(subgraph)
            else:
                subgraphs[node_id] = subgraph
    return graph.model_copy(
        update={
            "nodes": nodes,
            "wires": wires,
            "input_ports": ["input" if port == "target" else port for port in graph.input_ports],
            "input_bindings": input_bindings,
            "subgraphs": subgraphs,
        }
    )


def normalize_task_binding_spec_for_studio_authoring(
    task_binding_spec: Optional[StudioTaskBindingSpec],
    graph: GraphSpec,
) -> Optional[StudioTaskBindingSpec]:
    if task_binding_spec is None:
        return None
    changed = False
    bindings = []
    for binding in task_binding_spec.bindings:
        target = graph.nodes.get(binding.target_node_id)
        if target is not None and target.type == "Network" and binding.target_port == "target":
            binding = binding.model_copy(
                update={
                    "id": (f"task:{binding.source_data_id}->{binding.target_node_id}:input"),
                    "target_port": "input",
                }
            )
            changed = True
        bindings.append(binding)
    return (
        task_binding_spec.model_copy(update={"bindings": bindings})
        if changed
        else task_binding_spec
    )


def normalize_workspace_for_studio_authoring(
    workspace: Optional[StudioWorkspaceSpec],
) -> Optional[StudioWorkspaceSpec]:
    if workspace is None:
        return None
    changed = False
    scenarios = dict(workspace.scenarios)
    for scenario_id, scenario in workspace.scenarios.items():
        if scenario.graph is None:
            continue
        graph = normalize_graph_for_studio_authoring(scenario.graph)
        task_binding_spec = normalize_task_binding_spec_for_studio_authoring(
            scenario.task_binding_spec,
            graph,
        )
        if graph is scenario.graph and task_binding_spec is scenario.task_binding_spec:
            continue
        scenarios[scenario_id] = scenario.model_copy(
            update={"graph": graph, "task_binding_spec": task_binding_spec}
        )
        changed = True
    return workspace.model_copy(update={"scenarios": scenarios}) if changed else workspace


def normalize_project_for_studio_authoring(project: GraphProject) -> GraphProject:
    graph = normalize_graph_for_studio_authoring(project.graph)
    workspace = normalize_workspace_for_studio_authoring(project.workspace)
    if graph is project.graph and workspace is project.workspace:
        return project
    return project.model_copy(update={"graph": graph, "workspace": workspace})
