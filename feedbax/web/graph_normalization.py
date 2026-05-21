"""Studio authoring normalization for persisted graph specs."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from feedbax.web.models.graph import (
    ComponentSpec,
    GraphMetadata,
    GraphProject,
    GraphSpec,
    StudioTaskBindingSpec,
    StudioWorkspaceSpec,
    WireSpec,
)


def _authoring_component_type(component_type: str) -> str:
    if component_type == "SimpleStagedNetwork":
        return "Network"
    if component_type == "FeedbackChannel":
        return "Channel"
    if component_type == "PenzaiSubgraph":
        return "PenzaiAdapter"
    return component_type


def _int_param(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _standard_network_subgraph(node_id: str, params: dict) -> GraphSpec:
    now = datetime.now(timezone.utc).isoformat()
    hidden_type = params.get("hidden_type", "GRUCell")
    cell_type = "LSTM" if hidden_type in {"LSTM", "LSTMCell"} else "GRU"
    hidden_size = _int_param(params.get("hidden_size"), 100)
    input_size = _int_param(params.get("input_size"), 6)
    output_size = _int_param(params.get("out_size", params.get("output_size")), 2)
    activation = str(params.get("out_nonlinearity", "tanh"))
    cell_input_ports = ["input", "hidden", "cell"] if cell_type == "LSTM" else ["input", "hidden"]
    cell_output_ports = (
        ["output", "hidden", "cell"] if cell_type == "LSTM" else ["output", "hidden"]
    )
    return GraphSpec(
        nodes={
            "cell": ComponentSpec(
                type=cell_type,
                params={"input_size": input_size, "hidden_size": hidden_size},
                input_ports=cell_input_ports,
                output_ports=cell_output_ports,
            ),
            "readout": ComponentSpec(
                type="Linear",
                params={
                    "input_size": hidden_size,
                    "output_size": output_size,
                    "use_bias": True,
                    "activation": activation,
                },
                input_ports=["input"],
                output_ports=["output"],
            ),
        },
        wires=[
            WireSpec(
                source_node="cell",
                source_port="output",
                target_node="readout",
                target_port="input",
            )
        ],
        input_ports=["input", "feedback"],
        output_ports=["output", "hidden"],
        input_bindings={"input": ("cell", "input")},
        output_bindings={
            "output": ("readout", "output"),
            "hidden": ("cell", "output"),
        },
        taps=[],
        subgraphs={},
        metadata=GraphMetadata(
            name=f"{node_id} internals",
            description="Auto-generated Network subgraph",
            created_at=now,
            updated_at=now,
            version="1.0.0",
        ),
    )


def normalize_graph_for_studio_authoring(graph: GraphSpec) -> GraphSpec:
    """Normalize runtime/persisted component names to Studio authoring names."""

    nodes: dict[str, ComponentSpec] = {}
    generated_subgraphs: dict[str, GraphSpec] = {}
    for node_id, node_spec in graph.nodes.items():
        was_runtime_network = node_spec.type == "SimpleStagedNetwork"
        next_type = _authoring_component_type(node_spec.type)
        params = dict(node_spec.params)
        if next_type == "Network" and "output_size" in params and "out_size" not in params:
            params["out_size"] = params["output_size"]
        input_ports = list(node_spec.input_ports)
        if next_type == "Network":
            input_ports = ["input" if port == "target" else port for port in input_ports]
        nodes[node_id] = ComponentSpec(
            type=next_type,
            params=params,
            input_ports=input_ports,
            output_ports=list(node_spec.output_ports),
        )
        if was_runtime_network and node_id not in (graph.subgraphs or {}):
            generated_subgraphs[node_id] = _standard_network_subgraph(node_id, params)

    def rename_port(node_name: str, port: str) -> str:
        node = nodes.get(node_name)
        if node and node.type == "Network" and port == "target":
            return "input"
        return port

    wires = [
        WireSpec(
            source_node=wire.source_node,
            source_port=rename_port(wire.source_node, wire.source_port),
            target_node=wire.target_node,
            target_port=rename_port(wire.target_node, wire.target_port),
        )
        for wire in graph.wires
    ]
    input_bindings = {
        ("input" if name == "target" else name): (
            node,
            rename_port(node, port),
        )
        for name, (node, port) in graph.input_bindings.items()
    }
    subgraphs = None
    if graph.subgraphs or generated_subgraphs:
        subgraphs = {
            **generated_subgraphs,
            **(
                {
            node_id: normalize_graph_for_studio_authoring(subgraph)
            for node_id, subgraph in graph.subgraphs.items()
                }
                if graph.subgraphs
                else {}
            ),
        }
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
                    "id": (
                        f"task:{binding.source_data_id}->"
                        f"{binding.target_node_id}:input"
                    ),
                    "target_port": "input",
                }
            )
            changed = True
        bindings.append(binding)
    return task_binding_spec.model_copy(update={"bindings": bindings}) if changed else task_binding_spec


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
