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


def _recurrent_zero_initializer(width: int | None = None, *, state_slot: str) -> dict:
    initializer = {
        "kind": "zeros",
        "scope": "trial",
        "source": "state_initializer",
        "state_slot": state_slot,
    }
    if width is not None:
        initializer["shape"] = [width]
    return initializer


def _network_recurrent_wires(cell_type: str, hidden_size: int) -> list[WireSpec]:
    wires = [
        WireSpec(
            source_node="cell",
            source_port="hidden",
            target_node="cell",
            target_port="hidden",
            temporality="recurrent",
            recurrent_initializer=_recurrent_zero_initializer(
                hidden_size,
                state_slot="hidden",
            ),
        )
    ]
    if cell_type == "LSTM":
        wires.append(
            WireSpec(
                source_node="cell",
                source_port="cell",
                target_node="cell",
                target_port="cell",
                temporality="recurrent",
                recurrent_initializer=_recurrent_zero_initializer(
                    hidden_size,
                    state_slot="cell",
                ),
            )
        )
    return wires


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
            "input_mux": ComponentSpec(
                type="Mux",
                params={"n_inputs": 2},
                input_ports=["in_0", "in_1"],
                output_ports=["output"],
            ),
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
                source_node="input_mux",
                source_port="output",
                target_node="cell",
                target_port="input",
            ),
            WireSpec(
                source_node="cell",
                source_port="output",
                target_node="readout",
                target_port="input",
            ),
            *_network_recurrent_wires(cell_type, hidden_size),
        ],
        input_ports=["input", "feedback"],
        output_ports=["output", "hidden"],
        input_bindings={"input": ("input_mux", "in_0"), "feedback": ("input_mux", "in_1")},
        output_bindings={
            "output": ("readout", "output"),
            "hidden": ("cell", "hidden"),
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


def _network_input_ports(input_ports: list[str]) -> list[str]:
    ports = [
        "input" if port == "target" else port
        for port in input_ports
        if port not in {"hidden", "cell"}
    ]
    if "input" not in ports:
        ports.insert(0, "input")
    if "feedback" not in ports:
        ports.append("feedback")
    return list(dict.fromkeys(ports))


def _network_output_ports(output_ports: list[str]) -> list[str]:
    ports = list(output_ports) or ["output"]
    if "output" not in ports:
        ports.insert(0, "output")
    if "hidden" not in ports:
        ports.append("hidden")
    return list(dict.fromkeys(ports))


def _unwrap_legacy_network_model_layer(subgraph: GraphSpec) -> GraphSpec:
    """Collapse the old Network -> model wrapper when it is only a boundary proxy."""
    if set(subgraph.nodes) != {"model"}:
        return subgraph
    model_node = subgraph.nodes["model"]
    inner = (subgraph.subgraphs or {}).get("model")
    if model_node.type != "Subgraph" or inner is None or subgraph.wires:
        return subgraph

    forwards_inputs = all(
        binding == ("model", port) for port, binding in subgraph.input_bindings.items()
    )
    forwards_outputs = all(
        binding == ("model", port) for port, binding in subgraph.output_bindings.items()
    )
    if not forwards_inputs or not forwards_outputs:
        return subgraph

    return inner.model_copy(
        update={
            "input_ports": list(dict.fromkeys([*subgraph.input_ports, *inner.input_ports])),
            "output_ports": list(dict.fromkeys([*subgraph.output_ports, *inner.output_ports])),
            "metadata": subgraph.metadata or inner.metadata,
        }
    )


def _instant_path_exists(wires: list[WireSpec], source_node: str, target_node: str) -> bool:
    adjacency: dict[str, set[str]] = {}
    for wire in wires:
        if wire.temporality == "recurrent":
            continue
        adjacency.setdefault(wire.source_node, set()).add(wire.target_node)

    visited: set[str] = set()
    stack = [source_node]
    while stack:
        node = stack.pop()
        if node == target_node:
            return True
        if node in visited:
            continue
        visited.add(node)
        stack.extend(adjacency.get(node, set()))
    return False


def _normalize_recurrent_feedback_wire(wire: WireSpec, nodes: dict[str, ComponentSpec]) -> WireSpec:
    target = nodes.get(wire.target_node)
    if (
        wire.temporality == "instant"
        and target is not None
        and target.type == "Network"
        and wire.source_port == "output"
        and wire.target_port == "feedback"
    ):
        return wire.model_copy(
            update={
                "temporality": "recurrent",
                "recurrent_initializer": _recurrent_zero_initializer(state_slot="feedback"),
            }
        )
    return wire


def _normalize_network_subgraph(subgraph: GraphSpec) -> GraphSpec:
    subgraph = _unwrap_legacy_network_model_layer(subgraph)
    input_binding = subgraph.input_bindings.get("input")
    feedback_binding = subgraph.input_bindings.get("feedback")
    cell_id = None
    if input_binding and subgraph.nodes.get(input_binding[0], None) is not None:
        candidate = subgraph.nodes[input_binding[0]]
        if candidate.type in {"GRU", "LSTM"}:
            cell_id = input_binding[0]
    if cell_id is None:
        cell_id = next(
            (node_id for node_id, node in subgraph.nodes.items() if node.type in {"GRU", "LSTM"}),
            None,
        )
    if cell_id is None:
        return subgraph
    cell_type = subgraph.nodes[cell_id].type
    hidden_size = _int_param(subgraph.nodes[cell_id].params.get("hidden_size"), 100)
    has_mux_wire = any(
        wire.source_node == "input_mux"
        and wire.source_port == "output"
        and wire.target_node == cell_id
        and wire.target_port == "input"
        for wire in subgraph.wires
    )
    recurrent_wires = _network_recurrent_wires(cell_type, hidden_size)
    recurrent_targets = {(wire.target_node, wire.target_port) for wire in recurrent_wires}
    existing_recurrent_targets = {
        (wire.target_node, wire.target_port)
        for wire in subgraph.wires
        if wire.temporality == "recurrent"
    }
    missing_recurrent_wires = [
        wire
        for wire in recurrent_wires
        if (wire.target_node, wire.target_port) not in existing_recurrent_targets
    ]
    needs_hidden_binding = subgraph.output_bindings.get("hidden") != (cell_id, "hidden")
    if (
        "input_mux" in subgraph.nodes
        and has_mux_wire
        and feedback_binding
        and input_binding == ("input_mux", "in_0")
        and not missing_recurrent_wires
        and not needs_hidden_binding
    ):
        return subgraph.model_copy(
            update={
                "input_ports": _network_input_ports(list(subgraph.input_ports)),
                "output_ports": _network_output_ports(list(subgraph.output_ports)),
            }
        )
    nodes = dict(subgraph.nodes)
    nodes.setdefault(
        "input_mux",
        ComponentSpec(
            type="Mux",
            params={"n_inputs": 2},
            input_ports=["in_0", "in_1"],
            output_ports=["output"],
        ),
    )
    wires = [
        wire
        for wire in subgraph.wires
        if not (wire.target_node == cell_id and wire.target_port == "input")
        and (wire.target_node, wire.target_port) not in recurrent_targets
    ]
    wires.append(
        WireSpec(
            source_node="input_mux",
            source_port="output",
            target_node=cell_id,
            target_port="input",
        )
    )
    wires.extend(recurrent_wires)
    input_bindings = dict(subgraph.input_bindings)
    input_bindings["input"] = ("input_mux", "in_0")
    input_bindings["feedback"] = ("input_mux", "in_1")
    output_bindings = dict(subgraph.output_bindings)
    if needs_hidden_binding:
        output_bindings["hidden"] = (cell_id, "hidden")
    return subgraph.model_copy(
        update={
            "nodes": nodes,
            "wires": wires,
            "input_ports": _network_input_ports(list(subgraph.input_ports)),
            "output_ports": _network_output_ports(list(subgraph.output_ports)),
            "input_bindings": input_bindings,
            "output_bindings": output_bindings,
        }
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
        output_ports = list(node_spec.output_ports)
        if next_type == "Network":
            if node_id in (graph.subgraphs or {}) or was_runtime_network:
                input_ports = ["input" if port == "target" else port for port in input_ports]
            else:
                input_ports = _network_input_ports(input_ports)
                output_ports = _network_output_ports(output_ports)
        nodes[node_id] = ComponentSpec(
            type=next_type,
            params=params,
            input_ports=input_ports,
            output_ports=output_ports,
        )
        if was_runtime_network and node_id not in (graph.subgraphs or {}):
            generated_subgraphs[node_id] = _standard_network_subgraph(node_id, params)

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
        if _instant_path_exists(
            [*wires, *graph.wires], normalized_wire.target_node, normalized_wire.source_node
        ):
            normalized_wire = _normalize_recurrent_feedback_wire(normalized_wire, nodes)
        wires.append(normalized_wire)
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
                    node_id: (
                        _normalize_network_subgraph(normalize_graph_for_studio_authoring(subgraph))
                        if nodes.get(node_id) is not None and nodes[node_id].type == "Network"
                        else normalize_graph_for_studio_authoring(subgraph)
                    )
                    for node_id, subgraph in graph.subgraphs.items()
                }
                if graph.subgraphs
                else {}
            ),
        }
        for node_id, subgraph in subgraphs.items():
            node = nodes.get(node_id)
            if node is not None and node.type == "Network":
                nodes[node_id] = node.model_copy(
                    update={
                        "input_ports": list(subgraph.input_ports),
                        "output_ports": list(subgraph.output_ports),
                    }
                )
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
