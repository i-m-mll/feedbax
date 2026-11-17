from __future__ import annotations


from feedbax.contracts.graph import (
    GraphMetadata,
    GraphSpec,
    GraphUIState,
)


def _graph() -> GraphSpec:
    return GraphSpec(
        metadata=GraphMetadata(
            name="Workspace smoke",
            created_at="2026-05-17T00:00:00+00:00",
            updated_at="2026-05-17T00:00:00+00:00",
        )
    )


def _runtime_network_graph() -> GraphSpec:
    return GraphSpec(
        nodes={
            "network": {
                "type": "SimpleStagedNetwork",
                "params": {
                    "hidden_size": 100,
                    "input_size": 4,
                    "output_size": 2,
                    "hidden_type": "GRUCell",
                },
                "input_ports": ["target"],
                "output_ports": ["output"],
            }
        },
        input_ports=["target"],
        input_bindings={"target": ("network", "target")},
        metadata=GraphMetadata(
            name="Runtime network",
            created_at="2026-05-17T00:00:00+00:00",
            updated_at="2026-05-17T00:00:00+00:00",
        ),
    )


def _ui_state() -> GraphUIState:
    return GraphUIState()




def _analysis_page() -> dict[str, object]:
    return {
        "id": "analysis-page",
        "name": "Analysis page",
        "graph_spec": {
            "nodes": {
                "analysis-node": {
                    "id": "analysis-node",
                    "type": "ActivityPlot",
                    "label": "Activity plot",
                    "category": "Figures",
                    "inputPorts": [],
                    "outputPorts": [],
                    "params": {},
                    "role": "analysis",
                }
            },
            "wires": [],
            "dataSourceId": "__data_source__",
        },
        "eval_params": {},
        "eval_run_id": None,
        "expanded_field_paths": [],
    }


def test_graph_ui_state_accepts_payloads_without_assembly_view():
    state = GraphUIState.model_validate(
        {
            "viewport": {"x": 1, "y": 2, "zoom": 0.75},
            "node_states": {},
        }
    )

    assert state.assembly_view is None
    assert state.viewport.x == 1
