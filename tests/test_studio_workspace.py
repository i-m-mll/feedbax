from __future__ import annotations

import json

import pytest

from feedbax.contracts.graph import (
    ANALYSIS_CANVAS_LAYOUT_SCHEMA_ID,
    ANALYSIS_CANVAS_LAYOUT_SCHEMA_VERSION,
    GraphMetadata,
    GraphSpec,
    GraphUIState,
    STUDIO_PERSISTENCE_DOCUMENT_SCHEMA_ID,
    STUDIO_PERSISTENCE_DOCUMENT_SCHEMA_VERSION,
    build_default_studio_workspace,
)
from feedbax.web.services.graph_service import GraphService
from feedbax.contracts.canonical_json import CanonicalJsonError
from feedbax.contracts.migrations import (
    UnsupportedSpecVersion,
    admit_studio_persistence_document,
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


def _save_payload(**fields: object) -> dict[str, object]:
    return {
        "schema_id": STUDIO_PERSISTENCE_DOCUMENT_SCHEMA_ID,
        "schema_version": STUDIO_PERSISTENCE_DOCUMENT_SCHEMA_VERSION,
        **fields,
    }


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


def test_legacy_project_load_materializes_workspace(tmp_path):
    service = GraphService(storage_dir=tmp_path)
    graph_id = "legacy-project"
    graph = _graph()
    payload = {
        "metadata": graph.metadata.model_dump(),
        "graph": graph.model_dump(),
        "ui_state": _ui_state().model_dump(),
        "analysis_pages": [
            {
                "id": "analysis-page",
                "name": "Existing analysis",
                "graph_spec": {},
                "eval_params": {"condition": "baseline"},
                "viewport": {"x": 0, "y": 0, "zoom": 1},
                "eval_run_id": "eval-1",
                "expanded_field_paths": ["states.arm"],
            }
        ],
        "active_analysis_page_id": "analysis-page",
    }
    (tmp_path / f"{graph_id}.json").write_text(json.dumps(payload), encoding="utf-8")

    record = service.get_graph(graph_id)

    assert record.project.schema_id == "feedbax.spec.studio.graph_project"
    assert record.project.schema_version == "feedbax.spec.studio.graph_project.v1"
    assert record.project.workspace is not None
    analysis_stage = next(
        stage for stage in record.project.workspace.stages if stage.kind == "analysis"
    )
    analysis_scenario = record.project.workspace.scenarios[analysis_stage.scenario_id]
    assert analysis_scenario.analysis_spec["active_page_id"] == "analysis-page"
    assert analysis_scenario.analysis_spec["pages"][0]["eval_run_id"] == "eval-1"
    assert "viewport" not in analysis_scenario.analysis_spec["pages"][0]
    assert record.project.workspace_document.analysis_canvas_layout.model_dump(mode="json") == {
        "schema_id": ANALYSIS_CANVAS_LAYOUT_SCHEMA_ID,
        "schema_version": ANALYSIS_CANVAS_LAYOUT_SCHEMA_VERSION,
        "stages": {},
    }


def test_old_workspace_document_admission_explicitly_materializes_empty_analysis_layout(tmp_path):
    graph = _graph()
    service = GraphService(storage_dir=tmp_path)
    old_document = service._workspace_document(graph).model_dump(mode="json")
    old_document.pop("analysis_canvas_layout")

    admitted = admit_studio_persistence_document(
        _save_payload(
            graph=graph.model_dump(mode="json"),
            workspace_document=old_document,
        )
    )

    assert admitted.workspace_document is not None
    assert admitted.workspace_document.analysis_canvas_layout.model_dump(mode="json") == {
        "schema_id": ANALYSIS_CANVAS_LAYOUT_SCHEMA_ID,
        "schema_version": ANALYSIS_CANVAS_LAYOUT_SCHEMA_VERSION,
        "stages": {},
    }


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), 2**53])
def test_legacy_project_load_rejects_unsafely_encoded_numbers(tmp_path, invalid):
    service = GraphService(storage_dir=tmp_path)
    graph_id = "unsafe-legacy-project"
    graph = _graph()
    payload = {
        "metadata": graph.metadata.model_dump(),
        "graph": graph.model_dump(),
        "ui_state": {
            **_ui_state().model_dump(),
            "viewport": {"x": invalid, "y": 0, "zoom": 1},
        },
    }
    (tmp_path / f"{graph_id}.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CanonicalJsonError):
        service.get_graph(graph_id)


def test_legacy_project_load_does_not_generate_network_subgraph(tmp_path):
    service = GraphService(storage_dir=tmp_path)
    graph_id = "legacy-runtime-network"
    graph = _runtime_network_graph()
    payload = {
        "metadata": graph.metadata.model_dump(),
        "graph": graph.model_dump(),
        "ui_state": _ui_state().model_dump(),
    }
    (tmp_path / f"{graph_id}.json").write_text(json.dumps(payload), encoding="utf-8")

    record = service.get_graph(graph_id)

    assert record.project.graph.nodes["network"].type == "SimpleStagedNetwork"
    assert record.project.graph.nodes["network"].input_ports == ["target"]
    assert record.project.graph.input_ports == ["target"]
    assert record.project.graph.input_bindings == {"target": ("network", "target")}
    assert record.project.graph.subgraphs is None
    train_stage = next(stage for stage in record.project.workspace.stages if stage.kind == "train")
    assert (
        "graph" not in record.project.workspace.scenarios[train_stage.scenario_id].model_fields_set
    )


def test_project_load_migrates_workspace_task_binding_spec(tmp_path):
    service = GraphService(storage_dir=tmp_path)
    graph_id = "workspace-task-binding-v1"
    graph = _graph()
    payload = {
        "metadata": graph.metadata.model_dump(),
        "graph": graph.model_dump(),
        "workspace": {
            "id": "workspace:legacy-bindings",
            "schema_version": "feedbax.spec.studio.workspace.v1",
            "label": "Legacy bindings",
            "active_stage_id": "stage:train",
            "ui_state": {"top_pane": {"kind": "model"}},
            "stages": [
                {
                    "id": "stage:train",
                    "kind": "train",
                    "label": "Train",
                    "scenario_id": "scenario:train",
                    "ui_state": {"collapsed": True},
                }
            ],
            "scenarios": {
                "scenario:train": {
                    "id": "scenario:train",
                    "schema_version": "feedbax.spec.studio.scenario.v1",
                    "label": "Train",
                    "stage_id": "stage:train",
                    "graph": graph.model_dump(),
                    "ui_state": {"workspace_view_state": {"mode": "model"}},
                    "task_binding_spec": {
                        "schema_version": "feedbax.studio.task_bindings.v1",
                        "exposed_outputs": [],
                        "bindings": [
                            {
                                "id": "task:inputs->network:input",
                                "source_output_id": "inputs",
                                "target_node_id": "network",
                                "target_port": "input",
                                "role": "model_input",
                                "metadata": {},
                            }
                        ],
                        "metadata": {},
                    },
                }
            },
        },
    }
    (tmp_path / f"{graph_id}.json").write_text(json.dumps(payload), encoding="utf-8")

    record = service.get_graph(graph_id)

    workspace = record.project.workspace
    assert workspace is not None
    scenario = workspace.scenarios["scenario:train"]
    assert scenario.task_binding_spec is not None
    assert scenario.task_binding_spec.schema_version == "feedbax.spec.studio.task_bindings.v3"
    assert scenario.task_binding_spec.bindings[0].source_data_id == "inputs"
    assert record.project.workspace_document.workspace_ui_state == {"top_pane": {"kind": "model"}}
    assert record.project.workspace_document.stage_ui_state == {"stage:train": {"collapsed": True}}
    assert record.project.workspace_document.scenario_ui_state == {
        "scenario:train": {"workspace_view_state": {"mode": "model"}}
    }
    assert "ui_state" not in record.project.workspace.model_dump()
    assert "ui_state" not in record.project.workspace.stages[0].model_dump()
    assert "ui_state" not in scenario.model_dump()


def test_current_workspace_and_stage_require_exact_schema_identity():
    current = build_default_studio_workspace(label="identity")
    payload = current.model_dump(mode="json")
    payload.pop("schema_id")

    with pytest.raises(UnsupportedSpecVersion, match="exact schema identity"):
        admit_studio_persistence_document(_save_payload(workspace=payload))

    payload = current.model_dump(mode="json")
    payload["stages"][0].pop("schema_id")
    with pytest.raises(UnsupportedSpecVersion, match="exact schema identity"):
        admit_studio_persistence_document(_save_payload(workspace=payload))

    payload = current.model_dump(mode="json")
    payload.pop("schema_version")
    with pytest.raises(UnsupportedSpecVersion, match="declare their schema version"):
        admit_studio_persistence_document(_save_payload(workspace=payload))

    payload = current.model_dump(mode="json")
    payload["stages"][0].pop("schema_version")
    with pytest.raises(UnsupportedSpecVersion, match="declare their schema version"):
        admit_studio_persistence_document(_save_payload(workspace=payload))


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), 2**53])
def test_studio_save_refuses_nonfinite_and_unsafe_numbers_before_validation(invalid):
    with pytest.raises(CanonicalJsonError):
        admit_studio_persistence_document(
            _save_payload(
                workspace={
                    "schema_id": "feedbax.spec.studio.workspace",
                    "schema_version": "feedbax.spec.studio.workspace.v1",
                    "metadata": {"invalid": invalid},
                }
            )
        )
