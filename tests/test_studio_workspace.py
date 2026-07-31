from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.graph import (
    ComponentSpec,
    GraphMetadata,
    GraphSpec,
    GraphUIState,
    StudioStageSpec,
    StudioWorkspaceSpec,
)
from feedbax.contracts.array_values import (
    ARRAY_VALUE_SCHEMA_ID,
    ARRAY_VALUE_SCHEMA_VERSION,
)
from feedbax.web.app import create_app
import feedbax.web.api.graphs as graphs_api
from feedbax.web.services.graph_service import GraphSaveConflictError
from feedbax.web.services.graph_service import GraphService


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


def test_graph_ui_state_accepts_payloads_without_assembly_view():
    state = GraphUIState.model_validate(
        {
            "viewport": {"x": 1, "y": 2, "zoom": 0.75},
            "node_states": {},
        }
    )

    assert state.assembly_view is None
    assert state.viewport.x == 1


def test_create_graph_persists_default_workspace(tmp_path):
    service = GraphService(storage_dir=tmp_path)

    record = service.create_graph(_graph(), _ui_state())

    workspace = record.project.workspace
    assert workspace is not None
    assert workspace.schema_version == "feedbax.spec.studio.workspace.v1"
    assert workspace.active_stage_id == "stage:train"
    assert [stage.kind for stage in workspace.stages] == [
        "train",
        "eval",
        "analysis",
        "report",
    ]

    train_stage = next(stage for stage in workspace.stages if stage.kind == "train")
    train_scenario = workspace.scenarios[train_stage.scenario_id]
    assert train_scenario.graph == record.project.graph
    assert train_scenario.graph_ui_state == record.project.ui_state

    reloaded = service.get_graph(record.graph_id)
    assert reloaded.project.workspace == workspace


def test_studio_save_load_materializes_dynamic_ports_with_explicit_registry(tmp_path):
    service = GraphService(storage_dir=tmp_path)
    registry = ComponentRegistry(load_user_components=False)
    graph = GraphSpec(
        nodes={
            "mux": ComponentSpec(type="Mux", params={"n_inputs": 3}),
        },
        output_ports=["output"],
        output_bindings={"output": ("mux", "output")},
    )

    record = service.create_graph(
        graph,
        _ui_state(),
        component_registry=registry,
    )
    reloaded = service.get_graph(
        record.graph_id,
        component_registry=registry,
    )

    node = reloaded.project.graph.nodes["mux"]
    assert node.input_ports == ["in_0", "in_1", "in_2"]
    assert node.output_ports == ["output"]
    train_stage = next(
        stage for stage in reloaded.project.workspace.stages if stage.kind == "train"
    )
    scenario = reloaded.project.workspace.scenarios[train_stage.scenario_id]
    assert scenario.graph.nodes["mux"].input_ports == node.input_ports


def test_studio_save_load_preserves_read_only_array_value_envelopes(tmp_path):
    service = GraphService(storage_dir=tmp_path)
    declaration = {
        "schema_id": ARRAY_VALUE_SCHEMA_ID,
        "schema_version": ARRAY_VALUE_SCHEMA_VERSION,
        "encoding": "constant",
        "shape": [3, 2],
        "dtype": "float32",
        "nonfinite": "forbid",
        "value": 1.5,
    }
    graph = _graph().model_copy(
        update={
            "nodes": {
                "fixture": ComponentSpec(
                    type="fixture.Component",
                    params={"nested": {"matrix": declaration}},
                )
            }
        }
    )

    record = service.create_graph(graph, _ui_state())
    reloaded = service.get_graph(record.graph_id)

    assert reloaded.project.graph.nodes["fixture"].params["nested"]["matrix"] == declaration
    assert (
        reloaded.project.workspace.scenarios["scenario:train"]
        .graph.nodes["fixture"]
        .params["nested"]["matrix"]
        == declaration
    )


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

    assert record.project.workspace is not None
    analysis_stage = next(
        stage for stage in record.project.workspace.stages if stage.kind == "analysis"
    )
    analysis_scenario = record.project.workspace.scenarios[analysis_stage.scenario_id]
    assert analysis_scenario.analysis_spec["active_page_id"] == "analysis-page"
    assert analysis_scenario.analysis_spec["pages"][0]["eval_run_id"] == "eval-1"


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
    assert record.project.graph.input_bindings == {"input": ("network", "target")}
    assert record.project.graph.subgraphs is None
    train_stage = next(stage for stage in record.project.workspace.stages if stage.kind == "train")
    train_graph = record.project.workspace.scenarios[train_stage.scenario_id].graph
    assert train_graph is not None
    assert train_graph.nodes["network"].type == "SimpleStagedNetwork"
    assert train_graph.subgraphs is None


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
            "stages": [
                {
                    "id": "stage:train",
                    "kind": "train",
                    "label": "Train",
                    "scenario_id": "scenario:train",
                }
            ],
            "scenarios": {
                "scenario:train": {
                    "id": "scenario:train",
                    "schema_version": "feedbax.spec.studio.scenario.v1",
                    "label": "Train",
                    "stage_id": "stage:train",
                    "graph": graph.model_dump(),
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
    assert scenario.task_binding_spec.schema_version == "feedbax.spec.studio.task_bindings.v2"
    assert scenario.task_binding_spec.bindings[0].source_data_id == "inputs"


def test_update_graph_preserves_explicit_workspace_extensions(tmp_path):
    service = GraphService(storage_dir=tmp_path)
    record = service.create_graph(_graph(), _ui_state())
    workspace = record.project.workspace
    assert workspace is not None
    workspace.stages.append(
        StudioStageSpec(
            id="stage:custom-protocol",
            kind="protocol",
            label="Custom protocol",
            metadata={"future_product_field": {"do_not_drop": True}},
        )
    )

    updated = service.update_graph(
        record.graph_id,
        None,
        None,
        workspace=StudioWorkspaceSpec.model_validate(workspace.model_dump()),
    )

    custom_stage = next(
        stage for stage in updated.project.workspace.stages if stage.id == "stage:custom-protocol"
    )
    assert custom_stage.metadata["future_product_field"]["do_not_drop"] is True


def test_update_graph_bumps_and_checks_save_revision(tmp_path):
    service = GraphService(storage_dir=tmp_path)
    record = service.create_graph(_graph(), _ui_state())

    assert record.project.metadata.save_revision == 0

    updated = service.update_graph(
        record.graph_id,
        _graph(),
        _ui_state(),
        expected_save_revision=0,
        require_save_revision=True,
    )

    assert updated.project.metadata.save_revision == 1
    assert updated.project.graph.metadata is not None
    assert updated.project.graph.metadata.save_revision == 1

    with pytest.raises(GraphSaveConflictError) as exc_info:
        service.update_graph(
            record.graph_id,
            _graph(),
            _ui_state(),
            expected_save_revision=0,
            require_save_revision=True,
        )

    assert exc_info.value.current_revision == 1
    assert exc_info.value.expected_revision == 0


def test_legacy_project_load_defaults_save_revision_and_updates(tmp_path):
    service = GraphService(storage_dir=tmp_path)
    graph_id = "legacy-no-save-revision"
    graph = _graph()
    payload = {
        "metadata": graph.metadata.model_dump(exclude={"save_revision"}),
        "graph": graph.model_dump(),
        "ui_state": _ui_state().model_dump(),
    }
    assert graph.metadata is not None
    payload["graph"]["metadata"].pop("save_revision", None)
    (tmp_path / f"{graph_id}.json").write_text(json.dumps(payload), encoding="utf-8")

    record = service.get_graph(graph_id)

    assert record.project.metadata.save_revision == 0
    assert record.project.graph.metadata is not None
    assert record.project.graph.metadata.save_revision == 0

    updated = service.update_graph(
        graph_id,
        _graph(),
        _ui_state(),
        expected_save_revision=0,
        require_save_revision=True,
    )

    assert updated.project.metadata.save_revision == 1


def test_graph_update_api_rejects_stale_and_missing_revisions(tmp_path, monkeypatch):
    service = GraphService(storage_dir=tmp_path)
    record = service.create_graph(_graph(), _ui_state())
    monkeypatch.setattr(graphs_api, "service", service)
    client = TestClient(create_app())

    missing = client.put(
        f"/api/graphs/{record.graph_id}",
        json={"graph": _graph().model_dump(), "ui_state": _ui_state().model_dump()},
    )
    assert missing.status_code == 409
    assert missing.json()["detail"]["current_save_revision"] == 0

    ok = client.put(
        f"/api/graphs/{record.graph_id}",
        headers={"If-Match": "0"},
        json={"graph": _graph().model_dump(), "ui_state": _ui_state().model_dump()},
    )
    assert ok.status_code == 200
    assert ok.json()["data"]["metadata"]["save_revision"] == 1

    stale = client.put(
        f"/api/graphs/{record.graph_id}",
        headers={"If-Match": "0"},
        json={"graph": _graph().model_dump(), "ui_state": _ui_state().model_dump()},
    )
    assert stale.status_code == 409
    assert stale.json()["detail"]["expected_save_revision"] == 0
    assert stale.json()["detail"]["current_save_revision"] == 1


def test_beacon_update_uses_payload_save_revision(tmp_path, monkeypatch):
    service = GraphService(storage_dir=tmp_path)
    record = service.create_graph(_graph(), _ui_state())
    monkeypatch.setattr(graphs_api, "service", service)
    client = TestClient(create_app())

    stale = client.post(
        f"/api/graphs/{record.graph_id}/beacon",
        json={
            "graph": _graph().model_dump(),
            "ui_state": _ui_state().model_dump(),
            "expected_save_revision": 1,
        },
    )
    assert stale.status_code == 409

    ok = client.post(
        f"/api/graphs/{record.graph_id}/beacon",
        json={
            "graph": _graph().model_dump(),
            "ui_state": _ui_state().model_dump(),
            "expected_save_revision": 0,
        },
    )
    assert ok.status_code == 204
    assert service.get_graph(record.graph_id).project.metadata.save_revision == 1
