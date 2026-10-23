from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.graph import (
    ANALYSIS_CANVAS_LAYOUT_SCHEMA_ID,
    ANALYSIS_CANVAS_LAYOUT_SCHEMA_VERSION,
    ComponentSpec,
    GraphMetadata,
    GraphSpec,
    GraphUIState,
    StudioStageSpec,
    StudioWorkspaceSpec,
    STUDIO_PERSISTENCE_DOCUMENT_SCHEMA_ID,
    STUDIO_PERSISTENCE_DOCUMENT_SCHEMA_VERSION,
    build_default_studio_workspace,
)
from feedbax.contracts.array_values import (
    ARRAY_VALUE_SCHEMA_ID,
    ARRAY_VALUE_SCHEMA_VERSION,
)
from feedbax.web.app import create_app
import feedbax.web.api.graphs as graphs_api
from feedbax.web.services.graph_service import GraphSaveConflictError
from feedbax.web.services.graph_service import GraphService
from feedbax.contracts.canonical_json import CanonicalJsonError
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
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


def test_create_graph_persists_default_workspace(tmp_path):
    service = GraphService(storage_dir=tmp_path)

    record = service.create_graph(_graph())

    assert record.project.schema_id == "feedbax.spec.studio.graph_project"
    assert record.project.schema_version == "feedbax.spec.studio.graph_project.v1"
    workspace = record.project.workspace
    assert workspace is not None
    assert workspace.schema_version == "feedbax.spec.studio.workspace.v2"
    assert workspace.active_stage_id == "stage:train"
    assert [stage.kind for stage in workspace.stages] == [
        "train",
        "eval",
        "analysis",
        "report",
    ]

    train_stage = next(stage for stage in workspace.stages if stage.kind == "train")
    train_scenario = workspace.scenarios[train_stage.scenario_id]
    assert "graph" not in train_scenario.model_fields_set
    assert record.project.workspace_document.graph_ui_state == _ui_state()

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
    assert "graph" not in scenario.model_fields_set
    assert "/graph/nodes/mux" in reloaded.project.workspace_document.semantic_anchors


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

    record = service.create_graph(graph)
    reloaded = service.get_graph(record.graph_id)

    assert reloaded.project.graph.nodes["fixture"].params["nested"]["matrix"] == declaration
    assert "graph" not in reloaded.project.workspace.scenarios["scenario:train"].model_fields_set


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


def test_analysis_canvas_layout_round_trip_prunes_stale_semantic_keys(tmp_path, monkeypatch):
    service = GraphService(storage_dir=tmp_path)
    record = service.create_graph(_graph())
    monkeypatch.setattr(graphs_api, "service", service)
    document = record.project.workspace_document.model_dump(mode="json")
    document["analysis_pages"] = [_analysis_page()]
    document["active_analysis_page_id"] = "analysis-page"
    document["analysis_canvas_layout"] = {
        "schema_id": ANALYSIS_CANVAS_LAYOUT_SCHEMA_ID,
        "schema_version": ANALYSIS_CANVAS_LAYOUT_SCHEMA_VERSION,
        "stages": {
            "stage:analysis": {
                "pages": {
                    "analysis-page": {
                        "node_positions": {
                            "analysis-node": {"x": 321.5, "y": -88.25},
                            "__data_source__": {"x": 16, "y": 48},
                            "stale-node": {"x": 999, "y": 999},
                        },
                        "viewport": {"x": -110, "y": 72, "zoom": 1.4},
                    }
                }
            }
        },
    }

    response = TestClient(create_app()).put(
        f"/api/graphs/{record.graph_id}",
        headers={"If-Match": "0"},
        json=_save_payload(
            graph=record.project.graph.model_dump(mode="json"),
            workspace_document=document,
            workspace=record.project.workspace.model_dump(mode="json"),
        ),
    )

    assert response.status_code == 200
    reloaded = service.get_graph(record.graph_id).project.workspace_document
    page_layout = reloaded.analysis_canvas_layout.stages["stage:analysis"].pages["analysis-page"]
    assert page_layout.model_dump(mode="json") == {
        "node_positions": {
            "analysis-node": {"x": 321.5, "y": -88.25},
            "__data_source__": {"x": 16.0, "y": 48.0},
        },
        "viewport": {"x": -110.0, "y": 72.0, "zoom": 1.4},
    }


@pytest.mark.parametrize(
    "invalid_layout",
    [
        {
            "schema_id": ANALYSIS_CANVAS_LAYOUT_SCHEMA_ID,
            "schema_version": "feedbax.spec.studio.analysis_canvas_layout.v99",
            "stages": {},
        },
        {
            "schema_id": ANALYSIS_CANVAS_LAYOUT_SCHEMA_ID,
            "stages": {},
        },
        {
            "schema_id": "feedbax.spec.studio.some_other_layout",
            "schema_version": ANALYSIS_CANVAS_LAYOUT_SCHEMA_VERSION,
            "stages": {},
        },
        {
            "schema_id": ANALYSIS_CANVAS_LAYOUT_SCHEMA_ID,
            "schema_version": ANALYSIS_CANVAS_LAYOUT_SCHEMA_VERSION,
            "stages": {
                "stage:analysis": {
                    "pages": {
                        "analysis-page": {
                            "node_positions": {"analysis-node": {"x": 10_000_001, "y": 0}},
                            "viewport": {"x": 0, "y": 0, "zoom": 1},
                        }
                    }
                }
            },
        },
        {
            "schema_id": ANALYSIS_CANVAS_LAYOUT_SCHEMA_ID,
            "schema_version": ANALYSIS_CANVAS_LAYOUT_SCHEMA_VERSION,
            "stages": {
                "stage:analysis": {
                    "pages": {
                        "analysis-page": {
                            "node_positions": {},
                            "viewport": {"x": 0, "y": 0, "zoom": 0},
                        }
                    }
                }
            },
        },
    ],
)
def test_invalid_analysis_canvas_layout_is_rejected_atomically(
    tmp_path,
    monkeypatch,
    invalid_layout,
):
    service = GraphService(storage_dir=tmp_path)
    record = service.create_graph(_graph())
    monkeypatch.setattr(graphs_api, "service", service)
    path = tmp_path / f"{record.graph_id}.json"
    before = path.read_bytes()
    document = record.project.workspace_document.model_dump(mode="json")
    document["analysis_pages"] = [_analysis_page()]
    document["analysis_canvas_layout"] = invalid_layout

    response = TestClient(create_app()).put(
        f"/api/graphs/{record.graph_id}",
        headers={"If-Match": "0"},
        json=_save_payload(
            graph=record.project.graph.model_dump(mode="json"),
            workspace_document=document,
            workspace=record.project.workspace.model_dump(mode="json"),
        ),
    )

    assert response.status_code == 422
    assert path.read_bytes() == before


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
    assert record.project.graph.input_bindings == {"input": ("network", "target")}
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
    assert scenario.task_binding_spec.schema_version == "feedbax.spec.studio.task_bindings.v2"
    assert scenario.task_binding_spec.bindings[0].source_data_id == "inputs"
    assert record.project.workspace_document.workspace_ui_state == {"top_pane": {"kind": "model"}}
    assert record.project.workspace_document.stage_ui_state == {"stage:train": {"collapsed": True}}
    assert record.project.workspace_document.scenario_ui_state == {
        "scenario:train": {"workspace_view_state": {"mode": "model"}}
    }
    assert "ui_state" not in record.project.workspace.model_dump()
    assert "ui_state" not in record.project.workspace.stages[0].model_dump()
    assert "ui_state" not in scenario.model_dump()


def test_save_ingress_migrates_workspace_and_stage_as_one_document(tmp_path, monkeypatch):
    service = GraphService(storage_dir=tmp_path)
    record = service.create_graph(_graph())
    monkeypatch.setattr(graphs_api, "service", service)
    legacy_workspace = record.project.workspace.model_dump(mode="json")
    legacy_workspace.pop("schema_id")
    legacy_workspace["schema_version"] = "feedbax.spec.studio.workspace.v1"
    legacy_workspace["ui_state"] = {"pane": "model"}
    for stage in legacy_workspace["stages"]:
        stage.pop("schema_id")
        stage["schema_version"] = "feedbax.spec.studio.stage.v1"
        stage["ui_state"] = {"collapsed": True}

    response = TestClient(create_app()).put(
        f"/api/graphs/{record.graph_id}",
        headers={"If-Match": "0"},
        json=_save_payload(
            graph=record.project.graph.model_dump(mode="json"),
            workspace_document=record.project.workspace_document.model_dump(mode="json"),
            workspace=legacy_workspace,
        ),
    )

    assert response.status_code == 200
    stored = service.get_graph(record.graph_id).project
    assert stored.workspace.schema_id == "feedbax.spec.studio.workspace"
    assert stored.workspace.schema_version == "feedbax.spec.studio.workspace.v2"
    assert all(stage.schema_id == "feedbax.spec.studio.stage" for stage in stored.workspace.stages)
    assert all(
        stage.schema_version == "feedbax.spec.studio.stage.v2" for stage in stored.workspace.stages
    )


@pytest.mark.parametrize(
    ("location", "version"),
    [
        ("workspace", "feedbax.spec.studio.workspace.v99"),
        ("stage", "feedbax.spec.studio.stage.v99"),
    ],
)
def test_future_workspace_or_stage_version_is_rejected_before_write(
    tmp_path,
    monkeypatch,
    location,
    version,
):
    service = GraphService(storage_dir=tmp_path)
    record = service.create_graph(_graph())
    monkeypatch.setattr(graphs_api, "service", service)
    path = tmp_path / f"{record.graph_id}.json"
    before = path.read_bytes()
    workspace = record.project.workspace.model_dump(mode="json")
    if location == "workspace":
        workspace["schema_version"] = version
    else:
        workspace["stages"][0]["schema_version"] = version

    response = TestClient(create_app()).put(
        f"/api/graphs/{record.graph_id}",
        headers={"If-Match": "0"},
        json=_save_payload(
            graph=record.project.graph.model_dump(mode="json"),
            workspace_document=record.project.workspace_document.model_dump(mode="json"),
            workspace=workspace,
        ),
    )

    assert response.status_code == 422
    assert path.read_bytes() == before


def test_future_persistence_document_version_is_rejected_before_write(tmp_path, monkeypatch):
    service = GraphService(storage_dir=tmp_path)
    record = service.create_graph(_graph())
    monkeypatch.setattr(graphs_api, "service", service)
    path = tmp_path / f"{record.graph_id}.json"
    before = path.read_bytes()
    payload = _save_payload(
        graph=record.project.graph.model_dump(mode="json"),
        workspace_document=record.project.workspace_document.model_dump(mode="json"),
        workspace=record.project.workspace.model_dump(mode="json"),
    )
    payload["schema_version"] = "feedbax.spec.studio.persistence_document.v99"

    response = TestClient(create_app()).put(
        f"/api/graphs/{record.graph_id}",
        headers={"If-Match": "0"},
        json=payload,
    )

    assert response.status_code == 422
    assert path.read_bytes() == before


def test_whole_save_validation_is_atomic_before_write(tmp_path, monkeypatch):
    service = GraphService(storage_dir=tmp_path)
    record = service.create_graph(_graph())
    monkeypatch.setattr(graphs_api, "service", service)
    path = tmp_path / f"{record.graph_id}.json"
    before = path.read_bytes()
    workspace = record.project.workspace.model_dump(mode="json")
    workspace["stages"][0].pop("label")

    response = TestClient(create_app()).put(
        f"/api/graphs/{record.graph_id}",
        headers={"If-Match": "0"},
        json=_save_payload(
            graph=record.project.graph.model_dump(mode="json"),
            workspace_document=record.project.workspace_document.model_dump(mode="json"),
            workspace=workspace,
        ),
    )

    assert response.status_code == 422
    assert path.read_bytes() == before


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


def test_update_graph_preserves_explicit_workspace_extensions(tmp_path):
    service = GraphService(storage_dir=tmp_path)
    record = service.create_graph(_graph())
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
        _graph(),
        workspace=StudioWorkspaceSpec.model_validate(workspace.model_dump()),
    )

    custom_stage = next(
        stage for stage in updated.project.workspace.stages if stage.id == "stage:custom-protocol"
    )
    assert custom_stage.metadata["future_product_field"]["do_not_drop"] is True


def test_update_graph_bumps_and_checks_save_revision(tmp_path):
    service = GraphService(storage_dir=tmp_path)
    record = service.create_graph(_graph())

    assert record.project.metadata.save_revision == 0

    updated = service.update_graph(
        record.graph_id,
        _graph(),
        expected_save_revision=0,
        require_save_revision=True,
    )

    assert updated.project.metadata.save_revision == 1
    assert updated.project.graph.metadata is not None
    assert updated.project.graph.metadata.save_revision == 0

    with pytest.raises(GraphSaveConflictError) as exc_info:
        service.update_graph(
            record.graph_id,
            _graph(),
            expected_save_revision=0,
            require_save_revision=True,
        )

    assert exc_info.value.current_revision == 1
    assert exc_info.value.expected_revision == 0


def test_workspace_only_save_preserves_semantic_graph_revision(tmp_path) -> None:
    service = GraphService(storage_dir=tmp_path)
    record = service.create_graph(_graph())
    original_root = record.project.workspace_document.semantic_root
    moved_document = record.project.workspace_document.model_copy(
        update={"graph_ui_state": GraphUIState(viewport={"x": 400, "y": 240, "zoom": 0.75})}
    )

    updated = service.update_graph(
        record.graph_id,
        _graph(),
        workspace_document=moved_document,
        expected_save_revision=0,
        require_save_revision=True,
    )

    assert updated.project.workspace_document.semantic_root == original_root
    assert updated.project.graph == record.project.graph


def test_analysis_layout_only_save_preserves_semantic_workspace_and_execution_bytes(tmp_path):
    service = GraphService(storage_dir=tmp_path)
    record = service.create_graph(_graph())
    document = record.project.workspace_document.model_dump(mode="json")
    document["analysis_pages"] = [_analysis_page()]
    document["active_analysis_page_id"] = "analysis-page"
    document["analysis_canvas_layout"] = {
        "schema_id": ANALYSIS_CANVAS_LAYOUT_SCHEMA_ID,
        "schema_version": ANALYSIS_CANVAS_LAYOUT_SCHEMA_VERSION,
        "stages": {
            "stage:analysis": {
                "pages": {
                    "analysis-page": {
                        "node_positions": {"analysis-node": {"x": 100, "y": 200}},
                        "viewport": {"x": 0, "y": 0, "zoom": 1},
                    }
                }
            }
        },
    }
    first = service.update_graph(
        record.graph_id,
        record.project.graph,
        workspace=record.project.workspace,
        workspace_document=type(record.project.workspace_document).model_validate(document),
        expected_save_revision=0,
        require_save_revision=True,
    )
    graph_bytes = canonical_json_v2_bytes(first.project.graph.model_dump(mode="json"))
    workspace_bytes = canonical_json_v2_bytes(first.project.workspace.model_dump(mode="json"))
    moved_document = first.project.workspace_document.model_dump(mode="json")
    moved_layout = moved_document["analysis_canvas_layout"]["stages"]["stage:analysis"][
        "pages"
    ]["analysis-page"]
    moved_layout["node_positions"]["analysis-node"] = {"x": -250, "y": 640}
    moved_layout["viewport"] = {"x": 40, "y": -30, "zoom": 0.75}

    second = service.update_graph(
        record.graph_id,
        first.project.graph,
        workspace=first.project.workspace,
        workspace_document=type(first.project.workspace_document).model_validate(moved_document),
        expected_save_revision=1,
        require_save_revision=True,
    )

    assert canonical_json_v2_bytes(second.project.graph.model_dump(mode="json")) == graph_bytes
    assert canonical_json_v2_bytes(second.project.workspace.model_dump(mode="json")) == workspace_bytes
    analysis_stage_before = next(
        stage for stage in first.project.workspace.stages if stage.kind == "analysis"
    )
    analysis_stage_after = next(
        stage for stage in second.project.workspace.stages if stage.kind == "analysis"
    )
    assert analysis_stage_after.execution_spec == analysis_stage_before.execution_spec


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
        expected_save_revision=0,
        require_save_revision=True,
    )

    assert updated.project.metadata.save_revision == 1


def test_graph_update_api_rejects_stale_and_missing_revisions(tmp_path, monkeypatch):
    service = GraphService(storage_dir=tmp_path)
    record = service.create_graph(_graph())
    monkeypatch.setattr(graphs_api, "service", service)
    client = TestClient(create_app())
    workspace_document = record.project.workspace_document.model_dump(mode="json")

    missing = client.put(
        f"/api/graphs/{record.graph_id}",
        json=_save_payload(
            graph=_graph().model_dump(),
            workspace_document=workspace_document,
        ),
    )
    assert missing.status_code == 409
    assert missing.json()["detail"]["current_save_revision"] == 0

    ok = client.put(
        f"/api/graphs/{record.graph_id}",
        headers={"If-Match": "0"},
        json=_save_payload(
            graph=_graph().model_dump(),
            workspace_document=workspace_document,
        ),
    )
    assert ok.status_code == 200
    assert ok.json()["data"]["metadata"]["save_revision"] == 1

    stale = client.put(
        f"/api/graphs/{record.graph_id}",
        headers={"If-Match": "0"},
        json=_save_payload(
            graph=_graph().model_dump(),
            workspace_document=workspace_document,
        ),
    )
    assert stale.status_code == 409
    assert stale.json()["detail"]["expected_save_revision"] == 0
    assert stale.json()["detail"]["current_save_revision"] == 1


def test_beacon_update_uses_payload_save_revision(tmp_path, monkeypatch):
    service = GraphService(storage_dir=tmp_path)
    record = service.create_graph(_graph())
    monkeypatch.setattr(graphs_api, "service", service)
    client = TestClient(create_app())
    workspace_document = record.project.workspace_document.model_dump(mode="json")

    stale = client.post(
        f"/api/graphs/{record.graph_id}/beacon",
        json=_save_payload(
            graph=_graph().model_dump(),
            workspace_document=workspace_document,
            expected_save_revision=1,
        ),
    )
    assert stale.status_code == 409

    ok = client.post(
        f"/api/graphs/{record.graph_id}/beacon",
        json=_save_payload(
            graph=_graph().model_dump(),
            workspace_document=workspace_document,
            expected_save_revision=0,
        ),
    )
    assert ok.status_code == 204
    assert service.get_graph(record.graph_id).project.metadata.save_revision == 1
