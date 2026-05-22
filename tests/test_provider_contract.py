from __future__ import annotations

import sqlite3
import queue
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from feedbax.manifest import Provenance, load_manifest, sha256_file, write_training_run_manifest
from feedbax.manifest_index import rebuild_manifest_index
from feedbax.provider import (
    component_registry_snapshot,
    provider_manifest,
    validate_analysis_spec,
    validate_evaluation_spec,
    validate_graph_spec,
    validate_task_spec,
    validate_training_spec,
)
from feedbax.studio_schema import (
    RuntimeIntrospectionResult,
    RuntimeSampleLeafSchema,
    enumerate_studio_schema_registry,
    validate_graph_connection_schema,
)
from feedbax.web.graph_normalization import normalize_graph_for_studio_authoring
from feedbax.web.app import create_app
from feedbax.web.models.graph import (
    GraphMetadata,
    GraphSpec,
    StudioStageSpec,
    StudioTaskBindingSpec,
    StudioTaskTimelineSpec,
    TapSpec,
    build_default_studio_workspace,
)
from feedbax.web.worker.app import (
    WorkerStatus,
    _Job,
    _derive_worker_graph_lowering,
    _extract_graph_params,
    _extract_task_sampling_cfg,
    _extract_training_cfg,
    _require_worker_specs,
    _run_training,
    _run_training_stub,
)


def _minimal_graph_spec() -> dict:
    return {
        "nodes": {
            "gain": {
                "type": "Gain",
                "params": {"gain": 2.0},
                "input_ports": ["input"],
                "output_ports": ["output"],
            }
        },
        "wires": [],
        "input_ports": [],
        "output_ports": [],
        "input_bindings": {},
        "output_bindings": {},
    }


def _runtime_network_graph_spec() -> dict:
    return {
        "nodes": {
            "network": {
                "type": "SimpleStagedNetwork",
                "params": {
                    "hidden_size": 100,
                    "input_size": 4,
                    "output_size": 2,
                    "hidden_type": "GRUCell",
                    "out_nonlinearity": "tanh",
                },
                "input_ports": ["target"],
                "output_ports": ["output", "hidden"],
            },
            "mechanics": {
                "type": "PointMass",
                "params": {"dt": 0.02},
                "input_ports": ["force"],
                "output_ports": ["effector"],
            },
        },
        "wires": [
            {
                "source_node": "network",
                "source_port": "output",
                "target_node": "mechanics",
                "target_port": "force",
            }
        ],
        "input_ports": ["target"],
        "output_ports": ["effector"],
        "input_bindings": {"target": ("network", "target")},
        "output_bindings": {"effector": ("mechanics", "effector")},
    }


def _task_input_binding(
    *,
    target_node_id: str = "network",
    target_port: str = "input",
    expected_shape: list | None = None,
) -> dict:
    data = {
        "id": "inputs",
        "label": "Inputs",
        "kind": "signal",
        "role": "model_input",
        "path": "inputs",
        "bindable": True,
        "dtype": "vector",
        "metadata": {},
    }
    if expected_shape is not None:
        data["expected_shape"] = expected_shape
    return {
        "schema_version": "feedbax.studio.task_bindings.v2",
        "exposed_data": [data],
        "bindings": [
            {
                "id": f"task:inputs->{target_node_id}:{target_port}",
                "source_data_id": "inputs",
                "target_node_id": target_node_id,
                "target_port": target_port,
                "role": "model_input",
                "metadata": {},
            }
        ],
        "metadata": {},
    }


def _minimal_training_spec() -> dict:
    return {
        "optimizer": {"type": "adamw", "params": {"learning_rate": 0.001}},
        "loss": {
            "type": "composite",
            "label": "total",
            "weight": 1.0,
            "children": {
                "tracking": {
                    "type": "target",
                    "label": "tracking",
                    "weight": 1.0,
                }
            },
        },
        "n_batches": 2,
        "batch_size": 4,
    }


def _schema_workspace():
    graph = GraphSpec(
        nodes={
            "network": {
                "type": "Gain",
                "params": {"gain": 1.0},
                "input_ports": ["input"],
                "output_ports": ["output"],
            }
        },
        output_ports=["model_output"],
        output_bindings={"model_output": ("network", "output")},
        taps=[
            {
                "id": "activation-tap",
                "type": "probe",
                "position": {"afterNode": "network"},
                "paths": {"hidden": "state.network.hidden"},
            }
        ],
        metadata=GraphMetadata(
            name="Schema provider smoke",
            created_at="2026-05-20T00:00:00+00:00",
            updated_at="2026-05-20T00:00:00+00:00",
        ),
    )
    workspace = build_default_studio_workspace(label="Schema provider", graph=graph)
    train_stage = next(stage for stage in workspace.stages if stage.kind == "train")
    scenario = workspace.scenarios[train_stage.scenario_id]
    scenario.task_binding_spec = StudioTaskBindingSpec.model_validate(
        {
            "schema_version": "feedbax.studio.task_bindings.v2",
            "exposed_data": [
                {
                    "id": "inputs",
                    "label": "Inputs",
                    "kind": "signal",
                    "path": "inputs",
                    "bindable": True,
                    "dtype": "vector",
                    "metadata": {},
                },
                {
                    "id": "targets",
                    "label": "Targets",
                    "kind": "target",
                    "path": "targets",
                    "bindable": False,
                    "dtype": "state",
                    "metadata": {},
                },
            ],
            "bindings": [
                {
                    "id": "task:inputs->network:input",
                    "source_data_id": "inputs",
                    "target_node_id": "network",
                    "target_port": "input",
                    "role": "model_input",
                    "metadata": {},
                }
            ],
            "metadata": {},
        }
    )
    scenario.objective_spec = {
        "terms": [{"selector": "task_data:targets", "label": "Target tracking"}]
    }
    scenario.probe_specs = [{"id": "manual-probe", "label": "Manual probe"}]
    workspace.stages.append(
        StudioStageSpec(
            id="stage:broken",
            kind="eval",
            label="Broken",
            scenario_id="scenario:missing",
        )
    )
    return workspace


def test_provider_manifest_exposes_phase_one_capabilities() -> None:
    manifest = provider_manifest()

    assert manifest.provider == "feedbax"
    assert manifest.capabilities["validate_graph_spec"].input_schema == "GraphSpec"
    assert manifest.capabilities["start_training_run"].output_schema == "TrainingRunManifest"
    assert manifest.capabilities["enumerate_studio_schemas"].output_schema == "StudioSchemaRegistry"
    assert "training_checkpoint" in manifest.artifact_roles
    assert "TrainingRunManifest" in manifest.schemas
    assert "StudioSchemaRegistry" in manifest.schemas
    assert "TaskDataSchema" in manifest.schemas
    assert "RuntimeIntrospectionOptions" in manifest.schemas
    assert "RuntimeSampleLeafSchema" in manifest.schemas


def test_component_registry_snapshot_wraps_existing_registry() -> None:
    snapshot = component_registry_snapshot()
    type_ids = {entry.type_id for entry in snapshot.entries}

    assert snapshot.kind == "components"
    assert "feedbax.component.Gain" in type_ids
    gain = next(entry for entry in snapshot.entries if entry.type_id == "feedbax.component.Gain")
    assert gain.input_ports == ["input"]
    assert gain.output_ports == ["output"]


def test_validation_functions_accept_small_vertical_slice_specs() -> None:
    graph = _minimal_graph_spec()
    training = _minimal_training_spec()

    assert validate_graph_spec(graph).valid
    assert validate_training_spec(training, graph_spec=graph).valid
    assert validate_task_spec({"type": "SimpleReaches", "params": {}}).valid
    assert validate_evaluation_spec(
        {"evaluation_type": "default", "training_run_ids": ["feedbax-training-run:test"]}
    ).valid
    assert validate_analysis_spec(
        {
            "analysis_type": "feedbax.analysis.plot",
            "inputs": [{"kind": "TrainingRunManifest", "id": "feedbax-training-run:test"}],
        }
    ).valid


def test_task_validation_rejects_dense_delayed_reach_trajectory_params() -> None:
    result = validate_task_spec(
        {
            "type": "DelayedReaches",
            "params": {
                "n_steps": 140,
                "targets": [[[0.0, 0.0], [0.1, 0.1]]],
                "epoch_len_ranges": [[0, 1], [10, 30]],
                "target_on_epochs": [1, 2],
                "hold_epochs": [0, 1],
                "move_epochs": [2],
            },
        }
    )

    assert not result.valid
    assert result.errors[0].type == "dense_task_trajectory_not_allowed"
    assert result.errors[0].location == {"path": "/params/targets"}


def test_task_validation_rejects_invalid_delayed_reach_epoch_params() -> None:
    result = validate_task_spec(
        {
            "type": "DelayedReaches",
            "params": {
                "epoch_len_ranges": [[10, 1]],
                "target_on_epochs": [2],
                "hold_epochs": [0],
                "move_epochs": [1],
            },
        }
    )

    assert not result.valid
    assert {error.type for error in result.errors} == {
        "invalid_epoch_len_range",
        "invalid_epoch_index",
    }


def test_task_validation_reports_pathful_step_count_errors() -> None:
    invalid = validate_task_spec({"type": "DelayedReaches", "params": {"n_steps": 0}})
    mismatch = validate_task_spec(
        {
            "type": "DelayedReaches",
            "timeline": {"n_steps": 140},
            "params": {"n_steps": 120},
        }
    )

    assert invalid.errors[0].type == "invalid_task_n_steps"
    assert invalid.errors[0].location == {"path": "/params/n_steps"}
    assert {error.type for error in mismatch.errors} == {"task_n_steps_mismatch"}


def test_graph_validation_reports_unknown_components() -> None:
    graph = _minimal_graph_spec()
    graph["nodes"]["bad"] = {
        "type": "MissingComponent",
        "params": {},
        "input_ports": [],
        "output_ports": [],
    }

    result = validate_graph_spec(graph)

    assert not result.valid
    assert result.errors[0].type == "unknown_component_type"


def test_graph_validation_normalizes_runtime_network_authoring_payloads() -> None:
    result = validate_graph_spec(_runtime_network_graph_spec())

    assert result.valid
    assert {error.type for error in result.errors} == set()


def test_graph_validation_rejects_task_nodes() -> None:
    graph = _minimal_graph_spec()
    graph["nodes"]["task"] = {
        "type": "SimpleReaches",
        "params": {},
        "input_ports": [],
        "output_ports": ["inputs", "targets", "inits", "intervene"],
    }

    result = validate_graph_spec(graph)

    assert not result.valid
    assert result.errors[0].type == "task_node_not_allowed"


def test_graph_validation_uses_schema_for_direction_occupied_and_dtype_mismatch() -> None:
    graph = {
        "nodes": {
            "linear": {
                "type": "Linear",
                "params": {},
                "input_ports": ["input"],
                "output_ports": ["output"],
            },
            "muscle": {
                "type": "ReluMuscle",
                "params": {},
                "input_ports": ["excitation"],
                "output_ports": ["force", "activation"],
            },
        },
        "wires": [
            {
                "source_node": "linear",
                "source_port": "output",
                "target_node": "muscle",
                "target_port": "excitation",
            },
            {
                "source_node": "linear",
                "source_port": "input",
                "target_node": "muscle",
                "target_port": "excitation",
            },
        ],
        "input_ports": [],
        "output_ports": [],
        "input_bindings": {},
        "output_bindings": {},
    }

    result = validate_graph_spec(graph)
    issue_types = {issue.type for issue in result.errors}

    assert not result.valid
    assert "graph_wire_dtype_mismatch" in issue_types
    assert "wrong_source_port_direction" in issue_types
    assert "graph_input_occupied" in issue_types


def test_studio_task_timeline_spec_validates_value_specs() -> None:
    timeline = StudioTaskTimelineSpec.model_validate(
        {
            "schema_version": "feedbax.studio.task_timeline.v1",
            "epochs": [
                {
                    "id": "epoch:0",
                    "label": "hold",
                    "index": 0,
                    "length": {
                        "schema_version": "feedbax.studio.value.v1",
                        "mode": "constant",
                        "value": {"min": 0, "max": 1},
                        "metadata": {"scope": "trial"},
                    },
                    "metadata": {},
                }
            ],
            "signals": [
                {
                    "id": "hold",
                    "label": "Hold cue",
                    "kind": "signal",
                    "path": "inputs.hold",
                    "epoch_ids": ["epoch:0"],
                    "metadata": {},
                }
            ],
            "metadata": {"task_type": "DelayedReaches"},
        }
    )

    assert timeline.epochs[0].length.mode == "constant"
    assert timeline.signals[0].epoch_ids == ["epoch:0"]


def test_training_manifest_writes_artifacts_and_rebuildable_index(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.eqx"
    checkpoint.write_bytes(b"checkpoint bytes")

    root = tmp_path / "runs"
    manifest, path = write_training_run_manifest(
        job_id="job-1",
        total_batches=2,
        training_spec=_minimal_training_spec(),
        task_spec={"type": "SimpleReaches", "params": {}},
        task_binding_spec={
            "schema_version": "feedbax.studio.task_bindings.v2",
            "exposed_data": [],
            "bindings": [],
            "metadata": {},
        },
        graph_spec=_minimal_graph_spec(),
        checkpoint_path=checkpoint,
        history_events=[{"type": "training_progress", "batch": 1, "loss": 0.5}],
        final_loss=0.25,
        root=root,
        provenance=Provenance(source_commit="abc123", dirty=False),
        issues=["5429a23"],
    )

    assert path.exists()
    loaded = load_manifest(path)
    assert loaded.id == manifest.id
    assert loaded.status == "completed"
    assert loaded.summary_metrics["final_loss"] == 0.25
    assert loaded.task_binding_spec is not None
    assert loaded.provenance.issues == ["5429a23"]

    checkpoint_ref = next(
        artifact for artifact in loaded.artifacts if artifact.role == "training_checkpoint"
    )
    assert checkpoint_ref.sha256 == sha256_file(checkpoint)
    assert checkpoint_ref.uri is not None
    assert Path(checkpoint_ref.uri).exists()

    history_ref = next(
        artifact for artifact in loaded.artifacts if artifact.role == "training_history"
    )
    assert history_ref.media_type == "application/json"
    assert history_ref.uri is not None
    assert Path(history_ref.uri).exists()

    db_path = rebuild_manifest_index(root)
    with sqlite3.connect(db_path) as conn:
        manifest_count = conn.execute("SELECT COUNT(*) FROM manifests").fetchone()[0]
        artifact_count = conn.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0]

    assert manifest_count == 1
    assert artifact_count == 2


def test_provider_http_endpoints() -> None:
    client = TestClient(create_app())

    health = client.get("/api/provider/health")
    assert health.status_code == 200
    assert health.json()["provider"] == "feedbax"

    validation = client.post(
        "/api/provider/validate/graph",
        json={"spec": _minimal_graph_spec()},
    )
    assert validation.status_code == 200
    assert validation.json()["valid"] is True

    schemas = client.post(
        "/api/provider/studio/schemas",
        json={
            "workspace": _schema_workspace().model_dump(mode="json", exclude_none=True),
            "scenario_id": "scenario:train",
        },
    )
    assert schemas.status_code == 200
    payload = schemas.json()
    assert payload["kind"] == "studio_schema_registry"
    assert payload["scenario_id"] == "scenario:train"
    assert any(port["id"] == "port:network.input:input" for port in payload["ports"])
    assert any(item["id"] == "task_data:inputs" for item in payload["task_data"])
    assert any(
        target["selector"] == "path:states.mechanics.effector.pos"
        for target in payload["selector_targets"]
    )
    assert any(issue["type"] == "stage_missing_scenario" for issue in payload["issues"])

    runtime_schemas = client.post(
        "/api/provider/studio/schemas",
        json={
            "workspace": _schema_workspace().model_dump(mode="json", exclude_none=True),
            "scenario_id": "scenario:train",
            "runtime_introspection": {"enabled": True, "max_targets": 4},
        },
    )
    assert runtime_schemas.status_code == 200
    runtime_payload = runtime_schemas.json()
    assert runtime_payload["metadata"]["runtime_introspection"]["status"] == "unavailable"
    assert any(
        issue["type"] == "runtime_introspection_unavailable" and issue["severity"] == "warning"
        for issue in runtime_payload["issues"]
    )


def test_studio_schema_enumeration_returns_ports_task_data_targets_and_issues() -> None:
    registry = enumerate_studio_schema_registry(_schema_workspace(), "scenario:train")

    assert registry.workspace_id is not None
    assert registry.scenario_id == "scenario:train"
    assert any(port.id == "port:network.input:input" for port in registry.ports)
    bound = next(port for port in registry.ports if port.id == "port:network.input:input")
    assert bound.bound_task_data_id == "task_data:inputs"
    assert any(item.id == "task_data:targets" for item in registry.task_data)
    selectors = {target.selector for target in registry.selector_targets}
    assert "port:network.output" in selectors
    assert "task_data:inputs" in selectors
    assert "probe:activation-tap" in selectors
    assert "probe:manual-probe" in selectors
    assert "path:states.mechanics.effector.pos" in selectors
    assert any(issue.type == "stage_missing_scenario" for issue in registry.issues)
    assert registry.metadata["runtime_introspection"]["status"] == "not_requested"


def test_studio_schema_enumeration_normalizes_runtime_network_ports() -> None:
    workspace = build_default_studio_workspace(
        label="Runtime network",
        graph=GraphSpec.model_validate(_runtime_network_graph_spec()),
    )
    train_stage = next(stage for stage in workspace.stages if stage.kind == "train")
    scenario = workspace.scenarios[train_stage.scenario_id]
    scenario.task_binding_spec = StudioTaskBindingSpec.model_validate(
        {
            "schema_version": "feedbax.studio.task_bindings.v2",
            "exposed_data": [
                {
                    "id": "inputs",
                    "label": "Inputs",
                    "kind": "signal",
                    "path": "inputs",
                    "bindable": True,
                    "metadata": {},
                }
            ],
            "bindings": [
                {
                    "id": "task:inputs->network:target",
                    "source_data_id": "inputs",
                    "target_node_id": "network",
                    "target_port": "target",
                    "role": "model_input",
                    "metadata": {},
                }
            ],
            "metadata": {},
        }
    )

    registry = enumerate_studio_schema_registry(workspace, train_stage.scenario_id)

    network_input = next(port for port in registry.ports if port.id == "port:network.input:input")
    assert network_input.component_type == "Network"
    assert network_input.bound_task_data_id == "task_data:inputs"
    assert not any(issue.type == "unknown_task_binding_target_port" for issue in registry.issues)


def test_network_authoring_normalization_emits_recurrent_cell_edges() -> None:
    graph = GraphSpec.model_validate(_runtime_network_graph_spec())
    normalized = normalize_graph_for_studio_authoring(graph)
    subgraph = normalized.subgraphs["network"]

    hidden_wire = next(
        wire
        for wire in subgraph.wires
        if wire.source_node == "cell"
        and wire.source_port == "hidden"
        and wire.target_node == "cell"
        and wire.target_port == "hidden"
    )

    assert hidden_wire.temporality == "recurrent"
    assert hidden_wire.recurrent_initializer == {
        "kind": "zeros",
        "scope": "trial",
        "shape": [100],
        "source": "state_initializer",
        "state_slot": "hidden",
    }
    assert subgraph.output_bindings["hidden"] == ("cell", "hidden")


def test_network_authoring_normalization_flattens_legacy_model_wrapper() -> None:
    graph = GraphSpec.model_validate(_runtime_network_graph_spec())
    legacy_inner = (
        normalize_graph_for_studio_authoring(graph)
        .subgraphs["network"]
        .model_copy(
            update={
                "wires": [
                    wire.model_copy(
                        update={"temporality": "instant", "recurrent_initializer": None}
                    )
                    for wire in normalize_graph_for_studio_authoring(graph)
                    .subgraphs["network"]
                    .wires
                ],
                "output_ports": ["output"],
                "output_bindings": {"output": ("readout", "output")},
            }
        )
    )
    wrapped = graph.model_copy(
        update={
            "nodes": {
                **graph.nodes,
                "network": graph.nodes["network"].model_copy(update={"type": "Network"}),
            },
            "subgraphs": {
                "network": GraphSpec(
                    nodes={
                        "model": {
                            "type": "Subgraph",
                            "params": {},
                            "input_ports": ["input", "feedback"],
                            "output_ports": ["output"],
                        }
                    },
                    wires=[],
                    input_ports=["input", "feedback"],
                    output_ports=["output"],
                    input_bindings={
                        "input": ("model", "input"),
                        "feedback": ("model", "feedback"),
                    },
                    output_bindings={"output": ("model", "output")},
                    subgraphs={"model": legacy_inner},
                )
            },
        }
    )

    subgraph = normalize_graph_for_studio_authoring(wrapped).subgraphs["network"]

    assert "model" not in subgraph.nodes
    assert subgraph.nodes["cell"].type == "GRU"
    assert subgraph.output_bindings["hidden"] == ("cell", "hidden")
    hidden_wire = next(
        wire
        for wire in subgraph.wires
        if wire.source_node == "cell"
        and wire.source_port == "hidden"
        and wire.target_node == "cell"
        and wire.target_port == "hidden"
    )
    assert hidden_wire.temporality == "recurrent"
    assert hidden_wire.recurrent_initializer is not None


def test_network_authoring_normalization_marks_feedback_cut_recurrent() -> None:
    graph = GraphSpec(
        nodes={
            "network": {
                "type": "Network",
                "params": {},
                "input_ports": ["input", "feedback"],
                "output_ports": ["output"],
            },
            "mechanics": {
                "type": "PointMass",
                "params": {},
                "input_ports": ["force"],
                "output_ports": ["effector"],
            },
            "feedback": {
                "type": "FeedbackChannels",
                "params": {},
                "input_ports": ["input"],
                "output_ports": ["output"],
            },
        },
        wires=[
            {
                "source_node": "network",
                "source_port": "output",
                "target_node": "mechanics",
                "target_port": "force",
            },
            {
                "source_node": "mechanics",
                "source_port": "effector",
                "target_node": "feedback",
                "target_port": "input",
            },
            {
                "source_node": "feedback",
                "source_port": "output",
                "target_node": "network",
                "target_port": "feedback",
            },
        ],
    )

    recurrent_wire = normalize_graph_for_studio_authoring(graph).wires[2]

    assert recurrent_wire.temporality == "recurrent"
    assert recurrent_wire.recurrent_initializer == {
        "kind": "zeros",
        "scope": "trial",
        "source": "state_initializer",
        "state_slot": "feedback",
    }


def test_graph_connection_schema_rejects_instant_cycles_and_accepts_recurrent_cut() -> None:
    graph = GraphSpec(
        nodes={
            "a": {
                "type": "Gain",
                "params": {},
                "input_ports": ["input"],
                "output_ports": ["output"],
            },
            "b": {
                "type": "Gain",
                "params": {},
                "input_ports": ["input"],
                "output_ports": ["output"],
            },
        },
        wires=[
            {
                "source_node": "a",
                "source_port": "output",
                "target_node": "b",
                "target_port": "input",
            },
            {
                "source_node": "b",
                "source_port": "output",
                "target_node": "a",
                "target_port": "input",
            },
        ],
    )

    issues = validate_graph_connection_schema(graph)
    assert "instant_cycle" in {issue.type for issue in issues}

    recurrent_graph = graph.model_copy(
        update={
            "wires": [
                graph.wires[0],
                graph.wires[1].model_copy(
                    update={
                        "temporality": "recurrent",
                        "recurrent_initializer": {
                            "kind": "zeros",
                            "scope": "trial",
                            "shape": [1],
                        },
                    }
                ),
            ]
        }
    )
    recurrent_issues = validate_graph_connection_schema(recurrent_graph)
    assert "instant_cycle" not in {issue.type for issue in recurrent_issues}
    assert "recurrent_initializer_missing" not in {issue.type for issue in recurrent_issues}


def test_studio_schema_enumeration_projects_dynamic_mux_inputs() -> None:
    graph = GraphSpec(
        nodes={
            "mux": {
                "type": "Mux",
                "params": {"n_inputs": 2},
                "input_ports": ["in_0", "in_1"],
                "output_ports": ["output"],
            }
        },
        output_ports=["output"],
        output_bindings={"output": ("mux", "output")},
    )
    workspace = build_default_studio_workspace(label="Mux schema", graph=graph)
    train_stage = next(stage for stage in workspace.stages if stage.kind == "train")
    scenario = workspace.scenarios[train_stage.scenario_id]
    scenario.task_binding_spec = StudioTaskBindingSpec.model_validate(
        {
            "schema_version": "feedbax.studio.task_bindings.v2",
            "exposed_data": [
                {
                    "id": "target_on",
                    "label": "Target shown",
                    "kind": "signal",
                    "role": "model_input",
                    "path": "inputs.target_on",
                    "bindable": True,
                    "dtype": "float32",
                    "expected_shape": ["time", 1],
                    "metadata": {},
                }
            ],
            "bindings": [
                {
                    "id": "task:target_on->mux:in_2",
                    "source_data_id": "target_on",
                    "target_node_id": "mux",
                    "target_port": "in_2",
                    "role": "model_input",
                    "metadata": {},
                }
            ],
            "metadata": {},
        }
    )

    registry = enumerate_studio_schema_registry(workspace, train_stage.scenario_id)
    port = next(port for port in registry.ports if port.id == "port:mux.in_2:input")

    assert port.value_schema.dtype == "vector"
    assert port.origin == "declared"
    assert not any(issue.type == "unknown_task_binding_target_port" for issue in registry.issues)


def test_studio_schema_task_data_trajectory_bindings_use_sample_view() -> None:
    graph = GraphSpec(
        nodes={
            "network": {
                "type": "Network",
                "params": {"input_size": 2, "hidden_size": 100, "out_size": 6},
                "input_ports": ["input"],
                "output_ports": ["output"],
            }
        }
    )
    workspace = build_default_studio_workspace(label="Sample view schema", graph=graph)
    train_stage = next(stage for stage in workspace.stages if stage.kind == "train")
    scenario = workspace.scenarios[train_stage.scenario_id]
    scenario.task_binding_spec = StudioTaskBindingSpec.model_validate(
        {
            "schema_version": "feedbax.studio.task_bindings.v2",
            "exposed_data": [
                {
                    "id": "target_position",
                    "label": "Target position",
                    "kind": "signal",
                    "role": "model_input",
                    "path": "inputs.effector_target.pos",
                    "bindable": True,
                    "dtype": "float32",
                    "expected_shape": ["time", 2],
                    "metadata": {},
                }
            ],
            "bindings": [
                {
                    "id": "task:target_position->network:input",
                    "source_data_id": "target_position",
                    "target_node_id": "network",
                    "target_port": "input",
                    "role": "model_input",
                    "metadata": {},
                }
            ],
            "metadata": {},
        }
    )

    registry = enumerate_studio_schema_registry(workspace, train_stage.scenario_id)
    task_data = next(item for item in registry.task_data if item.id == "task_data:target_position")
    network_input = next(port for port in registry.ports if port.id == "port:network.input:input")

    assert task_data.value_schema.shape == ["time", 2]
    assert task_data.value_schema.metadata["sample_shape"] == [2]
    assert task_data.value_schema.metadata["time_axis"] == 0
    assert network_input.value_schema.shape is None
    issue_types = {issue.type for issue in registry.issues}
    assert "task_binding_rank_mismatch" not in issue_types
    assert "task_binding_shape_mismatch" not in issue_types
    assert "task_binding_dtype_mismatch" not in issue_types


def test_studio_schema_enumeration_infers_mux_output_width_from_sample_shapes() -> None:
    graph = GraphSpec(
        nodes={
            "mux": {
                "type": "Mux",
                "params": {"n_inputs": 2},
                "input_ports": ["in_0", "in_1"],
                "output_ports": ["output"],
            }
        },
        output_ports=["output"],
        output_bindings={"output": ("mux", "output")},
    )
    workspace = build_default_studio_workspace(label="Mux width schema", graph=graph)
    train_stage = next(stage for stage in workspace.stages if stage.kind == "train")
    scenario = workspace.scenarios[train_stage.scenario_id]
    scenario.task_binding_spec = StudioTaskBindingSpec.model_validate(
        {
            "schema_version": "feedbax.studio.task_bindings.v2",
            "exposed_data": [
                {
                    "id": "target_position",
                    "label": "Target position",
                    "kind": "signal",
                    "role": "model_input",
                    "path": "inputs.effector_target.pos",
                    "bindable": True,
                    "dtype": "float32",
                    "expected_shape": ["time", 2],
                    "metadata": {},
                },
                {
                    "id": "hold",
                    "label": "Hold/go cue",
                    "kind": "signal",
                    "role": "model_input",
                    "path": "inputs.hold",
                    "bindable": True,
                    "dtype": "float32",
                    "expected_shape": ["time", 1],
                    "metadata": {},
                },
            ],
            "bindings": [
                {
                    "id": "task:target_position->mux:in_0",
                    "source_data_id": "target_position",
                    "target_node_id": "mux",
                    "target_port": "in_0",
                    "role": "model_input",
                    "metadata": {},
                },
                {
                    "id": "task:hold->mux:in_1",
                    "source_data_id": "hold",
                    "target_node_id": "mux",
                    "target_port": "in_1",
                    "role": "model_input",
                    "metadata": {},
                },
            ],
            "metadata": {},
        }
    )

    registry = enumerate_studio_schema_registry(workspace, train_stage.scenario_id)
    mux_output = next(port for port in registry.ports if port.id == "port:mux.output:output")

    assert mux_output.value_schema.shape == [3]
    assert mux_output.value_schema.rank == 1
    assert mux_output.value_schema.metadata["dimension_source"] == "mux_concat_inputs"


def test_studio_schema_uses_subgraph_boundary_shapes_for_parent_ports() -> None:
    child_graph = GraphSpec(
        nodes={
            "cell": {
                "type": "GRU",
                "params": {"input_size": 4, "hidden_size": 100},
                "input_ports": ["input", "hidden"],
                "output_ports": ["output", "hidden"],
            },
            "readout": {
                "type": "Linear",
                "params": {"input_size": 100, "output_size": 2},
                "input_ports": ["input"],
                "output_ports": ["output"],
            },
        },
        wires=[
            {
                "source_node": "cell",
                "source_port": "output",
                "target_node": "readout",
                "target_port": "input",
            }
        ],
        input_ports=["input", "hidden"],
        output_ports=["output", "hidden"],
        input_bindings={"input": ("cell", "input"), "hidden": ("cell", "hidden")},
        output_bindings={"output": ("readout", "output"), "hidden": ("cell", "output")},
    )
    graph = GraphSpec(
        nodes={
            "task_mux": {
                "type": "Mux",
                "params": {"n_inputs": 3},
                "input_ports": ["in_0", "in_1", "in_2"],
                "output_ports": ["output"],
            },
            "network": {
                "type": "Network",
                "params": {"input_size": 4, "hidden_size": 100, "out_size": 2},
                "input_ports": ["input", "hidden"],
                "output_ports": ["output", "hidden"],
            },
        },
        subgraphs={"network": child_graph},
    )
    workspace = build_default_studio_workspace(label="Subgraph boundary schema", graph=graph)
    train_stage = next(stage for stage in workspace.stages if stage.kind == "train")
    scenario = workspace.scenarios[train_stage.scenario_id]
    scenario.task_binding_spec = StudioTaskBindingSpec.model_validate(
        {
            "schema_version": "feedbax.studio.task_bindings.v2",
            "exposed_data": [
                {
                    "id": "target_position",
                    "label": "Target position",
                    "kind": "signal",
                    "role": "model_input",
                    "path": "inputs.effector_target",
                    "bindable": True,
                    "dtype": "float32",
                    "expected_shape": ["time", 4],
                    "metadata": {},
                },
                {
                    "id": "hold",
                    "label": "Hold/go cue",
                    "kind": "signal",
                    "role": "model_input",
                    "path": "inputs.hold",
                    "bindable": True,
                    "dtype": "float32",
                    "expected_shape": ["time", 1],
                    "metadata": {},
                },
                {
                    "id": "target_on",
                    "label": "Target shown",
                    "kind": "signal",
                    "role": "model_input",
                    "path": "inputs.target_on",
                    "bindable": True,
                    "dtype": "float32",
                    "expected_shape": ["time", 1],
                    "metadata": {},
                },
            ],
            "bindings": [
                {
                    "id": "task:target_position->task_mux:in_0",
                    "source_data_id": "target_position",
                    "target_node_id": "task_mux",
                    "target_port": "in_0",
                    "role": "model_input",
                    "metadata": {},
                },
                {
                    "id": "task:hold->task_mux:in_1",
                    "source_data_id": "hold",
                    "target_node_id": "task_mux",
                    "target_port": "in_1",
                    "role": "model_input",
                    "metadata": {},
                },
                {
                    "id": "task:target_on->task_mux:in_2",
                    "source_data_id": "target_on",
                    "target_node_id": "task_mux",
                    "target_port": "in_2",
                    "role": "model_input",
                    "metadata": {},
                },
            ],
            "metadata": {},
        }
    )

    registry = enumerate_studio_schema_registry(workspace, train_stage.scenario_id)
    mux_output = next(port for port in registry.ports if port.id == "port:task_mux.output:output")
    feedback_input = next(
        port for port in registry.ports if port.id == "port:network.feedback:input"
    )
    hidden_inputs = [port for port in registry.ports if port.id == "port:network.hidden:input"]

    assert mux_output.value_schema.shape == [6]
    assert feedback_input.value_schema.dtype == "vector"
    assert hidden_inputs == []


def test_studio_schema_enumeration_runtime_introspection_hook_adds_sample_leaf_targets() -> None:
    def introspector(workspace, scenario_id, options):
        assert workspace.id
        assert scenario_id == "scenario:train"
        assert options.max_targets == 1
        return RuntimeIntrospectionResult(
            sample_leaves=[
                RuntimeSampleLeafSchema(
                    path="states.network.hidden",
                    label="Network hidden state",
                    value=[[0.0, 1.0], [2.0, 3.0]],
                    source={"sample": "representative"},
                ),
                RuntimeSampleLeafSchema(
                    path="task.validation_trials.targets",
                    label="Validation targets",
                    dtype="float32",
                    shape=[8, 2],
                ),
            ],
            metadata={"provider_hook": "test"},
        )

    registry = enumerate_studio_schema_registry(
        _schema_workspace(),
        "scenario:train",
        runtime_introspection={"enabled": True, "max_targets": 1},
        runtime_introspector=introspector,
    )

    runtime_targets = [
        target for target in registry.selector_targets if target.origin == "runtime_sample"
    ]
    assert len(runtime_targets) == 1
    target = runtime_targets[0]
    assert target.kind == "sample_leaf"
    assert target.selector == "path:states.network.hidden"
    assert target.value_schema.shape == [2, 2]
    assert target.value_schema.rank == 2
    assert target.value_schema.origin == "runtime_sample"
    assert registry.metadata["runtime_introspection"]["status"] == "completed"
    assert registry.metadata["runtime_introspection"]["target_count"] == 1
    assert registry.metadata["runtime_introspection"]["truncated"] is True


def test_studio_schema_enumeration_runtime_introspection_failure_is_warning() -> None:
    def introspector(_workspace, _scenario_id, _options):
        raise RuntimeError("sample unavailable")

    registry = enumerate_studio_schema_registry(
        _schema_workspace(),
        "scenario:train",
        runtime_introspection=True,
        runtime_introspector=introspector,
    )

    issue = next(issue for issue in registry.issues if issue.type == "runtime_introspection_failed")
    assert issue.severity == "warning"
    assert "sample unavailable" in issue.message
    assert registry.metadata["runtime_introspection"]["status"] == "failed"


def test_studio_schema_enumeration_reports_missing_scenario_graph_and_binding() -> None:
    missing = enumerate_studio_schema_registry(_schema_workspace(), "scenario:missing")
    assert any(issue.type == "missing_scenario" for issue in missing.issues)

    workspace = _schema_workspace()
    scenario = workspace.scenarios["scenario:train"]
    scenario.graph = None
    scenario.task_binding_spec = None
    registry = enumerate_studio_schema_registry(workspace, "scenario:train")

    issue_types = {issue.type for issue in registry.issues}
    assert "missing_graph" in issue_types
    assert "missing_task_binding_spec" in issue_types


def test_studio_schema_enumeration_validates_task_binding_schema_mismatch() -> None:
    workspace = _schema_workspace()
    scenario = next(iter(workspace.scenarios.values()))
    scenario.graph.nodes["network"].type = "Linear"
    assert scenario.task_binding_spec is not None
    scenario.task_binding_spec.exposed_data[0].dtype = "scalar"

    registry = enumerate_studio_schema_registry(workspace, scenario.id)
    issue_types = {issue.type for issue in registry.issues}

    assert "task_binding_dtype_mismatch" in issue_types


def test_studio_schema_enumeration_validates_task_binding_identity() -> None:
    workspace = _schema_workspace()
    scenario = next(iter(workspace.scenarios.values()))
    assert scenario.task_binding_spec is not None
    binding = scenario.task_binding_spec.bindings[0]
    scenario.task_binding_spec.bindings = [
        binding.model_copy(update={"id": "not-canonical"}),
        binding,
        binding.model_copy(),
    ]

    registry = enumerate_studio_schema_registry(workspace, scenario.id)
    issue_types = {issue.type for issue in registry.issues}

    assert "task_binding_id_mismatch" in issue_types
    assert "duplicate_task_binding" in issue_types


def test_studio_schema_enumerates_task_data_roles_and_rejects_protocol_bindings() -> None:
    workspace = _schema_workspace()
    scenario = workspace.scenarios["scenario:train"]
    assert scenario.task_binding_spec is not None
    scenario.task_binding_spec.exposed_data[1].bindable = True
    scenario.task_binding_spec.exposed_data[1].role = "target"
    scenario.task_binding_spec.bindings[0].source_data_id = "targets"

    registry = enumerate_studio_schema_registry(workspace, scenario.id)
    task_data = {item.path: item for item in registry.task_data}
    issue_types = {issue.type for issue in registry.issues}

    assert task_data["inputs"].role == "model_input"
    assert task_data["inputs"].bindable is True
    assert task_data["targets"].role == "target"
    assert task_data["targets"].bindable is False
    assert task_data["targets"].metadata["task_data_surface"] == "protocol"
    assert "task_data_bindable_role_mismatch" in issue_types
    assert "task_data_protocol_path_bindable" in issue_types
    assert "task_data_not_bindable" in issue_types


def test_studio_schema_enumeration_validates_intervention_targets() -> None:
    workspace = _schema_workspace()
    scenario = workspace.scenarios["scenario:train"]
    assert scenario.graph is not None
    scenario.graph.taps = [
        TapSpec.model_validate(item)
        for item in [
            {
                "id": "valid-clamp",
                "type": "intervention",
                "position": {"afterNode": "network"},
                "paths": {},
                "transform": {
                    "type": "intervention",
                    "params": {},
                    "intervention": {
                        "operation": "clamp",
                        "target_selector": {
                            "namespace": "state_path",
                            "compact": "path:states.net.output",
                            "path": "states.net.output",
                            "metadata": {},
                        },
                        "bounds": {"min": -1.0, "max": 1.0},
                        "metadata": {},
                    },
                },
            },
            {
                "id": "bad-constant",
                "type": "intervention",
                "position": {"afterNode": "network"},
                "paths": {},
                "transform": {
                    "type": "intervention",
                    "params": {},
                    "intervention": {
                        "operation": "constant",
                        "target_selector": {
                            "namespace": "task_data",
                            "compact": "task_data:targets",
                            "path": "targets",
                            "metadata": {},
                        },
                        "metadata": {},
                    },
                },
            },
        ]
    ]

    registry = enumerate_studio_schema_registry(workspace, "scenario:train")
    issue_types = {issue.type for issue in registry.issues}

    assert "intervention_missing_value" in issue_types
    assert "intervention_target_dtype_mismatch" in issue_types
    assert "intervention_missing_bounds" not in issue_types


def test_worker_stub_emits_durable_training_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("FEEDBAX_RUNS_DIR", str(tmp_path / "worker-runs"))
    event_queue: queue.Queue = queue.Queue()
    job = _Job(
        job_id="stub-job",
        total_batches=1,
        event_queue=event_queue,
        stop_event=threading.Event(),
        training_spec=_minimal_training_spec(),
        task_spec={"type": "SimpleReaches", "params": {}},
        task_binding_spec={
            "schema_version": "feedbax.studio.task_bindings.v2",
            "exposed_data": [],
            "bindings": [],
            "metadata": {},
        },
        graph_spec=_minimal_graph_spec(),
        status=WorkerStatus.RUNNING,
    )

    _run_training_stub(job)

    assert job.status == WorkerStatus.COMPLETED
    assert job.manifest_path is not None
    assert Path(job.manifest_path).exists()
    assert job.manifest_payload is not None
    assert job.manifest_payload["kind"] == "TrainingRunManifest"
    assert job.manifest_payload["artifacts"][0]["role"] == "training_history"

    events = []
    while not event_queue.empty():
        events.append(event_queue.get())
    complete = next(event for event in events if event["type"] == "training_complete")
    assert complete["manifest_id"] == job.manifest_payload["id"]
    assert complete["manifest_path"] == job.manifest_path


def _worker_contract_job(
    *,
    task_binding_spec: dict | None,
    graph_spec: dict | None = None,
) -> _Job:
    return _Job(
        job_id="worker-contract",
        total_batches=1,
        event_queue=queue.Queue(),
        stop_event=threading.Event(),
        training_spec=_minimal_training_spec(),
        task_spec={"type": "SimpleReaches", "params": {}},
        task_binding_spec=task_binding_spec,
        graph_spec=graph_spec or _minimal_graph_spec(),
        status=WorkerStatus.RUNNING,
    )


def test_worker_spec_contract_accepts_scenario_owned_task_binding_v2() -> None:
    _require_worker_specs(
        _worker_contract_job(
            task_binding_spec={
                "schema_version": "feedbax.studio.task_bindings.v2",
                "exposed_data": [
                    {
                        "id": "inputs",
                        "label": "Inputs",
                        "kind": "signal",
                        "path": "inputs",
                        "bindable": True,
                        "metadata": {},
                    }
                ],
                "bindings": [
                    {
                        "id": "task:inputs->gain:input",
                        "source_data_id": "inputs",
                        "target_node_id": "gain",
                        "target_port": "input",
                        "role": "model_input",
                        "metadata": {},
                    }
                ],
                "metadata": {},
            }
        )
    )


def test_worker_spec_contract_normalizes_runtime_network_payloads() -> None:
    job = _worker_contract_job(
        graph_spec=_runtime_network_graph_spec(),
        task_binding_spec={
            "schema_version": "feedbax.studio.task_bindings.v2",
            "exposed_data": [
                {
                    "id": "inputs",
                    "label": "Inputs",
                    "kind": "signal",
                    "path": "inputs",
                    "bindable": True,
                    "metadata": {},
                }
            ],
            "bindings": [
                {
                    "id": "task:inputs->network:target",
                    "source_data_id": "inputs",
                    "target_node_id": "network",
                    "target_port": "target",
                    "role": "model_input",
                    "metadata": {},
                }
            ],
            "metadata": {},
        },
    )

    _require_worker_specs(job)

    assert job.graph_spec["nodes"]["network"]["type"] == "Network"
    assert "network" in job.graph_spec["subgraphs"]
    assert job.task_binding_spec["bindings"][0]["target_port"] == "input"
    params = _extract_graph_params(job.graph_spec)
    assert params["hidden_size"] == 100
    assert params["input_size"] == 4
    assert params["out_size"] == 2


def test_worker_spec_contract_rejects_graph_incompatible_task_bindings() -> None:
    with pytest.raises(ValueError, match="unknown_task_binding_target_port"):
        _require_worker_specs(
            _worker_contract_job(
                task_binding_spec={
                    "schema_version": "feedbax.studio.task_bindings.v2",
                    "exposed_data": [
                        {
                            "id": "inputs",
                            "label": "Inputs",
                            "kind": "signal",
                            "path": "inputs",
                            "bindable": True,
                            "metadata": {},
                        }
                    ],
                    "bindings": [
                        {
                            "id": "task:inputs->gain:missing",
                            "source_data_id": "inputs",
                            "target_node_id": "gain",
                            "target_port": "missing",
                            "role": "model_input",
                            "metadata": {},
                        }
                    ],
                    "metadata": {},
                }
            )
        )


@pytest.mark.parametrize(
    ("task_binding_spec", "message"),
    [
        (
            None,
            "must not be inferred from graph task nodes",
        ),
        (
            {
                "schema_version": "feedbax.studio.task_bindings.v1",
                "exposed_outputs": [],
                "bindings": [],
                "metadata": {},
            },
            "schema v2",
        ),
        (
            {
                "schema_version": "feedbax.studio.task_bindings.v2",
                "exposed_outputs": [],
                "bindings": [],
                "metadata": {},
            },
            "exposed_outputs is not accepted",
        ),
        (
            {
                "schema_version": "feedbax.studio.task_bindings.v2",
                "exposed_data": [],
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
            "source_data_id",
        ),
    ],
)
def test_worker_spec_contract_rejects_legacy_or_inferred_task_bindings(
    task_binding_spec: dict | None,
    message: str,
) -> None:
    graph_with_task_node = _minimal_graph_spec()
    graph_with_task_node["nodes"]["task"] = {
        "type": "SimpleReaches",
        "params": {},
        "input_ports": [],
        "output_ports": ["inputs"],
    }
    with pytest.raises(ValueError, match=message):
        _require_worker_specs(
            _worker_contract_job(
                task_binding_spec=task_binding_spec,
                graph_spec=graph_with_task_node,
            )
        )


def test_worker_training_cfg_uses_task_n_steps() -> None:
    cfg = _extract_training_cfg(
        {"n_batches": 4, "n_reach_steps": 80},
        {"type": "DelayedReaches", "params": {"n_steps": 140}},
    )

    assert cfg.n_batches == 4
    assert cfg.n_reach_steps == 140


def test_worker_training_cfg_uses_timeline_task_n_steps() -> None:
    cfg = _extract_training_cfg(
        {"n_batches": 4, "n_reach_steps": 80},
        {"type": "DelayedReaches", "timeline": {"n_steps": 150}, "params": {"n_steps": 140}},
    )

    assert cfg.n_reach_steps == 150


def test_worker_task_sampling_cfg_uses_delayed_reaches_params() -> None:
    cfg = _extract_task_sampling_cfg(
        {
            "type": "DelayedReaches",
            "params": {
                "n_steps": 140,
                "workspace": [[-1.0, -0.5], [1.0, 0.5]],
                "train_endpoint_mode": "center_out",
                "epoch_len_ranges": [[0, 1], [10, 30]],
                "hold_epochs": [0, 1],
                "move_epochs": [2],
                "p_catch_trial": 0.5,
                "eval_reach_length": 0.4,
            },
        }
    )

    assert cfg.task_type == "DelayedReaches"
    assert cfg.workspace_min == (-1.0, -0.5)
    assert cfg.workspace_max == (1.0, 0.5)
    assert cfg.train_endpoint_mode == "center_out"
    assert cfg.epoch_len_ranges == ((0, 1), (10, 30))
    assert cfg.hold_epochs == (0, 1)
    assert cfg.move_epochs == (2,)
    assert cfg.p_catch_trial == 0.5
    assert cfg.reach_length == 0.4


def test_worker_rejects_dense_task_sampling_params() -> None:
    try:
        _extract_task_sampling_cfg(
            {
                "type": "DelayedReaches",
                "params": {
                    "n_steps": 140,
                    "targets": [[[0.0, 0.0], [0.1, 0.1]]],
                },
            }
        )
    except ValueError as exc:
        assert "compact task params only" in str(exc)
    else:
        raise AssertionError("Expected dense task params to be rejected")


def test_worker_graph_params_use_simple_staged_network_and_pointmass() -> None:
    params = _extract_graph_params(
        {
            "nodes": {
                "network": {
                    "type": "SimpleStagedNetwork",
                    "params": {
                        "hidden_size": 100,
                        "input_size": 4,
                        "out_size": 2,
                        "hidden_type": "GRUCell",
                        "out_nonlinearity": "tanh",
                    },
                    "input_ports": ["input"],
                    "output_ports": ["output"],
                },
                "mechanics": {
                    "type": "PointMass",
                    "params": {"dt": 0.02},
                    "input_ports": ["force"],
                    "output_ports": ["effector", "state"],
                },
            },
            "wires": [
                {
                    "source_node": "network",
                    "source_port": "output",
                    "target_node": "mechanics",
                    "target_port": "force",
                }
            ],
            "input_ports": [],
            "output_ports": [],
            "input_bindings": {},
            "output_bindings": {},
        }
    )

    assert params["hidden_size"] == 100
    assert params["input_size"] == 4
    assert params["out_size"] == 2
    assert params["dt"] == 0.02
    assert params["plant_type"] == "PointMass"
    assert params["action_size"] == 2
    assert params["mechanics_input_port"] == "force"
    assert params["action_path"] == ("network", "mechanics")


def test_worker_graph_lowering_uses_task_binding_network_and_passthrough_topology() -> None:
    graph = {
        "nodes": {
            "unused_network": {
                "type": "SimpleStagedNetwork",
                "params": {
                    "hidden_size": 8,
                    "input_size": 99,
                    "out_size": 2,
                    "hidden_type": "GRUCell",
                    "out_nonlinearity": "tanh",
                },
                "input_ports": ["input"],
                "output_ports": ["output"],
            },
            "network": {
                "type": "SimpleStagedNetwork",
                "params": {
                    "hidden_size": 16,
                    "input_size": 4,
                    "out_size": 2,
                    "hidden_type": "GRUCell",
                    "out_nonlinearity": "tanh",
                },
                "input_ports": ["input"],
                "output_ports": ["output", "hidden"],
            },
            "clamp": {
                "type": "NetworkClamp",
                "params": {},
                "input_ports": ["input"],
                "output_ports": ["output"],
            },
            "mechanics": {
                "type": "PointMass",
                "params": {"dt": 0.02},
                "input_ports": ["force"],
                "output_ports": ["effector", "state"],
            },
        },
        "wires": [
            {
                "source_node": "network",
                "source_port": "output",
                "target_node": "clamp",
                "target_port": "input",
            },
            {
                "source_node": "clamp",
                "source_port": "output",
                "target_node": "mechanics",
                "target_port": "force",
            },
        ],
        "input_ports": [],
        "output_ports": [],
        "input_bindings": {},
        "output_bindings": {},
    }
    task_binding = _task_input_binding(expected_shape=[4])

    lowering = _derive_worker_graph_lowering(graph, task_binding)
    params = _extract_graph_params(graph, task_binding)

    assert lowering.network_node_id == "network"
    assert lowering.model_input_data_id == "inputs"
    assert lowering.action_path == ("network", "clamp", "mechanics")
    assert params["hidden_size"] == 16
    assert params["input_size"] == 4
    assert params["observation_layout"] == "pointmass"


def test_worker_graph_lowering_derives_two_link_direct_action_size() -> None:
    graph = {
        "nodes": {
            "network": {
                "type": "SimpleStagedNetwork",
                "params": {
                    "hidden_size": 16,
                    "input_size": 13,
                    "out_size": 2,
                    "hidden_type": "GRUCell",
                    "out_nonlinearity": "tanh",
                },
                "input_ports": ["input"],
                "output_ports": ["output"],
            },
            "mechanics": {
                "type": "TwoLinkArm",
                "params": {"dt": 0.01},
                "input_ports": ["force"],
                "output_ports": ["effector", "state"],
            },
        },
        "wires": [
            {
                "source_node": "network",
                "source_port": "output",
                "target_node": "mechanics",
                "target_port": "force",
            }
        ],
        "input_ports": [],
        "output_ports": [],
        "input_bindings": {},
        "output_bindings": {},
    }

    params = _extract_graph_params(graph, _task_input_binding(expected_shape=[13]))

    assert params["plant_type"] == "TwoLinkArm"
    assert params["action_size"] == 2
    assert params["observation_layout"] == "two_link_direct"


def test_worker_graph_lowering_derives_arm6_excitation_action_size() -> None:
    graph = {
        "nodes": {
            "network": {
                "type": "SimpleStagedNetwork",
                "params": {
                    "hidden_size": 16,
                    "input_size": 17,
                    "out_size": 6,
                    "hidden_type": "GRUCell",
                    "out_nonlinearity": "sigmoid",
                },
                "input_ports": ["input"],
                "output_ports": ["output"],
            },
            "mechanics": {
                "type": "Arm6MuscleRigidTendon",
                "params": {"dt": 0.01},
                "input_ports": ["excitation"],
                "output_ports": ["torques", "forces", "activations"],
            },
        },
        "wires": [
            {
                "source_node": "network",
                "source_port": "output",
                "target_node": "mechanics",
                "target_port": "excitation",
            }
        ],
        "input_ports": [],
        "output_ports": [],
        "input_bindings": {},
        "output_bindings": {},
    }

    params = _extract_graph_params(graph, _task_input_binding(expected_shape=[17]))

    assert params["plant_type"] == "Arm6MuscleRigidTendon"
    assert params["action_size"] == 6
    assert params["observation_layout"] == "arm6_muscle"


def test_worker_graph_lowering_rejects_task_data_input_size_mismatch() -> None:
    graph = _runtime_network_graph_spec()
    with pytest.raises(ValueError, match="does not match task data"):
        _extract_graph_params(
            graph,
            _task_input_binding(target_port="target", expected_shape=[5]),
        )


def test_worker_graph_lowering_rejects_unwired_network_action() -> None:
    graph = _runtime_network_graph_spec()
    graph["wires"] = [
        {
            "source_node": "network",
            "source_port": "hidden",
            "target_node": "mechanics",
            "target_port": "force",
        }
    ]

    with pytest.raises(ValueError, match="supported mechanics action port"):
        _extract_graph_params(
            graph,
            _task_input_binding(target_port="target", expected_shape=[4]),
        )


def test_worker_training_errors_instead_of_stub_on_missing_task_binding() -> None:
    event_queue: queue.Queue = queue.Queue()
    job = _Job(
        job_id="invalid-job",
        total_batches=1,
        event_queue=event_queue,
        stop_event=threading.Event(),
        training_spec=_minimal_training_spec(),
        task_spec={"type": "SimpleReaches", "params": {}},
        task_binding_spec=None,
        graph_spec=_minimal_graph_spec(),
        status=WorkerStatus.RUNNING,
    )

    _run_training(job)

    assert job.status == WorkerStatus.ERROR
    events = []
    while not event_queue.empty():
        events.append(event_queue.get())
    assert events[0]["type"] == "training_error"
    assert "task_binding_spec" in events[0]["error"]
    assert all(event is None or event.get("type") != "training_progress" for event in events)
