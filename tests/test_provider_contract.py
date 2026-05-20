from __future__ import annotations

import sqlite3
import queue
import threading
from pathlib import Path

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
)
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
from feedbax.web.worker.app import WorkerStatus, _Job, _extract_training_cfg, _run_training_stub


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
    assert (
        manifest.capabilities["enumerate_studio_schemas"].output_schema
        == "StudioSchemaRegistry"
    )
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
        issue["type"] == "runtime_introspection_unavailable"
        and issue["severity"] == "warning"
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

    issue = next(
        issue for issue in registry.issues if issue.type == "runtime_introspection_failed"
    )
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


def test_worker_training_cfg_uses_task_n_steps() -> None:
    cfg = _extract_training_cfg(
        {"n_batches": 4, "n_reach_steps": 80},
        {"type": "DelayedReaches", "params": {"n_steps": 140}},
    )

    assert cfg.n_batches == 4
    assert cfg.n_reach_steps == 140
