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
from feedbax.web.app import create_app
from feedbax.web.worker.app import WorkerStatus, _Job, _run_training_stub


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


def test_provider_manifest_exposes_phase_one_capabilities() -> None:
    manifest = provider_manifest()

    assert manifest.provider == "feedbax"
    assert manifest.capabilities["validate_graph_spec"].input_schema == "GraphSpec"
    assert manifest.capabilities["start_training_run"].output_schema == "TrainingRunManifest"
    assert "training_checkpoint" in manifest.artifact_roles
    assert "TrainingRunManifest" in manifest.schemas


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


def test_training_manifest_writes_artifacts_and_rebuildable_index(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.eqx"
    checkpoint.write_bytes(b"checkpoint bytes")

    root = tmp_path / "runs"
    manifest, path = write_training_run_manifest(
        job_id="job-1",
        total_batches=2,
        training_spec=_minimal_training_spec(),
        task_spec={"type": "SimpleReaches", "params": {}},
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
