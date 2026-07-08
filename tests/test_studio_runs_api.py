from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from feedbax.contracts.manifest import (
    ParentRef,
    TrainingRunManifest,
    load_manifest,
    spec_payload,
    write_manifest,
)
from feedbax.web.api import runs
from feedbax.web.app import create_app

pytestmark = [pytest.mark.feedbax_contract, pytest.mark.no_silent_substitution_contract]


def _training_manifest(run_id: str, status: str) -> TrainingRunManifest:
    return TrainingRunManifest(
        id=run_id,
        status=status,
        graph_spec=spec_payload("GraphSpec", {"nodes": {}}),
        training_spec=spec_payload("TrainingSpec", {"n_batches": 25, "batch_size": 8}),
        task_spec=spec_payload("TaskSpec", {"type": "ReachingTask"}),
        summary_metrics={"final_validation_loss": 0.25},
        metadata={
            "planned": status == "pending",
            "studio": {
                "stage_id": "stage:train",
                "scenario_id": "scenario:train",
                "axis_coordinates": {"duration": 80},
            },
        },
    )


def test_create_eval_run_fails_when_persistence_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_db_session():
        raise RuntimeError("database offline")

    monkeypatch.setattr(runs, "db_session", fail_db_session)

    payload = runs.CreateEvalRunRequest(
        training_run_id="training-a",
        name="eval-a",
        eval_params={"perturbation": "none"},
    )

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(runs.create_eval_run(payload))

    assert excinfo.value.status_code == 500
    assert "Could not persist evaluation run" in str(excinfo.value.detail)


def test_training_run_index_lists_pending_manifest_rows(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FEEDBAX_RUNS_DIR", str(tmp_path))
    monkeypatch.setattr(runs, "_legacy_training_runs_from_model_db", lambda: [])
    write_manifest(_training_manifest("feedbax-training-run:pending", "pending"), root=tmp_path)
    client = TestClient(create_app())

    response = client.get("/api/runs/training")

    assert response.status_code == 200
    payload = response.json()
    assert payload[0]["id"] == "feedbax-training-run:pending"
    assert payload[0]["status"] == "pending"
    assert payload[0]["planned"] is True
    assert payload[0]["stage_id"] == "stage:train"
    assert payload[0]["hyperparams"]["n_batches"] == 25
    assert payload[0]["hyperparams"]["axis_duration"] == 80


def test_training_run_index_lists_sweep_axis_hyperparams(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FEEDBAX_RUNS_DIR", str(tmp_path))
    monkeypatch.setattr(runs, "_legacy_training_runs_from_model_db", lambda: [])
    manifest = _training_manifest("feedbax-training-run:sweep", "pending")
    manifest.metadata["studio"]["axis_coordinates"] = {"loss_weight": 1e-5}
    write_manifest(manifest, root=tmp_path)
    client = TestClient(create_app())

    response = client.get("/api/runs/training")

    assert response.status_code == 200
    assert response.json()[0]["hyperparams"]["axis_loss_weight"] == 1e-5


def test_training_run_index_surfaces_legacy_checkpoint_adoption_state(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FEEDBAX_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runs, "_legacy_training_runs_from_model_db", lambda: [])
    legacy_root = tmp_path / "legacy-checkpoints"
    checkpoint_dir = legacy_root / "checkpoint_000001"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "model.eqx").write_bytes(b"model")
    (checkpoint_dir / "optimizer_state.eqx").write_bytes(b"optimizer")
    (checkpoint_dir / "metadata.json").write_text("{}", encoding="utf-8")
    manifest = _training_manifest("feedbax-training-run:legacy-checkpoint", "completed")
    manifest.checkpoint_custody = [
        ParentRef(
            kind="TrainingCheckpointRoot",
            id="legacy-checkpoint-root",
            role="training_checkpoint_custody",
            uri=str(legacy_root),
        )
    ]
    write_manifest(manifest, root=tmp_path / "runs")
    client = TestClient(create_app())

    response = client.get("/api/runs/training")

    assert response.status_code == 200
    legacy = response.json()[0]["legacy_checkpoint"]
    assert legacy["layout_id"] == "rlrmp_eqx_stream_v0"
    assert "checkpoint predates checkpoint custody" in legacy["message"].lower()
    assert "docs/structure.md#legacy-checkpoint-adoption" in legacy["docs"]
    assert (
        legacy["adoption_entrypoint"]
        == "feedbax.training.legacy_checkpoint_adoption.adopt_legacy_checkpoint"
    )


def test_training_run_manifest_endpoint_returns_snapshot_payload(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FEEDBAX_RUNS_DIR", str(tmp_path))
    write_manifest(_training_manifest("feedbax-training-run:snapshot", "pending"), root=tmp_path)
    client = TestClient(create_app())

    response = client.get("/api/runs/training/feedbax-training-run:snapshot/manifest")

    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == "feedbax-training-run:snapshot"
    assert payload["training_spec"]["inline"]["n_batches"] == 25
    assert payload["task_spec"]["inline"]["type"] == "ReachingTask"


def test_training_run_index_merges_manifest_and_legacy_db_rows(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FEEDBAX_RUNS_DIR", str(tmp_path))
    write_manifest(_training_manifest("feedbax-training-run:pending", "pending"), root=tmp_path)
    monkeypatch.setattr(
        runs,
        "_legacy_training_runs_from_model_db",
        lambda: [
            runs.TrainingRunInfo(
                id="legacy-completed-run",
                name="Legacy completed run",
                created_at="2026-07-07T12:00:00",
                status="completed",
                hyperparams={"n_batches": 10},
            )
        ],
    )
    client = TestClient(create_app())

    response = client.get("/api/runs/training")

    assert response.status_code == 200
    ids = {row["id"] for row in response.json()}
    assert ids == {"feedbax-training-run:pending", "legacy-completed-run"}


def test_pending_training_manifest_lifecycle_is_status_guarded(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FEEDBAX_RUNS_DIR", str(tmp_path))
    monkeypatch.setattr(runs, "_legacy_training_runs_from_model_db", lambda: [])
    pending_path = write_manifest(
        _training_manifest("feedbax-training-run:pending", "pending"),
        root=tmp_path,
    )
    write_manifest(
        _training_manifest("feedbax-training-run:completed", "completed"),
        root=tmp_path,
    )
    client = TestClient(create_app())

    cancelled = client.post("/api/runs/training/feedbax-training-run:pending/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    assert load_manifest(pending_path).status == "cancelled"

    delete_cancelled = client.delete("/api/runs/training/feedbax-training-run:pending")
    assert delete_cancelled.status_code == 409

    completed_delete = client.delete("/api/runs/training/feedbax-training-run:completed")
    assert completed_delete.status_code == 409

    superseded = client.post(
        "/api/runs/training/feedbax-training-run:completed/supersede",
        json={"superseded_by": "feedbax-training-run:replacement", "reason": "new sweep"},
    )
    assert superseded.status_code == 200
    assert superseded.json()["superseded_by"] == "feedbax-training-run:replacement"


def test_pending_training_manifest_delete_removes_only_pending(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FEEDBAX_RUNS_DIR", str(tmp_path))
    monkeypatch.setattr(runs, "_legacy_training_runs_from_model_db", lambda: [])
    pending_path = write_manifest(
        _training_manifest("feedbax-training-run:delete-me", "pending"),
        root=tmp_path,
    )
    client = TestClient(create_app())

    deleted = client.delete("/api/runs/training/feedbax-training-run:delete-me")

    assert deleted.status_code == 200
    assert deleted.json()["status"] == "pending"
    assert not pending_path.exists()
    assert client.get("/api/runs/training").json() == []
