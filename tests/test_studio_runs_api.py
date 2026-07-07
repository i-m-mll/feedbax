from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from feedbax.contracts.manifest import (
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


def test_pending_training_manifest_lifecycle_is_status_guarded(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FEEDBAX_RUNS_DIR", str(tmp_path))
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
