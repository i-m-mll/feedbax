from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from feedbax.contracts.base import ParentRef
from feedbax.contracts.manifest import (
    EvaluationRunManifest,
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


def _evaluation_manifest(
    run_id: str,
    training_run_ids: list[str],
    *,
    status: str = "completed",
) -> EvaluationRunManifest:
    return EvaluationRunManifest(
        id=run_id,
        status=status,
        evaluation_spec=spec_payload(
            "EvaluationRunSpec",
            {
                "evaluation_type": "test",
                "training_run_ids": training_run_ids,
                "inputs": [],
                "params": {},
            },
        ),
        input_training_runs=[
            ParentRef(kind="TrainingRunManifest", id=value) for value in training_run_ids
        ],
    )


def test_create_eval_run_writes_versioned_manifest(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FEEDBAX_RUNS_DIR", str(tmp_path))
    write_manifest(_training_manifest("training-a", "completed"), root=tmp_path)

    payload = runs.CreateEvalRunRequest(
        training_run_id="training-a",
        name="eval-a",
        eval_params={"perturbation": "none"},
    )

    created = asyncio.run(runs.create_eval_run(payload))
    manifest, path = runs._load_evaluation_manifest_from_index(created.id)

    assert isinstance(manifest, EvaluationRunManifest)
    assert path.is_file()
    assert created.status == "pending"
    assert created.training_run_id == "training-a"
    assert created.name == "eval-a"
    assert manifest.evaluation_spec.schema_id == "feedbax.spec.evaluation_run"
    assert manifest.evaluation_spec.schema_version == "feedbax.spec.evaluation_run.v1"
    assert manifest.evaluation_spec.inline["params"] == {
        "perturbation": "none",
        "label": "eval-a",
    }
    assert manifest.input_training_runs == [
        ParentRef(kind="TrainingRunManifest", id="training-a", role="training_run")
    ]

    write_manifest(
        manifest.model_copy(update={"status": "completed", "summary_metrics": {"loss": 0.25}}),
        root=tmp_path,
    )
    repeated = asyncio.run(runs.create_eval_run(payload))
    preserved, _path = runs._load_evaluation_manifest_from_index(repeated.id)

    assert repeated.id == created.id
    assert repeated.status == "completed"
    assert preserved.status == "completed"
    assert preserved.summary_metrics == {"loss": 0.25}


def test_create_eval_run_fails_when_manifest_persistence_fails(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FEEDBAX_RUNS_DIR", str(tmp_path))
    write_manifest(_training_manifest("training-a", "completed"), root=tmp_path)

    def fail_write(*_args, **_kwargs):
        raise RuntimeError("manifest store offline")

    monkeypatch.setattr(runs, "write_manifest", fail_write)

    payload = runs.CreateEvalRunRequest(
        training_run_id="training-a",
        name="eval-a",
        eval_params={"perturbation": "none"},
    )

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(runs.create_eval_run(payload))

    assert excinfo.value.status_code == 500
    assert "Could not persist evaluation manifest" in str(excinfo.value.detail)


def test_create_eval_run_rejects_missing_parent_before_writing(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FEEDBAX_RUNS_DIR", str(tmp_path))

    def fail_write(*_args, **_kwargs):
        raise AssertionError("missing parent must be rejected before writing")

    monkeypatch.setattr(runs, "write_manifest", fail_write)
    payload = runs.CreateEvalRunRequest(
        training_run_id="feedbax-training-run:missing",
        name="eval-a",
    )

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(runs.create_eval_run(payload))

    assert excinfo.value.status_code == 404
    assert "Training run 'feedbax-training-run:missing' not found" == excinfo.value.detail


def test_create_eval_run_rejects_wrong_kind_parent_before_writing(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FEEDBAX_RUNS_DIR", str(tmp_path))
    parent_id = "feedbax-eval-run:not-training"
    write_manifest(_evaluation_manifest(parent_id, []), root=tmp_path)

    def fail_write(*_args, **_kwargs):
        raise AssertionError("wrong-kind parent must be rejected before writing")

    monkeypatch.setattr(runs, "write_manifest", fail_write)
    payload = runs.CreateEvalRunRequest(training_run_id=parent_id, name="eval-a")

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(runs.create_eval_run(payload))

    assert excinfo.value.status_code == 409
    assert "EvaluationRunManifest, not TrainingRunManifest" in excinfo.value.detail


def test_list_eval_runs_returns_empty_for_indexed_training_manifest(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FEEDBAX_RUNS_DIR", str(tmp_path))
    training_id = "feedbax-training-run:no-evaluations"
    write_manifest(_training_manifest(training_id, "completed"), root=tmp_path)

    assert asyncio.run(runs.list_eval_runs(training_id)) == []


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


def test_training_run_index_lists_sweep_axis_hyperparams(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FEEDBAX_RUNS_DIR", str(tmp_path))
    manifest = _training_manifest("feedbax-training-run:sweep", "pending")
    manifest.metadata["studio"]["axis_coordinates"] = {"loss_weight": 1e-5}
    write_manifest(manifest, root=tmp_path)
    client = TestClient(create_app())

    response = client.get("/api/runs/training")

    assert response.status_code == 200
    assert response.json()[0]["hyperparams"]["axis_loss_weight"] == 1e-5


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


def test_superseding_training_run_marks_dependent_evaluations_stale(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FEEDBAX_RUNS_DIR", str(tmp_path))
    training_id = "feedbax-training-run:completed"
    replacement_id = "feedbax-training-run:replacement"
    write_manifest(_training_manifest(training_id, "completed"), root=tmp_path)
    dependent_path = write_manifest(
        _evaluation_manifest("feedbax-eval-run:dependent", [training_id]),
        root=tmp_path,
    )
    unrelated_path = write_manifest(
        _evaluation_manifest("feedbax-eval-run:unrelated", ["feedbax-training-run:other"]),
        root=tmp_path,
    )
    client = TestClient(create_app())

    response = client.post(
        f"/api/runs/training/{training_id}/supersede",
        json={"superseded_by": replacement_id, "reason": "new sweep"},
    )

    assert response.status_code == 200
    dependent = load_manifest(dependent_path)
    unrelated = load_manifest(unrelated_path)
    assert isinstance(dependent, EvaluationRunManifest)
    assert dependent.status == "stale"
    assert dependent.metadata["staleness_reason"] == "upstream superseded"
    assert dependent.metadata["stale_from_status"] == "completed"
    assert dependent.metadata["stale_parent_ids"] == [training_id]
    assert dependent.metadata["upstream_supersessions"][training_id] == {
        "superseded_at": dependent.metadata["stale_at"],
        "superseded_by": replacement_id,
        "reason": "new sweep",
    }
    assert isinstance(unrelated, EvaluationRunManifest)
    assert unrelated.status == "completed"


def test_superseding_training_run_is_idempotent_and_rejects_self_supersession(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FEEDBAX_RUNS_DIR", str(tmp_path))
    training_id = "feedbax-training-run:completed"
    replacement_id = "feedbax-training-run:replacement"
    training_path = write_manifest(
        _training_manifest(training_id, "completed"),
        root=tmp_path,
    )
    evaluation_path = write_manifest(
        _evaluation_manifest("feedbax-eval-run:dependent", [training_id]),
        root=tmp_path,
    )
    client = TestClient(create_app())
    payload = {"superseded_by": replacement_id, "reason": "new sweep"}

    self_link = client.post(
        f"/api/runs/training/{training_id}/supersede",
        json={"superseded_by": training_id},
    )
    first = client.post(f"/api/runs/training/{training_id}/supersede", json=payload)
    first_training = load_manifest(training_path)
    first_evaluation = load_manifest(evaluation_path)
    second = client.post(f"/api/runs/training/{training_id}/supersede", json=payload)
    second_training = load_manifest(training_path)
    second_evaluation = load_manifest(evaluation_path)
    conflicting = client.post(
        f"/api/runs/training/{training_id}/supersede",
        json={"superseded_by": "feedbax-training-run:different"},
    )

    assert self_link.status_code == 409
    assert first.status_code == 200
    assert second.status_code == 200
    assert conflicting.status_code == 409
    assert first_training == second_training
    assert first_evaluation == second_evaluation


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
