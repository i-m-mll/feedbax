"""Corrupt durable Studio run state remains visible as an explicit failure."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from feedbax.orchestration.state import RowState, RunSetState, RunSetStateStore
from feedbax.web.api import runs as runs_api
from feedbax.web.api import training as training_api
from feedbax.web.services.training_service import (
    RunStateCorruptionError,
    TrainingService,
)


pytestmark = [pytest.mark.feedbax_contract, pytest.mark.no_silent_substitution_contract]


def _write_legacy_v2_run(root: Path, *, state_text: str | None = None) -> Path:
    run_dir = root / "private-run-root"
    run_dir.mkdir(parents=True)
    (run_dir / "bundle.json").write_text(
        json.dumps(
            {
                "schema_id": "feedbax.orchestration.run_bundle",
                "schema_version": "feedbax.orchestration.run_bundle.v2",
                "run_set_id": "set-visible",
                "rows": [
                    {
                        "row_id": "job-visible",
                        "metadata": {"worker_start": {"total_batches": 4}},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    state_path = run_dir / "state.json"
    if state_text is None:
        RunSetStateStore(state_path).save(
            RunSetState(
                run_set_id="set-visible",
                rows={"job-visible": RowState(status="running")},
            )
        )
    else:
        state_path.write_text(state_text, encoding="utf-8")
    return run_dir


def test_missing_bundle_raises_typed_corruption_and_logs_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("FEEDBAX_ORCHESTRATION_ROOT", str(tmp_path))
    run_dir = tmp_path / "private-missing-bundle"
    run_dir.mkdir()
    (run_dir / "state.json").write_text("{}", encoding="utf-8")

    with caplog.at_level(logging.ERROR), pytest.raises(RunStateCorruptionError) as raised:
        TrainingService().rebuild_cache_from_state_docs()

    assert raised.value.path == run_dir / "bundle.json"
    assert "bundle document is missing" in raised.value.reason
    assert str(run_dir / "bundle.json") in caplog.text


def test_malformed_non_v2_bundle_raises_typed_corruption(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("FEEDBAX_ORCHESTRATION_ROOT", str(tmp_path))
    run_dir = tmp_path / "private-malformed-bundle"
    run_dir.mkdir()
    (run_dir / "state.json").write_text("{}", encoding="utf-8")
    bundle_path = run_dir / "bundle.json"
    bundle_path.write_text(
        json.dumps(
            {
                "schema_id": "feedbax.orchestration.run_bundle",
                "schema_version": "feedbax.orchestration.run_bundle.v11",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RunStateCorruptionError) as raised:
        TrainingService().rebuild_cache_from_state_docs()

    assert raised.value.path == bundle_path
    assert "bundle validation failed" in raised.value.reason


def test_valid_legacy_v2_bundle_remains_visible_without_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("FEEDBAX_ORCHESTRATION_ROOT", str(tmp_path))
    run_dir = _write_legacy_v2_run(tmp_path)
    state_path = run_dir / "state.json"
    original_state = state_path.read_bytes()

    service = TrainingService()
    asyncio.run(service.reconcile_from_state_docs())
    rows = service.list_live_training_runs()

    assert [(row["job_id"], row["status"]) for row in rows] == [("job-visible", "running")]
    assert state_path.read_bytes() == original_state


def test_training_run_list_surfaces_sanitized_corruption_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("FEEDBAX_ORCHESTRATION_ROOT", str(tmp_path))
    run_dir = _write_legacy_v2_run(tmp_path, state_text="{not-json")
    monkeypatch.setattr(runs_api, "training_service", TrainingService())
    monkeypatch.setattr(runs_api, "iter_indexed_manifest_records_by_kind", lambda *_args: [])
    monkeypatch.setattr(runs_api, "_legacy_training_runs_from_model_db", lambda: [])
    app = FastAPI()
    app.include_router(runs_api.router, prefix="/api/runs")

    with caplog.at_level(logging.ERROR):
        response = TestClient(app).get("/api/runs/training")

    assert response.status_code == 500
    assert response.json() == {"detail": "Persisted Studio run state is corrupt"}
    assert str(run_dir) not in response.text
    assert str(run_dir / "state.json") in caplog.text


def test_training_status_surfaces_sanitized_corruption_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("FEEDBAX_ORCHESTRATION_ROOT", str(tmp_path))
    run_dir = _write_legacy_v2_run(tmp_path, state_text="{not-json")
    monkeypatch.setattr(training_api, "training_service", TrainingService())
    app = FastAPI()
    app.include_router(training_api.router, prefix="/api/training")

    with caplog.at_level(logging.ERROR):
        response = TestClient(app).get("/api/training/job-visible")

    assert response.status_code == 500
    assert response.json() == {"detail": "Persisted Studio run state is corrupt"}
    assert str(run_dir) not in response.text
    assert str(run_dir / "state.json") in caplog.text
