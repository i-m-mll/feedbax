"""Corrupt durable Studio run state remains visible as an explicit failure."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
import subprocess
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from feedbax.orchestration.stages import STAGE_ASSEMBLE
from feedbax.orchestration.state import RowState, RunSetState, RunSetStateStore, StageState
from feedbax.web.api import runs as runs_api
from feedbax.web.api import training as training_api
from feedbax.web.services.training_service import (
    RunStateCorruptionError,
    TrainingService,
)
from tests.test_orchestration_core import _bundle, _compiled_row


pytestmark = [pytest.mark.feedbax_contract, pytest.mark.no_silent_substitution_contract]


def _write_current_run(
    root: Path,
    *,
    state_text: str | None = None,
    state_run_set_id: str = "set-visible",
    state_row_ids: tuple[str, ...] = ("job-visible",),
) -> Path:
    run_dir = root / "private-run-root"
    run_dir.mkdir(parents=True)
    bundle = _bundle(
        root / "bundle-source",
        rows=[_compiled_row("job-visible")],
        run_set_id="set-visible",
    )
    (run_dir / "bundle.json").write_text(bundle.model_dump_json(), encoding="utf-8")
    state_path = run_dir / "state.json"
    if state_text is None:
        RunSetStateStore(state_path).save(
            RunSetState(
                run_set_id=state_run_set_id,
                rows={row_id: RowState(status="running") for row_id in state_row_ids},
                stages={STAGE_ASSEMBLE: StageState(status="completed")},
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
    RunSetStateStore(run_dir / "state.json").save(
        RunSetState(
            run_set_id="set-missing",
            stages={STAGE_ASSEMBLE: StageState(status="completed")},
        )
    )

    with caplog.at_level(logging.ERROR), pytest.raises(RunStateCorruptionError) as raised:
        TrainingService().rebuild_cache_from_state_docs()

    assert raised.value.path == run_dir / "bundle.json"
    assert "completed ASSEMBLE state is missing its bundle document" in raised.value.reason
    assert str(run_dir / "bundle.json") in caplog.text


def test_service_construction_does_not_scan_corrupt_records(tmp_path: Path) -> None:
    run_dir = tmp_path / "private-corrupt-at-import"
    run_dir.mkdir()
    (run_dir / "state.json").write_text("{not-json", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from feedbax.web.services.training_service import training_service; "
            "print(type(training_service).__name__)",
        ],
        env={**os.environ, "FEEDBAX_ORCHESTRATION_ROOT": str(tmp_path)},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "TrainingService"


def test_state_before_bundle_publication_is_a_valid_empty_window(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("FEEDBAX_ORCHESTRATION_ROOT", str(tmp_path))
    run_dir = tmp_path / "private-assembling"
    run_dir.mkdir()
    RunSetStateStore(run_dir / "state.json").save(
        RunSetState(
            run_set_id="set-assembling",
            stages={STAGE_ASSEMBLE: StageState(status="running")},
        )
    )

    assert TrainingService().list_live_training_runs() == []


def test_bundle_deletion_race_is_transient_not_corruption(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("FEEDBAX_ORCHESTRATION_ROOT", str(tmp_path))
    run_dir = _write_current_run(tmp_path)
    bundle_path = run_dir / "bundle.json"
    original_read_text = Path.read_text

    def raced_read_text(path: Path, *args, **kwargs) -> str:
        if path == bundle_path:
            raise FileNotFoundError(bundle_path)
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", raced_read_text)

    assert TrainingService().list_live_training_runs() == []


def test_status_uses_single_bundle_read_during_deletion_race(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("FEEDBAX_ORCHESTRATION_ROOT", str(tmp_path))
    run_dir = _write_current_run(tmp_path)
    bundle_path = run_dir / "bundle.json"
    service = TrainingService()
    service.rebuild_cache_from_state_docs()
    original_read_text = Path.read_text
    bundle_reads = 0

    def raced_read_text(path: Path, *args, **kwargs) -> str:
        nonlocal bundle_reads
        if path == bundle_path:
            bundle_reads += 1
            if bundle_reads > 1:
                raise FileNotFoundError(bundle_path)
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", raced_read_text)

    status = asyncio.run(service.get_status("job-visible"))

    assert status is not None
    assert status["total_batches"] == 1
    assert bundle_reads == 1


def test_duplicate_row_ids_raise_typed_corruption(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("FEEDBAX_ORCHESTRATION_ROOT", str(tmp_path))
    run_dir = _write_current_run(tmp_path)
    bundle_path = run_dir / "bundle.json"
    raw = json.loads(bundle_path.read_text(encoding="utf-8"))
    raw["rows"].append(dict(raw["rows"][0]))
    bundle_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(RunStateCorruptionError) as raised:
        TrainingService().list_live_training_runs()

    assert "duplicate row_id" in raised.value.reason


@pytest.mark.parametrize(
    ("state_run_set_id", "state_row_ids", "reason"),
    [
        ("set-other", ("job-visible",), "run_set_id values disagree"),
        ("set-visible", (), "row identities disagree"),
    ],
)
def test_cross_document_mismatch_raises_typed_corruption(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    state_run_set_id: str,
    state_row_ids: tuple[str, ...],
    reason: str,
) -> None:
    monkeypatch.setenv("FEEDBAX_ORCHESTRATION_ROOT", str(tmp_path))
    _write_current_run(
        tmp_path,
        state_run_set_id=state_run_set_id,
        state_row_ids=state_row_ids,
    )

    with pytest.raises(RunStateCorruptionError) as raised:
        TrainingService().list_live_training_runs()

    assert reason in raised.value.reason


def test_malformed_non_v2_bundle_raises_typed_corruption(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("FEEDBAX_ORCHESTRATION_ROOT", str(tmp_path))
    run_dir = tmp_path / "private-malformed-bundle"
    run_dir.mkdir()
    RunSetStateStore(run_dir / "state.json").save(
        RunSetState(
            run_set_id="set-malformed",
            stages={STAGE_ASSEMBLE: StageState(status="completed")},
        )
    )
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


def test_valid_current_bundle_remains_visible_without_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("FEEDBAX_ORCHESTRATION_ROOT", str(tmp_path))
    run_dir = _write_current_run(tmp_path)
    state_path = run_dir / "state.json"
    original_state = state_path.read_bytes()

    service = TrainingService()
    service.rebuild_cache_from_state_docs()
    rows = service.list_live_training_runs()

    assert [(row["job_id"], row["status"]) for row in rows] == [("job-visible", "running")]
    assert state_path.read_bytes() == original_state


def test_training_run_list_surfaces_sanitized_corruption_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("FEEDBAX_ORCHESTRATION_ROOT", str(tmp_path))
    run_dir = _write_current_run(tmp_path, state_text="{not-json")
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
    run_dir = _write_current_run(tmp_path, state_text="{not-json")
    monkeypatch.setattr(training_api, "training_service", TrainingService())
    app = FastAPI()
    app.include_router(training_api.router, prefix="/api/training")

    with caplog.at_level(logging.ERROR):
        response = TestClient(app).get("/api/training/job-visible")

    assert response.status_code == 500
    assert response.json() == {"detail": "Persisted Studio run state is corrupt"}
    assert str(run_dir) not in response.text
    assert str(run_dir / "state.json") in caplog.text
