from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from feedbax.bin import orchestrate
from feedbax.orchestration import (
    BudgetPolicy,
    EnvironmentDeclaration,
    LaunchPolicy,
    RunBundle,
    RunEventEmitter,
    RunRowSpec,
    RunSetState,
    RunSetStateStore,
    StageState,
    StateLockError,
)
from feedbax.orchestration.drivers.local import LocalOrchestrationDriver
from feedbax.orchestration.drivers.runpod import RunPodOrchestrationDriver
from feedbax.orchestration.stages import PreflightFailed
from feedbax.orchestration.state import RowState


def _bundle(
    tmp_path: Path,
    *,
    run_set_id: str = "2026-01-02-cli",
    rows: list[RunRowSpec] | None = None,
    max_wall_clock_seconds: float = 10.0,
) -> RunBundle:
    return RunBundle(
        run_set_id=run_set_id,
        rows=rows or [RunRowSpec(row_id="row-a", command=[sys.executable, "-c", "pass"])],
        environment=EnvironmentDeclaration(python_version="3.12"),
        launch_policy=LaunchPolicy(max_parallel_rows=2),
        budget=BudgetPolicy(max_wall_clock_seconds=max_wall_clock_seconds),
        orchestration_root=str(tmp_path),
    )


def _write_bundle(bundle: RunBundle, path: Path) -> Path:
    path.write_text(bundle.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def _save_state(bundle: RunBundle, state: RunSetState) -> None:
    bundle.run_set_dir.mkdir(parents=True, exist_ok=True)
    _write_bundle(bundle, bundle.run_set_dir / "bundle.json")
    RunSetStateStore(bundle.run_set_dir / "state.json").save(state)


def test_status_line_format_is_stable(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    run_set_dir = bundle.run_set_dir
    event_path = run_set_dir / "events" / "row-a.events.jsonl"
    event_path.parent.mkdir(parents=True, exist_ok=True)
    RunEventEmitter(
        run_set_id=bundle.run_set_id,
        row_id="row-a",
        path=event_path,
        heartbeat_seconds=None,
    ).emit_progress(
        {"batch": 3, "total_batches": 8, "loss": 0.125, "phase": "train"},
        batch=3,
        total_batches=8,
        force=True,
    )
    state = RunSetState(
        run_set_id=bundle.run_set_id,
        rows={"row-a": RowState(status="running", event_seq_high_water_mark=0)},
        stages={
            "ASSEMBLE": StageState(status="completed"),
            "PREFLIGHT": StageState(status="completed"),
            "LAUNCH": StageState(status="completed"),
        },
    )

    line = orchestrate.format_status_line(
        state,
        "row-a",
        run_set_dir=run_set_dir,
        now_ms=event_path.stat().st_mtime_ns // 1_000_000 + 10_000,
    )

    assert line.startswith("row=row-a status=running batch=3/8 last_loss=0.125 last_event_age_s=")
    assert " seq=0 stages=ASSEMBLE:completed,PREFLIGHT:completed,PROVISION:pending" in line


def test_json_status_validates_state_document(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: Any,
) -> None:
    monkeypatch.setenv("FEEDBAX_ORCHESTRATION_ROOT", str(tmp_path))
    bundle = _bundle(tmp_path, run_set_id="json-status")
    state = RunSetState(
        run_set_id=bundle.run_set_id,
        rows={"row-a": RowState(status="completed")},
    )
    _save_state(bundle, state)

    assert orchestrate.main(["status", "--run-set", bundle.run_set_id, "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert RunSetState.model_validate(payload).run_set_id == "json-status"


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (PreflightFailed("bad bundle"), orchestrate.EXIT_PREFLIGHT),
        (StateLockError("locked"), orchestrate.EXIT_LOCK),
        (RuntimeError("other"), orchestrate.EXIT_OTHER),
        (
            RunSetState(
                run_set_id="set",
                rows={"row": RowState(status="failed")},
            ),
            orchestrate.EXIT_ROW_FAILURE,
        ),
        (
            RunSetState(
                run_set_id="set",
                rows={"row": RowState(status="stopped")},
                abort_reason="budget-exceeded",
            ),
            orchestrate.EXIT_BUDGET,
        ),
    ],
)
def test_exit_code_classes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    result: BaseException | RunSetState,
    expected: int,
) -> None:
    bundle = _bundle(tmp_path)
    bundle_path = _write_bundle(bundle, tmp_path / "bundle.json")

    def fake_run_engine(*_args: Any, **_kwargs: Any) -> RunSetState:
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(orchestrate, "_run_engine", fake_run_engine)

    assert orchestrate.main(["launch", "--bundle", str(bundle_path)]) == expected


def test_watch_exits_after_all_rows_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: Any,
) -> None:
    monkeypatch.setenv("FEEDBAX_ORCHESTRATION_ROOT", str(tmp_path))
    bundle = _bundle(
        tmp_path,
        run_set_id="watch-terminal",
        rows=[
            RunRowSpec(row_id="a", command=[sys.executable, "-c", "pass"]),
            RunRowSpec(row_id="b", command=[sys.executable, "-c", "pass"]),
        ],
    )
    events_dir = bundle.run_set_dir / "events"
    for row_id in ("a", "b"):
        emitter = RunEventEmitter(
            run_set_id=bundle.run_set_id,
            row_id=row_id,
            path=events_dir / f"{row_id}.events.jsonl",
            heartbeat_seconds=None,
        )
        emitter.emit_terminal("complete", {"row": row_id})
        emitter.close()
    _save_state(
        bundle,
        RunSetState(
            run_set_id=bundle.run_set_id,
            rows={"a": RowState(status="completed"), "b": RowState(status="completed")},
        ),
    )

    assert orchestrate.main(["watch", "--run-set", bundle.run_set_id, "--poll-interval", "0"]) == 0

    assert capsys.readouterr().out.splitlines() == [
        'row=a seq=0 type=complete payload={"row":"a"}',
        'row=b seq=0 type=complete payload={"row":"b"}',
    ]


def test_collect_and_teardown_are_idempotent_after_completed_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FEEDBAX_ORCHESTRATION_ROOT", str(tmp_path))
    bundle = _bundle(tmp_path, run_set_id="idempotent")
    stages = {stage: StageState(status="completed") for stage in orchestrate.STAGE_ORDER}
    _save_state(
        bundle,
        RunSetState(
            run_set_id=bundle.run_set_id,
            rows={"row-a": RowState(status="completed")},
            stages=stages,
        ),
    )

    assert orchestrate.main(["collect", "--run-set", bundle.run_set_id]) == 0
    assert orchestrate.main(["collect", "--run-set", bundle.run_set_id]) == 0
    assert orchestrate.main(["teardown", "--run-set", bundle.run_set_id]) == 0
    assert orchestrate.main(["teardown", "--run-set", bundle.run_set_id, "--force"]) == 0


def test_two_row_local_driver_demo_through_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: Any,
) -> None:
    monkeypatch.setenv("FEEDBAX_ORCHESTRATION_ROOT", str(tmp_path / "orch"))
    script = tmp_path / "row.py"
    script.write_text(
        """
from pathlib import Path
import os
from feedbax.orchestration.events import RunEventEmitter

row = os.environ["FEEDBAX_ROW_ID"]
row_dir = Path(os.environ["FEEDBAX_ROW_DIR"])
with RunEventEmitter.from_env(heartbeat_seconds=None) as emitter:
    emitter.emit("ready", {"row": row})
    emitter.emit_progress(
        {"phase": "train", "batch": 1, "total_batches": 1, "loss": 0.5},
        batch=1,
        total_batches=1,
        force=True,
    )
    (row_dir / "payload.json").write_text('{"row": "%s"}\\n' % row, encoding="utf-8")
    emitter.emit_terminal("complete", {"row": row})
""".strip(),
        encoding="utf-8",
    )
    bundle = _bundle(
        tmp_path / "orch",
        run_set_id="local-demo",
        rows=[
            RunRowSpec(
                row_id="row-a",
                command=[sys.executable, str(script)],
                collect=["payload.json"],
            ),
            RunRowSpec(
                row_id="row-b",
                command=[sys.executable, str(script)],
                collect=["payload.json"],
            ),
        ],
    )
    bundle_path = _write_bundle(bundle, tmp_path / "bundle.json")

    class FastLocalDriver(LocalOrchestrationDriver):
        def __init__(self) -> None:
            super().__init__(cwd=tmp_path, freeze_lines=("feedbax==test",))

    monkeypatch.setattr(orchestrate, "LocalOrchestrationDriver", FastLocalDriver)

    assert orchestrate.main(["preflight", "--bundle", str(bundle_path)]) == 0
    assert orchestrate.main(["launch", "--bundle", str(bundle_path), "--driver", "local"]) == 0
    assert orchestrate.main(["status", "--run-set", "local-demo"]) == 0
    assert orchestrate.main(["certify", "--run-set", "local-demo"]) == 0
    assert orchestrate.main(["teardown", "--run-set", "local-demo"]) == 0
    assert orchestrate.main(["resume", "--run-set", "local-demo"]) == 0

    status_lines = [
        line for line in capsys.readouterr().out.splitlines() if line.startswith("row=")
    ]
    assert status_lines == [
        "row=row-a status=completed batch=1/1 last_loss=0.5 "
        "last_event_age_s=0 seq=2 stages=ASSEMBLE:completed,PREFLIGHT:completed,"
        "PROVISION:completed,REALIZE_ENV:completed,STAGE_INPUTS:completed,"
        "LAUNCH:completed,MONITOR:completed,COLLECT:completed,CERTIFY:completed,"
        "TEARDOWN:completed,REGISTER:completed",
        "row=row-b status=completed batch=1/1 last_loss=0.5 "
        "last_event_age_s=0 seq=2 stages=ASSEMBLE:completed,PREFLIGHT:completed,"
        "PROVISION:completed,REALIZE_ENV:completed,STAGE_INPUTS:completed,"
        "LAUNCH:completed,MONITOR:completed,COLLECT:completed,CERTIFY:completed,"
        "TEARDOWN:completed,REGISTER:completed",
    ]


def test_runpod_driver_is_constructed_from_durable_bundle_metadata(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path).model_copy(
        update={
            "driver": "runpod",
            "environment": EnvironmentDeclaration(
                python_version="3.12",
                image_id="runpod/pytorch:1.0.3",
                metadata={
                    "runpod_pod_id": "pod-123",
                    "runpod_ssh_host": "198.51.100.10",
                    "runpod_ssh_port": 2222,
                    "runpod_gpu_id": "NVIDIA GeForce RTX 4090",
                    "runpod_path_patches": [
                        {
                            "remote_file": "/workspace/feedbax/pyproject.toml",
                            "from": "/local/feedbax",
                            "to": "/workspace/feedbax",
                        }
                    ],
                },
            ),
        }
    )

    driver = orchestrate._driver_for_bundle(bundle)

    assert isinstance(driver, RunPodOrchestrationDriver)
    assert driver.config.pod_id == "pod-123"
    assert driver.config.path_patches[0][0] == "/workspace/feedbax/pyproject.toml"


def test_load_bundle_migrates_v1_deadman_defaults(tmp_path: Path) -> None:
    path = tmp_path / "bundle-v1.json"
    payload = _bundle(tmp_path).model_dump(mode="json")
    payload["schema_version"] = "feedbax.orchestration.run_bundle.v1"
    payload.pop("deadman_enabled")
    payload.pop("deadman_silence_seconds")
    path.write_text(json.dumps(payload), encoding="utf-8")

    bundle = orchestrate._load_bundle(path)

    assert bundle.deadman_enabled is False
    assert bundle.deadman_silence_seconds == 1800


def test_launch_cli_exposes_deadman_bundle_overrides() -> None:
    args = orchestrate.build_parser().parse_args(
        [
            "launch",
            "--bundle",
            "bundle.json",
            "--deadman",
            "--deadman-silence-seconds",
            "900",
        ]
    )

    assert args.deadman is True
    assert args.deadman_silence_seconds == 900
