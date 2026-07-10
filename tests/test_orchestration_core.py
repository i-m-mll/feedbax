from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from feedbax.contracts.migrations import UnsupportedSpecVersion, default_spec_registry
from feedbax.contracts.training import LrScheduleSpec, OptimizerSpec
from feedbax.orchestration import conformance, schedule_eval, stages
from feedbax.orchestration.bundle import (
    RUN_BUNDLE_SCHEMA_ID,
    RUN_BUNDLE_SCHEMA_VERSION,
    BudgetPolicy,
    EnvironmentDeclaration,
    LaunchPolicy,
    RepoRevision,
    RunBundle,
    RunRowSpec,
)
from feedbax.orchestration.conformance import CheckEntry, CheckRegistry
from feedbax.orchestration.drivers.base import DriverRowProbe
from feedbax.orchestration.drivers.local import (
    LocalDriverError,
    LocalOrchestrationDriver,
    compute_environment_fingerprint,
)
from feedbax.orchestration.stages import (
    STAGE_ORDER,
    STAGE_PREFLIGHT,
    OrchestrationStageError,
    PreflightFailed,
    StageEngine,
    run_preflight_checks,
)
from feedbax.orchestration.state import (
    RUN_SET_STATE_SCHEMA_ID,
    RUN_SET_STATE_SCHEMA_VERSION,
    RowState,
    RunSetState,
    RunSetStateStore,
    StateLockError,
)
from feedbax.training.interruption import CancellationAction, CancellationDecision


class FakeDriver:
    def __init__(self, *, fail: dict[str, int] | None = None) -> None:
        self.calls: list[str] = []
        self.fail = dict(fail or {})

    def _call(self, name: str) -> None:
        self.calls.append(name)
        remaining = self.fail.get(name, 0)
        if remaining > 0:
            self.fail[name] = remaining - 1
            raise RuntimeError(f"{name} failed")

    def provision(self, bundle: RunBundle, state: RunSetState) -> dict[str, Any]:
        self._call("provision")
        return {"provisioned": True}

    def realize_env(self, bundle: RunBundle, state: RunSetState) -> str:
        self._call("realize_env")
        return "fake-fingerprint"

    def stage_inputs(self, bundle: RunBundle, state: RunSetState) -> dict[str, Any]:
        self._call("stage_inputs")
        return {"inputs": True}

    def launch_row(self, bundle: RunBundle, row: RunRowSpec, state: RunSetState) -> dict[str, Any]:
        self._call(f"launch:{row.row_id}")
        return {"pid": 1000 + len(self.calls)}

    def probe(self, bundle: RunBundle, row: RunRowSpec, state: RunSetState) -> DriverRowProbe:
        self._call(f"probe:{row.row_id}")
        return DriverRowProbe(status="completed")

    def stop_row(self, bundle: RunBundle, row: RunRowSpec, state: RunSetState) -> dict[str, Any]:
        self._call(f"stop:{row.row_id}")
        return {"stopped": row.row_id}

    def collect(self, bundle: RunBundle, row: RunRowSpec, state: RunSetState) -> dict[str, str]:
        self._call(f"collect:{row.row_id}")
        return {"payload": str(bundle.run_set_dir / row.row_id / "payload.json")}

    def teardown(self, bundle: RunBundle, state: RunSetState) -> dict[str, Any]:
        self._call("teardown")
        return {"torn_down": True}


def _bundle(
    tmp_path: Path,
    *,
    rows: list[RunRowSpec] | None = None,
    launch_policy: LaunchPolicy | None = None,
    max_wall_clock_seconds: float = 10.0,
    run_set_id: str = "2026-01-02-deadbeef",
    python_version: str | None = "3.12",
) -> RunBundle:
    return RunBundle(
        run_set_id=run_set_id,
        rows=rows or [RunRowSpec(row_id="row-a", command=[sys.executable, "-c", "pass"])],
        environment=EnvironmentDeclaration(python_version=python_version),
        launch_policy=launch_policy or LaunchPolicy(max_parallel_rows=2),
        budget=BudgetPolicy(max_wall_clock_seconds=max_wall_clock_seconds),
        orchestration_root=str(tmp_path),
    )


def _scheduled_optimizer_payload() -> dict[str, Any]:
    return OptimizerSpec(
        type="adamw",
        params={"weight_decay": 0.0},
        lr_schedule=LrScheduleSpec(
            kind="warmup_cosine",
            learning_rate_0=0.1,
            total_steps=3500,
            constant_lr_iterations=500,
            warmup_init_fraction=0.1,
            cosine_annealing_alpha=0.2,
        ),
    ).model_dump(mode="json")


def _schedule_context(
    *,
    schedule_origin_step: int,
    current_step: int,
    optimizer_count_at_current_step: int,
) -> dict[str, int]:
    return {
        "schedule_origin_step": schedule_origin_step,
        "current_step": current_step,
        "optimizer_count_at_current_step": optimizer_count_at_current_step,
    }


def test_state_atomic_write_locking_and_schema_registration(tmp_path: Path) -> None:
    store = RunSetStateStore(tmp_path / "state.json")
    old = RunSetState(run_set_id="set", rows={"row": RowState(status="pending")})
    store.save(old)

    crashed_tmp = store.save(
        old.model_copy(update={"rows": {"row": RowState(status="completed")}}),
        crash_before_replace=True,
    )

    assert crashed_tmp.exists()
    assert store.load().rows["row"].status == "pending"

    with store.lock():
        with pytest.raises(StateLockError, match="active"):
            with store.lock():
                pass

    store.lock_path.write_text(json.dumps({"pid": 999999999}), encoding="utf-8")
    with pytest.raises(StateLockError, match="stale"):
        with store.lock():
            pass
    with store.lock(break_stale=True):
        assert store.lock_path.exists()

    assert default_spec_registry.resolve("RunBundle").identity == RUN_BUNDLE_SCHEMA_ID
    assert default_spec_registry.resolve("RunBundle").current_version == RUN_BUNDLE_SCHEMA_VERSION
    assert default_spec_registry.resolve("RunSetState").identity == RUN_SET_STATE_SCHEMA_ID
    assert (
        default_spec_registry.resolve("RunSetState").current_version == RUN_SET_STATE_SCHEMA_VERSION
    )
    with pytest.raises(UnsupportedSpecVersion, match="migration_intentionally_absent=yes"):
        default_spec_registry.migrate(
            "RunBundle",
            {"schema_version": "feedbax.orchestration.run_bundle.v0"},
        )


@pytest.mark.parametrize("stop_after", STAGE_ORDER[:-1])
def test_stage_engine_resumes_from_every_stage_boundary(
    tmp_path: Path,
    stop_after: str,
) -> None:
    bundle = _bundle(tmp_path)
    store = RunSetStateStore(bundle.run_set_dir / "state.json")
    first_driver = FakeDriver()
    StageEngine(bundle=bundle, driver=first_driver, store=store).run(stop_after_stage=stop_after)

    resumed_driver = FakeDriver()
    state = StageEngine(bundle=bundle, driver=resumed_driver, store=store).run()

    assert state.stage("REGISTER").status == "completed"
    if stop_after in (
        "PROVISION",
        "REALIZE_ENV",
        "STAGE_INPUTS",
        "LAUNCH",
        "MONITOR",
        "COLLECT",
        "CERTIFY",
        "TEARDOWN",
    ):
        assert "provision" not in resumed_driver.calls


def test_stage_retry_accounting_and_abort_teardown(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    store = RunSetStateStore(bundle.run_set_dir / "state.json")
    retry_driver = FakeDriver(fail={"provision": 2})

    state = StageEngine(bundle=bundle, driver=retry_driver, store=store).run()

    assert state.stage("PROVISION").attempts == 3
    assert retry_driver.calls.count("provision") == 3

    failing_bundle = _bundle(tmp_path / "abort", run_set_id="2026-01-02-feedface")
    failing_store = RunSetStateStore(failing_bundle.run_set_dir / "state.json")
    failing_driver = FakeDriver(fail={"realize_env": 3})

    with pytest.raises(RuntimeError, match="realize_env failed"):
        StageEngine(bundle=failing_bundle, driver=failing_driver, store=failing_store).run()

    failed_state = failing_store.load()
    assert failed_state.stage("REALIZE_ENV").attempts == 3
    assert failed_state.stage("TEARDOWN").status == "completed"
    assert "teardown" in failing_driver.calls


def test_preflight_failures_record_named_checks_and_do_not_call_driver(tmp_path: Path) -> None:
    invalid = _bundle(
        tmp_path,
        rows=[
            RunRowSpec(
                row_id="row-a",
                command=[sys.executable, "-c", "pass"],
                run_spec={"schema_version": "feedbax.spec.training_run.v0"},
            )
        ],
        python_version=None,
    )
    driver = FakeDriver()

    with pytest.raises(PreflightFailed):
        StageEngine(bundle=invalid, driver=driver).run()

    state = RunSetStateStore(invalid.run_set_dir / "state.json").load()
    checks = {check.name: check for check in state.stage(STAGE_PREFLIGHT).checks}
    assert checks["environment-declaration"].status == "fail"
    assert checks["manifest-payload-normalization"].status == "fail"
    assert driver.calls == []


def test_preflight_schedule_realization_uses_optimizer_builder(tmp_path: Path) -> None:
    bundle = _bundle(
        tmp_path,
        rows=[
            RunRowSpec(
                row_id="row-a",
                command=[sys.executable, "-c", "pass"],
                run_spec={
                    "optimizer": {
                        "type": "adamw",
                        "params": {"learning_rate": 0.001},
                    }
                },
            )
        ],
    )
    checks = {check.name: check for check in run_preflight_checks(bundle)}

    assert checks["schedule-realization"].status == "pass"
    assert checks["schedule-realization"].observed == {
        "row-a": [{"optimizer_index": 0, "scheduled": False, "points": 0}]
    }

    invalid = bundle.model_copy(
        update={
            "rows": [
                RunRowSpec(
                    row_id="row-a",
                    command=[sys.executable, "-c", "pass"],
                    run_spec={"optimizer": {"type": "adamw", "params": {}}},
                )
            ]
        }
    )
    invalid_checks = {check.name: check for check in run_preflight_checks(invalid)}
    assert invalid_checks["schedule-realization"].status == "fail"
    assert "/params/learning_rate is required" in (
        invalid_checks["schedule-realization"].detail or ""
    )


def test_preflight_schedule_realization_fails_miswired_resume_before_driver(
    tmp_path: Path,
) -> None:
    declared_restart_context = _schedule_context(
        schedule_origin_step=12_000,
        current_step=12_000,
        optimizer_count_at_current_step=12_000,
    )
    bundle = _bundle(
        tmp_path,
        rows=[
            RunRowSpec(
                row_id="row-a",
                command=[sys.executable, "-c", "pass"],
                run_spec={
                    "optimizer": _scheduled_optimizer_payload(),
                    "resume_context": declared_restart_context,
                    "optimizer_build_context": _schedule_context(
                        schedule_origin_step=0,
                        current_step=0,
                        optimizer_count_at_current_step=0,
                    ),
                },
            )
        ],
    )
    driver = FakeDriver()

    with pytest.raises(PreflightFailed):
        StageEngine(bundle=bundle, driver=driver).run()

    assert driver.calls == []
    state = RunSetStateStore(bundle.run_set_dir / "state.json").load()
    checks = {check.name: check for check in state.stage(STAGE_PREFLIGHT).checks}
    schedule_check = checks["schedule-realization"]
    assert schedule_check.status == "fail"
    assert "learning-rate mismatch" in (schedule_check.detail or "")
    row_observed = schedule_check.observed["row-a"][0]
    assert row_observed["expected_context"] == declared_restart_context
    assert row_observed["observed_context"] == {
        "schedule_origin_step": 0,
        "current_step": 0,
        "optimizer_count_at_current_step": 0,
    }
    assert len(row_observed["samples"]) >= 4
    assert row_observed["mismatches"][0]["expected"] != row_observed["mismatches"][0]["observed"]


def test_preflight_schedule_realization_passes_correct_resume_context(tmp_path: Path) -> None:
    resume_context = _schedule_context(
        schedule_origin_step=12_000,
        current_step=12_000,
        optimizer_count_at_current_step=12_000,
    )
    bundle = _bundle(
        tmp_path,
        rows=[
            RunRowSpec(
                row_id="row-a",
                command=[sys.executable, "-c", "pass"],
                run_spec={
                    "optimizer": _scheduled_optimizer_payload(),
                    "resume_context": resume_context,
                },
            )
        ],
    )

    checks = {check.name: check for check in run_preflight_checks(bundle)}

    schedule_check = checks["schedule-realization"]
    assert schedule_check.status == "pass"
    row_observed = schedule_check.observed["row-a"][0]
    assert row_observed["scheduled"] is True
    assert row_observed["expected_context"] == resume_context
    assert row_observed["observed_context"] == resume_context
    assert len(row_observed["samples"]) >= 4


def test_preflight_schedule_realization_fails_when_resume_context_is_dropped(
    tmp_path: Path,
) -> None:
    bundle = _bundle(
        tmp_path,
        rows=[
            RunRowSpec(
                row_id="row-a",
                command=[sys.executable, "-c", "pass"],
                run_spec={
                    "optimizer": _scheduled_optimizer_payload(),
                    "optimizer_build_context": _schedule_context(
                        schedule_origin_step=0,
                        current_step=0,
                        optimizer_count_at_current_step=0,
                    ),
                },
            )
        ],
    )

    checks = {check.name: check for check in run_preflight_checks(bundle)}

    assert checks["schedule-realization"].status == "fail"
    assert "resume_context missing" in (checks["schedule-realization"].detail or "")


def test_schedule_preflight_and_conformance_share_schedule_eval_helper() -> None:
    assert (
        conformance.learning_rate_from_build_optimizer
        is schedule_eval.learning_rate_from_build_optimizer
    )
    assert conformance.extract_resume_context is schedule_eval.extract_resume_context
    assert stages.schedule_eval is schedule_eval


def test_local_driver_warm_first_max_parallel_budget_and_demo(tmp_path: Path) -> None:
    script = tmp_path / "row_script.py"
    script.write_text(
        """
from pathlib import Path
import os
import time
from feedbax.orchestration.events import RunEventEmitter

row = os.environ["FEEDBAX_ROW_ID"]
row_dir = Path(os.environ["FEEDBAX_ROW_DIR"])
with RunEventEmitter.from_env(heartbeat_seconds=None) as emitter:
    if row == "warm":
        emitter.emit("ready", {"row": row})
        time.sleep(0.15)
    else:
        time.sleep(0.02)
    (row_dir / "payload.json").write_text('{"row": "%s"}\\n' % row, encoding="utf-8")
    emitter.emit_terminal("complete", {"row": row})
""".strip(),
        encoding="utf-8",
    )
    rows = [
        RunRowSpec(row_id="warm", command=[sys.executable, str(script)], collect=["payload.json"]),
        RunRowSpec(
            row_id="second", command=[sys.executable, str(script)], collect=["payload.json"]
        ),
    ]
    bundle = _bundle(
        tmp_path,
        rows=rows,
        launch_policy=LaunchPolicy(max_parallel_rows=1, warm_first=True),
        run_set_id="2026-01-02-cafebabe",
    )

    state = StageEngine(
        bundle=bundle,
        driver=LocalOrchestrationDriver(cwd=tmp_path, freeze_lines=("feedbax==test",)),
        poll_interval_seconds=0.01,
    ).run()

    assert state.stage("REGISTER").status == "completed"
    assert state.registration_payload and state.registration_payload["status"] == "completed"
    assert (
        state.registration_payload["certificate_sha256"]
        == hashlib.sha256((bundle.run_set_dir / "conformance.json").read_bytes()).hexdigest()
    )
    assert {row_id: row.status for row_id, row in state.rows.items()} == {
        "warm": "completed",
        "second": "completed",
    }
    assert (bundle.run_set_dir / "events" / "warm.events.jsonl").exists()
    assert (bundle.run_set_dir / "collected" / "second" / "payload.json").exists()

    slow = tmp_path / "slow.py"
    slow.write_text("import time; time.sleep(2)\n", encoding="utf-8")
    budget_bundle = _bundle(
        tmp_path / "budget",
        rows=[RunRowSpec(row_id="slow", command=[sys.executable, str(slow)])],
        max_wall_clock_seconds=0.05,
        run_set_id="2026-01-02-badf00d",
    )
    budget_state = StageEngine(
        bundle=budget_bundle,
        driver=LocalOrchestrationDriver(cwd=tmp_path, freeze_lines=("feedbax==test",)),
        poll_interval_seconds=0.01,
    ).run()

    assert budget_state.abort_reason == "budget-exceeded"
    assert budget_state.rows["slow"].status == "stopped"
    assert budget_state.registration_payload
    assert budget_state.registration_payload["status"] == "aborted"


def test_local_monitor_requests_checkpoint_stop_and_records_provenance(tmp_path: Path) -> None:
    script = tmp_path / "interruptible_row.py"
    script.write_text(
        """
import signal
import time
from feedbax.orchestration.events import RunEventEmitter

emitter = RunEventEmitter.from_env(heartbeat_seconds=None)
assert emitter is not None

def stop_at_checkpoint(_signum, _frame):
    emitter.emit_terminal("complete", {"status": "cancelled"})
    emitter.close()
    raise SystemExit(0)

signal.signal(signal.SIGINT, stop_at_checkpoint)
emitter.emit("ready", {"phase": "train"})
emitter.emit_progress(
    {"phase": "train", "batch": 1, "total_batches": 10},
    batch=1,
    total_batches=10,
    force=True,
)
while True:
    time.sleep(0.01)
""".strip(),
        encoding="utf-8",
    )
    bundle = _bundle(
        tmp_path,
        rows=[RunRowSpec(row_id="row", command=[sys.executable, str(script)])],
        run_set_id="checkpoint-stop",
    )
    event_path = bundle.run_set_dir / "events" / "row.events.jsonl"
    decision = CancellationDecision("stop", "test", 123.0)
    dispatched = False

    def interruption_probe() -> CancellationDecision | None:
        nonlocal dispatched
        if not dispatched and event_path.exists() and '"type":"ready"' in event_path.read_text():
            dispatched = True
            return decision
        return None

    state = StageEngine(
        bundle=bundle,
        driver=LocalOrchestrationDriver(cwd=Path.cwd(), freeze_lines=("feedbax==test",)),
        poll_interval_seconds=0.01,
        interruption_probe=interruption_probe,
    ).run(stop_after_stage="MONITOR")

    assert dispatched
    assert state.abort_reason == "operator-stop-after-checkpoint"
    assert state.rows["row"].status == "stopped"
    assert state.budget_counters["cancellation"] == decision.as_provenance()


@pytest.mark.parametrize(
    ("action", "expected_abort_reason", "expected_row_status"),
    [
        ("continue", None, "completed"),
        ("terminate", "operator-terminate", "stopped"),
    ],
)
def test_local_monitor_applies_continue_and_terminate_decisions(
    tmp_path: Path,
    action: CancellationAction,
    expected_abort_reason: str | None,
    expected_row_status: str,
) -> None:
    script = tmp_path / "row.py"
    script.write_text(
        """
import time
from feedbax.orchestration.events import RunEventEmitter

with RunEventEmitter.from_env(heartbeat_seconds=None) as emitter:
    assert emitter is not None
    emitter.emit("ready", {"phase": "train"})
    emitter.emit_progress(
        {"phase": "train", "batch": 1, "total_batches": 1},
        batch=1,
        total_batches=1,
        force=True,
    )
    time.sleep(0.1)
    emitter.emit_terminal("complete", {"status": "completed"})
""".strip(),
        encoding="utf-8",
    )
    bundle = _bundle(
        tmp_path,
        rows=[RunRowSpec(row_id="row", command=[sys.executable, str(script)])],
        run_set_id=f"{action}-decision",
    )
    event_path = bundle.run_set_dir / "events" / "row.events.jsonl"
    decision = CancellationDecision(action, "test", 123.0)
    dispatched = False

    def interruption_probe() -> CancellationDecision | None:
        nonlocal dispatched
        if not dispatched and event_path.exists() and '"type":"ready"' in event_path.read_text():
            dispatched = True
            return decision
        return None

    state = StageEngine(
        bundle=bundle,
        driver=LocalOrchestrationDriver(cwd=Path.cwd(), freeze_lines=("feedbax==test",)),
        poll_interval_seconds=0.01,
        interruption_probe=interruption_probe,
    ).run(stop_after_stage="MONITOR")

    assert dispatched
    assert state.abort_reason == expected_abort_reason
    assert state.rows["row"].status == expected_row_status
    if action == "terminate":
        assert state.budget_counters["cancellation"] == decision.as_provenance()


def test_register_writes_failed_certificate_payload_and_reentry_is_idempotent(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    store = RunSetStateStore(bundle.run_set_dir / "state.json")
    registry = CheckRegistry(
        {
            "fixture_fail": lambda row: CheckEntry(
                check_id="fixture_fail",
                status="fail",
                expected="pass",
                observed="fail",
            )
        }
    )

    with pytest.raises(ValueError, match="phase=completed"):
        StageEngine(
            bundle=bundle,
            driver=FakeDriver(),
            store=store,
            conformance_registry=registry,
        ).run()

    register_path = bundle.run_set_dir / "registration.json"
    certificate_path = bundle.run_set_dir / "conformance.json"
    payload = json.loads(register_path.read_text(encoding="utf-8"))
    certificate_digest = hashlib.sha256(certificate_path.read_bytes()).hexdigest()

    assert payload == {
        "abort_reason": None,
        "certificate_overall": "fail",
        "certificate_ref": str(certificate_path),
        "certificate_sha256": certificate_digest,
        "failure_reason": "conformance-failed",
        "run_set_id": bundle.run_set_id,
        "status": "failed",
    }
    failed_state = store.load()
    assert failed_state.stage("REGISTER").status == "failed"
    assert failed_state.registration_payload == payload

    registration_mtime = register_path.stat().st_mtime_ns
    with pytest.raises(ValueError, match="phase=completed"):
        StageEngine(
            bundle=bundle,
            driver=FakeDriver(),
            store=store,
            conformance_registry=registry,
        ).run()
    assert register_path.stat().st_mtime_ns == registration_mtime

    tampered = dict(payload)
    tampered["status"] = "completed"
    register_path.write_text(
        json.dumps(tampered, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(
        OrchestrationStageError,
        match=r"registration payload mismatch at .*registration\.json.*conformance\.json",
    ):
        StageEngine(
            bundle=bundle,
            driver=FakeDriver(),
            store=store,
            conformance_registry=registry,
        ).run()


def test_local_driver_adopts_live_started_pid_without_spawning(tmp_path: Path) -> None:
    marker = tmp_path / "spawned.txt"
    row = RunRowSpec(
        row_id="row-a",
        command=[
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(marker)!r}).write_text('spawned')",
        ],
    )
    bundle = _bundle(tmp_path, rows=[row])
    driver = LocalOrchestrationDriver(cwd=tmp_path, freeze_lines=("feedbax==test",))
    driver.provision(bundle, RunSetState(run_set_id=bundle.run_set_id, rows={"row-a": RowState()}))
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
    try:
        sentinels = bundle.run_set_dir / "sentinels"
        (sentinels / "row-a.started").write_text("1\n", encoding="utf-8")
        (sentinels / "row-a.pid").write_text(f"{process.pid}\n", encoding="utf-8")

        outputs = driver.launch_row(
            bundle,
            row,
            RunSetState(run_set_id=bundle.run_set_id, rows={"row-a": RowState()}),
        )
    finally:
        process.terminate()
        process.wait(timeout=5)

    assert outputs["pid"] == process.pid
    assert outputs["adopted"] is True
    assert not marker.exists()


def test_local_driver_marks_dead_started_pid_failed_without_spawning(tmp_path: Path) -> None:
    marker = tmp_path / "spawned.txt"
    row = RunRowSpec(
        row_id="row-a",
        command=[
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(marker)!r}).write_text('spawned')",
        ],
    )
    bundle = _bundle(tmp_path, rows=[row])
    driver = LocalOrchestrationDriver(cwd=tmp_path, freeze_lines=("feedbax==test",))
    driver.provision(bundle, RunSetState(run_set_id=bundle.run_set_id, rows={"row-a": RowState()}))
    sentinels = bundle.run_set_dir / "sentinels"
    (sentinels / "row-a.started").write_text("1\n", encoding="utf-8")
    (sentinels / "row-a.pid").write_text("999999999\n", encoding="utf-8")

    outputs = driver.launch_row(
        bundle,
        row,
        RunSetState(run_set_id=bundle.run_set_id, rows={"row-a": RowState()}),
    )

    assert outputs["status"] == "failed"
    assert outputs["event_discrepancies"][0]["code"] == "orphaned_launch"
    assert "orphaned launch" in (sentinels / "row-a.failed").read_text(encoding="utf-8")
    assert not marker.exists()


def test_stage_resume_records_orphaned_started_pid_as_failed(tmp_path: Path) -> None:
    marker = tmp_path / "spawned.txt"
    row = RunRowSpec(
        row_id="row-a",
        command=[
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(marker)!r}).write_text('spawned')",
        ],
    )
    bundle = _bundle(tmp_path, rows=[row])
    store = RunSetStateStore(bundle.run_set_dir / "state.json")
    driver = LocalOrchestrationDriver(cwd=tmp_path, freeze_lines=("feedbax==test",))
    state = RunSetState(run_set_id=bundle.run_set_id, rows={"row-a": RowState()})
    driver.provision(bundle, state)
    sentinels = bundle.run_set_dir / "sentinels"
    (sentinels / "row-a.started").write_text("1\n", encoding="utf-8")
    (sentinels / "row-a.pid").write_text("999999999\n", encoding="utf-8")

    final_state = StageEngine(
        bundle=bundle,
        driver=driver,
        store=store,
        poll_interval_seconds=0.01,
    ).run()

    assert final_state.rows["row-a"].status == "failed"
    assert final_state.rows["row-a"].event_discrepancies[0]["code"] == "orphaned_launch"
    assert not marker.exists()


def test_fingerprint_stability_package_changes_and_dirty_policy(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    (repo / "dirty.txt").write_text("dirty\n", encoding="utf-8")
    bundle = _bundle(
        tmp_path,
        rows=[RunRowSpec(row_id="row", command=[sys.executable, "-c", "pass"])],
    ).model_copy(
        update={
            "environment": EnvironmentDeclaration(
                python_version="3.12",
                repo_revisions=[RepoRevision(path=".", revision="HEAD", dirty_allowed=True)],
                image_id="local",
            )
        }
    )

    first = compute_environment_fingerprint(bundle, cwd=repo, freeze_lines=("a==1", "b==2"))
    second = compute_environment_fingerprint(bundle, cwd=repo, freeze_lines=("b==2", "a==1"))
    changed = compute_environment_fingerprint(bundle, cwd=repo, freeze_lines=("a==1", "b==3"))

    assert first == second
    assert first != changed

    disallow_dirty = bundle.model_copy(
        update={
            "environment": EnvironmentDeclaration(
                python_version="3.12",
                repo_revisions=[RepoRevision(path=".", revision="HEAD", dirty_allowed=False)],
            )
        }
    )
    with pytest.raises(LocalDriverError, match="dirty repo not allowed"):
        compute_environment_fingerprint(disallow_dirty, cwd=repo, freeze_lines=("a==1",))
