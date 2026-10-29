"""Output custody survives checkpoint stops and collection failures."""

from pathlib import Path
from datetime import datetime, timezone

import pytest

from feedbax.orchestration.drivers.local import LocalDriverError, LocalOrchestrationDriver
from feedbax.orchestration.drivers.base import DriverRowProbe
from feedbax.orchestration.drivers.native_execution import NATIVE_TRAINING_COLLECTION_OUTPUTS
from feedbax.orchestration.stages import (
    BudgetExceeded,
    CustodyPreservationRequired,
    OrchestrationStageError,
    StageEngine,
)
from feedbax.orchestration.state import RowState, RunSetState, StageState, utc_now
from feedbax.training import (
    CheckpointCustodyError,
    authenticate_published_checkpoint_custody,
    execute_training_run_spec,
)
from feedbax.training.interruption import CancellationDecision
from tests.test_orchestration_core import (
    CapabilityFakeDriver,
    _bundle,
    _compiled_row,
    _fixture_pass_registry,
)
from tests.test_training_run_executor import (
    _chunked_registry,
    _execution_context,
    _initial_slots,
    _run_spec,
)


@pytest.mark.parametrize("collection_fails", [False, True])
def test_native_checkpoint_stop_collects_authenticates_and_reclaims_after_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, collection_fails: bool
) -> None:
    bundle = _bundle(
        tmp_path,
        rows=[
            _compiled_row(
                "row-a",
                command=["python", "-m", "feedbax", "execute-training-run-spec"],
                collect=list(NATIVE_TRAINING_COLLECTION_OUTPUTS),
            )
        ],
    )
    row = bundle.rows[0]
    row = row.model_copy(
        update={
            "launch": row.launch.model_copy(
                update={
                    "payload_routing": {"kind": "registered-execution-payload"},
                }
            )
        }
    )
    bundle = bundle.model_copy(update={"rows": [row]})
    row_root = bundle.run_set_dir / "rows" / row.row_id
    registry, _program = _chunked_registry(stop_after_program_step=3)
    result = execute_training_run_spec(
        _run_spec(),
        initial_slots=_initial_slots(arrays=True),
        manifest_root=row_root / "manifests",
        checkpoint_root=row_root / "checkpoints",
        registry=registry,
        execution_context=_execution_context(collection_root=row_root),
        cancellation_probe=lambda coordinate: (
            CancellationDecision("stop", "test", 123.0) if coordinate.program_step == 1 else None
        ),
    )
    assert result.status == "cancelled"
    source = authenticate_published_checkpoint_custody(row_root / "checkpoints")
    driver = LocalOrchestrationDriver(cwd=tmp_path, freeze_lines=())
    engine = StageEngine(bundle=bundle, driver=driver)
    state = RunSetState(
        run_set_id=bundle.run_set_id,
        rows={row.row_id: RowState(status="stopped", started_at=utc_now())},
    )
    engine.store.preflight_and_reserve()
    engine.store.save(state)
    if collection_fails:
        original = driver.collect

        def fail_collection(*_args):
            raise OSError("injected collection failure")

        monkeypatch.setattr(driver, "collect", fail_collection)
        with pytest.raises(OSError, match="injected collection failure"):
            engine._run_stage("COLLECT", state)
        state = engine.store.load()
        assert state.stage("COLLECT").status == "failed"
        assert driver.teardown(bundle, state)["stopped_row_reclamation"] == []
        assert (
            authenticate_published_checkpoint_custody(row_root / "checkpoints").manifest_sha256
            == source.manifest_sha256
        )
        monkeypatch.setattr(driver, "collect", original)

    state = engine._run_stage("COLLECT", state)
    outputs = state.rows[row.row_id].collected_outputs
    assert set(outputs) == set(NATIVE_TRAINING_COLLECTION_OUTPUTS)
    assert Path(outputs["manifests"]).is_dir()
    destination = authenticate_published_checkpoint_custody(Path(outputs["checkpoints"]))
    assert destination.manifest_sha256 == source.manifest_sha256
    assert (
        state.stage("COLLECT").outputs["checkpoint_custody"][row.row_id]["manifest_sha256"]
        == source.manifest_sha256
    )
    recorded = state.stage("COLLECT").outputs
    drifted = state.with_stage(
        "COLLECT",
        StageState(
            status="completed",
            outputs={
                **recorded,
                "checkpoint_custody": {},
            },
        ),
    )
    with pytest.raises(LocalDriverError, match="checkpoint custody drifted"):
        driver.teardown(bundle, drifted)
    assert row_root.exists()
    latest = Path(outputs["checkpoints"]) / "latest.json"
    latest_bytes = latest.read_bytes()
    latest.write_bytes(b"{}\n")
    with pytest.raises(CheckpointCustodyError):
        driver.teardown(bundle, state)
    assert row_root.exists()
    latest.write_bytes(latest_bytes)
    assert driver.teardown(bundle, state)["stopped_row_reclamation"][0]["status"] == "reclaimed"
    assert not row_root.exists()
    assert (
        authenticate_published_checkpoint_custody(Path(outputs["checkpoints"])).manifest_sha256
        == source.manifest_sha256
    )


def test_ephemeral_collection_failure_blocks_delete_and_retry_releases(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, driver="runpod")
    driver = CapabilityFakeDriver("engine-acquired")
    driver.fail["collect:row-a"] = 5
    engine = StageEngine(
        bundle=bundle, driver=driver, conformance_registry=_fixture_pass_registry()
    )
    with pytest.raises(CustodyPreservationRequired, match="lack complete collection custody"):
        engine.run()
    assert driver.delete_calls == 0
    assert engine.store.load().stage("TEARDOWN").status == "failed"
    assert engine.store.load().stage("COLLECT").status == "failed"

    state = engine.run(stop_after_stage="TEARDOWN")
    assert driver.delete_calls == 1
    assert state.stage("TEARDOWN").status == "completed"
    assert state.stage("COLLECT").status == "completed"


def test_ephemeral_teardown_without_execution_allows_cleanup(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, driver="runpod")
    driver = CapabilityFakeDriver("engine-acquired")
    engine = StageEngine(bundle=bundle, driver=driver)
    engine.store.preflight_and_reserve()
    state = RunSetState(
        run_set_id=bundle.run_set_id,
        rows={"row-a": RowState(status="stopped", error="pre-launch abort")},
        stages={"PROVISION": StageState(status="completed")},
    )
    state = engine._run_teardown(state, abort=True)
    assert state.stage("TEARDOWN").status == "completed"
    assert driver.delete_calls == 1


@pytest.mark.parametrize("collection_outcome", ["complete", "error", "missing"])
def test_ephemeral_failed_executor_releases_only_after_output_custody(
    tmp_path: Path, collection_outcome: str
) -> None:
    class FailedExecutorDriver(CapabilityFakeDriver):
        def probe(self, bundle, row, state):
            self._call(f"probe:{row.row_id}")
            return DriverRowProbe(status="failed", detail="fixture executor failure")

        def collect(self, bundle, row, state):
            self._call(f"collect:{row.row_id}")
            if collection_outcome == "error":
                raise OSError("failure output pull failed")
            if collection_outcome == "missing":
                return {}
            destination = bundle.run_set_dir / "collected" / row.row_id / "failure.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text('{"error":"fixture executor failure"}\n')
            return {"failure.json": str(destination)}

    bundle = _bundle(
        tmp_path, driver="runpod", rows=[_compiled_row("row-a", collect=["failure.json"])]
    )
    driver = FailedExecutorDriver("engine-acquired")
    engine = StageEngine(bundle=bundle, driver=driver)
    expected_error = (
        OrchestrationStageError
        if collection_outcome == "complete"
        else (CustodyPreservationRequired)
    )
    with pytest.raises(expected_error):
        engine.run()
    state = engine.store.load()
    assert state.rows["row-a"].status == "failed"
    assert state.stage("COLLECT").status == "failed"
    assert state.stage("COLLECT").outputs["executor_failures"] == [
        {"row_id": "row-a", "error": "fixture executor failure"}
    ]
    if collection_outcome == "complete":
        outputs = state.rows["row-a"].collected_outputs
        assert Path(outputs["failure.json"]).read_text() == (
            '{"error":"fixture executor failure"}\n'
        )
        assert state.stage("COLLECT").outputs["rows"]["row-a"] == outputs
        assert state.stage("TEARDOWN").status == "completed"
        assert driver.delete_calls == 1
    else:
        assert state.stage("TEARDOWN").status == "failed"
        assert driver.delete_calls == 0


def test_ephemeral_spend_rejected_before_dispatch_still_releases_resource(tmp_path: Path) -> None:
    observed_time = 0.0

    class SpentDriver(CapabilityFakeDriver):
        def provision(self, bundle, state):
            return {
                **super().provision(bundle, state),
                "billing_started_at": datetime.fromtimestamp(0, tz=timezone.utc).isoformat(),
                "hourly_rate": 3600.0,
                "currency": "USD",
            }

        def stage_inputs(self, bundle, state):
            nonlocal observed_time
            outputs = super().stage_inputs(bundle, state)
            observed_time = 1.0
            return outputs

    bundle = _bundle(tmp_path, driver="runpod", max_spend_usd=0.5)
    driver = SpentDriver("engine-acquired")
    engine = StageEngine(bundle=bundle, driver=driver, wall_time=lambda: observed_time)
    with pytest.raises(BudgetExceeded, match="reached before LAUNCH"):
        engine.run()
    state = engine.store.load()
    assert state.stage("LAUNCH").started_at is not None
    assert all(row.status == "pending" and row.started_at is None for row in state.rows.values())
    assert not any(call.startswith("launch:") for call in driver.calls)
    assert state.stage("TEARDOWN").status == "completed"
    assert driver.delete_calls == 1


def test_ephemeral_uncertain_dispatch_preserves_resource(tmp_path: Path) -> None:
    class UncertainDispatchDriver(CapabilityFakeDriver):
        def launch_row(self, bundle, row, state):
            super().launch_row(bundle, row, state)
            raise OSError("dispatch acknowledgement lost")

    bundle = _bundle(tmp_path, driver="runpod")
    driver = UncertainDispatchDriver("engine-acquired")
    engine = StageEngine(bundle=bundle, driver=driver)
    with pytest.raises(CustodyPreservationRequired):
        engine.run()
    state = engine.store.load()
    assert driver.calls.count("launch:row-a") == 1
    assert state.rows["row-a"].started_at is not None
    assert state.stage("TEARDOWN").status == "failed"
    assert driver.delete_calls == 0
