"""Synchronous run-set stage engine."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from feedbax.orchestration.bundle import RunBundle, RunRowSpec
from feedbax.orchestration.conformance import (
    CheckRegistry,
    ConformanceRowArtifacts,
    RunConformanceCertificate,
    assert_certificate_allows_completed_registration,
    write_conformance_certificate,
)
from feedbax.orchestration.drivers.base import OrchestrationDriver
from feedbax.orchestration.events import RunEventReader
from feedbax.orchestration import schedule_eval
from feedbax.orchestration.state import (
    PreflightCheckEntry,
    RowState,
    RunSetState,
    RunSetStateStore,
    StageState,
    utc_now,
)
from feedbax.training.manifest_preflight import preflight_training_run_manifest_payloads
from feedbax.training.interruption import CancellationDecision


STAGE_ASSEMBLE = "ASSEMBLE"
STAGE_PREFLIGHT = "PREFLIGHT"
STAGE_PROVISION = "PROVISION"
STAGE_REALIZE_ENV = "REALIZE_ENV"
STAGE_STAGE_INPUTS = "STAGE_INPUTS"
STAGE_LAUNCH = "LAUNCH"
STAGE_MONITOR = "MONITOR"
STAGE_COLLECT = "COLLECT"
STAGE_CERTIFY = "CERTIFY"
STAGE_TEARDOWN = "TEARDOWN"
STAGE_REGISTER = "REGISTER"
STAGE_ORDER = (
    STAGE_ASSEMBLE,
    STAGE_PREFLIGHT,
    STAGE_PROVISION,
    STAGE_REALIZE_ENV,
    STAGE_STAGE_INPUTS,
    STAGE_LAUNCH,
    STAGE_MONITOR,
    STAGE_COLLECT,
    STAGE_CERTIFY,
    STAGE_TEARDOWN,
    STAGE_REGISTER,
)

RETRY_LIMITS = {
    STAGE_PROVISION: 3,
    STAGE_REALIZE_ENV: 3,
    STAGE_STAGE_INPUTS: 3,
    STAGE_COLLECT: 5,
}


class OrchestrationStageError(RuntimeError):
    """Raised when a run-set stage fails."""


class PreflightFailed(OrchestrationStageError):
    """Raised when one or more named preflight checks fail."""


class BudgetExceeded(OrchestrationStageError):
    """Raised when a run-set budget guard aborts monitoring."""


class StageEngine:
    """Execute a run bundle through the orchestration stage sequence."""

    def __init__(
        self,
        *,
        bundle: RunBundle,
        driver: OrchestrationDriver,
        store: RunSetStateStore | None = None,
        conformance_registry: CheckRegistry | None = None,
        poll_interval_seconds: float = 0.05,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        wall_time: Callable[[], float] = time.time,
        interruption_probe: Callable[[], CancellationDecision | None] | None = None,
    ) -> None:
        self.bundle = bundle
        self.driver = driver
        self.store = store or RunSetStateStore(bundle.run_set_dir / "state.json")
        self.conformance_registry = conformance_registry or CheckRegistry()
        self.poll_interval_seconds = poll_interval_seconds
        self._sleep = sleep
        self._monotonic = monotonic
        self._wall_time = wall_time
        self._interruption_probe = interruption_probe

    def run(
        self,
        *,
        break_stale_lock: bool = False,
        stop_after_stage: str | None = None,
    ) -> RunSetState:
        """Run or resume the bundle through all stages."""
        initial = self._initial_state()
        with self.store.lock(break_stale=break_stale_lock):
            state = self.store.initialize(initial)
            try:
                for stage_id in STAGE_ORDER:
                    if state.stage(stage_id).status == "completed":
                        continue
                    state = self._run_stage(stage_id, state)
                    if stop_after_stage == stage_id:
                        return state
                return state
            except Exception:
                latest = self.store.load() if self.store.path.exists() else state
                if self._provision_completed(latest) and not self.bundle.keep_alive:
                    latest = self._run_teardown(latest, abort=True)
                raise

    def _initial_state(self) -> RunSetState:
        return RunSetState(
            run_set_id=self.bundle.run_set_id,
            rows={row.row_id: RowState() for row in self.bundle.rows},
            stages={stage_id: StageState() for stage_id in STAGE_ORDER},
        )

    def _run_stage(self, stage_id: str, state: RunSetState) -> RunSetState:
        limit = RETRY_LIMITS.get(stage_id, 1)
        while True:
            stage = state.stage(stage_id)
            attempts = stage.attempts + 1
            started = stage.model_copy(
                update={
                    "status": "running",
                    "attempts": attempts,
                    "started_at": stage.started_at or utc_now(),
                    "error": None,
                }
            )
            state = state.with_stage(stage_id, started)
            self.store.save(state)
            try:
                handler = getattr(self, f"_stage_{stage_id.lower()}")
                state, outputs = handler(state)
            except Exception as exc:
                if self.store.path.exists():
                    state = self.store.load()
                failed = state.stage(stage_id).model_copy(
                    update={"status": "failed", "error": str(exc)}
                )
                state = state.with_stage(stage_id, failed)
                self.store.save(state)
                if attempts >= limit:
                    raise
                continue
            completed = state.stage(stage_id).model_copy(
                update={
                    "status": "completed",
                    "completed_at": utc_now(),
                    "outputs": dict(outputs),
                }
            )
            state = state.with_stage(stage_id, completed)
            self.store.save(state)
            return state

    def _stage_assemble(self, state: RunSetState) -> tuple[RunSetState, Mapping[str, Any]]:
        run_set_dir = self.bundle.run_set_dir
        run_set_dir.mkdir(parents=True, exist_ok=True)
        payload = self.bundle.model_dump(mode="json", exclude_none=True)
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        bundle_path = run_set_dir / "bundle.json"
        bundle_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return state, {
            "bundle_path": str(bundle_path),
            "bundle_sha256": hashlib.sha256(encoded).hexdigest(),
        }

    def _stage_preflight(self, state: RunSetState) -> tuple[RunSetState, Mapping[str, Any]]:
        checks = run_preflight_checks(self.bundle)
        stage = state.stage(STAGE_PREFLIGHT).model_copy(update={"checks": checks})
        state = state.with_stage(STAGE_PREFLIGHT, stage)
        self.store.save(state)
        failed = [check for check in checks if check.status == "fail"]
        if failed:
            names = ", ".join(check.name for check in failed)
            raise PreflightFailed(f"preflight failed: {names}")
        return state, {"checks": [check.model_dump(mode="json") for check in checks]}

    def _stage_provision(self, state: RunSetState) -> tuple[RunSetState, Mapping[str, Any]]:
        outputs = dict(self.driver.provision(self.bundle, state))
        state = state.model_copy(update={"provision_record": outputs, "updated_at": utc_now()})
        return state, outputs

    def _stage_realize_env(self, state: RunSetState) -> tuple[RunSetState, Mapping[str, Any]]:
        fingerprint = self.driver.realize_env(self.bundle, state)
        state = state.model_copy(
            update={"environment_fingerprint": fingerprint, "updated_at": utc_now()}
        )
        return state, {"environment_fingerprint": fingerprint}

    def _stage_stage_inputs(self, state: RunSetState) -> tuple[RunSetState, Mapping[str, Any]]:
        return state, dict(self.driver.stage_inputs(self.bundle, state))

    def _stage_launch(self, state: RunSetState) -> tuple[RunSetState, Mapping[str, Any]]:
        launched: list[str] = []
        for row in self._launchable_rows(state):
            state = self._launch_one(row, state)
            launched.append(row.row_id)
            if self.bundle.launch_policy.warm_first:
                break
            if len(launched) >= self.bundle.launch_policy.max_parallel_rows:
                break
        return state, {"launched_rows": launched}

    def _stage_monitor(self, state: RunSetState) -> tuple[RunSetState, Mapping[str, Any]]:
        started_at = self._wall_time()
        counters = dict(state.budget_counters)
        counters.setdefault("monitor_started_at", started_at)
        state = state.model_copy(update={"budget_counters": counters, "updated_at": utc_now()})
        self.store.save(state)
        budget_exceeded = False
        while True:
            decision = self._interruption_probe() if self._interruption_probe is not None else None
            if decision is not None and decision.action != "continue":
                state = self._apply_interruption(state, decision)
            state = self._refresh_rows(state)
            if self._all_terminal(state):
                break
            if self._wall_time() - started_at > self.bundle.budget.max_wall_clock_seconds:
                budget_exceeded = True
                state = self._stop_unfinished(state, reason="budget-exceeded")
                break
            if state.abort_reason != "operator-stop-after-checkpoint":
                state = self._launch_pending_if_allowed(state)
            self.store.save(state)
            if self._all_terminal(state):
                break
            self._sleep(self.poll_interval_seconds)
        state = self._refresh_rows(state)
        counters = dict(state.budget_counters)
        counters["wall_clock_seconds"] = max(0.0, self._wall_time() - started_at)
        if budget_exceeded:
            counters["budget_exceeded"] = "wall-clock"
            state = state.model_copy(
                update={
                    "budget_counters": counters,
                    "abort_reason": "budget-exceeded",
                    "updated_at": utc_now(),
                }
            )
            return state, counters
        state = state.model_copy(update={"budget_counters": counters, "updated_at": utc_now()})
        return state, counters

    def _stage_collect(self, state: RunSetState) -> tuple[RunSetState, Mapping[str, Any]]:
        collected: dict[str, Mapping[str, str]] = {}
        for row in self.bundle.rows:
            outputs = dict(self.driver.collect(self.bundle, row, state))
            collected[row.row_id] = outputs
            row_state = state.rows[row.row_id].model_copy(update={"collected_outputs": outputs})
            state = state.with_row(row.row_id, row_state)
        return state, {"rows": collected}

    def _stage_certify(self, state: RunSetState) -> tuple[RunSetState, Mapping[str, Any]]:
        rows = [
            ConformanceRowArtifacts(
                row_id=row.row_id,
                event_log=self.bundle.run_set_dir / "events" / f"{row.row_id}.events.jsonl",
                bundle_row_spec=row.model_dump(mode="json", exclude_none=True),
                recorded_environment_fingerprint=state.environment_fingerprint,
            )
            for row in self.bundle.rows
        ]
        certificate = write_conformance_certificate(
            run_set_dir=self.bundle.run_set_dir,
            run_set_id=self.bundle.run_set_id,
            rows=rows,
            registry=self.conformance_registry,
        )
        certificate_path = self.bundle.run_set_dir / "conformance.json"
        state = state.model_copy(
            update={"certificate_ref": str(certificate_path), "updated_at": utc_now()}
        )
        return state, {"certificate_ref": str(certificate_path), "overall": certificate.overall}

    def _stage_teardown(self, state: RunSetState) -> tuple[RunSetState, Mapping[str, Any]]:
        return self._run_teardown(state, abort=False), {}

    def _stage_register(self, state: RunSetState) -> tuple[RunSetState, Mapping[str, Any]]:
        certificate_path = Path(state.certificate_ref or "")
        certificate_bytes = certificate_path.read_bytes()
        certificate_payload = json.loads(certificate_bytes.decode("utf-8"))
        certificate = RunConformanceCertificate.model_validate(certificate_payload)
        certificate_digest = hashlib.sha256(certificate_bytes).hexdigest()
        if certificate.overall == "pass":
            status = "aborted" if state.abort_reason else "completed"
        else:
            status = "failed"
        payload = {
            "run_set_id": self.bundle.run_set_id,
            "status": status,
            "abort_reason": state.abort_reason,
            "certificate_ref": str(certificate_path),
            "certificate_sha256": certificate_digest,
            "certificate_overall": certificate.overall,
        }
        if status == "failed":
            payload["failure_reason"] = "conformance-failed"
        state = state.model_copy(update={"registration_payload": payload, "updated_at": utc_now()})
        register_path = self.bundle.run_set_dir / "registration.json"
        self._write_or_verify_registration(
            register_path=register_path,
            certificate_path=certificate_path,
            payload=payload,
        )
        self.store.save(state)
        assert_certificate_allows_completed_registration(certificate_payload)
        return state, payload

    def _write_or_verify_registration(
        self,
        *,
        register_path: Path,
        certificate_path: Path,
        payload: Mapping[str, Any],
    ) -> None:
        if register_path.exists():
            existing = json.loads(register_path.read_text(encoding="utf-8"))
            if existing == dict(payload):
                return
            raise OrchestrationStageError(
                "registration payload mismatch at "
                f"{register_path}; existing payload does not match current certificate "
                f"outcome from {certificate_path}: status={payload.get('status')!r}, "
                f"certificate_overall={payload.get('certificate_overall')!r}"
            )

        register_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{register_path.name}.",
            suffix=".tmp",
            dir=str(register_path.parent),
            text=True,
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(dict(payload), handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, register_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def _run_teardown(self, state: RunSetState, *, abort: bool) -> RunSetState:
        if self.bundle.keep_alive:
            return state
        stage = state.stage(STAGE_TEARDOWN)
        if stage.status == "completed":
            return state
        try:
            outputs = dict(self.driver.teardown(self.bundle, state))
            status = "completed"
            error = None
        except Exception as exc:
            outputs = {}
            status = "failed"
            error = str(exc)
        updated = stage.model_copy(
            update={
                "status": status,
                "attempts": stage.attempts + 1,
                "started_at": stage.started_at or utc_now(),
                "completed_at": utc_now(),
                "outputs": {**outputs, "abort_path": abort},
                "error": error,
            }
        )
        state = state.with_stage(STAGE_TEARDOWN, updated)
        self.store.save(state)
        return state

    def _provision_completed(self, state: RunSetState) -> bool:
        return state.stage(STAGE_PROVISION).status == "completed"

    def _launchable_rows(self, state: RunSetState) -> list[RunRowSpec]:
        return [row for row in self.bundle.rows if state.rows[row.row_id].status == "pending"]

    def _launch_one(self, row: RunRowSpec, state: RunSetState) -> RunSetState:
        outputs = self.driver.launch_row(self.bundle, row, state)
        output_status = outputs.get("status")
        status = "failed" if output_status == "failed" else "launched"
        row_state = state.rows[row.row_id].model_copy(
            update={
                "status": status,
                "pid": outputs.get("pid"),
                "started_at": utc_now(),
                "completed_at": utc_now() if status == "failed" else None,
                "error": outputs.get("detail") if status == "failed" else None,
                "event_discrepancies": [
                    dict(item) for item in outputs.get("event_discrepancies", [])
                ],
            }
        )
        state = state.with_row(row.row_id, row_state)
        self.store.save(state)
        return state

    def _launch_pending_if_allowed(self, state: RunSetState) -> RunSetState:
        if self.bundle.launch_policy.warm_first:
            first_id = self.bundle.rows[0].row_id
            first_status = state.rows[first_id].status
            if first_status == "failed":
                return self._stop_unfinished(state, reason="warm-first-failed")
            if first_status not in ("ready", "completed"):
                return state
        active = sum(
            1
            for row_state in state.rows.values()
            if row_state.status in ("launched", "ready", "running")
        )
        slots = self.bundle.launch_policy.max_parallel_rows - active
        if slots <= 0:
            return state
        for row in self._launchable_rows(state)[:slots]:
            state = self._launch_one(row, state)
            if self.bundle.launch_policy.stagger_seconds > 0:
                self._sleep(self.bundle.launch_policy.stagger_seconds)
        return state

    def _refresh_rows(self, state: RunSetState) -> RunSetState:
        for row in self.bundle.rows:
            row_state = state.rows[row.row_id]
            if row_state.status in ("completed", "failed", "stopped"):
                continue
            probe = self.driver.probe(self.bundle, row, state)
            event_path = self.bundle.run_set_dir / "events" / f"{row.row_id}.events.jsonl"
            done = self.bundle.run_set_dir / "sentinels" / f"{row.row_id}.done"
            failed = self.bundle.run_set_dir / "sentinels" / f"{row.row_id}.failed"
            events = RunEventReader(event_path).read_all()
            high_water = events[-1].seq if events else row_state.event_seq_high_water_mark
            last_type = events[-1].type if events else row_state.last_event_type
            ready = any(event.type == "ready" for event in events)
            reconciled = RunEventReader(event_path).reconcile_sentinels(
                done_sentinel=done,
                failed_sentinel=failed,
            )
            status = row_state.status
            completed_at = row_state.completed_at
            error = row_state.error
            if reconciled.status == "completed" or probe.status == "completed":
                terminal_status = (
                    reconciled.terminal_event.payload.get("status")
                    if reconciled.terminal_event is not None
                    else None
                )
                status = "stopped" if terminal_status == "cancelled" else "completed"
                completed_at = completed_at or utc_now()
                if terminal_status == "cancelled":
                    error = "operator-stop-after-checkpoint"
            elif reconciled.status in ("failed", "error") or probe.status == "failed":
                status = "failed"
                completed_at = completed_at or utc_now()
                error = probe.detail or error
            elif ready:
                status = "ready"
            elif probe.status == "running" and status in ("pending", "launched"):
                status = "running"
            updated = row_state.model_copy(
                update={
                    "status": status,
                    "pid": probe.pid or row_state.pid,
                    "event_seq_high_water_mark": high_water,
                    "last_event_type": last_type,
                    "event_discrepancies": [dict(item) for item in reconciled.discrepancies],
                    "completed_at": completed_at,
                    "error": error,
                }
            )
            state = state.with_row(row.row_id, updated)
        self.store.save(state)
        return state

    def _stop_unfinished(self, state: RunSetState, *, reason: str) -> RunSetState:
        for row in self.bundle.rows:
            row_state = state.rows[row.row_id]
            if row_state.status in ("completed", "failed", "stopped"):
                continue
            self.driver.stop_row(self.bundle, row, state)
            updated = row_state.model_copy(
                update={"status": "stopped", "completed_at": utc_now(), "error": reason}
            )
            state = state.with_row(row.row_id, updated)
        return state.model_copy(update={"abort_reason": reason, "updated_at": utc_now()})

    def _apply_interruption(
        self,
        state: RunSetState,
        decision: CancellationDecision,
    ) -> RunSetState:
        """Persist and propagate one monitor-level operator decision."""
        provenance = decision.as_provenance()
        counters = dict(state.budget_counters)
        counters["cancellation"] = provenance
        if decision.action == "terminate":
            stopped = self._stop_unfinished(state, reason="operator-terminate")
            return stopped.model_copy(update={"budget_counters": counters, "updated_at": utc_now()})

        assert decision.action == "stop"
        for row in self.bundle.rows:
            row_state = state.rows[row.row_id]
            if row_state.status in ("completed", "failed", "stopped"):
                continue
            if row_state.status == "pending":
                state = state.with_row(
                    row.row_id,
                    row_state.model_copy(
                        update={
                            "status": "stopped",
                            "completed_at": utc_now(),
                            "error": "operator-stop-after-checkpoint",
                        }
                    ),
                )
                continue
            request_stop = getattr(self.driver, "request_stop_at_checkpoint", None)
            if callable(request_stop):
                request_stop(self.bundle, row, state)
            else:
                self.driver.stop_row(self.bundle, row, state)
                state = state.with_row(
                    row.row_id,
                    row_state.model_copy(
                        update={
                            "status": "stopped",
                            "completed_at": utc_now(),
                            "error": "operator-stop-after-checkpoint",
                        }
                    ),
                )
        return state.model_copy(
            update={
                "abort_reason": "operator-stop-after-checkpoint",
                "budget_counters": counters,
                "updated_at": utc_now(),
            }
        )

    def _all_terminal(self, state: RunSetState) -> bool:
        return all(row.status in ("completed", "failed", "stopped") for row in state.rows.values())


def run_preflight_checks(bundle: RunBundle) -> list[PreflightCheckEntry]:
    """Run static preflight checks without driver calls or resource mutation."""
    checks: list[PreflightCheckEntry] = []
    checks.append(
        _check("schema-current", bundle.schema_version == "feedbax.orchestration.run_bundle.v1")
    )
    checks.append(_check("row-identity", True, observed=[row.row_id for row in bundle.rows]))
    checks.append(_check("budget-presence", bundle.budget.max_wall_clock_seconds > 0))
    checks.append(
        _check(
            "driver-preconditions",
            bundle.driver in {"local", "worker-http"},
            observed=bundle.driver,
        )
    )
    env_complete = bool(bundle.environment.python_version)
    checks.append(
        _check("environment-declaration", env_complete, observed=bundle.environment.python_version)
    )
    mutable_pin = any(
        "latest.json" in pin.checkpoint_transaction_id for pin in bundle.input_custody_pins
    )
    checks.append(_check("custody-pins", not mutable_pin))

    manifest_failures: list[str] = []
    normalized: dict[str, Any] = {}
    for row in bundle.rows:
        if isinstance(row.run_spec, Mapping):
            try:
                payloads = preflight_training_run_manifest_payloads(
                    row.run_spec,
                    row_id=row.row_id,
                )
                normalized[row.row_id] = payloads.model_dump()
            except Exception as exc:
                manifest_failures.append(f"{row.row_id}: {exc}")
    checks.append(
        _check(
            "manifest-payload-normalization",
            not manifest_failures,
            detail="; ".join(manifest_failures) if manifest_failures else None,
            observed=normalized or "no-inline-run-specs",
        )
    )
    schedule_failures, schedule_observed = _preflight_schedule_realization(bundle)
    checks.append(
        _check(
            "schedule-realization",
            not schedule_failures,
            detail="; ".join(schedule_failures) if schedule_failures else None,
            observed=schedule_observed or "no-inline-optimizer-specs",
        )
    )
    return checks


def _preflight_schedule_realization(bundle: RunBundle) -> tuple[list[str], dict[str, Any]]:
    failures: list[str] = []
    observed: dict[str, Any] = {}
    for row in bundle.rows:
        if not isinstance(row.run_spec, Mapping):
            continue
        optimizer_payloads = _optimizer_payloads(row.run_spec)
        if not optimizer_payloads:
            continue
        row_observed: list[dict[str, Any]] = []
        for index, payload in enumerate(optimizer_payloads):
            try:
                result = _evaluate_optimizer_schedule_at_preflight(row, index, payload)
            except Exception as exc:
                failures.append(f"{row.row_id}[{index}]: {exc}")
            else:
                row_observed.append(result)
                mismatches = result.get("mismatches")
                if mismatches:
                    failures.append(
                        f"{row.row_id}[{index}]: learning-rate mismatch: {mismatches!r}"
                    )
        observed[row.row_id] = row_observed
    return failures, observed


def _evaluate_optimizer_schedule_at_preflight(
    row: RunRowSpec,
    optimizer_index: int,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    from feedbax.contracts.training import OptimizerSpec
    from feedbax.training.optimizers import build_optimizer

    optimizer_spec = OptimizerSpec.model_validate(payload)
    if optimizer_spec.lr_schedule is None:
        optimizer = build_optimizer(
            optimizer_spec,
            schedule_origin_step=0,
            current_step=0,
            optimizer_count_at_current_step=0,
        )
        optimizer.init({"preflight": 0.0})
        return {"optimizer_index": optimizer_index, "scheduled": False, "points": 0}

    if not isinstance(row.run_spec, Mapping):
        raise ValueError("row run_spec must be an inline mapping for schedule preflight")
    expected_context = schedule_eval.require_schedule_context(
        schedule_eval.extract_resume_context(row.run_spec),
        label="resume_context",
    )
    observed_context = schedule_eval.require_schedule_context(
        schedule_eval.extract_optimizer_build_context(row.run_spec),
        label="optimizer_build_context",
    )
    samples, mismatches = schedule_eval.compare_schedule_samples(
        optimizer_spec,
        expected_context=expected_context,
        observed_context=observed_context,
        rel_tol=1e-9,
    )
    optimizer = build_optimizer(
        optimizer_spec,
        schedule_origin_step=observed_context.schedule_origin_step,
        current_step=observed_context.current_step,
        optimizer_count_at_current_step=observed_context.optimizer_count_at_current_step,
    )
    optimizer.init({"preflight": 0.0})
    result = {
        "optimizer_index": optimizer_index,
        "scheduled": True,
        "expected_context": expected_context.model_dump(),
        "observed_context": observed_context.model_dump(),
        "samples": samples,
    }
    if mismatches:
        result["mismatches"] = mismatches
    return result


def _optimizer_payloads(run_spec: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    payloads: list[Mapping[str, Any]] = []
    for value in (
        _path(run_spec, "optimizer"),
        _path(run_spec, "training", "optimizer"),
        _path(run_spec, "training_config", "optimizer"),
        _path(run_spec, "method_payload", "payload", "optimizer"),
        _path(run_spec, "method_payload", "inline", "payload", "optimizer"),
    ):
        if isinstance(value, Mapping):
            payloads.append(value)
    return payloads


def _check(
    name: str,
    passed: bool,
    *,
    detail: str | None = None,
    observed: Any = None,
) -> PreflightCheckEntry:
    return PreflightCheckEntry(
        name=name,
        status="pass" if passed else "fail",
        detail=detail,
        observed=observed,
    )


def _path(value: Mapping[str, Any] | None, *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current
