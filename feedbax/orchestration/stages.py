"""Synchronous run-set stage engine."""

from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import stat
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from feedbax.contracts.manifest import (
    ParentRef,
    TrainingRunManifest,
    load_manifest_bytes,
)
from feedbax.contracts.remote_smoke import RemoteSmokeEvidence, RemoteSmokeRowEvidence
from feedbax.orchestration.assembly import (
    AssemblyCompilerRegistry,
    AssemblyContext,
    RunAssemblyRequest,
    assemble_run_bundle,
    persist_assembly_request,
)
from feedbax.orchestration.bundle import (
    RUN_BUNDLE_SCHEMA_VERSION,
    RunBundle,
    RunRowSpec,
    canonical_run_bundle_sha256,
    default_orchestration_root,
    mint_run_set_id,
)
from feedbax.orchestration.conformance import (
    CheckRegistry,
    ConformanceRowArtifacts,
    RowConformanceRuntimeInputs,
    RunConformanceCertificate,
    assert_certificate_allows_completed_registration,
    write_conformance_certificate,
)
from feedbax.orchestration.drivers.base import (
    AcquisitionCreateError,
    OrchestrationDriver,
    ProvisioningAttemptError,
)
from feedbax.orchestration.drivers.native_execution import (
    NATIVE_TRAINING_COLLECTION_OUTPUTS,
    missing_native_training_collection_outputs,
    uses_registered_native_execution,
)
from feedbax.orchestration.events import RunEventReader
from feedbax.orchestration.input_materialization import preflight_resolved_inputs
from feedbax.orchestration.revision import FeedbaxRevisionError, assert_feedbax_revision_pin
from feedbax.orchestration import schedule_eval
from feedbax.orchestration.state import (
    AcquisitionIntent,
    PreflightCheckEntry,
    RegistrationHistory,
    RegistrationHistoryEntry,
    RowState,
    RunSetState,
    RunSetStateStore,
    StageState,
    dependency_skip_observed,
    utc_now,
)
from feedbax.training.checkpoint_custody import (
    authenticate_published_checkpoint_custody,
)
from feedbax.training.diagnostics import (
    TRAINING_DIAGNOSTICS_SCHEMA_ID,
    TRAINING_DIAGNOSTICS_SCHEMA_VERSION,
    TRAINING_DIAGNOSTICS_SCHEMA_VERSION_V2,
    TRAINING_DIAGNOSTICS_SCHEMA_VERSION_V3,
    TrainingDiagnostics,
)
from feedbax.training.interruption import CancellationDecision
from feedbax.training.manifest_preflight import preflight_training_run_manifest_payloads


STAGE_ASSEMBLE = "ASSEMBLE"
STAGE_PREFLIGHT = "PREFLIGHT"
STAGE_PROVISION = "PROVISION"
STAGE_REALIZE_ENV = "REALIZE_ENV"
STAGE_STAGE_INPUTS = "STAGE_INPUTS"
STAGE_SMOKE = "SMOKE"
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
    STAGE_SMOKE,
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


def _request_run_set_dir(request: RunAssemblyRequest | None, run_set_id: str) -> Path:
    if request is not None and request.orchestration_root:
        root = Path(request.orchestration_root).expanduser()
        return root if root.name == run_set_id else root / run_set_id
    return default_orchestration_root(run_set_id)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent), text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parse_observed_datetime(value: Any) -> datetime | None:
    """Parse an already-recorded provider time without manufacturing a fallback."""
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _observed_spend_usd(
    provision_record: Mapping[str, Any] | None,
    *,
    observed_at: datetime,
) -> tuple[float, float, datetime]:
    """Return accrued USD spend from observed provider billing facts."""
    record = provision_record or {}
    billing_started_at = _parse_observed_datetime(record.get("billing_started_at"))
    hourly_rate = record.get("hourly_rate")
    currency = record.get("currency")
    missing: list[str] = []
    if billing_started_at is None or billing_started_at > observed_at:
        missing.append("billing_started_at")
    if (
        isinstance(hourly_rate, bool)
        or not isinstance(hourly_rate, (int, float))
        or not math.isfinite(float(hourly_rate))
        or float(hourly_rate) < 0.0
    ):
        missing.append("hourly_rate")
    if currency != "USD":
        missing.append("currency=USD")
    if missing:
        raise BudgetExceeded(
            "capped remote execution requires usable observed provider billing evidence: "
            + ", ".join(missing)
        )
    assert billing_started_at is not None
    rate = float(hourly_rate)
    elapsed_hours = (observed_at - billing_started_at).total_seconds() / 3600.0
    return rate * elapsed_hours, rate, billing_started_at


class OrchestrationStageError(RuntimeError):
    """Raised when a run-set stage fails."""


class PreflightFailed(OrchestrationStageError):
    """Raised when one or more named preflight checks fail."""


class BudgetExceeded(OrchestrationStageError):
    """Raised when a run-set budget guard aborts monitoring."""


class _PrimaryExecutorFailure(OrchestrationStageError):
    """Carry terminal executor failure plus secondary collection evidence."""

    def __init__(self, message: str, *, stage_outputs: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.stage_outputs = dict(stage_outputs)


class _DeferredOperatorSignal(BaseException):
    """Carry an operator signal across bounded cleanup without losing its identity."""

    def __init__(self, signum: int) -> None:
        super().__init__(f"received signal {signum}")
        self.signum = signum


class _ScopedSignalSupervisor:
    """Convert SIGINT/SIGTERM into an orderly abort for one run boundary.

    Python only permits signal-handler installation on the main thread. Off the
    main thread this context is intentionally a no-op, so callers retain normal
    thread-level cancellation semantics instead of failing during setup.
    """

    def __init__(self) -> None:
        self._installed = False
        self._previous: dict[int, Any] = {}
        self._received: int | None = None
        self._deferring = False

    def __enter__(self) -> "_ScopedSignalSupervisor":
        if threading.current_thread() is not threading.main_thread():
            return self
        for signum in (signal.SIGINT, signal.SIGTERM):
            self._previous[signum] = signal.getsignal(signum)
            signal.signal(signum, self._handle)
        self._installed = True
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, traceback: Any) -> bool:
        if not self._installed:
            return False
        for signum, previous in self._previous.items():
            signal.signal(signum, previous)
        if isinstance(exc, _DeferredOperatorSignal):
            # Re-deliver only after cleanup and restoration. Default SIGTERM
            # terminates normally; Python's default SIGINT handler re-raises
            # KeyboardInterrupt. A custom prior handler retains its semantics.
            signal.raise_signal(exc.signum)
            if exc.signum == signal.SIGINT:
                raise KeyboardInterrupt
            raise SystemExit(128 + exc.signum)
        return False

    @contextmanager
    def defer_signals(self) -> Iterator[None]:
        """Defer operator signals until the bounded cleanup scope exits."""
        self._deferring = True
        body_failed = False
        try:
            yield
        except BaseException:
            body_failed = True
            raise
        finally:
            self._deferring = False
        if self._received is not None and not body_failed:
            raise _DeferredOperatorSignal(self._received)

    def _handle(self, signum: int, _frame: Any) -> None:
        if self._deferring:
            self._received = signum
            return
        if self._received is None:
            self._received = signum
            raise _DeferredOperatorSignal(signum)
        # A signal arriving while the first one is unwinding remains deferred.
        # Bounded transport/log/teardown operations ensure this cannot become an
        # unbounded signal trap.
        self._received = signum


class StageEngine:
    """Execute a run bundle through the orchestration stage sequence."""

    def __init__(
        self,
        *,
        bundle: RunBundle | None = None,
        driver: OrchestrationDriver | None = None,
        request: RunAssemblyRequest | None = None,
        assembly_context: AssemblyContext | None = None,
        assembly_registry: AssemblyCompilerRegistry | None = None,
        driver_factory: Callable[[RunBundle], OrchestrationDriver] | None = None,
        run_set_id: str | None = None,
        store: RunSetStateStore | None = None,
        conformance_registry: CheckRegistry | None = None,
        row_conformance_inputs: Mapping[str, RowConformanceRuntimeInputs | Mapping[str, Any]]
        | None = None,
        poll_interval_seconds: float | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        wall_time: Callable[[], float] = time.time,
        interruption_probe: Callable[[], CancellationDecision | None] | None = None,
    ) -> None:
        if (bundle is None) == (request is None):
            raise ValueError("StageEngine requires exactly one of bundle or request")
        if bundle is not None and driver is None:
            raise ValueError("bundle-based StageEngine construction requires driver")
        if request is not None and (
            assembly_context is None or assembly_registry is None or driver_factory is None
        ):
            raise ValueError(
                "request-based StageEngine construction requires assembly_context, "
                "assembly_registry, and driver_factory"
            )
        self.bundle = bundle
        self.request = request
        self.assembly_context = assembly_context
        self.assembly_registry = assembly_registry
        self.driver_factory = driver_factory
        self.run_set_id = (
            bundle.run_set_id if bundle is not None else (run_set_id or mint_run_set_id())
        )
        self.driver = driver
        run_set_dir = (
            bundle.run_set_dir
            if bundle is not None
            else _request_run_set_dir(request, self.run_set_id)
        )
        self.store = store or RunSetStateStore(run_set_dir / "state.json")
        self.conformance_registry = conformance_registry or CheckRegistry()
        self.row_conformance_inputs = {
            row_id: RowConformanceRuntimeInputs.model_validate(inputs)
            for row_id, inputs in (row_conformance_inputs or {}).items()
        }
        self._poll_interval_explicit = poll_interval_seconds is not None
        self.poll_interval_seconds = (
            float(getattr(driver, "poll_interval_seconds", 0.05))
            if poll_interval_seconds is None
            else poll_interval_seconds
        )
        self._sleep = sleep
        self._monotonic = monotonic
        self._wall_time = wall_time
        self._interruption_probe = interruption_probe
        self._signal_supervisor: _ScopedSignalSupervisor | None = None

    @classmethod
    def from_request(
        cls,
        request: RunAssemblyRequest,
        *,
        context: AssemblyContext,
        registry: AssemblyCompilerRegistry,
        driver_factory: Callable[[RunBundle], OrchestrationDriver],
        run_set_id: str | None = None,
        **kwargs: Any,
    ) -> "StageEngine":
        """Construct a new or resumed engine whose ASSEMBLE stage compiles the request."""
        return cls(
            request=request,
            assembly_context=context,
            assembly_registry=registry,
            driver_factory=driver_factory,
            run_set_id=run_set_id,
            **kwargs,
        )

    def run(
        self,
        *,
        break_stale_lock: bool = False,
        stop_after_stage: str | None = None,
        retry_failed_certification: bool = False,
    ) -> RunSetState:
        """Run or resume the bundle through all stages.

        Main-thread SIGINT/SIGTERM handling is scoped to this call. Observable
        exits run bounded cleanup before the original exception or signal is
        re-raised. The durable acquisition intent is the only automatic coverage
        for SIGKILL before an acquired pod's SSH endpoint becomes available; a
        later local process or operator must reconcile it.
        """
        initial = self._initial_state()
        with _ScopedSignalSupervisor() as signal_supervisor:
            self._signal_supervisor = signal_supervisor
            with self.store.lock(break_stale=break_stale_lock):
                state = self.store.initialize(initial)
                state = self._hydrate_completed_assembly(state)
                self._restore_driver_from_provision_record(state)
                with signal_supervisor.defer_signals():
                    state = self._reconcile_acquisition_intents(state)
                state = self._restore_completed_driver_preflight(state)
                if retry_failed_certification:
                    self._preserve_failed_registration_history(state)
                    retry_state = self._reset_failed_certification(state)
                    if retry_state is not state:
                        self.store.save(retry_state)
                    state = retry_state
                try:
                    for stage_id in STAGE_ORDER:
                        if state.stage(stage_id).status == "completed":
                            continue
                        if stage_id == STAGE_TEARDOWN:
                            with signal_supervisor.defer_signals():
                                state = self._run_stage(stage_id, state)
                        else:
                            state = self._run_stage(stage_id, state)
                        if stop_after_stage == stage_id:
                            break
                    return state
                except BaseException:
                    latest = self.store.load() if self.store.path.exists() else state
                    if (
                        self.bundle is not None
                        and self.driver is not None
                        and (
                            self._provision_completed(latest)
                            or bool(
                                getattr(self.driver, "has_pending_owned_resource", lambda: False)()
                            )
                        )
                    ):
                        with signal_supervisor.defer_signals():
                            self._run_teardown(latest, abort=True)
                    raise

    @staticmethod
    def _reset_failed_certification(state: RunSetState) -> RunSetState:
        """Make a completed failing certificate eligible for one explicit retry."""
        certify = state.stage(STAGE_CERTIFY)
        if certify.status != "completed" or certify.outputs.get("overall") != "fail":
            return state
        return state.model_copy(
            update={
                "certificate_ref": None,
                "stages": {
                    **state.stages,
                    STAGE_CERTIFY: certify.model_copy(
                        update={
                            "status": "pending",
                            "checks": [],
                            "started_at": None,
                            "completed_at": None,
                            "outputs": {},
                            "error": None,
                        }
                    ),
                },
                "updated_at": utc_now(),
            }
        )

    def _preserve_failed_registration_history(self, state: RunSetState) -> None:
        """Preserve the failed registration before an explicit certification retry."""
        certify = state.stage(STAGE_CERTIFY)
        if certify.status != "completed" or certify.outputs.get("overall") != "fail":
            return
        registration_payload = state.registration_payload
        register_path = self.bundle.run_set_dir / "registration.json"
        if registration_payload is None and not register_path.exists():
            return
        if not isinstance(registration_payload, Mapping):
            raise OrchestrationStageError(
                "failed CERTIFY retry requires a durable failed registration payload"
            )
        certificate_ref = certify.outputs.get("certificate_ref")
        certificate_sha256 = certify.outputs.get("certificate_sha256")
        if not isinstance(certificate_ref, str) or not isinstance(certificate_sha256, str):
            raise OrchestrationStageError(
                "failed CERTIFY retry requires the prior certificate identity"
            )
        certificate_path = Path(certificate_ref)
        if (
            state.certificate_ref != certificate_ref
            or not register_path.exists()
            or not certificate_path.exists()
        ):
            raise OrchestrationStageError(
                "failed CERTIFY retry requires the prior registration and certificate bytes"
            )
        registration_bytes = register_path.read_bytes()
        certificate_bytes = certificate_path.read_bytes()
        try:
            persisted_registration = json.loads(registration_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise OrchestrationStageError(
                "failed CERTIFY retry requires valid prior registration JSON"
            ) from exc
        if (
            persisted_registration != dict(registration_payload)
            or registration_payload.get("status") != "failed"
            or registration_payload.get("certificate_overall") != "fail"
            or registration_payload.get("certificate_sha256") != certificate_sha256
            or hashlib.sha256(certificate_bytes).hexdigest() != certificate_sha256
        ):
            raise OrchestrationStageError(
                "failed CERTIFY retry prior registration or certificate identity mismatch"
            )
        self._write_or_verify_registration_history(
            self._registration_history(
                run_set_id=state.run_set_id,
                registration_payload=registration_payload,
                registration_bytes=registration_bytes,
            ),
            mismatch_label="failed CERTIFY retry",
        )

    def _recover_post_pass_registration_history(
        self,
        *,
        state: RunSetState,
        current: Mapping[str, Any],
    ) -> None:
        """Recover history when recertification passed before history capture existed."""
        if current.get("status") != "completed" or current.get("certificate_overall") != "pass":
            return
        history_path = self.bundle.run_set_dir / "registration-history.json"
        if history_path.exists():
            return
        prior = state.registration_payload
        register_path = self.bundle.run_set_dir / "registration.json"
        if prior is None and not register_path.exists():
            return
        if not isinstance(prior, Mapping) or not register_path.exists():
            raise OrchestrationStageError(
                "post-pass registration recovery requires the durable failed state and payload"
            )
        registration_bytes = register_path.read_bytes()
        try:
            persisted = json.loads(registration_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise OrchestrationStageError(
                "post-pass registration recovery requires valid failed registration JSON"
            ) from exc
        stable_fields = ("run_set_id", "abort_reason", "stage_inputs_sha256", "certificate_ref")
        if (
            persisted != dict(prior)
            or prior.get("status") != "failed"
            or prior.get("certificate_overall") != "fail"
            or current.get("status") != "completed"
            or current.get("certificate_overall") != "pass"
            or prior.get("certificate_sha256") == current.get("certificate_sha256")
            or any(prior.get(field) != current.get(field) for field in stable_fields)
        ):
            raise OrchestrationStageError(
                "post-pass registration recovery state or registration mismatch"
            )
        self._write_or_verify_registration_history(
            self._registration_history(
                run_set_id=state.run_set_id,
                registration_payload=prior,
                registration_bytes=registration_bytes,
            ),
            mismatch_label="post-pass registration recovery",
        )

    @staticmethod
    def _registration_history(
        *,
        run_set_id: str,
        registration_payload: Mapping[str, Any],
        registration_bytes: bytes,
    ) -> RegistrationHistory:
        return RegistrationHistory(
            run_set_id=run_set_id,
            entries=[
                RegistrationHistoryEntry(
                    registration_payload=dict(registration_payload),
                    registration_sha256=hashlib.sha256(registration_bytes).hexdigest(),
                    certificate_sha256=str(registration_payload.get("certificate_sha256")),
                    original_certificate_ref=str(registration_payload.get("certificate_ref")),
                )
            ],
        )

    def _write_or_verify_registration_history(
        self,
        history: RegistrationHistory,
        *,
        mismatch_label: str,
    ) -> None:
        history_path = self.bundle.run_set_dir / "registration-history.json"
        if history_path.exists():
            try:
                existing = RegistrationHistory.model_validate_json(
                    history_path.read_text(encoding="utf-8")
                )
            except Exception as exc:
                raise OrchestrationStageError(
                    f"{mismatch_label} registration history is invalid"
                ) from exc
            if existing != history:
                raise OrchestrationStageError(
                    f"{mismatch_label} registration history mismatch"
                )
            return
        self._write_json_atomically(history_path, history.model_dump(mode="json"))

    def _initial_state(self) -> RunSetState:
        return RunSetState(
            run_set_id=self.run_set_id,
            rows=(
                {row.row_id: RowState() for row in self.bundle.rows}
                if self.bundle is not None
                else {}
            ),
            stages={stage_id: StageState() for stage_id in STAGE_ORDER},
        )

    def _restore_driver_from_provision_record(self, state: RunSetState) -> None:
        """Rehydrate process-local remote identity before any resumed stage runs."""
        if state.provision_record is None:
            return
        restore = getattr(self.driver, "restore_from_provision_record", None)
        if callable(restore):
            restore(state.provision_record)

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
                failed_update: dict[str, Any] = {"status": "failed", "error": str(exc)}
                if isinstance(exc, _PrimaryExecutorFailure):
                    failed_update["outputs"] = exc.stage_outputs
                failed = state.stage(stage_id).model_copy(update=failed_update)
                state = state.with_stage(stage_id, failed)
                self.store.save(state)
                governed_provision = stage_id == STAGE_PROVISION and bool(
                    getattr(self.driver, "govern_provisioning_retries", False)
                )
                if (
                    isinstance(exc, (_PrimaryExecutorFailure, BudgetExceeded))
                    or governed_provision
                    or attempts >= limit
                ):
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
        if self.request is not None:
            assert self.assembly_context is not None
            assert self.assembly_registry is not None
            run_set_dir = _request_run_set_dir(self.request, self.run_set_id)
            request_path = run_set_dir / "assembly-request.json"
            request_sha256 = persist_assembly_request(self.request, request_path)
            self.bundle = assemble_run_bundle(
                self.request,
                run_set_id=self.run_set_id,
                context=self.assembly_context,
                registry=self.assembly_registry,
            )
            assert self.driver_factory is not None
            self.driver = self.driver_factory(self.bundle)
            if not self._poll_interval_explicit:
                self.poll_interval_seconds = float(
                    getattr(self.driver, "poll_interval_seconds", 0.05)
                )
            state = state.model_copy(
                update={
                    "rows": {row.row_id: RowState() for row in self.bundle.rows},
                    "updated_at": utc_now(),
                }
            )
        else:
            assert self.bundle is not None
            run_set_dir = self.bundle.run_set_dir
            request_path = None
            request_sha256 = None
        run_set_dir.mkdir(parents=True, exist_ok=True)
        payload = self.bundle.model_dump(mode="json", exclude_none=True)
        bundle_path = run_set_dir / "bundle.json"
        _atomic_write_json(bundle_path, payload)
        outputs = {
            "bundle_path": str(bundle_path),
            "bundle_sha256": canonical_run_bundle_sha256(self.bundle),
        }
        if request_path is not None:
            outputs.update(
                {
                    "request_path": str(request_path),
                    "request_sha256": request_sha256,
                    "artifacts": [
                        ref.model_dump(mode="json")
                        for row in self.bundle.rows
                        for ref in (
                            row.execution.payload,
                            row.execution.authored_intent,
                            row.execution.resolved_snapshot,
                            row.execution.execution_capsule,
                        )
                    ],
                }
            )
        return state, outputs

    def _hydrate_completed_assembly(self, state: RunSetState) -> RunSetState:
        """Load and verify the persisted v3 bundle when resuming after ASSEMBLE."""
        if self.bundle is not None or state.stage(STAGE_ASSEMBLE).status != "completed":
            return state
        bundle_path = self.store.path.parent / "bundle.json"
        data = bundle_path.read_bytes()
        expected = state.stage(STAGE_ASSEMBLE).outputs.get("bundle_sha256")
        actual = canonical_run_bundle_sha256(RunBundle.model_validate_json(data))
        if expected != actual:
            raise OrchestrationStageError(
                f"persisted ASSEMBLE bundle hash mismatch: expected={expected!r} actual={actual!r}"
            )
        self.bundle = RunBundle.model_validate_json(data)
        assert self.driver_factory is not None
        self.driver = self.driver_factory(self.bundle)
        return state

    def _restore_completed_driver_preflight(self, state: RunSetState) -> RunSetState:
        """Restore driver-local preflight authority before skipping a completed stage."""
        if state.stage(STAGE_PREFLIGHT).status != "completed":
            return state
        restore = getattr(self.driver, "restore_completed_preflight", None)
        if callable(restore):
            try:
                reusable = restore(self.bundle, state)
            except Exception as exc:
                raise PreflightFailed(
                    f"persisted driver PREFLIGHT evidence is invalid: {exc}"
                ) from exc
            if reusable is False:
                later_started = [
                    stage_id
                    for stage_id in STAGE_ORDER[STAGE_ORDER.index(STAGE_PREFLIGHT) + 1 :]
                    if state.stage(stage_id).status != "pending"
                ]
                if later_started:
                    raise PreflightFailed(
                        "stale PREFLIGHT evidence cannot be rerun after later stages started; "
                        f"stages={later_started!r}"
                    )
                reset = state.stage(STAGE_PREFLIGHT).model_copy(
                    update={
                        "status": "pending",
                        "started_at": None,
                        "completed_at": None,
                        "outputs": {},
                        "error": None,
                        "checks": [],
                    }
                )
                state = state.with_stage(STAGE_PREFLIGHT, reset)
                self.store.save(state)
        return state

    def _stage_preflight(self, state: RunSetState) -> tuple[RunSetState, Mapping[str, Any]]:
        checks = run_preflight_checks(self.bundle)
        stage = state.stage(STAGE_PREFLIGHT).model_copy(update={"checks": checks})
        state = state.with_stage(STAGE_PREFLIGHT, stage)
        self.store.save(state)
        failed = [check for check in checks if check.status == "fail"]
        if failed:
            driver_static_preflight = getattr(self.driver, "static_preflight_checks", None)
            if callable(driver_static_preflight):
                checks.extend(
                    driver_static_preflight(
                        self.bundle,
                        upstream_failures=tuple(check.name for check in failed),
                    )
                )
                stage = state.stage(STAGE_PREFLIGHT).model_copy(update={"checks": checks})
                state = state.with_stage(STAGE_PREFLIGHT, stage)
                self.store.save(state)
                failed = [check for check in checks if check.status == "fail"]
            raise PreflightFailed(_format_preflight_failures(failed))
        driver_preflight = getattr(self.driver, "preflight_checks", None)
        if callable(driver_preflight):
            checks.extend(driver_preflight(self.bundle))
        realization_plan = getattr(self.driver, "repo_realization_plan", None)
        if callable(realization_plan):
            plan = realization_plan()
            if plan is not None:
                state = state.model_copy(
                    update={"repo_realization_plan": plan, "updated_at": utc_now()}
                )
        stage = state.stage(STAGE_PREFLIGHT).model_copy(update={"checks": checks})
        state = state.with_stage(STAGE_PREFLIGHT, stage)
        self.store.save(state)
        failed = [check for check in checks if check.status == "fail"]
        if failed:
            raise PreflightFailed(_format_preflight_failures(failed))
        outputs: dict[str, Any] = {"checks": [check.model_dump(mode="json") for check in checks]}
        evidence = getattr(self.driver, "preflight_evidence", None)
        if callable(evidence):
            outputs["driver_evidence"] = dict(evidence(self.bundle, state, checks))
        if state.repo_realization_plan is not None:
            outputs["repo_realization_plan"] = state.repo_realization_plan.model_dump(mode="json")
        return state, outputs

    def _uses_engine_acquisition(self) -> bool:
        required = getattr(self.driver, "engine_acquisition_required", None)
        return bool(callable(required) and required())

    def _replace_acquisition_intent(
        self,
        run_state: RunSetState,
        intent_id: str,
        **updates: Any,
    ) -> RunSetState:
        intents = list(run_state.acquisition_intents)
        for index, intent in enumerate(intents):
            if intent.intent_id == intent_id:
                intents[index] = intent.model_copy(update={**updates, "updated_at": utc_now()})
                return run_state.model_copy(
                    update={"acquisition_intents": intents, "updated_at": utc_now()}
                )
        raise OrchestrationStageError(f"unknown acquisition intent {intent_id!r}")

    def _new_acquisition_intent(
        self,
        state: RunSetState,
        *,
        attempt_ordinal: int,
        candidate_ordinal: int,
        candidate: str | None,
    ) -> tuple[RunSetState, AcquisitionIntent]:
        identity = self.driver.acquisition_config_identity(self.bundle)
        nonce = uuid.uuid4().hex[:12]
        intent = AcquisitionIntent(
            intent_id=(f"{state.run_set_id}-a{attempt_ordinal}-c{candidate_ordinal}-{nonce}"),
            datacenter_candidate=candidate,
            config_identity=identity,
        )
        state = state.model_copy(
            update={
                "acquisition_intents": [*state.acquisition_intents, intent],
                "updated_at": utc_now(),
            }
        )
        # This save, including file and parent-directory fsync, is the WAL boundary.
        self.store.save(state)
        return state, intent

    def _engine_owned_provision(
        self,
        state: RunSetState,
        *,
        attempt_ordinal: int,
    ) -> tuple[RunSetState, Mapping[str, Any]]:
        candidates = tuple(self.driver.acquisition_candidates(self.bundle))
        if not candidates:
            raise OrchestrationStageError("engine-owned acquisition has no candidates")
        last_clean_error = "provider rejected every acquisition candidate"
        for candidate_ordinal, candidate in enumerate(candidates, start=1):
            state, intent = self._new_acquisition_intent(
                state,
                attempt_ordinal=attempt_ordinal,
                candidate_ordinal=candidate_ordinal,
                candidate=candidate,
            )
            try:
                acquisition = self.driver.create_pod_once(self.bundle, candidate, intent.intent_id)
            except AcquisitionCreateError as exc:
                if exc.clean_rejection:
                    last_clean_error = str(exc)
                    state = self._replace_acquisition_intent(
                        state,
                        intent.intent_id,
                        state="failed-unacquired",
                        evidence={"create": exc.evidence},
                    )
                    self.store.save(state)
                    continue
                state = self._replace_acquisition_intent(
                    state,
                    intent.intent_id,
                    state="ambiguous",
                    evidence={"create": exc.evidence},
                )
                self.store.save(state)
                if self._signal_supervisor is None:
                    raise OrchestrationStageError(
                        "acquisition reconciliation requires the run signal supervisor"
                    ) from exc
                with self._signal_supervisor.defer_signals():
                    state = self._reconcile_acquisition_intents(state)
                raise ProvisioningAttemptError(
                    str(exc),
                    retryable=True,
                    attempt_record={
                        "driver": "runpod",
                        "acquired": False,
                        "intent_id": intent.intent_id,
                        "reconciliation": "resolved-torn-down",
                    },
                ) from exc

            state = self._replace_acquisition_intent(
                state,
                intent.intent_id,
                state="acquired",
                pod_ids=[acquisition.pod_id],
                evidence={
                    "create": {
                        "classification": "acquired",
                        "pod_id": acquisition.pod_id,
                    }
                },
            )
            self.store.save(state)
            try:
                outputs = dict(
                    self.driver.finish_acquired_pod(self.bundle, acquisition, intent.intent_id)
                )
            except BaseException as exc:
                cleanup: Mapping[str, Any] | None = None
                failure_evidence = getattr(
                    self.driver, "acquisition_failure_evidence", lambda: {}
                )()
                try:
                    cleanup = self.driver.teardown(self.bundle, state)
                except Exception as teardown_exc:
                    cleanup = dict(getattr(teardown_exc, "teardown_outputs", {}))
                    if not isinstance(exc, Exception):
                        raise exc
                    raise ProvisioningAttemptError(
                        f"{exc}; automatic teardown failed: {teardown_exc}",
                        retryable=False,
                        attempt_record={
                            "driver": "runpod",
                            "acquired": True,
                            "pod_id": acquisition.pod_id,
                            "intent_id": intent.intent_id,
                            **dict(failure_evidence),
                            "cleanup": cleanup,
                        },
                        stop_reason="teardown-failure",
                    ) from exc
                state = self._project_intent_teardown(state, cleanup)
                self.store.save(state)
                if not isinstance(exc, Exception):
                    raise
                raise ProvisioningAttemptError(
                    str(exc),
                    retryable=True,
                    attempt_record={
                        "driver": "runpod",
                        "acquired": True,
                        "pod_id": acquisition.pod_id,
                        "intent_id": intent.intent_id,
                        **dict(failure_evidence),
                        "cleanup": dict(cleanup),
                    },
                ) from exc
            return state, outputs
        raise ProvisioningAttemptError(
            last_clean_error,
            retryable=True,
            attempt_record={"driver": "runpod", "acquired": False},
        )

    def _reconcile_acquisition_intents(self, state: RunSetState) -> RunSetState:
        if not self._uses_engine_acquisition():
            return state
        provision_intent_id = (
            state.provision_record.get("intent_id")
            if isinstance(state.provision_record, Mapping)
            else None
        )
        pending = [
            intent
            for intent in state.acquisition_intents
            if intent.state in {"intended", "ambiguous", "ambiguous-unresolved"}
            or (intent.state == "acquired" and intent.intent_id != provision_intent_id)
        ]
        if not pending:
            return state
        timeout = float(getattr(self.driver, "reconciliation_timeout_seconds", 60.0))
        deadline = self._monotonic() + timeout
        records, inventory_evidence = self.driver.observe_pod_inventory(
            timeout_seconds=max(0.0, deadline - self._monotonic())
        )
        pod_name = getattr(self.driver, "acquisition_pod_name", None)
        if not callable(pod_name):
            raise OrchestrationStageError("acquisition driver does not expose intent pod names")
        for original in pending:
            expected_name = pod_name(original.intent_id)
            matches = [record for record in records if record.name == expected_name]
            evidence = {
                **original.evidence,
                "inventory": dict(inventory_evidence),
                "expected_name": expected_name,
                "matched_pod_ids": [record.pod_id for record in matches],
            }
            if not matches:
                evidence["unresolved_owned_pod"] = {
                    "pod_id": None,
                    "name": expected_name,
                    "last_known_state": original.state,
                    "reason": "bounded inventory absence cannot prove create finality",
                }
                state = self._replace_acquisition_intent(
                    state,
                    original.intent_id,
                    state="ambiguous-unresolved",
                    evidence=evidence,
                )
                self.store.save(state)
                self._stop_provisioning(
                    state,
                    list(state.provisioning_attempts),
                    "ambiguous-acquisition-unresolved",
                )
                raise OrchestrationStageError(
                    "provisioning stopped: ambiguous-acquisition-unresolved"
                )
            state = self._replace_acquisition_intent(
                state,
                original.intent_id,
                state="ambiguous",
                pod_ids=[record.pod_id for record in matches],
                evidence=evidence,
            )
            self.store.save(state)
            teardown_evidence = list(original.teardown_evidence)
            for record in matches:
                if self._monotonic() >= deadline:
                    raise OrchestrationStageError(
                        "acquisition reconciliation exceeded its teardown deadline"
                    )
                self.driver.adopt_owned_pod(
                    record.pod_id,
                    timeout_seconds=max(0.0, deadline - self._monotonic()),
                )
                adopted_state = state.model_copy(
                    update={
                        "provision_record": {
                            "driver": "runpod",
                            "pod_id": record.pod_id,
                            "provided_pod": False,
                            "provided_endpoint": False,
                            "teardown_allowed": True,
                            "intent_id": original.intent_id,
                        }
                    }
                )
                try:
                    teardown = dict(self.driver.teardown(self.bundle, adopted_state))
                except Exception as exc:
                    outputs = dict(getattr(exc, "teardown_outputs", {}))
                    evidence["unresolved_owned_pod"] = outputs.get(
                        "unresolved_owned_pod",
                        {
                            "pod_id": record.pod_id,
                            "last_known_state": "unknown",
                            "reason": str(exc),
                        },
                    )
                    state = self._replace_acquisition_intent(
                        state,
                        original.intent_id,
                        state="ambiguous-unresolved",
                        evidence=evidence,
                        teardown_evidence=[*teardown_evidence, outputs],
                    )
                    self.store.save(state)
                    raise
                teardown_evidence.append(teardown)
                state = self._replace_acquisition_intent(
                    state,
                    original.intent_id,
                    teardown_evidence=teardown_evidence,
                )
                self.store.save(state)
            state = self._replace_acquisition_intent(
                state,
                original.intent_id,
                state="resolved-torn-down",
                teardown_evidence=teardown_evidence,
            )
            self.store.save(state)
        return state

    def _stage_provision(self, state: RunSetState) -> tuple[RunSetState, Mapping[str, Any]]:
        """Provision RunPod through one durable, budget-governed retry authority."""
        if not getattr(self.driver, "govern_provisioning_retries", False):
            outputs = dict(self.driver.provision(self.bundle, state))
            state = state.model_copy(update={"provision_record": outputs, "updated_at": utc_now()})
            return state, outputs

        counters = dict(state.budget_counters)
        started_at = float(counters.setdefault("provisioning_started_at", self._wall_time()))
        deadline = float(
            counters.setdefault(
                "provisioning_deadline_at", started_at + self.bundle.budget.max_wall_clock_seconds
            )
        )
        state = state.model_copy(update={"budget_counters": counters, "updated_at": utc_now()})
        self.store.save(state)
        attempts = list(state.provisioning_attempts)
        while True:
            decision = self._interruption_probe() if self._interruption_probe is not None else None
            if decision is not None and decision.action != "continue":
                self._stop_provisioning(state, attempts, "cancelled", decision.as_provenance())
                raise BudgetExceeded("provisioning cancelled by operator")
            now = self._wall_time()
            if now >= deadline:
                self._stop_provisioning(state, attempts, "wall-clock-exceeded")
                raise BudgetExceeded("provisioning exceeded its wall-clock boundary")
            try:
                if self._uses_engine_acquisition():
                    state, provision_outputs = self._engine_owned_provision(
                        state,
                        attempt_ordinal=len(attempts) + 1,
                    )
                    outputs = dict(provision_outputs)
                else:
                    outputs = dict(self.driver.provision(self.bundle, state))
            except ProvisioningAttemptError as exc:
                finished_at = self._wall_time()
                attempt = {
                    **exc.attempt_record,
                    "attempt": len(attempts) + 1,
                    "started_at_unix_seconds": now,
                    "finished_at_unix_seconds": finished_at,
                    "error": str(exc),
                    "retryable": exc.retryable,
                }
                attempts.append(attempt)
                state = state.model_copy(
                    update={"provisioning_attempts": attempts, "updated_at": utc_now()}
                )
                if exc.stop_reason is not None:
                    self._stop_provisioning(state, attempts, exc.stop_reason)
                    raise OrchestrationStageError(
                        f"provisioning stopped: {exc.stop_reason}"
                    ) from exc
                cleanup = attempt.get("cleanup")
                absence = cleanup.get("pod_absence") if isinstance(cleanup, Mapping) else None
                if attempt.get("acquired") and not (
                    isinstance(absence, Mapping) and absence.get("verified") is True
                ):
                    self._stop_provisioning(state, attempts, "cleanup-proof-unavailable")
                    raise OrchestrationStageError(
                        "provisioning stopped: cleanup-proof-unavailable"
                    ) from exc
                state, exceeded = self._apply_failed_provision_cost(state, attempt)
                self.store.save(state)
                if exceeded:
                    reason = state.provisioning_stop_reason or "spend-exceeded"
                    self._stop_provisioning(state, attempts, reason)
                    raise BudgetExceeded(f"provisioning stopped: {reason}") from exc
                if not exc.retryable:
                    self._stop_provisioning(state, attempts, "non-retryable-error")
                    raise OrchestrationStageError(
                        "provisioning stopped: non-retryable-error"
                    ) from exc
                if self._wall_time() >= deadline:
                    self._stop_provisioning(state, attempts, "wall-clock-exceeded")
                    raise BudgetExceeded("provisioning exceeded its wall-clock boundary") from exc
                delay = float(getattr(self.driver, "provision_retry_delay_seconds", 0.0))
                if delay <= 0.0:
                    self._stop_provisioning(state, attempts, "invalid-retry-delay")
                    raise OrchestrationStageError(
                        "RunPod provisioning retry delay must be positive"
                    )
                self._sleep(min(delay, max(0.0, deadline - self._wall_time())))
                continue
            outputs["provisioning_attempts"] = attempts
            counters = dict(state.budget_counters)
            counters.pop("provisioning_stop_reason", None)
            prior_stop_reason = state.provisioning_stop_reason
            state = state.model_copy(
                update={
                    "provision_record": outputs,
                    "provisioning_attempts": attempts,
                    "provisioning_stop_reason": None,
                    "budget_counters": counters,
                    "abort_reason": (
                        None if state.abort_reason == prior_stop_reason else state.abort_reason
                    ),
                    "updated_at": utc_now(),
                }
            )
            return state, outputs

    def _stop_provisioning(
        self,
        state: RunSetState,
        attempts: list[dict[str, Any]],
        reason: str,
        cancellation: Mapping[str, Any] | None = None,
    ) -> RunSetState:
        counters = dict(state.budget_counters)
        counters["provisioning_stop_reason"] = reason
        if cancellation is not None:
            counters["provisioning_cancellation"] = dict(cancellation)
        state = state.model_copy(
            update={
                "provisioning_attempts": attempts,
                "provisioning_stop_reason": reason,
                "budget_counters": counters,
                "abort_reason": reason,
                "updated_at": utc_now(),
            }
        )
        self.store.save(state)
        return state

    def _apply_failed_provision_cost(
        self, state: RunSetState, attempt: Mapping[str, Any]
    ) -> tuple[RunSetState, bool]:
        """Account for a torn-down acquired attempt before permitting another one."""
        if not attempt.get("acquired") or self.bundle.budget.max_spend_usd is None:
            return state, False
        observed_at = datetime.fromtimestamp(
            float(attempt["finished_at_unix_seconds"]), tz=timezone.utc
        )
        try:
            cost, _rate, _started = _observed_spend_usd(attempt, observed_at=observed_at)
        except BudgetExceeded:
            return self._stop_provisioning(
                state, list(state.provisioning_attempts), "budget-evidence-unavailable"
            ), True
        counters = dict(state.budget_counters)
        total = float(counters.get("failed_provision_cost_usd", 0.0)) + cost
        counters["failed_provision_cost_usd"] = total
        counters["max_spend_usd"] = self.bundle.budget.max_spend_usd
        state = state.model_copy(update={"budget_counters": counters, "updated_at": utc_now()})
        return state, total >= self.bundle.budget.max_spend_usd

    def _stage_realize_env(self, state: RunSetState) -> tuple[RunSetState, Mapping[str, Any]]:
        state, spend_exceeded = self._apply_spend_budget(state)
        self.store.save(state)
        if spend_exceeded:
            raise BudgetExceeded("max_spend_usd reached before REALIZE_ENV")
        fingerprint = self.driver.realize_env(self.bundle, state)
        state = state.model_copy(
            update={"environment_fingerprint": fingerprint, "updated_at": utc_now()}
        )
        return state, {"environment_fingerprint": fingerprint}

    def _stage_stage_inputs(self, state: RunSetState) -> tuple[RunSetState, Mapping[str, Any]]:
        return state, dict(self.driver.stage_inputs(self.bundle, state))

    def _stage_smoke(self, state: RunSetState) -> tuple[RunSetState, Mapping[str, Any]]:
        launch = state.stage(STAGE_LAUNCH)
        if launch.status != "pending" or launch.started_at is not None:
            raise OrchestrationStageError(
                "SMOKE refuses to run after LAUNCH has started; pre-launch evidence is missing"
            )

        if not self.bundle.smoke_enabled:
            evidence = RemoteSmokeEvidence(
                run_set_id=self.bundle.run_set_id,
                bundle_sha256=canonical_run_bundle_sha256(self.bundle),
                rows=tuple(
                    RemoteSmokeRowEvidence(
                        row_id=row.row_id,
                        status="opted-out",
                        update_budget=self.bundle.smoke_update_budget,
                        payload_binding_status="not-run",
                        cleanup_status="not-created",
                        deadline_seconds=self.bundle.smoke_deadline_seconds,
                        opt_out_reason="bundle smoke_enabled=false",
                    )
                    for row in self.bundle.rows
                ),
            )
            return state, evidence.model_dump(mode="json")

        smoke_row = getattr(self.driver, "smoke_row", None)
        if not callable(smoke_row):
            evidence = RemoteSmokeEvidence(
                run_set_id=self.bundle.run_set_id,
                bundle_sha256=canonical_run_bundle_sha256(self.bundle),
                rows=tuple(
                    RemoteSmokeRowEvidence(
                        row_id=row.row_id,
                        status="opted-out",
                        update_budget=self.bundle.smoke_update_budget,
                        payload_binding_status="not-run",
                        cleanup_status="not-created",
                        deadline_seconds=self.bundle.smoke_deadline_seconds,
                        opt_out_reason=(
                            "remote smoke is inapplicable to driver "
                            f"{self.bundle.deployment_policy.driver!r}"
                        ),
                    )
                    for row in self.bundle.rows
                ),
            )
            return state, evidence.model_dump(mode="json")

        rows: list[dict[str, Any]] = []
        for row in self.bundle.rows:
            try:
                rows.append(dict(smoke_row(self.bundle, row, state)))
            except Exception as exc:
                raw_evidence = getattr(exc, "evidence", None)
                evidence = (
                    dict(raw_evidence)
                    if isinstance(raw_evidence, Mapping)
                    else {
                        "row_id": row.row_id,
                        "status": "failed",
                        "error": str(exc),
                    }
                )
                rows.append(evidence)
                stage_evidence = RemoteSmokeEvidence(
                    run_set_id=self.bundle.run_set_id,
                    bundle_sha256=canonical_run_bundle_sha256(self.bundle),
                    rows=tuple(RemoteSmokeRowEvidence.model_validate(item) for item in rows),
                )
                raise _PrimaryExecutorFailure(
                    f"remote smoke failed for row {row.row_id!r}: {exc}",
                    stage_outputs=stage_evidence.model_dump(mode="json"),
                ) from exc
        evidence = RemoteSmokeEvidence(
            run_set_id=self.bundle.run_set_id,
            bundle_sha256=canonical_run_bundle_sha256(self.bundle),
            rows=tuple(RemoteSmokeRowEvidence.model_validate(item) for item in rows),
        )
        return state, evidence.model_dump(mode="json")

    def _stage_launch(self, state: RunSetState) -> tuple[RunSetState, Mapping[str, Any]]:
        state, spend_exceeded = self._apply_spend_budget(state)
        self.store.save(state)
        if spend_exceeded:
            raise BudgetExceeded("max_spend_usd reached before LAUNCH")
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
        spend_exceeded = False
        while True:
            state, spend_exceeded = self._apply_spend_budget(state)
            if spend_exceeded:
                budget_exceeded = True
                state = self._stop_unfinished(state, reason="budget-exceeded")
                break
            decision = self._interruption_probe() if self._interruption_probe is not None else None
            if decision is not None and decision.action != "continue":
                state = self._apply_interruption(state, decision)
            state = self._refresh_rows(state)
            state, spend_exceeded = self._apply_spend_budget(state)
            if spend_exceeded:
                budget_exceeded = True
                state = self._stop_unfinished(state, reason="budget-exceeded")
                break
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
            counters["budget_exceeded"] = "spend" if spend_exceeded else "wall-clock"
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

    def _apply_spend_budget(self, state: RunSetState) -> tuple[RunSetState, bool]:
        """Record and enforce capped spend using only observed provider facts."""
        max_spend_usd = self.bundle.budget.max_spend_usd
        if self.bundle.deployment_policy.venue == "local" or max_spend_usd is None:
            return state, False
        observed_at = datetime.fromtimestamp(self._wall_time(), tz=timezone.utc)
        counters = dict(state.budget_counters)
        try:
            current_cost_usd, hourly_rate_usd, billing_started_at = _observed_spend_usd(
                state.provision_record,
                observed_at=observed_at,
            )
        except OrchestrationStageError:
            counters.update(
                {
                    "max_spend_usd": max_spend_usd,
                    "budget_exceeded": "spend-evidence-unavailable",
                }
            )
            unavailable = state.model_copy(
                update={
                    "budget_counters": counters,
                    "abort_reason": "budget-evidence-unavailable",
                    "updated_at": utc_now(),
                }
            )
            self.store.save(unavailable)
            raise
        failed_provision_cost_usd = float(counters.get("failed_provision_cost_usd", 0.0))
        accrued_cost_usd = failed_provision_cost_usd + current_cost_usd
        counters.update(
            {
                "billing_started_at": billing_started_at.isoformat(),
                "hourly_rate_usd": hourly_rate_usd,
                "accrued_cost_usd": accrued_cost_usd,
                "current_provision_cost_usd": current_cost_usd,
                "failed_provision_cost_usd": failed_provision_cost_usd,
                "max_spend_usd": max_spend_usd,
            }
        )
        exceeded = accrued_cost_usd >= max_spend_usd
        if exceeded:
            counters["budget_exceeded"] = "spend"
        state = state.model_copy(
            update={
                "budget_counters": counters,
                "abort_reason": "budget-exceeded" if exceeded else state.abort_reason,
                "updated_at": utc_now(),
            }
        )
        return state, exceeded

    def _stage_collect(self, state: RunSetState) -> tuple[RunSetState, Mapping[str, Any]]:
        collected: dict[str, Mapping[str, str]] = {}
        checkpoint_custody: dict[str, Mapping[str, Any]] = {}
        collection_recovery: dict[str, Mapping[str, Any]] = {}
        executor_failures = [
            {
                "row_id": row.row_id,
                "error": state.rows[row.row_id].error or "executor reported failure without detail",
            }
            for row in self.bundle.rows
            if state.rows[row.row_id].status == "failed"
        ]
        preserve_executor_failure = bool(executor_failures)
        secondary_evidence: list[dict[str, Any]] = []
        for row in self.bundle.rows:
            row_state = state.rows[row.row_id]
            try:
                outputs = dict(self.driver.collect(self.bundle, row, state))
            except Exception as exc:
                if not preserve_executor_failure:
                    raise
                outputs = {}
                secondary_evidence.append(
                    {
                        "kind": "collection_error_after_executor_failure",
                        "row_id": row.row_id,
                        "detail": str(exc),
                    }
                )
            missing_outputs = _missing_declared_collection_outputs(row, outputs)
            if missing_outputs:
                evidence = {
                    "kind": "absent_collection_outputs",
                    "row_id": row.row_id,
                    "missing_outputs": missing_outputs,
                }
                if not preserve_executor_failure:
                    raise OrchestrationStageError(
                        f"declared collection outputs are absent for row "
                        f"{row.row_id!r}: {missing_outputs!r}"
                    )
                secondary_evidence.append(evidence)
            elif uses_registered_native_execution(row):
                try:
                    checkpoint_custody[row.row_id] = _verify_collected_native_checkpoint_custody(
                        row, outputs
                    )
                except Exception as exc:
                    if not preserve_executor_failure:
                        raise
                    secondary_evidence.append(
                        {
                            "kind": "checkpoint_custody_verification_after_executor_failure",
                            "row_id": row.row_id,
                            "detail": str(exc),
                        }
                    )
            collected[row.row_id] = outputs
            recovery_evidence = getattr(self.driver, "collection_recovery_evidence", None)
            if callable(recovery_evidence):
                row_recovery = recovery_evidence(row.row_id)
                if row_recovery is not None:
                    collection_recovery[row.row_id] = dict(row_recovery)
            row_state = row_state.model_copy(update={"collected_outputs": outputs})
            state = state.with_row(row.row_id, row_state)
            self.store.save(state)
        stage_outputs: dict[str, Any] = {"rows": collected}
        if checkpoint_custody:
            stage_outputs["checkpoint_custody"] = checkpoint_custody
        if collection_recovery:
            stage_outputs["collection_recovery"] = collection_recovery
        if secondary_evidence:
            stage_outputs["secondary_evidence"] = secondary_evidence
        if executor_failures:
            stage_outputs["executor_failures"] = executor_failures
            primary = executor_failures[0]
            raise _PrimaryExecutorFailure(
                f"executor failed for row {primary['row_id']!r}: {primary['error']}",
                stage_outputs=stage_outputs,
            )
        return state, stage_outputs

    def _stage_certify(self, state: RunSetState) -> tuple[RunSetState, Mapping[str, Any]]:
        smoke = state.stage(STAGE_SMOKE)
        if smoke.status != "completed":
            raise OrchestrationStageError("CERTIFY requires completed SMOKE evidence")
        try:
            smoke_evidence = RemoteSmokeEvidence.model_validate(smoke.outputs)
        except ValueError as exc:
            raise OrchestrationStageError("CERTIFY requires typed per-row SMOKE evidence") from exc
        if smoke_evidence.run_set_id != self.bundle.run_set_id:
            raise OrchestrationStageError("CERTIFY rejected SMOKE run-set identity")
        if smoke_evidence.bundle_sha256 != canonical_run_bundle_sha256(self.bundle):
            raise OrchestrationStageError("CERTIFY rejected SMOKE bundle identity")
        smoke_by_row = {item.row_id: item for item in smoke_evidence.rows}
        missing_smoke = [row.row_id for row in self.bundle.rows if row.row_id not in smoke_by_row]
        invalid_smoke = [
            row.row_id
            for row in self.bundle.rows
            if row.row_id in smoke_by_row
            and smoke_by_row[row.row_id].status not in {"passed", "opted-out"}
        ]
        invalid_opt_out = [
            row.row_id
            for row in self.bundle.rows
            if row.row_id in smoke_by_row
            and smoke_by_row[row.row_id].status == "opted-out"
            and not smoke_by_row[row.row_id].opt_out_reason
        ]
        if missing_smoke or invalid_smoke or invalid_opt_out:
            raise OrchestrationStageError(
                "CERTIFY rejected SMOKE evidence: "
                f"missing={missing_smoke!r}, invalid={invalid_smoke!r}, "
                f"opt_out_without_reason={invalid_opt_out!r}"
            )
        if len(self.conformance_registry) == 0:
            raise OrchestrationStageError(
                "CERTIFY requires at least one registered conformance check"
            )
        stage_inputs = state.stage(STAGE_STAGE_INPUTS)
        if stage_inputs.status != "completed":
            raise OrchestrationStageError("CERTIFY requires completed STAGE_INPUTS authority")
        stage_inputs_sha256 = _canonical_json_sha256(stage_inputs.outputs)
        observed_at = utc_now()
        rows = [
            self._conformance_artifacts(row, state, observed_at=observed_at)
            for row in self.bundle.rows
        ]
        certificate = write_conformance_certificate(
            run_set_dir=self.bundle.run_set_dir,
            run_set_id=self.bundle.run_set_id,
            rows=rows,
            registry=self.conformance_registry,
            declared_inapplicable=self.bundle.metadata.get("conformance_inapplicable"),
            generated_at=observed_at,
        )
        certificate_path = self.bundle.run_set_dir / "conformance.json"
        certificate_sha256 = hashlib.sha256(certificate_path.read_bytes()).hexdigest()
        state = state.model_copy(
            update={"certificate_ref": str(certificate_path), "updated_at": utc_now()}
        )
        return state, {
            "certificate_ref": str(certificate_path),
            "certificate_sha256": certificate_sha256,
            "stage_inputs_sha256": stage_inputs_sha256,
            "overall": certificate.overall,
        }

    def _conformance_artifacts(
        self,
        row: RunRowSpec,
        state: RunSetState,
        *,
        observed_at: datetime | None = None,
    ) -> ConformanceRowArtifacts:
        """Assemble all check inputs from one row's collected bundle outputs."""
        observed_at = observed_at or utc_now()
        outputs = state.rows[row.row_id].collected_outputs
        discovered = _discover_conformance_artifacts(outputs)
        run_spec = _row_payload(row)
        preflight_payload = None
        if run_spec is not None:
            try:
                normalized = preflight_training_run_manifest_payloads(
                    run_spec,
                    row_id=row.row_id,
                ).model_dump()
                training_spec = normalized.get("training_spec")
                if isinstance(training_spec, Mapping):
                    inline = training_spec.get("inline")
                    if isinstance(inline, Mapping):
                        preflight_payload = inline
            except Exception:
                # PREFLIGHT owns normalization diagnostics. CERTIFY records the
                # absent normalized input as a failed manifest-valid verdict.
                pass
        return ConformanceRowArtifacts(
            row_id=row.row_id,
            execution=row.execution,
            execution_identity_adapter=self._execution_identity_adapter(),
            schema_registry=(
                self.assembly_context.schema_registry if self.assembly_context is not None else None
            ),
            event_log=self.bundle.run_set_dir / "events" / f"{row.row_id}.events.jsonl",
            row_status=state.rows[row.row_id].status,
            bundle_row_spec=run_spec,
            recorded_environment_fingerprint=state.environment_fingerprint,
            manifest_path=discovered.get("manifest_path"),
            manifest_payload=discovered.get("manifest_payload"),
            training_diagnostics=discovered.get("training_diagnostics"),
            checkpoint_custody_root=discovered.get("checkpoint_custody_root"),
            preflight_normalized_payload=preflight_payload,
            row_state=state.rows.get(row.row_id),
            runtime_inputs=self.row_conformance_inputs.get(row.row_id),
            deployment_policy=self.bundle.deployment_policy.model_dump(mode="json"),
            realized_deployment_evidence=self._realized_deployment_evidence(
                row, state, observed_at=observed_at
            ),
        )

    def _realized_deployment_evidence(
        self,
        row: RunRowSpec,
        state: RunSetState,
        *,
        observed_at: datetime,
    ) -> dict[str, Any]:
        """Project only state already observed before CERTIFY into raw evidence."""
        provision_stage = state.stage(STAGE_PROVISION)
        if (
            provision_stage.status != "completed"
            or state.provision_record is None
            or dict(provision_stage.outputs) != dict(state.provision_record)
        ):
            raise OrchestrationStageError(
                "CERTIFY requires provision_record to exactly match completed PROVISION outputs"
            )
        realize_env_stage = state.stage(STAGE_REALIZE_ENV)
        recorded_fingerprint = realize_env_stage.outputs.get("environment_fingerprint")
        if (
            realize_env_stage.status != "completed"
            or not isinstance(recorded_fingerprint, str)
            or recorded_fingerprint != state.environment_fingerprint
        ):
            raise OrchestrationStageError(
                "CERTIFY requires environment_fingerprint to exactly match completed "
                "REALIZE_ENV outputs"
            )
        provision = dict(state.provision_record or {})
        row_state = state.rows[row.row_id]
        venue = self.bundle.deployment_policy.venue
        unavailable: dict[str, str] = {}

        def absent(field_name: str, reason: str) -> None:
            unavailable[field_name] = reason

        started_at = row_state.started_at
        completed_at = row_state.completed_at
        wall_time = None
        if started_at is not None and completed_at is not None:
            wall_time = (completed_at - started_at).total_seconds()
        else:
            for name, value in (
                ("row_started_at", started_at),
                ("row_completed_at", completed_at),
                ("wall_time_seconds", wall_time),
            ):
                if value is None:
                    absent(name, "row state did not record this timing fact")

        provisioned_at = state.stage(STAGE_PROVISION).completed_at
        if provisioned_at is None:
            absent("provisioned_at", "PROVISION did not record a completion time")
        fingerprint = state.environment_fingerprint
        if fingerprint is None:
            absent("environment_fingerprint", "REALIZE_ENV did not record a fingerprint")

        evidence: dict[str, Any] = {
            "driver": provision.get("driver"),
            "venue": venue,
            "provider": provision.get("provider"),
            "gpu_model": None,
            "gpu_count": None,
            "region": provision.get("region"),
            "immutable_image_id": provision.get("immutable_image_id"),
            "environment_fingerprint": fingerprint,
            "provisioned_at": provisioned_at,
            "billing_started_at": provision.get("billing_started_at"),
            "row_started_at": started_at,
            "row_completed_at": completed_at,
            "observed_at": observed_at,
            "wall_time_seconds": wall_time,
            "hourly_rate": provision.get("hourly_rate"),
            "accrued_cost": None,
            "currency": provision.get("currency"),
            "cost_basis": "local-not-billable"
            if venue == "local"
            else ("billing-start-to-certify-observation"),
            "observation_basis": {
                "provider": provision.get(
                    "provider_observation_basis", "durable orchestration provision record"
                ),
                "environment": "validated REALIZE_ENV fingerprint preserved exactly",
                "timing": "durable PROVISION stage and row-state timestamps",
                "cost": (
                    "local route is non-billable"
                    if venue == "local"
                    else "provider billing start and observed hourly rate through CERTIFY"
                ),
            },
            "provider_observations": {
                "hourly_rate_raw": provision.get("hourly_rate_raw"),
                "immutable_image_id_raw": provision.get("immutable_image_id"),
                "billing_started_at_raw": provision.get(
                    "billing_started_at_raw", provision.get("billing_started_at")
                ),
                "region_raw": provision.get("region"),
            },
            "unavailable": unavailable,
        }
        if venue == "local":
            evidence.update(
                {
                    "provider": "local",
                    "hourly_rate": 0.0,
                    "accrued_cost": 0.0,
                    "currency": "USD",
                }
            )
            for name in ("gpu_model", "gpu_count", "region", "immutable_image_id"):
                absent(name, "not applicable to the local deployment route")
            absent("billing_started_at", "not applicable to the non-billable local route")
            return evidence

        try:
            realized = json.loads(fingerprint or "")
            runtime = realized.get("runtime", {})
            evidence["gpu_model"] = runtime.get("device_kind")
            evidence["gpu_count"] = runtime.get("device_count")
        except (json.JSONDecodeError, AttributeError):
            pass
        for name in ("provider", "gpu_model", "gpu_count", "region", "immutable_image_id"):
            if evidence[name] is None:
                absent(name, "remote observations did not prove this fact")
        billing_started = _parse_observed_datetime(evidence["billing_started_at"])
        hourly_rate = evidence["hourly_rate"]
        if evidence["billing_started_at"] is None:
            absent("billing_started_at", "provider response lacked billing start time")
        if hourly_rate is None:
            absent("hourly_rate", "provider response lacked an observed hourly rate")
        if evidence["currency"] is None:
            absent("currency", "provider response lacked rate currency")
        if (
            billing_started is not None
            and billing_started <= observed_at
            and isinstance(hourly_rate, (int, float))
        ):
            elapsed_hours = (observed_at - billing_started).total_seconds() / 3600.0
            evidence["accrued_cost"] = float(hourly_rate) * elapsed_hours
        else:
            absent(
                "accrued_cost",
                "valid non-future billing start and hourly rate are required for observation",
            )
        return evidence

    def _execution_identity_adapter(self) -> Any:
        if self.request is None or self.assembly_registry is None:
            return None
        return self.assembly_registry.resolve(self.request).identity_adapter

    def _stage_teardown(self, state: RunSetState) -> tuple[RunSetState, Mapping[str, Any]]:
        state = self._run_teardown(state, abort=False)
        return state, state.stage(STAGE_TEARDOWN).outputs

    def _stage_register(self, state: RunSetState) -> tuple[RunSetState, Mapping[str, Any]]:
        if len(self.conformance_registry) == 0:
            raise OrchestrationStageError(
                "REGISTER requires at least one registered conformance check"
            )
        if state.run_set_id != self.bundle.run_set_id:
            raise OrchestrationStageError(
                "REGISTER state run_set_id does not match the assembled bundle"
            )
        final_pod_inventory: Mapping[str, Any] | None = None
        if self.bundle.deployment_policy.driver == "runpod":
            final_pod_inventory = self._require_globally_empty_runpod_inventory(state)
        certify_stage = state.stage(STAGE_CERTIFY)
        certified_ref = certify_stage.outputs.get("certificate_ref")
        if (
            certify_stage.status != "completed"
            or not isinstance(certified_ref, str)
            or state.certificate_ref != certified_ref
        ):
            raise OrchestrationStageError(
                "REGISTER certificate_ref does not match the completed CERTIFY stage"
            )
        stage_inputs = state.stage(STAGE_STAGE_INPUTS)
        certified_stage_inputs_digest = certify_stage.outputs.get("stage_inputs_sha256")
        if stage_inputs.status != "completed":
            raise OrchestrationStageError("REGISTER requires completed STAGE_INPUTS authority")
        stage_inputs_digest = _canonical_json_sha256(stage_inputs.outputs)
        if (
            not isinstance(certified_stage_inputs_digest, str)
            or certified_stage_inputs_digest != stage_inputs_digest
        ):
            raise OrchestrationStageError(
                "REGISTER STAGE_INPUTS digest does not match the completed CERTIFY stage"
            )
        certificate_path = Path(certified_ref)
        certificate_bytes = certificate_path.read_bytes()
        certificate_digest = hashlib.sha256(certificate_bytes).hexdigest()
        certified_digest = state.stage(STAGE_CERTIFY).outputs.get("certificate_sha256")
        if not isinstance(certified_digest, str) or certified_digest != certificate_digest:
            raise OrchestrationStageError(
                "REGISTER certificate digest does not match the completed CERTIFY stage"
            )
        certificate_payload = json.loads(certificate_bytes.decode("utf-8"))
        certificate = RunConformanceCertificate.model_validate(certificate_payload)
        if certificate.run_set_id != self.bundle.run_set_id:
            raise OrchestrationStageError(
                "REGISTER certificate run_set_id does not match the assembled bundle"
            )
        bundle_row_ids = {row.row_id for row in self.bundle.rows}
        state_row_ids = set(state.rows)
        certificate_row_ids = set(certificate.rows)
        if bundle_row_ids != state_row_ids or bundle_row_ids != certificate_row_ids:
            raise OrchestrationStageError(
                "REGISTER requires exact bundle, state, and certificate row-ID equality"
            )
        row_statuses = {row_id: row.status for row_id, row in sorted(state.rows.items())}
        row_status_set = set(row_statuses.values())
        if certificate.overall == "fail":
            status = "failed"
        elif state.abort_reason:
            status = "aborted"
        elif row_status_set == {"completed"}:
            status = "completed"
        elif row_status_set == {"stopped"}:
            status = "stopped"
        else:
            status = "mixed"
        payload = {
            "run_set_id": self.bundle.run_set_id,
            "status": status,
            "abort_reason": state.abort_reason,
            "certificate_ref": str(certificate_path),
            "certificate_sha256": certificate_digest,
            "stage_inputs_sha256": stage_inputs_digest,
            "certificate_overall": certificate.overall,
        }
        if final_pod_inventory is not None:
            payload["final_pod_inventory"] = dict(final_pod_inventory)
        if status == "failed":
            payload["failure_reason"] = "conformance-failed"
        elif status in {"stopped", "mixed"}:
            payload["row_outcomes"] = {
                row_id: {
                    "status": row.status,
                    **({"reason": row.error} if row.error else {}),
                }
                for row_id, row in sorted(state.rows.items())
            }
        self._recover_post_pass_registration_history(state=state, current=payload)
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

    @staticmethod
    def _require_globally_empty_runpod_inventory(
        state: RunSetState,
    ) -> Mapping[str, Any]:
        """Require verified provider-wide RunPod absence before registration."""
        teardown = state.stage(STAGE_TEARDOWN)
        inventory = teardown.outputs.get("final_pod_inventory")
        valid = (
            teardown.status == "completed"
            and isinstance(inventory, Mapping)
            and inventory.get("scope") == "provider-account"
            and inventory.get("verified") is True
            and inventory.get("observation_basis") == "runpodctl pod list --output json"
            and inventory.get("outcome") == "empty"
            and type(inventory.get("pod_count")) is int
            and inventory.get("pod_count") == 0
            and type(inventory.get("pod_ids")) is list
            and inventory.get("pod_ids") == []
            and _parse_observed_datetime(inventory.get("observed_at")) is not None
        )
        if not valid:
            raise OrchestrationStageError(
                "REGISTER requires TEARDOWN evidence proving a globally empty "
                "RunPod provider inventory"
            )
        assert isinstance(inventory, Mapping)
        return inventory

    def _write_or_verify_registration(
        self,
        *,
        register_path: Path,
        certificate_path: Path,
        payload: Mapping[str, Any],
    ) -> None:
        if register_path.exists():
            existing_bytes = register_path.read_bytes()
            existing = json.loads(existing_bytes.decode("utf-8"))
            if existing == dict(payload):
                return
            if self._allows_failed_to_pass_registration_transition(
                existing=existing,
                existing_sha256=hashlib.sha256(existing_bytes).hexdigest(),
                current=dict(payload),
                history_path=register_path.with_name("registration-history.json"),
            ):
                self._write_json_atomically(register_path, dict(payload), replace=True)
                return
            raise OrchestrationStageError(
                "registration payload mismatch at "
                f"{register_path}; existing payload does not match current certificate "
                f"outcome from {certificate_path}: status={payload.get('status')!r}, "
                f"certificate_overall={payload.get('certificate_overall')!r}"
            )

        self._write_json_atomically(register_path, dict(payload))

    def _allows_failed_to_pass_registration_transition(
        self,
        *,
        existing: Mapping[str, Any],
        existing_sha256: str,
        current: Mapping[str, Any],
        history_path: Path,
    ) -> bool:
        """Return whether history authorizes one exact failed-to-completed transition."""
        if not history_path.exists():
            return False
        try:
            history = RegistrationHistory.model_validate_json(
                history_path.read_text(encoding="utf-8")
            )
        except Exception as exc:
            raise OrchestrationStageError("registration history is invalid") from exc
        if history.run_set_id != self.bundle.run_set_id or len(history.entries) != 1:
            return False
        prior = history.entries[0]
        stable_fields = ("run_set_id", "abort_reason", "stage_inputs_sha256")
        return (
            existing == prior.registration_payload
            and existing_sha256 == prior.registration_sha256
            and existing.get("status") == "failed"
            and existing.get("certificate_overall") == "fail"
            and existing.get("certificate_sha256") == prior.certificate_sha256
            and existing.get("certificate_ref") == prior.original_certificate_ref
            and current.get("status") == "completed"
            and current.get("certificate_overall") == "pass"
            and current.get("certificate_ref") == prior.original_certificate_ref
            and current.get("certificate_sha256") != prior.certificate_sha256
            and all(existing.get(field) == current.get(field) for field in stable_fields)
        )

    @staticmethod
    def _write_json_atomically(
        path: Path,
        payload: Mapping[str, Any],
        *,
        replace: bool = False,
    ) -> None:
        if path.exists() and not replace:
            raise OrchestrationStageError(f"refusing to overwrite existing history at {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=str(path.parent),
            text=True,
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(dict(payload), handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def _run_teardown(self, state: RunSetState, *, abort: bool) -> RunSetState:
        stage = state.stage(STAGE_TEARDOWN)
        if stage.status == "completed":
            return state
        failure_log_collection: dict[str, Any] | None = None
        if abort:
            collect_failure_logs = getattr(self.driver, "collect_failure_logs", None)
            if collect_failure_logs is not None:
                try:
                    diagnostic_outputs = dict(collect_failure_logs(self.bundle, state))
                    failure_log_collection = {
                        "status": "completed",
                        "outputs": diagnostic_outputs,
                    }
                except Exception as exc:
                    # Failure diagnostics are best-effort and must never mask
                    # the error that caused abort teardown.
                    failure_log_collection = {"status": "failed", "error": str(exc)}
        if self.bundle.keep_alive:
            describe_ownership = getattr(self.driver, "teardown_ownership", None)
            ownership = dict(describe_ownership(state)) if describe_ownership else {}
            outputs = {
                "teardown": "skipped",
                "skip_reason": "keep_alive",
                **({"ownership": ownership} if ownership else {}),
            }
            status = "completed"
            error = None
        else:
            try:
                outputs = dict(self.driver.teardown(self.bundle, state))
                status = "completed"
                error = None
            except Exception as exc:
                outputs = dict(getattr(exc, "teardown_outputs", {}))
                status = "failed"
                error = str(exc)
        teardown_outputs: dict[str, Any] = {**outputs, "abort_path": abort}
        if failure_log_collection is not None:
            teardown_outputs["failure_log_collection"] = failure_log_collection
        updated = stage.model_copy(
            update={
                "status": status,
                "attempts": stage.attempts + 1,
                "started_at": stage.started_at or utc_now(),
                "completed_at": utc_now(),
                "outputs": teardown_outputs,
                "error": error,
            }
        )
        state = state.with_stage(STAGE_TEARDOWN, updated)
        if status == "completed":
            state = self._project_intent_teardown(state, teardown_outputs)
        self.store.save(state)
        return state

    def _project_intent_teardown(
        self,
        state: RunSetState,
        teardown: Mapping[str, Any],
    ) -> RunSetState:
        """Join verified standard teardown evidence back to its owning intent."""
        absence = teardown.get("pod_absence")
        pod_id = teardown.get("pod_id")
        if not (
            isinstance(absence, Mapping)
            and absence.get("verified") is True
            and isinstance(pod_id, str)
        ):
            return state
        intent_id = (
            state.provision_record.get("intent_id")
            if isinstance(state.provision_record, Mapping)
            else None
        )
        for intent in state.acquisition_intents:
            if intent.intent_id == intent_id or pod_id in intent.pod_ids:
                return self._replace_acquisition_intent(
                    state,
                    intent.intent_id,
                    state="resolved-torn-down",
                    teardown_evidence=[*intent.teardown_evidence, dict(teardown)],
                )
        return state

    def _provision_completed(self, state: RunSetState) -> bool:
        return state.stage(STAGE_PROVISION).status == "completed"

    def _launchable_rows(self, state: RunSetState) -> list[RunRowSpec]:
        return [row for row in self.bundle.rows if state.rows[row.row_id].status == "pending"]

    def _launch_one(self, row: RunRowSpec, state: RunSetState) -> RunSetState:
        try:
            assert_feedbax_revision_pin(self.bundle.feedbax_revision)
        except FeedbaxRevisionError as exc:
            raise OrchestrationStageError(str(exc)) from exc
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
        unfinished = [
            row
            for row in self.bundle.rows
            if state.rows[row.row_id].status not in ("completed", "failed", "stopped")
        ]
        if not unfinished:
            self.store.save(state)
            return state
        probe_rows = getattr(self.driver, "probe_rows", None)
        if callable(probe_rows):
            probes = dict(probe_rows(self.bundle, unfinished, state))
        else:
            probes = {row.row_id: self.driver.probe(self.bundle, row, state) for row in unfinished}
        for row in self.bundle.rows:
            row_state = state.rows[row.row_id]
            if row_state.status in ("completed", "failed", "stopped"):
                continue
            probe = probes[row.row_id]
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
            terminal_event_without_sentinel = any(
                item.get("code") == "terminal_event_without_sentinel"
                for item in reconciled.discrepancies
            )
            status = row_state.status
            completed_at = row_state.completed_at
            error = row_state.error
            if (
                reconciled.status == "completed" and not terminal_event_without_sentinel
            ) or probe.status == "completed":
                terminal_status = (
                    reconciled.terminal_event.payload.get("status")
                    if reconciled.terminal_event is not None
                    else None
                )
                status = "stopped" if terminal_status == "cancelled" else "completed"
                completed_at = completed_at or utc_now()
                if terminal_status == "cancelled":
                    error = "operator-stop-after-checkpoint"
            elif (
                reconciled.status in ("failed", "error") and not terminal_event_without_sentinel
            ) or probe.status == "failed":
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


def _missing_declared_collection_outputs(
    row: RunRowSpec,
    collected: Mapping[str, str],
) -> list[str]:
    expected = {Path(source).name for source in row.launch.collect}
    return sorted(expected - set(collected))


def _verify_collected_native_checkpoint_custody(
    row: RunRowSpec,
    collected: Mapping[str, str],
) -> Mapping[str, Any]:
    """Authenticate the published native checkpoint transaction before teardown."""

    checkpoint_root = Path(collected["checkpoints"])
    authenticated = authenticate_published_checkpoint_custody(checkpoint_root)
    manifest = authenticated.manifest
    manifest_sha256 = authenticated.manifest_sha256
    transaction_root_sha256 = manifest.content_integrity_digest.transaction_root_sha256
    manifest_relative_path = authenticated.parent_ref.uri

    training_manifest = load_manifest_bytes(
        _read_collected_regular_file(
            collected["manifest.json"],
            context=f"collected training manifest for row {row.row_id!r}",
        )
    )
    if not isinstance(training_manifest, TrainingRunManifest):
        raise OrchestrationStageError(
            f"collected native manifest is not a TrainingRunManifest for row {row.row_id!r}"
        )
    if not training_manifest.checkpoint_custody:
        raise OrchestrationStageError(
            f"collected training manifest has no terminal checkpoint for row {row.row_id!r}"
        )
    terminal_ref = training_manifest.checkpoint_custody[-1]
    if not isinstance(terminal_ref, ParentRef):
        raise OrchestrationStageError(
            f"collected training manifest terminal checkpoint is not a ParentRef "
            f"for row {row.row_id!r}"
        )
    expected_terminal_ref = ParentRef(
        kind="TrainingCheckpointTransactionManifest",
        id=manifest.transaction_id,
        role="training_checkpoint_custody",
        uri=manifest_relative_path,
        metadata={"manifest_sha256": manifest_sha256},
    )
    if terminal_ref != expected_terminal_ref:
        raise OrchestrationStageError(
            f"collected checkpoint custody is not the terminal training manifest authority "
            f"for row {row.row_id!r}"
        )

    return {
        "transaction_id": authenticated.manifest.transaction_id,
        "manifest_sha256": authenticated.manifest_sha256,
        "transaction_root_sha256": transaction_root_sha256,
        "slot_names": sorted(authenticated.slot_names),
    }


def _read_collected_regular_file(path: str | Path, *, context: str) -> bytes:
    """Read one collected file without following or accepting a replaced entry."""
    path_obj = Path(path)
    parent_descriptor: int | None = None
    descriptor: int | None = None
    try:
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        directory_flags |= getattr(os, "O_CLOEXEC", 0)
        parent_descriptor = os.open(path_obj.parent, directory_flags)
        before = os.stat(
            path_obj.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if not stat.S_ISREG(before.st_mode):
            raise OrchestrationStageError(f"{context} is not a regular file")
        flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(path_obj.name, flags, dir_fd=parent_descriptor)
        opened = os.fstat(descriptor)
        after_open = os.stat(
            path_obj.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(opened.st_mode)
            or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
            or (after_open.st_dev, after_open.st_ino) != (opened.st_dev, opened.st_ino)
        ):
            raise OrchestrationStageError(f"{context} is not one stable regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        completed = os.fstat(descriptor)
        after_read = os.stat(
            path_obj.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            (completed.st_dev, completed.st_ino) != (opened.st_dev, opened.st_ino)
            or (after_read.st_dev, after_read.st_ino) != (opened.st_dev, opened.st_ino)
            or completed.st_size != opened.st_size
            or completed.st_mtime_ns != opened.st_mtime_ns
        ):
            raise OrchestrationStageError(f"{context} identity changed while reading")
        return b"".join(chunks)
    except OrchestrationStageError:
        raise
    except OSError as exc:
        raise OrchestrationStageError(f"{context} is unsafe or unavailable") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if parent_descriptor is not None:
            os.close(parent_descriptor)


def run_preflight_checks(bundle: RunBundle) -> list[PreflightCheckEntry]:
    """Run static preflight checks without driver calls or resource mutation."""
    return _run_static_preflight_checks(
        bundle,
        include_deployment_policy=True,
        include_manifest_payload_normalization=True,
    )


def run_authority_preflight_checks(bundle: RunBundle) -> list[PreflightCheckEntry]:
    """Run provider-neutral checks for an already assembled authority bundle."""
    return _run_static_preflight_checks(
        bundle,
        include_deployment_policy=False,
        include_manifest_payload_normalization=False,
    )


def _run_static_preflight_checks(
    bundle: RunBundle,
    *,
    include_deployment_policy: bool,
    include_manifest_payload_normalization: bool,
) -> list[PreflightCheckEntry]:
    checks: list[PreflightCheckEntry] = []
    schema_current = bundle.schema_version == RUN_BUNDLE_SCHEMA_VERSION
    checks.append(
        _check(
            "schema-current",
            schema_current,
            detail=(
                None
                if schema_current
                else f"expected {RUN_BUNDLE_SCHEMA_VERSION}, observed {bundle.schema_version}"
            ),
        )
    )
    try:
        observed_revision = assert_feedbax_revision_pin(bundle.feedbax_revision)
    except FeedbaxRevisionError as exc:
        checks.append(_check("feedbax-revision-pin", False, detail=str(exc)))
    else:
        checks.append(_check("feedbax-revision-pin", True, observed=observed_revision))
    checks.append(_check("row-identity", True, observed=[row.row_id for row in bundle.rows]))
    budget_present = bundle.budget.max_wall_clock_seconds > 0
    checks.append(
        _check(
            "budget-presence",
            budget_present,
            detail=None if budget_present else "budget.max_wall_clock_seconds must be positive",
        )
    )
    driver_supported = bundle.deployment_policy.driver in {"local", "worker-http", "runpod"}
    checks.append(
        _check(
            "driver-preconditions",
            driver_supported,
            detail=(
                None
                if driver_supported
                else f"unsupported deployment driver: {bundle.deployment_policy.driver!r}"
            ),
            observed=bundle.deployment_policy.driver,
        )
    )
    env_complete = bool(bundle.environment.python_version)
    checks.append(
        _check(
            "environment-declaration",
            env_complete,
            detail=(
                None
                if env_complete
                else "environment.python_version is required for deterministic realization"
            ),
            observed=bundle.environment.python_version,
        )
    )
    input_failures, input_observed = preflight_resolved_inputs(bundle)
    checks.append(
        _check(
            "input-custody-authority",
            not input_failures,
            detail="; ".join(input_failures) if input_failures else None,
            observed=input_observed or "no-resolved-inputs",
        )
    )

    if include_deployment_policy:
        policy_failures, policy_observed = _preflight_deployment_policy(bundle)
        checks.append(
            _check(
                "deployment-policy",
                not policy_failures,
                detail="; ".join(policy_failures) if policy_failures else None,
                observed=policy_observed,
            )
        )

    output_failures: list[str] = []
    output_observed: dict[str, Any] = {}
    for row in bundle.rows:
        if not uses_registered_native_execution(row):
            continue
        missing = missing_native_training_collection_outputs(row)
        output_observed[row.row_id] = {
            "declared": list(row.launch.collect),
            "required_for_registered_native_training": list(NATIVE_TRAINING_COLLECTION_OUTPUTS),
        }
        if missing:
            output_failures.append(f"{row.row_id}: missing {missing!r}")
    checks.append(
        _check(
            "native-output-custody",
            not output_failures,
            detail="; ".join(output_failures) if output_failures else None,
            observed=output_observed,
        )
    )

    row_payloads: dict[str, dict[str, Any] | None] = {}
    row_payload_errors: dict[str, str] = {}
    for row in bundle.rows:
        try:
            row_payloads[row.row_id] = _row_payload(row)
        except (OSError, ValueError) as exc:
            row_payloads[row.row_id] = None
            row_payload_errors[row.row_id] = str(exc)

    if include_manifest_payload_normalization:
        manifest_failures: list[str] = []
        normalized: dict[str, Any] = {}
        for row in bundle.rows:
            if row.row_id in row_payload_errors:
                manifest_failures.append(f"{row.row_id}: {row_payload_errors[row.row_id]}")
                continue
            run_spec = row_payloads[row.row_id]
            if _is_training_run_payload(run_spec):
                try:
                    payloads = preflight_training_run_manifest_payloads(
                        run_spec,
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
    schedule_failures, schedule_observed, schedule_skips = _preflight_schedule_realization(
        bundle,
        row_payloads=row_payloads,
        row_payload_errors=(row_payload_errors if include_manifest_payload_normalization else {}),
    )
    if not include_manifest_payload_normalization:
        schedule_failures = [
            *(f"{row_id}: {detail}" for row_id, detail in row_payload_errors.items()),
            *schedule_failures,
        ]
    schedule_detail = "; ".join(schedule_failures) if schedule_failures else None
    if schedule_skips:
        skip_detail = "skipped-due-to-dependency: " + ", ".join(schedule_skips)
        schedule_detail = f"{schedule_detail}; {skip_detail}" if schedule_detail else skip_detail
    checks.append(
        _check(
            "schedule-realization",
            not schedule_failures,
            detail=schedule_detail,
            observed=schedule_observed or "no-inline-optimizer-specs",
        )
    )
    return checks


def _preflight_deployment_policy(
    bundle: RunBundle,
) -> tuple[list[str], dict[str, Any]]:
    """Validate only the orchestration-owned deployment policy."""
    policy = bundle.deployment_policy
    failures: list[str] = []
    expected_venue = "local" if policy.driver == "local" else "remote"
    if policy.venue != expected_venue:
        failures.append(
            f"driver {policy.driver!r} requires venue={expected_venue!r}, observed {policy.venue!r}"
        )
    if policy.driver == "runpod" and not policy.cloud_authorized:
        failures.append("runpod deployment requires explicit cloud authorization")
    if policy.review_required and not policy.review_authorized:
        failures.append("required deployment review has not been explicitly authorized")
    return failures, policy.model_dump(mode="json")


def _row_payload(row: RunRowSpec) -> dict[str, Any] | None:
    """Load the registered executable payload without consulting launch metadata."""
    ref = row.execution.payload
    if ref.uri is None:
        return None
    data = Path(ref.uri).read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != ref.sha256:
        raise ValueError(
            f"row {row.row_id!r} executable payload digest mismatch: "
            f"expected={ref.sha256} actual={actual}"
        )
    payload = json.loads(data)
    if not isinstance(payload, dict):
        raise ValueError(f"row {row.row_id!r} executable payload must be a JSON object")
    if (
        payload.get("schema_id") != ref.schema_id
        or payload.get("schema_version") != ref.schema_version
    ):
        raise ValueError(
            f"row {row.row_id!r} executable payload schema does not match its artifact ref"
        )
    return payload


def _is_training_run_payload(payload: Any) -> bool:
    """Return whether generic payload checks apply to a TrainingRunSpec row."""
    return isinstance(payload, Mapping) and payload.get("schema_id") == "feedbax.spec.training_run"


def _preflight_schedule_realization(
    bundle: RunBundle,
    *,
    row_payloads: Mapping[str, dict[str, Any] | None] | None = None,
    row_payload_errors: Mapping[str, str] | None = None,
) -> tuple[list[str], dict[str, Any], list[str]]:
    failures: list[str] = []
    observed: dict[str, Any] = {}
    skipped: list[str] = []
    payloads = row_payloads or {}
    payload_errors = row_payload_errors or {}
    for row in bundle.rows:
        if row.row_id in payload_errors:
            observed[row.row_id] = dependency_skip_observed("manifest-payload-normalization")
            skipped.append(f"{row.row_id} depends on manifest-payload-normalization")
            continue
        if row_payloads is None:
            try:
                run_spec = _row_payload(row)
            except (OSError, ValueError) as exc:
                failures.append(f"{row.row_id}: {exc}")
                continue
        else:
            run_spec = payloads[row.row_id]
        if not _is_training_run_payload(run_spec):
            continue
        optimizer_payloads = _optimizer_payloads(run_spec)
        if not optimizer_payloads:
            failures.append(
                f"{row.row_id}: inline run_spec contains no optimizer at a supported typed path"
            )
            observed[row.row_id] = []
            continue
        row_observed: list[dict[str, Any]] = []
        for index, payload in enumerate(optimizer_payloads):
            try:
                result = _evaluate_optimizer_schedule_at_preflight(row, run_spec, index, payload)
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
    return failures, observed, skipped


def _format_preflight_failures(failed: Sequence[PreflightCheckEntry]) -> str:
    """Render ordered named failures with their actionable details."""
    rendered = [
        f"{check.name}: {check.detail}" if check.detail else check.name for check in failed
    ]
    return f"preflight failed: {'; '.join(rendered)}"


def _evaluate_optimizer_schedule_at_preflight(
    row: RunRowSpec,
    run_spec: Mapping[str, Any],
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
    if optimizer_spec.lr_schedule.kind == "constant":
        optimizer = build_optimizer(
            optimizer_spec,
            schedule_origin_step=0,
            current_step=0,
            optimizer_count_at_current_step=0,
        )
        optimizer.init({"preflight": 0.0})
        return {"optimizer_index": optimizer_index, "scheduled": True, "points": 1}

    expected_context = schedule_eval.require_schedule_context(
        schedule_eval.extract_resume_context(run_spec),
        label="resume_context",
    )
    observed_context = schedule_eval.require_schedule_context(
        schedule_eval.extract_optimizer_build_context(run_spec),
        label="optimizer_build_context",
    )
    run_end_step = _schedule_run_end_step(run_spec, expected_context)
    samples, mismatches = schedule_eval.compare_schedule_samples(
        optimizer_spec,
        expected_context=expected_context,
        observed_context=observed_context,
        run_end_step=run_end_step,
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


def _schedule_run_end_step(
    run_spec: Mapping[str, Any],
    context: schedule_eval.ScheduleEvalContext,
) -> int | None:
    """Return the segment post-completion coordinate when the run depth is declared."""
    for n_batches in (
        _path(run_spec, "training_config", "n_batches"),
        _path(run_spec, "training", "n_batches"),
        _path(run_spec, "n_batches"),
    ):
        if n_batches is not None:
            return context.current_step + int(n_batches)
    return None


def _optimizer_payloads(run_spec: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    payloads: list[Mapping[str, Any]] = []
    for value in (
        _path(run_spec, "optimizer"),
        _path(run_spec, "training", "optimizer"),
        _path(run_spec, "training_config", "optimizer"),
        _path(run_spec, "method_payload", "payload", "optimizer"),
        _path(run_spec, "method_payload", "payload", "training", "optimizer"),
        _path(run_spec, "method_payload", "payload", "controller_optimizer"),
        _path(run_spec, "method_payload", "inline", "payload", "optimizer"),
        _path(run_spec, "method_payload", "inline", "payload", "controller_optimizer"),
    ):
        if isinstance(value, Mapping):
            payloads.append(value)
    return payloads


def _discover_conformance_artifacts(outputs: Mapping[str, str]) -> dict[str, Any]:
    """Discover typed conformance inputs from driver-collected output paths.

    Discovery is deterministic and never decides conformance itself. Missing
    roles remain absent so the registered checks emit explicit failed verdicts.
    """
    result: dict[str, Any] = {}
    candidates: list[Path] = []
    for output_path in sorted(outputs.values()):
        path = Path(output_path)
        if path.is_dir():
            candidates.extend(sorted(path.rglob("*.json")))
        elif path.suffix == ".json":
            candidates.append(path)

    typed_diagnostics: list[Mapping[str, Any]] = []
    legacy_diagnostics: list[Mapping[str, Any]] = []
    for path in sorted(set(candidates)):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, Mapping):
            continue
        if _is_training_manifest(payload) and "manifest_path" not in result:
            result["manifest_path"] = path
            result["manifest_payload"] = payload
        if _is_typed_training_diagnostics(payload):
            typed_diagnostics.append(payload)
        elif _is_legacy_training_diagnostics(payload):
            legacy_diagnostics.append(payload)

    diagnostics = typed_diagnostics or legacy_diagnostics
    if len(diagnostics) == 1:
        result["training_diagnostics"] = diagnostics[0]

    for output_path in sorted(outputs.values()):
        path = Path(output_path)
        roots = [path] if path.is_dir() else [path.parent]
        if path.is_dir():
            roots.extend(latest.parent for latest in sorted(path.rglob("latest.json")))
        for root in roots:
            if (root / "latest.json").is_file() and (root / "manifest.json").is_file():
                result.setdefault("checkpoint_custody_root", root)
                break
    return result


def _is_training_manifest(payload: Mapping[str, Any]) -> bool:
    return payload.get("kind") == "TrainingRunManifest" or (
        "training_spec" in payload and ("summary_metrics" in payload or "provenance" in payload)
    )


def _is_training_diagnostics(payload: Mapping[str, Any]) -> bool:
    return _is_typed_training_diagnostics(payload) or _is_legacy_training_diagnostics(payload)


def _is_legacy_training_diagnostics(payload: Mapping[str, Any]) -> bool:
    if payload.get("kind") == "TrainingRunManifest":
        return False
    if "kind" in payload or "schema_id" in payload or "schema_version" in payload:
        return False
    diagnostic_keys = {
        "checkpoint_coordinates",
        "checkpoint_transactions",
        "learning_rate_trace",
        "lr_trace",
    }
    return bool(diagnostic_keys.intersection(payload))


def _is_typed_training_diagnostics(payload: Mapping[str, Any]) -> bool:
    if payload.get("kind") == "TrainingRunManifest":
        return False
    if (
        payload.get("kind") != "TrainingDiagnostics"
        or payload.get("schema_id") != TRAINING_DIAGNOSTICS_SCHEMA_ID
        or payload.get("schema_version")
        not in {
            TRAINING_DIAGNOSTICS_SCHEMA_VERSION_V2,
            TRAINING_DIAGNOSTICS_SCHEMA_VERSION_V3,
            TRAINING_DIAGNOSTICS_SCHEMA_VERSION,
        }
    ):
        return False
    try:
        TrainingDiagnostics.model_validate(payload)
    except ValueError:
        return False
    return True


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
