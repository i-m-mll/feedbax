"""Feedbax Studio headless training worker FastAPI app."""

from __future__ import annotations

import asyncio
import collections
import hmac
import json
import logging
import os
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import ValidationError as PydanticValidationError

from feedbax.plugins.bootstrap import BootstrapState
from feedbax.plugins.retained import compose_retained_application

from feedbax.studio.schema import validate_task_binding_schema
from feedbax.studio.protocol import infer_task_n_steps
from feedbax.compiler.normalization import (
    normalize_graph_for_studio_authoring,
)
from feedbax.contracts.migrations import migrate_studio_task_binding_spec
from feedbax.contracts.domain import DomainDiagnostic
from feedbax.contracts.graph import GraphSpec, StudioTaskBindingSpec
from feedbax.orchestration.events import (
    RUN_EVENT_TERMINAL_TYPES,
    run_event_from_legacy_worker_event,
    legacy_worker_event_from_run_event,
)
from feedbax.training.retained_diagnostics import (
    GraphCompilationError,
    exception_diagnostic,
    graph_compilation_error,
)
from feedbax.web.worker.diagnostics import (
    model_validation_diagnostics,
    object_required_diagnostic,
)
from feedbax.web.worker.checkpoint import (
    CheckpointCleanupError,
    cleanup_worker_checkpoint,
)
from feedbax.web.worker.identity import require_worker_job_id
from feedbax.web.worker.transport import WORKER_EVENT_CURSOR_EXPIRED


logger = logging.getLogger(__name__)


class WorkerStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    STOPPED = "stopped"
    ERROR = "error"


# Maximum number of events retained per job.
_EVENT_BUFFER_MAX = 1000
_STREAM_KEEPALIVE_SECONDS = 5

# Maximum number of terminal jobs retained for status/manifest lookup.
_TERMINAL_JOB_RETENTION_MAX = 32


@dataclass
class _Job:
    job_id: str
    run_set_id: str
    total_batches: int
    stop_event: threading.Event
    # Parsed training configuration dict passed from the API layer.
    training_config: Optional[Dict[str, Any]] = None
    event_log: collections.deque[dict[str, Any]] = field(
        default_factory=lambda: collections.deque(maxlen=_EVENT_BUFFER_MAX)
    )
    thread: Optional[threading.Thread] = None
    status: WorkerStatus = WorkerStatus.IDLE
    # Spec dicts forwarded from the API layer.
    training_spec: Optional[Dict[str, Any]] = None
    task_spec: Optional[Dict[str, Any]] = None
    task_binding_spec: Optional[Dict[str, Any]] = None
    # Graph spec dict forwarded from the API layer for network param extraction.
    graph_spec: Optional[Dict[str, Any]] = None
    # Path to the serialized checkpoint file after training completes.
    checkpoint_path: Optional[str] = None
    # Cleanup failure retained with the path so eviction can retry safely.
    checkpoint_cleanup_error: Optional[str] = None
    # Active ASGI responses that still require the checkpoint bytes.
    checkpoint_download_leases: int = 0
    # Path/payload for the durable manifest emitted after training completes.
    manifest_path: Optional[str] = None
    manifest_payload: Optional[Dict[str, Any]] = None
    retention_plan_payload: Optional[Dict[str, Any]] = None
    retained_observables_payload: Optional[Dict[str, Any]] = None
    batch: int = 0
    program_step: int | None = None
    last_loss: float = 0.0
    snapshot_interval: int = 100
    terminal_at: Optional[float] = None
    _state_lock: threading.RLock = field(default_factory=threading.RLock)
    _next_seq: int = 0
    _stream_finished: bool = False
    _stream_waiters: dict[asyncio.Event, asyncio.AbstractEventLoop] = field(default_factory=dict)

    def _wake_stream_waiters_locked(self) -> None:
        """Notify each waiting stream once until it reads the log again."""
        for ready, loop in self._stream_waiters.items():
            loop.call_soon_threadsafe(ready.set)
        self._stream_waiters.clear()

    def _oldest_seq_locked(self) -> int:
        return self._next_seq - len(self.event_log)

    def _terminal_event_published_locked(self) -> bool:
        return bool(self.event_log and self.event_log[-1]["type"] in RUN_EVENT_TERMINAL_TYPES)


def _is_terminal_status(status: WorkerStatus) -> bool:
    """Return whether *status* represents a finished registry entry."""
    return status is not WorkerStatus.RUNNING


def _mark_job_terminal(job: _Job, status: WorkerStatus) -> None:
    """Set *job* to a terminal status and record retention ordering metadata."""
    if not _is_terminal_status(status):
        raise ValueError(f"Worker status {status.value!r} is not terminal")
    with job._state_lock:
        job.status = status
        job.terminal_at = time.monotonic()


def _job_status_payload(job: _Job) -> dict[str, Any]:
    """Return the status route payload from one state snapshot."""
    with job._state_lock:
        return {
            "status": job.status,
            "batch": job.batch,
            "program_step": job.program_step,
            "total_batches": job.total_batches,
            "last_loss": job.last_loss,
            "job_id": job.job_id,
            "manifest_path": job.manifest_path,
        }


def _job_checkpoint_payload(job: _Job) -> dict[str, Any]:
    """Return the checkpoint metadata payload from one state snapshot."""
    with job._state_lock:
        return {
            "batch": job.batch,
            "program_step": job.program_step,
            "loss": job.last_loss,
            "weights_available": (
                job.checkpoint_path is not None and job.checkpoint_cleanup_error is None
            ),
            "job_id": job.job_id,
        }


def _cleanup_checkpoint_path(path: str | None) -> None:
    """Remove a managed worker checkpoint tempdir when a job is evicted."""
    cleanup_worker_checkpoint(path, context="worker job eviction cleanup failed")


def _manifest_history_events(job: _Job) -> list[dict[str, Any]]:
    """Return compact event history suitable for a durable JSON artifact."""
    history_types = {
        "training_progress",
        "training_log",
        "training_error",
        "training_complete",
        "training_stopped",
    }
    with job._state_lock:
        events = tuple(job.event_log)
    history = []
    for event in events:
        legacy_event = legacy_worker_event_from_run_event(event)
        if legacy_event.get("type") in history_types:
            history.append(legacy_event)
    return history


def _write_job_manifest(job: _Job) -> None:
    """Write a durable training-run manifest for a completed worker job."""
    try:
        from feedbax.contracts.manifest import write_training_run_manifest

        history_events = _manifest_history_events(job)
        with job._state_lock:
            manifest_kwargs = {
                "job_id": job.job_id,
                "run_set_id": job.run_set_id,
                "total_batches": job.total_batches,
                "training_spec": job.training_spec,
                "task_spec": job.task_spec,
                "task_binding_spec": job.task_binding_spec,
                "graph_spec": job.graph_spec,
                "checkpoint_path": job.checkpoint_path,
                "history_events": history_events,
                "retention_plan": job.retention_plan_payload,
                "retained_observables": job.retained_observables_payload,
                "status": job.status.value,
                "final_loss": job.last_loss,
            }
        manifest, path = write_training_run_manifest(**manifest_kwargs)
        manifest_payload = manifest.model_dump(mode="json", exclude_none=True)
        with job._state_lock:
            job.manifest_path = str(path)
            job.manifest_payload = manifest_payload
            batch = job.batch
        _emit(
            job,
            {
                "type": "training_log",
                "job_id": job.job_id,
                "batch": batch,
                "level": "info",
                "message": "Training manifest saved",
                "manifest_path": str(path),
                "manifest_id": manifest.id,
            },
        )
    except Exception as exc:
        with job._state_lock:
            batch = job.batch
        _emit(
            job,
            {
                "type": "training_log",
                "job_id": job.job_id,
                "batch": batch,
                "level": "warning",
                "message": f"Failed to save training manifest: {exc}",
            },
        )


# ---------------------------------------------------------------------------
# Training configuration extraction
# ---------------------------------------------------------------------------


@dataclass
class _TrainingCfg:
    """Normalized training configuration for _run_training_real."""

    n_batches: int = 2000
    batch_size: int = 128
    learning_rate: float = 1e-3
    grad_clip: float | None = 1.0
    hidden_dim: int = 128
    network_type: str = "gru"
    n_reach_steps: int | None = None
    effort_weight: float = 2.5
    snapshot_interval: int = 100


def _as_mapping(name: str, value: Any) -> Dict[str, Any]:
    """Return *value* as a dict or raise a clear worker-spec error."""
    if not isinstance(value, dict):
        raise GraphCompilationError(
            "Invalid training worker request",
            [object_required_diagnostic(name)],
        )
    return value


def _require_worker_specs(job: _Job, bootstrap_state: BootstrapState) -> None:
    """Validate the Studio payload shape required by the real worker path."""
    _as_mapping("training_spec", job.training_spec)
    _as_mapping("task_spec", job.task_spec)
    try:
        graph_spec = normalize_graph_for_studio_authoring(
            GraphSpec.model_validate(_as_mapping("graph_spec", job.graph_spec)),
            component_registry=bootstrap_state.bundle.components,
        )
    except PydanticValidationError as exc:
        raise GraphCompilationError(
            "Invalid graph_spec",
            model_validation_diagnostics("graph_spec", exc),
        ) from exc
    job.graph_spec = graph_spec.model_dump(mode="json", exclude_none=True)
    if job.task_binding_spec is None:
        raise GraphCompilationError(
            "Training worker requires task_binding_spec",
            [
                DomainDiagnostic(
                    severity="error",
                    code="worker.missing_task_binding_spec",
                    message=(
                        "Training worker requires scenario-owned task_binding_spec; "
                        "task data bindings must not be inferred from graph task nodes"
                    ),
                    location={"path": "/task_binding_spec"},
                )
            ],
        )
    task_binding_spec = _as_mapping("task_binding_spec", job.task_binding_spec)
    try:
        migrated_spec = migrate_studio_task_binding_spec(task_binding_spec).payload
        binding_spec = StudioTaskBindingSpec.model_validate(migrated_spec)
    except PydanticValidationError as exc:
        raise GraphCompilationError(
            "Invalid task_binding_spec",
            model_validation_diagnostics("task_binding_spec", exc),
        ) from exc
    except ValueError as exc:
        raise GraphCompilationError(
            "Invalid task_binding_spec",
            [
                exception_diagnostic(
                    exc,
                    code="binding.invalid_spec",
                    message=f"Invalid task_binding_spec: {exc}",
                    details={"path": "/task_binding_spec"},
                )
            ],
        ) from exc
    job.task_binding_spec = binding_spec.model_dump(mode="json", exclude_none=True)
    issues = [
        issue
        for issue in validate_task_binding_schema(
            binding_spec,
            graph_spec,
            "/task_binding_spec",
            component_registry=bootstrap_state.bundle.components,
        )
        if issue.severity == "error" or issue.type == "task_binding_unknown_schema"
    ]
    if issues:
        raise graph_compilation_error(
            "Invalid task_binding_spec for graph_spec",
            issues,
        )


def _extract_training_cfg(
    training_config: Optional[Dict[str, Any]],
    task_spec: Optional[Dict[str, Any]] = None,
) -> _TrainingCfg:
    """Parse a raw config dict into a normalized _TrainingCfg.

    Parses optional worker controls while leaving authored model facts absent.

    Args:
        training_config: Optional dict from the ``/start`` request body.
        task_spec: Optional task spec dict; overrides task params such as
            ``n_reach_steps`` and ``effort_weight`` when present.

    Returns:
        A normalized config. ``n_reach_steps`` remains absent unless the caller
        or task authored a rollout length.
    """
    cfg = _TrainingCfg()
    if training_config is None and task_spec is None:
        return cfg

    if training_config is not None:

        def _get(key: str, default, cast=None, *, allow_none: bool = False):
            if key not in training_config:
                return default
            val = training_config[key]
            if val is None:
                return None if allow_none else default
            try:
                return cast(val) if cast is not None else val
            except (TypeError, ValueError):
                return default

        cfg.n_batches = _get("n_batches", cfg.n_batches, int)
        cfg.batch_size = _get("batch_size", cfg.batch_size, int)
        cfg.learning_rate = _get("learning_rate", cfg.learning_rate, float)
        cfg.grad_clip = _get("grad_clip", cfg.grad_clip, float, allow_none=True)
        cfg.hidden_dim = _get("hidden_dim", cfg.hidden_dim, int)
        cfg.network_type = _get("network_type", cfg.network_type, str)
        cfg.n_reach_steps = _get("n_reach_steps", cfg.n_reach_steps, int)
        cfg.effort_weight = _get("effort_weight", cfg.effort_weight, float)
        cfg.snapshot_interval = _get("snapshot_interval", cfg.snapshot_interval, int)

    n_steps = infer_task_n_steps(task_spec)
    if n_steps is not None:
        cfg.n_reach_steps = n_steps

    if task_spec is not None:
        task_params = task_spec.get("params", {})
        for key, attr, cast in [
            ("effort_weight", "effort_weight", float),
        ]:
            if key in task_params:
                try:
                    setattr(cfg, attr, cast(task_params[key]))
                except (TypeError, ValueError):
                    pass

    return cfg


def _run_training_real(job: _Job, cfg: "_TrainingCfg", bootstrap_state: BootstrapState) -> None:
    """Real JAX training loop over the serialized graph boundary.

    Leaf components remain opaque executable components; the worker no longer
    reconstructs a hidden controller/plant/loss bridge from topology.
    """
    from feedbax.training.retained_graph import compile_training_run
    from feedbax.web.worker.execution import run_training_graph

    compiled = compile_training_run(
        graph_spec=_as_mapping("graph_spec", job.graph_spec),
        training_spec=_as_mapping("training_spec", job.training_spec),
        task_spec=_as_mapping("task_spec", job.task_spec),
        task_binding_spec=_as_mapping("task_binding_spec", job.task_binding_spec),
        cfg=cfg,
        component_registry=bootstrap_state.bundle.components,
    )
    result = run_training_graph(
        compiled,
        job_id=job.job_id,
        total_batches=job.total_batches,
        cfg=cfg,
        stop_event=job.stop_event,
        emit=lambda event: _emit(job, event),
    )
    try:
        with job._state_lock:
            job.last_loss = result.final_loss
            job.batch = result.final_batch
            job.program_step = result.final_program_step
            job.retention_plan_payload = result.retention_plan
            job.retained_observables_payload = result.retained_observables
            job.manifest_path = result.manifest_path
            job.manifest_payload = result.manifest_payload
            job.checkpoint_path = result.checkpoint_path
            job.checkpoint_cleanup_error = None
            batch = job.batch
            last_loss = job.last_loss
    except BaseException as exc:
        try:
            cleanup_worker_checkpoint(
                result.checkpoint_path,
                context="worker checkpoint ownership handoff failed",
            )
        except CheckpointCleanupError as cleanup_exc:
            with job._state_lock:
                job.checkpoint_path = cleanup_exc.checkpoint_path
                job.checkpoint_cleanup_error = str(cleanup_exc)
            raise cleanup_exc from exc
        raise
    if result.checkpoint_path is not None:
        _emit(
            job,
            {
                "type": "training_log",
                "job_id": job.job_id,
                "batch": batch,
                "level": "info",
                "message": "Checkpoint saved",
                "execution": "contract_executor",
            },
        )
    manifest_path = result.manifest_path
    manifest_payload = result.manifest_payload
    if manifest_path is not None:
        _emit(
            job,
            {
                "type": "training_log",
                "job_id": job.job_id,
                "batch": batch,
                "level": "info",
                "message": "Training manifest saved",
                "manifest_path": manifest_path,
                "manifest_id": manifest_payload.get("id") if manifest_payload else None,
                "execution": "contract_executor",
            },
        )
    terminal_event = {
        "type": "training_stopped" if result.stopped else "training_complete",
        "job_id": job.job_id,
        "batch": batch,
        "program_step": result.final_program_step,
        "loss": last_loss,
        "execution": "contract_executor",
    }
    if manifest_path is not None:
        terminal_event["manifest_path"] = manifest_path
    if manifest_payload is not None:
        terminal_event["manifest_id"] = manifest_payload.get("id")
    with job._state_lock:
        _emit(job, terminal_event)
        _mark_job_terminal(job, WorkerStatus.STOPPED if result.stopped else WorkerStatus.COMPLETED)


def _run_training(job: _Job, bootstrap_state: BootstrapState) -> None:
    """Training entry point. Always attempts real JAX training.

    Invalid Studio payloads terminate with a ``training_error`` event instead
    of falling through to synthetic output.
    """
    try:
        _require_worker_specs(job, bootstrap_state)
        cfg = _extract_training_cfg(job.training_config, job.task_spec)
        _run_training_real(job, cfg, bootstrap_state)
    except GraphCompilationError as exc:
        with job._state_lock:
            should_emit = job.status == WorkerStatus.RUNNING
            if should_emit:
                job.status = WorkerStatus.ERROR
                job.terminal_at = time.monotonic()
                batch = job.batch
        if should_emit:
            _emit(
                job,
                {
                    "type": "training_error",
                    "job_id": job.job_id,
                    "batch": batch,
                    "error": str(exc),
                    "diagnostics": [
                        diagnostic.model_dump(mode="json", exclude_none=True)
                        for diagnostic in exc.diagnostics
                    ],
                },
            )
    except Exception as exc:
        logger.exception("Unhandled worker exception while running training job %s", job.job_id)
        with job._state_lock:
            if isinstance(exc, CheckpointCleanupError):
                job.checkpoint_path = exc.checkpoint_path
                job.checkpoint_cleanup_error = str(exc)
            should_emit = job.status == WorkerStatus.RUNNING
            if should_emit:
                job.status = WorkerStatus.ERROR
                job.terminal_at = time.monotonic()
                batch = job.batch
        if should_emit:
            diagnostic = exception_diagnostic(
                exc,
                code="internal",
                message=f"{type(exc).__name__}: {exc}",
            )
            _emit(
                job,
                {
                    "type": "training_error",
                    "job_id": job.job_id,
                    "batch": batch,
                    "error": "Training failed unexpectedly",
                    "diagnostics": [diagnostic.model_dump(mode="json", exclude_none=True)],
                },
            )


def _emit(job: _Job, event: dict) -> None:
    """Publish a RunEvent to the job's retained log."""
    with job._state_lock:
        if job._stream_finished or job._terminal_event_published_locked():
            raise RuntimeError(f"Cannot publish to finished worker job {job.job_id}")
        seq = job._next_seq
        emitted_at_ms = int(time.time() * 1000)
        run_event = run_event_from_legacy_worker_event(
            event,
            run_set_id=job.run_set_id,
            row_id=job.job_id,
            seq=seq,
            emitted_at_ms=emitted_at_ms,
        ).model_dump(mode="json", exclude_none=True)
        if event.get("type") == "training_progress":
            if (batch := event.get("batch")) is not None:
                job.batch = int(batch)
            if (program_step := event.get("program_step")) is not None:
                job.program_step = int(program_step)
            if "loss" in event:
                job.last_loss = float(event["loss"])
        job.event_log.append(run_event)
        job._next_seq += 1
        job._wake_stream_waiters_locked()


def create_app(
    auth_token: Optional[str] = None, *, bootstrap_modules: tuple[str, ...] = ()
) -> FastAPI:
    """Create and return the worker FastAPI application.

    Args:
        auth_token: Optional shared secret. When provided every request must
            include ``Authorization: Bearer <token>``; requests without it
            receive HTTP 401.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.bootstrap_state = await compose_retained_application(modules=bootstrap_modules)
        yield

    app = FastAPI(title="Feedbax Training Worker", version="0.1.0", lifespan=lifespan)

    # ------------------------------------------------------------------
    # Auth dependency
    # ------------------------------------------------------------------

    _bearer_scheme = HTTPBearer(auto_error=False)

    def _require_auth(
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
    ) -> None:
        """FastAPI dependency that enforces the bearer token when one is configured."""
        if auth_token is None:
            # Auth not configured — allow all requests.
            return
        if credentials is None or not hmac.compare_digest(credentials.credentials, auth_token):
            raise HTTPException(status_code=401, detail="Unauthorized")

    # All routes share this dependency.
    _auth_dep = Depends(_require_auth)

    # ------------------------------------------------------------------
    # In-process job registry.
    # ------------------------------------------------------------------

    _jobs: Dict[str, _Job] = {}
    _jobs_lock = threading.Lock()

    def _evict_terminal_jobs_locked() -> None:
        """Drop oldest terminal jobs beyond the bounded retention window."""
        retained = max(0, _TERMINAL_JOB_RETENTION_MAX)
        terminal_jobs: list[tuple[float, str]] = []
        for job_id, job in _jobs.items():
            with job._state_lock:
                if job._stream_finished and _is_terminal_status(job.status):
                    terminal_jobs.append((job.terminal_at or 0.0, job_id))
        overflow = len(terminal_jobs) - retained
        if overflow <= 0:
            return
        for _, job_id in sorted(terminal_jobs)[:overflow]:
            job = _jobs[job_id]
            with job._state_lock:
                checkpoint_path = job.checkpoint_path
                if job.checkpoint_download_leases > 0:
                    continue
            try:
                _cleanup_checkpoint_path(checkpoint_path)
            except CheckpointCleanupError as exc:
                with job._state_lock:
                    job.checkpoint_cleanup_error = str(exc)
                raise
            if _jobs.get(job_id) is job:
                del _jobs[job_id]

    def _running_job_id_locked() -> Optional[str]:
        """Return the active job id if the single-worker slot is occupied."""
        for job in _jobs.values():
            with job._state_lock:
                if not job._stream_finished:
                    return job.job_id
        return None

    def _get_job(job_id: str) -> _Job:
        with _jobs_lock:
            job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return job

    async def _checkpoint_download_lease(
        job_id: str,
    ) -> AsyncIterator[tuple[str, str]]:
        """Hold checkpoint eviction custody through the request response lifecycle."""
        with _jobs_lock:
            job = _jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Job not found")
            with job._state_lock:
                checkpoint_path = job.checkpoint_path
                if job.checkpoint_cleanup_error is not None:
                    raise HTTPException(status_code=409, detail="Checkpoint cleanup pending")
                if checkpoint_path is None:
                    raise HTTPException(status_code=404, detail="No checkpoint available")
                if not os.path.exists(checkpoint_path):
                    raise HTTPException(status_code=410, detail="Checkpoint file gone")
                job.checkpoint_download_leases += 1

        try:
            yield checkpoint_path, job.job_id
        finally:
            with _jobs_lock:
                with job._state_lock:
                    job.checkpoint_download_leases -= 1
                try:
                    _evict_terminal_jobs_locked()
                except CheckpointCleanupError:
                    logger.exception("worker job eviction cleanup failed after checkpoint download")

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.get("/health", dependencies=[_auth_dep])
    def health():
        return {"status": "ok"}

    @app.post("/start", dependencies=[_auth_dep])
    def start(body: dict):
        total_batches = int(body.get("total_batches", 100))
        training_config: Optional[Dict[str, Any]] = body.get("training_config", None)
        training_spec: Optional[Dict[str, Any]] = body.get("training_spec", None)
        task_spec: Optional[Dict[str, Any]] = body.get("task_spec", None)
        task_binding_spec: Optional[Dict[str, Any]] = body.get("task_binding_spec", None)
        graph_spec: Optional[Dict[str, Any]] = body.get("graph_spec", None)
        snapshot_interval = int(body.get("snapshot_interval", 100))

        job_id_raw = body.get("job_id")
        run_set_id_raw = body.get("run_set_id")
        try:
            job_id = require_worker_job_id(job_id_raw)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not isinstance(run_set_id_raw, str) or not run_set_id_raw.strip():
            raise HTTPException(status_code=400, detail="start request requires run_set_id")
        run_set_id = run_set_id_raw.strip()
        stop_event = threading.Event()

        job = _Job(
            job_id=job_id,
            run_set_id=run_set_id,
            total_batches=total_batches,
            stop_event=stop_event,
            training_config=training_config,
            training_spec=training_spec,
            task_spec=task_spec,
            task_binding_spec=task_binding_spec,
            graph_spec=graph_spec,
            status=WorkerStatus.RUNNING,
            snapshot_interval=snapshot_interval,
        )

        def _run_training_and_evict() -> None:
            try:
                _run_training(job, app.state.bootstrap_state)
            finally:
                with job._state_lock:
                    job._stream_finished = True
                    job._wake_stream_waiters_locked()
                    if _is_terminal_status(job.status) and job.terminal_at is None:
                        job.terminal_at = time.monotonic()
                with _jobs_lock:
                    _evict_terminal_jobs_locked()

        thread = threading.Thread(target=_run_training_and_evict, daemon=True)
        job.thread = thread
        with _jobs_lock:
            _evict_terminal_jobs_locked()
            if job_id in _jobs:
                raise HTTPException(status_code=409, detail=f"Job {job_id} already exists")
            running_job_id = _running_job_id_locked()
            if running_job_id is not None:
                raise HTTPException(
                    status_code=409,
                    detail=f"Worker is already running job {running_job_id}",
                )
            _jobs[job_id] = job
        try:
            thread.start()
        except BaseException:
            with _jobs_lock:
                if _jobs.get(job_id) is job:
                    del _jobs[job_id]
                with job._state_lock:
                    job._stream_finished = True
                    job._wake_stream_waiters_locked()
            raise
        return {"job_id": job_id}

    @app.post("/jobs/{job_id}/stop", dependencies=[_auth_dep])
    def stop(job_id: str):
        job = _get_job(job_id)
        with job._state_lock:
            if job.status is WorkerStatus.RUNNING:
                job.stop_event.set()
        return {"ok": True}

    @app.get("/jobs/{job_id}/status", dependencies=[_auth_dep])
    def status(job_id: str):
        job = _get_job(job_id)
        return _job_status_payload(job)

    @app.get("/jobs/{job_id}/stream", dependencies=[_auth_dep])
    async def stream(job_id: str, from_seq: Optional[int] = Query(default=None, ge=0)) -> Response:
        """Read a job's SSE log from the inclusive cursor or oldest retained event."""
        job = _get_job(job_id)

        def cursor_expired(requested_seq: int, oldest_seq: int) -> dict[str, Any]:
            return {
                "code": WORKER_EVENT_CURSOR_EXPIRED,
                "requested_seq": requested_seq,
                "oldest_seq": oldest_seq,
            }

        with job._state_lock:
            oldest_seq = job._oldest_seq_locked()
            cursor = oldest_seq if from_seq is None else from_seq
            if cursor < oldest_seq:
                raise HTTPException(status_code=410, detail=cursor_expired(cursor, oldest_seq))
            if cursor > job._next_seq:
                raise HTTPException(status_code=422, detail="from_seq is ahead of the event log")
            if cursor == job._next_seq and job._terminal_event_published_locked():
                return Response(status_code=204)

        async def _generate() -> AsyncIterator[str]:
            loop = asyncio.get_running_loop()
            ready = asyncio.Event()
            next_seq = cursor

            while True:
                event = None
                with job._state_lock:
                    oldest_seq = job._oldest_seq_locked()
                    if oldest_seq <= next_seq < job._next_seq:
                        event = job.event_log[next_seq - oldest_seq]
                    elif next_seq == job._next_seq:
                        if job._stream_finished or job._terminal_event_published_locked():
                            return
                        ready.clear()
                        job._stream_waiters[ready] = loop

                if next_seq < oldest_seq:
                    detail = cursor_expired(next_seq, oldest_seq)
                    yield f"event: stream_gap\ndata: {json.dumps(detail)}\n\n"
                    return
                if event is None:
                    idle = False
                    try:
                        async with asyncio.timeout(_STREAM_KEEPALIVE_SECONDS):
                            await ready.wait()
                    except TimeoutError:
                        idle = True
                    finally:
                        with job._state_lock:
                            job._stream_waiters.pop(ready, None)
                    if idle:
                        yield ": keep-alive\n\n"
                    continue
                next_seq += 1
                yield f"data: {json.dumps(event)}\n\n"
                if event["type"] in RUN_EVENT_TERMINAL_TYPES:
                    return

        return StreamingResponse(_generate(), media_type="text/event-stream")

    @app.get("/jobs/{job_id}/checkpoint", dependencies=[_auth_dep])
    def checkpoint(job_id: str):
        """Return checkpoint metadata for a job."""
        job = _get_job(job_id)
        return _job_checkpoint_payload(job)

    @app.get("/jobs/{job_id}/checkpoint/download", dependencies=[_auth_dep])
    async def checkpoint_download(
        lease: tuple[str, str] = Depends(_checkpoint_download_lease),
    ):
        """Download the serialized checkpoint file for a job."""
        checkpoint_path, job_id = lease
        return FileResponse(
            checkpoint_path,
            media_type="application/octet-stream",
            filename=f"feedbax_checkpoint_{job_id}.eqx",
        )

    @app.get("/jobs/{job_id}/manifest", dependencies=[_auth_dep])
    def manifest(job_id: str):
        """Return the durable manifest for a job."""
        job = _get_job(job_id)
        with job._state_lock:
            manifest_payload = job.manifest_payload
        if manifest_payload is None:
            raise HTTPException(status_code=404, detail="No manifest available")
        return manifest_payload

    return app
