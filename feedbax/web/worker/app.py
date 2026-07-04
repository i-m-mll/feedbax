"""Feedbax Studio headless training worker FastAPI app."""

from __future__ import annotations

import asyncio
import collections
import json
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, Optional, Tuple

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from feedbax.studio.schema import validate_task_binding_schema
from feedbax.studio.protocol import infer_task_n_steps
from feedbax.contracts.graphs.normalization import (
    normalize_graph_for_studio_authoring,
    normalize_task_binding_spec_for_studio_authoring,
)
from feedbax.contracts.migrations import migrate_studio_task_binding_spec
from feedbax.contracts.graph import GraphSpec, StudioTaskBindingSpec


class WorkerStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"


# Maximum number of past events to buffer per job for from_seq replay.
_EVENT_BUFFER_MAX = 1000

# Maximum number of terminal jobs retained for status/manifest lookup.
_TERMINAL_JOB_RETENTION_MAX = 32


@dataclass
class _Job:
    job_id: str
    total_batches: int
    event_queue: queue.Queue
    stop_event: threading.Event
    # Parsed training configuration dict passed from the API layer.
    training_config: Optional[Dict[str, Any]] = None
    # Buffer of (seq, event_dict) for replay support.
    event_buffer: Deque[Tuple[int, dict]] = field(
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
    # Path/payload for the durable manifest emitted after training completes.
    manifest_path: Optional[str] = None
    manifest_payload: Optional[Dict[str, Any]] = None
    retention_plan_payload: Optional[Dict[str, Any]] = None
    retained_observables_payload: Optional[Dict[str, Any]] = None
    batch: int = 0
    last_loss: float = 0.0
    snapshot_interval: int = 100
    terminal_at: Optional[float] = None
    _state_lock: threading.RLock = field(default_factory=threading.RLock)
    # Monotonically increasing sequence counter; protected by _seq_lock.
    _seq: int = 0
    _seq_lock: threading.Lock = field(default_factory=threading.Lock)

    def next_seq(self) -> int:
        """Return the next sequence number and advance the counter."""
        with self._seq_lock:
            seq = self._seq
            self._seq += 1
            return seq


def _is_terminal_status(status: WorkerStatus) -> bool:
    """Return whether *status* represents a finished registry entry."""
    return status in {WorkerStatus.IDLE, WorkerStatus.COMPLETED, WorkerStatus.ERROR}


def _mark_job_terminal(job: _Job, status: WorkerStatus) -> None:
    """Set *job* to a terminal status and record retention ordering metadata."""
    if not _is_terminal_status(status):
        raise ValueError(f"Worker status {status.value!r} is not terminal")
    with job._state_lock:
        job.status = status
        job.terminal_at = time.monotonic()


def _job_status(job: _Job) -> WorkerStatus:
    """Return a consistent status value for *job*."""
    with job._state_lock:
        return job.status


def _job_status_payload(job: _Job) -> dict[str, Any]:
    """Return the status route payload from one state snapshot."""
    with job._state_lock:
        return {
            "status": job.status,
            "batch": job.batch,
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
            "loss": job.last_loss,
            "weights_available": job.checkpoint_path is not None,
            "job_id": job.job_id,
        }


def _manifest_history_events(job: _Job) -> list[dict[str, Any]]:
    """Return compact event history suitable for a durable JSON artifact."""
    history_types = {"training_progress", "training_log", "training_error", "training_complete"}
    with job._state_lock:
        return [dict(event) for _, event in job.event_buffer if event.get("type") in history_types]


def _write_job_manifest(job: _Job) -> None:
    """Write a durable training-run manifest for a completed worker job."""
    try:
        from feedbax.contracts.manifest import write_training_run_manifest

        history_events = _manifest_history_events(job)
        with job._state_lock:
            manifest_kwargs = {
                "job_id": job.job_id,
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
    n_reach_steps: int = 80
    effort_weight: float = 2.5
    snapshot_interval: int = 100


def _as_mapping(name: str, value: Any) -> Dict[str, Any]:
    """Return *value* as a dict or raise a clear worker-spec error."""
    if not isinstance(value, dict):
        raise ValueError(f"Training worker requires {name} to be an object")
    return value


def _require_worker_specs(job: _Job) -> None:
    """Validate the Studio payload shape required by the real worker path."""
    _as_mapping("training_spec", job.training_spec)
    _as_mapping("task_spec", job.task_spec)
    graph_spec = normalize_graph_for_studio_authoring(
        GraphSpec.model_validate(_as_mapping("graph_spec", job.graph_spec))
    )
    job.graph_spec = graph_spec.model_dump(mode="json", exclude_none=True)
    if job.task_binding_spec is None:
        raise ValueError(
            "Training worker requires scenario-owned task_binding_spec; "
            "task data bindings must not be inferred from graph task nodes"
        )
    task_binding_spec = _as_mapping("task_binding_spec", job.task_binding_spec)
    try:
        migrated_spec = migrate_studio_task_binding_spec(task_binding_spec).payload
        binding_spec = StudioTaskBindingSpec.model_validate(migrated_spec)
    except ValueError as exc:
        raise ValueError(f"Invalid task_binding_spec: {exc}") from exc
    binding_spec = normalize_task_binding_spec_for_studio_authoring(binding_spec, graph_spec)
    job.task_binding_spec = binding_spec.model_dump(mode="json", exclude_none=True)
    issues = validate_task_binding_schema(binding_spec, graph_spec, "/task_binding_spec")
    if issues:
        summary = "; ".join(f"{issue.type}: {issue.message}" for issue in issues)
        raise ValueError(f"Invalid task_binding_spec for graph_spec: {summary}")


def _extract_training_cfg(
    training_config: Optional[Dict[str, Any]],
    task_spec: Optional[Dict[str, Any]] = None,
) -> _TrainingCfg:
    """Parse a raw config dict into a normalized _TrainingCfg.

    Falls back to defaults for any missing or invalid field.

    Args:
        training_config: Optional dict from the ``/start`` request body.
        task_spec: Optional task spec dict; overrides task params such as
            ``n_reach_steps`` and ``effort_weight`` when present.

    Returns:
        A _TrainingCfg with all fields populated.
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


def _run_training_real(job: _Job, cfg: "_TrainingCfg") -> None:
    """Real JAX training loop over the serialized graph boundary.

    Leaf components remain opaque executable components; the worker no longer
    reconstructs a hidden controller/plant/loss bridge from topology.
    """
    from feedbax.web.worker.execution import compile_training_run, run_training_graph

    compiled = compile_training_run(
        graph_spec=_as_mapping("graph_spec", job.graph_spec),
        training_spec=_as_mapping("training_spec", job.training_spec),
        task_spec=_as_mapping("task_spec", job.task_spec),
        task_binding_spec=_as_mapping("task_binding_spec", job.task_binding_spec),
        cfg=cfg,
    )
    result = run_training_graph(
        compiled,
        job_id=job.job_id,
        total_batches=job.total_batches,
        cfg=cfg,
        stop_event=job.stop_event,
        emit=lambda event: _emit(job, event),
    )
    terminal_status = WorkerStatus.IDLE if job.stop_event.is_set() else WorkerStatus.COMPLETED
    with job._state_lock:
        job.last_loss = result.final_loss
        job.batch = job.total_batches
        job.checkpoint_path = result.checkpoint_path
        job.retention_plan_payload = result.retention_plan
        job.retained_observables_payload = result.retained_observables
        job.status = terminal_status
        job.terminal_at = time.monotonic()
        batch = job.batch
        last_loss = job.last_loss
    if result.checkpoint_path is not None:
        _emit(
            job,
            {
                "type": "training_log",
                "job_id": job.job_id,
                "batch": batch,
                "level": "info",
                "message": "Checkpoint saved",
                "execution": "generic_graph",
            },
        )
    _write_job_manifest(job)
    with job._state_lock:
        manifest_path = job.manifest_path
        manifest_payload = job.manifest_payload
    complete_event = {
        "type": "training_complete",
        "job_id": job.job_id,
        "batch": batch,
        "loss": last_loss,
        "execution": "generic_graph",
    }
    if manifest_path is not None:
        complete_event["manifest_path"] = manifest_path
    if manifest_payload is not None:
        complete_event["manifest_id"] = manifest_payload.get("id")
    _emit(job, complete_event)


def _run_training(job: _Job) -> None:
    """Training entry point. Always attempts real JAX training.

    Invalid Studio payloads terminate with a ``training_error`` event instead
    of falling through to synthetic output.
    """
    try:
        _require_worker_specs(job)
        cfg = _extract_training_cfg(job.training_config, job.task_spec)
        _run_training_real(job, cfg)
    except Exception as exc:
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
                },
            )
    finally:
        # Sentinel: tells SSE generator the stream is done.
        job.event_queue.put(None)


def _emit(job: _Job, event: dict) -> None:
    """Assign a seq number to *event*, buffer it, and enqueue it for SSE delivery."""
    seq = job.next_seq()
    event["seq"] = seq
    with job._state_lock:
        job.event_buffer.append((seq, event))
    job.event_queue.put(event)


def create_app(auth_token: Optional[str] = None) -> FastAPI:
    """Create and return the worker FastAPI application.

    Args:
        auth_token: Optional shared secret. When provided every request must
            include ``Authorization: Bearer <token>``; requests without it
            receive HTTP 401.
    """
    app = FastAPI(title="Feedbax Training Worker", version="0.1.0")

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
        if credentials is None or credentials.credentials != auth_token:
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
                if _is_terminal_status(job.status):
                    terminal_jobs.append((job.terminal_at or 0.0, job_id))
        overflow = len(terminal_jobs) - retained
        if overflow <= 0:
            return
        for _, job_id in sorted(terminal_jobs)[:overflow]:
            del _jobs[job_id]

    def _running_job_id_locked() -> Optional[str]:
        """Return the active job id if the single-worker slot is occupied."""
        for job in _jobs.values():
            if _job_status(job) == WorkerStatus.RUNNING:
                return job.job_id
        return None

    def _get_job(job_id: str) -> _Job:
        with _jobs_lock:
            job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return job

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

        job_id = str(uuid.uuid4())
        stop_event = threading.Event()
        event_queue: queue.Queue = queue.Queue()

        job = _Job(
            job_id=job_id,
            total_batches=total_batches,
            event_queue=event_queue,
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
                _run_training(job)
            finally:
                with job._state_lock:
                    if _is_terminal_status(job.status) and job.terminal_at is None:
                        job.terminal_at = time.monotonic()
                with _jobs_lock:
                    _evict_terminal_jobs_locked()

        thread = threading.Thread(target=_run_training_and_evict, daemon=True)
        job.thread = thread
        with _jobs_lock:
            _evict_terminal_jobs_locked()
            running_job_id = _running_job_id_locked()
            if running_job_id is not None:
                raise HTTPException(
                    status_code=409,
                    detail=f"Worker is already running job {running_job_id}",
                )
            _jobs[job_id] = job
        thread.start()
        return {"job_id": job_id}

    @app.post("/jobs/{job_id}/stop", dependencies=[_auth_dep])
    def stop(job_id: str):
        job = _get_job(job_id)
        job.stop_event.set()
        with job._state_lock:
            thread = job.thread
        if thread is None or not thread.is_alive():
            _mark_job_terminal(job, WorkerStatus.IDLE)
            with _jobs_lock:
                _evict_terminal_jobs_locked()
        return {"ok": True}

    @app.get("/jobs/{job_id}/status", dependencies=[_auth_dep])
    def status(job_id: str):
        job = _get_job(job_id)
        return _job_status_payload(job)

    @app.get("/jobs/{job_id}/stream", dependencies=[_auth_dep])
    def stream(job_id: str, from_seq: Optional[int] = Query(default=None, alias="from_seq")):
        """SSE stream of training events for a job.

        Args:
            job_id: Worker job identifier returned by ``POST /start``.
            from_seq: When provided, replay buffered events with seq >=
                *from_seq* before streaming live ones. Used by the client for
                reconnection.
        """
        job = _get_job(job_id)

        # Collect any buffered events to replay before the live stream.
        replay_events: list[dict] = []
        if from_seq is not None:
            with job._state_lock:
                replay_events = [evt for seq, evt in job.event_buffer if seq >= from_seq]

        async def _generate():
            loop = asyncio.get_running_loop()

            # --- Replay phase ---
            for event in replay_events:
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") in ("training_complete", "training_error"):
                    return

            # --- Live streaming phase ---
            while True:
                try:
                    # Poll the thread-safe queue without blocking the event loop.
                    event = await loop.run_in_executor(
                        None, lambda: job.event_queue.get(timeout=1.0)
                    )
                except queue.Empty:
                    # Worker still alive; keep the connection open.
                    with job._state_lock:
                        t = job.thread
                    if t is None or not t.is_alive():
                        break
                    continue

                if event is None:
                    # Sentinel: stream is finished.
                    break

                yield f"data: {json.dumps(event)}\n\n"

                # Stop streaming after the terminal events.
                if event.get("type") in ("training_complete", "training_error"):
                    break

        return StreamingResponse(_generate(), media_type="text/event-stream")

    @app.get("/jobs/{job_id}/checkpoint", dependencies=[_auth_dep])
    def checkpoint(job_id: str):
        """Return checkpoint metadata for a job."""
        job = _get_job(job_id)
        return _job_checkpoint_payload(job)

    @app.get("/jobs/{job_id}/checkpoint/download", dependencies=[_auth_dep])
    def checkpoint_download(job_id: str):
        """Download the serialized checkpoint file for a job."""
        import os

        job = _get_job(job_id)
        with job._state_lock:
            checkpoint_path = job.checkpoint_path
        if checkpoint_path is None:
            raise HTTPException(status_code=404, detail="No checkpoint available")
        if not os.path.exists(checkpoint_path):
            raise HTTPException(status_code=410, detail="Checkpoint file gone")
        return FileResponse(
            checkpoint_path,
            media_type="application/octet-stream",
            filename=f"feedbax_checkpoint_{job.job_id}.eqx",
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
