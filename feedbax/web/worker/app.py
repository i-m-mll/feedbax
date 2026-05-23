"""Feedbax Studio headless training worker FastAPI app."""

from __future__ import annotations

import asyncio
import collections
import json
import queue
import threading
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, Optional, Tuple

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from feedbax.studio_schema import validate_task_binding_schema
from feedbax.studio_protocol import infer_task_n_steps
from feedbax.web.graph_normalization import (
    normalize_graph_for_studio_authoring,
    normalize_task_binding_spec_for_studio_authoring,
)
from feedbax.web.models.graph import GraphSpec, StudioTaskBindingSpec


class WorkerStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"


# Maximum number of past events to buffer per job for from_seq replay.
_EVENT_BUFFER_MAX = 1000


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
    # Monotonically increasing sequence counter; protected by _seq_lock.
    _seq: int = 0
    _seq_lock: threading.Lock = field(default_factory=threading.Lock)

    def next_seq(self) -> int:
        """Return the next sequence number and advance the counter."""
        with self._seq_lock:
            seq = self._seq
            self._seq += 1
            return seq


def _manifest_history_events(job: _Job) -> list[dict[str, Any]]:
    """Return compact event history suitable for a durable JSON artifact."""
    history_types = {"training_progress", "training_log", "training_error", "training_complete"}
    return [dict(event) for _, event in job.event_buffer if event.get("type") in history_types]


def _write_job_manifest(job: _Job) -> None:
    """Write a durable training-run manifest for a completed worker job."""
    try:
        from feedbax.manifest import write_training_run_manifest

        manifest, path = write_training_run_manifest(
            job_id=job.job_id,
            total_batches=job.total_batches,
            training_spec=job.training_spec,
            task_spec=job.task_spec,
            task_binding_spec=job.task_binding_spec,
            graph_spec=job.graph_spec,
            checkpoint_path=job.checkpoint_path,
            history_events=_manifest_history_events(job),
            retention_plan=job.retention_plan_payload,
            retained_observables=job.retained_observables_payload,
            status=job.status.value,
            final_loss=job.last_loss,
        )
        job.manifest_path = str(path)
        job.manifest_payload = manifest.model_dump(mode="json", exclude_none=True)
        _emit(
            job,
            {
                "type": "training_log",
                "job_id": job.job_id,
                "batch": job.batch,
                "level": "info",
                "message": "Training manifest saved",
                "manifest_path": job.manifest_path,
                "manifest_id": manifest.id,
            },
        )
    except Exception as exc:
        _emit(
            job,
            {
                "type": "training_log",
                "job_id": job.job_id,
                "batch": job.batch,
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
    grad_clip: float = 1.0
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
    if task_binding_spec.get("schema_version") != "feedbax.studio.task_bindings.v2":
        raise ValueError("Training worker requires task_binding_spec schema v2")
    if "exposed_outputs" in task_binding_spec:
        raise ValueError("task_binding_spec.exposed_outputs is not accepted; use exposed_data")
    try:
        binding_spec = StudioTaskBindingSpec.model_validate(task_binding_spec)
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

        def _get(key: str, default, cast=None):
            val = training_config.get(key, default)
            if val is None:
                return default
            try:
                return cast(val) if cast is not None else val
            except (TypeError, ValueError):
                return default

        cfg.n_batches = _get("n_batches", cfg.n_batches, int)
        cfg.batch_size = _get("batch_size", cfg.batch_size, int)
        cfg.learning_rate = _get("learning_rate", cfg.learning_rate, float)
        cfg.grad_clip = _get("grad_clip", cfg.grad_clip, float)
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
    job.last_loss = result.final_loss
    job.batch = job.total_batches
    job.checkpoint_path = result.checkpoint_path
    job.retention_plan_payload = result.retention_plan
    job.retained_observables_payload = result.retained_observables
    job.status = WorkerStatus.IDLE if job.stop_event.is_set() else WorkerStatus.COMPLETED
    if result.checkpoint_path is not None:
        _emit(
            job,
            {
                "type": "training_log",
                "job_id": job.job_id,
                "batch": job.batch,
                "level": "info",
                "message": "Checkpoint saved",
                "execution": "generic_graph",
            },
        )
    _write_job_manifest(job)
    complete_event = {
        "type": "training_complete",
        "job_id": job.job_id,
        "batch": job.batch,
        "loss": job.last_loss,
        "execution": "generic_graph",
    }
    if job.manifest_path is not None:
        complete_event["manifest_path"] = job.manifest_path
    if job.manifest_payload is not None:
        complete_event["manifest_id"] = job.manifest_payload.get("id")
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
        if job.status == WorkerStatus.RUNNING:
            job.status = WorkerStatus.ERROR
            _emit(
                job,
                {
                    "type": "training_error",
                    "job_id": job.job_id,
                    "batch": job.batch,
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
    # Module-level state for the single active job.
    # ------------------------------------------------------------------

    _state: Dict[str, Optional[_Job]] = {"current": None}

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
        thread = threading.Thread(target=_run_training, args=(job,), daemon=True)
        job.thread = thread
        _state["current"] = job
        thread.start()
        return {"job_id": job_id}

    @app.post("/stop", dependencies=[_auth_dep])
    def stop():
        job = _state.get("current")
        if job is not None:
            job.stop_event.set()
            job.status = WorkerStatus.IDLE
        return {"ok": True}

    @app.get("/status", dependencies=[_auth_dep])
    def status():
        job = _state.get("current")
        if job is None:
            return {
                "status": WorkerStatus.IDLE,
                "batch": 0,
                "total_batches": 0,
                "last_loss": 0.0,
            }
        return {
            "status": job.status,
            "batch": job.batch,
            "total_batches": job.total_batches,
            "last_loss": job.last_loss,
            "manifest_path": job.manifest_path,
        }

    @app.get("/stream", dependencies=[_auth_dep])
    def stream(from_seq: Optional[int] = Query(default=None, alias="from_seq")):
        """SSE stream of training events for the current job.

        Args:
            from_seq: When provided, replay buffered events with seq >=
                *from_seq* before streaming live ones. Used by the client for
                reconnection.
        """
        job = _state.get("current")
        if job is None:
            # No job running; return an empty stream immediately.
            async def _empty():
                yield "data: {}\n\n"

            return StreamingResponse(_empty(), media_type="text/event-stream")

        # Collect any buffered events to replay before the live stream.
        replay_events: list[dict] = []
        if from_seq is not None:
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

    @app.get("/checkpoint", dependencies=[_auth_dep])
    def checkpoint():
        """Return checkpoint metadata for the current job."""
        job = _state.get("current")
        if job is None:
            return {"batch": 0, "loss": 0.0, "weights_available": False}
        return {
            "batch": job.batch,
            "loss": job.last_loss,
            "weights_available": job.checkpoint_path is not None,
        }

    @app.get("/checkpoint/download", dependencies=[_auth_dep])
    def checkpoint_download():
        """Download the serialized checkpoint file for the current job."""
        import os

        job = _state.get("current")
        if job is None or job.checkpoint_path is None:
            raise HTTPException(status_code=404, detail="No checkpoint available")
        if not os.path.exists(job.checkpoint_path):
            raise HTTPException(status_code=410, detail="Checkpoint file gone")
        return FileResponse(
            job.checkpoint_path,
            media_type="application/octet-stream",
            filename=f"feedbax_checkpoint_{job.job_id}.eqx",
        )

    @app.get("/manifest", dependencies=[_auth_dep])
    def manifest():
        """Return the durable manifest for the current job."""
        job = _state.get("current")
        if job is None or job.manifest_payload is None:
            raise HTTPException(status_code=404, detail="No manifest available")
        return job.manifest_payload

    return app
