"""Feedbax Studio headless training worker FastAPI app."""
from __future__ import annotations

import asyncio
import collections
import json
import queue
import random
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, Optional, Tuple

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer


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
    # Buffer of (seq, event_dict) for replay support.
    event_buffer: Deque[Tuple[int, dict]] = field(
        default_factory=lambda: collections.deque(maxlen=_EVENT_BUFFER_MAX)
    )
    thread: Optional[threading.Thread] = None
    status: WorkerStatus = WorkerStatus.IDLE
    batch: int = 0
    last_loss: float = 0.0
    # Monotonically increasing sequence counter; protected by _seq_lock.
    _seq: int = 0
    _seq_lock: threading.Lock = field(default_factory=threading.Lock)

    def next_seq(self) -> int:
        """Return the next sequence number and advance the counter."""
        with self._seq_lock:
            seq = self._seq
            self._seq += 1
            return seq


def _run_training(job: _Job) -> None:
    """Synthetic training loop — runs in a background thread."""
    try:
        start_loss = 1.0
        for batch in range(job.total_batches):
            if job.stop_event.is_set():
                job.status = WorkerStatus.IDLE
                return

            time.sleep(0.05)

            decay = 0.98 ** batch
            loss = start_loss * decay
            job.last_loss = loss
            job.batch = batch + 1

            noise = lambda: random.uniform(-0.005, 0.005)
            loss_terms = {
                "tracking": max(0.0, 0.70 * loss + noise()),
                "effort": max(0.0, 0.20 * loss + noise()),
                "smoothness": max(0.0, 0.07 * loss + noise()),
                "hidden_reg": max(0.0, 0.03 * loss + noise()),
            }
            grad_norm = max(0.01, 1.0 * decay + random.uniform(-0.02, 0.02))
            step_time_ms = random.uniform(30.0, 60.0)
            log_line = f"Step {batch + 1} | loss={loss:.4f} | grad_norm={grad_norm:.3f}"

            # Emit progress event.
            _emit(
                job,
                {
                    "type": "training_progress",
                    "job_id": job.job_id,
                    "batch": batch + 1,
                    "total_batches": job.total_batches,
                    "loss": loss,
                    "loss_terms": loss_terms,
                    "grad_norm": grad_norm,
                    "step_time_ms": step_time_ms,
                    "status": "running",
                },
            )
            # Emit log event.
            _emit(
                job,
                {
                    "type": "training_log",
                    "job_id": job.job_id,
                    "batch": batch + 1,
                    "level": "info",
                    "message": log_line,
                },
            )

        job.status = WorkerStatus.COMPLETED
        _emit(
            job,
            {
                "type": "training_complete",
                "job_id": job.job_id,
                "batch": job.total_batches,
                "loss": job.last_loss,
            },
        )
    except Exception as exc:
        job.status = WorkerStatus.ERROR
        _emit(
            job,
            {
                "type": "training_error",
                "job_id": job.job_id,
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
        job_id = str(uuid.uuid4())
        stop_event = threading.Event()
        event_queue: queue.Queue = queue.Queue()

        job = _Job(
            job_id=job_id,
            total_batches=total_batches,
            event_queue=event_queue,
            stop_event=stop_event,
            status=WorkerStatus.RUNNING,
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
            replay_events = [
                evt for seq, evt in job.event_buffer if seq >= from_seq
            ]

        async def _generate():
            loop = asyncio.get_event_loop()

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

    return app
