"""Feedbax Studio headless training worker FastAPI app."""
from __future__ import annotations

import asyncio
import queue
import random
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

from fastapi import FastAPI
from fastapi.responses import StreamingResponse


class WorkerStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class _Job:
    job_id: str
    total_batches: int
    event_queue: queue.Queue
    stop_event: threading.Event
    thread: threading.Thread
    status: WorkerStatus = WorkerStatus.IDLE
    batch: int = 0
    last_loss: float = 0.0


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
            job.event_queue.put(
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
                }
            )
            # Emit log event.
            job.event_queue.put(
                {
                    "type": "training_log",
                    "job_id": job.job_id,
                    "batch": batch + 1,
                    "level": "info",
                    "message": log_line,
                }
            )

        job.status = WorkerStatus.COMPLETED
        job.event_queue.put(
            {
                "type": "training_complete",
                "job_id": job.job_id,
                "batch": job.total_batches,
                "loss": job.last_loss,
            }
        )
    except Exception as exc:
        job.status = WorkerStatus.ERROR
        job.event_queue.put(
            {
                "type": "training_error",
                "job_id": job.job_id,
                "error": str(exc),
            }
        )
    finally:
        # Sentinel: tells SSE generator the stream is done.
        job.event_queue.put(None)


_SENTINEL = object()  # typed sentinel for type-narrowing clarity


def create_app() -> FastAPI:
    """Create and return the worker FastAPI application."""
    app = FastAPI(title="Feedbax Training Worker", version="0.1.0")

    # Module-level state for the single active job.
    _state: Dict[str, Optional[_Job]] = {"current": None}

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/start")
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
            thread=threading.Thread(target=_run_training, args=(job,), daemon=True),
            status=WorkerStatus.RUNNING,
        )
        _state["current"] = job
        job.thread.start()
        return {"job_id": job_id}

    @app.post("/stop")
    def stop():
        job = _state.get("current")
        if job is not None:
            job.stop_event.set()
            job.status = WorkerStatus.IDLE
        return {"ok": True}

    @app.get("/status")
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

    @app.get("/stream")
    def stream():
        """SSE stream of training events for the current job."""
        job = _state.get("current")
        if job is None:
            # No job running; return an empty stream immediately.
            async def _empty():
                yield "data: {}\n\n"
            return StreamingResponse(_empty(), media_type="text/event-stream")

        import json

        async def _generate():
            loop = asyncio.get_event_loop()
            while True:
                try:
                    # Poll the thread-safe queue without blocking the event loop.
                    event = await loop.run_in_executor(
                        None, lambda: job.event_queue.get(timeout=1.0)
                    )
                except queue.Empty:
                    # Worker still alive; keep the connection open.
                    if not job.thread.is_alive():
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
