from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, AsyncIterator
import asyncio
import queue
import threading
import time
import uuid


class TrainingStatus(str, Enum):
    IDLE = 'idle'
    RUNNING = 'running'
    PAUSED = 'paused'
    COMPLETED = 'completed'
    ERROR = 'error'


@dataclass
class TrainingProgress:
    job_id: str
    batch: int
    total_batches: int
    loss: float
    metrics: Dict[str, float]
    status: TrainingStatus


@dataclass
class TrainingJob:
    job_id: str
    total_batches: int
    progress_queue: queue.Queue[TrainingProgress]
    stop_event: threading.Event
    thread: threading.Thread
    status: TrainingStatus = TrainingStatus.IDLE
    last_loss: float = 0.0


class TrainingService:
    def __init__(self) -> None:
        self._jobs: Dict[str, TrainingJob] = {}

    def start_training(self, total_batches: int) -> str:
        job_id = str(uuid.uuid4())
        progress_queue: queue.Queue[TrainingProgress] = queue.Queue()
        stop_event = threading.Event()

        job = TrainingJob(
            job_id=job_id,
            total_batches=total_batches,
            progress_queue=progress_queue,
            stop_event=stop_event,
            thread=threading.Thread(target=self._run_training, args=(job_id,), daemon=True),
            status=TrainingStatus.RUNNING,
        )
        self._jobs[job_id] = job
        job.thread.start()
        return job_id

    def _run_training(self, job_id: str) -> None:
        job = self._jobs[job_id]
        try:
            start_loss = 1.0
            for batch in range(job.total_batches):
                if job.stop_event.is_set():
                    job.status = TrainingStatus.IDLE
                    break
                time.sleep(0.05)
                decay = 0.98 ** batch
                loss = start_loss * decay
                job.last_loss = loss
                job.progress_queue.put(
                    TrainingProgress(
                        job_id=job_id,
                        batch=batch + 1,
                        total_batches=job.total_batches,
                        loss=loss,
                        metrics={'loss': loss, 'grad_norm': max(0.05, 0.5 * decay)},
                        status=TrainingStatus.RUNNING,
                    )
                )
            if job.status == TrainingStatus.RUNNING:
                job.status = TrainingStatus.COMPLETED
        except Exception:
            job.status = TrainingStatus.ERROR

    def get_status(self, job_id: str) -> TrainingStatus:
        job = self._jobs[job_id]
        return job.status

    def stop_training(self, job_id: str) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        job.stop_event.set()
        job.status = TrainingStatus.IDLE

    async def stream_progress(self, job_id: str) -> AsyncIterator[TrainingProgress]:
        job = self._jobs[job_id]
        while job.status == TrainingStatus.RUNNING:
            try:
                progress = job.progress_queue.get_nowait()
                yield progress
            except queue.Empty:
                await asyncio.sleep(0.1)

        while not job.progress_queue.empty():
            yield job.progress_queue.get_nowait()

    def latest_checkpoint(self, job_id: str) -> Optional[dict]:
        job = self._jobs.get(job_id)
        if job is None:
            return None
        return {
            'checkpoint_path': None,
            'batch': job.total_batches if job.status == TrainingStatus.COMPLETED else 0,
            'loss': job.last_loss,
        }

    def last_loss(self, job_id: str) -> Optional[float]:
        job = self._jobs.get(job_id)
        if job is None:
            return None
        return job.last_loss


training_service = TrainingService()
