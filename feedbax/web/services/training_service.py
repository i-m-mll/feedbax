"""Studio training service — spawns a worker subprocess and relays its SSE stream."""
from __future__ import annotations

import socket
import subprocess
import sys
from dataclasses import dataclass
from typing import AsyncIterator, Optional

from feedbax.web.worker import client as worker_client


# ---------------------------------------------------------------------------
# Public event type
# ---------------------------------------------------------------------------

@dataclass
class TrainingEvent:
    """A single event relayed from the worker SSE stream."""
    raw: dict  # parsed JSON from the SSE data: line


# ---------------------------------------------------------------------------
# Port helper
# ---------------------------------------------------------------------------

def _find_free_port() -> int:
    """Bind to port 0 to let the OS assign a free ephemeral port, then release it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class TrainingService:
    """Manages the lifecycle of the headless training worker subprocess."""

    def __init__(self) -> None:
        self._port: Optional[int] = None
        self._process: Optional[subprocess.Popen] = None
        self._base_url: Optional[str] = None
        # Track last known job metadata for synchronous helpers.
        self._current_job_id: Optional[str] = None
        self._last_loss: float = 0.0

    # ------------------------------------------------------------------
    # Worker subprocess lifecycle
    # ------------------------------------------------------------------

    async def _ensure_worker(self) -> str:
        """Lazily start the worker subprocess and wait for it to be healthy.

        Returns:
            The worker base URL, e.g. ``"http://127.0.0.1:54321"``.

        Raises:
            RuntimeError: If the worker does not respond within 5 seconds.
        """
        if self._process is not None and self._process.poll() is None:
            # Worker is already alive.
            return self._base_url  # type: ignore[return-value]

        port = _find_free_port()
        self._port = port
        self._base_url = f"http://127.0.0.1:{port}"

        self._process = subprocess.Popen(
            [sys.executable, "-m", "feedbax.web.worker", "--port", str(port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        await worker_client.wait_for_health(self._base_url, timeout=5.0, interval=0.1)
        return self._base_url

    def _terminate_worker(self) -> None:
        """Terminate the worker subprocess if it is running."""
        if self._process is not None:
            try:
                self._process.terminate()
            except OSError:
                pass
            self._process = None
            self._base_url = None
            self._port = None

    # ------------------------------------------------------------------
    # Public interface (mirrors the old TrainingService API)
    # ------------------------------------------------------------------

    async def start_training(self, total_batches: int) -> str:
        """Start a training job on the worker.

        Args:
            total_batches: Number of training steps.

        Returns:
            The job ID assigned by the worker.
        """
        base_url = await self._ensure_worker()
        job_id = await worker_client.start_job(base_url, total_batches)
        self._current_job_id = job_id
        self._last_loss = 0.0
        return job_id

    async def stop_training(self) -> None:
        """Ask the worker to stop its current job.

        Also kills the subprocess if the HTTP request fails.
        """
        if self._base_url is None:
            return
        try:
            await worker_client.stop_job(self._base_url)
        except Exception:
            # If the HTTP call failed, forcibly kill the subprocess.
            if self._process is not None:
                try:
                    self._process.kill()
                except OSError:
                    pass

    async def get_status(self) -> dict:
        """Return the worker's current status dict.

        Returns an idle placeholder if the worker is not running.
        """
        if self._base_url is None or (
            self._process is not None and self._process.poll() is not None
        ):
            return {"status": "idle", "batch": 0, "total_batches": 0, "last_loss": 0.0}
        try:
            return await worker_client.get_status(self._base_url)
        except Exception:
            return {"status": "error", "batch": 0, "total_batches": 0, "last_loss": 0.0}

    async def stream_progress(self, job_id: str) -> AsyncIterator[TrainingEvent]:
        """Relay the worker SSE stream as :class:`TrainingEvent` objects.

        Args:
            job_id: The job ID returned by :meth:`start_training`.

        Yields:
            :class:`TrainingEvent` instances wrapping raw event dicts.
        """
        if self._base_url is None:
            return
        async for event in worker_client.stream_events(self._base_url):
            # Keep last_loss in sync for synchronous callers.
            if "loss" in event:
                self._last_loss = float(event["loss"])
            yield TrainingEvent(raw=event)

    def latest_checkpoint(self, job_id: str) -> Optional[dict]:
        """Return a stub checkpoint dict for the given job.

        Args:
            job_id: The job ID.

        Returns:
            A dict with checkpoint metadata, or ``None`` if unknown.
        """
        if self._current_job_id != job_id:
            return None
        return {
            "checkpoint_path": None,
            "job_id": job_id,
            "loss": self._last_loss,
        }

    def last_loss(self, job_id: str) -> Optional[float]:
        """Return the last recorded loss for the given job.

        Args:
            job_id: The job ID.

        Returns:
            The last loss value, or ``None`` if unknown.
        """
        if self._current_job_id != job_id:
            return None
        return self._last_loss

    def __del__(self) -> None:
        self._terminate_worker()


training_service = TrainingService()
