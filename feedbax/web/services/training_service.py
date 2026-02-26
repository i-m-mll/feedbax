"""Studio training service — spawns a worker subprocess and relays its SSE stream."""
from __future__ import annotations

import os
import socket
import subprocess
import sys
from dataclasses import dataclass
from typing import AsyncIterator, Optional

import feedbax.web.worker.client as worker_client


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
    """Manages the lifecycle of the headless training worker subprocess.

    Supports two operating modes:

    - **Local mode** (default): a worker subprocess is spawned on demand.
    - **Remote mode**: connects to a pre-existing worker at a given URL.
      Activated either by setting the ``FEEDBAX_WORKER_URL`` environment
      variable before construction, or by calling :meth:`connect_remote`.
    """

    def __init__(self) -> None:
        self._port: Optional[int] = None
        self._process: Optional[subprocess.Popen] = None
        self._base_url: Optional[str] = None
        self._auth_token: Optional[str] = None
        self._remote: bool = False
        # Track last known job metadata for synchronous helpers.
        self._current_job_id: Optional[str] = None
        self._last_loss: float = 0.0

        # Honour the FEEDBAX_WORKER_URL env var: skip subprocess and connect
        # directly to an external worker.
        env_url = os.environ.get("FEEDBAX_WORKER_URL")
        if env_url:
            self._base_url = env_url.rstrip("/")
            self._remote = True

    # ------------------------------------------------------------------
    # Remote mode
    # ------------------------------------------------------------------

    def connect_remote(self, url: str, auth_token: Optional[str] = None) -> None:
        """Switch to remote worker mode.

        Terminates any running local subprocess and configures the service to
        forward all requests to the given URL.

        Args:
            url: Base URL of the remote worker, e.g. ``"http://100.1.2.3:8765"``.
            auth_token: Optional bearer token required by the remote worker.
        """
        self._terminate_worker()
        self._base_url = url.rstrip("/")
        self._auth_token = auth_token
        self._remote = True

    def worker_mode(self) -> str:
        """Return ``"remote"`` or ``"local"`` depending on current configuration."""
        return "remote" if self._remote else "local"

    # ------------------------------------------------------------------
    # Worker subprocess lifecycle
    # ------------------------------------------------------------------

    async def _ensure_worker(self) -> str:
        """Lazily start the worker subprocess and wait for it to be healthy.

        In remote mode the URL is already configured — this simply returns it.

        Returns:
            The worker base URL, e.g. ``"http://127.0.0.1:54321"``.

        Raises:
            RuntimeError: If the worker does not respond within 5 seconds.
        """
        if self._remote:
            if self._base_url is None:
                raise RuntimeError("Remote worker URL is not configured")
            return self._base_url

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
        job_id = await worker_client.start_job(
            base_url, total_batches, auth_token=self._auth_token
        )
        self._current_job_id = job_id
        self._last_loss = 0.0
        return job_id

    async def stop_training(self) -> None:
        """Ask the worker to stop its current job.

        Also kills the subprocess if the HTTP request fails (local mode only).
        """
        if self._base_url is None:
            return
        try:
            await worker_client.stop_job(self._base_url, auth_token=self._auth_token)
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
            not self._remote
            and self._process is not None
            and self._process.poll() is not None
        ):
            return {"status": "idle", "batch": 0, "total_batches": 0, "last_loss": 0.0}
        try:
            return await worker_client.get_status(
                self._base_url, auth_token=self._auth_token
            )
        except Exception:
            return {"status": "error", "batch": 0, "total_batches": 0, "last_loss": 0.0}

    async def stream_progress(self, _job_id: str) -> AsyncIterator[TrainingEvent]:
        """Relay the worker SSE stream as :class:`TrainingEvent` objects.

        Args:
            job_id: The job ID returned by :meth:`start_training`.

        Yields:
            :class:`TrainingEvent` instances wrapping raw event dicts.
        """
        if self._base_url is None:
            return
        async for event in worker_client.stream_events(
            self._base_url, auth_token=self._auth_token
        ):
            # Keep last_loss in sync for synchronous callers.
            if "loss" in event:
                self._last_loss = float(event["loss"])
            yield TrainingEvent(raw=event)

    async def latest_checkpoint(self, job_id: str) -> Optional[dict]:
        """Return checkpoint metadata for the given job by querying the worker.

        Proxies to the worker's ``GET /checkpoint`` endpoint when a worker URL
        is configured and the requested job matches the current job.

        Args:
            job_id: The job ID.

        Returns:
            A dict with checkpoint metadata (keys: ``batch``, ``loss``,
            ``weights_available``), or ``None`` if the job is unknown.
        """
        if self._current_job_id != job_id:
            return None
        if self._base_url is None:
            return {"batch": 0, "loss": self._last_loss, "weights_available": False}
        try:
            data = await worker_client.get_checkpoint(
                self._base_url, auth_token=self._auth_token
            )
            data["job_id"] = job_id
            return data
        except Exception:
            return {"batch": 0, "loss": self._last_loss, "weights_available": False}

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
