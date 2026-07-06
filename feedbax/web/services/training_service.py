"""Studio training service — spawns a worker subprocess and relays its SSE stream."""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
from dataclasses import dataclass
from typing import AsyncIterator, Optional

import httpx

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


def _worker_stderr_excerpt(process: subprocess.Popen, *, limit: int = 4000) -> str:
    """Return captured stderr for an exited worker process."""
    if process.poll() is None or process.stderr is None:
        return ""
    try:
        _stdout, stderr = process.communicate(timeout=0.1)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if not isinstance(stderr, str):
        stderr = stderr.decode(errors="replace") if stderr else ""
    return stderr.strip()[-limit:]


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
        self._lock = asyncio.Lock()
        # Track last known job metadata for synchronous helpers and worker outages.
        self._last_status_by_job: dict[str, dict] = {}
        self._last_loss_by_job: dict[str, float] = {}

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
        async with self._lock:
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
                stderr=subprocess.PIPE,
                text=True,
            )

            try:
                await worker_client.wait_for_health(self._base_url, timeout=5.0, interval=0.1)
            except Exception as exc:
                stderr = _worker_stderr_excerpt(self._process)
                self._terminate_worker()
                detail = f": {stderr}" if stderr else ""
                raise RuntimeError(f"Worker subprocess failed health check{detail}") from exc
            return self._base_url

    def _terminate_worker(self) -> None:
        """Terminate the worker subprocess if it is running."""
        process = self._process
        self._process = None
        self._base_url = None
        self._port = None
        self._remote = False
        self._auth_token = None

        if process is not None:
            try:
                process.terminate()
            except OSError:
                pass
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except OSError:
                    pass
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    pass
            try:
                process.communicate(timeout=0.1)
            except (OSError, subprocess.TimeoutExpired):
                pass

    # ------------------------------------------------------------------
    # Public interface (mirrors the old TrainingService API)
    # ------------------------------------------------------------------

    async def start_training(
        self,
        total_batches: int,
        training_config: Optional[dict] = None,
        training_spec: Optional[dict] = None,
        task_spec: Optional[dict] = None,
        task_binding_spec: Optional[dict] = None,
        graph_spec: Optional[dict] = None,
    ) -> str:
        """Start a training job on the worker.

        Args:
            total_batches: Number of training steps.
            training_config: Optional dict forwarded to the worker as the
                ``training_config`` key in the ``/start`` request body.
                When present, the worker runs real JAX training; when ``None``
                it falls back to the synthetic stub.
            training_spec: Optional spec dict with optimizer/loss settings;
                forwarded to the worker for spec-driven configuration.
            task_spec: Optional task spec dict with task parameters;
                forwarded to the worker.

        Returns:
            The job ID assigned by the worker.
        """
        base_url = await self._ensure_worker()
        job_id = await worker_client.start_job(
            base_url,
            total_batches,
            training_config=training_config,
            training_spec=training_spec,
            task_spec=task_spec,
            task_binding_spec=task_binding_spec,
            graph_spec=graph_spec,
            auth_token=self._auth_token,
        )
        self._last_status_by_job[job_id] = {
            "status": "running",
            "batch": 0,
            "total_batches": total_batches,
            "last_loss": 0.0,
            "job_id": job_id,
        }
        self._last_loss_by_job[job_id] = 0.0
        return job_id

    async def stop_training(self, job_id: str) -> None:
        """Ask the worker to stop a job.

        Also kills the subprocess if the HTTP request fails (local mode only).
        """
        if self._base_url is None:
            if job_id not in self._last_status_by_job:
                raise ValueError(f"Unknown job {job_id!r}")
            return
        try:
            await worker_client.stop_job(
                self._base_url,
                job_id,
                auth_token=self._auth_token,
            )
            cached = self._last_status_by_job.setdefault(
                job_id,
                {
                    "batch": 0,
                    "total_batches": 0,
                    "last_loss": self._last_loss_by_job.get(job_id, 0.0),
                    "job_id": job_id,
                },
            )
            cached["status"] = "idle"
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise ValueError(f"Unknown job {job_id!r}") from exc
            raise
        except Exception:
            # If the HTTP call failed, forcibly kill the subprocess.
            if self._process is not None:
                try:
                    self._process.kill()
                except OSError:
                    pass

    async def worker_connected(self) -> bool:
        """Return whether the configured worker responds to health checks."""
        if self._base_url is None:
            return False
        try:
            await worker_client.wait_for_health(
                self._base_url,
                timeout=0.5,
                interval=0.05,
                auth_token=self._auth_token,
            )
            return True
        except Exception:
            return False

    async def get_status(self, job_id: str) -> Optional[dict]:
        """Return a job's worker status dict, or ``None`` if the job is unknown."""
        fallback = self._last_status_by_job.get(job_id)
        if self._base_url is None or (
            not self._remote and self._process is not None and self._process.poll() is not None
        ):
            return fallback
        try:
            status = await worker_client.get_status(
                self._base_url,
                job_id,
                auth_token=self._auth_token,
            )
            self._last_status_by_job[job_id] = status
            if "last_loss" in status:
                self._last_loss_by_job[job_id] = float(status["last_loss"])
            return status
        except Exception:
            return fallback

    async def stream_progress(self, job_id: str) -> AsyncIterator[TrainingEvent]:
        """Relay the worker SSE stream as :class:`TrainingEvent` objects.

        Args:
            job_id: The job ID returned by :meth:`start_training`.

        Yields:
            :class:`TrainingEvent` instances wrapping raw event dicts.
        """
        if self._base_url is None:
            return
        async for event in worker_client.stream_events(
            self._base_url,
            job_id,
            auth_token=self._auth_token,
        ):
            # Keep last_loss in sync for synchronous callers.
            if "loss" in event:
                self._last_loss_by_job[job_id] = float(event["loss"])
            if event.get("type") == "training_progress":
                self._last_status_by_job[job_id] = {
                    "status": "running",
                    "batch": event.get("batch", 0),
                    "total_batches": event.get("total_batches", 0),
                    "last_loss": event.get("loss", self._last_loss_by_job.get(job_id, 0.0)),
                    "job_id": job_id,
                }
            elif event.get("type") == "training_complete":
                self._last_status_by_job[job_id] = {
                    "status": "completed",
                    "batch": event.get("batch", 0),
                    "total_batches": event.get("batch", 0),
                    "last_loss": event.get("loss", self._last_loss_by_job.get(job_id, 0.0)),
                    "job_id": job_id,
                }
            elif event.get("type") == "training_error":
                cached = self._last_status_by_job.setdefault(
                    job_id,
                    {
                        "batch": event.get("batch", 0),
                        "total_batches": 0,
                        "last_loss": self._last_loss_by_job.get(job_id, 0.0),
                        "job_id": job_id,
                    },
                )
                cached["status"] = "error"
            yield TrainingEvent(raw=event)

    async def latest_checkpoint(self, job_id: str) -> Optional[dict]:
        """Return checkpoint metadata for the given job by querying the worker.

        Proxies to the worker's keyed checkpoint endpoint when a worker URL is
        configured.

        Args:
            job_id: The job ID.

        Returns:
            A dict with checkpoint metadata (keys: ``batch``, ``loss``,
            ``weights_available``), or ``None`` if the job is unknown.
        """
        if self._base_url is None:
            if job_id not in self._last_status_by_job:
                return None
            return {
                "batch": self._last_status_by_job[job_id].get("batch", 0),
                "loss": self._last_loss_by_job.get(job_id, 0.0),
                "weights_available": False,
                "job_id": job_id,
            }
        try:
            data = await worker_client.get_checkpoint(
                self._base_url,
                job_id,
                auth_token=self._auth_token,
            )
            data["job_id"] = job_id
            return data
        except Exception:
            if job_id not in self._last_status_by_job:
                return None
            return {
                "batch": self._last_status_by_job[job_id].get("batch", 0),
                "loss": self._last_loss_by_job.get(job_id, 0.0),
                "weights_available": False,
                "job_id": job_id,
            }

    async def latest_manifest(self, job_id: str) -> Optional[dict]:
        """Return the durable training manifest for *job_id* when available."""
        if self._base_url is None:
            return None
        try:
            return await worker_client.get_manifest(
                self._base_url,
                job_id,
                auth_token=self._auth_token,
            )
        except Exception:
            return None

    async def download_checkpoint(self, job_id: str, dest_path: str) -> None:
        """Download the serialized checkpoint from the worker to a local file.

        Args:
            job_id: The job ID whose checkpoint to download.
            dest_path: Local filesystem path to write the checkpoint file.

        Raises:
            ValueError: If *job_id* is unknown to the worker.
            RuntimeError: If no worker is configured.
        """
        if self._base_url is None:
            raise RuntimeError("No worker configured")
        try:
            await worker_client.download_checkpoint(
                self._base_url,
                job_id,
                dest_path,
                auth_token=self._auth_token,
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise ValueError(f"Unknown job {job_id!r}") from exc
            raise

    def last_loss(self, job_id: str) -> Optional[float]:
        """Return the last recorded loss for the given job.

        Args:
            job_id: The job ID.

        Returns:
            The last loss value, or ``None`` if unknown.
        """
        return self._last_loss_by_job.get(job_id)

    def __del__(self) -> None:
        self._terminate_worker()


training_service = TrainingService()
