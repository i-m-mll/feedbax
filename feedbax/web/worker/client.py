"""Async HTTP client for the Feedbax training worker.

Used by the Studio backend (TrainingService) to:
- Poll the worker's /health endpoint on startup.
- Start / stop training jobs via POST.
- Relay the SSE event stream to WebSocket clients.
"""

from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator, Optional

import httpx

from feedbax.orchestration.events import RUN_EVENT_TERMINAL_TYPES

# Maximum reconnection attempts for the SSE stream.
_MAX_RECONNECT_ATTEMPTS = 10
# Seconds to wait between reconnection attempts.
_RECONNECT_DELAY = 2.0


def _auth_headers(auth_token: Optional[str]) -> dict:
    """Return an ``Authorization`` header dict when *auth_token* is set."""
    if auth_token is None:
        return {}
    return {"Authorization": f"Bearer {auth_token}"}


async def wait_for_health(
    base_url: str,
    timeout: float = 5.0,
    interval: float = 0.1,
    auth_token: Optional[str] = None,
) -> None:
    """Poll GET /health until the worker responds OK, or raise RuntimeError on timeout.

    Args:
        base_url: Worker base URL, e.g. ``"http://127.0.0.1:8765"``.
        timeout: Maximum seconds to wait.
        interval: Seconds between poll attempts.
        auth_token: Optional shared secret to include in the Authorization header.

    Raises:
        RuntimeError: If the worker does not respond within *timeout* seconds.
    """
    deadline = asyncio.get_running_loop().time() + timeout
    headers = _auth_headers(auth_token)
    async with httpx.AsyncClient() as client:
        while True:
            try:
                resp = await client.get(f"{base_url}/health", headers=headers, timeout=2.0)
                if resp.status_code == 200:
                    return
            except (httpx.ConnectError, httpx.TimeoutException):
                pass

            if asyncio.get_running_loop().time() >= deadline:
                raise RuntimeError("Training worker failed to start")

            await asyncio.sleep(interval)


async def start_job(
    base_url: str,
    total_batches: int,
    *,
    job_id: str,
    run_set_id: str,
    training_config: Optional[dict] = None,
    training_spec: Optional[dict] = None,
    task_spec: Optional[dict] = None,
    task_binding_spec: Optional[dict] = None,
    graph_spec: Optional[dict] = None,
    auth_token: Optional[str] = None,
) -> str:
    """POST /start and return the assigned job_id.

    Args:
        base_url: Worker base URL.
        total_batches: Number of training steps to run.
        training_config: Optional dict forwarded to the worker as the
            ``training_config`` key. When ``None``, the worker uses default
            training configuration values; real JAX training is always attempted.
        training_spec: Optional spec dict with optimizer type/params and loss
            weights; forwarded to the worker for spec-driven optimizer construction.
        task_spec: Optional task spec dict with task parameters such as
            ``n_reach_steps`` and ``effort_weight``; forwarded to the worker.
        auth_token: Optional shared secret.

    Returns:
        The ``job_id`` string assigned by the worker.
    """
    body: dict = {
        "job_id": job_id,
        "run_set_id": run_set_id,
        "total_batches": total_batches,
    }
    if training_config is not None:
        body["training_config"] = training_config
    if training_spec is not None:
        body["training_spec"] = training_spec
    if task_spec is not None:
        body["task_spec"] = task_spec
    if task_binding_spec is not None:
        body["task_binding_spec"] = task_binding_spec
    if graph_spec is not None:
        body["graph_spec"] = graph_spec
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{base_url}/start",
            json=body,
            headers=_auth_headers(auth_token),
            timeout=10.0,
        )
        resp.raise_for_status()
        return resp.json()["job_id"]


async def stop_job(base_url: str, job_id: str, auth_token: Optional[str] = None) -> None:
    """POST /jobs/{job_id}/stop to request the worker halt a job.

    Args:
        base_url: Worker base URL.
        job_id: Worker job identifier.
        auth_token: Optional shared secret.
    """
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{base_url}/jobs/{job_id}/stop",
            headers=_auth_headers(auth_token),
            timeout=5.0,
        )
        resp.raise_for_status()


async def get_status(base_url: str, job_id: str, auth_token: Optional[str] = None) -> dict:
    """GET /jobs/{job_id}/status and return the raw status dict.

    Args:
        base_url: Worker base URL.
        job_id: Worker job identifier.
        auth_token: Optional shared secret.

    Returns:
        Dict with keys ``status``, ``batch``, ``total_batches``, ``last_loss``.
    """
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{base_url}/jobs/{job_id}/status",
            headers=_auth_headers(auth_token),
            timeout=5.0,
        )
        resp.raise_for_status()
        return resp.json()


async def get_checkpoint(base_url: str, job_id: str, auth_token: Optional[str] = None) -> dict:
    """GET /jobs/{job_id}/checkpoint and return the raw checkpoint metadata dict.

    Args:
        base_url: Worker base URL.
        job_id: Worker job identifier.
        auth_token: Optional shared secret.

    Returns:
        Dict with keys ``batch``, ``loss``, ``weights_available``.
    """
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{base_url}/jobs/{job_id}/checkpoint",
            headers=_auth_headers(auth_token),
            timeout=5.0,
        )
        resp.raise_for_status()
        return resp.json()


async def get_manifest(base_url: str, job_id: str, auth_token: Optional[str] = None) -> dict:
    """GET /jobs/{job_id}/manifest and return a job's durable manifest."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{base_url}/jobs/{job_id}/manifest",
            headers=_auth_headers(auth_token),
            timeout=5.0,
        )
        resp.raise_for_status()
        return resp.json()


async def download_checkpoint(
    base_url: str,
    job_id: str,
    dest_path: str,
    auth_token: Optional[str] = None,
) -> None:
    """Stream GET /jobs/{job_id}/checkpoint/download and write to *dest_path*.

    Args:
        base_url: Worker base URL.
        job_id: Worker job identifier.
        dest_path: Local filesystem path to write the checkpoint bytes to.
        auth_token: Optional shared secret.

    Raises:
        httpx.HTTPStatusError: If the worker returns a non-2xx status.
    """
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "GET",
            f"{base_url}/jobs/{job_id}/checkpoint/download",
            headers=_auth_headers(auth_token),
        ) as resp:
            resp.raise_for_status()
            with open(dest_path, "wb") as f:
                async for chunk in resp.aiter_bytes():
                    f.write(chunk)


async def stream_events(
    base_url: str,
    job_id: str,
    auth_token: Optional[str] = None,
) -> AsyncIterator[dict]:
    """Connect to GET /jobs/{job_id}/stream and yield parsed JSON event dicts.

    Automatically reconnects on connection errors up to
    ``_MAX_RECONNECT_ATTEMPTS`` times, using the ``seq`` field of the last
    received event to resume from the correct position via the ``from_seq``
    query parameter.

    The generator exits when the SSE stream closes (worker completed or
    errored), or when all reconnection attempts are exhausted.

    Args:
        base_url: Worker base URL.
        job_id: Worker job identifier.
        auth_token: Optional shared secret to include in the Authorization header.

    Yields:
        Parsed event dicts (the ``data:`` payload from each SSE line).
    """
    last_seq: Optional[int] = None
    attempt = 0
    headers = _auth_headers(auth_token)

    while attempt <= _MAX_RECONNECT_ATTEMPTS:
        url = f"{base_url}/jobs/{job_id}/stream"
        params: dict = {}
        expected_seq: Optional[int] = None
        if last_seq is not None:
            expected_seq = last_seq + 1
            params["from_seq"] = expected_seq

        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("GET", url, params=params, headers=headers) as resp:
                    resp.raise_for_status()
                    resumed_after_disconnect = attempt > 0
                    # Successful connection — reset attempt counter.
                    attempt = 0
                    reported_resume = False
                    async for line in resp.aiter_lines():
                        line = line.strip()
                        if not line.startswith("data:"):
                            continue
                        payload = line[len("data:") :].strip()
                        if not payload:
                            continue
                        try:
                            event = json.loads(payload)
                        except json.JSONDecodeError:
                            continue
                        if not event:
                            continue
                        event_seq: Optional[int] = None
                        if "seq" in event:
                            event_seq = int(event["seq"])
                        if resumed_after_disconnect and not reported_resume:
                            missed_events = 0
                            reason = "resumed"
                            message = "Training stream resumed after reconnect."
                            if (
                                expected_seq is not None
                                and event_seq is not None
                                and event_seq > expected_seq
                            ):
                                missed_events = event_seq - expected_seq
                                reason = "gap"
                                message = (
                                    "Training stream resumed after reconnect with "
                                    f"{missed_events} missed event(s)."
                                )
                            yield {
                                "type": "training_resync",
                                "job_id": job_id,
                                "expected_worker_seq": expected_seq,
                                "observed_worker_seq": event_seq,
                                "worker_seq": last_seq,
                                "missed_events": missed_events,
                                "reason": reason,
                                "message": message,
                            }
                            reported_resume = True
                        # Track highest seq seen for reconnection.
                        if event_seq is not None:
                            last_seq = event_seq
                        yield event
                        # Terminal events — stop cleanly, no reconnect.
                        if event.get("type") in RUN_EVENT_TERMINAL_TYPES or event.get(
                            "type"
                        ) in ("training_complete", "training_error"):
                            return
        except (
            httpx.ConnectError,
            httpx.RemoteProtocolError,
            httpx.ReadError,
        ):
            attempt += 1
            if attempt > _MAX_RECONNECT_ATTEMPTS:
                return
            await asyncio.sleep(_RECONNECT_DELAY)
        except Exception:
            # Unknown error — stop cleanly rather than reconnecting blindly.
            return
