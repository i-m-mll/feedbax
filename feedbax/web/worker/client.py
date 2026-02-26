"""Async HTTP client for the Feedbax training worker.

Used by the Studio backend (TrainingService) to:
- Poll the worker's /health endpoint on startup.
- Start / stop training jobs via POST.
- Relay the SSE event stream to WebSocket clients.
"""
from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator

import httpx


async def wait_for_health(base_url: str, timeout: float = 5.0, interval: float = 0.1) -> None:
    """Poll GET /health until the worker responds OK, or raise RuntimeError on timeout.

    Args:
        base_url: Worker base URL, e.g. ``"http://127.0.0.1:8765"``.
        timeout: Maximum seconds to wait.
        interval: Seconds between poll attempts.

    Raises:
        RuntimeError: If the worker does not respond within *timeout* seconds.
    """
    deadline = asyncio.get_event_loop().time() + timeout
    async with httpx.AsyncClient() as client:
        while True:
            try:
                resp = await client.get(f"{base_url}/health", timeout=2.0)
                if resp.status_code == 200:
                    return
            except (httpx.ConnectError, httpx.TimeoutException):
                pass

            if asyncio.get_event_loop().time() >= deadline:
                raise RuntimeError("Training worker failed to start")

            await asyncio.sleep(interval)


async def start_job(base_url: str, total_batches: int) -> str:
    """POST /start and return the assigned job_id.

    Args:
        base_url: Worker base URL.
        total_batches: Number of training steps to run.

    Returns:
        The ``job_id`` string assigned by the worker.
    """
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{base_url}/start",
            json={"total_batches": total_batches},
            timeout=10.0,
        )
        resp.raise_for_status()
        return resp.json()["job_id"]


async def stop_job(base_url: str) -> None:
    """POST /stop to request the worker halt the current job.

    Args:
        base_url: Worker base URL.
    """
    async with httpx.AsyncClient() as client:
        await client.post(f"{base_url}/stop", timeout=5.0)


async def get_status(base_url: str) -> dict:
    """GET /status and return the raw status dict.

    Args:
        base_url: Worker base URL.

    Returns:
        Dict with keys ``status``, ``batch``, ``total_batches``, ``last_loss``.
    """
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{base_url}/status", timeout=5.0)
        resp.raise_for_status()
        return resp.json()


async def stream_events(base_url: str) -> AsyncIterator[dict]:
    """Connect to GET /stream and yield parsed JSON event dicts as they arrive.

    The generator exits when the SSE stream closes (worker completed or errored)
    or when the worker process disappears (connection error).

    Args:
        base_url: Worker base URL.

    Yields:
        Parsed event dicts (the ``data:`` payload from each SSE line).
    """
    async with httpx.AsyncClient(timeout=None) as client:
        try:
            async with client.stream("GET", f"{base_url}/stream") as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[len("data:"):].strip()
                    if not payload:
                        continue
                    try:
                        event = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    if event:
                        yield event
        except (httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadError):
            # Worker process died mid-stream; stop cleanly.
            return
