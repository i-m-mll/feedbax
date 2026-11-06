"""Async HTTP client for the Feedbax training worker.

Used by the Studio backend (TrainingService) to:
- Poll the worker's /health endpoint on startup.
- Start / stop training jobs via POST.
- Relay the SSE event stream to WebSocket clients.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import replace
from typing import AsyncIterator, Optional

import httpx

from feedbax.orchestration.events import RUN_EVENT_TERMINAL_TYPES
from feedbax.web.worker.identity import require_worker_job_id
from feedbax.web.worker.transport import (
    SSEConsumptionBudget,
    WorkerConsumptionError,
    WorkerEndpoint,
    WorkerEventCursorExpired,
    WorkerResponseError,
    WorkerTransportError,
    _check_content_length,
    _check_response,
    check_event_stream_response,
    decode_worker_event,
    request_json,
)

# Maximum reconnection attempts for the SSE stream.
_MAX_RECONNECT_ATTEMPTS = 10
# Seconds to wait between reconnection attempts.
_RECONNECT_DELAY = 2.0


class WorkerEventStreamError(WorkerTransportError):
    """Raised when the worker event stream cannot be relayed safely."""


class _WorkerEventStreamEnded(Exception):
    """Internal signal for a non-terminal clean end of the SSE response."""


async def wait_for_health(
    endpoint: WorkerEndpoint,
    timeout: float = 5.0,
    interval: float = 0.1,
) -> None:
    """Poll GET /health until the worker responds OK, or raise RuntimeError on timeout.

    Args:
        endpoint: Validated worker endpoint and request policy.
        timeout: Maximum seconds to wait.
        interval: Seconds between poll attempts.

    Raises:
        RuntimeError: If the worker does not respond within *timeout* seconds.
    """
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise WorkerTransportError("Training worker health check failed")
        limits = endpoint.limits
        attempt_endpoint = replace(
            endpoint,
            limits=replace(
                limits,
                connect_seconds=min(limits.connect_seconds, remaining),
                read_seconds=min(limits.read_seconds, remaining),
                write_seconds=min(limits.write_seconds, remaining),
                pool_seconds=min(limits.pool_seconds, remaining),
                request_seconds=min(limits.request_seconds, remaining),
            ),
        )
        try:
            await request_json(attempt_endpoint, "GET", "/health")
            return
        except (WorkerTransportError, WorkerResponseError):
            if asyncio.get_running_loop().time() >= deadline:
                raise WorkerTransportError("Training worker health check failed")
        await asyncio.sleep(interval)


async def start_job(
    endpoint: WorkerEndpoint,
    total_batches: int,
    *,
    job_id: str,
    run_set_id: str,
    training_config: Optional[dict] = None,
    training_spec: Optional[dict] = None,
    task_spec: Optional[dict] = None,
    task_binding_spec: Optional[dict] = None,
    graph_spec: Optional[dict] = None,
) -> str:
    """POST /start and return the assigned job_id.

    Args:
        endpoint: Validated worker endpoint and request policy.
        total_batches: Number of training steps to run.
        training_config: Optional dict forwarded to the worker as the
            ``training_config`` key. When ``None``, the worker uses default
            training configuration values; real JAX training is always attempted.
        training_spec: Optional spec dict with optimizer type/params and loss
            weights; forwarded to the worker for spec-driven optimizer construction.
        task_spec: Optional task spec dict with task parameters such as
            ``n_reach_steps`` and ``effort_weight``; forwarded to the worker.

    Returns:
        The ``job_id`` string assigned by the worker.
    """
    require_worker_job_id(job_id)
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
    response = await request_json(endpoint, "POST", "/start", json_body=body)
    job_id_response = response.get("job_id")
    if not isinstance(job_id_response, str):
        raise WorkerTransportError("Training worker returned an invalid job identity")
    return job_id_response


async def stop_job(endpoint: WorkerEndpoint, job_id: str) -> None:
    """POST /jobs/{job_id}/stop to request the worker halt a job.

    Args:
        endpoint: Validated worker endpoint and request policy.
        job_id: Worker job identifier.
    """
    require_worker_job_id(job_id)
    await request_json(endpoint, "POST", f"/jobs/{job_id}/stop")


async def get_status(endpoint: WorkerEndpoint, job_id: str) -> dict:
    """GET /jobs/{job_id}/status and return the raw status dict.

    Args:
        endpoint: Validated worker endpoint and request policy.
        job_id: Worker job identifier.

    Returns:
        Dict with keys ``status``, ``batch``, ``total_batches``, ``last_loss``.
    """
    require_worker_job_id(job_id)
    return await request_json(endpoint, "GET", f"/jobs/{job_id}/status")


async def get_checkpoint(endpoint: WorkerEndpoint, job_id: str) -> dict:
    """GET /jobs/{job_id}/checkpoint and return the raw checkpoint metadata dict.

    Args:
        endpoint: Validated worker endpoint and request policy.
        job_id: Worker job identifier.

    Returns:
        Dict with keys ``batch``, ``loss``, ``weights_available``.
    """
    require_worker_job_id(job_id)
    return await request_json(endpoint, "GET", f"/jobs/{job_id}/checkpoint")


async def get_manifest(endpoint: WorkerEndpoint, job_id: str) -> dict:
    """GET /jobs/{job_id}/manifest and return a job's durable manifest."""
    require_worker_job_id(job_id)
    return await request_json(endpoint, "GET", f"/jobs/{job_id}/manifest")


async def download_checkpoint(
    endpoint: WorkerEndpoint,
    job_id: str,
    dest_path: str,
) -> None:
    """Stream GET /jobs/{job_id}/checkpoint/download and write to *dest_path*.

    Args:
        endpoint: Validated worker endpoint and request policy.
        job_id: Worker job identifier.
        dest_path: Local filesystem path to write the checkpoint bytes to.

    Raises:
        WorkerTransportError: If the worker response cannot be consumed safely.
    """
    require_worker_job_id(job_id)
    partial_path = dest_path + ".part"
    try:
        async with asyncio.timeout(endpoint.limits.checkpoint_seconds):
            await endpoint.revalidate_address_async()
            async with httpx.AsyncClient(
                timeout=endpoint.limits.httpx_timeout(),
                follow_redirects=False,
            ) as client:
                async with client.stream(
                    "GET",
                    endpoint.url(f"/jobs/{job_id}/checkpoint/download"),
                    headers=endpoint.authorization_headers(),
                ) as response:
                    _check_response(response)
                    _check_content_length(response, endpoint.limits.checkpoint_bytes)
                    total = 0
                    with open(partial_path, "wb") as handle:
                        async for chunk in response.aiter_bytes():
                            total += len(chunk)
                            if total > endpoint.limits.checkpoint_bytes:
                                raise WorkerConsumptionError(
                                    "Training worker response exceeded its byte budget"
                                )
                            handle.write(chunk)
        os.replace(partial_path, dest_path)
    except (WorkerTransportError, WorkerResponseError):
        raise
    except TimeoutError as exc:
        raise WorkerConsumptionError(
            "Training worker checkpoint exceeded its overall time budget"
        ) from exc
    except httpx.HTTPError as exc:
        raise WorkerTransportError("Training worker checkpoint request failed") from exc
    finally:
        try:
            os.unlink(partial_path)
        except FileNotFoundError:
            pass


async def stream_events(
    endpoint: WorkerEndpoint,
    job_id: str,
    *,
    from_seq: int | None = None,
) -> AsyncIterator[dict]:
    """Connect to GET /jobs/{job_id}/stream and yield parsed JSON event dicts.

    Resume inclusively from ``from_seq`` when supplied. After a connection
    error, retry from one past the last delivered worker event, up to
    ``_MAX_RECONNECT_ATTEMPTS`` times.

    The generator exits cleanly after a terminal event. Stream failures are
    retried where classified as reconnectable, then surfaced to the caller.

    Args:
        endpoint: Validated worker endpoint and request policy.
        job_id: Worker job identifier.
        from_seq: Inclusive cursor for the first request.

    Yields:
        Parsed event dicts (the ``data:`` payload from each SSE line).

    Raises:
        WorkerEventStreamError: If reconnect attempts are exhausted or an
            unclassified stream failure occurs.
    """
    require_worker_job_id(job_id)
    last_seq: Optional[int] = None
    attempt = 0
    consumption = SSEConsumptionBudget(endpoint.limits)

    while attempt <= _MAX_RECONNECT_ATTEMPTS:
        expected_seq = from_seq if last_seq is None else last_seq + 1
        params = {} if expected_seq is None else {"from_seq": expected_seq}
        # A truncated line from a closed response cannot continue in a new response.
        consumption.buffered.clear()

        try:
            consumption.check_lifetime()
            await endpoint.revalidate_address_async()
            async with httpx.AsyncClient(
                timeout=endpoint.limits.httpx_timeout(),
                follow_redirects=False,
            ) as client:
                async with client.stream(
                    "GET",
                    endpoint.url(f"/jobs/{job_id}/stream"),
                    params=params,
                    headers=endpoint.authorization_headers(),
                ) as resp:
                    check_event_stream_response(resp)
                    if resp.status_code == 204:
                        return
                    report_resume = attempt > 0
                    async for chunk in resp.aiter_bytes():
                        for payload_bytes in consumption.feed(chunk):
                            event = decode_worker_event(payload_bytes)
                            event_seq = event.get("seq")
                            if event_seq is not None and (
                                type(event_seq) is not int or event_seq < 0
                            ):
                                raise WorkerEventStreamError("Training worker event stream failed.")
                            if expected_seq is not None:
                                if event_seq is None or event_seq > expected_seq:
                                    raise WorkerEventStreamError(
                                        "Training worker event stream has a sequence gap."
                                    )
                                if event_seq < expected_seq:
                                    continue
                            if report_resume:
                                yield {
                                    "type": "training_resync",
                                    "job_id": job_id,
                                    "expected_worker_seq": expected_seq,
                                    "observed_worker_seq": event_seq,
                                    "missed_events": 0,
                                    "reason": "resumed",
                                    "message": "Training stream resumed after reconnect.",
                                }
                                report_resume = False
                            if event_seq is not None:
                                last_seq = event_seq
                                expected_seq = event_seq + 1
                            yield event
                            if event.get("type") in RUN_EVENT_TERMINAL_TYPES or event.get(
                                "type"
                            ) in ("training_complete", "training_stopped", "training_error"):
                                return
            raise _WorkerEventStreamEnded
        except (
            _WorkerEventStreamEnded,
            httpx.ConnectError,
            httpx.RemoteProtocolError,
            httpx.ReadError,
            httpx.ReadTimeout,
        ) as exc:
            attempt += 1
            if attempt > _MAX_RECONNECT_ATTEMPTS:
                raise WorkerEventStreamError("Training worker event stream failed.") from exc
            await asyncio.sleep(_RECONNECT_DELAY)
        except (WorkerEventCursorExpired, WorkerEventStreamError):
            raise
        except Exception as exc:
            # Do not expose upstream response bodies, URLs, or transport details.
            raise WorkerEventStreamError("Training worker event stream failed.") from exc
