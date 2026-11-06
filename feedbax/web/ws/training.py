"""WebSocket handler that relays worker SSE events to the frontend."""

from __future__ import annotations

import asyncio
import json
from contextlib import aclosing
from typing import Annotated

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from feedbax.web.services.training_service import training_service
from feedbax.web.worker.transport import (
    SSEConsumptionBudget,
    WorkerConsumptionLimits,
    WorkerTransportError,
)

router = APIRouter()


async def _wait_for_disconnect(websocket: WebSocket) -> None:
    while (await websocket.receive())["type"] != "websocket.disconnect":
        pass


async def _send_json(websocket: WebSocket, payload: dict, write_seconds: float) -> None:
    async with asyncio.timeout(write_seconds):
        await websocket.send_json(payload)


async def _relay_progress(
    websocket: WebSocket, job_id: str, from_seq: int | None, limits: WorkerConsumptionLimits
) -> None:
    consumption = SSEConsumptionBudget(limits)
    try:
        async with aclosing(training_service.stream_progress(job_id, from_seq=from_seq)) as events:
            async for event in events:
                consumption.admit_event(
                    json.dumps(event.raw, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
                )
                await _send_json(websocket, event.raw, limits.write_seconds)
        return
    except WebSocketDisconnect:
        return
    except WorkerTransportError as exc:
        message = str(exc)
    except RuntimeError as exc:
        if "websocket.close" in str(exc) or "websocket.disconnect" in str(exc):
            return
        raise
    except Exception:
        message = "Training worker stream failed."

    if websocket.application_state == WebSocketState.CONNECTED:
        try:
            error = training_service.make_error_event(job_id, message)
            await _send_json(websocket, error.raw, limits.write_seconds)
        except (WebSocketDisconnect, RuntimeError, TimeoutError):
            return


@router.websocket("/training/{job_id}")
async def training_ws(
    websocket: WebSocket,
    job_id: str,
    from_seq: Annotated[int | None, Query(ge=0)] = None,
):
    await websocket.accept()
    limits = training_service.worker_limits()
    relay = asyncio.create_task(_relay_progress(websocket, job_id, from_seq, limits))
    disconnect = asyncio.create_task(_wait_for_disconnect(websocket))
    try:
        done, _ = await asyncio.wait((relay, disconnect), return_when=asyncio.FIRST_COMPLETED)
        if disconnect in done:
            disconnect.result()
            return
        relay.result()
    finally:
        relay.cancel()
        disconnect.cancel()
        await asyncio.gather(relay, disconnect, return_exceptions=True)
        if (
            websocket.client_state != WebSocketState.DISCONNECTED
            and websocket.application_state == WebSocketState.CONNECTED
        ):
            try:
                async with asyncio.timeout(limits.write_seconds):
                    await websocket.close()
            except (WebSocketDisconnect, RuntimeError, TimeoutError):
                pass
