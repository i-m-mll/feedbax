"""WebSocket handler that relays worker SSE events to the frontend."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from feedbax.web.services.training_service import training_service
from feedbax.web.worker.transport import SSEConsumptionBudget, WorkerTransportError

router = APIRouter()


@router.websocket("/training/{job_id}")
async def training_ws(websocket: WebSocket, job_id: str):
    await websocket.accept()
    consumption = SSEConsumptionBudget(training_service.worker_limits())
    try:
        async for event in training_service.stream_progress(job_id):
            consumption.admit_event(
                json.dumps(event.raw, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            )
            async with asyncio.timeout(consumption.limits.write_seconds):
                await websocket.send_json(event.raw)
    except WebSocketDisconnect:
        return
    except WorkerTransportError as exc:
        if websocket.application_state == WebSocketState.CONNECTED:
            try:
                await websocket.send_json(training_service.make_error_event(job_id, str(exc)).raw)
            except (WebSocketDisconnect, RuntimeError):
                return
    except RuntimeError as exc:
        if "websocket.close" in str(exc) or "websocket.disconnect" in str(exc):
            return
        raise
    except Exception:
        if websocket.application_state == WebSocketState.CONNECTED:
            try:
                await websocket.send_json(
                    training_service.make_error_event(job_id, "Training worker stream failed.").raw
                )
            except (WebSocketDisconnect, RuntimeError):
                return
    finally:
        if websocket.application_state == WebSocketState.CONNECTED:
            try:
                await websocket.close()
            except RuntimeError:
                return
