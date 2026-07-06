"""WebSocket handler that relays worker SSE events to the frontend."""
from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from feedbax.web.services.training_service import training_service

router = APIRouter()


@router.websocket('/training/{job_id}')
async def training_ws(websocket: WebSocket, job_id: str):
    await websocket.accept()
    try:
        async for event in training_service.stream_progress(job_id):
            await websocket.send_json(event.raw)
    except WebSocketDisconnect:
        return
    except RuntimeError as exc:
        if "websocket.close" in str(exc) or "websocket.disconnect" in str(exc):
            return
        raise
    except Exception as exc:
        if websocket.application_state == WebSocketState.CONNECTED:
            try:
                await websocket.send_json(
                    {
                        'type': 'training_error',
                        'job_id': job_id,
                        'error': str(exc),
                    }
                )
            except (WebSocketDisconnect, RuntimeError):
                return
    finally:
        if websocket.application_state == WebSocketState.CONNECTED:
            try:
                await websocket.close()
            except RuntimeError:
                return
