"""WebSocket handler that relays worker SSE events to the frontend."""
from __future__ import annotations

from fastapi import APIRouter, WebSocket

from feedbax.web.services.training_service import training_service

router = APIRouter()


@router.websocket('/training/{job_id}')
async def training_ws(websocket: WebSocket, job_id: str):
    await websocket.accept()
    try:
        async for event in training_service.stream_progress(job_id):
            await websocket.send_json(event.raw)
    except Exception as exc:
        await websocket.send_json(
            {
                'type': 'training_error',
                'job_id': job_id,
                'error': str(exc),
            }
        )
    finally:
        await websocket.close()
