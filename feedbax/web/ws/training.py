from __future__ import annotations
from fastapi import APIRouter, WebSocket
from feedbax.web.services.training_service import training_service, TrainingStatus

router = APIRouter()


@router.websocket('/training/{job_id}')
async def training_ws(websocket: WebSocket, job_id: str):
    await websocket.accept()
    try:
        async for progress in training_service.stream_progress(job_id):
            await websocket.send_json(
                {
                    'type': 'training_progress',
                    'job_id': job_id,
                    'batch': progress.batch,
                    'total_batches': progress.total_batches,
                    'loss': progress.loss,
                    'metrics': progress.metrics,
                }
            )

        status = training_service.get_status(job_id)
        if status == TrainingStatus.COMPLETED:
            await websocket.send_json(
                {
                    'type': 'training_complete',
                    'job_id': job_id,
                    'final_loss': training_service.last_loss(job_id) or 0.0,
                    'checkpoint_path': None,
                }
            )
    except KeyError:
        await websocket.send_json(
            {
                'type': 'training_error',
                'job_id': job_id,
                'error': 'Job not found',
            }
        )
    finally:
        await websocket.close()
