from __future__ import annotations
from fastapi import APIRouter, WebSocket
from feedbax.web.services.training_service import training_service, TrainingStatus

router = APIRouter()


@router.websocket('/training/{job_id}')
async def training_ws(websocket: WebSocket, job_id: str):
    await websocket.accept()
    try:
        # Emit a log message at the start of training.
        try:
            job_total = training_service._jobs[job_id].total_batches
        except KeyError:
            await websocket.send_json(
                {
                    'type': 'training_error',
                    'job_id': job_id,
                    'error': 'Job not found',
                }
            )
            return

        await websocket.send_json(
            {
                'type': 'training_log',
                'job_id': job_id,
                'batch': 0,
                'level': 'info',
                'message': f'Training started: {job_total} steps',
            }
        )

        async for progress in training_service.stream_progress(job_id):
            await websocket.send_json(
                {
                    'type': 'training_progress',
                    'job_id': job_id,
                    'batch': progress.batch,
                    'total_batches': progress.total_batches,
                    'loss': progress.loss,
                    'loss_terms': progress.loss_terms,
                    'grad_norm': progress.grad_norm,
                    'step_time_ms': progress.step_time_ms,
                    'status': progress.status,
                }
            )
            for line in progress.log_lines:
                await websocket.send_json(
                    {
                        'type': 'training_log',
                        'job_id': job_id,
                        'batch': progress.batch,
                        'level': 'info',
                        'message': line,
                    }
                )

        status = training_service.get_status(job_id)
        final_loss = training_service.last_loss(job_id) or 0.0

        if status == TrainingStatus.COMPLETED:
            await websocket.send_json(
                {
                    'type': 'training_log',
                    'job_id': job_id,
                    'batch': job_total,
                    'level': 'info',
                    'message': f'Training complete | final loss={final_loss:.4f}',
                }
            )
            await websocket.send_json(
                {
                    'type': 'training_complete',
                    'job_id': job_id,
                    'final_loss': final_loss,
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
