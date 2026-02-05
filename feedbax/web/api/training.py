from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from feedbax.web.models.training import TrainingSpec, TaskSpec
from feedbax.web.services.training_service import training_service

router = APIRouter()


class TrainingRequest(BaseModel):
    graph_id: str
    training_spec: TrainingSpec
    task_spec: TaskSpec


@router.post('')
async def start_training(payload: TrainingRequest):
    job_id = training_service.start_training(payload.training_spec.n_batches)
    return {'job_id': job_id}


@router.get('/{job_id}')
async def get_training_status(job_id: str):
    try:
        status = training_service.get_status(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail='Job not found') from exc
    return {'status': status}


@router.delete('/{job_id}')
async def stop_training(job_id: str):
    training_service.stop_training(job_id)
    return {'success': True}


@router.get('/{job_id}/checkpoint')
async def get_checkpoint(job_id: str):
    checkpoint = training_service.latest_checkpoint(job_id)
    if checkpoint is None:
        raise HTTPException(status_code=404, detail='Job not found')
    return checkpoint
