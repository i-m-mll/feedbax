from __future__ import annotations
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from feedbax.web.models.graph import GraphSpec
from feedbax.web.models.training import LossTermSpec, TrainingSpec, TaskSpec
from feedbax.web.services.graph_service import GraphService
from feedbax.web.services.loss_service import loss_service, ProbeInfo
from feedbax.web.services.training_service import training_service

router = APIRouter()
graph_service = GraphService()


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


# --- Probe and Loss Configuration Endpoints ---


class ProbeResponse(BaseModel):
    id: str
    label: str
    node: str
    timing: str
    selector: str
    description: Optional[str] = None


@router.get('/probes/{graph_id}')
async def get_available_probes(graph_id: str) -> List[ProbeResponse]:
    """Get all available probes for a graph."""
    try:
        record = graph_service.get_graph(graph_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail='Graph not found') from exc

    probes = loss_service.get_available_probes(record.project.graph)
    return [
        ProbeResponse(
            id=p.id,
            label=p.label,
            node=p.node,
            timing=p.timing,
            selector=p.selector,
            description=p.description,
        )
        for p in probes
    ]


class ValidateLossRequest(BaseModel):
    graph_id: str
    loss_spec: LossTermSpec


class ValidationErrorResponse(BaseModel):
    path: List[str]
    field: str
    message: str


class ValidateLossResponse(BaseModel):
    valid: bool
    errors: List[ValidationErrorResponse]


@router.post('/loss/validate')
async def validate_loss_spec(payload: ValidateLossRequest) -> ValidateLossResponse:
    """Validate a loss specification against a graph."""
    try:
        record = graph_service.get_graph(payload.graph_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail='Graph not found') from exc

    errors = loss_service.validate_loss_spec(payload.loss_spec, record.project.graph)
    return ValidateLossResponse(
        valid=len(errors) == 0,
        errors=[
            ValidationErrorResponse(
                path=e['path'],
                field=e['field'],
                message=e['message'],
            )
            for e in errors
        ],
    )


class ResolveSelectorRequest(BaseModel):
    graph_id: str
    selector: str


@router.post('/loss/resolve-selector')
async def resolve_selector(payload: ResolveSelectorRequest) -> Dict[str, Any]:
    """Resolve a probe selector to its specification."""
    try:
        record = graph_service.get_graph(payload.graph_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail='Graph not found') from exc

    result = loss_service.resolve_probe_selector(
        payload.selector, record.project.graph
    )
    if result is None:
        raise HTTPException(status_code=404, detail='Selector not found')
    return result
