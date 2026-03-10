from __future__ import annotations
import os
import tempfile
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask

from feedbax.web.models.graph import GraphSpec
from feedbax.web.models.training import LossTermSpec, TrainingConfig, TrainingSpec, TaskSpec
from feedbax.web.services.graph_service import GraphService
from feedbax.web.services.loss_service import loss_service, ProbeInfo
from feedbax.web.services.training_service import training_service

router = APIRouter()
graph_service = GraphService()


# ---------------------------------------------------------------------------
# Remote worker management
# ---------------------------------------------------------------------------


class WorkerConnectRequest(BaseModel):
    url: str
    auth_token: Optional[str] = None


class WorkerConnectResponse(BaseModel):
    ok: bool
    url: str


class WorkerStatusResponse(BaseModel):
    mode: str  # "local" | "remote"
    url: Optional[str]
    connected: bool


@router.post('/worker/connect', response_model=WorkerConnectResponse)
async def connect_worker(payload: WorkerConnectRequest):
    """Configure the Studio backend to use a remote training worker.

    Body:
        url: Base URL of the remote worker, e.g. ``http://100.x.x.x:8765``.
        auth_token: Optional bearer token required by the worker.
    """
    training_service.connect_remote(payload.url, payload.auth_token)
    return WorkerConnectResponse(ok=True, url=payload.url)


@router.get('/worker/status', response_model=WorkerStatusResponse)
async def get_worker_status():
    """Return the current worker configuration (local vs remote, URL, health)."""
    mode = training_service.worker_mode()
    url = training_service._base_url
    connected = False
    if url is not None:
        try:
            status = await training_service.get_status()
            connected = status.get("status") != "error"
        except Exception:
            connected = False
    return WorkerStatusResponse(mode=mode, url=url, connected=connected)


class TrainingRequest(BaseModel):
    graph_id: str
    training_spec: TrainingSpec
    task_spec: TaskSpec
    # Optional structured config for Phase 6 real JAX training.
    # When present, passed to the worker which runs actual JAX training.
    # When absent, the worker falls back to the synthetic stub.
    training_config: Optional[TrainingConfig] = None
    # Optional graph spec forwarded to the worker for config inference.
    graph_spec: Optional[dict] = None


@router.post('')
async def start_training(payload: TrainingRequest):
    training_config = (
        payload.training_config.model_dump() if payload.training_config is not None else None
    )
    job_id = await training_service.start_training(
        payload.training_spec.n_batches,
        training_config=training_config,
        training_spec=payload.training_spec.model_dump(),
        task_spec=payload.task_spec.model_dump(),
        graph_spec=payload.graph_spec,
    )
    return {'job_id': job_id}


@router.get('/{job_id}')
async def get_training_status(job_id: str):
    status = await training_service.get_status()
    return {'status': status}


@router.delete('/{job_id}')
async def stop_training(job_id: str):
    await training_service.stop_training()
    return {'success': True}


@router.get('/{job_id}/checkpoint')
async def get_checkpoint(job_id: str):
    checkpoint = await training_service.latest_checkpoint(job_id)
    if checkpoint is None:
        raise HTTPException(status_code=404, detail='Job not found')
    return checkpoint


@router.get('/{job_id}/checkpoint/download')
async def download_checkpoint(job_id: str):
    """Download the serialized checkpoint file for a completed training job.

    Proxies the binary ``.eqx`` file from the worker through the Studio
    backend to the browser.  A temporary file is used for the proxy and
    cleaned up automatically after the response is sent.
    """
    fd, dest = tempfile.mkstemp(suffix=".eqx")
    os.close(fd)
    try:
        await training_service.download_checkpoint(job_id, dest)
    except ValueError:
        os.unlink(dest)
        raise HTTPException(status_code=404, detail='Job not found')
    except RuntimeError as exc:
        os.unlink(dest)
        raise HTTPException(status_code=503, detail=str(exc))
    return FileResponse(
        dest,
        media_type="application/octet-stream",
        filename=f"feedbax_checkpoint_{job_id}.eqx",
        background=BackgroundTask(os.unlink, dest),
    )


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
