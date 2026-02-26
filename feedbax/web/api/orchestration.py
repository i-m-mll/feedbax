"""REST endpoints for GCP cloud instance orchestration.

Endpoints
---------
POST /api/orchestration/launch
    Start creating a GCP instance; returns immediately with ``status="creating"``.

GET /api/orchestration/status
    Return the current orchestration state, refreshing from GCP if active.

DELETE /api/orchestration/instance
    Terminate the running instance and disconnect the TrainingService.
"""
from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from feedbax.web.orchestration.gcp import InstanceConfig
from feedbax.web.orchestration.manager import orchestration_manager
from feedbax.web.services.training_service import training_service

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class LaunchRequest(BaseModel):
    project: str
    zone: str
    machine_type: str = "n1-standard-4"
    preemptible: bool = True
    worker_port: int = 8765
    auth_token: Optional[str] = None
    ts_auth_key: Optional[str] = None


class LaunchResponse(BaseModel):
    status: str
    instance_name: Optional[str] = None
    worker_url: Optional[str] = None


class StatusResponse(BaseModel):
    status: str
    instance_name: Optional[str] = None
    worker_url: Optional[str] = None
    internal_ip: Optional[str] = None
    external_ip: Optional[str] = None
    error: Optional[str] = None


class TerminateResponse(BaseModel):
    ok: bool


# ---------------------------------------------------------------------------
# Background launch helper
# ---------------------------------------------------------------------------


async def _launch_background(config: InstanceConfig, instance_name: str) -> None:
    """Background coroutine that drives the full instance launch sequence."""
    await orchestration_manager.launch(config, training_service, instance_name)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/launch", response_model=LaunchResponse)
async def launch_instance(payload: LaunchRequest, background_tasks: BackgroundTasks):
    """Create a GCP compute instance and start the Feedbax worker.

    The endpoint returns immediately with ``status="creating"`` while the
    instance is being provisioned in the background.  Poll
    ``GET /api/orchestration/status`` to track progress.

    Body:
        project: GCP project ID.
        zone: GCP zone, e.g. ``"us-central1-a"``.
        machine_type: GCP machine type (default ``"n1-standard-4"``).
        preemptible: Use a preemptible/spot instance (default ``true``).
        worker_port: Port the worker will bind to (default ``8765``).
        auth_token: Optional bearer token for the worker.
        ts_auth_key: Optional Tailscale auth key.
    """
    config = InstanceConfig(
        project=payload.project,
        zone=payload.zone,
        machine_type=payload.machine_type,
        preemptible=payload.preemptible,
        worker_port=payload.worker_port,
        auth_token=payload.auth_token,
        ts_auth_key=payload.ts_auth_key,
    )

    short_id = uuid.uuid4().hex[:6]
    instance_name = f"feedbax-worker-{short_id}"

    background_tasks.add_task(_launch_background, config, instance_name)

    return LaunchResponse(status="creating", instance_name=instance_name)


@router.get("/status", response_model=StatusResponse)
async def get_orchestration_status():
    """Return the current orchestration state.

    When an instance is active (status in ``creating``, ``connecting``, or
    ``running``), also refreshes from GCP to detect preemption.
    """
    state = orchestration_manager.state

    if state.instance is not None and state.status in ("creating", "connecting", "running"):
        try:
            state = await orchestration_manager.refresh_status()
        except Exception:
            # Refresh failed — return cached state.
            pass

    instance = state.instance
    return StatusResponse(
        status=state.status,
        instance_name=instance.name if instance else None,
        worker_url=state.worker_url,
        internal_ip=instance.internal_ip if instance else None,
        external_ip=instance.external_ip if instance else None,
        error=state.error,
    )


@router.delete("/instance", response_model=TerminateResponse)
async def terminate_instance():
    """Terminate the current GCP instance and disconnect the TrainingService.

    If no instance is running, returns ``ok=true`` without error.
    """
    state = orchestration_manager.state
    if state.instance is None:
        return TerminateResponse(ok=True)

    try:
        await orchestration_manager.terminate(training_service)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return TerminateResponse(ok=True)
