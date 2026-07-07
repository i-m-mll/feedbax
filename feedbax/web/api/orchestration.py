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
from typing import Literal, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field, field_validator

from feedbax.web.orchestration.gcp import InstanceConfig
from feedbax.web.orchestration.manager import orchestration_manager
from feedbax.web.orchestration.startup_script import (
    DEFAULT_FEEDBAX_REF,
    DEFAULT_FEEDBAX_REPOSITORY,
    INSTALL_SPEC_SCHEMA_VERSION,
    FeedbaxInstallSpec,
)
from feedbax.web.services.training_service import training_service

router = APIRouter()

_BILLABLE_CONFIRMATION_TOKEN = "launch-billable-gcp-worker"
_MACHINE_TYPE_ESTIMATES_USD = {
    "n1-standard-4": 0.20,
    "n1-standard-8": 0.40,
    "n2-standard-4": 0.22,
    "n2-standard-8": 0.44,
}
_PREEMPTIBLE_DISCOUNT = 0.30


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class LaunchCostEstimate(BaseModel):
    currency: Literal["USD"] = "USD"
    hourly_estimate: float
    machine_type: str
    preemptible: bool
    basis: str


class InstallSpecRequest(BaseModel):
    schema_version: Literal["feedbax.orchestration.install.v1"] = INSTALL_SPEC_SCHEMA_VERSION
    source: Literal["git"] = "git"
    repository: Literal["https://github.com/mlll-io/feedbax.git"] = (
        DEFAULT_FEEDBAX_REPOSITORY
    )
    ref: str = DEFAULT_FEEDBAX_REF
    extras: tuple[str, ...] = ()

    @field_validator("ref")
    @classmethod
    def _validate_ref(cls, value: str) -> str:
        FeedbaxInstallSpec(ref=value)
        return value

    @field_validator("extras")
    @classmethod
    def _validate_extras(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        FeedbaxInstallSpec(extras=value)
        return value

    def to_domain(self) -> FeedbaxInstallSpec:
        return FeedbaxInstallSpec(
            schema_version=self.schema_version,
            source=self.source,
            repository=self.repository,
            ref=self.ref,
            extras=self.extras,
        )


class LaunchRequest(BaseModel):
    project: str
    zone: str
    machine_type: str = "n1-standard-4"
    preemptible: bool = True
    worker_port: int = 8765
    auth_token: Optional[str] = None
    ts_auth_key: Optional[str] = None
    install_spec: InstallSpecRequest = Field(default_factory=InstallSpecRequest)
    confirm_billable_launch: bool = False
    confirmation_token: Optional[str] = None
    max_hourly_cost_usd: Optional[float] = None


class LaunchResponse(BaseModel):
    status: str
    instance_name: Optional[str] = None
    worker_url: Optional[str] = None
    cost_estimate: Optional[LaunchCostEstimate] = None


class StatusResponse(BaseModel):
    status: str
    instance_name: Optional[str] = None
    worker_url: Optional[str] = None
    internal_ip: Optional[str] = None
    external_ip: Optional[str] = None
    error: Optional[str] = None
    orphaned_instance: Optional[str] = None
    worker_health_failures: int = 0


class TerminateResponse(BaseModel):
    ok: bool


class OrchestrationTarget(BaseModel):
    id: Literal["local", "gcp", "runpod", "manual"]
    label: str
    billable: bool
    launch_mode: Literal["local", "web-orchestration", "execution-plan", "manual-export"]
    available: bool
    confirmation_token: Optional[str] = None
    notes: list[str] = Field(default_factory=list)


class OrchestrationTargetsResponse(BaseModel):
    targets: list[OrchestrationTarget]


# ---------------------------------------------------------------------------
# Background launch helper
# ---------------------------------------------------------------------------


async def _launch_background(config: InstanceConfig, instance_name: str) -> None:
    """Background coroutine that drives the full instance launch sequence."""
    await orchestration_manager.launch(
        config,
        training_service,
        instance_name,
        reserved=True,
    )


def _estimate_launch_cost(payload: LaunchRequest) -> LaunchCostEstimate:
    """Return a deterministic launch cost surface for confirmation."""
    base = _MACHINE_TYPE_ESTIMATES_USD.get(payload.machine_type)
    basis = "static machine-type estimate"
    if base is None:
        base = 0.05
        basis = "fallback vCPU estimate"
        pieces = payload.machine_type.rsplit("-", 1)
        if len(pieces) == 2 and pieces[1].isdigit():
            base *= int(pieces[1])
        else:
            base *= 4
    hourly = base * (_PREEMPTIBLE_DISCOUNT if payload.preemptible else 1.0)
    return LaunchCostEstimate(
        hourly_estimate=round(hourly, 4),
        machine_type=payload.machine_type,
        preemptible=payload.preemptible,
        basis=basis,
    )


def _require_launch_confirmation(payload: LaunchRequest) -> LaunchCostEstimate:
    """Reject billable launches unless the caller acknowledged cost and cap."""
    estimate = _estimate_launch_cost(payload)
    if (
        not payload.confirm_billable_launch
        or payload.confirmation_token != _BILLABLE_CONFIRMATION_TOKEN
    ):
        raise HTTPException(
            status_code=412,
            detail={
                "error": "billable_launch_confirmation_required",
                "required_confirmation_token": _BILLABLE_CONFIRMATION_TOKEN,
                "cost_estimate": estimate.model_dump(),
            },
        )
    if payload.max_hourly_cost_usd is None:
        raise HTTPException(
            status_code=412,
            detail={
                "error": "max_hourly_cost_usd_required",
                "cost_estimate": estimate.model_dump(),
            },
        )
    if payload.max_hourly_cost_usd < estimate.hourly_estimate:
        raise HTTPException(
            status_code=412,
            detail={
                "error": "max_hourly_cost_usd_below_estimate",
                "cost_estimate": estimate.model_dump(),
                "max_hourly_cost_usd": payload.max_hourly_cost_usd,
            },
        )
    return estimate


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
    cost_estimate = _require_launch_confirmation(payload)

    config = InstanceConfig(
        project=payload.project,
        zone=payload.zone,
        machine_type=payload.machine_type,
        preemptible=payload.preemptible,
        worker_port=payload.worker_port,
        auth_token=payload.auth_token,
        ts_auth_key=payload.ts_auth_key,
        install_spec=payload.install_spec.to_domain(),
    )

    short_id = uuid.uuid4().hex[:6]
    instance_name = f"feedbax-worker-{short_id}"

    try:
        await orchestration_manager.reserve_launch(config, instance_name)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    background_tasks.add_task(_launch_background, config, instance_name)

    return LaunchResponse(
        status="creating",
        instance_name=instance_name,
        cost_estimate=cost_estimate,
    )


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
        orphaned_instance=state.orphaned_instance,
        worker_health_failures=state.worker_health_failures,
    )


@router.get("/targets", response_model=OrchestrationTargetsResponse)
async def list_orchestration_targets():
    """Return execution targets available to Studio queue orchestration."""
    return OrchestrationTargetsResponse(
        targets=[
            OrchestrationTarget(
                id="local",
                label="Local worker",
                billable=False,
                launch_mode="local",
                available=True,
                notes=["Uses the existing local Studio execution path."],
            ),
            OrchestrationTarget(
                id="gcp",
                label="GCP",
                billable=True,
                launch_mode="web-orchestration",
                available=True,
                confirmation_token=_BILLABLE_CONFIRMATION_TOKEN,
                notes=["Uses the existing GCP VM orchestration launch endpoint."],
            ),
            OrchestrationTarget(
                id="runpod",
                label="RunPod",
                billable=True,
                launch_mode="execution-plan",
                available=True,
                confirmation_token="confirm-runpod-queue-launch",
                notes=[
                    "Queue launch prepares a RunPod ExecutionPlan with repository script commands.",
                    "Real pod acquisition remains owned by scripts/deploy/runpod_deploy.sh.",
                ],
            ),
            OrchestrationTarget(
                id="manual",
                label="Manual export",
                billable=False,
                launch_mode="manual-export",
                available=True,
                notes=["Exports operator instructions without starting compute."],
            ),
        ]
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
