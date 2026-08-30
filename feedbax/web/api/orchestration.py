"""Studio operations over the durable controller protocol."""

from __future__ import annotations

import uuid
from typing import Any, Literal

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel, Field

from feedbax.execution.records import invocation_from_document
from feedbax.orchestration.controller import ControllerProtocolError, OperatorGateError
from feedbax.orchestration.realization import backend_plan_from_document
from feedbax.web.orchestration.controller import get_studio_controller
from feedbax.web.orchestration.startup_script import (
    DEFAULT_FEEDBAX_REF,
    DEFAULT_FEEDBAX_REPOSITORY,
    INSTALL_SPEC_SCHEMA_VERSION,
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


class LaunchCostEstimate(BaseModel):
    currency: Literal["USD"] = "USD"
    hourly_estimate: float
    machine_type: str
    preemptible: bool
    basis: str


class InstallSpecRequest(BaseModel):
    schema_version: Literal["feedbax.orchestration.install.v1"] = INSTALL_SPEC_SCHEMA_VERSION
    source: Literal["git"] = "git"
    repository: Literal["https://github.com/mlll-io/feedbax.git"] = DEFAULT_FEEDBAX_REPOSITORY
    ref: str = DEFAULT_FEEDBAX_REF
    extras: tuple[str, ...] = ()


class LaunchRequest(BaseModel):
    invocation: dict[str, Any]
    backend_plan: dict[str, Any]
    project: str
    zone: str
    machine_type: str = "n1-standard-4"
    preemptible: bool = True
    worker_port: int = 8765
    worker_auth_token: str | None = None
    tailscale_auth_key: str | None = None
    install_spec: InstallSpecRequest = Field(default_factory=InstallSpecRequest)
    reservation_ttl_seconds: int = Field(default=300, ge=30, le=3600)


class LaunchResponse(BaseModel):
    status: Literal["awaiting_authentication"]
    intent_id: str
    reservation_id: str
    expires_at: str
    instance_name: str
    cost_estimate: LaunchCostEstimate
    expected_cost: dict[str, Any]


class AuthenticateLaunchRequest(BaseModel):
    operator_identity: str = Field(min_length=1)
    authentication_id: str = Field(min_length=1)
    confirmation_token: str
    max_cost_usd: float = Field(ge=0)


class AuthenticateLaunchResponse(BaseModel):
    status: Literal["authenticated"]
    intent_id: str
    reservation_id: str


class StatusResponse(BaseModel):
    status: str
    intent_id: str | None = None
    reservation_id: str | None = None
    attempt_id: str | None = None
    instance_name: str | None = None
    worker_url: str | None = None
    internal_ip: str | None = None
    external_ip: str | None = None
    error: str | None = None
    orphaned_instance: str | None = None
    expected_cost: dict[str, Any] | None = None
    observed_cost: dict[str, Any] | None = None


class TerminateResponse(BaseModel):
    ok: bool


class OrchestrationTarget(BaseModel):
    id: Literal["local", "gcp", "runpod", "manual"]
    label: str
    billable: bool
    launch_mode: Literal["local", "durable-controller", "execution-plan", "manual-export"]
    available: bool
    notes: list[str] = Field(default_factory=list)


class OrchestrationTargetsResponse(BaseModel):
    targets: list[OrchestrationTarget]


def _estimate_launch_cost(machine_type: str, preemptible: bool) -> LaunchCostEstimate:
    base = _MACHINE_TYPE_ESTIMATES_USD.get(machine_type)
    basis = "static machine-type estimate"
    if base is None:
        pieces = machine_type.rsplit("-", 1)
        base = 0.05 * (int(pieces[1]) if len(pieces) == 2 and pieces[1].isdigit() else 4)
        basis = "vCPU-name estimate"
    hourly = base * (0.30 if preemptible else 1.0)
    return LaunchCostEstimate(
        hourly_estimate=round(hourly, 4),
        machine_type=machine_type,
        preemptible=preemptible,
        basis=basis,
    )


@router.post("/launch", response_model=LaunchResponse)
async def reserve_launch(payload: LaunchRequest) -> LaunchResponse:
    """Create an inert, exact reservation without contacting GCP."""
    try:
        invocation = invocation_from_document(payload.invocation)
        plan = backend_plan_from_document(payload.backend_plan)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if plan.expected_cost is None or plan.billable_confirmation_class != (
        "authenticated-effect-reservation"
    ):
        raise HTTPException(
            status_code=422,
            detail="GCP BackendPlan must carry an authenticated effect reservation cost boundary",
        )
    instance_name = f"feedbax-worker-{uuid.uuid4().hex[:12]}"
    parameters = {
        "project": payload.project,
        "zone": payload.zone,
        "machine_type": payload.machine_type,
        "preemptible": payload.preemptible,
        "worker_port": payload.worker_port,
        "instance_name": instance_name,
        "install_spec": payload.install_spec.model_dump(mode="json"),
        "startup_timeout_seconds": min(plan.timeout_seconds, 900),
        "worker_health_timeout_seconds": min(plan.timeout_seconds, 900),
        "poll_interval_seconds": 2.0,
    }
    try:
        intent, reservation = get_studio_controller(training_service).reserve_cloud_launch(
            invocation,
            plan,
            parameters=parameters,
            ttl_seconds=payload.reservation_ttl_seconds,
            secrets={
                key: value
                for key, value in {
                    "worker_auth_token": payload.worker_auth_token,
                    "tailscale_auth_key": payload.tailscale_auth_key,
                }.items()
                if value is not None
            },
        )
    except (ValueError, ControllerProtocolError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    assert reservation.expires_at is not None
    return LaunchResponse(
        status="awaiting_authentication",
        intent_id=intent.intent_id,
        reservation_id=reservation.reservation_id,
        expires_at=reservation.expires_at.isoformat(),
        instance_name=instance_name,
        cost_estimate=_estimate_launch_cost(payload.machine_type, payload.preemptible),
        expected_cost=plan.expected_cost.model_dump(mode="json"),
    )


@router.post(
    "/intents/{intent_id}/reservations/{reservation_id}/authenticate",
    response_model=AuthenticateLaunchResponse,
)
async def authenticate_launch(
    intent_id: str,
    reservation_id: str,
    payload: AuthenticateLaunchRequest,
    background_tasks: BackgroundTasks,
) -> AuthenticateLaunchResponse:
    """Authenticate one named reservation, then schedule its single dispatch."""
    controller = get_studio_controller(training_service)
    projection = controller.status(intent_id)
    try:
        reserved = projection.reservations[reservation_id]
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Effect reservation not found") from exc
    if payload.confirmation_token != _BILLABLE_CONFIRMATION_TOKEN:
        raise HTTPException(status_code=412, detail="Billable launch confirmation is required")
    expected_cost = reserved.reservation.expected_cost
    if expected_cost is None or payload.max_cost_usd < expected_cost.maximum:
        raise HTTPException(status_code=412, detail="Cost cap is below the exact reserved cost")
    try:
        controller.controller.authenticate_reservation(
            intent_id,
            reservation_id,
            operator_identity=payload.operator_identity,
            authentication_id=payload.authentication_id,
            evidence={
                "confirmation_class": "authenticated-effect-reservation",
                "maximum_cost_usd": payload.max_cost_usd,
            },
        )
    except (OperatorGateError, ControllerProtocolError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    background_tasks.add_task(
        controller.controller.dispatch,
        intent_id,
        reservation_id,
        controller.gcp,
    )
    return AuthenticateLaunchResponse(
        status="authenticated", intent_id=intent_id, reservation_id=reservation_id
    )


@router.get("/status", response_model=StatusResponse)
async def get_orchestration_status(
    intent_id: str | None = Query(default=None),
    refresh: bool = Query(default=True),
) -> StatusResponse:
    controller = get_studio_controller(training_service)
    intent_id = intent_id or controller.latest_intent_id()
    if intent_id is None:
        return StatusResponse(status="idle")
    try:
        projection = (
            await controller.refresh(intent_id) if refresh else controller.status(intent_id)
        )
    except ControllerProtocolError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    reservation = next(reversed(projection.reservations.values()), None)
    attempt = next(reversed(projection.attempts.values()), None)
    observation = attempt.observations[-1] if attempt is not None and attempt.observations else {}
    status = projection.status
    if reservation is not None and reservation.status == "inert":
        status = "awaiting_authentication"
    elif attempt is not None:
        status = attempt.status
    return StatusResponse(
        status=status,
        intent_id=intent_id,
        reservation_id=reservation.reservation.reservation_id if reservation else None,
        attempt_id=attempt.attempt_id if attempt else None,
        instance_name=attempt.provider_resource_handle if attempt else None,
        worker_url=attempt.worker_identity if attempt else None,
        internal_ip=observation.get("internal_ip"),
        external_ip=observation.get("external_ip"),
        error=observation.get("error")
        if attempt is not None and attempt.status == "failed"
        else None,
        orphaned_instance=(
            attempt.provider_resource_handle
            if attempt is not None and attempt.status == "unknown"
            else None
        ),
        expected_cost=(
            reservation.reservation.expected_cost.model_dump(mode="json")
            if reservation is not None and reservation.reservation.expected_cost is not None
            else None
        ),
        observed_cost=(
            reservation.observed_cost.model_dump(mode="json")
            if reservation is not None and reservation.observed_cost is not None
            else None
        ),
    )


@router.get("/intents/{intent_id}/artifacts")
async def inspect_artifacts(intent_id: str) -> dict[str, Any]:
    return {
        "intent_id": intent_id,
        "artifact_refs": list(get_studio_controller(training_service).inspect_artifacts(intent_id)),
    }


@router.delete("/instance", response_model=TerminateResponse)
async def terminate_instance(intent_id: str | None = Query(default=None)) -> TerminateResponse:
    controller = get_studio_controller(training_service)
    intent_id = intent_id or controller.latest_intent_id()
    if intent_id is None:
        return TerminateResponse(ok=True)
    try:
        await controller.terminate(intent_id)
    except (ControllerProtocolError, RuntimeError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return TerminateResponse(ok=True)


@router.get("/targets", response_model=OrchestrationTargetsResponse)
async def list_orchestration_targets() -> OrchestrationTargetsResponse:
    return OrchestrationTargetsResponse(
        targets=[
            OrchestrationTarget(
                id="local",
                label="Local worker",
                billable=False,
                launch_mode="local",
                available=True,
                notes=["Uses StageEngine through the durable controller."],
            ),
            OrchestrationTarget(
                id="gcp",
                label="GCP",
                billable=True,
                launch_mode="durable-controller",
                available=True,
                notes=["Requires authentication of one exact inert reservation."],
            ),
            OrchestrationTarget(
                id="runpod",
                label="RunPod",
                billable=True,
                launch_mode="execution-plan",
                available=True,
                notes=["Uses the same reservation-bound controller protocol."],
            ),
            OrchestrationTarget(
                id="manual",
                label="Manual export",
                billable=False,
                launch_mode="manual-export",
                available=True,
                notes=["Exports an inert plan without starting compute."],
            ),
        ]
    )
