"""Studio operations over the durable controller protocol."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from feedbax.execution.records import invocation_from_document
from feedbax.orchestration.controller import ControllerProtocolError, OperatorGateError
from feedbax.web.orchestration.controller import (
    StudioController,
    StudioRecoveryUnavailable,
    get_studio_controller,
)
from feedbax.web.services.training_service import training_service


router = APIRouter()
_BILLABLE_CONFIRMATION_TOKEN = "launch-billable-gcp-worker"


async def _studio_controller(request: Request) -> StudioController:
    controller = getattr(request.app.state, "studio_controller", None)
    return controller if controller is not None else get_studio_controller(training_service)


_Controller = Annotated[StudioController, Depends(_studio_controller)]


class AuthenticateLaunchRequest(BaseModel):
    operator_identity: str = Field(min_length=1)
    authentication_id: str = Field(min_length=1)
    confirmation_token: str
    max_cost_usd: float = Field(ge=0)


class AuthenticateLaunchResponse(BaseModel):
    status: Literal["authenticated"]
    intent_id: str
    reservation_id: str


class RetryLaunchRequest(BaseModel):
    invocation: dict[str, Any]
    reservation_ttl_seconds: int = Field(default=300, ge=30, le=3600)


class RetryLaunchResponse(BaseModel):
    status: Literal["awaiting_authentication"]
    intent_id: str
    reservation_id: str
    external_effect_key: str
    expires_at: str


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
    orphaned_resources: list[dict[str, Any]] = Field(default_factory=list)
    expected_cost: dict[str, Any] | None = None
    observed_cost: dict[str, Any] | None = None


class TerminateResponse(BaseModel):
    ok: bool


@router.post(
    "/intents/{intent_id}/reservations/{reservation_id}/authenticate",
    response_model=AuthenticateLaunchResponse,
)
async def authenticate_launch(
    intent_id: str,
    reservation_id: str,
    payload: AuthenticateLaunchRequest,
    background_tasks: BackgroundTasks,
    controller: _Controller,
) -> AuthenticateLaunchResponse:
    """Authenticate one named reservation, then schedule its single dispatch."""
    try:
        controller.require_recovered(intent_id)
        projection = controller.status(intent_id)
        reserved = projection.reservations[reservation_id]
    except StudioRecoveryUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Effect reservation not found") from exc
    if payload.confirmation_token != _BILLABLE_CONFIRMATION_TOKEN:
        raise HTTPException(status_code=412, detail="Billable launch confirmation is required")
    expected_cost = reserved.reservation.expected_cost
    if expected_cost is None or payload.max_cost_usd < expected_cost.maximum:
        raise HTTPException(status_code=412, detail="Cost cap is below the exact reserved cost")
    try:
        controller.authenticate_reservation(
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


@router.post(
    "/intents/{intent_id}/reservations/{reservation_id}/retry",
    response_model=RetryLaunchResponse,
)
async def admit_retry(
    intent_id: str,
    reservation_id: str,
    payload: RetryLaunchRequest,
    controller: _Controller,
) -> RetryLaunchResponse:
    """Admit one same-effect retry under the exact Invocation execution policy."""

    try:
        invocation = invocation_from_document(payload.invocation)
        reservation = controller.admit_retry(
            intent_id,
            invocation,
            reservation_id,
            ttl_seconds=payload.reservation_ttl_seconds,
        )
    except StudioRecoveryUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (ValueError, ControllerProtocolError, OperatorGateError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    assert reservation.expires_at is not None
    return RetryLaunchResponse(
        status="awaiting_authentication",
        intent_id=intent_id,
        reservation_id=reservation.reservation_id,
        external_effect_key=reservation.external_effect_key,
        expires_at=reservation.expires_at.isoformat(),
    )


@router.get("/status", response_model=StatusResponse)
async def get_orchestration_status(
    controller: _Controller,
    intent_id: str | None = Query(default=None),
    refresh: bool = Query(default=True),
) -> StatusResponse:
    intent_id = intent_id or controller.latest_intent_id()
    if intent_id is None:
        return StatusResponse(status="idle")
    try:
        projection = (
            await controller.refresh(intent_id) if refresh else controller.status(intent_id)
        )
    except StudioRecoveryUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ControllerProtocolError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    reservation = next(reversed(projection.reservations.values()), None)
    attempt = next(reversed(projection.attempts.values()), None)
    observation = attempt.observations[-1] if attempt is not None and attempt.observations else {}
    status = projection.status
    if reservation is not None and reservation.status == "inert":
        status = "awaiting_authentication"
    elif attempt is not None and status != "operator_action_required":
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
        orphaned_resources=[
            {
                "backend_id": orphan.backend_id,
                "provider_resource_handle": orphan.resource.provider_resource_handle,
                "external_effect_key": orphan.resource.external_effect_key,
                "status": orphan.status,
                "handling_policy": (
                    orphan.handling_policy.model_dump(mode="json")
                    if orphan.handling_policy is not None
                    else None
                ),
            }
            for orphan in projection.orphans.values()
        ],
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
async def inspect_artifacts(intent_id: str, controller: _Controller) -> dict[str, Any]:
    return {
        "intent_id": intent_id,
        "artifact_refs": list(controller.inspect_artifacts(intent_id)),
    }


@router.delete("/instance", response_model=TerminateResponse)
async def terminate_instance(
    controller: _Controller, intent_id: str | None = Query(default=None)
) -> TerminateResponse:
    intent_id = intent_id or controller.latest_intent_id()
    if intent_id is None:
        return TerminateResponse(ok=True)
    try:
        await controller.terminate(intent_id)
    except StudioRecoveryUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (ControllerProtocolError, RuntimeError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return TerminateResponse(ok=True)
