"""Provider contract endpoints for Feedbax Studio and orchestrators."""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from feedbax.provider import (
    ProviderHealth,
    ProviderManifest,
    ProviderValidationResult,
    RegistrySnapshot,
    health,
    provider_manifest,
    registry_snapshot,
    validate_spec,
)
from feedbax.execution import ExecutionPlan, ExecutionSpec, prepare_execution_plan


router = APIRouter()


class ValidateSpecRequest(BaseModel):
    spec: dict[str, Any] = Field(default_factory=dict)
    graph_spec: Optional[dict[str, Any]] = None


@router.get("/health", response_model=ProviderHealth)
async def get_provider_health() -> ProviderHealth:
    return health()


@router.get("/manifest", response_model=ProviderManifest)
async def get_provider_manifest() -> ProviderManifest:
    return provider_manifest()


@router.get("/registries/{kind}", response_model=RegistrySnapshot)
async def get_registry_snapshot(kind: str) -> RegistrySnapshot:
    try:
        return registry_snapshot(kind)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/validate/{kind}", response_model=ProviderValidationResult)
async def validate_provider_spec(
    kind: str,
    payload: ValidateSpecRequest,
) -> ProviderValidationResult:
    try:
        return validate_spec(kind, payload.spec, graph_spec=payload.graph_spec)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/execution/plan", response_model=ExecutionPlan)
async def prepare_provider_execution_plan(payload: ExecutionSpec) -> ExecutionPlan:
    return prepare_execution_plan(payload)
