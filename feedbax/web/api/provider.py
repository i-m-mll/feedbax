"""Provider contract endpoints for Feedbax Studio and orchestrators."""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from feedbax.integrations.provider import (
    ProviderHealth,
    ProviderManifest,
    ProviderValidationResult,
    RegistrySnapshot,
    health,
    provider_manifest,
    registry_snapshot,
    validate_spec,
)
from feedbax.studio.schema import (
    StudioSchemaEnumerationRequest,
    StudioSchemaRegistry,
    enumerate_studio_schema_registry,
)


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
async def get_registry_snapshot(kind: str, request: Request) -> RegistrySnapshot:
    try:
        return registry_snapshot(
            kind,
            component_registry=request.app.state.bootstrap_state.bundle.components,
            experiment_registry=request.app.state.bootstrap_state.bundle.experiment_packages,
            analysis_registry=request.app.state.bootstrap_state.bundle.analysis_recipes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/validate/{kind}", response_model=ProviderValidationResult)
async def validate_provider_spec(
    kind: str,
    payload: ValidateSpecRequest,
    request: Request,
) -> ProviderValidationResult:
    try:
        return validate_spec(
            kind,
            payload.spec,
            graph_spec=payload.graph_spec,
            component_registry=request.app.state.bootstrap_state.bundle.components,
            training_method_registry=request.app.state.bootstrap_state.bundle.training_programs,
            analysis_registry=request.app.state.bootstrap_state.bundle.analysis_recipes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post(
    "/studio/schemas",
    response_model=StudioSchemaRegistry,
)
async def enumerate_studio_schemas(
    payload: StudioSchemaEnumerationRequest,
    request: Request,
) -> StudioSchemaRegistry:
    return enumerate_studio_schema_registry(
        payload.workspace,
        payload.scenario_id,
        graph=payload.graph,
        runtime_introspection=payload.runtime_introspection,
        component_registry=request.app.state.bootstrap_state.bundle.components,
    )
