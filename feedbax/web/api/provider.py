"""Provider contract endpoints for Feedbax Studio and orchestrators."""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException
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
from feedbax.execution import ExecutionPlan, ExecutionSpec, prepare_execution_plan
from feedbax.studio.schema import (
    StudioSchemaEnumerationRequest,
    StudioSchemaRegistry,
    enumerate_studio_schema_registry,
)
from feedbax.studio.execution import (
    StudioExecutionPreparationError,
    StudioPipelineMaterializationRequest,
    StudioPipelineMaterializationResult,
    StudioTrainingLocalRunRequest,
    StudioTrainingLocalRunResult,
    StudioTrainingExecutionPreparation,
    StudioTrainingExecutionRequest,
    materialize_studio_pipeline,
    prepare_studio_training_execution,
    run_studio_training_local_execution,
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


@router.post(
    "/studio/training/plan",
    response_model=StudioTrainingExecutionPreparation,
)
async def prepare_studio_training_plan(
    payload: StudioTrainingExecutionRequest,
) -> StudioTrainingExecutionPreparation:
    try:
        return prepare_studio_training_execution(payload)
    except StudioExecutionPreparationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post(
    "/studio/training/run-local",
    response_model=StudioTrainingLocalRunResult,
)
async def run_studio_training_local(
    payload: StudioTrainingLocalRunRequest,
) -> StudioTrainingLocalRunResult:
    try:
        return run_studio_training_local_execution(payload)
    except StudioExecutionPreparationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post(
    "/studio/pipeline/materialize",
    response_model=StudioPipelineMaterializationResult,
)
async def materialize_studio_pipeline_stages(
    payload: StudioPipelineMaterializationRequest,
) -> StudioPipelineMaterializationResult:
    try:
        return materialize_studio_pipeline(payload)
    except StudioExecutionPreparationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post(
    "/studio/schemas",
    response_model=StudioSchemaRegistry,
)
async def enumerate_studio_schemas(
    payload: StudioSchemaEnumerationRequest,
) -> StudioSchemaRegistry:
    return enumerate_studio_schema_registry(
        payload.workspace,
        payload.scenario_id,
        runtime_introspection=payload.runtime_introspection,
    )
