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
from feedbax.execution.models import ExecutionPlan, ExecutionSpec
from feedbax.execution.planning import prepare_execution_plan
from feedbax.studio.schema import (
    StudioSchemaEnumerationRequest,
    StudioSchemaRegistry,
    enumerate_studio_schema_registry,
)
from feedbax.studio.execution import (
    StudioExecutionPreparationError,
    StudioEvaluationLocalRunRequest,
    StudioEvaluationLocalRunResult,
    StudioEvaluationMatrixPreview,
    StudioEvaluationMatrixRequest,
    StudioEvaluationStagingResult,
    StudioPipelineMaterializationRequest,
    StudioPipelineMaterializationResult,
    StudioTrainingLocalRunRequest,
    StudioTrainingLocalRunResult,
    StudioTrainingExecutionPreparation,
    StudioTrainingExecutionRequest,
    materialize_studio_pipeline,
    prepare_studio_training_execution,
    preview_studio_evaluation_matrix,
    run_studio_evaluation_local_execution,
    run_studio_training_local_execution,
    stage_studio_evaluation_matrix,
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
            training_method_registry=request.app.state.bootstrap_state.bundle.training_methods,
            analysis_registry=request.app.state.bootstrap_state.bundle.analysis_recipes,
        )
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
    request: Request,
) -> StudioTrainingExecutionPreparation:
    try:
        return prepare_studio_training_execution(
            payload,
            registry_bundle=request.app.state.bootstrap_state.bundle,
        )
    except StudioExecutionPreparationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post(
    "/studio/training/run-local",
    response_model=StudioTrainingLocalRunResult,
)
async def run_studio_training_local(
    payload: StudioTrainingLocalRunRequest,
    request: Request,
) -> StudioTrainingLocalRunResult:
    try:
        return run_studio_training_local_execution(
            payload,
            registry_bundle=request.app.state.bootstrap_state.bundle,
        )
    except StudioExecutionPreparationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post(
    "/studio/evaluation/preview",
    response_model=StudioEvaluationMatrixPreview,
)
async def preview_studio_evaluation(
    payload: StudioEvaluationMatrixRequest,
) -> StudioEvaluationMatrixPreview:
    try:
        return preview_studio_evaluation_matrix(payload)
    except StudioExecutionPreparationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post(
    "/studio/evaluation/stage",
    response_model=StudioEvaluationStagingResult,
)
async def stage_studio_evaluation(
    payload: StudioEvaluationMatrixRequest,
) -> StudioEvaluationStagingResult:
    try:
        return stage_studio_evaluation_matrix(payload)
    except StudioExecutionPreparationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post(
    "/studio/evaluation/run-local",
    response_model=StudioEvaluationLocalRunResult,
)
async def run_studio_evaluation_local(
    payload: StudioEvaluationLocalRunRequest,
    request: Request,
) -> StudioEvaluationLocalRunResult:
    try:
        return run_studio_evaluation_local_execution(
            payload,
            registry_bundle=request.app.state.bootstrap_state.bundle,
        )
    except StudioExecutionPreparationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post(
    "/studio/pipeline/materialize",
    response_model=StudioPipelineMaterializationResult,
)
async def materialize_studio_pipeline_stages(
    payload: StudioPipelineMaterializationRequest,
    request: Request,
) -> StudioPipelineMaterializationResult:
    try:
        return materialize_studio_pipeline(
            payload,
            registry_bundle=request.app.state.bootstrap_state.bundle,
        )
    except StudioExecutionPreparationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


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
