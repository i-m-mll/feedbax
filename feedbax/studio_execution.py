"""Lower Studio workspace stages into provider execution plans."""

from __future__ import annotations

import uuid
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from feedbax.execution import (
    ArtifactPolicy,
    ExecutionBackend,
    ExecutionPlan,
    ExecutionSpec,
    LocalBackendConfig,
    RepoSource,
    default_feedbax_sources,
    prepare_execution_plan,
)
from feedbax.manifest import utc_now
from feedbax.web.models.graph import (
    StudioArtifactRef,
    StudioManifestRef,
    StudioStageSpec,
    StudioValidationIssue,
    StudioValidationState,
    StudioWorkspaceSpec,
)


class StudioExecutionModel(BaseModel):
    """Base model for Studio execution preparation records."""

    model_config = ConfigDict(extra="forbid")


class StudioTrainingExecutionRequest(StudioExecutionModel):
    """Request to prepare an execution plan from a Studio train stage."""

    workspace: StudioWorkspaceSpec
    stage_id: Optional[str] = None
    backend: ExecutionBackend = "local"
    job_id: Optional[str] = None
    local_cwd: Optional[str] = None
    feedbax_ref: Optional[str] = None
    repos: Optional[list[RepoSource]] = None
    primary_repo: Optional[str] = None
    issues: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class StudioTrainingExecutionPreparation(StudioExecutionModel):
    """Prepared provider execution plan plus workspace updates."""

    workspace: StudioWorkspaceSpec
    stage_id: str
    scenario_id: str
    execution_spec: ExecutionSpec
    plan: ExecutionPlan


class StudioExecutionPreparationError(ValueError):
    """Raised when a workspace cannot be lowered to an execution plan."""


def prepare_studio_training_execution(
    request: StudioTrainingExecutionRequest,
) -> StudioTrainingExecutionPreparation:
    """Prepare a provider execution plan from the active Studio train scenario.

    This function is intentionally provider/domain owned. Frontend state is
    submitted as a typed workspace snapshot, then Feedbax validates and lowers
    it into the same provider-neutral ``ExecutionSpec`` used by local, SSH,
    RunPod, and Modal plans.
    """

    workspace = request.workspace.model_copy(deep=True)
    stage = _select_train_stage(workspace, request.stage_id)
    if stage.scenario_id is None:
        raise StudioExecutionPreparationError(
            f"Train stage {stage.id!r} does not reference a scenario"
        )
    scenario = workspace.scenarios.get(stage.scenario_id)
    if scenario is None:
        raise StudioExecutionPreparationError(
            f"Train stage {stage.id!r} references missing scenario {stage.scenario_id!r}"
        )
    if scenario.graph is None:
        raise StudioExecutionPreparationError(
            f"Scenario {scenario.id!r} cannot execute without a graph"
        )
    if scenario.training_spec is None:
        raise StudioExecutionPreparationError(
            f"Scenario {scenario.id!r} cannot execute without a training_spec"
        )
    if scenario.task_spec is None:
        raise StudioExecutionPreparationError(
            f"Scenario {scenario.id!r} cannot execute without a task_spec"
        )

    validation = _validate_training_scenario(
        graph=scenario.graph.model_dump(mode="json", exclude_none=True),
        training_spec=scenario.training_spec,
        task_spec=scenario.task_spec,
    )
    if validation.errors:
        stage.validation = validation
        stage.status = "invalid"
        _replace_stage(workspace, stage)
        raise StudioExecutionPreparationError(
            "Studio train scenario is not valid for provider execution"
        )

    job_id = request.job_id or f"studio-train-{uuid.uuid4().hex[:12]}"
    execution_spec = _build_execution_spec(
        request=request,
        workspace=workspace,
        stage=stage,
        job_id=job_id,
    )
    plan = prepare_execution_plan(execution_spec)
    plan.warnings.extend(
        issue.message for issue in validation.warnings if issue.message not in plan.warnings
    )

    prepared_at = utc_now().isoformat()
    plan_ref = StudioArtifactRef(
        kind="ExecutionPlan",
        id=f"execution-plan:{plan.job_id}",
        role="execution_plan",
        uri=f"{plan.run_directory.rstrip('/')}/execution-plan.json",
        media_type="application/json",
        metadata={
            "backend": plan.backend,
            "stage_id": stage.id,
            "scenario_id": stage.scenario_id,
            "prepared_at": prepared_at,
        },
    )
    stage.execution_spec = execution_spec.model_dump(mode="json", exclude_none=True)
    stage.status = "ready"
    stage.validation = StudioValidationState(
        valid=True,
        checked_at=prepared_at,
        warnings=validation.warnings,
        metadata={
            **validation.metadata,
            "execution_job_id": plan.job_id,
            "execution_backend": plan.backend,
            "execution_run_directory": plan.run_directory,
        },
    )
    stage.artifact_refs = _upsert_artifact_ref(stage.artifact_refs, plan_ref)
    stage.manifest_refs = _upsert_manifest_ref(
        stage.manifest_refs,
        StudioManifestRef(
            kind="ExecutionPlan",
            id=f"execution-plan:{plan.job_id}",
            role="execution_plan",
            uri=plan_ref.uri,
            metadata=plan_ref.metadata,
        ),
    )
    stage.metadata = {
        **stage.metadata,
        "last_execution_plan": {
            "job_id": plan.job_id,
            "backend": plan.backend,
            "prepared_at": prepared_at,
            "run_directory": plan.run_directory,
        },
    }
    workspace.artifact_refs = _upsert_artifact_ref(workspace.artifact_refs, plan_ref)
    _replace_stage(workspace, stage)

    return StudioTrainingExecutionPreparation(
        workspace=workspace,
        stage_id=stage.id,
        scenario_id=stage.scenario_id,
        execution_spec=execution_spec,
        plan=plan,
    )


def _select_train_stage(
    workspace: StudioWorkspaceSpec,
    stage_id: Optional[str],
) -> StudioStageSpec:
    if stage_id is not None:
        stage = next((item for item in workspace.stages if item.id == stage_id), None)
        if stage is None:
            raise StudioExecutionPreparationError(f"Workspace has no stage {stage_id!r}")
        if stage.kind != "train":
            raise StudioExecutionPreparationError(
                f"Stage {stage.id!r} has kind {stage.kind!r}; expected 'train'"
            )
        return stage.model_copy(deep=True)

    active = next(
        (item for item in workspace.stages if item.id == workspace.active_stage_id),
        None,
    )
    if active is not None and active.kind == "train":
        return active.model_copy(deep=True)
    train = next((item for item in workspace.stages if item.kind == "train"), None)
    if train is None:
        raise StudioExecutionPreparationError("Workspace has no train stage")
    return train.model_copy(deep=True)


def _validate_training_scenario(
    *,
    graph: dict[str, Any],
    training_spec: dict[str, Any],
    task_spec: dict[str, Any],
) -> StudioValidationState:
    from feedbax.provider import validate_graph_spec, validate_task_spec, validate_training_spec

    graph_result = validate_graph_spec(graph)
    training_result = validate_training_spec(training_spec, graph_spec=graph)
    task_result = validate_task_spec(task_spec)

    errors = [
        *_provider_issues_to_studio(graph_result.errors, prefix="graph"),
        *_provider_issues_to_studio(training_result.errors, prefix="training_spec"),
        *_provider_issues_to_studio(task_result.errors, prefix="task_spec"),
    ]
    warnings = [
        *_provider_issues_to_studio(graph_result.warnings, prefix="graph", severity="warning"),
        *_provider_issues_to_studio(
            training_result.warnings, prefix="training_spec", severity="warning"
        ),
        *_provider_issues_to_studio(task_result.warnings, prefix="task_spec", severity="warning"),
    ]
    warnings.append(
        StudioValidationIssue(
            type="execution_runner_pending",
            message=(
                "Prepared plan validates the Studio training snapshot; binding it to the "
                "real JAX training runner remains a later execution slice."
            ),
            location={"path": "/execution_spec/command"},
            severity="info",
        )
    )
    return StudioValidationState(
        valid=not errors,
        checked_at=utc_now().isoformat(),
        errors=errors,
        warnings=warnings,
        metadata={"validated_by": "feedbax.studio_execution"},
    )


def _provider_issues_to_studio(
    issues: list[Any],
    *,
    prefix: str,
    severity: str = "error",
) -> list[StudioValidationIssue]:
    converted: list[StudioValidationIssue] = []
    for issue in issues:
        location = dict(issue.location or {})
        if "path" in location:
            location["path"] = f"/{prefix}{location['path']}"
        else:
            location["path"] = f"/{prefix}"
        converted.append(
            StudioValidationIssue(
                type=issue.type,
                message=issue.message,
                location=location,
                severity=severity,
            )
        )
    return converted


def _build_execution_spec(
    *,
    request: StudioTrainingExecutionRequest,
    workspace: StudioWorkspaceSpec,
    stage: StudioStageSpec,
    job_id: str,
) -> ExecutionSpec:
    scenario = workspace.scenarios[stage.scenario_id or ""]
    repos = request.repos
    if repos is None and request.backend != "local":
        repos = default_feedbax_sources(feedbax_ref=request.feedbax_ref or "develop")
    metadata = {
        **request.metadata,
        "studio": {
            "workspace_id": workspace.id,
            "workspace_schema_version": workspace.schema_version,
            "stage_id": stage.id,
            "stage_kind": stage.kind,
            "scenario_id": scenario.id,
            "scenario_schema_version": scenario.schema_version,
            "graph_spec": scenario.graph.model_dump(mode="json", exclude_none=True)
            if scenario.graph is not None
            else None,
            "training_spec": scenario.training_spec,
            "task_spec": scenario.task_spec,
            "objective_spec": scenario.objective_spec,
            "temporal_spec": scenario.temporal_spec,
        },
        "command_contract": {
            "expected_files": [
                "execution-spec.json",
                "workspace-snapshot.json",
                "graph-spec.json",
                "training-spec.json",
                "task-spec.json",
            ],
            "current_command_role": "validate_stage_snapshot",
            "future_command_role": "launch_training_runner",
        },
    }
    return ExecutionSpec(
        kind="training",
        job_id=job_id,
        backend=request.backend,
        command="feedbax-provider validate training training-spec.json --graph graph-spec.json && feedbax-provider validate task task-spec.json",
        repos=repos or [],
        primary_repo=request.primary_repo,
        local=LocalBackendConfig(cwd=request.local_cwd),
        artifact_policy=ArtifactPolicy(
            tracked_paths=[
                "execution-spec.json",
                "workspace-snapshot.json",
                "graph-spec.json",
                "training-spec.json",
                "task-spec.json",
            ],
            bulk_paths=["artifacts"],
            metadata={"studio_stage_id": stage.id, "studio_scenario_id": scenario.id},
        ),
        issues=request.issues,
        env={
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "FEEDBAX_STUDIO_WORKSPACE_ID": workspace.id,
            "FEEDBAX_STUDIO_STAGE_ID": stage.id,
            "FEEDBAX_STUDIO_SCENARIO_ID": scenario.id,
        },
        metadata=metadata,
    )


def _replace_stage(workspace: StudioWorkspaceSpec, updated: StudioStageSpec) -> None:
    workspace.stages = [updated if stage.id == updated.id else stage for stage in workspace.stages]


def _upsert_artifact_ref(
    refs: list[StudioArtifactRef],
    ref: StudioArtifactRef,
) -> list[StudioArtifactRef]:
    return [item for item in refs if not (item.kind == ref.kind and item.id == ref.id)] + [ref]


def _upsert_manifest_ref(
    refs: list[StudioManifestRef],
    ref: StudioManifestRef,
) -> list[StudioManifestRef]:
    return [item for item in refs if not (item.kind == ref.kind and item.id == ref.id)] + [ref]
