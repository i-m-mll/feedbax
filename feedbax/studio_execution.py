"""Lower Studio workspace stages into provider execution plans."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from feedbax.execution import (
    ArtifactPolicy,
    ExecutionBackend,
    LocalExecutionResult,
    ExecutionPlan,
    ExecutionSpec,
    LocalBackendConfig,
    RepoSource,
    default_feedbax_sources,
    prepare_execution_plan,
    run_local_execution,
)
from feedbax.manifest import (
    AnalysisRunManifest,
    AnalysisRunSpec,
    ArtifactRef,
    EntrypointRef,
    EvaluationRunManifest,
    EvaluationRunSpec,
    ParentRef,
    Provenance,
    ReportManifest,
    ReportSpec,
    default_manifest_root,
    spec_payload,
    store_json_artifact,
    utc_now,
    write_manifest,
)
from feedbax.studio_schema import SchemaValidationIssue, validate_task_binding_schema
from feedbax.web.models.graph import (
    GraphSpec,
    StudioArtifactRef,
    StudioCollectionRef,
    StudioManifestRef,
    StudioStageSpec,
    StudioTaskBindingSpec,
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


class StudioTrainingLocalRunRequest(StudioExecutionModel):
    """Request to run the active Studio train-stage scenario locally."""

    workspace: StudioWorkspaceSpec
    stage_id: Optional[str] = None
    job_id: Optional[str] = None
    local_cwd: Optional[str] = None
    root: Optional[str] = None
    timeout: Optional[float] = None
    issues: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class StudioTrainingLocalRunResult(StudioExecutionModel):
    """Result from a local Studio train-stage provider execution."""

    workspace: StudioWorkspaceSpec
    stage_id: str
    scenario_id: str
    execution_spec: ExecutionSpec
    result: LocalExecutionResult
    snapshot_dir: str


StudioPipelineMaterializationStage = Literal["eval", "analysis", "report"]


class StudioPipelineMaterializationRequest(StudioExecutionModel):
    """Request to materialize downstream Studio pipeline stages."""

    workspace: StudioWorkspaceSpec
    stages: list[StudioPipelineMaterializationStage] = Field(
        default_factory=lambda: ["eval", "analysis", "report"]
    )
    job_id: Optional[str] = None
    root: Optional[str] = None
    issues: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class StudioPipelineMaterializationResult(StudioExecutionModel):
    """Result from materializing eval/analysis/report Studio stages."""

    workspace: StudioWorkspaceSpec
    stage_ids: list[str]
    manifest_paths: dict[str, str]
    artifact_refs: list[StudioArtifactRef] = Field(default_factory=list)


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
        task_binding_spec=scenario.task_binding_spec.model_dump(mode="json", exclude_none=True)
        if scenario.task_binding_spec is not None
        else None,
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


def run_studio_training_local_execution(
    request: StudioTrainingLocalRunRequest,
) -> StudioTrainingLocalRunResult:
    """Run a Studio train-stage scenario through the local provider boundary."""

    job_id = request.job_id or f"studio-train-{uuid.uuid4().hex[:12]}"
    root_path = Path(request.root).expanduser() if request.root else default_manifest_root()
    snapshot_dir = (
        Path(request.local_cwd).expanduser()
        if request.local_cwd
        else root_path / "executions" / job_id / "inputs"
    )
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    preparation = prepare_studio_training_execution(
        StudioTrainingExecutionRequest(
            workspace=request.workspace,
            stage_id=request.stage_id,
            backend="local",
            job_id=job_id,
            local_cwd=str(snapshot_dir),
            issues=request.issues,
            metadata=request.metadata,
        )
    )
    _materialize_local_execution_snapshot(preparation, snapshot_dir)

    result = run_local_execution(
        preparation.execution_spec,
        root=root_path,
        timeout=request.timeout,
    )
    workspace = preparation.workspace.model_copy(deep=True)
    stage = _select_train_stage(workspace, preparation.stage_id)
    completed_at = utc_now().isoformat()
    manifest_ref = StudioManifestRef(
        kind=result.manifest_payload.get("kind", "TrainingRunManifest"),
        id=str(result.manifest_payload.get("id", f"training-run:{result.job_id}")),
        role="training_run",
        uri=result.manifest_path,
        metadata={
            "job_id": result.job_id,
            "status": result.status,
            "stage_id": stage.id,
            "scenario_id": preparation.scenario_id,
            "completed_at": completed_at,
        },
    )
    stage.status = "completed" if result.status == "completed" else "failed"
    stage.validation = StudioValidationState(
        valid=result.status == "completed",
        checked_at=completed_at,
        errors=(
            []
            if result.status == "completed"
            else [
                StudioValidationIssue(
                    type="local_execution_failed",
                    message=f"Local execution returned code {result.return_code}",
                    location={"path": "/execution_spec/command"},
                    severity="error",
                )
            ]
        ),
        warnings=stage.validation.warnings,
        metadata={
            **stage.validation.metadata,
            "execution_job_id": result.job_id,
            "execution_status": result.status,
            "execution_return_code": result.return_code,
            "snapshot_dir": str(snapshot_dir),
            "manifest_path": result.manifest_path,
        },
    )
    stage.manifest_refs = _upsert_manifest_ref(stage.manifest_refs, manifest_ref)
    stage.artifact_refs = _upsert_many_artifact_refs(
        stage.artifact_refs,
        _local_result_artifact_refs(result, snapshot_dir, stage.id, preparation.scenario_id),
    )
    stage.output_collections = _upsert_training_manifest_in_outputs(
        stage.output_collections,
        manifest_ref,
        stage.id,
    )
    stage.metadata = {
        **stage.metadata,
        "last_execution_result": {
            "job_id": result.job_id,
            "status": result.status,
            "return_code": result.return_code,
            "completed_at": completed_at,
            "manifest_path": result.manifest_path,
            "snapshot_dir": str(snapshot_dir),
        },
    }
    workspace.manifest_refs = _upsert_manifest_ref(workspace.manifest_refs, manifest_ref)
    workspace.artifact_refs = _upsert_many_artifact_refs(
        workspace.artifact_refs,
        _local_result_artifact_refs(result, snapshot_dir, stage.id, preparation.scenario_id),
    )
    workspace.collections = _upsert_many_collection_refs(
        workspace.collections,
        stage.output_collections,
    )
    _replace_stage(workspace, stage)

    return StudioTrainingLocalRunResult(
        workspace=workspace,
        stage_id=stage.id,
        scenario_id=preparation.scenario_id,
        execution_spec=preparation.execution_spec,
        result=result,
        snapshot_dir=str(snapshot_dir),
    )


def materialize_studio_pipeline(
    request: StudioPipelineMaterializationRequest,
) -> StudioPipelineMaterializationResult:
    """Materialize the first Studio train -> eval -> analysis -> report path.

    This is the Phase 4 product-path bridge. It consumes the durable collections
    produced by upstream stages and writes provider manifests for downstream
    stages without introducing a hidden frontend-only interpretation path.
    """

    workspace = request.workspace.model_copy(deep=True)
    root_path = Path(request.root).expanduser() if request.root else default_manifest_root()
    base_job_id = request.job_id or f"studio-pipeline-{uuid.uuid4().hex[:12]}"
    executed_stage_ids: list[str] = []
    manifest_paths: dict[str, str] = {}
    artifact_refs: list[StudioArtifactRef] = []

    for stage_kind in request.stages:
        if stage_kind == "eval":
            manifest_path, stage_artifacts = _materialize_eval_stage(
                workspace,
                root_path=root_path,
                job_id=f"{base_job_id}-eval",
                issues=request.issues,
                request_metadata=request.metadata,
            )
        elif stage_kind == "analysis":
            manifest_path, stage_artifacts = _materialize_analysis_stage(
                workspace,
                root_path=root_path,
                job_id=f"{base_job_id}-analysis",
                issues=request.issues,
                request_metadata=request.metadata,
            )
        elif stage_kind == "report":
            manifest_path, stage_artifacts = _materialize_report_stage(
                workspace,
                root_path=root_path,
                job_id=f"{base_job_id}-report",
                issues=request.issues,
                request_metadata=request.metadata,
            )
        else:  # pragma: no cover - Literal keeps this unreachable.
            raise StudioExecutionPreparationError(f"Unsupported stage kind {stage_kind!r}")

        stage = _select_stage_by_kind(workspace, stage_kind)
        executed_stage_ids.append(stage.id)
        manifest_paths[stage.id] = str(manifest_path)
        artifact_refs.extend(stage_artifacts)

    return StudioPipelineMaterializationResult(
        workspace=workspace,
        stage_ids=executed_stage_ids,
        manifest_paths=manifest_paths,
        artifact_refs=artifact_refs,
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
    task_binding_spec: dict[str, Any] | None = None,
) -> StudioValidationState:
    from feedbax.provider import validate_graph_spec, validate_task_spec, validate_training_spec

    graph_result = validate_graph_spec(graph)
    training_result = validate_training_spec(training_spec, graph_spec=graph)
    task_result = validate_task_spec(task_spec)
    task_binding_errors = _validate_task_binding_spec(graph, task_binding_spec)

    errors = [
        *_provider_issues_to_studio(graph_result.errors, prefix="graph"),
        *_provider_issues_to_studio(training_result.errors, prefix="training_spec"),
        *_provider_issues_to_studio(task_result.errors, prefix="task_spec"),
        *task_binding_errors,
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


def _validate_task_binding_spec(
    graph: dict[str, Any],
    task_binding_spec: dict[str, Any] | None,
) -> list[StudioValidationIssue]:
    if task_binding_spec is None:
        return []
    try:
        validated_spec = StudioTaskBindingSpec.model_validate(task_binding_spec)
    except ValidationError as exc:
        issues: list[StudioValidationIssue] = []
        for error in exc.errors():
            loc = error.get("loc", ())
            suffix = "".join(f"/{part}" for part in loc) if loc else ""
            message = str(error.get("msg", "Invalid task binding spec"))
            if message.startswith("Value error, "):
                message = message.removeprefix("Value error, ")
            issues.append(
                StudioValidationIssue(
                    type="invalid_task_binding_spec",
                    message=message,
                    location={"path": f"/task_binding_spec{suffix}"},
                )
            )
        return issues
    graph_spec = GraphSpec.model_validate(graph)
    return _schema_issues_to_studio(
        validate_task_binding_schema(validated_spec, graph_spec, "/task_binding_spec")
    )


def _schema_issues_to_studio(
    issues: list[SchemaValidationIssue],
) -> list[StudioValidationIssue]:
    return [
        StudioValidationIssue(
            type=issue.type,
            message=issue.message,
            location=issue.location,
            severity=issue.severity,
        )
        for issue in issues
    ]


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
            "task_binding_spec": scenario.task_binding_spec.model_dump(mode="json", exclude_none=True)
            if scenario.task_binding_spec is not None
            else None,
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
                "task-binding-spec.json",
                "artifacts/training-summary.json",
            ],
            "current_command_role": "materialize_mvp_training_result",
            "future_command_role": "launch_training_runner",
        },
    }
    return ExecutionSpec(
        kind="training",
        job_id=job_id,
        backend=request.backend,
        command=(
            "python -m feedbax.bin.studio_pipeline materialize-training "
            "--graph graph-spec.json "
            "--training training-spec.json "
            "--task task-spec.json "
            "--task-binding task-binding-spec.json "
            "--output artifacts/training-summary.json"
        ),
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
                "task-binding-spec.json",
                "artifacts/training-summary.json",
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


def _materialize_local_execution_snapshot(
    preparation: StudioTrainingExecutionPreparation,
    snapshot_dir: Path,
) -> None:
    scenario = preparation.workspace.scenarios[preparation.scenario_id]
    files = {
        "execution-spec.json": preparation.execution_spec.model_dump(
            mode="json", exclude_none=True
        ),
        "workspace-snapshot.json": preparation.workspace.model_dump(
            mode="json", exclude_none=True
        ),
        "graph-spec.json": scenario.graph.model_dump(mode="json", exclude_none=True)
        if scenario.graph is not None
        else {},
        "training-spec.json": scenario.training_spec or {},
        "task-spec.json": scenario.task_spec or {},
        "task-binding-spec.json": scenario.task_binding_spec.model_dump(
            mode="json", exclude_none=True
        )
        if scenario.task_binding_spec is not None
        else {},
    }
    for filename, payload in files.items():
        _write_json(snapshot_dir / filename, payload)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _local_result_artifact_refs(
    result: LocalExecutionResult,
    snapshot_dir: Path,
    stage_id: str,
    scenario_id: str,
) -> list[StudioArtifactRef]:
    metadata = {
        "job_id": result.job_id,
        "stage_id": stage_id,
        "scenario_id": scenario_id,
        "status": result.status,
    }
    run_dir = Path(result.stdout_path).parent
    refs = [
        StudioArtifactRef(
            kind="ExecutionPlan",
            id=f"execution-plan:{result.job_id}",
            role="execution_plan",
            uri=str(run_dir / "execution-plan.json"),
            media_type="application/json",
            metadata=metadata,
        ),
        StudioArtifactRef(
            kind="ExecutionLog",
            id=f"execution-log:{result.job_id}:stdout",
            role="execution_stdout",
            uri=result.stdout_path,
            media_type="text/plain",
            metadata=metadata,
        ),
        StudioArtifactRef(
            kind="ExecutionLog",
            id=f"execution-log:{result.job_id}:stderr",
            role="execution_stderr",
            uri=result.stderr_path,
            media_type="text/plain",
            metadata=metadata,
        ),
        StudioArtifactRef(
            kind="StudioExecutionSnapshot",
            id=f"studio-execution-snapshot:{result.job_id}",
            role="execution_input_snapshot",
            uri=str(snapshot_dir),
            media_type="application/x-directory",
            metadata={
                **metadata,
                "files": [
                    "execution-spec.json",
                    "workspace-snapshot.json",
                    "graph-spec.json",
                    "training-spec.json",
                    "task-spec.json",
                    "task-binding-spec.json",
                ],
            },
        ),
    ]
    training_summary_path = snapshot_dir / "artifacts" / "training-summary.json"
    if training_summary_path.exists():
        refs.append(
            StudioArtifactRef(
                kind="StudioTrainingResult",
                id=f"studio-training-result:{result.job_id}",
                role="training_result",
                uri=str(training_summary_path),
                media_type="application/json",
                metadata=metadata,
            )
        )
    return refs


def _materialize_eval_stage(
    workspace: StudioWorkspaceSpec,
    *,
    root_path: Path,
    job_id: str,
    issues: list[str],
    request_metadata: dict[str, Any],
) -> tuple[Path, list[StudioArtifactRef]]:
    train_stage = _select_stage_by_kind(workspace, "train")
    eval_stage = _select_stage_by_kind(workspace, "eval")
    training_collection = _require_output_collection(train_stage, "training_runs")
    _require_collection_items(training_collection, "training runs", eval_stage.id)
    eval_stage.input_collections = _upsert_collection_ref(
        eval_stage.input_collections,
        training_collection,
    )

    train_scenario = workspace.scenarios.get(train_stage.scenario_id or "")
    eval_scenario = workspace.scenarios.get(eval_stage.scenario_id or "")
    if eval_scenario is not None:
        eval_scenario.parent_scenario_id = (
            eval_scenario.parent_scenario_id or train_stage.scenario_id
        )
        eval_scenario.metadata = {
            **eval_scenario.metadata,
            "inheritance": eval_scenario.metadata.get(
                "inheritance",
                "training_default",
            ),
            "inherits_from_stage_id": train_stage.id,
        }
        if eval_scenario.task_spec is None and train_scenario is not None:
            eval_scenario.task_spec = train_scenario.task_spec
        if eval_scenario.task_binding_spec is None and train_scenario is not None:
            eval_scenario.task_binding_spec = train_scenario.task_binding_spec
        workspace.scenarios[eval_scenario.id] = eval_scenario

    input_refs = _collection_manifest_parents(training_collection)
    spec = EvaluationRunSpec(
        evaluation_type="studio_default_eval",
        training_run_ids=[ref.id for ref in input_refs if ref.kind == "TrainingRunManifest"],
        inputs=input_refs,
        params={
            "stage_id": eval_stage.id,
            "scenario_id": eval_stage.scenario_id,
            "selection_spec": eval_stage.selection_spec,
            "input_collection_id": training_collection.id,
            "inherited_from_scenario_id": eval_scenario.parent_scenario_id
            if eval_scenario is not None
            else train_stage.scenario_id,
        },
    )
    summary = {
        "kind": "StudioEvaluationSummary",
        "job_id": job_id,
        "stage_id": eval_stage.id,
        "input_training_runs": [ref.id for ref in input_refs],
        "status": "completed",
    }
    artifact = store_json_artifact(
        summary,
        root=root_path,
        role="evaluation_result",
        logical_name=f"{job_id}-evaluation-summary.json",
        metadata={"stage_id": eval_stage.id, "job_id": job_id},
    )
    manifest = EvaluationRunManifest(
        id=f"feedbax-evaluation-run:{job_id}",
        status="completed",
        evaluation_spec=spec_payload(
            "EvaluationRunSpec",
            spec.model_dump(mode="json", exclude_none=True),
        ),
        input_training_runs=input_refs,
        summary_metrics={"input_training_runs": len(input_refs)},
        provenance=_stage_provenance(
            stage_kind="eval",
            issues=issues,
            parents=input_refs,
            request_metadata=request_metadata,
            job_id=job_id,
        ),
        artifacts=[artifact],
        metadata={"studio": _stage_manifest_metadata(workspace, eval_stage, job_id)},
    )
    path = write_manifest(manifest, root=root_path)
    manifest_ref = _studio_manifest_ref(manifest.kind, manifest.id, "evaluation_run", path, job_id)
    artifact_refs = [_studio_artifact_ref(artifact, kind="EvaluationResult")]
    _complete_stage_with_manifest(
        workspace,
        eval_stage,
        manifest_ref=manifest_ref,
        artifact_refs=artifact_refs,
        output_collection_kind="evaluation_runs",
        output_collection_id="collection:evaluation-runs",
        output_collection_label="Evaluation runs",
        completed_metadata={
            "input_collection_id": training_collection.id,
            "input_manifest_ids": [ref.id for ref in input_refs],
        },
    )
    return path, artifact_refs


def _materialize_analysis_stage(
    workspace: StudioWorkspaceSpec,
    *,
    root_path: Path,
    job_id: str,
    issues: list[str],
    request_metadata: dict[str, Any],
) -> tuple[Path, list[StudioArtifactRef]]:
    eval_stage = _select_stage_by_kind(workspace, "eval")
    analysis_stage = _select_stage_by_kind(workspace, "analysis")
    evaluation_collection = _require_output_collection(eval_stage, "evaluation_runs")
    _require_collection_items(evaluation_collection, "evaluation runs", analysis_stage.id)
    analysis_stage.input_collections = _upsert_collection_ref(
        analysis_stage.input_collections,
        evaluation_collection,
    )
    input_refs = _collection_manifest_parents(evaluation_collection)
    scenario = workspace.scenarios.get(analysis_stage.scenario_id or "")
    analysis_spec_payload = scenario.analysis_spec if scenario is not None else None
    spec = AnalysisRunSpec(
        analysis_type=str(
            (analysis_spec_payload or {}).get("analysis_type", "feedbax.analysis.activity")
        ),
        inputs=input_refs,
        # Contract-level forwarding; legacy analysis modules still consume eval refs.
        input_requirements=list((analysis_spec_payload or {}).get("input_requirements", [])),
        params={
            "stage_id": analysis_stage.id,
            "scenario_id": analysis_stage.scenario_id,
            "selection_spec": analysis_stage.selection_spec,
            "input_collection_id": evaluation_collection.id,
            "analysis_spec": analysis_spec_payload or {},
        },
    )
    summary = {
        "kind": "StudioAnalysisSummary",
        "job_id": job_id,
        "stage_id": analysis_stage.id,
        "input_evaluation_runs": [ref.id for ref in input_refs],
        "analysis_type": spec.analysis_type,
        "status": "completed",
    }
    artifact = store_json_artifact(
        summary,
        root=root_path,
        role="analysis_table",
        logical_name=f"{job_id}-analysis-summary.json",
        metadata={"stage_id": analysis_stage.id, "job_id": job_id},
    )
    manifest = AnalysisRunManifest(
        id=f"feedbax-analysis-run:{job_id}",
        status="completed",
        analysis_spec=spec_payload(
            "AnalysisRunSpec",
            spec.model_dump(mode="json", exclude_none=True),
        ),
        inputs=input_refs,
        summary_metrics={"input_evaluation_runs": len(input_refs)},
        provenance=_stage_provenance(
            stage_kind="analysis",
            issues=issues,
            parents=input_refs,
            request_metadata=request_metadata,
            job_id=job_id,
        ),
        artifacts=[artifact],
        metadata={"studio": _stage_manifest_metadata(workspace, analysis_stage, job_id)},
    )
    path = write_manifest(manifest, root=root_path)
    manifest_ref = _studio_manifest_ref(manifest.kind, manifest.id, "analysis_run", path, job_id)
    artifact_refs = [_studio_artifact_ref(artifact, kind="AnalysisTable")]
    _complete_stage_with_manifest(
        workspace,
        analysis_stage,
        manifest_ref=manifest_ref,
        artifact_refs=artifact_refs,
        output_collection_kind="analysis_products",
        output_collection_id="collection:analysis-products",
        output_collection_label="Analysis products",
        completed_metadata={
            "input_collection_id": evaluation_collection.id,
            "input_manifest_ids": [ref.id for ref in input_refs],
        },
    )
    return path, artifact_refs


def _materialize_report_stage(
    workspace: StudioWorkspaceSpec,
    *,
    root_path: Path,
    job_id: str,
    issues: list[str],
    request_metadata: dict[str, Any],
) -> tuple[Path, list[StudioArtifactRef]]:
    analysis_stage = _select_stage_by_kind(workspace, "analysis")
    report_stage = _select_stage_by_kind(workspace, "report")
    analysis_collection = _require_output_collection(analysis_stage, "analysis_products")
    _require_collection_items(analysis_collection, "analysis products", report_stage.id)
    report_stage.input_collections = _upsert_collection_ref(
        report_stage.input_collections,
        analysis_collection,
    )
    input_refs = _collection_manifest_parents(analysis_collection)
    scenario = workspace.scenarios.get(report_stage.scenario_id or "")
    report_spec_payload = scenario.report_spec if scenario is not None else None
    spec = ReportSpec(
        report_type=str((report_spec_payload or {}).get("report_type", "studio_report_stub")),
        inputs=input_refs,
        params={
            "stage_id": report_stage.id,
            "scenario_id": report_stage.scenario_id,
            "selection_spec": report_stage.selection_spec,
            "input_collection_id": analysis_collection.id,
            "report_spec": report_spec_payload or {},
        },
        narrative="MVP report stub assembled from selected Studio analysis products.",
    )
    report_body = {
        "kind": "StudioReportProduct",
        "job_id": job_id,
        "stage_id": report_stage.id,
        "input_analysis_products": [ref.id for ref in input_refs],
        "title": workspace.label,
        "status": "completed",
    }
    artifact = store_json_artifact(
        report_body,
        root=root_path,
        role="report",
        logical_name=f"{job_id}-report.json",
        metadata={"stage_id": report_stage.id, "job_id": job_id},
    )
    manifest = ReportManifest(
        id=f"feedbax-report:{job_id}",
        status="completed",
        report_spec=spec_payload(
            "ReportSpec",
            spec.model_dump(mode="json", exclude_none=True),
        ),
        inputs=input_refs,
        provenance=_stage_provenance(
            stage_kind="report",
            issues=issues,
            parents=input_refs,
            request_metadata=request_metadata,
            job_id=job_id,
        ),
        artifacts=[artifact],
        metadata={"studio": _stage_manifest_metadata(workspace, report_stage, job_id)},
    )
    path = write_manifest(manifest, root=root_path)
    manifest_ref = _studio_manifest_ref(manifest.kind, manifest.id, "report", path, job_id)
    artifact_refs = [_studio_artifact_ref(artifact, kind="ReportArtifact")]
    _complete_stage_with_manifest(
        workspace,
        report_stage,
        manifest_ref=manifest_ref,
        artifact_refs=artifact_refs,
        output_collection_kind="reports",
        output_collection_id="collection:reports",
        output_collection_label="Reports",
        completed_metadata={
            "input_collection_id": analysis_collection.id,
            "input_manifest_ids": [ref.id for ref in input_refs],
        },
    )
    return path, artifact_refs


def _select_stage_by_kind(
    workspace: StudioWorkspaceSpec,
    kind: str,
) -> StudioStageSpec:
    stage = next((item for item in workspace.stages if item.kind == kind), None)
    if stage is None:
        raise StudioExecutionPreparationError(f"Workspace has no {kind!r} stage")
    return stage.model_copy(deep=True)


def _require_output_collection(
    stage: StudioStageSpec,
    kind: str,
) -> StudioCollectionRef:
    collection = next(
        (item for item in stage.output_collections if item.kind == kind),
        None,
    )
    if collection is None:
        raise StudioExecutionPreparationError(
            f"Stage {stage.id!r} has no {kind!r} output collection"
        )
    return collection.model_copy(deep=True)


def _require_collection_items(
    collection: StudioCollectionRef,
    label: str,
    consumer_stage_id: str,
) -> None:
    if not collection.item_refs:
        raise StudioExecutionPreparationError(
            f"Cannot materialize stage {consumer_stage_id!r}; no {label} are available"
        )


def _collection_manifest_parents(collection: StudioCollectionRef) -> list[ParentRef]:
    return [
        ParentRef(
            kind=ref.kind,
            id=ref.id,
            role=ref.role,
            uri=ref.uri,
            metadata={
                **ref.metadata,
                "provider": ref.provider,
                "collection_id": collection.id,
                "collection_kind": collection.kind,
            },
        )
        for ref in collection.item_refs
    ]


def _stage_provenance(
    *,
    stage_kind: str,
    issues: list[str],
    parents: list[ParentRef],
    request_metadata: dict[str, Any],
    job_id: str,
) -> Provenance:
    return Provenance(
        entrypoint=EntrypointRef(
            kind="feedbax-studio-pipeline",
            name=f"materialize_{stage_kind}_stage",
            metadata={"job_id": job_id},
        ),
        issues=list(issues),
        parents=parents,
        metadata=request_metadata,
    )


def _stage_manifest_metadata(
    workspace: StudioWorkspaceSpec,
    stage: StudioStageSpec,
    job_id: str,
) -> dict[str, Any]:
    return {
        "workspace_id": workspace.id,
        "workspace_schema_version": workspace.schema_version,
        "stage_id": stage.id,
        "stage_kind": stage.kind,
        "scenario_id": stage.scenario_id,
        "job_id": job_id,
    }


def _studio_manifest_ref(
    kind: str,
    manifest_id: str,
    role: str,
    path: Path,
    job_id: str,
) -> StudioManifestRef:
    return StudioManifestRef(
        kind=kind,
        id=manifest_id,
        role=role,
        uri=str(path),
        metadata={"job_id": job_id},
    )


def _studio_artifact_ref(
    artifact: ArtifactRef,
    *,
    kind: str,
) -> StudioArtifactRef:
    return StudioArtifactRef(
        kind=kind,
        id=artifact.artifact_id or f"artifact:{artifact.logical_name}",
        role=artifact.role,
        uri=artifact.uri,
        media_type=artifact.media_type,
        metadata={
            **artifact.metadata,
            "logical_name": artifact.logical_name,
            "sha256": artifact.sha256,
            "storage_backend": artifact.storage_backend,
        },
    )


def _complete_stage_with_manifest(
    workspace: StudioWorkspaceSpec,
    stage: StudioStageSpec,
    *,
    manifest_ref: StudioManifestRef,
    artifact_refs: list[StudioArtifactRef],
    output_collection_kind: str,
    output_collection_id: str,
    output_collection_label: str,
    completed_metadata: dict[str, Any],
) -> None:
    completed_at = utc_now().isoformat()
    stage.status = "completed"
    stage.validation = StudioValidationState(
        valid=True,
        checked_at=completed_at,
        metadata={
            "materialized_by": "feedbax.studio_execution",
            "manifest_id": manifest_ref.id,
            **completed_metadata,
        },
    )
    stage.manifest_refs = _upsert_manifest_ref(stage.manifest_refs, manifest_ref)
    stage.artifact_refs = _upsert_many_artifact_refs(stage.artifact_refs, artifact_refs)
    stage.output_collections = _upsert_manifest_in_output_collection(
        stage.output_collections,
        collection_kind=output_collection_kind,
        collection_id=output_collection_id,
        collection_label=output_collection_label,
        stage_id=stage.id,
        manifest_ref=manifest_ref,
    )
    stage.metadata = {
        **stage.metadata,
        "last_materialization": {
            "manifest_id": manifest_ref.id,
            "completed_at": completed_at,
            **completed_metadata,
        },
    }
    workspace.manifest_refs = _upsert_manifest_ref(workspace.manifest_refs, manifest_ref)
    workspace.artifact_refs = _upsert_many_artifact_refs(
        workspace.artifact_refs,
        artifact_refs,
    )
    workspace.collections = _upsert_many_collection_refs(
        workspace.collections,
        [*stage.input_collections, *stage.output_collections],
    )
    _replace_stage(workspace, stage)


def _upsert_manifest_in_output_collection(
    collections: list[StudioCollectionRef],
    *,
    collection_kind: str,
    collection_id: str,
    collection_label: str,
    stage_id: str,
    manifest_ref: StudioManifestRef,
) -> list[StudioCollectionRef]:
    updated: list[StudioCollectionRef] = []
    added = False
    for collection in collections:
        if collection.kind == collection_kind:
            collection = collection.model_copy(deep=True)
            collection.item_refs = _upsert_manifest_ref(collection.item_refs, manifest_ref)
            added = True
        updated.append(collection)
    if not added:
        updated.append(
            StudioCollectionRef(
                id=collection_id,
                kind=collection_kind,
                label=collection_label,
                source_stage_id=stage_id,
                item_refs=[manifest_ref],
            )
        )
    return updated


def _upsert_collection_ref(
    collections: list[StudioCollectionRef],
    ref: StudioCollectionRef,
) -> list[StudioCollectionRef]:
    return [item for item in collections if item.id != ref.id] + [ref]


def _upsert_many_collection_refs(
    collections: list[StudioCollectionRef],
    refs: list[StudioCollectionRef],
) -> list[StudioCollectionRef]:
    merged = collections
    for ref in refs:
        merged = _upsert_collection_ref(merged, ref)
    return merged


def _upsert_training_manifest_in_outputs(
    collections: list[StudioCollectionRef],
    manifest_ref: StudioManifestRef,
    stage_id: str,
) -> list[StudioCollectionRef]:
    updated: list[StudioCollectionRef] = []
    added = False
    for collection in collections:
        if collection.kind == "training_runs":
            collection = collection.model_copy(deep=True)
            collection.item_refs = _upsert_manifest_ref(collection.item_refs, manifest_ref)
            added = True
        updated.append(collection)
    if not added:
        updated.append(
            StudioCollectionRef(
                id="collection:training-runs",
                kind="training_runs",
                label="Training runs",
                source_stage_id=stage_id,
                item_refs=[manifest_ref],
            )
        )
    return updated


def _replace_stage(workspace: StudioWorkspaceSpec, updated: StudioStageSpec) -> None:
    workspace.stages = [updated if stage.id == updated.id else stage for stage in workspace.stages]


def _upsert_artifact_ref(
    refs: list[StudioArtifactRef],
    ref: StudioArtifactRef,
) -> list[StudioArtifactRef]:
    return [item for item in refs if not (item.kind == ref.kind and item.id == ref.id)] + [ref]


def _upsert_many_artifact_refs(
    refs: list[StudioArtifactRef],
    new_refs: list[StudioArtifactRef],
) -> list[StudioArtifactRef]:
    merged = refs
    for ref in new_refs:
        merged = _upsert_artifact_ref(merged, ref)
    return merged


def _upsert_manifest_ref(
    refs: list[StudioManifestRef],
    ref: StudioManifestRef,
) -> list[StudioManifestRef]:
    return [item for item in refs if not (item.kind == ref.kind and item.id == ref.id)] + [ref]
