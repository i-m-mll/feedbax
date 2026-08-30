"""Bind Studio workspace stages to invocations and inert backend plans."""

from __future__ import annotations

import json
import uuid
import copy
import hashlib
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from feedbax.execution.records import Invocation, InvocationExecutionPolicy, invocation_for_operation
from feedbax.orchestration.drivers.local import local_driver_registration
from feedbax.orchestration.drivers.runpod import runpod_driver_registration
from feedbax.orchestration.realization import (
    BackendPlan,
    BackendRealizationRequest,
    MachineShape,
    OrchestrationBackend,
)
from feedbax.workflow.plan import LogicalKey, Operation, PlanNode, build_workflow_plan
from feedbax.analysis.evaluation import (
    EvaluationRecipeExecutionError,
    execute_evaluation_run_spec,
)
from feedbax.analysis.reports import STUDIO_REPORT_TYPE, execute_report_spec
from feedbax.analysis.manifest_inputs import authenticated_manifest_ref
from feedbax.analysis.specs import execute_analysis_run_spec
from feedbax.contracts.manifest import (
    AnalysisRunSpec,
    ArtifactRef,
    EntrypointRef,
    EvaluationRunSpec,
    EvaluationRunManifest,
    CheckpointCandidateRef,
    CheckpointScorerIdentity,
    CheckpointSelectionBank,
    CheckpointSelectionGroup,
    CheckpointSelectionManifest,
    CheckpointSelectionSpec,
    ParentRef,
    Provenance,
    ReportSpec,
    TrainingRunManifest,
    checkpoint_selection_manifest_id,
    default_manifest_root,
    evaluation_run_manifest_id,
    evaluation_states_cache_path,
    load_manifest,
    planned_training_run_manifest_id,
    spec_payload,
    utc_now,
    write_manifest,
)
from feedbax.contracts.selection import (
    SelectionSpec,
    manifest_index_rows_from_records,
    preview_selection_spec,
)
from feedbax.persistence.manifest_index import (
    find_manifest_paths_by_id,
    iter_indexed_manifest_records_by_kind,
)
from feedbax.contracts.migrations import migrate_studio_task_binding_spec
from feedbax.studio.sweep_matrix import (
    SweepMatrixError,
    materialize_sweep_matrix,
    matrix_spec_from_selection,
    _coordinate_label,
    _expand_coordinates,
    _parse_axes,
    _parse_combination,
    _validate_group_axes,
    _variation_values,
)
from feedbax.training.run_matrix import MaterializedMatrixRow
from feedbax.studio.schema import SchemaValidationIssue, validate_task_binding_schema
from feedbax.contracts.graph import (
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

if TYPE_CHECKING:
    from feedbax.plugins.application import ApplicationRegistryBundle

ExecutionTarget = Literal["local", "gcp", "runpod", "manual"]

class _Unset:
    """Sentinel distinguishing an omitted argument from an explicit `None`."""


_UNSET = _Unset()


class StudioExecutionModel(BaseModel):
    """Base model for Studio execution preparation records."""

    model_config = ConfigDict(extra="forbid")


class StudioTrainingExecutionRequest(StudioExecutionModel):
    """Request to bind a Studio train stage to an inert backend realization."""

    workspace: StudioWorkspaceSpec
    graph: GraphSpec
    stage_id: Optional[str] = None
    backend: Literal["local", "runpod"] = "local"
    job_id: Optional[str] = None
    local_cwd: Optional[str] = None
    backend_realization: BackendRealizationRequest | None = None
    queue_target: Optional[ExecutionTarget] = None
    queue_manifest_ids: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class StudioTrainingExecutionPreparation(StudioExecutionModel):
    """Prepared invocation and inert backend plan plus workspace updates."""

    workspace: StudioWorkspaceSpec
    graph: GraphSpec
    stage_id: str
    scenario_id: str
    invocation: Invocation
    backend_plan: BackendPlan


EvalCheckpointPolicyMode = Literal["last", "best-by-metric", "every-k"]
EvalReprocessMode = Literal["missing", "missing_failed", "all", "stale"]


class StudioEvaluationCheckpointPolicy(StudioExecutionModel):
    """Checkpoint policy applied while lowering a selected training run to evals."""

    mode: EvalCheckpointPolicyMode = "last"
    metric: Optional[str] = None
    objective: Literal["minimize", "maximize"] = "minimize"
    every_k: Optional[int] = Field(default=None, ge=1)
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_policy_parameters(self) -> "StudioEvaluationCheckpointPolicy":
        if self.mode == "best-by-metric" and not (self.metric or "").strip():
            raise ValueError("best-by-metric checkpoint policy requires metric")
        if self.mode == "every-k" and self.every_k is None:
            raise ValueError("every-k checkpoint policy requires every_k")
        return self


class StudioEvaluationMatrixRequest(StudioExecutionModel):
    """Request to preview or stage a Studio eval matrix."""

    workspace: StudioWorkspaceSpec
    stage_id: Optional[str] = None
    selection_spec: Optional[SelectionSpec] = None
    training_run_ids: list[str] = Field(default_factory=list)
    eval_params: dict[str, Any] = Field(default_factory=dict)
    condition_matrix: dict[str, Any] = Field(default_factory=dict)
    checkpoint_policy: StudioEvaluationCheckpointPolicy = Field(
        default_factory=StudioEvaluationCheckpointPolicy
    )
    reprocess: EvalReprocessMode = "missing"
    job_id: Optional[str] = None
    root: Optional[str] = None
    issues: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class StudioEvaluationMatrixPreview(StudioExecutionModel):
    """Preview counts for a staged Studio eval matrix."""

    workspace: StudioWorkspaceSpec
    stage_id: str
    selected_training_run_count: int
    condition_count: int
    checkpoint_policy_count: int
    total_eval_count: int
    materialized_count: int
    pending_count: int
    failed_count: int
    new_manifest_count: int
    launch_count: int
    evaluation_run_ids: list[str]
    checkpoint_selection_ids: list[str]
    summary: str


class StudioEvaluationStagingResult(StudioExecutionModel):
    """Result from writing pending EvaluationRunManifest rows."""

    workspace: StudioWorkspaceSpec
    stage_id: str
    preview: StudioEvaluationMatrixPreview
    manifest_refs: list[StudioManifestRef]
    checkpoint_selection_refs: list[StudioManifestRef]


class StudioEvaluationLocalRunRequest(StudioEvaluationMatrixRequest):
    """Request to stage and execute selected Studio evaluations locally."""

    timeout: Optional[float] = None


class StudioEvaluationLocalRunResult(StudioExecutionModel):
    """Result from executing staged Studio evaluations."""

    workspace: StudioWorkspaceSpec
    stage_id: str
    preview: StudioEvaluationMatrixPreview
    manifest_refs: list[StudioManifestRef]
    completed_count: int
    failed_count: int
    skipped_count: int = 0
    skipped_failed_count: int = 0
    errors: list[str] = Field(default_factory=list)


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
    """Raised when a workspace cannot be bound to an invocation and backend plan."""


def prepare_studio_training_execution(
    request: StudioTrainingExecutionRequest,
    *,
    registry_bundle: ApplicationRegistryBundle,
) -> StudioTrainingExecutionPreparation:
    """Prepare one provider-neutral invocation and one inert backend plan."""

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
    if scenario.training_spec is None:
        raise StudioExecutionPreparationError(
            f"Scenario {scenario.id!r} cannot execute without a training_spec"
        )
    if scenario.task_spec is None:
        raise StudioExecutionPreparationError(
            f"Scenario {scenario.id!r} cannot execute without a task_spec"
        )

    validation = _validate_training_scenario(
        graph=request.graph.model_dump(mode="json", exclude_none=True),
        training_spec=scenario.training_spec,
        task_spec=scenario.task_spec,
        component_registry=registry_bundle.components,
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
    execution_target = _request_execution_target(stage, request)
    invocation, backend_plan = _build_invocation_backend_plan(
        request=request,
        workspace=workspace,
        stage=stage,
        job_id=job_id,
    )

    prepared_at = utc_now().isoformat()
    plan_ref = StudioArtifactRef(
        kind="BackendPlan",
        id=f"backend-plan:{backend_plan.backend_plan_id}",
        role="backend_plan",
        uri=f"backend-plan:{backend_plan.backend_plan_id}",
        media_type="application/json",
        metadata={
            "backend": backend_plan.backend_id,
            "invocation_id": invocation.invocation_id,
            "stage_id": stage.id,
            "scenario_id": stage.scenario_id,
            "prepared_at": prepared_at,
        },
    )
    stage.execution_spec = invocation.model_dump(mode="json", exclude_none=True)
    stage.status = "ready"
    stage.validation = StudioValidationState(
        valid=True,
        checked_at=prepared_at,
        warnings=validation.warnings,
        metadata={
            **validation.metadata,
            "execution_job_id": job_id,
            "invocation_id": invocation.invocation_id,
            "backend_plan_id": backend_plan.backend_plan_id,
            "execution_backend": backend_plan.backend_id,
        },
    )
    stage.artifact_refs = _upsert_artifact_ref(stage.artifact_refs, plan_ref)
    stage.manifest_refs = _upsert_manifest_ref(
        stage.manifest_refs,
        StudioManifestRef(
            kind="BackendPlan",
            id=f"backend-plan:{backend_plan.backend_plan_id}",
            role="backend_plan",
            uri=plan_ref.uri,
            metadata=plan_ref.metadata,
        ),
    )
    if request.queue_manifest_ids:
        staged_training_refs, staged_run_set_ref, staged_summary = _queue_training_manifest_subset(
            stage=stage,
            request=request,
            execution_target=execution_target,
        )
    else:
        staged_training_refs, staged_run_set_ref, staged_summary = (
            _stage_pending_training_manifests(
                workspace=workspace,
                stage=stage,
                scenario_id=stage.scenario_id,
                graph_spec=request.graph.model_dump(mode="json", exclude_none=True),
                training_spec=scenario.training_spec,
                task_spec=scenario.task_spec,
                task_binding_spec=scenario.task_binding_spec.model_dump(
                    mode="json", exclude_none=True
                )
                if scenario.task_binding_spec is not None
                else None,
                request=request,
                job_id=job_id,
                execution_target=execution_target,
                registry_bundle=registry_bundle,
            )
        )
    for staged_ref in staged_training_refs:
        stage.manifest_refs = _upsert_manifest_ref(stage.manifest_refs, staged_ref)
        stage.output_collections = _upsert_training_manifest_in_outputs(
            stage.output_collections,
            staged_ref,
            stage.id,
        )
    if staged_run_set_ref is not None:
        stage.manifest_refs = _upsert_manifest_ref(stage.manifest_refs, staged_run_set_ref)
        stage.output_collections = _upsert_manifest_in_output_collection(
            stage.output_collections,
            collection_kind="training_run_sets",
            collection_id="collection:training-run-sets",
            collection_label="Training run sets",
            stage_id=stage.id,
            manifest_ref=staged_run_set_ref,
        )
    stage.metadata = {
        **stage.metadata,
        "last_backend_plan": {
            "job_id": job_id,
            "invocation_id": invocation.invocation_id,
            "backend_plan_id": backend_plan.backend_plan_id,
            "backend": backend_plan.backend_id,
            "prepared_at": prepared_at,
        },
        "last_staged_training": staged_summary,
    }
    workspace.artifact_refs = _upsert_artifact_ref(workspace.artifact_refs, plan_ref)
    for staged_ref in staged_training_refs:
        workspace.manifest_refs = _upsert_manifest_ref(workspace.manifest_refs, staged_ref)
    if staged_run_set_ref is not None:
        workspace.manifest_refs = _upsert_manifest_ref(workspace.manifest_refs, staged_run_set_ref)
    workspace.collections = _upsert_many_collection_refs(
        workspace.collections,
        stage.output_collections,
    )
    _replace_stage(workspace, stage)

    return StudioTrainingExecutionPreparation(
        workspace=workspace,
        graph=request.graph,
        stage_id=stage.id,
        scenario_id=stage.scenario_id,
        invocation=invocation,
        backend_plan=backend_plan,
    )


def preview_studio_evaluation_matrix(
    request: StudioEvaluationMatrixRequest,
) -> StudioEvaluationMatrixPreview:
    """Preview deterministic evaluation matrix lowering without writing manifests."""

    plan = _evaluation_matrix_plan(request)
    return _evaluation_preview_from_plan(plan)


def stage_studio_evaluation_matrix(
    request: StudioEvaluationMatrixRequest,
) -> StudioEvaluationStagingResult:
    """Write pending EvaluationRunManifest rows for a Studio eval matrix."""

    plan = _evaluation_matrix_plan(request)
    root_path = _request_root(request.root)
    workspace = plan["workspace"].model_copy(deep=True)
    eval_stage = plan["stage"].model_copy(deep=True)
    eval_refs: list[StudioManifestRef] = []
    checkpoint_refs: list[StudioManifestRef] = []
    for item in plan["items"]:
        checkpoint_manifest, checkpoint_path = _write_checkpoint_selection_manifest(
            item,
            root=root_path,
        )
        checkpoint_ref = _studio_manifest_ref(
            checkpoint_manifest.kind,
            checkpoint_manifest.id,
            "checkpoint_selection",
            checkpoint_path,
            plan["job_id"],
        )
        checkpoint_ref.metadata = {
            **checkpoint_ref.metadata,
            "status": checkpoint_manifest.status,
            "stage_id": eval_stage.id,
            "selected_training_run_id": item["training_ref"].id,
            "checkpoint_policy": item["checkpoint_policy"],
        }
        checkpoint_refs.append(checkpoint_ref)

        manifest, path = _write_pending_evaluation_manifest(
            item,
            checkpoint_ref=checkpoint_ref,
            request=request,
            root=root_path,
        )
        manifest_ref = _pending_evaluation_manifest_ref(
            manifest,
            path,
            stage=eval_stage,
            job_id=plan["job_id"],
        )
        eval_refs.append(manifest_ref)

    eval_stage.input_collections = _upsert_collection_ref(
        eval_stage.input_collections,
        StudioCollectionRef(
            id="collection:selected-training-runs",
            kind="training_runs",
            label="Selected training runs",
            source_stage_id=eval_stage.id,
            item_refs=plan["training_stage_refs"],
            metadata={"selection_spec": plan["selection_spec"]},
        ),
    )
    for checkpoint_ref in checkpoint_refs:
        eval_stage.manifest_refs = _upsert_manifest_ref(eval_stage.manifest_refs, checkpoint_ref)
    for manifest_ref in eval_refs:
        eval_stage.manifest_refs = _upsert_manifest_ref(eval_stage.manifest_refs, manifest_ref)
        eval_stage.output_collections = _upsert_manifest_in_output_collection(
            eval_stage.output_collections,
            collection_kind="evaluation_runs",
            collection_id="collection:evaluation-runs",
            collection_label="Evaluation runs",
            stage_id=eval_stage.id,
            manifest_ref=manifest_ref,
        )
    eval_stage.status = "ready"
    eval_stage.metadata = {
        **eval_stage.metadata,
        "last_staged_evaluation": {
            "staged_at": utc_now().isoformat(),
            "total_eval_count": len(eval_refs),
            "checkpoint_policy": plan["checkpoint_policy"],
            "reprocess": request.reprocess,
        },
    }
    workspace.manifest_refs = _upsert_many_manifest_refs(workspace.manifest_refs, checkpoint_refs)
    workspace.manifest_refs = _upsert_many_manifest_refs(workspace.manifest_refs, eval_refs)
    workspace.collections = _upsert_many_collection_refs(
        workspace.collections,
        eval_stage.output_collections,
    )
    _replace_stage(workspace, eval_stage)
    staged_plan = {**plan, "workspace": workspace, "stage": eval_stage}
    preview = _evaluation_preview_from_plan(staged_plan)
    return StudioEvaluationStagingResult(
        workspace=workspace,
        stage_id=eval_stage.id,
        preview=preview,
        manifest_refs=eval_refs,
        checkpoint_selection_refs=checkpoint_refs,
    )


def run_studio_evaluation_local_execution(
    request: StudioEvaluationLocalRunRequest,
    *,
    registry_bundle: ApplicationRegistryBundle,
) -> StudioEvaluationLocalRunResult:
    """Stage and execute selected Studio evaluations through registered recipes."""

    stale_launch_ids = (
        _stale_evaluation_launch_ids(request) if request.reprocess == "stale" else None
    )
    staged = stage_studio_evaluation_matrix(request)
    root_path = _request_root(request.root)
    workspace = staged.workspace.model_copy(deep=True)
    eval_stage = _select_stage_by_kind(workspace, "eval")
    completed = 0
    failed = 0
    skipped = 0
    skipped_failed = 0
    errors: list[str] = []
    updated_refs: list[StudioManifestRef] = []

    for ref in staged.manifest_refs:
        manifest = load_manifest(ref.uri)
        if not isinstance(manifest, EvaluationRunManifest):
            failed += 1
            errors.append(f"Manifest {ref.id!r} is not an EvaluationRunManifest")
            continue
        should_launch = (
            ref.id in stale_launch_ids
            if stale_launch_ids is not None
            else _should_launch_status(manifest.status, request.reprocess)
        )
        if not should_launch:
            skipped += 1
            if manifest.status == "failed":
                failed += 1
                skipped_failed += 1
            updated_refs.append(ref)
            continue
        try:
            spec = EvaluationRunSpec.model_validate(manifest.evaluation_spec.inline)
            executed, path = execute_evaluation_run_spec(
                spec,
                root=root_path,
                provenance=manifest.provenance,
                issues=request.issues,
                metadata=manifest.metadata,
                force=request.reprocess == "all",
                registry=registry_bundle.evaluation_recipes,
            )
            completed += 1 if executed.status == "completed" else 0
            failed += 1 if executed.status == "failed" else 0
            updated_refs.append(
                _pending_evaluation_manifest_ref(
                    executed,
                    path,
                    stage=eval_stage,
                    job_id=request.job_id or ref.metadata.get("job_id") or "studio-eval",
                )
            )
        except EvaluationRecipeExecutionError as exc:
            failed += 1
            errors.append(str(exc.__cause__ or exc))
            updated_refs.append(
                _pending_evaluation_manifest_ref(
                    exc.manifest,
                    exc.path,
                    stage=eval_stage,
                    job_id=request.job_id or ref.metadata.get("job_id") or "studio-eval",
                )
            )
        except Exception as exc:  # pragma: no cover - defensive API boundary.
            failed += 1
            errors.append(str(exc))

    for manifest_ref in updated_refs:
        eval_stage.manifest_refs = _upsert_manifest_ref(eval_stage.manifest_refs, manifest_ref)
        eval_stage.output_collections = _upsert_manifest_in_output_collection(
            eval_stage.output_collections,
            collection_kind="evaluation_runs",
            collection_id="collection:evaluation-runs",
            collection_label="Evaluation runs",
            stage_id=eval_stage.id,
            manifest_ref=manifest_ref,
        )
    eval_stage.status = "completed" if failed == 0 else "failed"
    eval_stage.metadata = {
        **eval_stage.metadata,
        "last_evaluation_launch": {
            "completed_count": completed,
            "failed_count": failed,
            "skipped_count": skipped,
            "skipped_failed_count": skipped_failed,
            "launched_at": utc_now().isoformat(),
            "reprocess": request.reprocess,
        },
    }
    workspace.manifest_refs = _upsert_many_manifest_refs(workspace.manifest_refs, updated_refs)
    workspace.collections = _upsert_many_collection_refs(
        workspace.collections,
        eval_stage.output_collections,
    )
    _replace_stage(workspace, eval_stage)
    preview = preview_studio_evaluation_matrix(
        StudioEvaluationMatrixRequest(
            workspace=workspace,
            stage_id=eval_stage.id,
            selection_spec=request.selection_spec,
            training_run_ids=request.training_run_ids,
            eval_params=request.eval_params,
            condition_matrix=request.condition_matrix,
            checkpoint_policy=request.checkpoint_policy,
            reprocess=request.reprocess,
            job_id=request.job_id,
            root=request.root,
            issues=request.issues,
            metadata=request.metadata,
        )
    )
    return StudioEvaluationLocalRunResult(
        workspace=workspace,
        stage_id=eval_stage.id,
        preview=preview,
        manifest_refs=updated_refs,
        completed_count=completed,
        failed_count=failed,
        skipped_count=skipped,
        skipped_failed_count=skipped_failed,
        errors=errors,
    )


def _stale_evaluation_launch_ids(request: StudioEvaluationMatrixRequest) -> set[str]:
    plan = _evaluation_matrix_plan(request)
    root = plan["root"]
    launch_ids: set[str] = set()
    for item in plan["items"]:
        manifest = _existing_manifest(item["evaluation_id"], root=root)
        status = manifest.status if isinstance(manifest, EvaluationRunManifest) else None
        if _should_launch_status(status, "stale"):
            launch_ids.add(item["evaluation_id"])
    return launch_ids


def materialize_studio_pipeline(
    request: StudioPipelineMaterializationRequest,
    *,
    registry_bundle: ApplicationRegistryBundle,
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
                registry_bundle=registry_bundle,
            )
        elif stage_kind == "analysis":
            manifest_path, stage_artifacts = _materialize_analysis_stage(
                workspace,
                root_path=root_path,
                job_id=f"{base_job_id}-analysis",
                issues=request.issues,
                request_metadata=request.metadata,
                registry_bundle=registry_bundle,
            )
        elif stage_kind == "report":
            manifest_path, stage_artifacts = _materialize_report_stage(
                workspace,
                root_path=root_path,
                job_id=f"{base_job_id}-report",
                issues=request.issues,
                request_metadata=request.metadata,
                registry_bundle=registry_bundle,
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
    component_registry: Any,
) -> StudioValidationState:
    from feedbax.integrations.provider import (
        validate_graph_spec,
        validate_task_spec,
        validate_training_spec,
    )

    graph_result = validate_graph_spec(graph, component_registry=component_registry)
    training_result = validate_training_spec(
        training_spec,
        graph_spec=graph,
        task_spec=task_spec,
        component_registry=component_registry,
    )
    task_result = validate_task_spec(task_spec)
    task_binding_errors = _validate_task_binding_spec(
        graph, task_binding_spec, component_registry=component_registry
    )

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
        metadata={"validated_by": "feedbax.studio.execution"},
    )


def _validate_task_binding_spec(
    graph: dict[str, Any],
    task_binding_spec: dict[str, Any] | None,
    *,
    component_registry: Any,
) -> list[StudioValidationIssue]:
    if task_binding_spec is None:
        return []
    try:
        migrated_spec = migrate_studio_task_binding_spec(task_binding_spec).payload
        validated_spec = StudioTaskBindingSpec.model_validate(migrated_spec)
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
    except ValueError as exc:
        return [
            StudioValidationIssue(
                type="invalid_task_binding_spec",
                message=str(exc),
                location={"path": "/task_binding_spec/schema_version"},
            )
        ]
    graph_spec = GraphSpec.model_validate(graph)
    return _schema_issues_to_studio(
        validate_task_binding_schema(
            validated_spec,
            graph_spec,
            "/task_binding_spec",
            component_registry=component_registry,
        )
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


def _build_invocation_backend_plan(
    *,
    request: StudioTrainingExecutionRequest,
    workspace: StudioWorkspaceSpec,
    stage: StudioStageSpec,
    job_id: str,
) -> tuple[Invocation, BackendPlan]:
    scenario = workspace.scenarios[stage.scenario_id or ""]
    semantic_payload = {
        "graph": request.graph.model_dump(mode="json", exclude_none=True),
        "training": scenario.training_spec,
        "task": scenario.task_spec,
        "task_binding": (
            scenario.task_binding_spec.model_dump(mode="json", exclude_none=True)
            if scenario.task_binding_spec is not None
            else None
        ),
        "objective": scenario.objective_spec,
        "temporal": scenario.temporal_spec,
    }
    semantic_hash = hashlib.sha256(
        json.dumps(semantic_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    key = LogicalKey("campaign", f"studio/{stage.id}/{scenario.id}")
    node = PlanNode(
        key=key,
        source_ref=f"studio:{workspace.id}:{stage.id}:{scenario.id}",
        operation=Operation(
            type_id="feedbax.operation.train",
            parameters={
                "compiled_schema_id": "feedbax.studio.training_scenario",
                "semantic_hash": semantic_hash,
            },
            output_types={"training_run": "feedbax.training_run"},
            determinism="seeded",
            cache_policy="never",
            effect="external",
            capabilities=("training",),
        ),
        content_hash=semantic_hash,
        execution_identity=semantic_hash,
    )
    workflow = build_workflow_plan(key, (node,), ())
    realization = request.backend_realization
    if realization is None:
        if request.backend != "local":
            raise StudioExecutionPreparationError(
                "paid-resource-capable Studio planning requires an exact backend_realization; "
                "backend selection alone cannot mint machine shape or expected cost"
            )
        realization = _default_local_realization(job_id=job_id, cwd=request.local_cwd)
    invocation = invocation_for_operation(
        workflow,
        key,
        bound_inputs={},
        execution_policy=InvocationExecutionPolicy(
            timeout_seconds=realization.timeout_seconds,
            max_attempts=1,
        ),
        scientific_seeds=_studio_scientific_seeds(scenario.training_spec),
    )
    registration = (
        local_driver_registration() if request.backend == "local" else runpod_driver_registration()
    )
    backend = OrchestrationBackend(
        backend_id=registration.name,
        supported_scientific_capabilities=frozenset({"training"}),
        driver_capabilities=registration.supported_capabilities,
    )
    if realization.configuration.get("job_id") not in {None, job_id}:
        raise StudioExecutionPreparationError(
            "backend realization job_id does not match the Studio preparation job_id"
        )
    plan = backend.realize("training", (invocation, realization))
    if not isinstance(plan, BackendPlan):
        raise TypeError("backend realization did not produce a BackendPlan")
    return invocation, plan


def _default_local_realization(*, job_id: str, cwd: str | None) -> BackendRealizationRequest:
    repo_root = Path(__file__).resolve().parents[2]
    revision = subprocess.run(
        ["git", "--no-optional-locks", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    ).stdout.strip()
    listed = subprocess.run(
        ["git", "--no-optional-locks", "ls-files", "-co", "--exclude-standard", "feedbax"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    ).stdout.splitlines()
    code_hash = hashlib.sha256(revision.encode("ascii") + b"\0")
    for relative in sorted(set(listed)):
        path = repo_root / relative
        if not path.is_file():
            continue
        code_hash.update(relative.encode("utf-8") + b"\0")
        code_hash.update(path.read_bytes())
    code_digest = code_hash.hexdigest()
    lock_path = repo_root / "uv.lock"
    environment_digest = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    return BackendRealizationRequest(
        adapter_id="feedbax.orchestration.local",
        adapter_version="1",
        capability_variant="local-stop",
        code_bundle_id=f"feedbax-working-tree:sha256:{code_digest}",
        environment_bundle_id=f"uv-lock:sha256:{environment_digest}",
        command=(
            sys.executable,
            "-m",
            "feedbax.bin.studio_pipeline",
            "materialize-training",
            "--graph",
            "graph-spec.json",
            "--training",
            "training-spec.json",
            "--task",
            "task-spec.json",
            "--task-binding",
            "task-binding-spec.json",
            "--output",
            "artifacts/training-summary.json",
        ),
        machine=MachineShape(),
        timeout_seconds=3600,
        retry_classification="same-plan",
        external_effect_key=f"studio-local:{job_id}",
        configuration={"job_id": job_id, "cwd": cwd},
    )


def _studio_scientific_seeds(training_spec: Mapping[str, Any]) -> dict[str, int]:
    seeds = training_spec.get("seeds")
    if not isinstance(seeds, Mapping):
        seed = training_spec.get("seed")
        return {"training": seed} if isinstance(seed, int) and not isinstance(seed, bool) else {}
    return {
        str(name): value
        for name, value in sorted(seeds.items())
        if isinstance(value, int) and not isinstance(value, bool)
    }


def _restaged_metadata(
    existing_metadata: Mapping[str, Any],
    *,
    manifest_id: str,
    restaged_from_status: str,
) -> dict[str, Any]:
    metadata = {
        **existing_metadata,
        "planned": True,
        "restaged_at": utc_now().isoformat(),
        "restaged_from_status": restaged_from_status,
    }
    for key in ("superseded_by", "supersedes"):
        if metadata.get(key) == manifest_id:
            metadata.pop(key)
    return metadata


def _reuse_pending_training_manifest(
    manifest_id: str,
    root: Path,
) -> tuple[TrainingRunManifest, Path] | None:
    """Return an already-staged manifest for `manifest_id`, restaging cancelled runs.

    Returns `None` when nothing indexed under `manifest_id` can stand in for a
    freshly built pending manifest, in which case the caller builds one.
    """
    from feedbax.contracts.manifest import load_manifest
    from feedbax.persistence.manifest_index import find_manifest_paths_by_id

    existing_paths = find_manifest_paths_by_id(manifest_id, root=root)
    if not existing_paths:
        return None
    existing = load_manifest(existing_paths[0])
    if not isinstance(existing, TrainingRunManifest):
        return None
    if existing.status != "cancelled":
        return existing, existing_paths[0]
    restaged = existing.model_copy(
        update={
            "status": "pending",
            "completed_at": None,
            "metadata": _restaged_metadata(
                existing.metadata,
                manifest_id=manifest_id,
                restaged_from_status="cancelled",
            ),
        }
    )
    return restaged, write_manifest(restaged, root=root)


def _build_pending_training_manifest(
    *,
    manifest_id: str,
    root: Path,
    workspace: StudioWorkspaceSpec,
    stage: StudioStageSpec,
    scenario_id: str | None,
    graph_spec: dict[str, Any],
    training_spec: dict[str, Any],
    task_spec: dict[str, Any],
    task_binding_spec: dict[str, Any] | None,
    training_spec_kind: str,
    request: StudioTrainingExecutionRequest,
    execution_target: str,
    axis_coordinates: dict[str, Any],
    studio_run_identity: dict[str, Any],
    entrypoint_metadata: dict[str, Any],
    label: str | None | _Unset = _UNSET,
    run_set_id: str | None = None,
    overrides: list[Any] | None = None,
    total_batches: Any | None = None,
) -> tuple[TrainingRunManifest, Path]:
    """Write the single durable pending `TrainingRunManifest` document.

    Both the single-run and the matrix-row staging paths emit through here.
    `studio_run_identity` carries the identity fields that distinguish those
    paths inside `metadata["studio"]` (the single-run `seed`, or the matrix
    row's `axis_value_indices` and `run_set_id`); it is spliced in immediately
    after `axis_coordinates` so the emitted key order is stable per path.
    """
    now = utc_now()
    studio_metadata = {
        "workspace_id": workspace.id,
        "workspace_schema_version": workspace.schema_version,
        "stage_id": stage.id,
        "stage_kind": stage.kind,
        "scenario_id": scenario_id,
        "selection_spec": stage.selection_spec,
        "axis_coordinates": axis_coordinates,
        **studio_run_identity,
        "planned_training_run_id": manifest_id,
        "execution_target": execution_target,
        "execution_backend": request.backend,
    }
    metadata: dict[str, Any] = {}
    if not isinstance(label, _Unset):
        metadata["name"] = label
        metadata["label"] = label
    metadata["studio"] = studio_metadata
    metadata["planned"] = True
    metadata["staged_at"] = now.isoformat()
    metadata["execution_target"] = execution_target
    metadata["execution_backend"] = request.backend
    metadata["spec_hashes"] = _ui_spec_hashes(
        {
            "graph_spec": graph_spec,
            "training_spec": training_spec,
            "task_spec": task_spec,
            "task_binding_spec": task_binding_spec,
        }
    )
    manifest = TrainingRunManifest(
        id=manifest_id,
        run_set_id=run_set_id,
        job_id=request.job_id,
        status="pending",
        graph_spec=spec_payload("GraphSpec", graph_spec),
        training_spec=spec_payload(training_spec_kind, training_spec),
        task_spec=spec_payload("TaskSpec", task_spec),
        task_binding_spec=spec_payload("StudioTaskBindingSpec", task_binding_spec)
        if task_binding_spec is not None
        else None,
        overrides=overrides if overrides is not None else [],
        summary_metrics={
            key: value
            for key, value in {"total_batches": total_batches}.items()
            if value is not None
        },
        provenance=Provenance(
            entrypoint=EntrypointRef(
                kind="feedbax-studio-pipeline",
                name="stage_training_run",
                metadata=entrypoint_metadata,
            ),
            issues=list(request.issues),
            metadata={**request.metadata, "studio": studio_metadata},
        ),
        metadata=metadata,
    )
    return manifest, write_manifest(manifest, root=root)


def _write_pending_training_manifest(
    *,
    workspace: StudioWorkspaceSpec,
    stage: StudioStageSpec,
    scenario_id: str | None,
    graph_spec: dict[str, Any],
    training_spec: dict[str, Any],
    task_spec: dict[str, Any],
    task_binding_spec: dict[str, Any] | None,
    request: StudioTrainingExecutionRequest,
    execution_target: str,
) -> tuple[TrainingRunManifest, Path]:
    seed = _training_seed(training_spec)
    axis_coordinates = _stage_axis_coordinates(stage)
    manifest_id = planned_training_run_manifest_id(
        graph_spec=graph_spec,
        training_spec=training_spec,
        task_spec=task_spec,
        task_binding_spec=task_binding_spec,
        seed=seed,
        axis_coordinates=axis_coordinates,
    )

    root_path = default_manifest_root()
    reused = _reuse_pending_training_manifest(manifest_id, root_path)
    if reused is not None:
        return reused

    return _build_pending_training_manifest(
        manifest_id=manifest_id,
        root=root_path,
        workspace=workspace,
        stage=stage,
        scenario_id=scenario_id,
        graph_spec=graph_spec,
        training_spec=training_spec,
        task_spec=task_spec,
        task_binding_spec=task_binding_spec,
        training_spec_kind="TrainingSpec",
        request=request,
        execution_target=execution_target,
        axis_coordinates=axis_coordinates,
        studio_run_identity={"seed": seed},
        entrypoint_metadata={"job_id": request.job_id},
        total_batches=training_spec.get("n_batches"),
    )


def _stage_pending_training_manifests(
    *,
    workspace: StudioWorkspaceSpec,
    stage: StudioStageSpec,
    scenario_id: str | None,
    graph_spec: dict[str, Any],
    training_spec: dict[str, Any],
    task_spec: dict[str, Any],
    task_binding_spec: dict[str, Any] | None,
    request: StudioTrainingExecutionRequest,
    job_id: str,
    execution_target: str,
    registry_bundle: ApplicationRegistryBundle,
) -> tuple[list[StudioManifestRef], StudioManifestRef | None, dict[str, Any]]:
    matrix_spec = matrix_spec_from_selection(stage.selection_spec)
    if matrix_spec is None:
        pending_manifest, pending_path = _write_pending_training_manifest(
            workspace=workspace,
            stage=stage,
            scenario_id=scenario_id,
            graph_spec=graph_spec,
            training_spec=training_spec,
            task_spec=task_spec,
            task_binding_spec=task_binding_spec,
            request=request,
            execution_target=execution_target,
        )
        pending_ref = _pending_training_manifest_ref(
            pending_manifest,
            pending_path,
            stage=stage,
            job_id=job_id,
        )
        return (
            [pending_ref],
            None,
            {
                "manifest_id": pending_manifest.id,
                "status": pending_manifest.status,
                "path": str(pending_path),
                "staged_at": utc_now().isoformat(),
                "run_count": 1,
                "execution_target": execution_target,
            },
        )

    try:
        materialized = materialize_sweep_matrix(
            matrix_spec,
            graph_spec=graph_spec,
            training_spec=training_spec,
            task_spec=task_spec,
            task_binding_spec=task_binding_spec,
            default_name=f"{stage.label} matrix",
            method_registry=registry_bundle.training_programs,
            row_lowerer_registry=registry_bundle.row_lowerers,
        )
    except SweepMatrixError as exc:
        raise StudioExecutionPreparationError(str(exc)) from exc

    for row in materialized.rows:
        _validate_materialized_training_row(
            row,
            component_registry=registry_bundle.components,
        )

    root_path = default_manifest_root()
    materialized_run_set = materialized.run_set_manifest
    run_set_metadata = {
        **materialized_run_set.metadata,
        "studio": {
            "workspace_id": workspace.id,
            "workspace_schema_version": workspace.schema_version,
            "stage_id": stage.id,
            "stage_kind": stage.kind,
            "scenario_id": scenario_id,
            "selection_spec": stage.selection_spec,
        },
        "planned": True,
        "staged_at": utc_now().isoformat(),
        "run_count": len(materialized.rows),
        "execution_target": execution_target,
        "execution_backend": request.backend,
    }
    run_set = materialized_run_set.model_copy(
        update={
            "status": "pending",
            "provenance": Provenance(
                entrypoint=EntrypointRef(
                    kind="feedbax-studio-pipeline",
                    name="stage_training_run_set",
                    metadata={"job_id": request.job_id},
                ),
                issues=list(request.issues),
                metadata={
                    **request.metadata,
                    "studio": {
                        "workspace_id": workspace.id,
                        "workspace_schema_version": workspace.schema_version,
                        "stage_id": stage.id,
                        "stage_kind": stage.kind,
                        "scenario_id": scenario_id,
                        "selection_spec": stage.selection_spec,
                        "run_count": len(materialized.rows),
                        "execution_target": execution_target,
                        "execution_backend": request.backend,
                    },
                },
            ),
            "metadata": run_set_metadata,
        }
    )
    run_set_path = write_manifest(run_set, root=root_path)
    run_refs: list[StudioManifestRef] = []
    run_paths: dict[str, str] = {}
    for row in materialized.rows:
        manifest, path = _write_pending_training_manifest_for_matrix_row(
            row,
            workspace=workspace,
            stage=stage,
            scenario_id=scenario_id,
            run_set_id=materialized.run_set_id,
            request=request,
            root=root_path,
            execution_target=execution_target,
        )
        run_refs.append(
            _pending_training_manifest_ref(
                manifest,
                path,
                stage=stage,
                job_id=job_id,
            )
        )
        run_paths[manifest.id] = str(path)

    run_set_ref = _studio_manifest_ref(
        run_set.kind,
        run_set.id,
        "training_run_set",
        run_set_path,
        job_id,
    )
    run_set_ref.metadata = {
        **run_set_ref.metadata,
        "status": run_set.status,
        "stage_id": stage.id,
        "scenario_id": scenario_id,
        "planned": True,
        "run_count": len(materialized.rows),
        "axes": materialized_run_set.axes.model_dump(mode="json", exclude_none=True),
        "execution_target": execution_target,
        "execution_backend": request.backend,
    }
    return (
        run_refs,
        run_set_ref,
        {
            "manifest_id": run_set.id,
            "status": run_set.status,
            "path": str(run_set_path),
            "staged_at": utc_now().isoformat(),
            "run_count": len(materialized.rows),
            "run_ids": [row.planned_run_id for row in materialized.rows],
            "run_paths": run_paths,
            "execution_target": execution_target,
        },
    )


def _write_pending_training_manifest_for_matrix_row(
    row: MaterializedMatrixRow,
    *,
    workspace: StudioWorkspaceSpec,
    stage: StudioStageSpec,
    scenario_id: str | None,
    run_set_id: str,
    request: StudioTrainingExecutionRequest,
    root: Path,
    execution_target: str,
) -> tuple[TrainingRunManifest, Path]:
    reused = _reuse_pending_training_manifest(row.planned_run_id, root)
    if reused is not None:
        return reused

    graph_spec, training_spec, task_spec, task_binding_spec, training_kind = (
        _materialized_row_specs(row)
    )
    coordinate = row.coordinate
    axis_coordinates = (
        coordinate.values
        if coordinate is not None
        else {
            "row_id": row.row_id,
            "overrides": [
                override.model_dump(mode="json", exclude_none=True) for override in row.overrides
            ],
        }
    )
    coordinate_label = coordinate.label if coordinate is not None else row.row_id
    value_indices = coordinate.value_indices if coordinate is not None else {}
    return _build_pending_training_manifest(
        manifest_id=row.planned_run_id,
        root=root,
        workspace=workspace,
        stage=stage,
        scenario_id=scenario_id,
        graph_spec=graph_spec,
        training_spec=training_spec,
        task_spec=task_spec,
        task_binding_spec=task_binding_spec,
        training_spec_kind=training_kind,
        request=request,
        execution_target=execution_target,
        axis_coordinates=axis_coordinates,
        studio_run_identity={"axis_value_indices": value_indices, "run_set_id": run_set_id},
        entrypoint_metadata={"job_id": request.job_id, "run_set_id": run_set_id},
        label=coordinate_label,
        run_set_id=run_set_id,
        overrides=row.overrides,
        total_batches=_matrix_row_n_batches(row),
    )


def _validate_materialized_training_row(
    row: MaterializedMatrixRow,
    *,
    component_registry: Any,
) -> None:
    if row.spec is not None:
        return
    graph_spec, training_spec, task_spec, task_binding_spec, _ = _materialized_row_specs(row)
    validation = _validate_training_scenario(
        graph=graph_spec,
        training_spec=training_spec,
        task_spec=task_spec,
        task_binding_spec=task_binding_spec,
        component_registry=component_registry,
    )
    if not validation.errors:
        return
    details = "; ".join(
        f"{issue.location.get('path') if issue.location else '<unknown>'}: {issue.message}"
        for issue in validation.errors
    )
    raise StudioExecutionPreparationError(
        "Expanded sweep training run is invalid: "
        f"run_id={row.planned_run_id!r}, coordinate="
        f"{row.coordinate.values if row.coordinate is not None else row.row_id!r}; "
        f"{details}"
    )


def _materialized_row_specs(
    row: MaterializedMatrixRow,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any] | None, str]:
    if row.spec is not None:
        graph_spec = row.spec.graph.inline
        if graph_spec is None:
            raise StudioExecutionPreparationError(
                "Studio staging requires inline graph payloads in governed run matrices"
            )
        return (
            dict(graph_spec),
            row.payload,
            row.spec.task.model_dump(mode="json", exclude_none=True),
            None,
            "TrainingRunSpec",
        )
    graph_spec = row.payload.get("graph_spec")
    training_spec = row.payload.get("training_spec")
    task_spec = row.payload.get("task_spec")
    task_binding_spec = row.payload.get("task_binding_spec")
    if (
        not isinstance(graph_spec, dict)
        or not isinstance(training_spec, dict)
        or not isinstance(task_spec, dict)
    ):
        raise StudioExecutionPreparationError(
            "Materialized Studio matrix row is missing its spec envelope"
        )
    return (
        graph_spec,
        training_spec,
        task_spec,
        task_binding_spec if isinstance(task_binding_spec, dict) else None,
        "TrainingSpec",
    )


def _matrix_row_n_batches(row: MaterializedMatrixRow) -> Any | None:
    if row.spec is not None:
        return row.spec.training_config.n_batches
    training_spec = row.payload.get("training_spec")
    return training_spec.get("n_batches") if isinstance(training_spec, dict) else None


def _pending_training_manifest_ref(
    pending_manifest: TrainingRunManifest,
    pending_path: Path,
    *,
    stage: StudioStageSpec,
    job_id: str,
) -> StudioManifestRef:
    pending_ref = _studio_manifest_ref(
        pending_manifest.kind,
        pending_manifest.id,
        "training_run",
        pending_path,
        job_id,
    )
    pending_ref.metadata = {
        **pending_ref.metadata,
        "status": pending_manifest.status,
        "stage_id": stage.id,
        "scenario_id": stage.scenario_id,
        "planned": True,
        "spec_hashes": _manifest_spec_hashes(pending_manifest),
        **pending_manifest.metadata,
    }
    return pending_ref


def _manifest_spec_hashes(manifest: Any) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for key in ("graph_spec", "training_spec", "task_spec", "task_binding_spec", "evaluation_spec"):
        payload = getattr(manifest, key, None)
        inline = getattr(payload, "inline", None)
        if isinstance(inline, dict):
            hashes[key] = _stable_ui_hash(inline)
    return hashes


def _ui_spec_hashes(payloads: dict[str, Any]) -> dict[str, str]:
    return {
        key: _stable_ui_hash(value) for key, value in payloads.items() if isinstance(value, dict)
    }


def _stable_ui_hash(value: Any) -> str:
    """Match Studio's synchronous FNV-1a draft hash used for stale badges."""

    text = _stable_ui_stringify(value)
    hash_value = 2166136261
    for character in text:
        hash_value ^= ord(character)
        hash_value = (hash_value * 16777619) & 0xFFFFFFFF
    return f"fnv1a:{hash_value:08x}"


def _stable_ui_stringify(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return json.dumps(value, separators=(",", ":"), allow_nan=False)
    if isinstance(value, str):
        return json.dumps(value, separators=(",", ":"))
    if isinstance(value, list):
        return "[" + ",".join(_stable_ui_stringify(item) for item in value) + "]"
    if isinstance(value, dict):
        return (
            "{"
            + ",".join(
                f"{json.dumps(key, separators=(',', ':'))}:{_stable_ui_stringify(value[key])}"
                for key in sorted(value)
            )
            + "}"
        )
    return json.dumps(value, separators=(",", ":"), sort_keys=True, allow_nan=False)


def _training_seed(training_spec: dict[str, Any]) -> Any | None:
    if "seed" in training_spec:
        return training_spec["seed"]
    params = training_spec.get("params")
    if isinstance(params, dict) and "seed" in params:
        return params["seed"]
    return None


def _request_execution_target(
    stage: StudioStageSpec,
    request: StudioTrainingExecutionRequest,
) -> ExecutionTarget:
    if request.queue_target is not None:
        if not request.queue_manifest_ids:
            raise StudioExecutionPreparationError(
                "Queue launch preparation requires queue_manifest_ids with queue_target"
            )
        return request.queue_target
    return _stage_execution_target(stage, request.backend)


def _queue_training_manifest_subset(
    *,
    stage: StudioStageSpec,
    request: StudioTrainingExecutionRequest,
    execution_target: ExecutionTarget,
) -> tuple[list[StudioManifestRef], None, dict[str, Any]]:
    selected_ids = list(dict.fromkeys(request.queue_manifest_ids))
    refs_by_id = _stage_manifest_refs_by_id(stage)
    missing_ids = [manifest_id for manifest_id in selected_ids if manifest_id not in refs_by_id]
    if missing_ids:
        raise StudioExecutionPreparationError(
            "Queue launch references manifest IDs that are not present on the selected train "
            f"stage: {', '.join(missing_ids)}"
        )

    selected_refs = [refs_by_id[manifest_id] for manifest_id in selected_ids]
    for ref in selected_refs:
        if not _is_training_manifest_ref(ref):
            raise StudioExecutionPreparationError(
                f"Queue launch manifest {ref.id!r} is not a training run"
            )
        ref_target = _manifest_ref_execution_target(ref, stage)
        if ref_target != execution_target:
            raise StudioExecutionPreparationError(
                f"Queue launch manifest {ref.id!r} targets {ref_target!r}, "
                f"not selected target {execution_target!r}"
            )

    now = utc_now().isoformat()
    return (
        selected_refs,
        None,
        {
            "manifest_ids": selected_ids,
            "status": "pending",
            "prepared_at": now,
            "run_count": sum(_manifest_ref_run_count(ref) for ref in selected_refs),
            "execution_target": execution_target,
            "source": "queue_manifest_subset",
        },
    )


def _stage_manifest_refs_by_id(stage: StudioStageSpec) -> dict[str, StudioManifestRef]:
    refs_by_id: dict[str, StudioManifestRef] = {}
    for ref in stage.manifest_refs:
        refs_by_id[ref.id] = ref
    for collection in stage.output_collections:
        for ref in collection.item_refs:
            refs_by_id[ref.id] = ref
    return refs_by_id


def _is_training_manifest_ref(ref: StudioManifestRef) -> bool:
    return ref.role == "training_run" or ref.kind in {"TrainingRun", "TrainingRunManifest"}


def _manifest_ref_execution_target(
    ref: StudioManifestRef,
    stage: StudioStageSpec,
) -> ExecutionTarget:
    for key in ("execution_target", "compute_target", "target"):
        value = ref.metadata.get(key)
        if not isinstance(value, str):
            continue
        if value == "managed":
            return "gcp"
        if value in {"local", "gcp", "runpod", "manual"}:
            return value
    return _stage_execution_target(stage, "local")


def _manifest_ref_run_count(ref: StudioManifestRef) -> int:
    value = ref.metadata.get("run_count")
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, float) and value > 0 and value.is_integer():
        return int(value)
    return 1


def _stage_execution_target(stage: StudioStageSpec, backend: str) -> ExecutionTarget:
    realization = stage.metadata.get("backend_realization")
    execution_target = (
        realization.get("execution_target") if isinstance(realization, dict) else None
    )
    if execution_target in {"local", "gcp", "runpod", "manual"}:
        return str(execution_target)
    if backend == "runpod":
        return "runpod"
    return "local"


def _stage_axis_coordinates(stage: StudioStageSpec) -> dict[str, Any]:
    for key in ("axis_coordinates", "sweep_coordinates", "axis_values"):
        value = stage.selection_spec.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _request_root(root: str | None) -> Path:
    return Path(root).expanduser() if root else default_manifest_root()


def _evaluation_matrix_plan(request: StudioEvaluationMatrixRequest) -> dict[str, Any]:
    workspace = request.workspace.model_copy(deep=True)
    eval_stage = _select_stage_by_kind(workspace, "eval")
    if request.stage_id is not None and eval_stage.id != request.stage_id:
        eval_stage = _select_stage(workspace, request.stage_id)
        if eval_stage.kind != "eval":
            raise StudioExecutionPreparationError(
                f"Stage {eval_stage.id!r} is {eval_stage.kind!r}, not 'eval'"
            )
    training_refs = _selected_training_refs(request, workspace, eval_stage)
    conditions = _expand_eval_conditions(
        request.condition_matrix or _stage_condition_matrix(eval_stage),
        base_params={**_stage_eval_params(eval_stage), **request.eval_params},
    )
    job_id = request.job_id or f"studio-eval-{uuid.uuid4().hex[:12]}"
    checkpoint_policy = request.checkpoint_policy.model_dump(mode="json", exclude_none=True)
    items: list[dict[str, Any]] = []
    for training_ref in training_refs:
        checkpoint_spec = _checkpoint_selection_spec(training_ref, checkpoint_policy)
        checkpoint_id = checkpoint_selection_manifest_id(checkpoint_spec)
        checkpoint_ref = ParentRef(
            kind="CheckpointSelectionManifest",
            id=checkpoint_id,
            role="checkpoint_selection",
            metadata={"checkpoint_policy": checkpoint_policy},
        )
        for condition in conditions:
            spec = _evaluation_spec_for_condition(
                training_ref,
                checkpoint_ref=checkpoint_ref,
                condition=condition,
                checkpoint_policy=checkpoint_policy,
                workspace=workspace,
                stage=eval_stage,
            )
            items.append(
                {
                    "workspace": workspace,
                    "stage": eval_stage,
                    "job_id": job_id,
                    "training_ref": training_ref,
                    "checkpoint_spec": checkpoint_spec,
                    "checkpoint_id": checkpoint_id,
                    "checkpoint_policy": checkpoint_policy,
                    "checkpoint_ref": checkpoint_ref,
                    "condition": condition,
                    "evaluation_spec": spec,
                    "evaluation_id": evaluation_run_manifest_id(spec),
                }
            )
    return {
        "workspace": workspace,
        "stage": eval_stage,
        "job_id": job_id,
        "selection_spec": _selection_spec_payload(request, eval_stage),
        "training_stage_refs": [_stage_ref_for_parent(ref) for ref in training_refs],
        "conditions": conditions,
        "checkpoint_policy": checkpoint_policy,
        "items": items,
        "reprocess": request.reprocess,
        "root": _request_root(request.root),
    }


def _evaluation_preview_from_plan(plan: dict[str, Any]) -> StudioEvaluationMatrixPreview:
    root = plan["root"]
    materialized = pending = failed = launch = new_count = 0
    eval_ids: list[str] = []
    checkpoint_ids: list[str] = []
    for item in plan["items"]:
        eval_id = item["evaluation_id"]
        checkpoint_ids.append(item["checkpoint_id"])
        eval_ids.append(eval_id)
        manifest = _existing_manifest(eval_id, root=root)
        status = manifest.status if isinstance(manifest, EvaluationRunManifest) else None
        if status == "completed" or evaluation_states_cache_path(eval_id, root=root).exists():
            materialized += 1
        elif status in {"pending", "running"}:
            pending += 1
        elif status == "failed":
            failed += 1
        if manifest is None:
            new_count += 1
        if _should_launch_status(status, plan["reprocess"]):
            launch += 1
    total = len(plan["items"])
    selected_count = len(plan["training_stage_refs"])
    condition_count = len(plan["conditions"])
    summary = (
        f"{selected_count} runs x {condition_count} conditions x 1 checkpoint policy = "
        f"{total} evals - {materialized} already materialized"
    )
    return StudioEvaluationMatrixPreview(
        workspace=plan["workspace"],
        stage_id=plan["stage"].id,
        selected_training_run_count=selected_count,
        condition_count=condition_count,
        checkpoint_policy_count=1,
        total_eval_count=total,
        materialized_count=materialized,
        pending_count=pending,
        failed_count=failed,
        new_manifest_count=new_count,
        launch_count=launch,
        evaluation_run_ids=eval_ids,
        checkpoint_selection_ids=sorted(set(checkpoint_ids)),
        summary=summary,
    )


def _selected_training_refs(
    request: StudioEvaluationMatrixRequest,
    workspace: StudioWorkspaceSpec,
    eval_stage: StudioStageSpec,
) -> list[ParentRef]:
    refs_by_id = _available_training_refs(workspace, eval_stage)
    ids = list(request.training_run_ids)
    selection_spec = request.selection_spec or _selection_spec_from_eval_stage(eval_stage)
    if selection_spec is not None:
        rows = manifest_index_rows_from_records(
            iter_indexed_manifest_records_by_kind(selection_spec.manifest_kind)
        )
        preview = preview_selection_spec(selection_spec, rows, limit=None)
        ids = [ref.id for ref in preview.parent_refs]
    if not ids:
        ids = [
            item
            for item in eval_stage.selection_spec.get("training_run_ids", [])
            if isinstance(item, str)
        ]
    refs = [refs_by_id.get(id_) or _indexed_training_parent_ref(id_) for id_ in ids]
    refs = [ref for ref in refs if ref is not None]
    if not refs:
        raise StudioExecutionPreparationError(
            "Evaluation staging requires at least one training run"
        )
    return refs


def _available_training_refs(
    workspace: StudioWorkspaceSpec,
    eval_stage: StudioStageSpec,
) -> dict[str, ParentRef]:
    refs: dict[str, ParentRef] = {}
    for collection in eval_stage.input_collections:
        if collection.kind == "training_runs":
            refs.update({ref.id: _parent_ref_from_stage_ref(ref) for ref in collection.item_refs})
    train_stage = next((stage for stage in workspace.stages if stage.kind == "train"), None)
    if train_stage is not None:
        for collection in train_stage.output_collections:
            if collection.kind == "training_runs":
                refs.update(
                    {ref.id: _parent_ref_from_stage_ref(ref) for ref in collection.item_refs}
                )
    return refs


def _indexed_training_parent_ref(run_id: str) -> ParentRef | None:
    row = next(
        (
            item
            for item in iter_indexed_manifest_records_by_kind("TrainingRunManifest")
            if item["id"] == run_id
        ),
        None,
    )
    if row is None:
        return None
    return ParentRef(
        kind="TrainingRunManifest",
        id=run_id,
        role="training_run",
        uri=str(row["path"]),
        metadata={"status": row.get("status")},
    )


def _parent_ref_from_stage_ref(ref: StudioManifestRef) -> ParentRef:
    return ParentRef(
        kind=ref.kind,
        id=ref.id,
        role=ref.role,
        uri=ref.uri,
        metadata={**ref.metadata, "provider": ref.provider},
    )


def _stage_ref_for_parent(ref: ParentRef) -> StudioManifestRef:
    return StudioManifestRef(
        kind=ref.kind,
        id=ref.id,
        role=ref.role or "training_run",
        uri=ref.uri,
        metadata=ref.metadata,
    )


def _selection_spec_from_eval_stage(eval_stage: StudioStageSpec) -> SelectionSpec | None:
    raw = eval_stage.selection_spec.get("selection_spec") or eval_stage.selection_spec.get(
        "training_selection"
    )
    if isinstance(raw, dict):
        return SelectionSpec.model_validate(raw)
    ids = [
        item
        for item in eval_stage.selection_spec.get("training_run_ids", [])
        if isinstance(item, str)
    ]
    if ids:
        return SelectionSpec(mode="explicit", manifest_kind="TrainingRunManifest", ids=ids)
    return None


def _selection_spec_payload(
    request: StudioEvaluationMatrixRequest,
    eval_stage: StudioStageSpec,
) -> dict[str, Any] | None:
    spec = request.selection_spec or _selection_spec_from_eval_stage(eval_stage)
    return spec.model_dump(mode="json", exclude_none=True) if spec is not None else None


def _stage_eval_params(eval_stage: StudioStageSpec) -> dict[str, Any]:
    execution_spec = (
        eval_stage.execution_spec if isinstance(eval_stage.execution_spec, dict) else {}
    )
    raw = execution_spec.get("eval_params") or eval_stage.selection_spec.get("eval_params")
    return dict(raw) if isinstance(raw, dict) else {}


def _stage_condition_matrix(eval_stage: StudioStageSpec) -> dict[str, Any]:
    execution_spec = (
        eval_stage.execution_spec if isinstance(eval_stage.execution_spec, dict) else {}
    )
    raw = (
        eval_stage.selection_spec.get("condition_matrix")
        or eval_stage.selection_spec.get("matrix")
        or execution_spec.get("condition_matrix")
    )
    return dict(raw) if isinstance(raw, dict) else {}


def _expand_eval_conditions(
    matrix_spec: dict[str, Any],
    *,
    base_params: dict[str, Any],
) -> list[dict[str, Any]]:
    if not matrix_spec:
        return [
            {
                "index": 0,
                "label": str(base_params.get("label") or "default"),
                "params": copy.deepcopy(base_params),
                "axis_coordinates": {},
                "axis_value_indices": {},
            }
        ]
    try:
        axes = _parse_axes(matrix_spec)
        combination = _parse_combination(matrix_spec)
        _validate_group_axes(axes, combination)
        indexed_coordinates = _expand_coordinates(axes, combination)
    except SweepMatrixError as exc:
        raise StudioExecutionPreparationError(str(exc)) from exc
    axes_with_values = [
        axis.model_copy(update={"values": _variation_values(axis.variation)}) for axis in axes
    ]
    axis_by_id = {axis.id: axis for axis in axes_with_values}
    conditions: list[dict[str, Any]] = []
    for index, value_indices in enumerate(indexed_coordinates):
        params = copy.deepcopy(base_params)
        values = {
            axis_id: axis_by_id[axis_id].values[value_index]
            for axis_id, value_index in value_indices.items()
        }
        for axis_id, value in values.items():
            _set_eval_axis_value(params, axis_by_id[axis_id].path, value)
        conditions.append(
            {
                "index": index,
                "label": _coordinate_label(axis_by_id, values) or f"condition {index + 1}",
                "params": params,
                "axis_coordinates": values,
                "axis_value_indices": value_indices,
            }
        )
    return conditions


def _set_eval_axis_value(params: dict[str, Any], path: str, value: Any) -> None:
    if path.startswith("/"):
        parts = [part for part in path.split("/") if part]
    else:
        parts = [part for part in path.split(".") if part]
    if parts and parts[0] in {"eval_params", "params"}:
        parts = parts[1:]
    if not parts:
        raise StudioExecutionPreparationError("eval condition axis path must name a parameter")
    current: Any = params
    for part in parts[:-1]:
        if not isinstance(current, dict):
            raise StudioExecutionPreparationError(
                f"eval condition axis path {path!r} cannot traverse {part!r}"
            )
        current = current.setdefault(part, {})
    if not isinstance(current, dict):
        raise StudioExecutionPreparationError(
            f"eval condition axis path {path!r} cannot set {parts[-1]!r}"
        )
    current[parts[-1]] = value


def _checkpoint_selection_spec(
    training_ref: ParentRef,
    checkpoint_policy: dict[str, Any],
) -> CheckpointSelectionSpec:
    mode = str(checkpoint_policy.get("mode", "last"))
    return CheckpointSelectionSpec(
        selection_type="feedbax.studio.checkpoint_policy",
        scorer=CheckpointScorerIdentity(
            scorer_id=f"feedbax.studio.checkpoint_policy.{mode}",
            name=mode,
            parameters=checkpoint_policy,
        ),
        bank=CheckpointSelectionBank(
            role="fixed",
            status="available",
            ref=training_ref,
            logical_name=f"{training_ref.id}:{mode}",
            metadata={"checkpoint_policy": checkpoint_policy},
        ),
        group_by="run",
        inputs=[training_ref],
        params={"checkpoint_policy": checkpoint_policy},
        metadata={"training_run_id": training_ref.id},
    )


def _write_checkpoint_selection_manifest(
    item: dict[str, Any],
    *,
    root: Path,
) -> tuple[CheckpointSelectionManifest, Path]:
    spec = item["checkpoint_spec"]
    manifest_id = item["checkpoint_id"]
    existing = _existing_manifest(manifest_id, root=root)
    if isinstance(existing, CheckpointSelectionManifest):
        return existing, _manifest_path(manifest_id, root=root)
    training_ref = item["training_ref"]
    checkpoint_ref = ParentRef(
        kind="CheckpointPolicy",
        id=f"{training_ref.id}:{item['checkpoint_policy'].get('mode', 'last')}",
        role="checkpoint_policy",
        metadata=item["checkpoint_policy"],
    )
    candidate = CheckpointCandidateRef(
        id=checkpoint_ref.id,
        checkpoint=checkpoint_ref,
        run_id=training_ref.id,
        training_run=training_ref,
        metadata={"checkpoint_policy": item["checkpoint_policy"]},
    )
    manifest = CheckpointSelectionManifest(
        id=manifest_id,
        status="pending",
        selection_spec=spec_payload(
            "CheckpointSelectionSpec",
            spec.model_dump(mode="json", exclude_none=True),
        ),
        scorer=spec.scorer,
        bank=spec.bank,
        selection_status="selected",
        inputs=[training_ref],
        selections=[
            CheckpointSelectionGroup(
                scope="run",
                run_id=training_ref.id,
                candidate_checkpoints=[candidate],
                selected_checkpoint=candidate,
                metadata={"checkpoint_policy": item["checkpoint_policy"]},
            )
        ],
        provenance=Provenance(
            entrypoint=EntrypointRef(
                kind="feedbax-studio-pipeline",
                name="stage_checkpoint_selection",
                metadata={"job_id": item["job_id"]},
            ),
            parents=[training_ref],
            metadata={"checkpoint_policy": item["checkpoint_policy"]},
        ),
        metadata={
            "planned": True,
            "checkpoint_policy": item["checkpoint_policy"],
            "selected_training_run_id": training_ref.id,
        },
    )
    return manifest, write_manifest(manifest, root=root)


def _evaluation_spec_for_condition(
    training_ref: ParentRef,
    *,
    checkpoint_ref: ParentRef,
    condition: dict[str, Any],
    checkpoint_policy: dict[str, Any],
    workspace: StudioWorkspaceSpec,
    stage: StudioStageSpec,
) -> EvaluationRunSpec:
    return EvaluationRunSpec(
        evaluation_type="feedbax.studio.default_eval",
        training_run_ids=[training_ref.id],
        inputs=[training_ref, checkpoint_ref],
        params={
            **condition["params"],
            "stage_id": stage.id,
            "scenario_id": stage.scenario_id,
            "selection_spec": stage.selection_spec,
            "condition_axis_coordinates": condition["axis_coordinates"],
            "condition_axis_value_indices": condition["axis_value_indices"],
            "condition_label": condition["label"],
            "checkpoint_policy": checkpoint_policy,
            "checkpoint_selection_manifest_id": checkpoint_ref.id,
            "workspace_id": workspace.id,
        },
    )


def _write_pending_evaluation_manifest(
    item: dict[str, Any],
    *,
    checkpoint_ref: StudioManifestRef,
    request: StudioEvaluationMatrixRequest,
    root: Path,
) -> tuple[EvaluationRunManifest, Path]:
    manifest_id = item["evaluation_id"]
    existing = _existing_manifest(manifest_id, root=root)
    if isinstance(existing, EvaluationRunManifest):
        if _should_restaging_overwrite(existing.status, request.reprocess):
            restaged = existing.model_copy(
                update={
                    "status": "pending",
                    "artifacts": [],
                    "summary_metrics": {"input_training_runs": len(existing.input_training_runs)},
                    "metadata": _restaged_metadata(
                        existing.metadata,
                        manifest_id=manifest_id,
                        restaged_from_status=existing.status,
                    ),
                }
            )
            return restaged, write_manifest(restaged, root=root)
        return existing, _manifest_path(manifest_id, root=root)

    training_ref = item["training_ref"]
    condition = item["condition"]
    label = _evaluation_condition_label(training_ref, condition)
    manifest = EvaluationRunManifest(
        id=manifest_id,
        status="pending",
        evaluation_spec=spec_payload(
            "EvaluationRunSpec",
            item["evaluation_spec"].model_dump(mode="json", exclude_none=True),
        ),
        input_training_runs=[training_ref],
        summary_metrics={"input_training_runs": 1},
        provenance=Provenance(
            entrypoint=EntrypointRef(
                kind="feedbax-studio-pipeline",
                name="stage_evaluation_run",
                metadata={"job_id": item["job_id"]},
            ),
            issues=list(request.issues),
            parents=list(item["evaluation_spec"].inputs),
            metadata={
                **request.metadata,
                "checkpoint_policy": item["checkpoint_policy"],
                "condition_axis_coordinates": condition["axis_coordinates"],
            },
        ),
        metadata={
            "name": label,
            "label": label,
            "planned": True,
            "staged_at": utc_now().isoformat(),
            "selected_training_run_id": training_ref.id,
            "training_run_ids": [training_ref.id],
            "eval_protocol": _eval_protocol_metadata(condition),
            "checkpoint_policy": item["checkpoint_policy"],
            "checkpoint_selection_manifest_id": checkpoint_ref.id,
            "condition_axis_coordinates": condition["axis_coordinates"],
            "spec_hashes": _ui_spec_hashes(
                {
                    "evaluation_spec": item["evaluation_spec"].model_dump(
                        mode="json",
                        exclude_none=True,
                    )
                }
            ),
            "studio": {
                "workspace_id": item["workspace"].id,
                "workspace_schema_version": item["workspace"].schema_version,
                "stage_id": item["stage"].id,
                "stage_kind": item["stage"].kind,
                "scenario_id": item["stage"].scenario_id,
                "selection_spec": item["stage"].selection_spec,
            },
        },
    )
    return manifest, write_manifest(manifest, root=root)


def _pending_evaluation_manifest_ref(
    manifest: EvaluationRunManifest,
    path: Path,
    *,
    stage: StudioStageSpec,
    job_id: str,
) -> StudioManifestRef:
    ref = _studio_manifest_ref(
        manifest.kind,
        manifest.id,
        "evaluation_run",
        path,
        str(job_id),
        manifest=manifest,
    )
    ref.metadata = {
        **ref.metadata,
        "status": manifest.status,
        "stage_id": stage.id,
        "scenario_id": stage.scenario_id,
        "planned": manifest.status == "pending",
        **manifest.metadata,
    }
    return ref


def _evaluation_condition_label(training_ref: ParentRef, condition: dict[str, Any]) -> str:
    label = condition["label"]
    return f"{label} on {training_ref.id}" if label != "default" else f"Eval {training_ref.id}"


def _eval_protocol_metadata(condition: dict[str, Any]) -> dict[str, Any]:
    params = condition["params"]
    return {
        "targets": params.get("targets") or params.get("target_set") or "default",
        "sisu": params.get("sisu") or params.get("sisu_value"),
        "perturbation": params.get("perturbation") or params.get("perturbation_type"),
    }


def _should_launch_status(status: str | None, reprocess: EvalReprocessMode) -> bool:
    if reprocess == "stale":
        return status == "stale"
    if status == "stale":
        return reprocess == "all"
    if status is None:
        return True
    if status == "pending":
        return True
    if status == "failed" and reprocess in {"missing_failed", "all"}:
        return True
    if status == "completed" and reprocess == "all":
        return True
    return status == "cancelled" and reprocess == "all"


def _should_restaging_overwrite(status: str | None, reprocess: EvalReprocessMode) -> bool:
    return status in {"failed", "cancelled", "completed", "stale"} and _should_launch_status(
        status,
        reprocess,
    )


def _existing_manifest(manifest_id: str, *, root: Path) -> object | None:
    paths = find_manifest_paths_by_id(manifest_id, root=root)
    if not paths:
        return None
    return load_manifest(paths[0])


def _manifest_path(manifest_id: str, *, root: Path) -> Path:
    paths = find_manifest_paths_by_id(manifest_id, root=root)
    if not paths:
        raise StudioExecutionPreparationError(f"Manifest {manifest_id!r} is not indexed")
    return paths[0]


def _materialize_eval_stage(
    workspace: StudioWorkspaceSpec,
    *,
    root_path: Path,
    job_id: str,
    issues: list[str],
    request_metadata: dict[str, Any],
    registry_bundle: ApplicationRegistryBundle,
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
    analysis_stage = _select_stage_by_kind(workspace, "analysis")
    analysis_scenario = workspace.scenarios.get(analysis_stage.scenario_id or "")
    downstream_states_policy = (
        (analysis_scenario.analysis_spec or {}).get(
            "evaluation_states_policy",
            "recompute",
        )
        if analysis_scenario is not None
        else "recompute"
    )
    spec = EvaluationRunSpec(
        evaluation_type="feedbax.studio.default_eval",
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
            **(
                {"states_custody": "durable"}
                if downstream_states_policy == "require_durable"
                else {}
            ),
        },
    )
    manifest, path = execute_evaluation_run_spec(
        spec,
        root=root_path,
        provenance=_stage_provenance(
            stage_kind="eval",
            issues=issues,
            parents=input_refs,
            request_metadata=request_metadata,
            job_id=job_id,
        ),
        metadata={"studio": _stage_manifest_metadata(workspace, eval_stage, job_id)},
        registry=registry_bundle.evaluation_recipes,
    )
    manifest_ref = _studio_manifest_ref(
        manifest.kind,
        manifest.id,
        "evaluation_run",
        path,
        job_id,
        manifest=manifest,
    )
    artifact_refs = [
        _studio_artifact_ref(artifact, kind="EvaluationResult") for artifact in manifest.artifacts
    ]
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
    registry_bundle: ApplicationRegistryBundle,
) -> tuple[Path, list[StudioArtifactRef]]:
    eval_stage = _select_stage_by_kind(workspace, "eval")
    analysis_stage = _select_stage_by_kind(workspace, "analysis")
    evaluation_collection = _require_output_collection(eval_stage, "evaluation_runs")
    _require_collection_items(evaluation_collection, "evaluation runs", analysis_stage.id)
    analysis_stage.input_collections = _upsert_collection_ref(
        analysis_stage.input_collections,
        evaluation_collection,
    )
    input_refs = _collection_manifest_parents(
        evaluation_collection,
        authenticated_kind="EvaluationRunManifest",
    )
    scenario = workspace.scenarios.get(analysis_stage.scenario_id or "")
    analysis_spec_payload = scenario.analysis_spec if scenario is not None else None
    analysis_type = (analysis_spec_payload or {}).get("analysis_type")
    if not analysis_type:
        raise StudioExecutionPreparationError(
            f"Analysis stage {analysis_stage.id!r} requires scenario "
            f"{analysis_stage.scenario_id!r} to declare analysis_spec.analysis_type"
        )
    spec = AnalysisRunSpec(
        analysis_type=str(analysis_type),
        inputs=input_refs,
        input_requirements=list((analysis_spec_payload or {}).get("input_requirements", [])),
        evaluation_states_policy=(analysis_spec_payload or {}).get(
            "evaluation_states_policy", "recompute"
        ),
        params={
            "stage_id": analysis_stage.id,
            "scenario_id": analysis_stage.scenario_id,
            "selection_spec": analysis_stage.selection_spec,
            "input_collection_id": evaluation_collection.id,
            "analysis_spec": analysis_spec_payload or {},
            **_analysis_requested_outputs_params(analysis_spec_payload),
        },
    )
    manifest, path = execute_analysis_run_spec(
        spec,
        root=root_path,
        provenance=_stage_provenance(
            stage_kind="analysis",
            issues=issues,
            parents=input_refs,
            request_metadata=request_metadata,
            job_id=job_id,
        ),
        metadata={
            "studio": _stage_manifest_metadata(workspace, analysis_stage, job_id),
            "input_evaluation_runs": [ref.id for ref in input_refs],
        },
        fig_dump_formats=("json",),
        registry=registry_bundle.analysis_recipes,
        evaluation_registry=registry_bundle.evaluation_recipes,
        experiment_registry=registry_bundle.experiment_packages,
    )
    manifest_ref = _studio_manifest_ref(
        manifest.kind,
        manifest.id,
        "analysis_run",
        path,
        job_id,
        manifest=manifest,
    )
    artifact_refs = [
        _studio_artifact_ref(artifact, kind="AnalysisArtifact") for artifact in manifest.artifacts
    ]
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


def _analysis_requested_outputs_params(
    analysis_spec_payload: dict[str, Any] | None,
) -> dict[str, list[str]]:
    if not analysis_spec_payload:
        return {}
    requested = analysis_spec_payload.get("requested_outputs", analysis_spec_payload.get("outputs"))
    if requested is None:
        return {}
    if not isinstance(requested, list):
        raise StudioExecutionPreparationError(
            "Analysis scenario requested_outputs/outputs must be a list"
        )
    return {"requested_outputs": [str(item) for item in requested]}


def _materialize_report_stage(
    workspace: StudioWorkspaceSpec,
    *,
    root_path: Path,
    job_id: str,
    issues: list[str],
    request_metadata: dict[str, Any],
    registry_bundle: ApplicationRegistryBundle,
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
        report_type=str((report_spec_payload or {}).get("report_type", STUDIO_REPORT_TYPE)),
        inputs=input_refs,
        params={
            "stage_id": report_stage.id,
            "scenario_id": report_stage.scenario_id,
            "selection_spec": report_stage.selection_spec,
            "input_collection_id": analysis_collection.id,
            "report_spec": report_spec_payload or {},
            "studio": {
                "job_id": job_id,
                "stage_id": report_stage.id,
                "title": workspace.label,
            },
        },
        narrative="MVP report stub assembled from selected Studio analysis products.",
    )
    manifest, path = execute_report_spec(
        spec,
        root=root_path,
        provenance=_stage_provenance(
            stage_kind="report",
            issues=issues,
            parents=input_refs,
            request_metadata=request_metadata,
            job_id=job_id,
        ),
        metadata={"studio": _stage_manifest_metadata(workspace, report_stage, job_id)},
        registry=registry_bundle.report_recipes,
    )
    manifest_ref = _studio_manifest_ref(
        manifest.kind,
        manifest.id,
        "report",
        path,
        job_id,
        manifest=manifest,
    )
    artifact_refs = [
        _studio_artifact_ref(artifact, kind="ReportArtifact") for artifact in manifest.artifacts
    ]
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


def _select_stage(
    workspace: StudioWorkspaceSpec,
    stage_id: str,
) -> StudioStageSpec:
    stage = next((item for item in workspace.stages if item.id == stage_id), None)
    if stage is None:
        raise StudioExecutionPreparationError(f"Workspace has no stage {stage_id!r}")
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


def _collection_manifest_parents(
    collection: StudioCollectionRef,
    *,
    authenticated_kind: str | None = None,
) -> list[ParentRef]:
    parents: list[ParentRef] = []
    for ref in collection.item_refs:
        parent = ParentRef(
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
        if ref.kind == authenticated_kind:
            if not ref.uri:
                raise StudioExecutionPreparationError(
                    f"Studio manifest ref {ref.id!r} has no exact runtime path to authenticate"
                )
            manifest = load_manifest(ref.uri)
            if manifest.kind != ref.kind or manifest.id != ref.id:
                raise StudioExecutionPreparationError(
                    f"Studio manifest ref {ref.id!r} disagrees with its exact runtime bytes"
                )
            authority = authenticated_manifest_ref(manifest, ref.uri, ref.role)
            parent = authority.model_copy(
                update={"metadata": {**authority.metadata, **parent.metadata}}
            )
        parents.append(parent)
    return parents


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
    *,
    manifest: Any | None = None,
) -> StudioManifestRef:
    metadata: dict[str, Any] = {"job_id": job_id}
    if manifest is not None:
        metadata.update(_manifest_parent_ref_metadata(manifest))
    return StudioManifestRef(
        kind=kind,
        id=manifest_id,
        role=role,
        uri=str(path),
        metadata=metadata,
    )


def _manifest_parent_ref_metadata(manifest: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    parents = getattr(getattr(manifest, "provenance", None), "parents", [])
    if parents:
        metadata["parent_refs"] = [
            parent.model_dump(mode="json", exclude_none=True) for parent in parents
        ]
    for field in ("inputs", "input_training_runs"):
        refs = getattr(manifest, field, [])
        if refs:
            metadata[field] = [ref.model_dump(mode="json", exclude_none=True) for ref in refs]
    return metadata


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
            "materialized_by": "feedbax.studio.execution",
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


def _upsert_many_manifest_refs(
    refs: list[StudioManifestRef],
    new_refs: list[StudioManifestRef],
) -> list[StudioManifestRef]:
    merged = refs
    for ref in new_refs:
        merged = _upsert_manifest_ref(merged, ref)
    return merged


def _upsert_manifest_ref(
    refs: list[StudioManifestRef],
    ref: StudioManifestRef,
) -> list[StudioManifestRef]:
    return [item for item in refs if not (item.kind == ref.kind and item.id == ref.id)] + [ref]
