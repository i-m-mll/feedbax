from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from feedbax.bin.studio_pipeline import main as studio_pipeline_main
from feedbax.studio.execution import (
    STUDIO_TRAINING_CONTRACT_FILES,
    StudioPipelineMaterializationRequest,
    StudioExecutionPreparationError,
    StudioEvaluationCheckpointPolicy,
    StudioEvaluationMatrixRequest,
    StudioTrainingLocalRunRequest,
    StudioTrainingExecutionRequest,
    _build_execution_spec,
    _build_pending_training_manifest,
    _write_pending_training_manifest,
    _write_pending_training_manifest_for_matrix_row,
    materialize_studio_pipeline,
    prepare_studio_training_execution,
    preview_studio_evaluation_matrix,
    run_studio_evaluation_local_execution,
    run_studio_training_local_execution,
    stage_studio_evaluation_matrix,
)
from feedbax.analysis.evaluation import (
    EvaluationRecipeResult,
)
from feedbax.analysis.specs import (
    AnalysisRecipeResult,
)
from feedbax.plugins.application import new_application_registry_bundle
from feedbax.plugins.bootstrap import BootstrapState
from feedbax.contracts.manifest import (
    CheckpointSelectionManifest,
    EvaluationRunManifest,
    EvaluationRunSpec,
    TrainingRunAxisCoordinate,
    TrainingRunManifest,
    TrainingRunSetManifest,
    load_manifest,
    store_json_artifact,
    utc_now,
    write_manifest,
)
from feedbax.contracts.run_matrix import TrainingRowProvenance
from feedbax.training.run_matrix import MaterializedMatrixRow
from tests.analysis_fixtures import ToyAnalysis, build_toy_analysis_data
from feedbax.web.app import create_app
from feedbax.contracts.graph import (
    GraphMetadata,
    GraphSpec,
    StudioCollectionRef,
    StudioManifestRef,
    StudioStageSpec,
    StudioTaskBindingSpec,
    build_default_studio_workspace,
)


@pytest.fixture(autouse=True)
def _isolated_manifest_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep every Studio execution test's manifests in its unique temp directory."""
    monkeypatch.setenv("FEEDBAX_RUNS_DIR", str(tmp_path))


def _graph() -> GraphSpec:
    return GraphSpec(
        nodes={
            "network": {
                "type": "Gain",
                "params": {"gain": 1.0},
                "input_ports": ["input"],
                "output_ports": ["output"],
            }
        },
        metadata=GraphMetadata(
            name="Studio execution smoke",
            created_at="2026-05-18T00:00:00+00:00",
            updated_at="2026-05-18T00:00:00+00:00",
        ),
    )


def _workspace():
    workspace = build_default_studio_workspace(label="Studio execution")
    train_stage = next(stage for stage in workspace.stages if stage.kind == "train")
    scenario = workspace.scenarios[train_stage.scenario_id]
    scenario.training_spec = {
        "optimizer": {"type": "adam", "params": {"learning_rate": 0.001}},
        "loss": {
            "type": "Composite",
            "label": "loss",
            "weight": 1.0,
            "selector": "port:network.output",
        },
        "n_batches": 25,
        "batch_size": 8,
    }
    scenario.task_spec = {
        "type": "ReachingTask",
        "params": {"n_targets": 4, "target_radius": 0.02},
    }
    scenario.task_binding_spec = StudioTaskBindingSpec.model_validate(
        {
            "schema_version": "feedbax.spec.studio.task_bindings.v2",
            "exposed_data": [
                {
                    "id": "inputs",
                    "label": "Inputs",
                    "kind": "signal",
                    "path": "inputs",
                    "bindable": True,
                    "metadata": {},
                },
                {
                    "id": "targets",
                    "label": "Targets",
                    "kind": "target",
                    "path": "targets",
                    "bindable": False,
                    "metadata": {},
                },
            ],
            "bindings": [
                {
                    "id": "task:inputs->network:input",
                    "source_data_id": "inputs",
                    "target_node_id": "network",
                    "target_port": "input",
                    "role": "model_input",
                    "metadata": {},
                }
            ],
            "metadata": {},
        }
    )
    workspace.stages.append(
        StudioStageSpec(
            id="stage:future-report-packaging",
            kind="protocol",
            label="Future report packaging",
            metadata={"later_product_surface": {"keep": True}},
        )
    )
    return workspace


def _workspace_with_analysis_type(analysis_type: str):
    workspace = _workspace()
    analysis_stage = next(stage for stage in workspace.stages if stage.kind == "analysis")
    scenario = workspace.scenarios[analysis_stage.scenario_id]
    scenario.analysis_spec = {
        **(scenario.analysis_spec or {}),
        "analysis_type": analysis_type,
    }
    return workspace


@pytest.fixture
def registry_bundle():
    return new_application_registry_bundle(local_component_source=None)


@pytest.fixture
def studio_client(registry_bundle, monkeypatch):
    async def compose(*, modules=()):
        assert modules == ()
        return BootstrapState(registry_bundle, ())

    monkeypatch.setattr("feedbax.web.app.compose_application", compose)
    with TestClient(create_app()) as client:
        yield client


@pytest.fixture
def studio_default_eval_recipe(registry_bundle):
    def recipe(
        spec: EvaluationRunSpec,
        root: Path,
        states_path: Path,
        _execution_context,
    ) -> EvaluationRecipeResult:
        artifact = store_json_artifact(
            {
                "kind": "StudioEvaluationSummary",
                "input_training_runs": [ref.id for ref in spec.inputs],
                "status": "completed",
            },
            root=root,
            role="evaluation_result",
            logical_name="studio-default-evaluation-summary.json",
            metadata={"states_path": str(states_path)},
        )
        return EvaluationRecipeResult(
            states={"input_training_runs": [ref.id for ref in spec.inputs]},
            summary_metrics={"toy_rollouts": len(spec.inputs)},
            artifacts=[artifact],
        )

    registry_bundle.evaluation_recipes.register("feedbax.studio.default_eval", recipe)
    yield


@pytest.fixture
def studio_default_analysis_recipe(registry_bundle):
    def recipe(spec, _root: Path, inputs, _execution_context) -> AnalysisRecipeResult:
        return AnalysisRecipeResult(
            analyses={"studio_summary": ToyAnalysis(variant="studio", cache_result=True)},
            data=build_toy_analysis_data(value=len(inputs)),
            common_inputs={"studio": spec.params.get("stage_id")},
        )

    registry_bundle.analysis_recipes.register("feedbax.analysis.activity", recipe)
    yield


def test_prepare_studio_training_execution_lowers_workspace_to_provider_plan(registry_bundle):
    request = StudioTrainingExecutionRequest(
        workspace=_workspace(),
        graph=_graph(),
        job_id="studio-plan",
        local_cwd="/tmp/feedbax-studio",
        issues=["ddd3758"],
    )

    prepared = prepare_studio_training_execution(request, registry_bundle=registry_bundle)

    assert prepared.stage_id == "stage:train"
    assert prepared.scenario_id == "scenario:train"
    assert prepared.execution_spec.kind == "training"
    assert prepared.execution_spec.backend == "local"
    assert prepared.execution_spec.issues == ["ddd3758"]
    assert prepared.execution_spec.metadata["studio"]["workspace_id"] == prepared.workspace.id
    assert prepared.execution_spec.metadata["studio"]["training_spec"]["n_batches"] == 25
    task_binding_spec = prepared.execution_spec.metadata["studio"]["task_binding_spec"]
    assert task_binding_spec["schema_version"] == "feedbax.spec.studio.task_bindings.v2"
    assert task_binding_spec["exposed_data"][0]["id"] == "inputs"
    assert "exposed_outputs" not in task_binding_spec
    assert (
        task_binding_spec["bindings"][0]["source_data_id"],
        task_binding_spec["bindings"][0]["target_node_id"],
        task_binding_spec["bindings"][0]["target_port"],
    ) == ("inputs", "network", "input")
    assert (
        "source_output_id"
        not in prepared.execution_spec.metadata["studio"]["task_binding_spec"]["bindings"][0]
    )
    assert (
        prepared.execution_spec.metadata["command_contract"]["expected_files"]
        == prepared.execution_spec.artifact_policy.tracked_paths
    )
    assert (
        prepared.execution_spec.metadata["command_contract"]["expected_files"][-2]
        == "task-binding-spec.json"
    )
    assert (
        prepared.execution_spec.metadata["command_contract"]["current_command_role"]
        == "materialize_mvp_training_result"
    )
    assert (
        prepared.execution_spec.metadata["command_contract"]["future_command_role"]
        == "launch_training_runner"
    )
    assert set(prepared.execution_spec.metadata["studio"]["graph_spec"]["nodes"]) == {"network"}
    assert prepared.plan.job_id == "studio-plan"
    assert prepared.plan.run_directory == "/tmp/feedbax-studio/feedbax_runs/studio-plan"
    assert any(route.uri == "training-spec.json" for route in prepared.plan.artifact_routes)
    assert any("real JAX training runner" in warning for warning in prepared.plan.warnings)

    train_stage = next(stage for stage in prepared.workspace.stages if stage.kind == "train")
    assert train_stage.status == "ready"
    assert train_stage.validation.valid is True
    assert train_stage.execution_spec["job_id"] == "studio-plan"
    assert train_stage.artifact_refs[0].role == "execution_plan"
    assert train_stage.manifest_refs[0].kind == "ExecutionPlan"

    future_stage = next(
        stage for stage in prepared.workspace.stages if stage.id == "stage:future-report-packaging"
    )
    assert future_stage.metadata["later_product_surface"]["keep"] is True


def test_studio_training_plan_endpoint_returns_updated_workspace(studio_client):
    client = studio_client

    response = client.post(
        "/api/provider/studio/training/plan",
        json={
            "workspace": _workspace().model_dump(mode="json", exclude_none=True),
            "graph": _graph().model_dump(mode="json", exclude_none=True),
            "job_id": "http-studio-plan",
            "issues": ["ddd3758"],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["plan"]["job_id"] == "http-studio-plan"
    assert payload["execution_spec"]["metadata"]["studio"]["stage_id"] == "stage:train"
    train_stage = next(
        stage for stage in payload["workspace"]["stages"] if stage["kind"] == "train"
    )
    assert train_stage["status"] == "ready"
    assert train_stage["artifact_refs"][0]["role"] == "execution_plan"


def test_prepare_studio_training_execution_writes_idempotent_pending_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    registry_bundle,
):
    monkeypatch.setenv("FEEDBAX_RUNS_DIR", str(tmp_path))
    request = StudioTrainingExecutionRequest(
        workspace=_workspace(),
        graph=_graph(),
        job_id="studio-plan",
        local_cwd="/tmp/feedbax-studio",
        issues=["9aa8ff2"],
    )

    first = prepare_studio_training_execution(request, registry_bundle=registry_bundle)
    second = prepare_studio_training_execution(request, registry_bundle=registry_bundle)

    train_stage = next(stage for stage in first.workspace.stages if stage.kind == "train")
    training_ref = next(ref for ref in train_stage.manifest_refs if ref.role == "training_run")
    training_collection = next(
        collection
        for collection in train_stage.output_collections
        if collection.kind == "training_runs"
    )
    second_ref = next(
        ref
        for stage in second.workspace.stages
        if stage.kind == "train"
        for ref in stage.manifest_refs
        if ref.role == "training_run"
    )

    assert training_ref.id == second_ref.id
    assert training_collection.item_refs[0].id == training_ref.id
    assert training_ref.metadata["status"] == "pending"
    manifest = load_manifest(training_ref.uri)
    assert manifest.status == "pending"
    assert manifest.training_spec.inline["n_batches"] == 25
    assert manifest.task_binding_spec.inline["bindings"][0]["target_port"] == "input"
    assert manifest.provenance.issues == ["9aa8ff2"]


def test_prepare_studio_training_execution_expands_sweep_matrix_to_pending_run_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    registry_bundle,
):
    monkeypatch.setenv("FEEDBAX_RUNS_DIR", str(tmp_path))
    workspace = _workspace()
    train_stage = next(stage for stage in workspace.stages if stage.kind == "train")
    train_stage.execution_spec = {"protocol": {"compute_target": "runpod"}}
    train_stage.selection_spec["matrix"] = {
        "name": "Loss weight sweep",
        "axes": [
            {
                "id": "loss_weight",
                "label": "loss.weight",
                "path": "training_spec.loss.weight",
                "values": [0, 1e-5],
            }
        ],
        "mode": "cross",
    }
    request = StudioTrainingExecutionRequest(
        workspace=workspace,
        graph=_graph(),
        job_id="studio-plan",
        local_cwd="/tmp/feedbax-studio",
        issues=["c199a9c"],
    )

    prepared = prepare_studio_training_execution(request, registry_bundle=registry_bundle)

    train_stage = next(stage for stage in prepared.workspace.stages if stage.kind == "train")
    run_set_ref = next(ref for ref in train_stage.manifest_refs if ref.role == "training_run_set")
    run_refs = [ref for ref in train_stage.manifest_refs if ref.role == "training_run"]
    training_collection = next(
        collection
        for collection in train_stage.output_collections
        if collection.kind == "training_runs"
    )
    run_set = load_manifest(run_set_ref.uri)
    runs = [load_manifest(ref.uri) for ref in run_refs]

    assert isinstance(run_set, TrainingRunSetManifest)
    assert run_set.name == "Loss weight sweep"
    assert run_set.metadata["matrix_schema_version"] == ("feedbax.spec.training_run_matrix.v1")
    assert run_set.metadata["studio_legacy_adapter"] is True
    assert run_set.axes.axes[0].role == "authored_sweep"
    assert run_set.axes.axes[0].values == [0, 1e-5]
    assert len(run_set.axes.runs) == 2
    assert len(run_refs) == 2
    assert {ref.id for ref in training_collection.item_refs} == {ref.id for ref in run_refs}
    assert {ref.metadata["execution_target"] for ref in run_refs} == {"runpod"}
    assert run_set_ref.metadata["execution_target"] == "runpod"
    assert all(isinstance(run, TrainingRunManifest) for run in runs)
    assert [run.training_spec.inline["loss"]["weight"] for run in runs] == [0, 1e-5]
    assert {run.metadata["execution_target"] for run in runs} == {"runpod"}
    assert {run.metadata["execution_backend"] for run in runs} == {"local"}
    assert [run.metadata["studio"]["axis_coordinates"]["loss_weight"] for run in runs] == [
        0,
        1e-5,
    ]
    assert {run.run_set_id for run in runs} == {run_set.id}


def test_prepare_studio_training_execution_uses_queue_subset_target_not_stale_stage_protocol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    registry_bundle,
):
    monkeypatch.setenv("FEEDBAX_RUNS_DIR", str(tmp_path))
    workspace = _workspace()
    train_stage = next(stage for stage in workspace.stages if stage.kind == "train")
    train_stage.execution_spec = {"protocol": {"compute_target": "gcp"}}
    runpod_ref = StudioManifestRef(
        kind="TrainingRunManifest",
        id="train:runpod",
        role="training_run",
        uri="/tmp/feedbax/runpod.json",
        metadata={"status": "pending", "planned": True, "execution_target": "runpod"},
    )
    gcp_ref = StudioManifestRef(
        kind="TrainingRunManifest",
        id="train:gcp",
        role="training_run",
        uri="/tmp/feedbax/gcp.json",
        metadata={"status": "pending", "planned": True, "execution_target": "gcp"},
    )
    train_stage.manifest_refs = [runpod_ref, gcp_ref]
    train_stage.output_collections = [
        StudioCollectionRef(
            id="collection:training-runs",
            kind="training_runs",
            label="Training runs",
            source_stage_id=train_stage.id,
            item_refs=[runpod_ref, gcp_ref],
        )
    ]

    prepared = prepare_studio_training_execution(
        StudioTrainingExecutionRequest(
            workspace=workspace,
            graph=_graph(),
            backend="runpod",
            job_id="studio-plan",
            queue_target="runpod",
            queue_manifest_ids=["train:runpod"],
            issues=["12e49a2"],
        ),
        registry_bundle=registry_bundle,
    )

    prepared_train_stage = next(
        stage for stage in prepared.workspace.stages if stage.kind == "train"
    )
    staged_summary = prepared_train_stage.metadata["last_staged_training"]
    assert prepared.plan.backend == "runpod"
    assert staged_summary["source"] == "queue_manifest_subset"
    assert staged_summary["execution_target"] == "runpod"
    assert staged_summary["manifest_ids"] == ["train:runpod"]
    assert not (tmp_path / "manifests").exists()


def test_prepare_studio_training_execution_rejects_queue_subset_target_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    registry_bundle,
):
    monkeypatch.setenv("FEEDBAX_RUNS_DIR", str(tmp_path))
    workspace = _workspace()
    train_stage = next(stage for stage in workspace.stages if stage.kind == "train")
    train_stage.execution_spec = {"protocol": {"compute_target": "gcp"}}
    train_stage.manifest_refs = [
        StudioManifestRef(
            kind="TrainingRunManifest",
            id="train:gcp",
            role="training_run",
            uri="/tmp/feedbax/gcp.json",
            metadata={"status": "pending", "planned": True, "execution_target": "gcp"},
        )
    ]

    with pytest.raises(
        StudioExecutionPreparationError,
        match="targets 'gcp', not selected target 'runpod'",
    ):
        prepare_studio_training_execution(
            StudioTrainingExecutionRequest(
                workspace=workspace,
                graph=_graph(),
                backend="runpod",
                job_id="studio-plan",
                queue_target="runpod",
                queue_manifest_ids=["train:gcp"],
                issues=["12e49a2"],
            ),
            registry_bundle=registry_bundle,
        )

    assert not (tmp_path / "manifests").exists()


def test_prepare_studio_training_execution_rejects_invalid_expanded_sweep_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    registry_bundle,
):
    monkeypatch.setenv("FEEDBAX_RUNS_DIR", str(tmp_path))
    workspace = _workspace()
    train_stage = next(stage for stage in workspace.stages if stage.kind == "train")
    train_stage.selection_spec["matrix"] = {
        "name": "Invalid batch sweep",
        "axes": [
            {
                "id": "n_batches",
                "path": "training_spec.n_batches",
                "values": [25, -1],
            }
        ],
        "mode": "cross",
    }
    request = StudioTrainingExecutionRequest(
        workspace=workspace,
        graph=_graph(),
        job_id="studio-plan",
        local_cwd="/tmp/feedbax-studio",
        issues=["c199a9c"],
    )

    with pytest.raises(StudioExecutionPreparationError, match="n_batches must be positive"):
        prepare_studio_training_execution(request, registry_bundle=registry_bundle)

    assert not (tmp_path / "manifests").exists()


def test_prepare_studio_training_execution_restages_cancelled_deterministic_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    registry_bundle,
    studio_client,
):
    monkeypatch.setenv("FEEDBAX_RUNS_DIR", str(tmp_path))
    request = StudioTrainingExecutionRequest(
        workspace=_workspace(),
        graph=_graph(),
        job_id="studio-plan",
        local_cwd="/tmp/feedbax-studio",
        issues=["9aa8ff2"],
    )

    first = prepare_studio_training_execution(request, registry_bundle=registry_bundle)
    train_stage = next(stage for stage in first.workspace.stages if stage.kind == "train")
    training_ref = next(ref for ref in train_stage.manifest_refs if ref.role == "training_run")
    client = studio_client

    cancelled = client.post(f"/api/runs/training/{training_ref.id}/cancel")
    assert cancelled.status_code == 200
    assert load_manifest(training_ref.uri).status == "cancelled"

    restaged = prepare_studio_training_execution(request, registry_bundle=registry_bundle)
    restaged_stage = next(stage for stage in restaged.workspace.stages if stage.kind == "train")
    restaged_ref = next(ref for ref in restaged_stage.manifest_refs if ref.role == "training_run")
    restaged_manifest = load_manifest(restaged_ref.uri)

    assert restaged_ref.id == training_ref.id
    assert restaged_manifest.status == "pending"
    assert restaged_manifest.completed_at is None
    assert restaged_manifest.metadata["restaged_from_status"] == "cancelled"
    assert "superseded_by" not in restaged_manifest.metadata
    assert "supersedes" not in restaged_manifest.metadata
    assert "superseded_by" not in restaged_ref.metadata
    assert "supersedes" not in restaged_ref.metadata
    assert restaged_ref.metadata["spec_hashes"]["training_spec"].startswith("fnv1a:")


def test_stage_studio_evaluation_matrix_records_checkpoint_policy_and_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    registry_bundle,
):
    monkeypatch.setenv("FEEDBAX_RUNS_DIR", str(tmp_path))
    training_request = StudioTrainingExecutionRequest(
        workspace=_workspace(),
        graph=_graph(),
        job_id="studio-plan",
        issues=["717e8fb"],
    )
    prepared_training = prepare_studio_training_execution(
        training_request, registry_bundle=registry_bundle
    )
    train_stage = next(
        stage for stage in prepared_training.workspace.stages if stage.kind == "train"
    )
    training_ref = next(ref for ref in train_stage.manifest_refs if ref.role == "training_run")
    eval_stage = next(stage for stage in prepared_training.workspace.stages if stage.kind == "eval")
    eval_stage.input_collections = [
        collection
        for collection in train_stage.output_collections
        if collection.kind == "training_runs"
    ]
    eval_stage.selection_spec["training_run_ids"] = [training_ref.id]
    request = StudioEvaluationMatrixRequest(
        workspace=prepared_training.workspace,
        training_run_ids=[training_ref.id],
        eval_params={"targets": "8-direction center-out"},
        condition_matrix={
            "axes": [
                {
                    "id": "sisu",
                    "label": "SISU",
                    "path": "eval_params.sisu",
                    "values": [0.25, 0.5],
                }
            ],
            "mode": "cross",
        },
        checkpoint_policy=StudioEvaluationCheckpointPolicy(
            mode="best-by-metric",
            metric="final_validation_loss",
            objective="minimize",
        ),
        issues=["717e8fb"],
    )

    preview = preview_studio_evaluation_matrix(request)
    first = stage_studio_evaluation_matrix(request)
    second = stage_studio_evaluation_matrix(request)

    assert preview.total_eval_count == 2
    assert preview.new_manifest_count == 2
    assert first.preview.pending_count == 2
    assert second.preview.new_manifest_count == 0
    assert {ref.id for ref in first.manifest_refs} == {ref.id for ref in second.manifest_refs}

    eval_manifest = load_manifest(first.manifest_refs[0].uri)
    checkpoint_manifest = load_manifest(first.checkpoint_selection_refs[0].uri)
    assert isinstance(eval_manifest, EvaluationRunManifest)
    assert isinstance(checkpoint_manifest, CheckpointSelectionManifest)
    assert eval_manifest.status == "pending"
    assert eval_manifest.evaluation_spec.inline["params"]["checkpoint_policy"] == {
        "mode": "best-by-metric",
        "metric": "final_validation_loss",
        "objective": "minimize",
        "params": {},
    }
    assert (
        eval_manifest.provenance.parents
        == EvaluationRunSpec.model_validate(eval_manifest.evaluation_spec.inline).inputs
    )
    assert first.manifest_refs[0].metadata["parent_refs"][0]["id"] == training_ref.id
    assert first.manifest_refs[0].metadata["spec_hashes"]["evaluation_spec"].startswith("fnv1a:")
    assert eval_manifest.provenance.metadata["checkpoint_policy"]["mode"] == "best-by-metric"
    assert checkpoint_manifest.metadata["checkpoint_policy"]["metric"] == "final_validation_loss"


def test_studio_evaluation_preview_filters_stale_manifests_explicitly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    registry_bundle,
):
    monkeypatch.setenv("FEEDBAX_RUNS_DIR", str(tmp_path))
    prepared_training = prepare_studio_training_execution(
        StudioTrainingExecutionRequest(workspace=_workspace(), graph=_graph(), job_id="studio-plan"),
        registry_bundle=registry_bundle,
    )
    train_stage = next(
        stage for stage in prepared_training.workspace.stages if stage.kind == "train"
    )
    training_ref = next(ref for ref in train_stage.manifest_refs if ref.role == "training_run")
    eval_stage = next(stage for stage in prepared_training.workspace.stages if stage.kind == "eval")
    eval_stage.input_collections = [
        collection
        for collection in train_stage.output_collections
        if collection.kind == "training_runs"
    ]
    request = StudioEvaluationMatrixRequest(
        workspace=prepared_training.workspace,
        training_run_ids=[training_ref.id],
        eval_params={"perturbation": "none"},
        checkpoint_policy=StudioEvaluationCheckpointPolicy(mode="last"),
        reprocess="missing",
        root=str(tmp_path),
    )
    staged = stage_studio_evaluation_matrix(request)
    existing_manifest = load_manifest(staged.manifest_refs[0].uri)
    assert isinstance(existing_manifest, EvaluationRunManifest)
    stale_manifest = existing_manifest.model_copy(
        update={
            "status": "stale",
            "metadata": {
                **existing_manifest.metadata,
                "staleness_reason": "upstream superseded",
            },
        }
    )
    write_manifest(stale_manifest, root=tmp_path)

    default_preview = preview_studio_evaluation_matrix(request)
    stale_preview = preview_studio_evaluation_matrix(
        request.model_copy(update={"reprocess": "stale"})
    )
    restaged = stage_studio_evaluation_matrix(request.model_copy(update={"reprocess": "stale"}))
    restaged_manifest = load_manifest(restaged.manifest_refs[0].uri)

    assert default_preview.launch_count == 0
    assert stale_preview.launch_count == 1
    assert restaged_manifest.status == "pending"
    assert "superseded_by" not in restaged_manifest.metadata
    assert "supersedes" not in restaged_manifest.metadata
    assert "superseded_by" not in restaged.manifest_refs[0].metadata
    assert "supersedes" not in restaged.manifest_refs[0].metadata


def test_studio_evaluation_run_local_reprocesses_stale_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    studio_default_eval_recipe,
    registry_bundle,
):
    monkeypatch.setenv("FEEDBAX_RUNS_DIR", str(tmp_path))
    prepared_training = prepare_studio_training_execution(
        StudioTrainingExecutionRequest(workspace=_workspace(), graph=_graph(), job_id="studio-plan"),
        registry_bundle=registry_bundle,
    )
    train_stage = next(
        stage for stage in prepared_training.workspace.stages if stage.kind == "train"
    )
    training_ref = next(ref for ref in train_stage.manifest_refs if ref.role == "training_run")
    eval_stage = next(stage for stage in prepared_training.workspace.stages if stage.kind == "eval")
    eval_stage.input_collections = [
        collection
        for collection in train_stage.output_collections
        if collection.kind == "training_runs"
    ]
    request = StudioEvaluationMatrixRequest(
        workspace=prepared_training.workspace,
        training_run_ids=[training_ref.id],
        eval_params={"perturbation": "none"},
        checkpoint_policy=StudioEvaluationCheckpointPolicy(mode="last"),
        reprocess="missing",
        root=str(tmp_path),
    )
    staged = stage_studio_evaluation_matrix(request)
    existing_manifest = load_manifest(staged.manifest_refs[0].uri)
    assert isinstance(existing_manifest, EvaluationRunManifest)
    stale_manifest = existing_manifest.model_copy(
        update={
            "status": "stale",
            "metadata": {
                **existing_manifest.metadata,
                "staleness_reason": "upstream superseded",
            },
        }
    )
    write_manifest(stale_manifest, root=tmp_path)

    launched = run_studio_evaluation_local_execution(
        request.model_copy(update={"reprocess": "stale"}),
        registry_bundle=registry_bundle,
    )
    launched_manifest = load_manifest(launched.manifest_refs[0].uri)

    assert launched.completed_count == 1
    assert launched.failed_count == 0
    assert launched.skipped_count == 0
    assert isinstance(launched_manifest, EvaluationRunManifest)
    assert launched_manifest.status == "completed"


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"mode": "best-by-metric"}, "requires metric"),
        ({"mode": "best-by-metric", "metric": "  "}, "requires metric"),
        ({"mode": "every-k"}, "requires every_k"),
        ({"mode": "every-k", "every_k": 0}, "greater than or equal to 1"),
    ],
)
def test_studio_evaluation_checkpoint_policy_rejects_incomplete_modes(payload, message):
    with pytest.raises(ValidationError, match=message):
        StudioEvaluationCheckpointPolicy.model_validate(payload)


def test_studio_evaluation_run_local_preserves_skipped_failed_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    studio_default_eval_recipe,
    registry_bundle,
):
    monkeypatch.setenv("FEEDBAX_RUNS_DIR", str(tmp_path))
    prepared_training = prepare_studio_training_execution(
        StudioTrainingExecutionRequest(workspace=_workspace(), graph=_graph(), job_id="studio-plan"),
        registry_bundle=registry_bundle,
    )
    train_stage = next(
        stage for stage in prepared_training.workspace.stages if stage.kind == "train"
    )
    training_ref = next(ref for ref in train_stage.manifest_refs if ref.role == "training_run")
    eval_stage = next(stage for stage in prepared_training.workspace.stages if stage.kind == "eval")
    eval_stage.input_collections = [
        collection
        for collection in train_stage.output_collections
        if collection.kind == "training_runs"
    ]
    request = StudioEvaluationMatrixRequest(
        workspace=prepared_training.workspace,
        training_run_ids=[training_ref.id],
        eval_params={"perturbation": "none"},
        checkpoint_policy=StudioEvaluationCheckpointPolicy(mode="last"),
        reprocess="missing",
        root=str(tmp_path),
    )
    staged = stage_studio_evaluation_matrix(request)
    existing_manifest = load_manifest(staged.manifest_refs[0].uri)
    assert isinstance(existing_manifest, EvaluationRunManifest)
    failed_manifest = existing_manifest.model_copy(update={"status": "failed"})
    write_manifest(failed_manifest, root=tmp_path)

    launched = run_studio_evaluation_local_execution(request, registry_bundle=registry_bundle)
    launched_eval_stage = next(stage for stage in launched.workspace.stages if stage.kind == "eval")

    assert launched.completed_count == 0
    assert launched.failed_count == 1
    assert launched.skipped_count == 1
    assert launched.skipped_failed_count == 1
    assert launched.errors == []
    assert launched_eval_stage.status == "failed"
    assert launched_eval_stage.metadata["last_evaluation_launch"] == {
        "completed_count": 0,
        "failed_count": 1,
        "skipped_count": 1,
        "skipped_failed_count": 1,
        "launched_at": launched_eval_stage.metadata["last_evaluation_launch"]["launched_at"],
        "reprocess": "missing",
    }


def test_studio_evaluation_endpoints_preview_stage_and_run_local(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    studio_default_eval_recipe,
    registry_bundle,
    studio_client,
):
    monkeypatch.setenv("FEEDBAX_RUNS_DIR", str(tmp_path))
    prepared_training = prepare_studio_training_execution(
        StudioTrainingExecutionRequest(workspace=_workspace(), graph=_graph(), job_id="studio-plan"),
        registry_bundle=registry_bundle,
    )
    train_stage = next(
        stage for stage in prepared_training.workspace.stages if stage.kind == "train"
    )
    training_ref = next(ref for ref in train_stage.manifest_refs if ref.role == "training_run")
    eval_stage = next(stage for stage in prepared_training.workspace.stages if stage.kind == "eval")
    eval_stage.input_collections = [
        collection
        for collection in train_stage.output_collections
        if collection.kind == "training_runs"
    ]
    payload = {
        "workspace": prepared_training.workspace.model_dump(mode="json", exclude_none=True),
        "selection_spec": {
            "mode": "explicit",
            "manifest_kind": "TrainingRunManifest",
            "ids": [training_ref.id],
        },
        "eval_params": {"perturbation": "none"},
        "checkpoint_policy": {"mode": "last"},
        "issues": ["717e8fb"],
        "root": str(tmp_path),
    }
    client = studio_client

    preview = client.post("/api/provider/studio/evaluation/preview", json=payload)
    staged = client.post("/api/provider/studio/evaluation/stage", json=payload)
    launched = client.post("/api/provider/studio/evaluation/run-local", json=payload)

    assert preview.status_code == 200
    assert preview.json()["total_eval_count"] == 1
    assert staged.status_code == 200
    assert staged.json()["preview"]["pending_count"] == 1
    assert launched.status_code == 200
    assert launched.json()["completed_count"] == 1
    eval_stage_payload = next(
        stage for stage in launched.json()["workspace"]["stages"] if stage["kind"] == "eval"
    )
    assert eval_stage_payload["status"] == "completed"
    assert (
        eval_stage_payload["output_collections"][0]["item_refs"][0]["metadata"]["status"]
        == "completed"
    )


def test_studio_training_plan_endpoint_rejects_missing_training_spec(studio_client):
    workspace = build_default_studio_workspace(label="Missing spec")
    client = studio_client

    response = client.post(
        "/api/provider/studio/training/plan",
        json={
            "workspace": workspace.model_dump(mode="json", exclude_none=True),
            "graph": _graph().model_dump(mode="json", exclude_none=True),
        },
    )

    assert response.status_code == 422
    assert "training_spec" in response.json()["detail"]


def test_task_binding_spec_rejects_legacy_v1_contract():
    with pytest.raises(ValueError, match="task_bindings.v1.*exposed_data.*source_data_id"):
        StudioTaskBindingSpec.model_validate(
            {
                "schema_version": "feedbax.studio.task_bindings.v1",
                "exposed_outputs": [],
                "bindings": [],
                "metadata": {},
            }
        )


def test_task_binding_spec_rejects_source_output_id():
    with pytest.raises(ValueError, match="source_output_id.*source_data_id"):
        StudioTaskBindingSpec.model_validate(
            {
                "schema_version": "feedbax.spec.studio.task_bindings.v2",
                "exposed_data": [
                    {
                        "id": "inputs",
                        "label": "Inputs",
                        "kind": "signal",
                        "path": "inputs",
                        "bindable": True,
                        "metadata": {},
                    },
                ],
                "bindings": [
                    {
                        "id": "task:inputs->network:input",
                        "source_output_id": "inputs",
                        "target_node_id": "network",
                        "target_port": "input",
                        "role": "model_input",
                        "metadata": {},
                    }
                ],
                "metadata": {},
            }
        )


def test_run_studio_training_local_execution_materializes_snapshot_and_refs(
    tmp_path: Path,
    registry_bundle,
):
    result = run_studio_training_local_execution(
        StudioTrainingLocalRunRequest(
            workspace=_workspace(),
            graph=_graph(),
            job_id="studio-local-run",
            root=str(tmp_path),
            issues=["ff19bc8"],
        ),
        registry_bundle=registry_bundle,
    )

    snapshot_dir = Path(result.snapshot_dir)
    assert (snapshot_dir / "execution-spec.json").exists()
    assert (snapshot_dir / "workspace-snapshot.json").exists()
    assert (snapshot_dir / "graph-spec.json").exists()
    assert (snapshot_dir / "training-spec.json").exists()
    assert (snapshot_dir / "task-spec.json").exists()
    assert (snapshot_dir / "task-binding-spec.json").exists()
    assert (snapshot_dir / "artifacts" / "training-summary.json").exists()
    execution_spec = json.loads((snapshot_dir / "execution-spec.json").read_text())
    task_binding_spec = json.loads((snapshot_dir / "task-binding-spec.json").read_text())
    workspace_snapshot = json.loads((snapshot_dir / "workspace-snapshot.json").read_text())
    assert (
        execution_spec["metadata"]["studio"]["task_binding_spec"]
        == task_binding_spec
        == workspace_snapshot["scenarios"]["scenario:train"]["task_binding_spec"]
    )
    assert task_binding_spec["schema_version"] == "feedbax.spec.studio.task_bindings.v2"
    assert task_binding_spec["exposed_data"][0]["id"] == "inputs"
    assert "exposed_outputs" not in task_binding_spec
    assert task_binding_spec["bindings"][0]["source_data_id"] == "inputs"
    assert "source_output_id" not in task_binding_spec["bindings"][0]
    assert result.result.status == "completed"
    assert result.result.return_code == 0
    assert Path(result.result.manifest_path).exists()
    assert result.result.manifest_payload["kind"] == "TrainingRunManifest"
    assert result.result.manifest_payload["training_spec"]["inline"]["n_batches"] == 25
    assert result.result.manifest_payload["task_spec"]["inline"]["type"] == "ReachingTask"
    assert (
        result.result.manifest_payload["task_binding_spec"]["inline"]["bindings"][0]["target_port"]
        == "input"
    )
    assert (
        result.result.manifest_payload["provenance"]["metadata"]["execution_metadata"]["studio"][
            "task_binding_spec"
        ]
        == task_binding_spec
    )

    train_stage = next(stage for stage in result.workspace.stages if stage.kind == "train")
    assert train_stage.status == "completed"
    assert any(ref.role == "training_run" for ref in train_stage.manifest_refs)
    assert any(ref.role == "training_result" for ref in train_stage.artifact_refs)
    assert any(ref.role == "execution_stdout" for ref in train_stage.artifact_refs)
    assert any(ref.role == "execution_input_snapshot" for ref in train_stage.artifact_refs)
    training_collection = next(
        collection
        for collection in train_stage.output_collections
        if collection.kind == "training_runs"
    )
    assert training_collection.item_refs[0].role == "training_run"
    workspace_training_collection = next(
        collection
        for collection in result.workspace.collections
        if collection.kind == "training_runs"
    )
    assert workspace_training_collection.item_refs[0].role == "training_run"

    future_stage = next(
        stage for stage in result.workspace.stages if stage.id == "stage:future-report-packaging"
    )
    assert future_stage.metadata["later_product_surface"]["keep"] is True


def test_studio_training_run_local_endpoint_returns_execution_result(tmp_path: Path, studio_client):
    client = studio_client

    response = client.post(
        "/api/provider/studio/training/run-local",
        json={
            "workspace": _workspace().model_dump(mode="json", exclude_none=True),
            "graph": _graph().model_dump(mode="json", exclude_none=True),
            "job_id": "http-studio-local-run",
            "root": str(tmp_path),
            "issues": ["ff19bc8"],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["result"]["status"] == "completed"
    assert payload["result"]["return_code"] == 0
    assert payload["snapshot_dir"].endswith("inputs")
    train_stage = next(
        stage for stage in payload["workspace"]["stages"] if stage["kind"] == "train"
    )
    assert train_stage["status"] == "completed"
    assert any(ref["role"] == "training_run" for ref in train_stage["manifest_refs"])


def test_studio_pipeline_materialize_training_writes_validation_only_artifact(
    tmp_path: Path,
) -> None:
    workspace = _workspace()
    train_stage = next(stage for stage in workspace.stages if stage.kind == "train")
    scenario = workspace.scenarios[train_stage.scenario_id]
    graph_path = tmp_path / "graph-spec.json"
    training_path = tmp_path / "training-spec.json"
    task_path = tmp_path / "task-spec.json"
    binding_path = tmp_path / "task-binding-spec.json"
    output_path = tmp_path / "artifacts" / "training-summary.json"
    graph_path.write_text(
        _graph().model_dump_json(indent=2, exclude_none=True) + "\n",
        encoding="utf-8",
    )
    training_path.write_text(json.dumps(scenario.training_spec), encoding="utf-8")
    task_path.write_text(json.dumps(scenario.task_spec), encoding="utf-8")
    binding_path.write_text(
        scenario.task_binding_spec.model_dump_json(indent=2, exclude_none=True) + "\n",
        encoding="utf-8",
    )

    rc = studio_pipeline_main(
        [
            "materialize-training",
            "--graph",
            str(graph_path),
            "--training",
            str(training_path),
            "--task",
            str(task_path),
            "--task-binding",
            str(binding_path),
            "--output",
            str(output_path),
        ]
    )

    assert rc == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["kind"] == "StudioTrainingValidationArtifact"
    assert payload["runner"] == "studio_validation_only"
    assert "final_loss" not in payload
    assert "history" not in payload


def test_materialize_studio_pipeline_requires_registered_eval_recipe(
    tmp_path: Path, registry_bundle
):
    training = run_studio_training_local_execution(
        StudioTrainingLocalRunRequest(
            workspace=_workspace(),
            graph=_graph(),
            job_id="studio-pipeline-train-unregistered",
            root=str(tmp_path),
            issues=["d30d4c2"],
        ),
        registry_bundle=registry_bundle,
    )

    with pytest.raises(ValueError, match="feedbax\\.studio\\.default_eval.*not registered"):
        materialize_studio_pipeline(
            StudioPipelineMaterializationRequest(
                workspace=training.workspace,
                job_id="studio-pipeline-unregistered",
                root=str(tmp_path),
                issues=["d30d4c2"],
            ),
            registry_bundle=registry_bundle,
        )


def test_materialize_studio_pipeline_requires_explicit_analysis_type(
    tmp_path: Path,
    studio_default_eval_recipe,
    registry_bundle,
):
    training = run_studio_training_local_execution(
        StudioTrainingLocalRunRequest(
            workspace=_workspace(),
            graph=_graph(),
            job_id="studio-pipeline-train-no-analysis-type",
            root=str(tmp_path),
            issues=["d30d4c2"],
        ),
        registry_bundle=registry_bundle,
    )

    with pytest.raises(
        StudioExecutionPreparationError,
        match="analysis_spec\\.analysis_type",
    ):
        materialize_studio_pipeline(
            StudioPipelineMaterializationRequest(
                workspace=training.workspace,
                job_id="studio-pipeline-no-analysis-type",
                root=str(tmp_path),
                issues=["d30d4c2"],
            ),
            registry_bundle=registry_bundle,
        )


def test_materialize_studio_pipeline_consumes_stage_collections(
    tmp_path: Path,
    studio_default_eval_recipe,
    studio_default_analysis_recipe,
    registry_bundle,
):
    training = run_studio_training_local_execution(
        StudioTrainingLocalRunRequest(
            workspace=_workspace_with_analysis_type("feedbax.analysis.activity"),
            graph=_graph(),
            job_id="studio-pipeline-train",
            root=str(tmp_path),
            issues=["d30d4c2"],
        ),
        registry_bundle=registry_bundle,
    )

    materialized = materialize_studio_pipeline(
        StudioPipelineMaterializationRequest(
            workspace=training.workspace,
            job_id="studio-pipeline",
            root=str(tmp_path),
            issues=["d30d4c2"],
        ),
        registry_bundle=registry_bundle,
    )

    assert materialized.stage_ids == ["stage:eval", "stage:analysis", "stage:report"]
    assert set(materialized.manifest_paths) == {"stage:eval", "stage:analysis", "stage:report"}
    assert all(Path(path).exists() for path in materialized.manifest_paths.values())

    eval_stage = next(stage for stage in materialized.workspace.stages if stage.kind == "eval")
    analysis_stage = next(
        stage for stage in materialized.workspace.stages if stage.kind == "analysis"
    )
    report_stage = next(stage for stage in materialized.workspace.stages if stage.kind == "report")

    assert eval_stage.status == "completed"
    assert analysis_stage.status == "completed"
    assert report_stage.status == "completed"
    assert eval_stage.input_collections[0].item_refs[0].role == "training_run"
    assert analysis_stage.input_collections[0].item_refs[0].role == "evaluation_run"
    assert report_stage.input_collections[0].item_refs[0].role == "analysis_run"
    assert eval_stage.output_collections[0].item_refs[0].kind == "EvaluationRunManifest"
    assert analysis_stage.output_collections[0].item_refs[0].kind == "AnalysisRunManifest"
    assert report_stage.output_collections[0].item_refs[0].kind == "ReportManifest"
    assert (
        eval_stage.output_collections[0]
        .item_refs[0]
        .metadata["parent_refs"][0]["id"]
        .startswith("feedbax-training-run:")
    )
    assert (
        analysis_stage.output_collections[0].item_refs[0].metadata["parent_refs"][0]["id"]
        == eval_stage.output_collections[0].item_refs[0].id
    )
    assert (
        report_stage.output_collections[0].item_refs[0].metadata["parent_refs"][0]["id"]
        == analysis_stage.output_collections[0].item_refs[0].id
    )

    eval_scenario = materialized.workspace.scenarios["scenario:eval"]
    assert eval_scenario.parent_scenario_id == "scenario:train"
    assert eval_scenario.task_spec["type"] == "ReachingTask"
    assert eval_scenario.task_binding_spec.bindings[0].target_node_id == "network"
    assert len(materialized.workspace.manifest_refs) >= 4
    assert any(ref.role == "report" for ref in materialized.workspace.artifact_refs)
    eval_manifest = json.loads(Path(materialized.manifest_paths["stage:eval"]).read_text())
    analysis_manifest = json.loads(Path(materialized.manifest_paths["stage:analysis"]).read_text())
    assert eval_manifest["status"] == "completed"
    assert eval_manifest["evaluation_spec"]["inline"]["evaluation_type"] == (
        "feedbax.studio.default_eval"
    )
    assert eval_manifest["summary_metrics"]["toy_rollouts"] == 1
    assert eval_manifest["provenance"]["parents"][0]["id"].startswith("feedbax-training-run:")
    assert "cache/states" in eval_manifest["metadata"]["cache"]["states_path"]
    assert analysis_manifest["status"] == "completed"
    assert analysis_manifest["analysis_spec"]["inline"]["analysis_type"] == (
        "feedbax.analysis.activity"
    )
    assert analysis_manifest["inputs"][0]["id"] == eval_manifest["id"]
    assert analysis_manifest["provenance"]["parents"][0]["id"] == eval_manifest["id"]
    assert analysis_manifest["summary_metrics"]["analysis_count"] == 1
    assert analysis_manifest["artifacts"][0]["role"] == "figure"

    future_stage = next(
        stage
        for stage in materialized.workspace.stages
        if stage.id == "stage:future-report-packaging"
    )
    assert future_stage.metadata["later_product_surface"]["keep"] is True


def test_materialize_studio_pipeline_carries_authored_evaluation_states_policy(
    tmp_path: Path,
    studio_default_eval_recipe,
    studio_default_analysis_recipe,
    registry_bundle,
) -> None:
    workspace = _workspace_with_analysis_type("feedbax.analysis.activity")
    analysis_stage = next(stage for stage in workspace.stages if stage.kind == "analysis")
    scenario = workspace.scenarios[analysis_stage.scenario_id]
    scenario.analysis_spec = {
        **(scenario.analysis_spec or {}),
        "evaluation_states_policy": "require_durable",
    }
    training = run_studio_training_local_execution(
        StudioTrainingLocalRunRequest(
            workspace=workspace,
            graph=_graph(),
            job_id="studio-policy-train",
            root=str(tmp_path),
            issues=["b594b56"],
        ),
        registry_bundle=registry_bundle,
    )

    materialized = materialize_studio_pipeline(
        StudioPipelineMaterializationRequest(
            workspace=training.workspace,
            job_id="studio-policy",
            root=str(tmp_path),
            issues=["b594b56"],
        ),
        registry_bundle=registry_bundle,
    )

    evaluation = load_manifest(materialized.manifest_paths["stage:eval"])
    analysis = load_manifest(materialized.manifest_paths["stage:analysis"])
    assert evaluation.evaluation_spec.inline["params"]["states_custody"] == "durable"
    assert analysis.analysis_spec.inline["evaluation_states_policy"] == "require_durable"
    source = analysis.evaluation_state_sources[0]
    assert source.source_kind == "durable"
    assert source.evaluation_manifest_authority.metadata["ref_schema_version"] == (
        "feedbax.ref.authenticated_manifest.v1"
    )
    assert source.evaluation_manifest_authority.uri is None


def test_materialize_studio_pipeline_endpoint_returns_updated_workspace(
    tmp_path: Path,
    studio_default_eval_recipe,
    studio_default_analysis_recipe,
    registry_bundle,
    studio_client,
):
    training = run_studio_training_local_execution(
        StudioTrainingLocalRunRequest(
            workspace=_workspace_with_analysis_type("feedbax.analysis.activity"),
            graph=_graph(),
            job_id="http-studio-pipeline-train",
            root=str(tmp_path),
            issues=["d30d4c2"],
        ),
        registry_bundle=registry_bundle,
    )

    response = studio_client.post(
        "/api/provider/studio/pipeline/materialize",
        json={
            "workspace": training.workspace.model_dump(mode="json", exclude_none=True),
            "job_id": "http-studio-pipeline",
            "root": str(tmp_path),
            "issues": ["d30d4c2"],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["stage_ids"] == ["stage:eval", "stage:analysis", "stage:report"]
    report_stage = next(
        stage for stage in payload["workspace"]["stages"] if stage["kind"] == "report"
    )
    assert report_stage["status"] == "completed"
    assert report_stage["manifest_refs"][0]["kind"] == "ReportManifest"


def _matrix_row_for_scenario(
    workspace,
    *,
    planned_run_id: str,
    payload: dict | None = None,
    coordinate=None,
    overrides=(),
) -> MaterializedMatrixRow:
    """Build a materialized matrix row carrying the train scenario's spec envelope."""
    stage = next(stage for stage in workspace.stages if stage.kind == "train")
    scenario = workspace.scenarios[stage.scenario_id]
    envelope = {
        "graph_spec": _graph().model_dump(mode="json", exclude_none=True),
        "training_spec": scenario.training_spec,
        "task_spec": scenario.task_spec,
        "task_binding_spec": scenario.task_binding_spec.model_dump(mode="json", exclude_none=True),
    }
    return MaterializedMatrixRow(
        row_id=f"row-for-{planned_run_id}",
        planned_run_id=planned_run_id,
        spec=None,
        authored_payload=dict(payload if payload is not None else envelope),
        payload=dict(payload if payload is not None else envelope),
        provenance=TrainingRowProvenance(
            row_id=f"row-for-{planned_run_id}",
            row_index=0,
            planned_run_id=planned_run_id,
            authored_payload_hash="0" * 64,
            lowered_execution_payload_hash="1" * 64,
            axis_coordinates={},
        ),
        coordinate=coordinate,
        overrides=list(overrides),
    )


def _single_run_emitter_kwargs(workspace) -> dict:
    stage = next(stage for stage in workspace.stages if stage.kind == "train")
    scenario = workspace.scenarios[stage.scenario_id]
    return {
        "workspace": workspace,
        "stage": stage,
        "scenario_id": stage.scenario_id,
        "graph_spec": _graph().model_dump(mode="json", exclude_none=True),
        "training_spec": scenario.training_spec,
        "task_spec": scenario.task_spec,
        "task_binding_spec": scenario.task_binding_spec.model_dump(mode="json", exclude_none=True),
        "request": StudioTrainingExecutionRequest(
            workspace=workspace, graph=_graph(), job_id="job-shared"
        ),
        "execution_target": "local",
    }


def test_studio_training_contract_files_are_declared_once(registry_bundle):
    """The command contract and the artifact policy read one declared file list."""
    workspace = _workspace()
    stage = next(stage for stage in workspace.stages if stage.kind == "train")
    spec = _build_execution_spec(
        request=StudioTrainingExecutionRequest(
            workspace=workspace, graph=_graph(), job_id="contract-files"
        ),
        workspace=workspace,
        stage=stage,
        job_id="contract-files",
    )

    declared = list(STUDIO_TRAINING_CONTRACT_FILES)
    assert spec.metadata["command_contract"]["expected_files"] == declared
    assert spec.artifact_policy.tracked_paths == declared
    # Distinct list objects, so mutating one emitted spec cannot corrupt the other
    # call site or the module constant.
    assert (
        spec.metadata["command_contract"]["expected_files"]
        is not spec.artifact_policy.tracked_paths
    )


def test_pending_training_manifest_paths_emit_through_one_builder(tmp_path: Path, monkeypatch):
    """Single-run and matrix-row staging both reach _build_pending_training_manifest."""
    calls: list[str] = []
    original = _build_pending_training_manifest

    def spy(**kwargs):
        calls.append(kwargs["manifest_id"])
        return original(**kwargs)

    monkeypatch.setattr("feedbax.studio.execution._build_pending_training_manifest", spy)

    workspace = _workspace()
    _write_pending_training_manifest(**_single_run_emitter_kwargs(workspace))
    _write_pending_training_manifest_for_matrix_row(
        _matrix_row_for_scenario(workspace, planned_run_id="planned-shared-builder"),
        workspace=workspace,
        stage=next(stage for stage in workspace.stages if stage.kind == "train"),
        scenario_id="scenario:train",
        run_set_id="run-set-shared",
        request=StudioTrainingExecutionRequest(
            workspace=workspace, graph=_graph(), job_id="job-shared"
        ),
        root=tmp_path,
        execution_target="local",
    )

    assert len(calls) == 2
    assert calls[1] == "planned-shared-builder"


def test_single_run_and_matrix_row_pending_manifests_agree_except_on_run_identity(
    tmp_path: Path,
):
    """Both pending manifests carry one shared metadata contract.

    The only permitted divergence is run identity: a matrix row additionally
    carries its coordinate name/label, axis value indices, and run-set id, while
    a single run carries its resolved seed. Everything else -- including the
    Studio spec hashes -- must agree for equivalent inputs.
    """
    workspace = _workspace()
    single, _ = _write_pending_training_manifest(**_single_run_emitter_kwargs(workspace))
    matrix, _ = _write_pending_training_manifest_for_matrix_row(
        _matrix_row_for_scenario(
            workspace,
            planned_run_id="planned-parity",
            coordinate=TrainingRunAxisCoordinate(
                run_id="planned-parity", index=0, value_indices={}, values={}, label="parity"
            ),
        ),
        workspace=workspace,
        stage=next(stage for stage in workspace.stages if stage.kind == "train"),
        scenario_id="scenario:train",
        run_set_id="run-set-parity",
        request=StudioTrainingExecutionRequest(
            workspace=workspace, graph=_graph(), job_id="job-shared"
        ),
        root=tmp_path,
        execution_target="local",
    )

    assert single.metadata["spec_hashes"] == matrix.metadata["spec_hashes"]
    assert single.summary_metrics == matrix.summary_metrics == {"total_batches": 25}
    assert set(matrix.metadata) - set(single.metadata) == {"name", "label"}
    assert set(single.metadata) - set(matrix.metadata) == set()

    single_studio = single.metadata["studio"]
    matrix_studio = matrix.metadata["studio"]
    assert set(single_studio) - set(matrix_studio) == {"seed"}
    assert set(matrix_studio) - set(single_studio) == {"axis_value_indices", "run_set_id"}
    shared = set(single_studio) & set(matrix_studio) - {"planned_training_run_id"}
    assert {key: single_studio[key] for key in shared} == {
        key: matrix_studio[key] for key in shared
    }
    assert single_studio["planned_training_run_id"] == single.id
    assert matrix_studio["planned_training_run_id"] == matrix.id
    for manifest in (single, matrix):
        assert manifest.provenance.metadata["studio"] == manifest.metadata["studio"]


def test_pending_matrix_manifest_reuses_existing_before_deriving_row_specs(tmp_path: Path):
    """An already-staged manifest short-circuits before the row envelope is read."""
    workspace = _workspace()
    stage = next(stage for stage in workspace.stages if stage.kind == "train")
    existing = TrainingRunManifest(
        id="planned-reuse-first",
        status="completed",
        job_id="prior-job",
        metadata={"planned": True, "prior": True},
    )
    write_manifest(existing, root=tmp_path)

    manifest, path = _write_pending_training_manifest_for_matrix_row(
        # A payload with no spec envelope would raise if it were read first.
        _matrix_row_for_scenario(
            workspace, planned_run_id="planned-reuse-first", payload={"graph_spec": {}}
        ),
        workspace=workspace,
        stage=stage,
        scenario_id="scenario:train",
        run_set_id="run-set-reuse",
        request=StudioTrainingExecutionRequest(
            workspace=workspace, graph=_graph(), job_id="job-reuse"
        ),
        root=tmp_path,
        execution_target="local",
    )

    assert manifest.status == "completed"
    assert manifest.job_id == "prior-job"
    assert path == write_manifest(existing, root=tmp_path)


def test_pending_matrix_manifest_restages_cancelled_run(tmp_path: Path):
    """The shared reuse helper restages a cancelled matrix row back to pending."""
    workspace = _workspace()
    stage = next(stage for stage in workspace.stages if stage.kind == "train")
    cancelled = TrainingRunManifest(
        id="planned-restage-row",
        status="cancelled",
        completed_at=utc_now(),
        metadata={"planned": True, "superseded_by": "planned-restage-row"},
    )
    write_manifest(cancelled, root=tmp_path)

    manifest, _ = _write_pending_training_manifest_for_matrix_row(
        _matrix_row_for_scenario(workspace, planned_run_id="planned-restage-row"),
        workspace=workspace,
        stage=stage,
        scenario_id="scenario:train",
        run_set_id="run-set-restage",
        request=StudioTrainingExecutionRequest(
            workspace=workspace, graph=_graph(), job_id="job-restage"
        ),
        root=tmp_path,
        execution_target="local",
    )

    assert manifest.status == "pending"
    assert manifest.completed_at is None
    assert manifest.metadata["restaged_from_status"] == "cancelled"
    assert "superseded_by" not in manifest.metadata
