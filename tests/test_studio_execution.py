from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from feedbax.studio_execution import (
    StudioPipelineMaterializationRequest,
    StudioTrainingLocalRunRequest,
    StudioTrainingExecutionRequest,
    materialize_studio_pipeline,
    prepare_studio_training_execution,
    run_studio_training_local_execution,
)
from feedbax.web.app import create_app
from feedbax.web.models.graph import (
    GraphMetadata,
    GraphSpec,
    StudioStageSpec,
    build_default_studio_workspace,
)


def _graph() -> GraphSpec:
    return GraphSpec(
        metadata=GraphMetadata(
            name="Studio execution smoke",
            created_at="2026-05-18T00:00:00+00:00",
            updated_at="2026-05-18T00:00:00+00:00",
        )
    )


def _workspace():
    workspace = build_default_studio_workspace(label="Studio execution", graph=_graph())
    train_stage = next(stage for stage in workspace.stages if stage.kind == "train")
    scenario = workspace.scenarios[train_stage.scenario_id]
    scenario.training_spec = {
        "optimizer": {"type": "adam", "params": {"learning_rate": 0.001}},
        "loss": {"type": "Composite", "label": "loss", "weight": 1.0, "children": {}},
        "n_batches": 25,
        "batch_size": 8,
    }
    scenario.task_spec = {
        "type": "ReachingTask",
        "params": {"n_targets": 4, "target_radius": 0.02},
    }
    workspace.stages.append(
        StudioStageSpec(
            id="stage:future-report-packaging",
            kind="protocol",
            label="Future report packaging",
            metadata={"later_product_surface": {"keep": True}},
        )
    )
    return workspace


def test_prepare_studio_training_execution_lowers_workspace_to_provider_plan():
    request = StudioTrainingExecutionRequest(
        workspace=_workspace(),
        job_id="studio-plan",
        local_cwd="/tmp/feedbax-studio",
        issues=["ddd3758"],
    )

    prepared = prepare_studio_training_execution(request)

    assert prepared.stage_id == "stage:train"
    assert prepared.scenario_id == "scenario:train"
    assert prepared.execution_spec.kind == "training"
    assert prepared.execution_spec.backend == "local"
    assert prepared.execution_spec.issues == ["ddd3758"]
    assert prepared.execution_spec.metadata["studio"]["workspace_id"] == prepared.workspace.id
    assert prepared.execution_spec.metadata["studio"]["training_spec"]["n_batches"] == 25
    assert prepared.plan.job_id == "studio-plan"
    assert prepared.plan.run_directory == "/tmp/feedbax-studio/feedbax_runs/studio-plan"
    assert any(route.source == "training-spec.json" for route in prepared.plan.artifact_routes)
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


def test_studio_training_plan_endpoint_returns_updated_workspace():
    client = TestClient(create_app())

    response = client.post(
        "/api/provider/studio/training/plan",
        json={
            "workspace": _workspace().model_dump(mode="json", exclude_none=True),
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


def test_studio_training_plan_endpoint_rejects_missing_training_spec():
    workspace = build_default_studio_workspace(label="Missing spec", graph=_graph())
    client = TestClient(create_app())

    response = client.post(
        "/api/provider/studio/training/plan",
        json={"workspace": workspace.model_dump(mode="json", exclude_none=True)},
    )

    assert response.status_code == 422
    assert "training_spec" in response.json()["detail"]


def test_run_studio_training_local_execution_materializes_snapshot_and_refs(
    tmp_path: Path,
):
    result = run_studio_training_local_execution(
        StudioTrainingLocalRunRequest(
            workspace=_workspace(),
            job_id="studio-local-run",
            root=str(tmp_path),
            issues=["ff19bc8"],
        )
    )

    snapshot_dir = Path(result.snapshot_dir)
    assert (snapshot_dir / "execution-spec.json").exists()
    assert (snapshot_dir / "workspace-snapshot.json").exists()
    assert (snapshot_dir / "graph-spec.json").exists()
    assert (snapshot_dir / "training-spec.json").exists()
    assert (snapshot_dir / "task-spec.json").exists()
    assert (snapshot_dir / "artifacts" / "training-summary.json").exists()
    assert result.result.status == "completed"
    assert result.result.return_code == 0
    assert Path(result.result.manifest_path).exists()
    assert result.result.manifest_payload["kind"] == "TrainingRunManifest"
    assert result.result.manifest_payload["training_spec"]["inline"]["n_batches"] == 25
    assert result.result.manifest_payload["task_spec"]["inline"]["type"] == "ReachingTask"

    train_stage = next(stage for stage in result.workspace.stages if stage.kind == "train")
    assert train_stage.status == "completed"
    assert any(ref.role == "training_run" for ref in train_stage.manifest_refs)
    assert any(ref.role == "training_result" for ref in train_stage.artifact_refs)
    assert any(ref.role == "execution_stdout" for ref in train_stage.artifact_refs)
    assert any(ref.role == "execution_input_snapshot" for ref in train_stage.artifact_refs)
    training_collection = next(
        collection for collection in train_stage.output_collections if collection.kind == "training_runs"
    )
    assert training_collection.item_refs[0].role == "training_run"
    workspace_training_collection = next(
        collection for collection in result.workspace.collections if collection.kind == "training_runs"
    )
    assert workspace_training_collection.item_refs[0].role == "training_run"

    future_stage = next(
        stage for stage in result.workspace.stages if stage.id == "stage:future-report-packaging"
    )
    assert future_stage.metadata["later_product_surface"]["keep"] is True


def test_studio_training_run_local_endpoint_returns_execution_result(tmp_path: Path):
    client = TestClient(create_app())

    response = client.post(
        "/api/provider/studio/training/run-local",
        json={
            "workspace": _workspace().model_dump(mode="json", exclude_none=True),
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


def test_materialize_studio_pipeline_consumes_stage_collections(tmp_path: Path):
    training = run_studio_training_local_execution(
        StudioTrainingLocalRunRequest(
            workspace=_workspace(),
            job_id="studio-pipeline-train",
            root=str(tmp_path),
            issues=["d30d4c2"],
        )
    )

    materialized = materialize_studio_pipeline(
        StudioPipelineMaterializationRequest(
            workspace=training.workspace,
            job_id="studio-pipeline",
            root=str(tmp_path),
            issues=["d30d4c2"],
        )
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

    eval_scenario = materialized.workspace.scenarios["scenario:eval"]
    assert eval_scenario.parent_scenario_id == "scenario:train"
    assert eval_scenario.task_spec["type"] == "ReachingTask"
    assert len(materialized.workspace.manifest_refs) >= 4
    assert any(ref.role == "report" for ref in materialized.workspace.artifact_refs)

    future_stage = next(
        stage
        for stage in materialized.workspace.stages
        if stage.id == "stage:future-report-packaging"
    )
    assert future_stage.metadata["later_product_surface"]["keep"] is True


def test_materialize_studio_pipeline_endpoint_returns_updated_workspace(tmp_path: Path):
    training = run_studio_training_local_execution(
        StudioTrainingLocalRunRequest(
            workspace=_workspace(),
            job_id="http-studio-pipeline-train",
            root=str(tmp_path),
            issues=["d30d4c2"],
        )
    )
    client = TestClient(create_app())

    response = client.post(
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
