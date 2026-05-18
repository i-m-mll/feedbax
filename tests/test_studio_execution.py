from __future__ import annotations

from fastapi.testclient import TestClient

from feedbax.studio_execution import (
    StudioTrainingExecutionRequest,
    prepare_studio_training_execution,
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
