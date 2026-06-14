from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from feedbax.analysis.evaluation import (
    EvaluationRecipeResult,
    execute_evaluation_run_spec,
    register_evaluation_recipe,
    unregister_evaluation_recipe,
)
from feedbax.analysis.specs import (
    AnalysisRecipeResult,
    execute_analysis_run_spec,
    register_analysis_recipe,
    unregister_analysis_recipe,
)
from feedbax.contracts.manifest import (
    AnalysisRunSpec,
    EvaluationRunSpec,
    ParentRef,
    analysis_run_manifest_id,
    load_manifest,
)
from feedbax.web.app import create_app
from tests.analysis_fixtures import ToyAnalysis, build_toy_analysis_data


TOY_JOB_EVAL_TYPE = "feedbax_test_studio_job_eval"
TOY_JOB_ANALYSIS_TYPE = "feedbax_test_studio_job_analysis"
MISMATCHED_JOB_ANALYSIS_TYPE = "feedbax_test_studio_job_mismatched_analysis"


def _register_job_eval_recipe() -> None:
    def recipe(
        spec: EvaluationRunSpec,
        _root: Path,
        _states_path: Path,
    ) -> EvaluationRecipeResult:
        return EvaluationRecipeResult(
            states={"value": spec.params["value"]},
            summary_metrics={"value": spec.params["value"]},
        )

    register_evaluation_recipe(TOY_JOB_EVAL_TYPE, recipe, replace=True)


def _register_job_analysis_recipe() -> None:
    def recipe(spec: AnalysisRunSpec, _root: Path, inputs) -> AnalysisRecipeResult:
        value = sum(int(resolved.states["value"]) for resolved in inputs)
        return AnalysisRecipeResult(
            analyses={TOY_JOB_ANALYSIS_TYPE: ToyAnalysis(variant="studio", cache_result=True)},
            data=build_toy_analysis_data(value=value),
            common_inputs={"studio": spec.params["studio"]},
        )

    register_analysis_recipe(TOY_JOB_ANALYSIS_TYPE, recipe, replace=True)


def _register_mismatched_job_analysis_recipe() -> None:
    def recipe(spec: AnalysisRunSpec, _root: Path, inputs) -> AnalysisRecipeResult:
        value = sum(int(resolved.states["value"]) for resolved in inputs)
        return AnalysisRecipeResult(
            analyses={"actual_output": ToyAnalysis(variant="studio", cache_result=True)},
            data=build_toy_analysis_data(value=value),
            common_inputs={"studio": spec.params["studio"]},
        )

    register_analysis_recipe(MISMATCHED_JOB_ANALYSIS_TYPE, recipe, replace=True)


def _execute_eval(root: Path):
    spec = EvaluationRunSpec(
        evaluation_type=TOY_JOB_EVAL_TYPE,
        inputs=[
            ParentRef(
                kind="TrainingRunManifest",
                id="feedbax-training-run:studio-job",
                role="training_run",
            )
        ],
        params={"value": 5},
    )
    return execute_evaluation_run_spec(spec, root=root, issues=["studio-eval-fixture"])


def test_studio_analysis_job_routes_eval_run_through_executable_spec(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("FEEDBAX_RUNS_DIR", str(tmp_path))
    _register_job_eval_recipe()
    _register_job_analysis_recipe()
    try:
        eval_manifest, _eval_path = _execute_eval(tmp_path)
        spec = AnalysisRunSpec(
            analysis_type=TOY_JOB_ANALYSIS_TYPE,
            inputs=[
                ParentRef(
                    kind="EvaluationRunManifest",
                    id=eval_manifest.id,
                    role="evaluation_run",
                )
            ],
            params={
                "requested_outputs": [TOY_JOB_ANALYSIS_TYPE],
                "studio": {
                    "node_id": TOY_JOB_ANALYSIS_TYPE,
                    "force_rerun": False,
                },
            },
        )

        with TestClient(create_app()) as client:
            response = client.post(
                "/api/analyses/jobs",
                json={
                    "node_id": TOY_JOB_ANALYSIS_TYPE,
                    "eval_run_id": eval_manifest.id,
                    "force_rerun": False,
                },
            )
            assert response.status_code == 200
            payload = response.json()["data"]
            assert payload["manifest_id"] == analysis_run_manifest_id(spec)

            status_payload = None
            for _ in range(50):
                status_response = client.get(f"/api/analyses/jobs/status/{payload['request_id']}")
                status_payload = status_response.json()["data"]
                if status_payload["status"] in {"complete", "error"}:
                    break
                time.sleep(0.05)

        assert status_payload is not None
        assert status_payload["status"] == "complete", status_payload.get("error")
        assert status_payload["manifest_id"] == payload["manifest_id"]
        assert Path(status_payload["manifest_path"]).exists()
        assert status_payload["artifact_ids"]
        assert status_payload["artifact_paths"]

        manifest = load_manifest(status_payload["manifest_path"])
        assert manifest.kind == "AnalysisRunManifest"
        assert manifest.inputs[0].id == eval_manifest.id
        assert manifest.provenance.parents[0].id == eval_manifest.id
        assert manifest.provenance.issues == []
        assert manifest.analysis_spec.inline["params"]["requested_outputs"] == [
            TOY_JOB_ANALYSIS_TYPE
        ]
        assert manifest.summary_metrics["analysis_count"] == 1

        rerun_manifest, _rerun_path = execute_analysis_run_spec(
            spec,
            root=tmp_path,
            fig_dump_formats=("json",),
        )
        assert rerun_manifest.id == manifest.id
    finally:
        unregister_analysis_recipe(TOY_JOB_ANALYSIS_TYPE)
        unregister_evaluation_recipe(TOY_JOB_EVAL_TYPE)


def test_studio_analysis_job_errors_when_requested_node_matches_no_analysis_key(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("FEEDBAX_RUNS_DIR", str(tmp_path))
    _register_job_eval_recipe()
    _register_mismatched_job_analysis_recipe()
    try:
        eval_manifest, _eval_path = _execute_eval(tmp_path)

        with TestClient(create_app()) as client:
            response = client.post(
                "/api/analyses/jobs",
                json={
                    "node_id": MISMATCHED_JOB_ANALYSIS_TYPE,
                    "eval_run_id": eval_manifest.id,
                    "force_rerun": False,
                },
            )
            assert response.status_code == 200
            payload = response.json()["data"]

            status_payload = None
            for _ in range(50):
                status_response = client.get(f"/api/analyses/jobs/status/{payload['request_id']}")
                status_payload = status_response.json()["data"]
                if status_payload["status"] in {"complete", "error"}:
                    break
                time.sleep(0.05)

        assert status_payload is not None
        assert status_payload["status"] == "error"
        assert f"requested_outputs=['{MISMATCHED_JOB_ANALYSIS_TYPE}']" in status_payload["error"]
        assert "available_analysis_keys=['actual_output']" in status_payload["error"]
    finally:
        unregister_analysis_recipe(MISMATCHED_JOB_ANALYSIS_TYPE)
        unregister_evaluation_recipe(TOY_JOB_EVAL_TYPE)


def test_studio_analysis_job_requires_eval_run_id() -> None:
    with TestClient(create_app()) as client:
        response = client.post(
            "/api/analyses/jobs",
            json={"node_id": TOY_JOB_ANALYSIS_TYPE},
        )

    assert response.status_code == 400
    assert "eval_run_id is required" in response.json()["detail"]
