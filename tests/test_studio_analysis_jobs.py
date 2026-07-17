from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from feedbax.analysis.evaluation import (
    EvaluationRecipeResult,
    execute_evaluation_run_spec,
    register_evaluation_recipe,
    unregister_evaluation_recipe,
)
from feedbax.analysis.context import AnalysisRunContext
import feedbax.analysis.specs as analysis_specs
from feedbax.analysis.specs import (
    AnalysisRecipeResult,
    execute_analysis_run_spec,
    register_analysis_recipe,
    unregister_analysis_recipe,
)
from feedbax.analysis.manifest_inputs import is_authenticated_manifest_ref
from feedbax.contracts.manifest import (
    AnalysisRunSpec,
    ArtifactRef,
    EvaluationRunSpec,
    ParentRef,
    analysis_run_manifest_id,
    load_manifest,
)
from feedbax.contracts.studio_api import GenerateAnalysisRequest
from feedbax.web.api.analysis import _run_analysis_sync, _spec_for_analysis_request
from feedbax.web.app import create_app
from tests.analysis_fixtures import ToyAnalysis, build_toy_analysis_data


TOY_JOB_EVAL_TYPE = "feedbax.test.studio_job_eval"
TOY_JOB_ANALYSIS_TYPE = "feedbax.test.studio_job_analysis"
MISMATCHED_JOB_ANALYSIS_TYPE = "feedbax.test.studio_job_mismatched_analysis"


def _register_job_eval_recipe() -> None:
    def recipe(
        spec: EvaluationRunSpec,
        _root: Path,
        _states_path: Path,
        _execution_context,
    ) -> EvaluationRecipeResult:
        return EvaluationRecipeResult(
            states={"value": spec.params["value"]},
            summary_metrics={"value": spec.params["value"]},
        )

    register_evaluation_recipe(TOY_JOB_EVAL_TYPE, recipe, replace=True)


def _register_job_analysis_recipe() -> None:
    def recipe(
        spec: AnalysisRunSpec,
        _root: Path,
        inputs,
        _execution_context,
    ) -> AnalysisRecipeResult:
        value = sum(int(resolved.states["value"]) for resolved in inputs)
        return AnalysisRecipeResult(
            analyses={TOY_JOB_ANALYSIS_TYPE: ToyAnalysis(variant="studio", cache_result=True)},
            data=build_toy_analysis_data(value=value),
            common_inputs={"studio": spec.params["studio"]},
        )

    register_analysis_recipe(TOY_JOB_ANALYSIS_TYPE, recipe, replace=True)


def _register_mismatched_job_analysis_recipe() -> None:
    def recipe(
        spec: AnalysisRunSpec,
        _root: Path,
        inputs,
        _execution_context,
    ) -> AnalysisRecipeResult:
        value = sum(int(resolved.states["value"]) for resolved in inputs)
        return AnalysisRecipeResult(
            analyses={"actual_output": ToyAnalysis(variant="studio", cache_result=True)},
            data=build_toy_analysis_data(value=value),
            common_inputs={"studio": spec.params["studio"]},
        )

    register_analysis_recipe(MISMATCHED_JOB_ANALYSIS_TYPE, recipe, replace=True)


def _execute_eval(root: Path, *, durable: bool = False):
    spec = EvaluationRunSpec(
        evaluation_type=TOY_JOB_EVAL_TYPE,
        inputs=[
            ParentRef(
                kind="TrainingRunManifest",
                id="feedbax-training-run:studio-job",
                role="training_run",
            )
        ],
        params={"value": 5, **({"states_custody": "durable"} if durable else {})},
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
        assert status_payload["artifact_paths"] == [
            artifact.uri
            if (artifact.uri or "").startswith("artifact://sha256/")
            else str(Path(artifact.uri))
            for artifact in manifest.artifacts
            if artifact.uri is not None
        ]
        assert all("artifact:/sha256/" not in uri for uri in status_payload["artifact_paths"])
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


def test_run_analysis_sync_preserves_canonical_uri_and_legacy_path_shape(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context = AnalysisRunContext(
        spec=AnalysisRunSpec(analysis_type="test.studio_canonical_location"),
        root=tmp_path,
        index_manifest=False,
    )
    canonical = context.record_json_artifact(
        {"location": "canonical"},
        role="analysis",
        logical_name="canonical.json",
    )
    legacy_path = tmp_path / "legacy.json"
    legacy = ArtifactRef(
        role="analysis",
        logical_name="legacy.json",
        uri=str(legacy_path),
    )
    manifest = SimpleNamespace(
        id="feedbax-analysis-run:studio-artifact-locations",
        artifacts=[canonical, legacy],
    )

    def execute_stub(_spec, *, fig_dump_formats):
        assert fig_dump_formats == ("json",)
        return manifest, tmp_path / "manifest.json"

    monkeypatch.setattr(analysis_specs, "execute_analysis_run_spec", execute_stub)

    result = _run_analysis_sync(AnalysisRunSpec(analysis_type="test.studio_locations"))

    assert canonical.uri == canonical.artifact_id
    assert result.artifact_paths == [canonical.uri, str(Path(legacy.uri))]
    assert all("artifact:/sha256/" not in uri for uri in result.artifact_paths)


def test_studio_analysis_job_executes_require_durable_with_exact_authority(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("FEEDBAX_RUNS_DIR", str(tmp_path))
    _register_job_eval_recipe()
    _register_job_analysis_recipe()
    try:
        evaluation, _ = _execute_eval(tmp_path, durable=True)
        request = GenerateAnalysisRequest(
            node_id=TOY_JOB_ANALYSIS_TYPE,
            eval_run_id=evaluation.id,
            evaluation_states_policy="require_durable",
        )
        spec = _spec_for_analysis_request(request, root=tmp_path)
        assert spec.evaluation_states_policy == "require_durable"
        assert is_authenticated_manifest_ref(spec.inputs[0])

        with TestClient(create_app()) as client:
            response = client.post("/api/analyses/jobs", json=request.model_dump(mode="json"))
            assert response.status_code == 200
            request_id = response.json()["data"]["request_id"]
            status_payload = None
            for _ in range(50):
                status_response = client.get(f"/api/analyses/jobs/status/{request_id}")
                status_payload = status_response.json()["data"]
                if status_payload["status"] in {"complete", "error"}:
                    break
                time.sleep(0.05)

        assert status_payload is not None
        assert status_payload["status"] == "complete", status_payload.get("error")
        analysis = load_manifest(status_payload["manifest_path"])
        assert analysis.evaluation_state_sources[0].source_kind == "durable"
        assert is_authenticated_manifest_ref(
            analysis.evaluation_state_sources[0].evaluation_manifest_authority
        )
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
