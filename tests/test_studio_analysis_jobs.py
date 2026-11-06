from __future__ import annotations

import asyncio
import time
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace

import httpx
from fastapi.testclient import TestClient
import pytest

from feedbax.analysis.analysis import IdentityNode, SinglePort
from feedbax.analysis.evaluation import (
    EvaluationRecipeResult,
    execute_evaluation_run_spec,
)
from feedbax.analysis.context import AnalysisRunContext
import feedbax.analysis.specs as analysis_specs
from feedbax.analysis.specs import (
    AnalysisRecipeExecutionError,
    AnalysisRecipeResult,
    execute_analysis_run_spec,
)
from feedbax.analysis.manifest_inputs import is_authenticated_manifest_ref
from feedbax.contracts.base import (
    ArtifactRef,
    ParentRef,
)
from feedbax.contracts.manifest import (
    AnalysisRunSpec,
    EvaluationRunSpec,
    analysis_run_manifest_id,
    load_manifest,
)
from feedbax.contracts.studio_api import GenerateAnalysisRequest
from feedbax.web.api import analysis as analysis_api
from feedbax.web.api.analysis import _run_analysis_sync, _spec_for_analysis_request
from feedbax.web.app import create_app
from feedbax.plugins.bootstrap import BootstrapState
from tests.support.analysis_fixtures import ToyAnalysis, build_toy_analysis_data
from tests.support.application_registry import registry_bundle_fixture as registry_bundle_fixture


TOY_JOB_EVAL_TYPE = "feedbax.test.studio_job_eval"
TOY_JOB_ANALYSIS_TYPE = "feedbax.test.studio_job_analysis"
MISMATCHED_JOB_ANALYSIS_TYPE = "feedbax.test.studio_job_mismatched_analysis"
UNWIRED_JOB_OUTPUT = "unwired_output"


def _register_job_eval_recipe(registries) -> None:
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

    registries.evaluation_recipes.register(TOY_JOB_EVAL_TYPE, recipe)


def _register_job_analysis_recipe(registries, *, revision: str | None = None) -> None:
    def recipe(
        spec: AnalysisRunSpec,
        _root: Path,
        inputs,
        _execution_context,
    ) -> AnalysisRecipeResult:
        value = sum(int(resolved.states["value"]) for resolved in inputs)
        return AnalysisRecipeResult(
            analyses={
                TOY_JOB_ANALYSIS_TYPE: ToyAnalysis(variant="studio", cache_result=True),
                UNWIRED_JOB_OUTPUT: IdentityNode(inputs=SinglePort(input="missing_dependency")),
            },
            data=build_toy_analysis_data(value=value),
            common_inputs={"studio": spec.params["studio"]},
        )

    registries.analysis_recipes.register(
        TOY_JOB_ANALYSIS_TYPE, recipe, computation_revision=revision
    )


def _register_mismatched_job_analysis_recipe(registries) -> None:
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

    registries.analysis_recipes.register(MISMATCHED_JOB_ANALYSIS_TYPE, recipe)


def _execute_eval(root: Path, registries, *, durable: bool = False):
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
    return execute_evaluation_run_spec(
        spec,
        registry=registries.evaluation_recipes,
        root=root,
        issues=["studio-eval-fixture"],
    )


@pytest.fixture
def studio_registries(application_registry_bundle, monkeypatch):
    async def compose(*, modules=()):
        assert modules == ()
        return BootstrapState(application_registry_bundle, ())

    monkeypatch.setattr("feedbax.web.app.compose_application", compose)
    return application_registry_bundle


def test_studio_analysis_jobs_bound_and_drain_per_app(
    tmp_path: Path,
    monkeypatch,
    studio_registries,
) -> None:
    monkeypatch.setenv("FEEDBAX_RUNS_DIR", str(tmp_path))
    jobs_path = "/api/analyses/jobs"
    release_analysis = Event()
    figure_calls: list[str] = []

    def figure_request(name: str) -> dict:
        return {
            "node_id": name,
            "job_kind": "figure",
            "figure_spec": {"name": name, "assembler": "test.studio_figure"},
        }

    def result(manifest_id: str) -> analysis_api.AnalysisJobResult:
        return analysis_api.AnalysisJobResult(
            manifest_id=manifest_id,
            manifest_path=str(tmp_path / "manifest.json"),
            figure_hashes=[],
            artifact_ids=[],
            artifact_paths=[],
        )

    async def submit(client: httpx.AsyncClient, payload: dict) -> str:
        response = await client.post(jobs_path, json=payload)
        assert response.status_code == 200, response.text
        data = response.json()["data"]
        assert data["status"] == "pending"
        return data["request_id"]

    async def status(client: httpx.AsyncClient, request_id: str) -> dict:
        response = await client.get(f"{jobs_path}/status/{request_id}")
        assert response.status_code == 200, response.text
        return response.json()["data"]

    async def settle(client: httpx.AsyncClient, request_id: str) -> dict:
        async with asyncio.timeout(5):
            while True:
                data = await status(client, request_id)
                if data["status"] in {"complete", "error"}:
                    return data
                await asyncio.sleep(0.01)

    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        analysis_started = asyncio.Event()
        first_app_ready = asyncio.Event()
        stop_first_app = asyncio.Event()

        def run_analysis(spec, _registries, *, computation):
            loop.call_soon_threadsafe(analysis_started.set)
            assert release_analysis.wait(15), "test did not release running analysis"
            return result(analysis_run_manifest_id(spec, computation=computation))

        def run_figure(spec, _registries):
            figure_calls.append(spec.name)
            return result(analysis_api.figure_manifest_id(spec))

        monkeypatch.setattr(analysis_api, "_run_analysis_sync", run_analysis)
        monkeypatch.setattr(analysis_api, "_run_figure_sync", run_figure)
        first_app, second_app = create_app(), create_app()

        async def serve_first_app() -> None:
            async with first_app.router.lifespan_context(first_app):
                first_app_ready.set()
                await stop_first_app.wait()

        async with (
            second_app.router.lifespan_context(second_app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=first_app), base_url="http://studio"
            ) as first,
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=second_app), base_url="http://studio"
            ) as second,
        ):
            first_lifespan = asyncio.create_task(serve_first_app())
            try:
                await asyncio.wait_for(first_app_ready.wait(), 5)
                running_id = await submit(
                    first,
                    {
                        "node_id": TOY_JOB_ANALYSIS_TYPE,
                        "eval_run_id": "feedbax-evaluation-run:blocked",
                    },
                )
                await asyncio.wait_for(analysis_started.wait(), 5)
                assert (await status(first, running_id))["status"] == "running"

                queued_id = await submit(first, figure_request("queued"))
                assert (await status(first, queued_id))["status"] == "pending"
                rejected = await first.post(jobs_path, json=figure_request("rejected"))
                assert rejected.status_code == 503, rejected.text

                assert (await second.get(f"{jobs_path}/status/{running_id}")).status_code == 404
                independent_id = await submit(second, figure_request("independent"))
                assert (await first.get(f"{jobs_path}/status/{independent_id}")).status_code == 404
                independent = await settle(second, independent_id)
                assert independent["status"] == "complete", independent["error"]
                assert (await status(first, running_id))["status"] == "running"
                assert (await status(first, queued_id))["status"] == "pending"

                stop_first_app.set()
                cancelled = await settle(first, queued_id)
                assert cancelled["status"] == "error"
                assert cancelled["error"]
                assert not first_lifespan.done()
                assert (
                    await first.post(jobs_path, json=figure_request("closing"))
                ).status_code == 503
                assert (await status(first, running_id))["status"] == "running"
            finally:
                release_analysis.set()
                stop_first_app.set()
                await asyncio.wait_for(first_lifespan, 5)

            completed = await status(first, running_id)
            assert completed["status"] == "complete", completed["error"]
            assert (await status(first, queued_id))["status"] == "error"
            assert figure_calls == ["independent"]

    asyncio.run(scenario())


def test_studio_analysis_job_thread_start_failure_closes_admission(
    tmp_path: Path,
    monkeypatch,
    studio_registries,
) -> None:
    monkeypatch.setenv("FEEDBAX_RUNS_DIR", str(tmp_path))
    worker_called = Event()
    start_thread = Thread.start
    payload = {
        "node_id": TOY_JOB_ANALYSIS_TYPE,
        "eval_run_id": "feedbax-evaluation-run:thread-start-failure",
    }

    def fail_analysis_thread_start(thread: Thread) -> None:
        if thread.name.startswith("analysis_"):
            raise RuntimeError("can't start new thread")
        start_thread(thread)

    def run_analysis(_spec, _registries, *, computation):
        worker_called.set()
        raise AssertionError("rejected work must not execute")

    monkeypatch.setattr(analysis_api, "_run_analysis_sync", run_analysis)
    with TestClient(create_app()) as client:
        with monkeypatch.context() as patch:
            patch.setattr(Thread, "start", fail_analysis_thread_start)
            failed = client.post("/api/analyses/jobs", json=payload)
        assert failed.status_code == 503, failed.text

        later = client.post("/api/analyses/jobs", json=payload)
        assert later.status_code == 503, later.text

    assert not worker_called.is_set()


@pytest.mark.parametrize("revision", ["studio-test-v1", None])
def test_studio_analysis_job_routes_eval_run_through_executable_spec(
    tmp_path: Path,
    monkeypatch,
    studio_registries,
    revision: str | None,
) -> None:
    monkeypatch.setenv("FEEDBAX_RUNS_DIR", str(tmp_path))
    _register_job_eval_recipe(studio_registries)
    _register_job_analysis_recipe(studio_registries, revision=revision)
    eval_manifest, _eval_path = _execute_eval(tmp_path, studio_registries)
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
    assert manifest.computation is not None
    assert manifest.id == payload["manifest_id"]
    assert manifest.id == analysis_run_manifest_id(spec, computation=manifest.computation)
    assert manifest.inputs[0].id == eval_manifest.id
    assert manifest.provenance.parents[0].id == eval_manifest.id
    assert manifest.provenance.issues == []
    assert manifest.analysis_spec.inline["params"]["requested_outputs"] == [TOY_JOB_ANALYSIS_TYPE]
    assert manifest.summary_metrics["analysis_count"] == 1

    rerun_manifest, _rerun_path = execute_analysis_run_spec(
        spec,
        registry=studio_registries.analysis_recipes,
        evaluation_registry=studio_registries.evaluation_recipes,
        root=tmp_path,
        fig_dump_formats=("json",),
        experiment_registry=studio_registries.experiment_packages,
    )
    if revision is None:
        assert rerun_manifest.id != manifest.id
    else:
        assert rerun_manifest.id == manifest.id


@pytest.mark.parametrize("outputs", [[UNWIRED_JOB_OUTPUT], None])
def test_unwired_studio_recipe_output_fails_when_requested(
    tmp_path: Path, studio_registries, outputs: list[str] | None
) -> None:
    _register_job_eval_recipe(studio_registries)
    _register_job_analysis_recipe(studio_registries)
    evaluation, _ = _execute_eval(tmp_path, studio_registries)
    spec = AnalysisRunSpec(
        analysis_type=TOY_JOB_ANALYSIS_TYPE,
        inputs=[ParentRef(kind="EvaluationRunManifest", id=evaluation.id, role="evaluation_run")],
        params={"studio": {}, **({"requested_outputs": outputs} if outputs is not None else {})},
    )

    with pytest.raises(AnalysisRecipeExecutionError) as excinfo:
        execute_analysis_run_spec(
            spec,
            registry=studio_registries.analysis_recipes,
            evaluation_registry=studio_registries.evaluation_recipes,
            experiment_registry=studio_registries.experiment_packages,
            root=tmp_path,
        )
    cause = excinfo.value.__cause__
    assert isinstance(cause, ValueError)
    assert "String dependency 'missing_dependency'" in str(cause)


def test_run_analysis_sync_preserves_canonical_uri_and_legacy_path_shape(
    tmp_path: Path,
    monkeypatch,
    application_registry_bundle,
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
    spec = AnalysisRunSpec(analysis_type="test.studio_locations")
    requested_computation = application_registry_bundle.analysis_recipes.new_computation(
        spec.analysis_type
    )

    def execute_stub(
        _spec,
        *,
        registry,
        computation,
        evaluation_registry,
        experiment_registry,
        fig_dump_formats,
    ):
        assert registry is application_registry_bundle.analysis_recipes
        assert computation == requested_computation
        assert evaluation_registry is application_registry_bundle.evaluation_recipes
        assert experiment_registry is application_registry_bundle.experiment_packages
        assert fig_dump_formats == ("json",)
        return manifest, tmp_path / "manifest.json"

    monkeypatch.setattr(analysis_specs, "execute_analysis_run_spec", execute_stub)

    result = _run_analysis_sync(
        spec, application_registry_bundle, computation=requested_computation
    )

    assert canonical.uri == canonical.artifact_id
    assert result.artifact_paths == [canonical.uri, str(Path(legacy.uri))]
    assert all("artifact:/sha256/" not in uri for uri in result.artifact_paths)


def test_studio_analysis_job_executes_require_durable_with_exact_authority(
    tmp_path: Path,
    monkeypatch,
    studio_registries,
) -> None:
    monkeypatch.setenv("FEEDBAX_RUNS_DIR", str(tmp_path))
    _register_job_eval_recipe(studio_registries)
    _register_job_analysis_recipe(studio_registries)
    evaluation, _ = _execute_eval(tmp_path, studio_registries, durable=True)
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


def test_studio_analysis_job_errors_when_requested_node_matches_no_analysis_key(
    tmp_path: Path,
    monkeypatch,
    studio_registries,
) -> None:
    monkeypatch.setenv("FEEDBAX_RUNS_DIR", str(tmp_path))
    _register_job_eval_recipe(studio_registries)
    _register_mismatched_job_analysis_recipe(studio_registries)
    eval_manifest, _eval_path = _execute_eval(tmp_path, studio_registries)

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


def test_studio_analysis_job_requires_eval_run_id() -> None:
    with TestClient(create_app()) as client:
        response = client.post(
            "/api/analyses/jobs",
            json={"node_id": TOY_JOB_ANALYSIS_TYPE},
        )

    assert response.status_code == 400
    assert "eval_run_id is required" in response.json()["detail"]
