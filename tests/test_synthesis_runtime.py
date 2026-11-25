"""A synthesis method fills the shipped graph's declared model parameters."""

import jax
import numpy as np
from fastapi.testclient import TestClient

from feedbax.application.composition import load_application
from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.effective_success import success_content_ref
from feedbax.contracts.model_checkpoint import MODEL_TREE_MEDIA_TYPE, SemanticModelEncoding
from feedbax.contracts.model_origin import ModelProductKey, SynthesizedModelOrigin
from feedbax.contracts.project import Project
from feedbax.contracts.request_identity import decode_request
from feedbax.contracts.request_specs import SynthesisRequestSpec, TrialDocuments
from feedbax.examples import load_shipped_projects
from feedbax.host.synthesis import PreparedSynthesis, synthesize_model
from feedbax.jobs import JobService
from feedbax.persistence.array_container import decode_array_container
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.effective_success import SuccessView
from feedbax.persistence.model_tree import MODEL_CHECKPOINT_STORE, read_model_tree
from feedbax.planner.project_plan import RunKey, compile_project
from feedbax.web.services.project_folder import ProjectFolderService
from feedbax.web.app import create_app
from tests.support.host import job_row, token, wait_until
from tests.support.managed_train_fixture import managed_evaluation_step
from tests.test_canonical_evaluate_admission import _save_project


def synthesis_project():
    raw = load_shipped_projects()["point-mass-reaching"].model_dump(mode="json", by_alias=True, exclude_unset=True)
    evaluate = managed_evaluation_step("synthesize")
    evaluate["inputs"]["subject"].pop("checkpoint")
    raw["steps"] = {
        "synthesize": {"kind": "synthesize", "base": {"inline": {
            "model": {"document": "point-mass"},
            "task": {"document": "simple-reaches-task"},
            "objective": {"document": "simple-reaches-objective"},
            "method": "feedbax.constant_parameters",
        }}},
        "baseline": evaluate,
    }
    return raw


def test_synthesize_and_evaluate_plan_from_authored_documents():
    raw = synthesis_project()
    plan = compile_project(Project.model_validate(raw))
    assert not plan.diagnostics, plan.diagnostics
    runs = {run.kind: run for run in plan.runs}
    assert set(runs) == {"synthesize", "evaluate"}
    assert not any(run.admission_problem for run in runs.values())
    synthesis = runs["synthesize"]
    decoded = decode_request(plan.requests[synthesis.request_id], plan.documents.__getitem__)
    assert isinstance(decoded.spec, SynthesisRequestSpec)
    assert decoded.spec.settings == {"value": 0}
    chosen = runs["evaluate"].inputs["subject"].member.reference
    assert chosen.request_id == synthesis.request_id and chosen.role == "model" and chosen.select is None

    raw["documents"]["point-mass"]["payload"]["interior"]["components"]["plant"]["fields"]["mass"] = 1.5
    changed = compile_project(Project.model_validate(raw))
    before = {run.key: run.request_id for run in plan.runs}
    assert len(changed.runs) == len(before)
    assert all(run.request_id != before[run.key] for run in changed.runs)


def test_served_synthesis_documents_bind_in_trials_and_model_view(host, tmp_path, monkeypatch):
    monkeypatch.setenv("FEEDBAX_HOST_ROOT", str(host.root))
    raw = synthesis_project()
    project = "synthesis-authoring"
    raw["documents"]["extra-model"] = raw["documents"]["point-mass"]
    _save_project(ProjectFolderService(tmp_path), project, Project.model_validate(raw))
    route = f"/api/projects/{project}"
    run = {"variant": "default", "step": "synthesize", "row": "default"}
    with TestClient(create_app(projects_dir=tmp_path, automatic_run="off")) as client:
        revision = client.get(route).json()["revision"]
        body = {"revision": revision, "run": run}
        response = client.post(f"{route}/authoring/trials", json=body)
        assert response.status_code == 200, response.text
        view = response.json()
        assert view["model"] == "point-mass" and view["task_document"] == "simple-reaches-task"
        assert view["objective_document"] == "simple-reaches-objective" and not view["problems"]
        assert view["epoch_instances"]
        response = client.post(f"{route}/model/projection", json=body | {"document": "point-mass"})
        assert response.status_code == 200, response.text
        assert response.json()["task_signals"]
        response = client.get(f"{route}/train-runs", params={"revision": revision, "window_map": True})
        assert response.status_code == 200, response.text
        window = response.json()["window_map"]
        selected = next(row for row in window["rows"] if row["key"] == run)
        assert selected["evaluation_conditions"]
        step = next(step for step in window["steps"] if step["step_id"] == run["step"])
        assert step["document_ids"] == {
            "payload.model": "point-mass", "payload.task": "simple-reaches-task",
            "payload.objective": "simple-reaches-objective",
        }
        for role, document in (
            ("model", "extra-model"), ("task", "delayed-reaches-task"), ("objective", "delayed-reaches-objective"),
        ):
            response = client.post(f"{route}/authoring/trials", json=body | {
                "change": {"kind": "select_document", "role": role, "document": document},
            })
            assert response.status_code == 200, response.text
            files = response.json()["files"]
            assert len(files) == 1 and files[0]["address"] == {"kind": "step", "id": "synthesize"}, response.json()
            assert files[0]["patches"] == [{
                "op": "replace", "path": "/base/inline",
                "value": raw["steps"]["synthesize"]["base"]["inline"] | {role: {"document": document}},
            }]


def test_saved_synthesize_is_admitted_and_owns_a_model_product(host, tmp_path):
    registry = load_application().bundle
    jobs = JobService(host, registry)
    folders = ProjectFolderService(tmp_path / "projects")
    project = "synthesized-reaching"
    _save_project(folders, project, Project.model_validate(synthesis_project()))
    plan = folders.read_plan(project, revision=None)
    synthesis = RunKey("default", "synthesize", "default")
    (admitted,) = jobs.admit_plan(folders, plan, run_keys=(synthesis,), token=token("synthesize")).items
    assert admitted.disposition == "admitted", admitted
    job = wait_until(lambda: (
        job if (job := job_row(host, admitted.job_id)) and job["outcome"] is not None else None
    ))
    assert job["state"] == "succeeded", job["outcome"]
    store = ImmutableArtifactBlobProvider(plan.store_path)
    view = SuccessView(store)
    receipt = view.read_attempt(admitted.job_id).receipt
    assert tuple(receipt.products) == ("model",)
    product, = receipt.products["model"]
    assert product.content.tree and product.content.media_type == MODEL_TREE_MEDIA_TYPE
    assert receipt.request.id.startswith("synthesize:")
    model = read_model_tree(store, success_content_ref(product.content), SemanticModelEncoding.model_validate(product.descriptor))
    replica, = model.replicas
    parameters = decode_array_container(
        store.get_object(replica.arrays), expected_descriptor_sha256=replica.descriptor.array_descriptor_sha256,
    ).tensors
    assert parameters and all(value.dtype == np.float64 for value in parameters.values())

    evaluate = RunKey("default", "baseline", "default")
    (rolled_out,) = jobs.admit_plan(folders, plan, run_keys=(evaluate,), token=token("evaluate-synthesis")).items
    assert rolled_out.disposition == "admitted", rolled_out
    job = wait_until(lambda: (
        job if (job := job_row(host, rolled_out.job_id)) and job["outcome"] is not None else None
    ))
    assert job["state"] == "succeeded", job["outcome"]
    result = jobs.read_evaluation_result(project=plan.project_id, store=plan.store_path, job_id=rolled_out.job_id)
    assert result._manifest().model_origin == SynthesizedModelOrigin(
        kind="synthesized", key=ModelProductKey(variant="default", step="synthesize", row="default"),
        job_id=admitted.job_id, request_id=receipt.request.id,
    )
    with result.open_states() as dataset:
        assert "checkpoint" not in dataset.descriptor.dimensions
        assert dataset.descriptor.dimensions["trial"] > 0


def test_synthesis_writes_one_replica_with_every_parameter(tmp_path):
    project = load_shipped_projects()["point-mass-reaching"]
    spec = SynthesisRequestSpec(
        trial_documents=TrialDocuments.model_validate({
            "model": project.documents["point-mass"].payload,
            "task": project.documents["simple-reaches-task"].payload,
            "objective": project.documents["simple-reaches-objective"].payload,
        }),
        method="feedbax.constant_parameters",
        settings={"value": 0.125},
    )
    registry = load_application().bundle
    artifacts = ImmutableArtifactBlobProvider(tmp_path / "artifacts")
    transaction = tmp_path / "attempt"
    with jax.enable_x64():
        result = synthesize_model(spec, registry=registry, artifacts=artifacts)
        assert isinstance(result, PreparedSynthesis), result
        published = result.writer.publish(transaction, (result.model,), templates=(result.template,), replica_ids=("0",))
    store = ImmutableArtifactBlobProvider(transaction / MODEL_CHECKPOINT_STORE)
    model = read_model_tree(store, published.tree, published.encoding)
    replica, = model.replicas
    assert replica.descriptor.id == "0"
    arrays = decode_array_container(
        store.get_object(replica.arrays), expected_descriptor_sha256=replica.descriptor.array_descriptor_sha256,
    )
    parameters = [
        arrays.tensors[address] for address, metadata in arrays.descriptor.metadata.items()
        if metadata["representation"] == "jax-array"
    ]
    assert parameters
    for array in parameters:
        assert array.dtype == np.float64
        np.testing.assert_equal(array, np.full_like(array, 0.125))

    restricted = spec.model_copy(update={"target": ModelAddress.at_stored_path(("interior", "components", "plant"))})
    with jax.enable_x64():
        problems = synthesize_model(restricted, registry=registry, artifacts=artifacts)
    assert isinstance(problems, tuple) and problems
    assert all(problem.model_address is not None and "parameters" in problem.path for problem in problems)
