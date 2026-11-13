"""Selected-run authoring uses the same task and objective as local Host."""


import numpy as np
from fastapi.testclient import TestClient

from feedbax.contracts.expression_values import CallValue, LiteralValue, NameValue
from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.project import Project
from feedbax.contracts.project_plan import compile_project
from feedbax.contracts.studio_trials import StudioTrialAuthoring
from feedbax.jobs import JobService
from feedbax.contracts.request_identity import decode_request
from feedbax.tasks.reaching import REACHING_TASK
from feedbax.web.app import create_app
from tests.test_canonical_evaluate_admission import _save_project
from tests.test_document_task_parity import _model_payload
from tests.test_document_train_pair import authored_documents, document_project
from tests.test_local_job_host import host, job_row, wait_until  # noqa: F401


def _python_project():
    raw = document_project(authored_documents(), evaluation=True)
    model = raw["documents"]["reach-model"]["payload"]
    raw["documents"]["reach-model"]["payload"] = {"step_seconds": model["step_seconds"], "ports": {}, "interior": {}}
    raw["documents"]["task"] = {
        "schema": "feedbax.spec.python_task",
        "payload": {"type": REACHING_TASK, "params": {}},
    }
    raw["documents"]["objective"]["payload"]["terms"] = []
    documents = raw["steps"]["train"]["base"]["inline"]["trial_documents"]
    documents.pop("input_overrides")
    documents.update(task=None, python_task={"document": "task"})
    return raw


def test_revision_bound_task_and_objective_edits_run_on_host(host, tmp_path):  # noqa: F811
    app = create_app(projects_dir=tmp_path / "projects", enable_evaluation_candidate=True)
    with TestClient(app) as client:
        registry = app.state.bootstrap_state.bundle
        folders = app.state.project_folders
        jobs = JobService(host, registry, enable_evaluation_candidate=True)
        app.state.job_service = jobs
        project = "trial-authoring"
        raw = _python_project()
        _save_project(folders, project, Project.model_validate(raw))
        revision = folders.read(project).revision
        run = {"variant": "default", "step": "train", "row": "default"}

        def author(revision, change=None):
            body = {"run": run, "revision": revision, "change": change}
            response = client.post(f"/api/projects/{project}/authoring/trials", json=body)
            assert response.status_code == 200, response.text
            return StudioTrialAuthoring.model_validate(response.json())

        def save(view):
            response = client.patch(f"/api/projects/{project}", json={
                "revision": view.revision,
                "files": [file.model_dump(mode="json") for file in view.files],
            })
            assert response.status_code == 200, response.text
            return response.json()["revision"]

        initial = author(revision)
        assert initial.task_document == "task"
        assert initial.objective_document == "objective"
        assert initial.task_types[0].type == REACHING_TASK
        assert [(signal.id, signal.name) for signal in initial.signals] == [("reach_target", "reach_target")]
        assert [epoch.id for epoch in initial.epochs] == ["move"]
        assert not initial.inputs
        assert initial.tasks[0].document == "task"
        assert any(field.path == "/python_task/params/n_steps" for field in initial.fields)
        edited = author(revision, {"kind": "task", "edits": [
            {"op": "add", "path": "/params/x", "value": 0.22},
            {"op": "add", "path": "/params/n_steps", "value": 12},
        ]})
        old = revision
        revision = save(edited)
        stale = client.post(f"/api/projects/{project}/authoring/trials", json={"run": run, "revision": old})
        assert stale.status_code == 409, stale.text

        # Install the same authored Model shape that Studio topology commands produce.
        # Model edit mechanics have their own addressed end-to-end coverage.
        saved = folders.read(project).project.model_dump(mode="json", by_alias=True, exclude_unset=True)
        model = _model_payload()
        model["ports"]["inputs"][0]["name"] = "reach_input"
        saved["documents"]["reach-model"]["payload"] = model
        _save_project(folders, project, Project.model_validate(saved))
        revision = folders.read(project).revision
        view = author(revision)
        (port,) = view.inputs
        assert port.signal is None and port.required
        assert any("reach_input" in problem.reason for problem in view.problems)
        assert view.outputs and view.resets
        revision = save(author(revision, {
            "kind": "binding", "input": port.address.model_dump(mode="json"), "signal": "reach_target",
        }))
        view = author(revision)
        assert view.inputs[0].explicit and view.inputs[0].signal == "reach_target"
        assert not view.problems
        (output,) = view.outputs
        assert isinstance(output.address, ModelAddress)
        term = {"id": "effort", "name": "Control effort", "measure": CallValue(
            kind="call", function="sqnorm", args=(NameValue(kind="name", reference=output.value.id),)
        ).model_dump(mode="json"), "weight": LiteralValue(value=2).model_dump(mode="json")}
        revised = author(revision, {"kind": "objective", "edits": [
            {"op": "add", "path": "/terms/-", "value": term},
        ]})
        revision = save(revised)
        view = author(revision)
        assert view.n_steps == 12
        assert view.objective.terms[0].name == "Control effort"
        assert any(field.path == "/objective/terms/effort/measure" for field in view.fields)
        assert not view.problems
        refused = author(revision, {"kind": "objective", "edits": [
            {"op": "replace", "path": "/terms/0/measure/args/0/reference", "value": "unknown"},
        ]})
        assert not refused.files
        assert any(problem.code == "unknown_name" for problem in refused.problems)

        plan = folders.read_plan(project, revision=revision)
        assert not plan.plan.diagnostics, plan.plan.diagnostics
        projection = client.post(f"/api/projects/{project}/authoring/trials", json={
            "run": {**run, "step": "baseline"}, "revision": revision,
        })
        assert projection.status_code == 200, projection.text
        evaluated_task = StudioTrialAuthoring.model_validate(projection.json())
        assert evaluated_task.objective is None
        assert evaluated_task.task.params == {"x": 0.22, "y": 0.0, "n_steps": 12}
        assert [epoch.id for epoch in evaluated_task.epochs] == ["move"]
        sample = client.post("/api/execution/task-trials/sample", json={
            "model_name": "reach-model", "model": model,
            "python_task": view.task.model_dump(mode="json"),
            "input_overrides": {"reach_input": "reach_target"}, "count": 1,
        })
        assert sample.status_code == 200, sample.text
        np.testing.assert_allclose(sample.json()["trials"][0]["inputs"]["reach_target"], [[0.22, 0]] * 12)

        def submit(step):
            response = client.post(f"/api/projects/{project}/train-runs", json={
                "run_keys": [{**run, "step": step}], "revision": revision, "token": step,
            })
            assert response.status_code == 202, response.text
            item, = response.json()["items"]
            assert item["disposition"] == "admitted", item
            return item["job_id"]

        trained, evaluated = submit("train"), submit("baseline")
        for job in (trained, evaluated):
            row = wait_until(lambda: job_row(host, job, "succeeded") or job_row(host, job, "failed"), timeout=24)
            assert row["state"] == "succeeded", row.get("outcome")
        result = jobs.read_evaluation_result(project=project, store=plan.store_path, job_id=evaluated)
        with result.open_states() as dataset:
            assert np.isfinite(dataset.data("objective/total")).all()
            assert int(dataset.data("objective/coverage/effort").item()) == 1


def test_python_task_integer_sweep_uses_registered_site():
    raw = _python_project()
    model = _model_payload()
    raw["documents"]["reach-model"]["payload"] = model
    raw["documents"]["task"]["payload"]["params"]["n_steps"] = 11
    raw["steps"].pop("baseline")
    raw["steps"]["train"]["sweep"] = {"axes": {"length": {
        "path": "trial_documents.python_task.params.n_steps",
        "range": {"start": 11, "stop": 15, "count": 3},
    }}}
    plan = compile_project(Project.model_validate(raw))
    assert not plan.diagnostics, plan.diagnostics
    selected = [
        decode_request(plan.requests[run.request_id], plan.documents.__getitem__).spec
        for run in plan.runs if run.kind == "train"
    ]
    assert [request.trial_documents.python_task.params["n_steps"] for request in selected] == [11, 13, 15]
