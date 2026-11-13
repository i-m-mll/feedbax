"""Selected-run authoring uses the same task and objective as local Host."""


import numpy as np
from fastapi.testclient import TestClient

from feedbax.contracts.expression_values import CallValue, LiteralValue, NameValue
from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.project import Project
from feedbax.contracts.project_plan import RunKey
from feedbax.contracts.studio_trials import StudioTrialAuthoring
from feedbax.jobs import JobService
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
        train = RunKey("default", "train", "default")
        evaluate = RunKey("default", "baseline", "default")
        (trained,) = jobs.admit_plan(folders, plan, run_keys=(train,), token="train").items
        assert trained.disposition == "admitted", trained
        (evaluated,) = jobs.admit_plan(folders, plan, run_keys=(evaluate,), token="evaluate").items
        assert evaluated.disposition == "admitted", evaluated
        for job in (trained, evaluated):
            row = wait_until(lambda: job_row(host, job.job_id, "succeeded") or job_row(host, job.job_id, "failed"),
                             timeout=24)
            assert row["state"] == "succeeded", row.get("outcome")
        result = jobs.read_evaluation_result(project=project, store=plan.store_path, job_id=evaluated.job_id)
        with result.open_states() as dataset:
            assert np.isfinite(dataset.data("objective/total")).all()
            assert int(dataset.data("objective/coverage/effort").item()) == 1
