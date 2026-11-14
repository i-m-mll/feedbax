"""Selected-run authoring uses the same task and objective as local Host."""


from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from feedbax.contracts.expression_values import LENGTH, CallValue, LiteralValue, NameValue
from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.project import Project
from feedbax.contracts.project_plan import ProjectPlanMemo, RunKey, compile_project
from feedbax.contracts.studio_trials import StudioTrialAuthoring
from feedbax.jobs import JobService
from feedbax.contracts.request_identity import decode_request
from feedbax.tasks.reaching import REACHING_TASK, ReachingParameters, builtin_python_tasks, reaching_document
from feedbax.tasks.registry import PythonTaskRegistry
from feedbax.web.app import create_app
from feedbax.web.services.project_folder import ProjectFolderService
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
    raw["steps"]["baseline"]["base"]["inline"]["params"]["trial_set"] = {"kind": "sample", "count": 1}
    return raw


def test_revision_bound_task_and_objective_edits_run_on_host(host, tmp_path):  # noqa: F811
    app = create_app(projects_dir=tmp_path / "projects")
    with TestClient(app) as client:
        registry = app.state.bootstrap_state.bundle
        folders = app.state.project_folders
        jobs = JobService(host, registry)
        app.state.job_service = jobs
        project = "trial-authoring"
        raw = _python_project()
        raw["documents"]["test-task"] = {
            "schema": "feedbax.spec.python_task",
            "payload": {"type": REACHING_TASK, "params": {"n_steps": 12, "x": 0.31}},
        }
        task = reaching_document(ReachingParameters(n_steps=12, x=0.22)).model_dump(mode="json")
        task["epochs"][0]["id"] = "probe"
        task["variables"][0].update(id="probe_target", name="probe_target")
        task["signals"][0]["cells"][0].update(
            first_epoch="probe", last_epoch="probe", value={"kind": "name", "reference": "probe_target"},
        )
        raw["documents"]["probe-task"] = {"schema": "feedbax.spec.task_document", "payload": task}
        invalid_task = deepcopy(task)
        invalid_task["signals"] = []
        raw["documents"]["unbound-task"] = {"schema": "feedbax.spec.task_document", "payload": invalid_task}
        raw["steps"]["baseline"]["base"]["inline"]["params"]["variation"] = {}
        raw["steps"]["baseline"]["sweep"] = {"axes": {"seed": {"values": {
            "control": {"patches": {"params.seed": {"value": 1}}},
            "test": {"patches": {"params.seed": {"value": 1}}},
        }}}}
        _save_project(folders, project, Project.model_validate(raw))
        revision = folders.read(project).revision
        run = {"variant": "default", "step": "train", "row": "default"}

        def author(revision, change=None, *, selected=run, scope=None):
            body = {"run": selected, "revision": revision, "change": change, "scope": scope}
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
        assert {choice.document for choice in initial.tasks} == {"probe-task", "task", "test-task", "unbound-task"}
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
        windowed = term | {"id": "move-effort", "epochs": ["move"]}

        def squared_target(identifier, source):
            return {
                "id": identifier,
                "measure": CallValue(
                    kind="call", function="sqnorm", args=(NameValue(kind="name", reference=source),),
                ).model_dump(mode="json"),
                "measure_type": {"base": "real", "dimensions": LENGTH.raised(2).model_dump(mode="json")},
            }
        revised = author(revision, {"kind": "objective", "edits": [
            {"op": "add", "path": "/terms/-", "value": value}
            for value in (term, windowed, squared_target("target-variable", "target"), squared_target("target-signal", "reach_target"))
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

        def ids():
            return {item.key: item.request_id for item in folders.read_plan(project, revision=revision).plan.runs}

        before = ids()
        test_run = {**run, "step": "baseline", "row": "seed=test"}
        control_run = {**test_run, "row": "seed=control"}
        evaluated_task = author(revision, selected=control_run)
        assert evaluated_task.objective is None
        assert evaluated_task.task.params == {"x": 0.22, "y": 0.0, "n_steps": 12}
        assert [epoch.id for epoch in evaluated_task.epochs] == ["move"]
        scope = {"kind": "rows", "rows": ["seed=test"]}
        choices = {choice.document: choice.problems for choice in evaluated_task.tasks}
        assert not choices["probe-task"]
        assert any("reach_target" in problem.reason for problem in choices["unbound-task"])
        incompatible = author(revision, {"kind": "select_task", "document": "unbound-task"},
                              selected=test_run, scope=scope)
        assert not incompatible.files
        assert any("reach_target" in problem.reason for problem in incompatible.problems)
        revision = save(author(revision, {"kind": "select_task", "document": "test-task"},
                               selected=test_run, scope=scope))
        chosen = author(revision, selected=test_run)
        assert chosen.task_document == "test-task"
        assert chosen.task.params["x"] == 0.31
        revision = save(author(revision, {"kind": "task", "edits": [
            {"op": "replace", "path": "/params/x", "value": 0.38},
        ]}, selected=test_run, scope=scope))
        assert author(revision, selected=test_run).task.params["x"] == 0.38
        assert folders.read(project).project.documents["test-task"].payload["params"]["x"] == 0.31
        revision = save(author(revision, {"kind": "select_task", "document": "probe-task"},
                               selected=test_run, scope=scope))
        probe = author(revision, selected=test_run)
        assert not probe.problems and probe.objective is None
        assert [epoch.id for epoch in probe.epochs] == ["probe"]
        after = ids()
        assert {key for key in before if before[key] != after[key]} == {RunKey("default", "baseline", "seed=test")}

        plan = folders.read_plan(project, revision=revision)
        assert not plan.plan.diagnostics, plan.plan.diagnostics
        sample = client.post("/api/execution/task-trials/sample", json={
            "model_name": "reach-model", "model": model,
            "python_task": view.task.model_dump(mode="json"),
            "input_overrides": {"reach_input": "reach_target"}, "count": 1,
        })
        assert sample.status_code == 200, sample.text
        np.testing.assert_allclose(sample.json()["trials"][0]["inputs"]["reach_target"], [[0.22, 0]] * 12)

        def submit(selected, token):
            response = client.post(f"/api/projects/{project}/train-runs", json={
                "run_keys": [selected], "revision": revision, "token": token,
            })
            assert response.status_code == 202, response.text
            item, = response.json()["items"]
            assert item["disposition"] == "admitted", item
            return item["job_id"]

        trained, control, evaluated = submit(run, "train"), submit(control_run, "control"), submit(test_run, "probe")
        finished = {}
        for job in (trained, control, evaluated):
            row = wait_until(lambda: job_row(host, job, "succeeded") or job_row(host, job, "failed"), timeout=24)
            assert row["state"] == "succeeded", row.get("outcome")
            finished[job] = row
        checkpoint = finished[trained]["outcome"]["completion"]
        assert all(finished[job]["edges"][0]["binding"]["completion"] == checkpoint for job in (control, evaluated))
        with jobs.read_evaluation_result(project=project, store=plan.store_path, job_id=control).open_states() as dataset:
            assert dataset.data("trial").tolist() == ["0"]
            assert all(dataset.data(f"objective/coverage/{term}").item() == 1
                       for term in ("effort", "move-effort", "target-variable", "target-signal"))
            position = "/v1/model/state/plant/effector~1position"
            control_positions = dataset.data(position).copy()
            control_effort = dataset.data("objective/terms/effort").item()
        result = jobs.read_evaluation_result(project=project, store=plan.store_path, job_id=evaluated)
        with result.open_states() as dataset:
            assert dataset.data("trial").tolist() == ["0"]
            for key in ("effort", "target-signal"):
                assert dataset.data(f"objective/coverage/{key}").item() == 1
            for key in ("move-effort", "target-variable"):
                assert dataset.data(f"objective/coverage/{key}").item() == 0
                assert dataset.data(f"objective/terms/{key}").item() == 0
            np.testing.assert_allclose(dataset.data("/v1/task/output/reach_target")[..., 0], 0.22)
            np.testing.assert_allclose(dataset.data("objective/terms/target-signal"), 0.22**2)
            np.testing.assert_array_equal(dataset.data(position), control_positions)
            np.testing.assert_allclose(dataset.data("objective/terms/effort"), control_effort)
            np.testing.assert_allclose(
                dataset.data("objective/total"),
                dataset.data("objective/terms/effort") + dataset.data("objective/terms/target-signal"),
            )

        revision = save(author(revision, {"kind": "objective", "edits": [
            {"op": "replace", "path": "/terms/0/name", "value": "Effort"},
        ]}))
        assert author(revision).objective.terms[0].name == "Effort"
        assert ids() == after
        with jobs.read_evaluation_result(project=project, store=plan.store_path, job_id=evaluated).open_states() as dataset:
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


def test_memoized_plan_uses_supplied_python_tasks():
    raw = _python_project()
    raw["documents"]["reach-model"]["payload"] = _model_payload()
    raw["steps"].pop("baseline")
    declaration = builtin_python_tasks().resolve(REACHING_TASK)
    assert declaration is not None
    registry = PythonTaskRegistry()
    registry.register(replace(declaration, key="test.reaching"))
    registry.seal()
    raw["documents"]["task"]["payload"]["type"] = "test.reaching"
    project = Project.model_validate(raw)
    memo = ProjectPlanMemo()
    expected = compile_project(project, python_tasks=registry)
    actual = compile_project(project, memo=memo, python_tasks=registry)
    assert not actual.diagnostics
    assert len(actual.runs) == 1
    assert actual == expected
    assert compile_project(project, memo=memo).diagnostics[0].path == "payload.trial_documents.python_task.type"


@pytest.mark.parametrize("kind", ["step", "axis_value", "rows", "variant"])
def test_scoped_objective_writes_preserve_other_requests(tmp_path, kind):
    raw = _python_project()
    raw["documents"]["reach-model"]["payload"] = _model_payload()
    raw["steps"].pop("baseline")
    selected, other = RunKey("default", "train", "default"), RunKey("default", "other", "default")
    scope = {"kind": kind}
    if kind == "step":
        raw["steps"]["other"] = {"kind": "train", "base": {"step": "train"}}
        scope["lit_steps"] = ["train"]
        raw["documents"]["objective"]["payload"]["terms"] = [
            {"id": "base", "measure": {"kind": "literal", "value": 1}, "weight": {"kind": "literal", "value": 1}},
        ]
    elif kind == "variant":
        raw["sweep"] = {"rows": {"control": {}, "treated": {}}}
        selected, other = RunKey("treated", "train", "default"), RunKey("control", "train", "default")
        scope["variant"] = "treated"
    else:
        raw["steps"]["train"]["sweep"] = {"axes": {"length": {"values": {
            "short": {"patches": {"trial_documents.python_task.params.n_steps": {"op": "add", "value": 11}}},
            "long": {"patches": {"trial_documents.python_task.params.n_steps": {"op": "add", "value": 13}}},
        }}}}
        selected, other = RunKey("default", "train", "length=short"), RunKey("default", "train", "length=long")
        scope.update({"axis": "length", "value": "short"} if kind == "axis_value" else {"rows": [selected.row]})
    folders = ProjectFolderService(tmp_path)
    project = "scoped-objective"
    _save_project(folders, project, Project.model_validate(raw))
    before = folders.read_plan(project, revision=None)
    previous = {item.key: item.request_id for item in before.plan.runs}
    assert {selected, other} <= previous.keys(), before.plan.diagnostics
    body = {
        "run": {"variant": selected.variant, "step": selected.step, "row": selected.row},
        "revision": before.revision, "scope": scope,
        "change": {"kind": "objective", "edits": [
            {"op": "replace", "path": "/terms/0/weight/value", "value": 2},
        ] if kind == "step" else [
            {"op": "add", "path": "/terms/-", "value": {"id": "term", "measure": {"kind": "literal", "value": 1}}},
        ]},
    }
    with TestClient(create_app(projects_dir=tmp_path)) as client:
        route = f"/api/projects/{project}"
        prepared = client.post(f"{route}/authoring/trials", json=body)
        assert prepared.status_code == 200, prepared.text
        files = prepared.json()["files"]
        assert len(files) == 1, prepared.text
        assert files[0]["address"]["kind"] == ("variant" if kind == "variant" else "step")
        saved = client.patch(route, json={"revision": before.revision, "files": files})
        assert saved.status_code == 200, saved.text
        assert client.patch(route, json={"revision": before.revision, "files": files}).status_code == 409
        current = folders.read_plan(project, revision=saved.json()["revision"])
        assert not current.plan.diagnostics
        request = next(item for item in current.plan.runs if item.key == selected)
        trained = decode_request(current.plan.requests[request.request_id], current.plan.documents.__getitem__).spec
        assert [term.id for term in trained.trial_documents.objective.terms] == ["base" if kind == "step" else "term"]
        assert {item.key for item in current.plan.runs if previous[item.key] != item.request_id} == {selected}
        if kind == "rows":
            expression = body | {
                "revision": saved.json()["revision"],
                "change": {"kind": "expression", "term": "term", "field": "weight", "text": "2"},
            }
            prepared = client.post(f"{route}/authoring/trials", json=expression)
            assert prepared.status_code == 200, prepared.text
            files = prepared.json()["files"]
            assert len(files) == 1, prepared.text
            applied = client.patch(route, json={"revision": expression["revision"], "files": files})
            assert applied.status_code == 200, applied.text
            updated = folders.read_plan(project, revision=applied.json()["revision"]).plan
            edited = next(item for item in updated.runs if item.key == selected)
            spec = decode_request(updated.requests[edited.request_id], updated.documents.__getitem__).spec
            assert spec.trial_documents.objective.terms[0].weight == LiteralValue(value=2)
            assert next(item for item in updated.runs if item.key == other).request_id == previous[other]
        if kind == "step":
            shared = body | {
                "revision": saved.json()["revision"], "scope": {"kind": "step", "lit_steps": ["train", "other"]},
                "change": {"kind": "objective", "edits": [
                    {"op": "replace", "path": "/terms/0/weight/value", "value": 3},
                ]},
            }
            prepared = client.post(f"{route}/authoring/trials", json=shared)
            assert prepared.status_code == 200, prepared.text
            shared_files = prepared.json()["files"]
            assert [file["address"]["kind"] for file in shared_files] == ["document", "step"]
            proposed = client.patch(route, json={"revision": shared["revision"], "files": shared_files})
            assert proposed.status_code == 200, proposed.text
            weights = {}
            for key in (selected, other):
                view = client.post(f"{route}/authoring/trials", json={
                    "revision": proposed.json()["revision"],
                    "run": {"variant": key.variant, "step": key.step, "row": key.row},
                })
                weights[key] = view.json()["objective"]["terms"][0]["weight"]["value"]
            assert weights == {selected: 3, other: 3}


def test_scoped_rename_refuses_a_name_change_in_an_unselected_inheritor(tmp_path):
    raw = _python_project()
    raw["documents"]["reach-model"]["payload"] = _model_payload()
    raw["steps"].pop("baseline")
    raw["steps"]["other"] = {"kind": "train", "base": {"step": "train"}}
    raw["documents"]["objective"]["payload"]["terms"] = [
        {"id": "base", "name": "Base", "measure": {"kind": "literal", "value": 1}},
    ]
    raw["steps"]["train"]["sweep"] = {"axes": {"length": {"values": {
        "short": {"patches": {"trial_documents.python_task.params.n_steps": {"op": "add", "value": 11}}},
        "long": {"patches": {"trial_documents.python_task.params.n_steps": {"op": "add", "value": 13}}},
    }}}}
    folders = ProjectFolderService(tmp_path)
    _save_project(folders, "scoped-name", Project.model_validate(raw))
    revision = folders.read("scoped-name").revision
    with TestClient(create_app(projects_dir=tmp_path)) as client:
        refused = client.post("/api/projects/scoped-name/authoring/trials", json={
            "revision": revision, "run": {"variant": "default", "step": "train", "row": "length=short"},
            "scope": {"kind": "axis_value", "axis": "length", "value": "short"},
            "change": {"kind": "objective", "edits": [
                {"op": "replace", "path": "/terms/0/name", "value": "Renamed"},
            ]},
        })
    assert refused.status_code == 422, refused.text
    assert "outside the selected scope" in refused.json()["detail"]
    assert folders.read("scoped-name").revision == revision
