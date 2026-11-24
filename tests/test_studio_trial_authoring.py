"""Selected-run authoring uses the same task and objective as local Host."""


from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import AliasChoices, Field, StrictFloat, StrictInt

from feedbax.execution.trial_resolution import evaluation_conditions
from feedbax.contracts.expression_values import LENGTH, CallValue, LiteralValue, NameValue
from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.project import Project
from feedbax.planner.project_plan import ProjectPlanMemo, RunKey, compile_project
from feedbax.contracts.studio_trials import StudioTrialAuthoring
from feedbax.contracts.request_identity import decode_request
from feedbax.application.composition import application_python_tasks
from feedbax.tasks.registry import PythonTaskRegistry
from feedbax.web.app import create_app
from feedbax.web.services import project_plan as window_service
from feedbax.web.services.project_folder import ProjectFolderService
from tests.test_canonical_evaluate_admission import _save_project
from tests.test_document_task_parity import _model_payload
from tests.test_document_train_pair import authored_documents, document_project
from tests.support.host import job_row, token, wait_until
from tests.support.python_task import reaching_python_task  # noqa: F401
from tests.support.plugins.reaching_task import REACHING_TASK, ReachingParameters, reaching_document

pytestmark = pytest.mark.usefixtures("reaching_python_task")


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


def test_revision_bound_task_and_objective_edits_run_on_host(private_host, tmp_path, monkeypatch):
    host = private_host
    monkeypatch.setenv("FEEDBAX_HOST_ROOT", str(host.root))
    app = create_app(projects_dir=tmp_path / "projects")
    with TestClient(app) as client:
        folders = app.state.project_folders
        jobs = app.state.job_service
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
        assert REACHING_TASK in {task.type for task in initial.task_types}
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
        incompatible = author(revision, {"kind": "select_document", "role": "task", "document": "unbound-task"},
                              selected=test_run, scope=scope)
        assert incompatible.files
        revision = save(author(revision, {"kind": "select_document", "role": "task", "document": "test-task"},
                               selected=test_run, scope=scope))
        chosen = author(revision, selected=test_run)
        assert chosen.task_document == "test-task"
        assert chosen.task.params["x"] == 0.31
        revision = save(author(revision, {"kind": "task", "edits": [
            {"op": "replace", "path": "/params/x", "value": 0.38},
        ]}, selected=test_run, scope=scope))
        assert author(revision, selected=test_run).task.params["x"] == 0.38
        assert folders.read(project).project.documents["test-task"].payload["params"]["x"] == 0.31
        revision = save(author(revision, {"kind": "select_document", "role": "task", "document": "probe-task"},
                               selected=test_run, scope=scope))
        probe = author(revision, selected=test_run)
        assert not probe.problems and probe.objective is None
        assert [epoch.id for epoch in probe.epochs] == ["probe"]
        after = ids()
        assert {key for key in before if before[key] != after[key]} == {RunKey("default", "baseline", "seed=test")}

        plan = folders.read_plan(project, revision=revision)
        assert not plan.plan.diagnostics, plan.plan.diagnostics
        sample = client.post("/api/execution/task-trials/sample", json={
            "model": model,
            "python_task": view.task.model_dump(mode="json"),
            "input_overrides": {"reach_input": "reach_target"}, "count": 1,
        })
        assert sample.status_code == 200, sample.text
        np.testing.assert_allclose(sample.json()["trials"][0]["inputs"]["reach_target"], [[0.22, 0]] * 12)

        def submit(selected, name):
            response = client.post(f"/api/projects/{project}/train-runs", json={
                "run_keys": [selected], "revision": revision, "token": token(name),
            })
            assert response.status_code == 202, response.text
            item, = response.json()["items"]
            assert item["disposition"] == "admitted", item
            return item["job_id"]

        trained, control, evaluated = submit(run, "train"), submit(control_run, "control"), submit(test_run, "probe")
        finished = {}
        for job in (trained, control, evaluated):
            row = wait_until(lambda: job_row(host, job, "succeeded") or job_row(host, job, "failed"))
            assert row["state"] == "succeeded", row.get("outcome")
            finished[job] = row
        train_receipt = finished[trained]["outcome"]["store_success"]
        assert all(
            finished[job]["edges"][0]["binding"]["terminal"]["receipt"] == train_receipt
            for job in (control, evaluated)
        )
        with jobs.read_evaluation_result(project=plan.project_id, store=plan.store_path, job_id=control).open_states() as dataset:
            assert dataset.data("trial").tolist() == ["0"]
            assert all(dataset.data(f"objective/coverage/{term}").item() == 1
                       for term in ("effort", "move-effort", "target-variable", "target-signal"))
            position = "/v1/model/state/plant/effector~1position"
            control_positions = dataset.data(position).copy()
            control_effort = dataset.data("objective/terms/effort").item()
        result = jobs.read_evaluation_result(project=plan.project_id, store=plan.store_path, job_id=evaluated)
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
        with jobs.read_evaluation_result(project=plan.project_id, store=plan.store_path, job_id=evaluated).open_states() as dataset:
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
    reaching = application_python_tasks().declaration_record(REACHING_TASK)
    registry = PythonTaskRegistry()
    registry.register_declaration(reaching.defining_object, key="test.reaching", source=reaching.source)
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


def test_saved_window_applies_current_registry_to_canonical_python_selection(tmp_path, monkeypatch):
    class Parameters(ReachingParameters):
        scale: StrictInt | StrictFloat = Field(default=1.0, validation_alias=AliasChoices("gain", "scale"))

    reaching = application_python_tasks().declaration_record(REACHING_TASK)

    def registry(prefix):
        class Scaled(reaching.defining_object):
            settings_model = Parameters

            @staticmethod
            def template(params):
                base = reaching.defining_object.template(params)
                conditions = (f"{prefix}-{type(params.scale).__name__}-{params.scale:.0f}",)

                def prepare(*args):
                    return replace(base.prepare(*args), condition_ids=conditions)

                return replace(base, prepare=prepare, condition_ids=conditions)

        tasks = PythonTaskRegistry()
        tasks.register_declaration(Scaled, key="test.reaching", source=reaching.source)
        tasks.seal()
        return tasks

    compiler_tasks, window_tasks = registry("compiler"), registry("window")
    raw = _python_project()
    raw["documents"]["reach-model"]["payload"] = _model_payload()
    raw["steps"].pop("baseline")
    raw["documents"]["objective"]["payload"]["terms"] = [
        {"id": "command", "measure": {"kind": "literal", "value": 1}},
    ]
    raw["documents"]["task"]["payload"]["type"] = "test.reaching"
    folders = ProjectFolderService(tmp_path, python_tasks=compiler_tasks)

    def read(expected):
        _save_project(folders, "cohort", Project.model_validate(raw))
        settled = folders.read_plan("cohort", revision=None)
        (run,) = settled.plan.runs
        original = decode_request(settled.plan.requests[run.request_id], settled.plan.documents.__getitem__).spec
        assert evaluation_conditions(original.trial_documents.task_source, window_tasks) == (expected,)
        return window_service.ProjectPlanRead.without_history(settled)

    before = read("window-int-1")
    raw["documents"]["task"]["payload"]["params"]["gain"] = 2.0
    after = read("window-int-2")
    monkeypatch.setattr(window_service, "decode_request", lambda *args: pytest.fail("Decoded a compiled request"))
    for view, expected in ((before, "window-int-1"), (after, "window-int-2")):
        (row,) = view.window_map(evaluation_tasks=window_tasks).rows
        assert row.evaluation_conditions == [expected]
        (row,) = view.window_map(evaluation_tasks=PythonTaskRegistry()).rows
        assert row.evaluation_conditions is None


def test_window_gives_unadmitted_train_runs_their_task_conditions(tmp_path):
    raw = _python_project()
    raw["documents"]["reach-model"]["payload"] = _model_payload()
    raw["steps"].pop("baseline")
    tasks = application_python_tasks()
    folders = ProjectFolderService(tmp_path, python_tasks=tasks)

    def row(terms):
        raw["documents"]["objective"]["payload"]["terms"] = terms
        _save_project(folders, "cohort", Project.model_validate(raw))
        view = window_service.ProjectPlanRead.without_history(folders.read_plan("cohort", revision=None))
        (row,) = view.window_map(evaluation_tasks=tasks).rows
        return row

    admitted = row([{"id": "command", "measure": {"kind": "literal", "value": 1}}])
    unadmitted = row([])
    assert admitted.status != "cannot_run"
    assert unadmitted.status == "cannot_run"
    assert admitted.evaluation_conditions is not None
    assert unadmitted.evaluation_conditions == admitted.evaluation_conditions


@pytest.mark.parametrize("kind", ["step", "axis_value", "rows", "variant"])
def test_scoped_objective_writes_preserve_other_requests(tmp_path, kind):
    raw = _python_project()
    raw["documents"]["reach-model"]["payload"] = _model_payload()
    raw["steps"].pop("baseline")
    selected, other = RunKey("default", "train", "default"), RunKey("default", "other", "default")
    scope = {"kind": "step" if kind == "variant" else kind}
    if kind == "step":
        raw["steps"]["other"] = deepcopy(raw["steps"]["train"])
        scope["chosen_steps"] = ["train"]
        raw["documents"]["objective"]["payload"]["terms"] = [
            {"id": "base", "measure": {"kind": "literal", "value": 1}, "weight": {"kind": "literal", "value": 1}},
        ]
    elif kind == "variant":
        raw["sweep"] = {"rows": {"control": {}, "treated": {}}}
        selected, other = RunKey("treated", "train", "default"), RunKey("control", "train", "default")
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
        files, created = prepared.json()["files"], prepared.json()["created"]
        assert len(files) == 1, prepared.text
        assert files[0]["address"]["kind"] == ("variant" if kind == "variant" else "step")
        revision = before.revision
        if kind == "step":
            assert [(file["id"], file["document"]["base"]) for file in created] == [
                ("objective-2", {"document": "objective"}),
            ]
            made = client.post(f"{route}/files", json={"revision": revision, "file": created[0]})
            assert made.status_code == 201, made.text
            revision = folders.read_plan(project, revision=None).revision
        else:
            assert not created
        saved = client.patch(route, json={"revision": revision, "files": files})
        assert saved.status_code == 200, saved.text
        assert client.patch(route, json={"revision": revision, "files": files}).status_code == 409
        current = folders.read_plan(project, revision=saved.json()["revision"])
        assert not current.plan.diagnostics
        request = next(item for item in current.plan.runs if item.key == selected)
        trained = decode_request(current.plan.requests[request.request_id], current.plan.documents.__getitem__).spec
        assert [term.id for term in trained.trial_documents.objective.terms] == ["base" if kind == "step" else "term"]
        assert {item.key for item in current.plan.runs if previous[item.key] != item.request_id} == {selected}
        assert not current.plan.diagnostics
        if kind == "rows":
            expression = body | {
                "revision": saved.json()["revision"],
                "change": {"kind": "expression", "path": "/objective/terms/term/weight", "edit": {"kind": "text", "text": "2"}},
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
                "run": {"variant": other.variant, "step": other.step, "row": other.row},
                "revision": saved.json()["revision"], "scope": {"kind": "step", "chosen_steps": ["other", "train"]},
                "change": {"kind": "objective", "edits": [
                    {"op": "replace", "path": "/terms/0/weight/value", "value": 3},
                ]},
            }
            prepared = client.post(f"{route}/authoring/trials", json=shared)
            assert prepared.status_code == 200, prepared.text
            shared_files = prepared.json()["files"]
            assert [file["address"]["id"] for file in shared_files] == ["objective", "objective-2"]
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


def test_an_objective_edit_for_a_step_leaves_the_steps_based_on_it_on_their_document(tmp_path):
    raw = _python_project()
    raw["documents"]["reach-model"]["payload"] = _model_payload()
    raw["steps"].pop("baseline")
    raw["steps"]["other"] = {"kind": "train", "base": {"step": "train"}}
    raw["steps"]["independent"] = deepcopy(raw["steps"]["train"])
    raw["documents"]["objective"]["payload"]["terms"] = [
        {"id": "base", "measure": {"kind": "literal", "value": 1}, "weight": {"kind": "literal", "value": 1}},
    ]
    train = RunKey("default", "train", "default")
    folders = ProjectFolderService(tmp_path)
    project = "inherited-objective"
    _save_project(folders, project, Project.model_validate(raw))
    route = f"/api/projects/{project}"

    def plan():
        return folders.read_plan(project, revision=None).plan

    with TestClient(create_app(projects_dir=tmp_path)) as client:
        before = {item.key: item.request_id for item in plan().runs}
        response = client.post(f"{route}/authoring/trials", json={
            "run": {"variant": "default", "step": "train", "row": "default"},
            "revision": folders.read_plan(project, revision=None).revision, "scope": {"kind": "step", "chosen_steps": ["train"]},
            "change": {"kind": "objective", "edits": [{"op": "replace", "path": "/terms/0/weight/value", "value": 2}]},
        })
        assert response.status_code == 200, response.text
        (created,) = response.json()["created"]
        made = client.post(f"{route}/files", json={
            "revision": folders.read_plan(project, revision=None).revision, "file": created,
        })
        assert made.status_code == 201, made.text
        saved = client.patch(route, json={
            "revision": folders.read_plan(project, revision=None).revision, "files": response.json()["files"],
        })
        assert saved.status_code == 200, saved.text
        after = {item.key: item.request_id for item in plan().runs}
        assert {key for key in before if before[key] != after[key]} == {train}
        assert plan().document_users("default", "objective-2") == {"train"}


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


def test_an_evaluate_task_choice_in_a_variant_is_its_row_s_task_reference(tmp_path):
    raw = _python_project()
    raw["documents"]["test-task"] = {
        "schema": "feedbax.spec.python_task", "payload": {"type": REACHING_TASK, "params": {"n_steps": 12, "x": 0.31}},
    }
    raw["sweep"] = {"rows": {"control": {}, "treated": {}}}
    folders = ProjectFolderService(tmp_path)
    _save_project(folders, "variants", Project.model_validate(raw))
    before = folders.read_plan("variants", revision=None)
    with TestClient(create_app(projects_dir=tmp_path)) as client:
        reply = client.post("/api/projects/variants/authoring/trials", json={
            "revision": before.revision, "run": {"variant": "treated", "step": "baseline", "row": "default"},
            "change": {"kind": "select_document", "role": "task", "document": "test-task"},
        })
        assert reply.status_code == 200, reply.text
        view = reply.json()
        assert not view["problems"], view["problems"]
        saved = client.patch("/api/projects/variants", json={"revision": before.revision, "files": view["files"]})
        assert saved.status_code == 200, saved.text
    project = folders.read("variants").project
    assert project.steps == Project.model_validate(raw).steps
    assert {path: write.model_dump(mode="json") for path, write in project.sweep.rows["treated"].patches.items()} == {
        "steps.baseline.base.inline.trial_documents": {"op": "add", "value": {"task": {"document": "test-task"}}},
    }
    after = folders.read_plan("variants", revision=None)
    prior = {run.key: run.request_id for run in before.plan.runs}
    changed = {run.key for run in after.plan.runs if prior[run.key] != run.request_id}
    assert changed == {RunKey("treated", "baseline", "default")}
    assert sorted(
        (problem.variant, problem.blocked_by.variant) for problem in after.plan.diagnostics
        if problem.step == "baseline" and problem.code == "blocked_input"
    ) == [("control", "control"), ("treated", "treated")]


@pytest.mark.parametrize("variant", ["default", "treated"])
def test_a_whole_step_task_edit_of_a_plain_evaluate_is_its_own_document(tmp_path, variant):
    from feedbax.contracts.rollout import ROLLOUT_TYPE
    from web.e2e.studio.fixtures.evaluate_authoring import _train_definition

    raw = _train_definition()
    raw["steps"]["check"] = {"kind": "evaluate", "base": {"inline": {"evaluation_type": ROLLOUT_TYPE, "params": {
        "seed": 731, "trial_policy": "paired", "variation": [], "trial_set": {"kind": "every_condition"},
    }}}, "inputs": {"subject": {"step": "train", "checkpoint": "last"}}}
    if variant != "default":
        raw["sweep"] = {"rows": {"control": {}, "treated": {}}}
    folders = ProjectFolderService(tmp_path)
    _save_project(folders, "check", Project.model_validate(raw))
    before = folders.read_plan("check", revision=None)
    run = {"variant": variant, "step": "check", "row": "default"}
    with TestClient(create_app(projects_dir=tmp_path)) as client:
        reply = client.post("/api/projects/check/authoring/trials", json={
            "revision": before.revision, "run": run, "scope": {"kind": "step"},
            "change": {"kind": "task", "edits": [
                {"op": "replace", "path": "/variables/1/value/args/0/value", "value": 0.24},
            ]},
        })
        assert reply.status_code == 200, reply.text
        edited = reply.json()
        assert not edited["problems"], edited["problems"]
        (created,) = edited["created"]
        assert client.post("/api/projects/check/files", json={
            "revision": before.revision, "file": created,
        }).status_code == 201
        saved = client.patch("/api/projects/check", json={"revision": folders.read("check").revision, "files": edited["files"]})
        assert saved.status_code == 200, saved.text
        view = client.post("/api/projects/check/authoring/trials", json={"revision": saved.json()["revision"], "run": run})
        assert view.status_code == 200, view.text
    assert view.json()["task"]["variables"][1]["value"]["args"][0]["value"] == 0.24
    assert view.json()["task_document"] == created["id"]
    trained = {row.key: row for row in before.plan.rows if row.key.step == "train"}
    after = folders.read_plan("check", revision=None).plan
    assert {row.key: row for row in after.rows if row.key.step == "train"}.keys() == trained.keys()
    assert {run.key: run.request_id for run in after.runs if run.key.step == "train"} == {
        run.key: run.request_id for run in before.plan.runs if run.key.step == "train"
    }
    if variant != "default":
        assert folders.read("check").project.steps == Project.model_validate(raw).steps
