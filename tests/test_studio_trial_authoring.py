"""Selected-run authoring uses the same task and objective as local Host."""


from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import AliasChoices, Field, StrictFloat, StrictInt

from feedbax.execution.trial_resolution import evaluation_conditions
from feedbax.contracts.expression_values import LENGTH, CallValue, LiteralValue, NameValue, UnresolvedValue
from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.project import Project
from feedbax.planner.project_plan import ProjectPlanMemo, RunKey, compile_project
from feedbax.contracts.studio_trials import StudioTrialAuthoring
from feedbax.contracts.request_identity import decode_request
from feedbax.application.composition import application_python_tasks
from feedbax.examples import load_shipped_projects
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
        "label": "Task",
        "schema": "feedbax.spec.python_task",
        "payload": {"type": REACHING_TASK, "params": {}},
    }
    raw["documents"]["objective"]["payload"]["terms"] = []
    documents = raw["steps"]["train"]["base"]["inline"]["trial_documents"]
    documents.pop("input_bindings")
    documents.update(task=None, python_task={"document": "task"})
    raw["steps"]["baseline"]["base"]["inline"]["params"]["trial_set"] = {"kind": "sample", "count": 1}
    return raw


@pytest.mark.parametrize("competing_label", [False, True])
def test_task_pairing_prefers_the_matching_signal_id(tmp_path, competing_label):
    project = load_shipped_projects()["point-mass-reaching"]
    original = compile_project(project)
    raw = project.model_dump(mode="json", by_alias=True, exclude_unset=True)
    signals = raw["documents"]["simple-reaches-task"]["payload"]["signals"]
    next(signal for signal in signals if signal["id"] == "target_pos")["name"] = "target position"
    if competing_label:
        next(signal for signal in signals if signal["id"] == "target_vel")["name"] = "target_pos"
    renamed = Project.model_validate(raw)
    run = RunKey("default", "simple-reaches", "default")
    with TestClient(create_app(projects_dir=tmp_path, automatic_run="off")) as client:
        folders = client.app.state.project_folders
        _save_project(folders, "implicit-binding", renamed)
        snapshot = folders.read("implicit-binding")
        response = client.post("/api/projects/implicit-binding/authoring/trials", json={
            "revision": snapshot.revision,
            "run": {"variant": run.variant, "step": run.step, "row": run.row},
        })
        assert response.status_code == 200, response.text
        view = StudioTrialAuthoring.model_validate(response.json())
        target = next(port for port in view.inputs if port.id == "target_pos")
        assert (target.signal_id, target.explicit) == ("target_pos", False)
        assert not view.problems
        assert folders.read_plan("implicit-binding", revision=None).plan.row(run).request_id == original.row(run).request_id


@pytest.mark.parametrize("rename", ["signal", "root input"])
def test_renaming_keeps_each_delayed_reaches_pairing(tmp_path, rename):
    """Catches a rename that drops a signal, changes a varied rollout's signal, or feeds an idle input."""
    import jax.random as jr

    from feedbax.execution.trial_resolution import pair_trial_source
    from feedbax.tasks.document import TaskParameters

    functions, *_ = authored = authored_documents()
    raw = document_project(authored, evaluation=True)
    training = raw["steps"]["train"]["base"]["inline"]["trial_documents"]
    del training["input_bindings"]
    model = raw["documents"]["reach-model"]["payload"]
    root = model["ports"]["inputs"][0]
    root["name"] = "target"
    raw["documents"]["idle-model"] = deepcopy(raw["documents"]["reach-model"])
    raw["documents"]["idle-model"]["payload"]["ports"]["inputs"].append(root | {
        "id": "idle-input", "name": "idle", "default": {"value": [0.0, 0.0]},
    })
    raw["steps"]["optional"] = deepcopy(raw["steps"]["train"])
    raw["steps"]["optional"]["base"]["inline"]["trial_documents"]["model"] = {"document": "idle-model"}
    task = raw["documents"]["task"]["payload"]
    next(variable for variable in task["variables"] if variable["id"] == "target")["name"] = "requested"
    for signal in task["signals"]:
        if signal["id"] == "target_pos":
            signal["name"] = "target"
        elif signal["id"] == "goal":
            signal["name"] = "spare"
    params = raw["steps"]["baseline"]["base"]["inline"]["params"]
    params["variation"] = [
        {"op": "replace", "path": "task.signals.target_pos.name", "value": "legacy"},
        {"op": "replace", "path": "task.signals.goal.name", "value": "target"},
    ]
    app = create_app(projects_dir=tmp_path)
    with TestClient(app) as client:
        folders = app.state.project_folders
        bundle = app.state.bootstrap_state.bundle
        _save_project(folders, "rename", Project.model_validate(raw))
        url = "/api/projects/rename"
        train = {"variant": "default", "step": "train", "row": "default"}
        baseline = train | {"step": "baseline"}
        optional = train | {"step": "optional"}

        def pairing(run):
            snapshot = folders.read("rename")
            response = client.post(url + "/authoring/trials", json={"revision": snapshot.revision, "run": run})
            assert response.status_code == 200, response.text
            view = StudioTrialAuthoring.model_validate(response.json())
            assert not view.problems, view.problems
            model = snapshot.project.documents[view.model].payload
            paired = pair_trial_source(
                type(authored[1]).model_validate(model), view.task, bundle.components,
                functions, python_tasks=bundle.python_tasks, input_bindings=view.input_bindings,
            )
            assert not isinstance(paired, tuple) and not paired.problems, paired
            trial = paired.task.ahead(TaskParameters(), jr.key(1), jr.key(2))
            fed, = [trial.inputs[signal] for _, signal in paired.input_bindings]
            return view, fed

        def signal_name(id_, name):
            snapshot = folders.read("rename")
            request = {
                "revision": snapshot.revision, "run": train,
                "change": {"kind": "structure", "changes": [
                    {"op": "rename", "member": "signal", "id": id_, "name": name},
                ]},
            }
            response = client.post(url + "/authoring/trials", json=request)
            assert response.status_code == 200, response.text
            if choices := response.json()["store_choices"]:
                shared = next(choice for choice in choices if choice["kind"] == "all_readers")
                response = client.post(url + "/authoring/trials", json=request | {"store_choice": shared["id"]})
                assert response.status_code == 200, response.text
            saved = client.patch(url, json={
                "revision": snapshot.revision,
                "files": response.json()["files"],
            })
            assert saved.status_code == 200, saved.text

        _, original_train = pairing(train)
        _, original_baseline = pairing(baseline)
        _, original_optional = pairing(optional)
        assert not np.array_equal(original_train, original_baseline)
        train_key = RunKey("default", "train", "default")
        original_request = folders.read_plan("rename", revision=None).plan.row(train_key).request_id
        assert original_request is not None

        if rename == "signal":
            signal_name("target_pos", "goal")
            assert folders.read_plan("rename", revision=None).plan.row(train_key).request_id == original_request
        else:
            snapshot = folders.read("rename")
            response = client.post(url + "/model/edit", json={
                "revision": snapshot.revision, "presentation_revision": snapshot.presentation_revision,
                "document": "reach-model", "command": {"operation": {
                    "op": "rename", "name": "goal", "address": {"composites": [], "element": {
                        "kind": "port", "end": {"owner": "inputs", "port": "task-input"},
                    }},
                }},
            })
            assert response.status_code == 200 and response.json()["result"]["status"] == "applied", response.text

        shown_train, trained = pairing(train)
        shown_baseline, evaluated = pairing(baseline)
        assert shown_train.input_bindings["task-input"] == "target_pos"
        assert shown_baseline.input_bindings["task-input"] == "goal"
        np.testing.assert_array_equal(trained, original_train)
        np.testing.assert_array_equal(evaluated, original_baseline)

        signal_name("goal", "idle")
        shown_optional, optional_inputs = pairing(optional)
        assert shown_optional.input_bindings["idle-input"] is None
        assert next(row for row in shown_optional.inputs if row.id == "idle-input").signal_id is None
        np.testing.assert_array_equal(optional_inputs, original_optional)
        updated = folders.read_plan("rename", revision=None).plan
        assert not updated.diagnostics, [(problem.path, problem.detail) for problem in updated.diagnostics]


def test_root_rename_keeps_different_bindings_in_the_same_variant(tmp_path):
    """Catches a variant-wide repair that replaces distinct signals selected by individual rows."""
    raw = document_project(authored_documents())
    train = raw["steps"]["train"]
    del train["base"]["inline"]["trial_documents"]["input_bindings"]
    train["sweep"] = {"axes": {"context": {"values": {
        "left": {},
        "right": {"patches": {
            "documents.task.payload.signals.target_pos.name": {"value": "spare"},
            "documents.task.payload.signals.goal.name": {"value": "reach_target"},
        }},
    }}}}
    signals = raw["documents"]["task"]["payload"]["signals"]
    next(signal for signal in signals if signal["id"] == "target_pos")["name"] = "reach_target"
    next(signal for signal in signals if signal["id"] == "goal")["name"] = "spare"
    raw["sweep"] = {"rows": {"control": {}, "treated": {"patches": {
        "documents.reach-model.payload.ports.inputs.task-input.name": {"value": "spare"},
    }}}}

    app = create_app(projects_dir=tmp_path)
    with TestClient(app) as client:
        folders = app.state.project_folders
        _save_project(folders, "rename-variant", Project.model_validate(raw))
        url = "/api/projects/rename-variant"
        snapshot = folders.read("rename-variant")

        def selections(revision):
            result = {}
            for variant in ("control", "treated"):
                for row in ("context=left", "context=right"):
                    shown = client.post(url + "/authoring/trials", json={
                        "revision": revision, "run": {"variant": variant, "step": "train", "row": row},
                    })
                    assert shown.status_code == 200, shown.text
                    view = StudioTrialAuthoring.model_validate(shown.json())
                    assert not view.problems, view.problems
                    result[variant, row] = view.inputs[0].signal_id
            return result

        before = selections(snapshot.revision)
        assert before == {
            ("control", "context=left"): "target_pos", ("control", "context=right"): "goal",
            ("treated", "context=left"): "goal", ("treated", "context=right"): "target_pos",
        }
        response = client.post(url + "/model/edit", json={
            "revision": snapshot.revision, "presentation_revision": snapshot.presentation_revision,
            "document": "reach-model", "command": {"operation": {
                "op": "rename", "name": "destination", "address": {"composites": [], "element": {
                    "kind": "port", "end": {"owner": "inputs", "port": "task-input"},
                }},
            }},
        })
        assert response.status_code == 200 and response.json()["result"]["status"] == "applied", response.text
        assert selections(folders.read("rename-variant").revision) == before


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
            "label": "Test task",
            "schema": "feedbax.spec.python_task",
            "payload": {"type": REACHING_TASK, "params": {"n_steps": 12, "x": 0.31}},
        }
        task = reaching_document(ReachingParameters(n_steps=12, x=0.22)).model_dump(mode="json")
        task["epochs"][0]["id"] = "probe"
        task["variables"][0].update(id="probe_target", name="probe_target")
        task["signals"][0]["cells"][0].update(
            first_epoch="probe", last_epoch="probe", value={"kind": "name", "reference": "probe_target"},
        )
        raw["documents"]["probe-task"] = {"label": "Probe task", "schema": "feedbax.spec.task_document", "payload": task}
        invalid_task = deepcopy(task)
        invalid_task["signals"] = []
        raw["documents"]["unbound-task"] = {"label": "Unbound task", "schema": "feedbax.spec.task_document", "payload": invalid_task}
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
            {"op": "add", "path": "/params", "value": {"x": 0.22, "n_steps": 12}},
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
        assert port.signal_id is None and port.required
        assert any("reach_input" in problem.reason for problem in view.problems)
        assert view.outputs and view.resets
        revision = save(author(revision, {
            "kind": "binding", "input_id": port.id, "signal_id": "reach_target",
        }))
        view = author(revision)
        assert view.inputs[0].explicit and view.inputs[0].signal_id == "reach_target"
        assert not view.problems
        (output,) = view.outputs
        assert isinstance(output.address, ModelAddress)
        term = {"id": "effort", "name": "Control effort", "measure": CallValue(
            kind="call", function="sqnorm", args=(NameValue(kind="name", reference=output.value.id),)
        ).model_dump(mode="json"), "weight": LiteralValue(value=2).model_dump(mode="json")}
        windowed = term | {"id": "move-effort", "window": {"spans": [{"kind": "epoch", "epoch": "move"}]}}

        def squared_target(identifier, source):
            return {
                "id": identifier,
                "measure": CallValue(
                    kind="call", function="sqnorm", args=(NameValue(kind="name", reference=source),),
                ).model_dump(mode="json"),
                "measure_type": {"base": "real", "dimensions": LENGTH.raised(2).model_dump(mode="json")},
            }
        revised = author(revision, {"kind": "objective", "edits": [{
            "op": "add", "path": "/terms",
            "value": [term, windowed, squared_target("target-variable", "target"), squared_target("target-signal", "reach_target")],
        }]})
        revision = save(revised)
        view = author(revision)
        assert view.n_steps == 12
        assert view.objective.terms[0].name == "Control effort"
        assert any(field.path == "/objective/terms/effort/measure" for field in view.fields)
        assert not view.problems
        marked = author(revision, {"kind": "objective", "edits": [
            {"op": "replace", "path": "/terms/0/measure/args/0/reference", "value": "unknown"},
        ]})
        assert marked.files
        assert any(problem.code == "unknown_name" for problem in marked.problems)

        def ids():
            return {item.key: item.request_id for item in folders.read_plan(project, revision=revision).plan.runs}

        before = ids()
        test_run = {**run, "step": "baseline", "row": "seed=test"}
        control_run = {**test_run, "row": "seed=control"}
        evaluated_task = author(revision, selected=control_run)
        assert [term.id for term in evaluated_task.objective.terms] == [
            "effort", "move-effort", "target-variable", "target-signal",
        ]
        assert evaluated_task.objective_source.model_dump() == {"kind": "train", "step": "train"}
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
        assert not probe.problems
        assert probe.objective_source.model_dump() == {"kind": "train", "step": "train"}
        assert [epoch.id for epoch in probe.epochs] == ["probe"]
        after = ids()
        assert {key for key in before if before[key] != after[key]} == {RunKey("default", "baseline", "seed=test")}

        plan = folders.read_plan(project, revision=revision)
        assert not plan.plan.diagnostics, plan.plan.diagnostics
        sample = client.post("/api/execution/task-trials/sample", json={
            "model": model,
            "python_task": view.task.model_dump(mode="json"),
            "input_bindings": {"task-input": "reach_target"}, "count": 1,
        })
        assert sample.status_code == 200, sample.text
        np.testing.assert_allclose(sample.json()["drawn"]["trials"][0]["inputs"]["reach_target"], [[0.22, 0]] * 12)

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


def test_task_saves_unresolved_text_and_binds_it_when_a_variable_appears(tmp_path):
    app = create_app(projects_dir=tmp_path / "projects")
    raw = document_project(authored_documents())
    task_file = raw["documents"]["task"]["payload"]
    target = next(variable for variable in task_file["variables"] if variable["id"] == "target")
    terms = raw["documents"]["objective"]["payload"]["terms"]
    terms.append(terms[0] | {"id": "disabled", "enabled": False})
    project = "held-expression"
    folders = app.state.project_folders
    _save_project(folders, project, Project.model_validate(raw))
    revision = folders.read(project).revision
    run = {"variant": "default", "step": "train", "row": "default"}
    path = "/task/variables/target/value"

    with TestClient(app) as client:
        def author(change=None):
            response = client.post(f"/api/projects/{project}/authoring/trials", json={
                "revision": revision, "run": run, "change": change,
            })
            assert response.status_code == 200, response.text
            return StudioTrialAuthoring.model_validate(response.json())

        def save(view):
            response = client.patch(f"/api/projects/{project}", json={
                "revision": view.revision,
                "files": [file.model_dump(mode="json") for file in view.files],
            })
            assert response.status_code == 200, response.text
            return response.json()["revision"]

        def target_field(view):
            return next(field for field in view.fields if field.path == path)

        assert not author().problems
        disabled_path = "/objective/terms/disabled/weight"
        disabled = author({"kind": "expression", "path": disabled_path, "edit": {
            "kind": "text", "text": "(",
        }})
        disabled_field = next(field for field in disabled.fields if field.path == disabled_path)
        assert disabled.files and disabled_field.text == "("
        (disabled_problem,) = disabled_field.problems
        assert disabled_problem.code == "parse" and disabled_problem in disabled.problems
        edit = author({"kind": "expression", "path": path, "edit": {
            "kind": "text", "text": "start + offset",
        }})
        assert edit.files
        revision = save(edit)
        reopened = author()
        for view in (edit, reopened):
            field = target_field(view)
            problem, = field.problems
            assert field.text == "start + offset"
            assert problem.code == "unknown_name"
            assert field.text[problem.span.start:problem.span.end] == "offset"
        assert isinstance(
            next(variable.value for variable in reopened.task.variables if variable.id == "target"), UnresolvedValue,
        )
        (unrunnable,) = folders.read_plan(project, revision=revision).plan.runs
        assert unrunnable.admission_problem is not None
        assert "offset" in unrunnable.admission_problem.detail

        added = author({"kind": "task", "edits": [{"op": "add", "path": "/variables/-", "value": {
            "id": "offset-id", "name": "offset", "type": target["type"],
            "value": {"kind": "call", "function": "vector", "args": [
                {"kind": "literal", "value": 0.1}, {"kind": "literal", "value": 0.2},
            ]},
        }}]})
        assert not added.problems
        revision = save(added)
        resolved = next(variable.value for variable in author().task.variables if variable.id == "target")
        assert isinstance(resolved, CallValue)
        assert resolved.args[1] == NameValue(reference="offset-id")
        rename = author({"kind": "structure", "changes": [{
            "op": "rename", "member": "variable", "id": "offset-id", "name": "displacement",
        }]})
        revision = save(rename)
        renamed = author()
        assert not renamed.problems
        assert target_field(renamed).text == "start + displacement"
        assert next(variable.value for variable in renamed.task.variables if variable.id == "target") == resolved


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
    (unresolved,) = compile_project(project, memo=memo).runs
    assert unresolved.admission_problem is not None
    assert unresolved.admission_problem.path == "/trial_documents/python_task/type"


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
            {"op": "add", "path": "/terms/0/weight", "value": {"kind": "literal", "value": 2}},
        ] if kind == "step" else [
            {"op": "add", "path": "/terms", "value": [{"id": "term", "measure": {"kind": "literal", "value": 1}}]},
        ]},
    }
    with TestClient(create_app(projects_dir=tmp_path)) as client:
        route = f"/api/projects/{project}"
        prepared = client.post(f"{route}/authoring/trials", json=body)
        assert prepared.status_code == 200, prepared.text
        if kind == "step":
            choice = next(option for option in prepared.json()["store_choices"] if option["kind"] == "selected_readers")
            prepared = client.post(f"{route}/authoring/trials", json=body | {"store_choice": choice["id"]})
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
                "change": {
                    "kind": "expression", "path": "/objective/terms/base/weight",
                    "edit": {"kind": "text", "text": "3"},
                },
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


def test_trial_authoring_refuses_a_row_removed_by_a_saved_sweep(tmp_path):
    raw = document_project(authored_documents())
    raw["steps"]["train"]["sweep"] = {"axes": {
        "seed": {"values": {"one": {"patches": {"seed": {"value": 1}}}}},
    }}
    app = create_app(projects_dir=tmp_path)
    with TestClient(app) as client:
        folders = app.state.project_folders
        _save_project(folders, "reach", Project.model_validate(raw))
        snapshot = folders.read("reach")
        response = client.post("/api/projects/reach/authoring/trials", json={
            "revision": snapshot.revision,
            "run": {"variant": "default", "step": "train", "row": "default"},
        })
        assert response.status_code == 422, response.text
        assert response.json()["detail"] == "The selected run no longer belongs to this step."


def test_trial_authoring_preserves_field_presence_when_editing_objective_terms(tmp_path):
    raw = document_project(authored_documents())
    task = raw["documents"]["task"]["payload"]
    del task["fixed_steps"]
    objective = raw["documents"]["objective"]["payload"]
    term = objective["terms"][0]
    for field in ("enabled", "reduction", "window"):
        del term[field]
    folders = ProjectFolderService(tmp_path)
    project = "term-defaults"
    _save_project(folders, project, Project.model_validate(raw))
    route = f"/api/projects/{project}"
    body = {
        "revision": folders.read(project).revision,
        "run": {"variant": "default", "step": "train", "row": "default"},
    }

    with TestClient(create_app(projects_dir=tmp_path)) as client:
        def author(**change):
            response = client.post(f"{route}/authoring/trials", json=body | change)
            assert response.status_code == 200, response.text
            return response.json()

        view = author()
        assert "fixed_steps" not in view["task"]
        stored = folders.read(project).project.documents["objective"].payload
        assert view["objective"] == stored
        assert not {"enabled", "reduction", "window", "weight", "measure_type"} & stored["terms"][0].keys()
        for op, values in (
            ("add", {"enabled": False, "reduction": "sum", "window": {"spans": [{"kind": "epoch", "epoch": "move"}]}}),
            ("replace", {"enabled": True, "reduction": "mean", "window": None}),
        ):
            shown = view["objective"]["terms"][0]
            edits = [
                {"op": "replace" if field in shown else "add", "path": f"/terms/0/{field}", "value": value}
                for field, value in values.items()
            ]
            assert {edit["op"] for edit in edits} == {op}
            proposed = author(change={"kind": "objective", "edits": edits})
            saved = client.patch(route, json={"revision": body["revision"], "files": proposed["files"]})
            assert saved.status_code == 200, saved.text
            body["revision"] = saved.json()["revision"]
            view = author()
            expected = stored["terms"][0] | values if op == "add" else stored["terms"][0]
            assert view["objective"]["terms"][0] == expected
            assert folders.read(project).project.documents["objective"].payload["terms"][0] == expected


def test_studio_saves_an_objective_ramp_weighted_by_a_delayed_signal(tmp_path):
    project = "objective-ramp"
    folders = ProjectFolderService(tmp_path)
    _save_project(folders, project, Project.model_validate(document_project(authored_documents())))
    route = f"/api/projects/{project}"
    body = {
        "revision": folders.read(project).revision,
        "run": {"variant": "default", "step": "train", "row": "default"},
    }
    path = "/objective/terms/effort/weight"
    text = "(t / T) ** 2 * sqnorm(lag(command, 5))"
    with TestClient(create_app(projects_dir=tmp_path)) as client:
        edited = client.post(f"{route}/authoring/trials", json=body | {
            "change": {"kind": "expression", "path": path, "edit": {"kind": "text", "text": text}},
        })
        assert edited.status_code == 200, edited.text
        proposed = edited.json()
        assert proposed["files"]
        assert not proposed["problems"], proposed["problems"]
        saved = client.patch(route, json={"revision": body["revision"], "files": proposed["files"]})
        assert saved.status_code == 200, saved.text
        reread = client.post(f"{route}/authoring/trials", json=body | {"revision": saved.json()["revision"]})
        assert reread.status_code == 200, reread.text
        assert not reread.json()["problems"], reread.text
        field = next(field for field in reread.json()["fields"] if field["path"] == path)
        assert field["text"] == text


def test_trial_authoring_shows_authored_fields_for_a_windowed_set_entry(tmp_path):
    from feedbax.desktop.release import AppPaths
    from web.e2e.studio.fixtures.set_authoring import seed_window

    project = seed_window(str(tmp_path), "windowed-set")
    folders = ProjectFolderService(AppPaths.resolve(instance_root=tmp_path).web / "projects")
    snapshot = folders.read(project)
    plan = folders.read_plan(project, revision=snapshot.revision).plan
    assert not plan.diagnostics
    selected = next(row for row in plan.rows if row.coordinates["pulse"].value_id == "force")
    body = {
        "revision": snapshot.revision,
        "run": {"variant": selected.key.variant, "step": selected.key.step, "row": selected.key.row},
    }
    with TestClient(create_app(projects_dir=folders.root)) as client:
        response = client.post(f"/api/projects/{project}/authoring/trials", json=body)
        sample = client.post("/api/execution/task-trials/sample", json=body | {"project_id": project, "nominal": True})
        assert sample.status_code == 200, sample.text
        trial, = sample.json()["drawn"]["trials"]
        window, = trial["intervention_windows"].values()
        assert [index for index, active in enumerate(window) if active] == list(range(10, 15))
    assert response.status_code == 200, response.text
    view = response.json()
    authored_signals = {signal["id"] for signal in view["task"]["signals"]}
    signal_fields = [field["path"].split("/")[3] for field in view["fields"]
                     if field["path"].startswith("/task/signals/")]
    assert signal_fields and set(signal_fields) <= authored_signals
    assert view["epochs"]
    assert any(variable["name"].endswith("Onset") for variable in view["variables"])


def test_objective_offsets_project_paired_editable_fields(tmp_path):
    raw = document_project(authored_documents())
    term = raw["documents"]["objective"]["payload"]["terms"][0]
    term["window"] = {"spans": [{
        "kind": "interval",
        "start": {
            "anchor": {"kind": "epoch", "epoch": "move", "edge": "start"},
        },
        "end": {"offset": {"value": 5, "unit": "st"}},
    }]}
    task = raw["documents"]["task"]["payload"]
    task["variables"].append({
        "id": "onset", "name": "onset", "type": {"base": "real", "dimensions": {"time": 1}},
        "value": {"kind": "literal", "value": .1, "unit": "s"},
    })
    folders = ProjectFolderService(tmp_path)
    _save_project(folders, "window", Project.model_validate(raw))
    route = "/api/projects/window"
    body = {
        "revision": folders.read("window").revision,
        "run": {"variant": "default", "step": "train", "row": "default"},
    }
    start = "/objective/terms/effort/window/spans/0/start/offset"
    end = "/objective/terms/effort/window/spans/0/end/offset"

    with TestClient(create_app(projects_dir=tmp_path)) as client:
        def author(change=None):
            response = client.post(f"{route}/authoring/trials", json=body | ({"change": change} if change else {}))
            assert response.status_code == 200, response.text
            return response.json()

        def edit(path, text):
            answer = author({"kind": "expression", "path": path, "edit": {"kind": "text", "text": text}})
            assert not answer["problems"], answer["problems"]
            response = client.patch(route, json={"revision": body["revision"], "files": answer["files"]})
            assert response.status_code == 200, response.text
            body["revision"] = response.json()["revision"]
            return answer

        def field(view, path):
            return next(item for item in view["fields"] if item["path"] == path)

        shown = author()
        assert field(shown, start)["text"] == ""
        assert field(shown, start)["conversion"] is None
        assert field(shown, end)["conversion"] == {"value": .05, "unit": "s"}
        assert field(shown, start)["can_sweep"]
        assert {"value", "formula", "link", "draw", "sweep"} <= set(field(shown, start)["site"]["kinds"])

        edit(start, "-25 ms")
        reopened = author()
        start_offset = reopened["objective"]["terms"][0]["window"]["spans"][0]["start"]["offset"]
        assert start_offset == {"value": -.025, "unit": "s"}
        assert field(reopened, start)["conversion"] == {"value": -3, "unit": "st"}

        edit(start, "onset")
        link = author()["objective"]["terms"][0]["window"]["spans"][0]["start"]["offset"]
        assert link == {"kind": "name", "reference": "onset"}

        edit(start, "")
        cleared = author()
        assert "offset" not in cleared["objective"]["terms"][0]["window"]["spans"][0]["start"]
        assert field(cleared, start)["text"] == ""

        edit(end, "linspace(3 st, 5 st, 3)")
        runs = folders.read_plan("window", revision=body["revision"]).plan.runs
        assert len(runs) == 3 and all(run.admission_problem is None for run in runs)
        assert len({run.request_id for run in runs}) == 3


def test_shared_objective_rename_asks_before_writing_and_can_copy_for_one_reader(tmp_path):
    raw = _python_project()
    raw["documents"]["reach-model"]["payload"] = _model_payload()
    raw["steps"].pop("baseline")
    raw["steps"]["other"] = {"label": "Other", "kind": "train", "base": {"step": "train"}}
    raw["documents"]["objective"]["label"] = "Simple reaches Objective"
    raw["documents"]["objective"]["payload"]["terms"] = [
        {"id": "base", "name": "Before", "measure": {"kind": "literal", "value": 1}},
    ]
    folders = ProjectFolderService(tmp_path)
    project = "inherited-objective"
    _save_project(folders, project, Project.model_validate(raw))
    route = f"/api/projects/{project}"
    revision = folders.read(project).revision
    body = {
        "run": {"variant": "default", "step": "train", "row": "default"},
        "revision": revision,
        "change": {"kind": "objective", "edits": [{"op": "replace", "path": "/terms/0/name", "value": "After"}]},
    }

    with TestClient(create_app(projects_dir=tmp_path)) as client:
        answer = client.post(f"{route}/authoring/trials", json=body)
        assert answer.status_code == 200, answer.text
        asked = answer.json()
        assert asked["files"] == asked["created"] == []
        assert folders.read(project).revision == revision
        all_steps, only_train = asked["store_choices"]
        assert (all_steps["kind"], all_steps["label"], all_steps["steps"]) == (
            "all_readers", "All 2 Steps", ["other", "train"],
        )
        assert all_steps["runs"] == [body["run"] | {"step": "other"}, body["run"]]
        assert (only_train["kind"], only_train["label"], only_train["runs"]) == (
            "selected_readers", "Only This Step", [body["run"]],
        )
        altered = body | {
            "store_choice": only_train["id"],
            "change": {"kind": "objective", "edits": [{"op": "replace", "path": "/terms/0/name", "value": "Wrong"}]},
        }
        assert client.post(f"{route}/authoring/trials", json=altered).status_code == 422
        assert folders.read(project).revision == revision
        replay = body | {"store_choice": only_train["id"]}
        response = client.post(f"{route}/authoring/trials", json=replay)
        assert response.status_code == 200, response.text
        (created,) = response.json()["created"]
        assert created["document"]["base"] == {"document": "objective"}
        assert created["document"]["label"] == "Simple reaches Objective 2"
        batch = {"revision": revision, "created": [created], "files": response.json()["files"]}
        bad_file = batch["files"][0] | {
            "patches": [{"op": "replace", "path": "/missing", "value": "bad"}],
        }
        failed = client.patch(route, json=batch | {"files": [bad_file]})
        assert failed.status_code == 422, failed.text
        assert folders.read(project).revision == revision
        assert created["id"] not in folders.read(project).project.documents
        saved = client.patch(route, json=batch)
        assert saved.status_code == 200, saved.text
        assert client.post(f"{route}/authoring/trials", json=replay).status_code == 409
        after = folders.read_plan(project, revision=saved.json()["revision"])
        assert after.plan.document_users("default", created["id"]) == {"train"}
        for step, name in (("train", "After"), ("other", "Before")):
            view = client.post(f"{route}/authoring/trials", json={
                "revision": after.revision, "run": body["run"] | {"step": step},
            })
            assert view.status_code == 200, view.text
            assert view.json()["objective"]["terms"][0]["name"] == name


def test_whole_step_weight_sweep_writes_without_asking_about_its_new_rows(tmp_path):
    raw = _python_project()
    raw["steps"].pop("baseline")
    raw["documents"]["objective"]["payload"]["terms"] = [
        {"id": "effort", "name": "Effort", "measure": {"kind": "literal", "value": 1}},
    ]
    folders = ProjectFolderService(tmp_path)
    _save_project(folders, "weight", Project.model_validate(raw))
    with TestClient(create_app(projects_dir=tmp_path)) as client:
        route = "/api/projects/weight"
        revision = folders.read("weight").revision
        response = client.post(f"{route}/authoring/trials", json={
            "revision": revision, "run": {"variant": "default", "step": "train", "row": "default"},
            "scope": {"kind": "step"},
            "change": {
                "kind": "expression", "path": "/objective/terms/effort/weight",
                "edit": {"kind": "text", "text": "0.1, 1, 10"},
            },
        })
        assert response.status_code == 200, response.text
        answer = response.json()
        assert answer["files"] and not answer.get("store_choices"), answer
        saved = client.patch(route, json={"revision": revision, "files": answer["files"]})
        assert saved.status_code == 200, saved.text
        plan = folders.read_plan("weight", revision=saved.json()["revision"]).plan
        assert len([row for row in plan.rows if row.key.step == "train"]) == 3


def test_scoped_rename_asks_before_changing_an_unselected_inheritor(tmp_path):
    raw = _python_project()
    raw["documents"]["reach-model"]["payload"] = _model_payload()
    raw["steps"].pop("baseline")
    raw["steps"]["other"] = {"label": "Other", "kind": "train", "base": {"step": "train"}}
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
    assert refused.status_code == 200, refused.text
    response = refused.json()
    assert response["files"] == []
    sweep = next(choice for choice in response["store_choices"] if choice["kind"] == "sweep_value")
    assert {(run["step"], run["row"]) for run in sweep["runs"]} == {
        ("train", "length=short"), ("other", "length=short"),
    }
    assert folders.read("scoped-name").revision == revision


def test_an_evaluate_task_choice_in_a_variant_is_its_row_s_task_reference(tmp_path):
    raw = _python_project()
    raw["documents"]["test-task"] = {
        "label": "Test task",
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
    raw["steps"]["check"] = {"label": "Check", "kind": "evaluate", "base": {"inline": {"evaluation_type": ROLLOUT_TYPE, "params": {
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


def test_initial_model_evaluate_uses_and_edits_its_own_objective(tmp_path):
    """The baseline must score its own Objective, and an edit must reach that document."""
    from feedbax.contracts.rollout import ROLLOUT_TYPE

    app = create_app(projects_dir=tmp_path / "projects")
    raw = document_project(authored_documents())
    raw["steps"] = {
        "evaluate": {
            "label": "Evaluate",
            "kind": "evaluate",
            "base": {"inline": {"evaluation_type": ROLLOUT_TYPE, "params": {
                "seed": 0, "trial_policy": "paired", "variation": [],
                "trial_set": {"kind": "every_condition"},
                "input_bindings": {"task-input": "target_pos"},
            }}},
            "inputs": {
                "subject": {"kind": "initial_model", "source": {"document": "reach-model"}},
                "task": {"kind": "task", "source": {"document": "task"}},
                "objective": {"kind": "objective", "source": {"document": "objective"}},
            },
        },
    }
    raw["steps"]["unscored"] = deepcopy(raw["steps"]["evaluate"])
    del raw["steps"]["unscored"]["inputs"]["objective"]

    with TestClient(app) as client:
        folders = app.state.project_folders
        project = "initial-evaluate"
        _save_project(folders, project, Project.model_validate(raw))
        revision = folders.read(project).revision
        planned = folders.read_plan(project, revision=revision).plan
        hand = "/v1/model/state/plant/effector~1position"
        for run in planned.runs:
            port = run.dataset_port
            assert port.variables[hand].dims == ("condition", "trial", "state_time", "xy")
            assert "checkpoint" not in port.coordinates and "update" not in port.coordinates

        def author(step, change=None):
            response = client.post(f"/api/projects/{project}/authoring/trials", json={
                "revision": revision, "run": {"variant": "default", "step": step, "row": "default"},
                "change": change,
            })
            assert response.status_code == 200, response.text
            return response.json()

        view = author("evaluate")
        assert view["objective_document"] == "objective"
        assert view["objective_source"] == {"kind": "own", "step": "evaluate"}
        assert view["objective"]["terms"][0]["id"] == "effort"
        assert any(field["path"] == "/objective/terms/effort/measure" for field in view["fields"])
        assert author("unscored")["objective"] is None
        assert author("unscored")["objective_source"] is None
        changed = author("evaluate", {"kind": "objective", "edits": [
            {"op": "add", "path": "/terms/0/name", "value": "Command effort"},
        ]})
        response = client.patch(f"/api/projects/{project}", json={"revision": revision, "files": changed["files"]})
        assert response.status_code == 200, response.text
        revision = response.json()["revision"]
        assert author("evaluate")["objective"]["terms"][0]["name"] == "Command effort"


@pytest.mark.parametrize(("condition", "boundary", "outcome", "weight"), [
    ("true", 0, 2, 2.0),
    ("held(dist(effector.pos, point(0.01 m, 0 m)) < 0.00999 m, 0.1 s)", 12, 2, 2.0),
    ("held(dist(effector.pos, point(1 m, 0 m)) < 0.00999 m, 0.1 s)", 14, 3, 3.0),
])
def test_until_scene_ends_at_entry_after_holding_or_at_limit(tmp_path, condition, boundary, outcome, weight):
    """Euler moves the hand on the second transition; a ten-step hold ends at boundary 12."""
    model = {
        "step_seconds": 0.01,
        "ports": {
            "inputs": [{"id": "force", "name": "force", "type": {"shape": [2], "unit": "N"}}],
            "outputs": [{"id": "effector", "name": "effector", "type": {"base": "state"}}],
        },
        "interior": {
            "components": {"plant": {"kind": "leaf", "name": "Plant", "type": "feedbax.PointMass"}},
            "wires": {
                "force": {
                    "source": {"owner": "inputs", "port": "force"},
                    "target": {"owner": "component", "component": "plant", "port": "force"},
                },
                "effector": {
                    "source": {"owner": "component", "component": "plant", "port": "effector"},
                    "target": {"owner": "outputs", "port": "effector"},
                },
            },
        },
    }

    def expression(text):
        return {"kind": "unresolved", "text": text}

    task = {
        "id": "reach",
        "variables": (),
        "epochs": [
            {"id": "reach", "name": "Reach", "end": {
                "kind": "until",
                "condition": expression(condition),
                "limit": {"value": 0.14, "unit": "s"},
            }},
            {"id": "exit", "name": "Exit", "end": {
                "kind": "length", "length": {"kind": "literal", "value": 2, "unit": "st"},
            }},
        ],
        "signals": [{"id": "force", "name": "force", "type": {
            "base": "real", "shape": [2], "dimensions": {"mass": 1, "length": 1, "time": -2},
        }, "cells": [
            {"id": "push", "first_epoch": "reach", "last_epoch": "reach", "value": expression("vector(1 N, 0 N)")},
        ]}],
    }
    objective = {
        "id": "score",
        "terms": [{
            "id": "ending",
            "measure": expression("1."),
            "weight": expression("where(epoch.Reach.condition, 2, where(epoch.Reach.limit, 3, 0))"),
        }],
    }
    app = create_app(projects_dir=tmp_path / "projects")
    with TestClient(app) as client:
        response = client.post("/api/execution/task-trials/sample", json={
            "model": model, "task": task, "objective": objective, "nominal": True,
        })
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["drawn"]["kind"] == "trials", data["drawn"]
    (trial,) = data["drawn"]["trials"]
    assert [(cue["start"], cue["end"], cue["outcome"]) for cue in trial["timeline"]] == [
        (0, boundary, outcome), (boundary, boundary + 2, 1),
    ]
    assert trial["present"]["force"] == [True] * boundary + [False] * (16 - boundary)
    assert trial["terms"]["ending"]["weights"] == [weight] * 16


def test_until_authoring_switches_and_refuses_a_limit_that_rounds_to_zero(tmp_path):
    app = create_app(projects_dir=tmp_path / "projects")
    raw = document_project(authored_documents())
    project = "until-expression"
    folders = app.state.project_folders
    _save_project(folders, project, Project.model_validate(raw))
    revision = folders.read(project).revision
    run = {"variant": "default", "step": "train", "row": "default"}
    name = raw["documents"]["task"]["payload"]["epochs"][0]["id"]
    end = f"/task/epochs/{name}/end"
    with TestClient(app) as client:
        def author(change=None):
            response = client.post(f"/api/projects/{project}/authoring/trials", json={
                "revision": revision, "run": run, "change": change,
            })
            assert response.status_code == 200, response.text
            return StudioTrialAuthoring.model_validate(response.json())

        def change(field, text):
            return author({"kind": "expression", "path": f"{end}/{field}", "edit": {
                "kind": "text", "text": text,
            }})

        def assert_empty_condition(result, kind):
            assert result.task.epochs[0].end.kind == kind
            fields = {field.path: field for field in result.fields}
            assert f"{end}/length" in fields
            condition = fields[f"{end}/condition"]
            assert condition.text == ""
            assert condition.site.site.expected.base == "boolean"
            assert {value.name for value in condition.site.site.names} >= {"command", "time", "t"}

        assert_empty_condition(author(), "length")
        offered = change("length", "true ≤ 0.3 s")
        assert offered.files and not offered.problems
        assert offered.task.epochs[0].end.kind == "until"
        assert offered.epochs[0].steps.least == 0
        assert offered.epochs[0].steps.greatest == offered.epochs[0].steps.nominal == 30
        fields = {field.path: field for field in offered.fields}
        assert f"{end}/condition" in fields
        assert fields[f"{end}/limit"].conversion.model_dump() == {"value": 30, "unit": "st"}
        response = client.patch(f"/api/projects/{project}", json={
            "revision": revision, "files": [file.model_dump(mode="json") for file in offered.files],
        })
        assert response.status_code == 200, response.text
        revision = response.json()["revision"]
        refused = change("limit", "0.001 s")
        assert not refused.files
        assert any(problem.path == ("task", "epochs", name, "end", "limit") for problem in refused.problems)
        restored = change("length", "20 st")
        assert restored.files and not restored.problems
        assert restored.task.epochs[0].end.kind == "length"
        assert_empty_condition(change("length", "rest"), "rest")


def test_expression_menu_reuses_the_exact_projected_pairing(tmp_path, monkeypatch):
    from feedbax.contracts.expression_site import project_site

    app = create_app(projects_dir=tmp_path)
    project = "expression-menu"
    _save_project(app.state.project_folders, project, Project.model_validate(document_project(authored_documents())))
    url = f"/api/projects/{project}/authoring/trials"
    run = {"variant": "default", "step": "train", "row": "default"}
    pointer = "/task/variables/target/value"

    with TestClient(app) as client:
        folders = app.state.project_folders
        revision = folders.read(project).revision
        request = {"revision": revision, "run": run}
        authored = client.post(url, json=request)
        assert authored.status_code == 200, authored.text
        view = StudioTrialAuthoring.model_validate(authored.json())
        assert all(not field.site.parts for field in view.fields)

        edit = client.post(url, json=request | {"change": {
            "kind": "expression", "path": pointer, "edit": {"kind": "text", "text": "start"},
        }})
        assert edit.status_code == 200, edit.text
        preview = StudioTrialAuthoring.model_validate(edit.json())
        assert preview.files and preview.revision == revision
        assert preview.expression_context != view.expression_context

        def forbid_plan(*args, **kwargs):
            pytest.fail("Opening an expression menu attempted to prepare another trial pairing.")

        monkeypatch.setattr(folders, "read_plan", forbid_plan)
        field = next(field for field in view.fields if field.path == pointer)
        expression = next(variable.value for variable in view.task.variables if variable.id == "target")
        functions = app.state.bootstrap_state.bundle.expression_functions
        expected = project_site(field.site.site, functions, expression).parts

        def menu(**updates):
            return client.post(url + "/expression-menu", json=(request | {
                "expression_context": view.expression_context, "field": pointer, "part": ["args", 0],
            } | updates))

        response = menu()
        assert response.status_code == 200, response.text
        assert response.json() == next(part.model_dump(mode="json") for part in expected if part.path == ("args", 0))
        assert menu(expression_context=preview.expression_context).status_code == 422
        assert menu(part=["args", 999]).status_code == 422
        assert menu(field="/unknown").status_code == 422
        expired = menu(run=run | {"step": "other"})
        assert expired.status_code == 409
        assert expired.json() == {"detail": {
            "code": "trial_expression_context_unavailable",
            "message": "Reload trial authoring before requesting another expression menu.",
        }}
        stale = menu(revision="0" * 64)
        assert stale.status_code == 409 and stale.json()["detail"]["current"] == revision
