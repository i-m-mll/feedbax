"""An Evaluate step's own fields take axes addressed as a Train's are."""

import numpy as np
import pytest

from feedbax.application.composition import load_application
from feedbax.contracts.project import Project
from feedbax.contracts.request_identity import decode_request
from feedbax.jobs import JobService
from feedbax.planner.project_plan import compile_project
from feedbax.web.services.project_folder import ProjectFolderService
from tests.support.host import job_row, token, wait_until
from tests.support.python_task import REACHING_TASK, reaching_python_task  # noqa: F401
from tests.test_canonical_evaluate_admission import _save_project, managed_evaluation_definition

pytestmark = pytest.mark.usefixtures("reaching_python_task")

X = "trial_documents.python_task.params.x"


def _axis(path, values, *, op="replace"):
    return {"values": {f"v{index}": {"value": value, "patches": {path: {"op": op, "value": value}}}
                       for index, value in enumerate(values)}}


def _project(seeds, axes):
    raw = managed_evaluation_definition()
    train = raw["steps"]["train"]
    train["base"]["inline"]["trial_documents"].update(task=None, python_task={"document": "task"}, input_bindings={})
    raw["documents"]["task"] = {
        "label": "Task",
        "schema": "feedbax.spec.python_task",
        "payload": {"type": REACHING_TASK, "params": {"x": 0.1, "n_steps": 5}},
    }
    (term,) = raw["documents"]["objective"]["payload"]["terms"].values()
    term.pop("window")
    if seeds:
        train["sweep"] = {"axes": {"seed": _axis("seed", seeds, op="add")}}
    evaluate = raw["steps"]["baseline"]
    evaluate["sweep"] = {"axes": {"trained": {"input": "subject"}, **axes}}
    params = evaluate["base"]["inline"]["params"]
    params["trial_set"] = {"kind": "sample", "count": 1}
    params["variation"] = {}
    raw["steps"] = {"train": train, "baseline": evaluate}
    return Project.model_validate(raw)


def _compile(project):
    bundle = load_application().bundle
    return compile_project(
        project, components=bundle.components, functions=bundle.expression_functions, python_tasks=bundle.python_tasks,
    )


def test_task_and_trials_axes_cross_the_subject_runs():
    """Catches an Evaluate axis on its Task or Trials that is dropped, refused, or not crossed with its subjects."""
    plan = _compile(_project([0, 1, 2], {
        "x": _axis(X, [0.2, 0.3]),
        "count": _axis("params.trial_set.count", [1, 2, 3]),
    }))
    assert plan.diagnostics == ()
    runs = [run for run in plan.runs if run.kind == "evaluate"]
    authored = set()
    for run in runs:
        spec = decode_request(plan.requests[run.request_id], plan.documents.__getitem__).spec
        (edit,) = spec.params["variation"].values()
        assert edit["path"] == "task.params.x"
        authored.add((run.inputs["subject"].producer, edit["value"], len(spec.params["trials"])))
    assert len(runs) == len({run.request_id for run in runs}) == 18
    assert {(x, count) for _, x, count in authored} == {(x, n) for x in (0.2, 0.3) for n in (1, 2, 3)}
    assert len({producer for producer, _, _ in authored}) == 3


@pytest.mark.parametrize("path", ["inputs.subject", "inputs.subject.where"])
def test_an_axis_on_an_input_is_refused_at_its_path(path):
    """Catches an axis on the Subject or its Runs rule that plans, or is refused somewhere the author cannot find."""
    plan = _compile(_project([], {"pick": _axis(path, [{"step": "train"}, {"step": "train"}])}))
    assert not any(run.kind == "evaluate" for run in plan.runs)
    assert {(problem.step, problem.path) for problem in plan.diagnostics} == {("baseline", path)}


def test_a_task_field_value_reaches_the_evaluated_states(private_host, tmp_path, monkeypatch):
    """Catches a planned Task axis whose value the Host never applies to the rollout."""
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("JAX_ENABLE_X64", "0")
    folders = ProjectFolderService(tmp_path / "projects")
    _save_project(folders, "own-sweeps", _project([], {"x": _axis(X, [0.2, 0.3])}))
    plan = folders.read_plan("own-sweeps", revision=None)
    train = next(run for run in plan.plan.runs if run.kind == "train")
    row = next(run for run in plan.plan.runs if run.kind == "evaluate" and run.key.row.endswith("x=v1"))
    host = private_host
    jobs = JobService(host, load_application().bundle)
    started = jobs.admit_plan(folders, plan, run_keys=(train.key, row.key), token=token("own-sweeps")).items
    assert all(item.disposition == "admitted" for item in started)
    job = started[-1].job_id
    settled = wait_until(lambda: job_row(host, job, "succeeded") or job_row(host, job, "failed"))
    assert settled["state"] == "succeeded", settled["outcome"]
    result = jobs.read_evaluation_result(project=plan.project_id, store=plan.store_path, job_id=job)
    with result.open_states() as states:
        np.testing.assert_allclose(states.data("/v1/task/output/reach_target")[..., 0], 0.3)


def test_a_task_document_field_edit_on_some_evaluate_runs_reaches_their_requests(tmp_path):
    """Catches an Evaluate's edit of a Task document field that is refused, or reaches the wrong runs' requests."""
    from fastapi.testclient import TestClient

    from feedbax.web.app import create_app

    raw = managed_evaluation_definition()
    evaluate = raw["steps"]["baseline"]
    evaluate["sweep"] = {"axes": {"probe": {"values": {"near": {"value": 1}, "far": {"value": 2}}}}}
    raw["steps"] = {"train": raw["steps"]["train"], "baseline": evaluate}
    index = next(i for i, item in enumerate(raw["documents"]["task"]["payload"]["variables"]) if item["id"] == "target")
    target = {"kind": "call", "function": "vector", "args": [
        {"kind": "literal", "value": coordinate, "unit": "m"} for coordinate in (0.25, -0.1)
    ]}
    folders = ProjectFolderService(tmp_path)
    _save_project(folders, "fields", Project.model_validate(raw))
    before = folders.read_plan("fields", revision=None)
    far = next(run.key for run in before.plan.runs if run.kind == "evaluate" and run.key.row.endswith("far"))
    with TestClient(create_app(projects_dir=tmp_path)) as client:
        reply = client.post("/api/projects/fields/authoring/trials", json={
            "revision": before.revision, "run": {"variant": far.variant, "step": far.step, "row": far.row},
            "scope": {"kind": "rows", "rows": [far.row]},
            "change": {"kind": "task", "edits": [{"op": "replace", "path": f"/variables/{index}/value", "value": target}]},
        })
        assert reply.status_code == 200, reply.text
        edited = reply.json()
        assert not edited["problems"], edited["problems"]
        saved = client.patch("/api/projects/fields", json={"revision": before.revision, "files": edited["files"]})
        assert saved.status_code == 200, saved.text
    after = folders.read_plan("fields", revision=None).plan
    assert after.diagnostics == ()
    variations = {}
    for run in after.runs:
        if run.kind == "evaluate":
            spec = decode_request(after.requests[run.request_id], after.documents.__getitem__).spec
            variation = spec.params["variation"]
            variations[run.key] = list(variation.values()) if isinstance(variation, dict) else variation
    assert variations.pop(far) == [{"op": "replace", "path": "task.variables.target.value", "value": target}]
    assert list(variations.values()) == [[]]


def test_a_set_applies_to_an_evaluate_that_runs_its_subjects_model():
    """Catches a Set an Evaluate cannot sweep unless it names a Model of its own, or whose entries plan alike."""
    from feedbax.contracts.set_document import sets_offered
    from tests.test_set_document_plan import _plan, _project as _set_project

    raw = _set_project()
    del raw["steps"]["baseline"]["base"]["inline"]["trial_documents"]
    del raw["steps"]["train"]["sweep"]
    assert sets_offered(Project.model_validate(raw), "baseline") == ("alterations",)
    swept = _plan(raw)
    assert swept.diagnostics == ()
    runs = {run.label: run.request_id for run in swept.runs if run.key.step == "baseline"}
    assert runs.keys() == {"Gain", "Scaled force", "Heavy"}
    del raw["steps"]["baseline"]["sweep"]
    (plain,) = (run.request_id for run in _plan(raw).runs if run.key.step == "baseline")
    assert len({plain, *runs.values()}) == 4


def test_a_list_typed_in_an_evaluate_s_trials_sweeps_them_across_its_subjects(tmp_path):
    """Catches an Evaluate's Trials field that offers no Sweep, or whose sweep is not crossed with the subject's runs."""
    from fastapi.testclient import TestClient

    from feedbax.web.app import create_app

    folders = ProjectFolderService(tmp_path)
    _save_project(folders, "trials", _project([0, 1, 2], {}))
    before = folders.read_plan("trials", revision=None)
    assert sum(run.kind == "evaluate" for run in before.plan.runs) == 3
    with TestClient(create_app(projects_dir=tmp_path)) as client:
        url = "/api/projects/trials/authoring/train/baseline/fields"
        shown = client.post(url, json={"revision": before.revision})
        assert shown.status_code == 200, shown.text
        view = shown.json()
        trials = next(field for field in view["fields"] if field["id"] == "/params/trial_set/count")
        assert "sweep" in view["sites"][trials["site"]]["kinds"]
        reply = client.post(url, json={
            "revision": before.revision, "field": "/params/trial_set/count", "edit": {"kind": "text", "text": "1, 4"},
        })
        assert reply.status_code == 200, reply.text
        assert not reply.json().get("problems"), reply.json()["problems"]
        saved = client.patch("/api/projects/trials", json={"revision": before.revision, "files": reply.json()["files"]})
        assert saved.status_code == 200, saved.text
    after = folders.read_plan("trials", revision=None).plan
    counts = sorted(
        len(decode_request(after.requests[run.request_id], after.documents.__getitem__).spec.params["trials"])
        for run in after.runs if run.kind == "evaluate"
    )
    assert counts == [1, 1, 1, 4, 4, 4]


def test_a_range_on_an_integer_task_parameter_plans_whole_values():
    """Catches an Evaluate's range over an integer Task parameter planned as fractional or refused values."""
    plan = _compile(_project([], {"length": {
        "path": "trial_documents.python_task.params.n_steps", "range": {"start": 5, "stop": 9, "count": 3},
    }}))
    assert plan.diagnostics == ()
    values = sorted(
        decode_request(plan.requests[run.request_id], plan.documents.__getitem__).spec.params["variation"]
        ["task-params-n_steps"]["value"]
        for run in plan.runs if run.kind == "evaluate"
    )
    assert values == [5, 7, 9] and all(type(value) is int for value in values)


def test_a_sweep_typed_in_an_evaluate_s_task_field_leaves_its_subject_alone(tmp_path):
    """Catches an Evaluate's Task field that offers no Sweep, or whose sweep rewrites the Task its subject trains on."""
    from fastapi.testclient import TestClient

    from feedbax.web.app import create_app

    raw = managed_evaluation_definition()
    raw["steps"] = {"train": raw["steps"]["train"], "baseline": raw["steps"]["baseline"]}
    folders = ProjectFolderService(tmp_path)
    _save_project(folders, "task-sweep", Project.model_validate(raw))
    before = folders.read_plan("task-sweep", revision=None)
    (run,) = (run.key for run in before.plan.runs if run.kind == "evaluate")
    field = "/task/signals/fixation/cells/hold-cue/value"
    with TestClient(create_app(projects_dir=tmp_path)) as client:
        url = "/api/projects/task-sweep/authoring/trials"
        selected = {"variant": run.variant, "step": run.step, "row": run.row}
        shown = client.post(url, json={"revision": before.revision, "run": selected}).json()
        assert next(item for item in shown["fields"] if item["path"] == field)["can_sweep"]
        reply = client.post(url, json={"revision": before.revision, "run": selected, "change": {
            "kind": "expression", "path": field, "edit": {"kind": "text", "text": "0.5, 0.75"},
        }})
        assert reply.status_code == 200, reply.text
        assert not reply.json()["problems"], reply.json()["problems"]
        saved = client.patch("/api/projects/task-sweep", json={"revision": before.revision, "files": reply.json()["files"]})
        assert saved.status_code == 200, saved.text
    after = folders.read_plan("task-sweep", revision=None).plan
    assert after.diagnostics == ()
    trained = {run.key: run.request_id for run in before.plan.runs if run.kind == "train"}
    assert {run.key: run.request_id for run in after.runs if run.kind == "train"} == trained
    values = sorted(
        edit["value"]
        for run in after.runs if run.kind == "evaluate"
        for edit in decode_request(after.requests[run.request_id], after.documents.__getitem__).spec.params["variation"]
        if edit["path"].startswith("task.signals.fixation.")
    )
    assert values == [0.5, 0.75]
