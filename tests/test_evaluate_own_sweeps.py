"""An Evaluate step's own fields take axes addressed as a Train's are."""

import numpy as np
import pytest

from feedbax.application.bundle import new_application_registry_bundle
from feedbax.builtins.reaching import REACHING_TASK
from feedbax.contracts.project import Project
from feedbax.contracts.request_identity import decode_request
from feedbax.jobs import JobService
from feedbax.planner.project_plan import compile_project
from feedbax.web.services.project_folder import ProjectFolderService
from tests.support.host import job_row, token, wait_until
from tests.test_canonical_evaluate_admission import _save_project, managed_evaluation_definition

X = "trial_documents.python_task.params.x"


def _axis(path, values, *, op="replace"):
    return {"values": {f"v{index}": {"value": value, "patches": {path: {"op": op, "value": value}}}
                       for index, value in enumerate(values)}}


def _project(seeds, axes):
    raw = managed_evaluation_definition()
    train = raw["steps"]["train"]
    train["base"]["inline"]["trial_documents"].update(task=None, python_task={"document": "task"}, input_overrides={})
    raw["documents"]["task"] = {
        "schema": "feedbax.spec.python_task",
        "payload": {"type": REACHING_TASK, "params": {"x": 0.1, "n_steps": 5}},
    }
    raw["documents"]["objective"]["payload"]["terms"][0].pop("epochs")
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
    bundle = new_application_registry_bundle()
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


def test_a_task_field_value_reaches_the_evaluated_states(host, tmp_path, monkeypatch):
    """Catches a planned Task axis whose value the Host never applies to the rollout."""
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("JAX_ENABLE_X64", "0")
    folders = ProjectFolderService(tmp_path / "projects")
    _save_project(folders, "own-sweeps", _project([], {"x": _axis(X, [0.2, 0.3])}))
    plan = folders.read_plan("own-sweeps", revision=None)
    train = next(run for run in plan.plan.runs if run.kind == "train")
    row = next(run for run in plan.plan.runs if run.kind == "evaluate" and run.key.row.endswith("x=v1"))
    jobs = JobService(host, new_application_registry_bundle())
    started = jobs.admit_plan(folders, plan, run_keys=(train.key, row.key), token=token("own-sweeps")).items
    assert all(item.disposition == "admitted" for item in started)
    job = started[-1].job_id
    settled = wait_until(lambda: job_row(host, job, "succeeded") or job_row(host, job, "failed"))
    assert settled["state"] == "succeeded", settled["outcome"]
    result = jobs.read_evaluation_result(project=plan.project_id, store=plan.store_path, job_id=job)
    with result.open_states() as states:
        np.testing.assert_allclose(states.data("/v1/task/output/reach_target")[..., 0], 0.3)
