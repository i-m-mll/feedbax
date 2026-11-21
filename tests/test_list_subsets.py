"""A list input's selections are whole references: Best per item, To Rows per item, and named subsets."""

from copy import deepcopy

from feedbax.contracts.project import Project
from feedbax.contracts.project_plan import compile_project
from feedbax.contracts.request_refs import BestRank, ProductRef
from feedbax.contracts.studio_project import StudioProjectPatchBatch
from feedbax.contracts.studio_subset import StudioSubsetToRowsRequest
from feedbax.jobs import JobService
from feedbax.plugins.application import new_application_registry_bundle
from feedbax.web.services.project_folder import ProjectFolderService
from feedbax.web.services.project_plan import ProjectPlanService
from tests.support.host import job_row, token, wait_until
from tests.test_canonical_evaluate_admission import _save_project
from tests.test_saved_best_evaluate import authored_best

RATE = "settings.optimizer.learning_rate.value"


def _rates(values):
    return {"axes": {"rate": {"values": {
        name: {"value": value, "patches": {RATE: {"value": value}}} for name, value in values.items()
    }}}}


def listed_best():
    """One Evaluate over the best two runs of each of two Train steps, and a named subset of the first."""
    raw = authored_best()
    steps = raw["steps"]
    steps["train_b"] = deepcopy(steps["train"]) | {"sweep": _rates({"d": 0.004, "e": 0.0005, "f": 0.0015})}
    best = {"best": {"by": "train_loss", "k": 2, "order": "min"}}
    raw["documents"]["chosen"] = {"schema": "subset", "payload": {"rows": [
        {"key": "chosen", "variant": "default", "row": "rate=b"},
    ]}}
    steps["baseline"]["inputs"]["subject"] = [
        {"step": "train", "checkpoint": "last", "where": deepcopy(best)},
        {"step": "train_b", "checkpoint": "last", "where": deepcopy(best)},
        {"step": "train", "checkpoint": "last", "where": {"subset": "chosen"}},
    ]
    return raw


def _subjects(plan):
    """Each Evaluate row's subject member, by row."""
    return {run.key.row: run.inputs["subject"] for run in plan.runs if run.kind == "evaluate"}


def test_each_selection_ranks_its_own_step_and_records_its_subset():
    plan = compile_project(Project.model_validate(listed_best()))
    assert not plan.diagnostics, plan.diagnostics
    subsets = {key[2]: subset for key, subset in plan.subsets.items() if key[1] == "baseline"}
    assert set(subsets) == {"subject.0", "subject.1", "subject.2"}
    assert [len(subsets[item].ranks) for item in ("subject.0", "subject.1")] == [2, 2]
    assert subsets["subject.2"].document == "chosen" and not subsets["subject.2"].ranks
    subjects = _subjects(plan)
    assert len(subjects) == 5
    ranked = {item: {subjects[row].sources[0].step for row in subjects if subjects[row].member.key in subsets[item].ranks}
              for item in ("subject.0", "subject.1")}
    assert ranked == {"subject.0": {"train"}, "subject.1": {"train_b"}}
    assert sum(isinstance(binding.member.reference, ProductRef) for binding in subjects.values()) == 1

    edited = listed_best()
    edited["steps"]["train_b"]["sweep"] = _rates({"d": 0.004, "e": 0.0006, "f": 0.0015})
    after_plan = compile_project(Project.model_validate(edited))
    after = _subjects(after_plan)
    before_requests = {run.key.row: run.request_id for run in plan.runs if run.kind == "evaluate"}
    after_requests = {run.key.row: run.request_id for run in after_plan.runs if run.kind == "evaluate"}
    kept = {row for row in before_requests if after_requests.get(row) == before_requests[row]}
    second = {row for row, binding in subjects.items() if binding.member.key in subsets["subject.1"].ranks}
    assert kept == set(before_requests) - second
    assert all(after[row].sources[0].step == "train_b" for row in set(after) - kept)


def test_to_rows_freezes_one_selection_and_leaves_the_others_measured(host, tmp_path, monkeypatch):
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("JAX_ENABLE_X64", "0")
    host.policy(compute=2)
    project = "listed-best"
    folders = ProjectFolderService(tmp_path / "projects")
    jobs = JobService(host, new_application_registry_bundle())
    service = ProjectPlanService(folders, jobs)
    _save_project(folders, project, Project.model_validate(listed_best()))
    saved = folders.read_plan(project, revision=None)
    plan = saved.plan
    assert not plan.diagnostics, plan.diagnostics
    admitted = jobs.admit_plan(folders, saved, run_keys=tuple(run.key for run in plan.runs), token=token("all"))
    assert {item.disposition for item in admitted.items} == {"admitted"}
    for item in admitted.items:
        row = wait_until(lambda: current if (current := job_row(host, item.job_id))["outcome"] is not None else None)
        assert row["state"] == "succeeded", row["outcome"]

    response = service.response(service.read(project, revision=saved.revision), include_window_map=True)
    card = next(step for step in response.window_map.steps if step.step_id == "baseline")
    inputs = {item.id: item for item in card.inputs}
    assert [(inputs[id].subset.count, inputs[id].subset.total) for id in ("subject.0", "subject.1", "subject.2")] == [
        (2, 3), (2, 3), (1, 3),
    ]
    first = inputs["subject.0"].subset.ranks
    attempts = {
        item.bound_subject.key: item.result.job_id for item in response.window_map.evaluations
        if item.key.step == "baseline" and item.bound_subject is not None
    }
    assert set(attempts) == {*first, *inputs["subject.1"].subset.ranks}
    edit = service.subset_to_rows(project, StudioSubsetToRowsRequest(
        revision=saved.revision, variant="default", step="baseline", input="subject.0",
        attempts=[attempts[key] for key in first],
    ))
    snapshot = folders.patch(project, StudioProjectPatchBatch(revision=edit.revision, files=edit.files))
    subject = snapshot.project.steps["baseline"].inputs["subject"]
    assert [row.key for row in subject[0].where.rows] == first
    assert subject[1:] == Project.model_validate(listed_best()).steps["baseline"].inputs["subject"][1:]

    frozen = folders.read_plan(project, revision=None).plan
    assert not frozen.diagnostics, frozen.diagnostics
    before, after = _subjects(plan), _subjects(frozen)
    assert after.keys() == before.keys()
    for row, binding in after.items():
        if before[row].member.key in first:
            assert isinstance(binding.member.reference, ProductRef) and binding.producer.step == "train"
        else:
            assert binding.member == before[row].member
    assert sum(isinstance(binding.member.reference, BestRank) for binding in after.values()) == 2
