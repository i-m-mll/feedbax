"""Checkpoint rule authoring uses the saved project's Plan."""

import pytest
from fastapi.testclient import TestClient

from feedbax.contracts.project import Project
from feedbax.contracts.project_inputs import StepReference
from feedbax.contracts.expression_values import UnresolvedValue
from feedbax.contracts.project_sweep import CHECKPOINT_RULE_TARGET
from feedbax.contracts.request_refs import UpdateSeriesMember
from feedbax.contracts.studio_checkpoint_rule import StudioCheckpointRuleRequest
from feedbax.web.app import create_app
from feedbax.web.services.checkpoint_rule import prepare_checkpoint_rule_edit
from feedbax.web.services.project_folder import (
    ProjectFolderInvalid, ProjectFolderService, ProjectRevisionConflict, preview_authored_files,
)
from tests.test_canonical_evaluate_admission import _save_project
from tests.test_saved_update_series import definition


def test_checkpoint_draft_is_only_raw_text_in_a_saved_reference():
    text = "  1, huh  "
    reference = StepReference.model_validate({"step": "train", "checkpoint": text})
    assert reference.checkpoint == UnresolvedValue(text=text)
    assert reference.model_dump(mode="json", exclude_unset=True) == {"step": "train", "checkpoint": text}
    with pytest.raises(ValueError, match="Save an invalid checkpoint rule as its text"):
        StepReference.model_validate({"step": "train", "checkpoint": {"kind": "unresolved", "text": text}})
    for rule in ("last", {"at": 3}, [{"at": 3}], {"every": 2, "through": 8}):
        reference = StepReference.model_validate({"step": "train", "checkpoint": rule})
        assert reference.model_dump(mode="json", exclude_unset=True, exclude_none=True)["checkpoint"] == rule


@pytest.mark.parametrize("site", ["run", "checkpoint"])
def test_bad_rule_commits_and_reopens_for_repair(tmp_path, monkeypatch, site):
    """An invalid edit survives reload, blocks only its consumer, and can later be corrected."""
    project = "bad-rule"
    route = f"/api/projects/{project}"
    folders = ProjectFolderService(tmp_path)
    raw = definition({"every": 2})
    if site == "checkpoint":
        raw["sweep"] = {"rows": {"case": {}}}
    _save_project(folders, project, Project.model_validate(raw))
    before = folders.read_plan(project, revision=None)
    run = next(run for run in before.plan.runs if run.kind == "evaluate")
    variant, row = run.key.variant, run.key.row
    text = "  seed=  " if site == "run" else "  1, huh  "
    held = {"kind": "unresolved", "text": text}
    expected_path = "inputs.subject.where.rule" if site == "run" else "inputs.subject.checkpoint"

    app = create_app(projects_dir=tmp_path, automatic_run="off")
    with TestClient(app) as client:
        monkeypatch.setattr(app.state.job_service.host, "jobs_snapshot", lambda **_: {"cursor": 0, "jobs": []})

        def prepare(revision, rule, row_id=row):
            result = client.post(f"{route}/authoring/checkpoint-rule", json={
                "revision": revision, "run": {"variant": variant, "step": "baseline", "row": row_id},
                "edit": {"kind": "set", "scope": {"kind": "rows", "rows": [row_id]}, "rule": rule},
            })
            return result

        def apply(revision, files):
            assert files and all(file["patches"] for file in files)
            response = client.patch(route, json={"revision": revision, "files": files})
            assert response.status_code == 200, response.text
            assert response.json()["revision"] != revision
            return response.json()["revision"]

        if site == "run":
            response = client.post(f"{route}/authoring/run-rule", json={
                "revision": before.revision, "variant": variant, "step": "train", "rule": text,
            })
            assert response.status_code == 200, response.text
            answer = response.json()["result"]
            assert answer["kind"] == "refused" and answer["span"] is not None
            files = [{"address": {"kind": "step", "id": "baseline"}, "patches": [
                {"op": "add", "path": "/inputs/subject/where", "value": {"rule": text}},
            ]}]
        else:
            refused = prepare(before.revision, held, "vanished")
            assert refused.status_code == 422 and "run changed" in refused.text
            response = prepare(before.revision, held)
            assert response.status_code == 200, response.text
            [problem] = response.json()["problems"]
            assert problem["path"] == expected_path
            files = response.json()["files"]
        assert folders.read(project).revision == before.revision

        revision = apply(before.revision, files)
        reopened = ProjectFolderService(tmp_path)
        persisted = reopened.read(project, revision=revision).project.steps["baseline"]
        if site == "run":
            assert persisted.inputs["subject"].where.rule == text
        else:
            [exception] = persisted.sweep.exceptions.values()
            assert exception.patches[CHECKPOINT_RULE_TARGET].value == text
        plan = reopened.read_plan(project, revision=revision).plan
        assert {run.kind for run in plan.runs} == {"train"}
        [problem] = plan.diagnostics
        assert (problem.step, problem.variant, problem.path) == ("baseline", variant, expected_path)
        assert problem.span is not None

        response = client.get(f"{route}/train-runs", params={"revision": revision, "window_map": True})
        assert response.status_code == 200, response.text
        window = response.json()["window_map"]
        [reopened_problem] = window["problems"]
        assert (reopened_problem["address"], reopened_problem["path"]) == ("steps/baseline.json", expected_path)
        assert reopened_problem["authored_path"].endswith("/where/rule" if site == "run" else "/value")
        step = next(step for step in window["steps"] if step["step_id"] == "baseline")
        [subject] = step["inputs"]
        if site == "run":
            assert subject["subset"]["terms"]["rule"] == text
            files[0]["patches"] = [{"op": "replace", "path": "/inputs/subject/where/rule", "value": ""}]
        else:
            [exception] = step["exceptions"]
            assert exception["rule"]["patches"][CHECKPOINT_RULE_TARGET]["value"] == text
            response = prepare(revision, {"at": 2})
            assert response.status_code == 200, response.text
            assert response.json()["problems"] == []
            files = response.json()["files"]
        repaired = reopened.read_plan(project, revision=apply(revision, files)).plan
        assert not repaired.diagnostics and {run.kind for run in repaired.runs} == {"train", "evaluate"}


def test_checkpoint_rule_sweep_keeps_subject_fanout_and_replans_horizons(tmp_path):
    folders = ProjectFolderService(tmp_path / "projects")
    raw = definition({"every": 2})
    train = raw["steps"]["train"]
    train["sweep"] = {"axes": {"budget": {"values": {
        "short": {},
        "long": {"patches": {
            "training_config.n_batches": {"value": 6},
        }},
    }}}}
    evaluate = raw["steps"]["baseline"]
    evaluate["sweep"] = {"axes": {
        "subject": {"input": "subject"},
        "replicate": {"values": {"control": {}, "test": {"patches": {"params.seed": {"value": 9}}}}},
    }}
    project = Project.model_validate(raw)
    _save_project(folders, "series", project)

    def plan():
        return folders.read_plan("series", revision=None)

    def prepare(settled, row, edit, choice=None):
        return prepare_checkpoint_rule_edit(folders, "series", StudioCheckpointRuleRequest.model_validate({
            "revision": settled.revision,
            "run": {"variant": row.key.variant, "step": row.key.step, "row": row.key.row},
            "edit": edit,
            "store_choice": choice,
        }))

    before = plan()
    baseline = {row.key: row for row in before.plan.runs}
    test_row = next(row for row in baseline.values() if row.key.step == "baseline"
                    and row.coordinates["replicate"].value_id == "test")
    axis_edit = prepare(before, test_row, {
        "kind": "set", "scope": {"kind": "axis_value", "axis": "replicate", "value": "test"},
        "rule": [{"at": 3}],
    })
    named = preview_authored_files(project, axis_edit.files)
    _save_project(folders, "series", named)
    after_axis = plan()
    axis_rows = {row.key: row for row in after_axis.plan.runs}
    assert baseline.keys() == axis_rows.keys()
    for key, row in axis_rows.items():
        selected = key.step == "baseline" and row.coordinates["replicate"].value_id == "test"
        assert (row.request_id != baseline[key].request_id) is selected
    control = next(row for row in axis_rows.values() if row.key.step == "baseline"
                   and row.coordinates["replicate"].value_id == "control")
    row_edit = prepare(after_axis, control, {
        "kind": "set", "scope": {"kind": "rows", "rows": [control.key.row]}, "rule": [{"at": 5}],
    })
    _save_project(folders, "series", preview_authored_files(named, row_edit.files))
    assert {
        row.key for row in plan().plan.runs if row.request_id != axis_rows[row.key].request_id
    } == {control.key}
    after_row = plan()
    change = {
        "kind": "set", "scope": {"kind": "rows", "rows": [test_row.key.row]}, "rule": [{"at": 1}],
    }
    review = prepare(after_row, test_row, change)
    assert review.files == []
    assert plan().revision == after_row.revision
    option, = review.store_choices
    assert option.kind == "sweep_value"
    assert {(key.variant, key.step, key.row) for key in option.runs} == {
        (key.variant, key.step, key.row) for key, run in axis_rows.items()
        if key.step == "baseline" and run.coordinates["replicate"].value_id == "test"
    }
    with pytest.raises(ProjectRevisionConflict):
        prepare(after_row, test_row, change | {"rule": "last"}, option.id)
    assert plan().revision == after_row.revision
    chosen = prepare(after_row, test_row, change, option.id)
    saved = folders.read("series", revision=after_row.revision)
    _save_project(folders, "series", preview_authored_files(saved.project, chosen.files))
    original_requests = {run.key: run.request_id for run in after_row.plan.runs}
    assert {
        (run.key.variant, run.key.step, run.key.row) for run in plan().plan.runs
        if run.request_id != original_requests[run.key]
    } == {(key.variant, key.step, key.row) for key in option.runs}

    _save_project(folders, "series", project)
    conversion = prepare(plan(), test_row, {"kind": "to_sweep"})
    swept = preview_authored_files(project, conversion.files)
    _save_project(folders, "series", swept)
    split = plan()
    assert not split.plan.diagnostics
    assert swept.steps["baseline"].inputs["subject"].checkpoint is None
    original = split.plan.runs
    assert {row.key: row.request_id for row in original if row.kind == "train"} == {
        key: row.request_id for key, row in baseline.items() if row.kind == "train"
    }
    by_budget = {}
    for row in original:
        if row.kind == "evaluate":
            point = row.inputs["subject"].member.reference.select
            update = row.coordinates["update"].value_id
            assert isinstance(point, UpdateSeriesMember) and point.updates == (int(update),)
            by_budget.setdefault(row.coordinates["budget"].value_id, set()).add(update)
    assert by_budget == {"short": {"2", "4"}, "long": {"2", "4", "6"}}
    train = swept.steps["train"]
    patches = train.sweep.axes["budget"].values["long"].patches
    for write in patches.values():
        write.value = 8
    _save_project(folders, "series", swept)
    grown = plan().plan
    assert not grown.diagnostics
    assert {row.key for row in original} <= {row.key for row in grown.runs}
    assert {row.coordinates["update"].value_id for row in grown.runs if row.kind == "evaluate"
            and row.coordinates["budget"].value_id == "long"} == {"2", "4", "6", "8"}

    inherited = definition({"every": 2})
    inherited["steps"]["baseline"]["sweep"] = {"rows": {
        "chosen": {"patches": {CHECKPOINT_RULE_TARGET: {"value": [{"at": 3}]}}},
    }}
    inherited["steps"]["child"] = {"label": "Child", "kind": "evaluate", "base": {"step": "baseline", "row": "chosen"}}
    _save_project(folders, "series", Project.model_validate(inherited))
    inherited_rows = {row.key.step: row for row in plan().plan.runs}
    assert inherited_rows["child"].request_id == inherited_rows["baseline"].request_id
    assert inherited_rows["child"].inputs["subject"].member.reference.select.updates == (3,)
    with pytest.raises(ProjectFolderInvalid, match="cross of generated rows"):
        prepare(plan(), inherited_rows["baseline"], {"kind": "to_sweep"})
