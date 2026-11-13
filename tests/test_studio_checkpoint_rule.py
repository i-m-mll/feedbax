"""Checkpoint rule authoring uses the saved project's Plan."""

import pytest

from feedbax.contracts.project import Project
from feedbax.contracts.project_sweep import CHECKPOINT_RULE_TARGET
from feedbax.contracts.request_refs import UpdateSeriesMember
from feedbax.contracts.studio_checkpoint_rule import StudioCheckpointRuleRequest
from feedbax.web.services.checkpoint_rule import prepare_checkpoint_rule_edit
from feedbax.web.services.project_folder import ProjectFolderInvalid, ProjectFolderService, preview_authored_files
from tests.test_canonical_evaluate_admission import _save_project
from tests.test_saved_update_series import definition


def test_checkpoint_rule_sweep_keeps_subject_fanout_and_replans_horizons(tmp_path):
    folders = ProjectFolderService(tmp_path / "projects")
    raw = definition({"every": 2})
    train = raw["steps"]["train"]
    train["sweep"] = {"axes": {"budget": {"values": {
        "short": {},
        "long": {"patches": {
            "training_config.n_batches": {"value": 6},
            "method_payload.payload.target_total": {"value": 6},
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

    def prepare(settled, row, edit):
        return prepare_checkpoint_rule_edit(folders, "series", StudioCheckpointRuleRequest.model_validate({
            "revision": settled.revision,
            "run": {"variant": row.key.variant, "step": row.key.step, "row": row.key.row},
            "edit": edit,
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
    with pytest.raises(ProjectFolderInvalid, match="already writes"):
        prepare(plan(), axis_rows[test_row.key], {
            "kind": "set", "scope": {"kind": "rows", "rows": [test_row.key.row]}, "rule": [{"at": 7}],
        })

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
    inherited["steps"]["child"] = {"kind": "evaluate", "base": {"step": "baseline", "row": "chosen"}}
    _save_project(folders, "series", Project.model_validate(inherited))
    inherited_rows = {row.key.step: row for row in plan().plan.runs}
    assert inherited_rows["child"].request_id == inherited_rows["baseline"].request_id
    assert inherited_rows["child"].inputs["subject"].member.reference.select.updates == (3,)
    with pytest.raises(ProjectFolderInvalid, match="cross of generated rows"):
        prepare(plan(), inherited_rows["baseline"], {"kind": "to_sweep"})
