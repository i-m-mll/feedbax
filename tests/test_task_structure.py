"""A task document's structure is changed by the service, and the document it leaves compiles."""

from feedbax.contracts.studio_trials import StudioTrialAuthoring
from tests.support.managed_train_fixture import managed_train_project
from tests.test_model_document_edits import _URL, _Project
from tests.test_task_document_fields import _save

TASK = "delayed-reaches-task"
LINKED = (
    '{"composites":[],"element":{"kind":"field","parent":{"id":"plant","kind":"component"},'
    '"path":["reset","effector","position"]}}'
)
DURATION = {"base": "real", "dimensions": {"time": 1}}


def _project(tmp_path):
    return _Project(tmp_path, managed_train_project(scenario="delayed-reaches").model_dump(
        mode="json", by_alias=True, exclude_unset=True,
    ))


def _change(project, change) -> StudioTrialAuthoring:
    row = next(row for row in project.plan().rows if row.kind == "train")
    response = project.client.post(f"{_URL}/authoring/trials", json={
        "revision": project.snapshot()["revision"],
        "run": {"variant": row.key.variant, "step": row.key.step, "row": row.key.row},
        "change": None if change is None else {"kind": "structure", "change": change},
    })
    assert response.status_code == 200, response.text
    return StudioTrialAuthoring.model_validate(response.json())


def _made(project, change) -> list:
    """The problems of the task once ``change`` is saved."""
    view = _change(project, change)
    assert view.files and not view.problems, (change, view.problems)
    _save(project, view)
    return _change(project, None).problems


def test_each_structural_change_leaves_a_document_that_compiles(tmp_path):
    project = _project(tmp_path)
    with project.client:
        for change in (
            {"op": "add", "member": "variable", "name": "gain"},
            {"op": "rename", "member": "variable", "id": "gain", "name": "g"},
            {"op": "move", "member": "variable", "id": "gain", "index": 0},
            {"op": "set_type", "member": "variable", "id": "gain", "type": DURATION},
            {"op": "remove", "member": "variable", "id": "gain"},
            {"op": "add", "member": "epoch", "name": "settle", "index": 1},
            {"op": "rename", "member": "epoch", "id": "settle", "name": "rest"},
            {"op": "move", "member": "epoch", "id": "settle", "index": 0},
            {"op": "remove", "member": "epoch", "id": "settle"},
            {"op": "add", "member": "signal", "name": "extra"},
            {"op": "rename", "member": "signal", "id": "extra", "name": "more"},
            {"op": "move", "member": "signal", "id": "extra", "index": 0},
            {"op": "set_type", "member": "signal", "id": "extra", "type": DURATION},
            {"op": "present", "signal": "extra", "epoch": "hold"},
            {"op": "set_span", "signal": "extra", "cell": "hold", "first_epoch": "hold", "last_epoch": "cue"},
            {"op": "absent", "signal": "extra", "cell": "hold"},
            {"op": "remove", "member": "signal", "id": "extra"},
            {"op": "link", "model_field": LINKED, "variable": "target"},
        ):
            assert _made(project, change) == [], change
        task = project.composed(TASK)
        assert [item["id"] for item in task["variables"]] == ["start", "target", "catch"]
        assert [item["id"] for item in task["epochs"]] == ["hold", "cue", "move"]
        assert task["links"] == [{"model_field": LINKED, "variable": "target"}]


def test_removing_an_epoch_shortens_the_cells_that_span_it(tmp_path):
    project = _project(tmp_path)
    with project.client:
        assert _made(project, {"op": "remove", "member": "epoch", "id": "cue"}) == []
        spans = {
            signal["id"]: [(cell["first_epoch"], cell["last_epoch"]) for cell in signal["cells"]]
            for signal in project.composed(TASK)["signals"]
        }
        assert spans["goal"] == [("hold", "hold"), ("move", "move")]
        assert spans["target_pos"] == [("move", "move")]


def test_removing_a_variable_that_is_read_leaves_a_problem_where_it_is_read(tmp_path):
    project = _project(tmp_path)
    with project.client:
        problems = _made(project, {"op": "remove", "member": "variable", "id": "catch"})
        assert "catch" not in [item["id"] for item in project.composed(TASK)["variables"]]
        read = ("signals", "goal", "cells", "move", "value")
        assert any(problem.path[:len(read)] == read for problem in problems), problems
