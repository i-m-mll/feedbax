"""A task document's structure is changed by the service, and the document it leaves compiles."""

from feedbax.contracts.studio_trials import StudioTrialAuthoring
from feedbax.contracts.expression_values import LiteralValue
from feedbax.contracts.task_document import BLANK_TASK, TaskDocument
from tests.support.managed_train_fixture import managed_train_project
from tests.test_model_document_edits import _URL, _Project
from tests.test_task_document_fields import _save

TASK = "delayed-reaches-task"
LINKED = (
    '{"composites":[],"element":{"kind":"field","parent":{"id":"plant","kind":"component"},'
    '"path":["reset","effector","position"]}}'
)
DURATION = {"base": "real", "dimensions": {"time": 1}}


POINT = {"base": "real", "shape": [2], "dimensions": {"length": 1}}


def _project(tmp_path, task=None):
    raw = managed_train_project(scenario="delayed-reaches").model_dump(mode="json", by_alias=True, exclude_unset=True)
    if task is not None:
        raw["documents"][TASK]["payload"] = task.model_dump(mode="json")
    return _Project(tmp_path, raw)


def _change(project, change, *, body=None) -> StudioTrialAuthoring:
    row = next(row for row in project.plan().rows if row.kind == "train")
    response = project.client.post(f"{_URL}/authoring/trials", json={
        "revision": project.snapshot()["revision"],
        "run": {"variant": row.key.variant, "step": row.key.step, "row": row.key.row},
        "change": body if change is None else {"kind": "structure", "change": change},
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
    """Catches a change, such as a new type on a member whose values were of the old one, that leaves a Problem."""
    project = _project(tmp_path)
    with project.client:
        for change in (
            {"op": "add", "member": "variable", "name": "gain"},
            {"op": "rename", "member": "variable", "id": "gain", "name": "g"},
            {"op": "move", "member": "variable", "id": "gain", "index": 0},
            {"op": "set_type", "member": "variable", "id": "gain", "type": POINT},
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
            {"op": "set_type", "member": "signal", "id": "extra", "type": POINT},
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


def test_a_value_is_stored_in_a_task_that_already_has_problems(tmp_path):
    """Catches a value refused for the problems compiling reaches once it is mended: a new task's first variables are
    typed before their values, and its signals do not yet feed the model."""
    project = _project(tmp_path, BLANK_TASK)
    with project.client:
        for change in ({"op": "add", "member": "variable", "name": "start"},
                       {"op": "set_type", "member": "variable", "id": "start", "type": POINT}):
            _save(project, _change(project, change))
        view = _change(project, None, body={
            "kind": "expression", "path": "/task/variables/start/value", "edit": {"kind": "text", "text": "[0 m, 0 m]"},
        })
        assert view.files, view.problems
        _save(project, view)
        assert project.composed(TASK)["variables"][0]["value"]["function"] == "vector"


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


def test_a_problem_about_a_name_carries_the_name_s_place_in_the_field_s_text(tmp_path):
    """Catches a field that loses its text, or marks its whole value, when one name in it does not resolve."""
    project = _project(tmp_path)
    with project.client:
        _made(project, {"op": "remove", "member": "variable", "id": "catch"})
        path = "/task/signals/goal/cells/move/value"
        field = next(field for field in _change(project, None).fields if field.path == path)
        (problem,) = field.problems
        assert field.text[problem.span.start:problem.span.end] == "catch", (field.text, problem)

        row = next(row for row in project.plan().rows if row.kind == "train")
        text = "where(scale, start, target)"
        response = project.client.post(f"{_URL}/authoring/trials", json={
            "revision": project.snapshot()["revision"],
            "run": {"variant": row.key.variant, "step": row.key.step, "row": row.key.row},
            "change": {"kind": "expression", "path": path, "edit": {"kind": "text", "text": text}},
        })
        view = StudioTrialAuthoring.model_validate(response.json())
        refused = [problem for problem in view.problems if problem.path == ("task", "signals", "goal", "cells", "move", "value")]
        assert not view.files and refused, view.problems
        assert text[refused[-1].span.start:refused[-1].span.end] == "scale", refused


def test_a_variable_with_problems_marks_each_field_that_reads_it_where_it_reads_it(tmp_path):
    """Catches a bad variable whose readers go unmarked, are marked without the name's place, or hide fields that do not
    read it."""
    raw = managed_train_project(scenario="delayed-reaches").model_dump(mode="json", by_alias=True, exclude_unset=True)
    task = TaskDocument.model_validate(raw["documents"][TASK]["payload"])
    task = task.model_copy(update={"variables": tuple(
        item.model_copy(update={"value": LiteralValue(value=1, unit="m")}) if item.id == "catch" else item
        for item in task.variables
    )})
    project = _project(tmp_path, task)
    with project.client:
        fields = {field.path: field for field in _change(project, None).fields}
        marked = {path: field for path, field in fields.items() if field.problems}
        assert "/task/variables/catch/value" in marked and "/task/variables/start/value" not in marked
        field = marked["/task/signals/goal/cells/move/value"]
        (problem,) = field.problems
        assert field.text[problem.span.start:problem.span.end] == "catch", (field.text, problem)
