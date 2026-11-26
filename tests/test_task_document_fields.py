"""A Task document's fields are edited, and its numbers swept, by the paths of the members they lie in."""

from feedbax.contracts.request_identity import decode_request
from feedbax.contracts.studio_trials import StudioTrialAuthoring
from tests.test_evaluate_documents import TASK, _data
from tests.test_model_document_edits import _URL, _Project


def _author(project, step, change=None, **body):
    response = project.client.post(f"{_URL}/authoring/trials", json={
        "revision": project.snapshot()["revision"], "run": project.run(step), "change": change, **body,
    })
    assert response.status_code == 200, response.text
    return StudioTrialAuthoring.model_validate(response.json())


def _save(project, view):
    for created in view.created:
        response = project.client.post(f"{_URL}/files", json={
            "revision": project.snapshot()["revision"], "file": created.model_dump(mode="json", by_alias=True),
        })
        assert response.status_code == 201, response.text
    response = project.client.patch(_URL, json={
        "revision": project.snapshot()["revision"], "files": [file.model_dump(mode="json") for file in view.files],
    })
    assert response.status_code == 200, response.text


def _text(project, step, path):
    return next(field.text for field in _author(project, step).fields if field.path == path)


def _gain(project):
    """Each Train request's value of the task's ``gain``, in the order of the step's rows."""
    plan = project.plan()
    values = []
    for run in plan.runs:
        if run.kind == "train":
            task = decode_request(plan.requests[run.request_id], plan.documents.__getitem__).spec.trial_documents.task
            (gain,) = (variable for variable in task.variables if variable.id == "gain")
            values.append(gain.value.value)
    return values


def _with_gain():
    data = _data()
    payload = data["documents"][TASK]["payload"]
    payload["variables"].insert(0, {
        "id": "gain", "name": "gain", "type": {"base": "real"}, "value": {"kind": "literal", "value": 0.1},
    })
    return data


def test_a_sweep_of_a_task_variable_stays_on_it_when_the_variables_are_reordered(tmp_path):
    project = _Project(tmp_path, _with_gain())
    with project.client:
        path = "/task/variables/gain/value"
        view = _author(project, "fee", {"kind": "expression", "path": path, "edit": {"kind": "text", "text": "0.1, 0.15, 0.2"}})
        assert not view.problems, view.problems
        _save(project, view)
        assert sorted(_gain(project)) == [0.1, 0.15, 0.2]
        swept = next(field for field in _author(project, "fee").fields if field.path == path)
        assert swept.sweep is not None

        data = project.project().model_dump(mode="json", by_alias=True, exclude_unset=True)
        variables = data["documents"][TASK]["payload"]["variables"]
        variables.append(variables.pop(0))
        reordered = _Project(tmp_path / "reordered", data)
        assert sorted(_gain(reordered)) == [0.1, 0.15, 0.2]


def test_a_task_field_s_text_is_what_its_edit_stored(tmp_path):
    project = _Project(tmp_path, _data())
    with project.client:
        path = "/task/epochs/move/end/length"
        assert _text(project, "fee", path) == "80 st"
        view = _author(project, "fee", {"kind": "expression", "path": path, "edit": {"kind": "text", "text": "60 st"}})
        assert not view.problems, view.problems
        _save(project, view)
        assert _text(project, "fee", path) == "60 st"
        (epoch,) = project.composed(TASK)["epochs"]
        assert epoch["end"]["length"] == {"kind": "literal", "value": 60, "unit": "st"}


def _start_low(project):
    """Each Train request's lower bound of the draw that gives ``start``'s first coordinate."""
    plan = project.plan()
    values = []
    for run in plan.runs:
        if run.kind == "train":
            task = decode_request(plan.requests[run.request_id], plan.documents.__getitem__).spec.trial_documents.task
            (start,) = (variable for variable in task.variables if variable.id == "start")
            values.append(start.value.args[0].args[0].value)
    return values


def test_a_number_given_to_a_draw_is_set_and_swept_as_a_field_of_its_own(tmp_path):
    project = _Project(tmp_path, _data())
    with project.client:
        path = "/task/variables/start/value/args/0/args/0"
        view = _author(project, "fee", {"kind": "expression", "path": path, "edit": {"kind": "text", "text": "-0.35"}})
        assert not view.problems, view.problems
        _save(project, view)
        assert _text(project, "fee", path) == "-0.35"
        assert _start_low(project) == [-0.35]
        view = _author(project, "fee", {"kind": "expression", "path": path, "edit": {"kind": "text", "text": "-0.4, -0.3"}})
        assert not view.problems, view.problems
        _save(project, view)
        assert sorted(_start_low(project)) == [-0.4, -0.3]
