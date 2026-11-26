"""Studio keeps family expressions and their edit paths on the authored member."""

from fastapi.testclient import TestClient
import pytest

from feedbax.contracts.project import Project
from feedbax.contracts.set_document import SET_DOCUMENT_SCHEMA_ID
from feedbax.web.app import create_app
from tests.test_canonical_evaluate_admission import _save_project
from tests.test_document_train_pair import authored_documents, document_project


def _post(client, body, change=None):
    if change is not None:
        body = body | {"change": change}
    response = client.post("/api/projects/families/authoring/trials", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def _field(view, path):
    return next(field for field in view["fields"] if field["path"] == path)


def _index(view, field):
    return next(name for name in view["sites"][field["site"]]["names"] if name["id"] == "k")


def _edit(path, text):
    return {"kind": "expression", "path": path, "edit": {"kind": "text", "text": text}}


def _save(client, body, recipe):
    response = client.patch("/api/projects/families", json={"revision": body["revision"], "files": recipe["files"]})
    assert response.status_code == 200, response.text
    return body | {"revision": response.json()["revision"]}


def test_objective_family_fields_keep_the_authored_member_and_scope(tmp_path):
    raw = document_project(authored_documents())
    term = raw["documents"]["objective"]["payload"]["terms"]["effort"]
    term.update(over={"k": "0..3"}, weight="1 + k")
    app = create_app(projects_dir=tmp_path)
    folders = app.state.project_folders
    _save_project(folders, "families", Project.model_validate(raw))
    body = {"revision": folders.read("families").revision,
            "run": {"variant": "default", "step": "train", "row": "default"}}
    path = "/objective/terms/effort"
    with TestClient(app) as client:
        view = _post(client, body)
        assert list(view["objective"]["terms"]) == ["effort"]
        assert view["objective"]["terms"]["effort"]["over"] == {"k": "0..3"}
        assert not view["problems"], view["problems"]
        weight = _field(view, path + "/weight")
        assert weight["text"] == "1 + k"
        index = _index(view, weight)
        assert index["type"]["base"] == "integer"
        assert _field(view, path + "/over/k")["text"] == "0..3"

        body = _save(client, body, _post(client, body, _edit(path + "/weight", "2 + k")))
        body = _save(client, body, _post(client, body, _edit(path + "/over/k", "1..2")))
        reopened = _post(client, body)
        assert reopened["objective"]["terms"]["effort"]["over"] == {"k": "1..2"}
        assert _field(reopened, path + "/weight")["text"] == "2 + k"
        assert not any("effort-1" in field["path"] for field in reopened["fields"])


def test_objective_family_problem_names_the_stored_field_and_combination(tmp_path):
    raw = document_project(authored_documents())
    raw["documents"]["objective"]["payload"]["terms"]["effort"].update(
        over={"k": "0..3"}, weight="k + true",
    )
    app = create_app(projects_dir=tmp_path)
    folders = app.state.project_folders
    _save_project(folders, "families", Project.model_validate(raw))
    body = {"revision": folders.read("families").revision,
            "run": {"variant": "default", "step": "train", "row": "default"}}
    with TestClient(app) as client:
        view = _post(client, body)
    assert [term for term in view["objective"]["terms"]] == ["effort"]
    problem = next(problem for problem in view["problems"] if problem["path"][:4] == ["objective", "terms", "effort", "weight"])
    assert "k=0" in problem["reason"]


@pytest.mark.parametrize("placement", ["trial", "run"])
def test_set_entry_family_variable_edits_the_authored_value(tmp_path, placement):
    raw = document_project(authored_documents())
    raw["documents"]["bank"] = {
        "label": "Bank", "schema": SET_DOCUMENT_SCHEMA_ID, "payload": {"entries": {
            "pulse": {"label": "Pulse {k}", "over": {"k": "0..1"}, "task_variables": {
                "amount": {"label": "Amount", "type": {"base": "real"}, "value": "1 + k"},
            }},
        }},
    }
    differences = {}
    if placement == "trial":
        differences = {"/interior/wires/task-wire/attachments": {"value": {"pulse": {
            "kind": "intervention", "operation": "Scale",
            "operands": {"factor": {"kind": "value", "value": "1"}},
        }}}}
        raw["documents"]["bank"]["payload"]["entries"]["pulse"]["differences"] = differences
        raw["steps"]["train"]["set_fields"] = {"pulse": {"axis": {"set": "bank"}, "sampling": {"kind": "listed"}}}
    else:
        raw["steps"]["train"]["sweep"] = {"axes": {"pulse": {"set": "bank"}}}
    app = create_app(projects_dir=tmp_path)
    folders = app.state.project_folders
    _save_project(folders, "families", Project.model_validate(raw))
    body = {"revision": folders.read("families").revision,
            "run": {"variant": "default", "step": "train", "row": "default" if placement == "trial" else "pulse=pulse-1"}}
    with TestClient(app) as client:
        view = _post(client, body)
        owned = next(variable for variable in view["variables"] if variable.get("set_entry", {}).get("entry") == "pulse")
        path = f"/task/variables/{owned['id']}/value"
        field = _field(view, path)
        assert field["text"] == "1 + k"
        assert _index(view, field)["type"]["base"] == "integer"
        _save(client, body, _post(client, body, _edit(path, "3 + k")))
    entry = folders.read("families").project.documents["bank"].payload["entries"]["pulse"]
    assert entry["task_variables"]["amount"]["value"] == "3 + k"
    assert entry["over"] == {"k": "0..1"}
    assert entry.get("differences", {}) == differences
