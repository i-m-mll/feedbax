"""Studio keeps family expressions and their edit paths on the authored member."""

from fastapi.testclient import TestClient
import pytest

from feedbax.contracts.project import Project
from feedbax.contracts.project_sources import pointer
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
    term.update(over={"k": "0..5"}, weight="1 + k")
    app = create_app(projects_dir=tmp_path)
    folders = app.state.project_folders
    _save_project(folders, "families", Project.model_validate(raw))
    body = {"revision": folders.read("families").revision,
            "run": {"variant": "default", "step": "train", "row": "default"}}
    path = "/objective/terms/effort"
    with TestClient(app) as client:
        view = _post(client, body)
        assert list(view["objective"]["terms"]) == ["effort"]
        assert view["objective"]["terms"]["effort"]["over"] == {"k": "0..5"}
        assert view["families"][path] == {"member_count": 6, "indices": {"k": {"count": 6}}}
        assert not view["problems"], view["problems"]
        weight = _field(view, path + "/weight")
        assert weight["text"] == "1 + k"
        index = _index(view, weight)
        assert index["type"]["base"] == "integer"
        assert _field(view, path + "/over/k")["text"] == "0..5"

        body = _save(client, body, _post(client, body, _edit(path + "/weight", "2 + k")))
        body = _save(client, body, _post(client, body, _edit(path + "/over/k", "0..7")))
        reopened = _post(client, body)
        assert reopened["objective"]["terms"]["effort"]["over"] == {"k": "0..7"}
        assert reopened["families"][path]["member_count"] == 8
        assert _field(reopened, path + "/weight")["text"] == "2 + k"
        assert not any("effort-1" in field["path"] for field in reopened["fields"])


def test_objective_family_problem_names_the_stored_field_and_combination(tmp_path):
    raw = document_project(authored_documents())
    raw["documents"]["objective"]["payload"]["terms"]["effort"].update(
        over={"k": {"token": {"value": 0, "label": "First choice"}}}, weight="k + true",
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
    assert "k=First choice" in problem["reason"]
    assert "k=First choice" in _field(view, "/objective/terms/effort/weight")["problems"][0]["reason"]


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


def test_validation_trial_family_projects_and_edits_existing_members(tmp_path):
    raw = document_project(authored_documents())
    task = raw["documents"]["task"]["payload"]
    task["variables"]["catch"]["label"] = "Catch trial"
    task["epochs"]["hold"]["label"] = "Hold still"
    task["validation_trials"] = {"cohort": {
        "label": "Validation {k} · {load}",
        "over": {"k": "0..1", "load": {"small": 1, "big": {"value": 3, "label": "Heavy"}}},
        "variables": {"catch": "k > 0"}, "epoch_steps": {"hold": "5 + load"},
    }}
    app = create_app(projects_dir=tmp_path)
    folders = app.state.project_folders
    _save_project(folders, "families", Project.model_validate(raw))
    body = {"revision": folders.read("families").revision,
            "run": {"variant": "default", "step": "train", "row": "default"}}
    path = "/task/validation_trials/cohort"

    def change(*edits):
        return {"kind": "task", "edits": list(edits)}

    with TestClient(app) as client:
        view = _post(client, body)
        assert not view["problems"], view["problems"]
        assert list(view["task"]["validation_trials"]) == ["cohort"]
        assert view["families"][path] == {
            "member_count": 4,
            "indices": {"k": {"count": 2}, "load": {"count": 2, "values": [
                {"key": "small", "name": "small"}, {"key": "big", "name": "Heavy"},
            ]}},
        }
        sequence = _field(view, path + "/over/k")
        assert sequence["text"] == "0..1"
        assert not view["sites"][sequence["site"]]["names"]
        named = _field(view, path + "/over/load/big")
        assert named["text"] == "3"
        assert named["label"] == "Heavy"
        assert not view["sites"][named["site"]]["names"]
        variable = _field(view, path + "/variables/catch")
        assert variable["text"] == "k > 0"
        assert variable["label"] == "Catch trial"
        assert [name["id"] for name in view["sites"][variable["site"]]["names"]] == ["k", "load"]
        epoch = _field(view, path + "/epoch_steps/hold")
        site = view["sites"][epoch["site"]]
        assert (epoch["text"], epoch["label"]) == ("5 + load", "Hold still")
        assert site["structural"] and site["expected"]["base"] == "integer"

        body = _save(client, body, _post(client, body, _edit(path + "/over/load/big", "4")))
        body = _save(client, body, _post(client, body, change(
            {"op": "add", "path": "/validation_trials/cohort/over/load/small/label", "value": "Light"},
            {"op": "add", "path": "/validation_trials/cohort/combination", "value": "zip"},
            {"op": "replace", "path": "/validation_trials/cohort/label", "value": "Renamed {k} · {load}"},
        )))
        body = _save(client, body, _post(client, body, _edit(path + "/variables/catch", "k > 1")))
        body = _save(client, body, _post(client, body, _edit(path + "/epoch_steps/hold", "6 + load")))
        reopened = _post(client, body)
        assert reopened["families"][path]["member_count"] == 2
        assert _field(reopened, path + "/over/load/small")["label"] == "Light"
        assert _field(reopened, path + "/over/load/big")["text"] == "4"
        assert _field(reopened, path + "/variables/catch")["text"] == "k > 1"
        assert _field(reopened, path + "/epoch_steps/hold")["text"] == "6 + load"
        trial = reopened["task"]["validation_trials"]["cohort"]
        assert trial["label"] == "Renamed {k} · {load}"
        assert trial["over"]["load"] == {
            "small": {"value": 1, "label": "Light"}, "big": {"value": 4, "label": "Heavy"},
        }
        assert not reopened["problems"], reopened["problems"]

        body = _save(client, body, _post(client, body, change(
            {"op": "remove", "path": "/validation_trials/cohort/over/load/small/label"},
            {"op": "remove", "path": "/validation_trials/cohort/over/load/big"},
        )))
        assert _post(client, body)["task"]["validation_trials"]["cohort"]["over"]["load"] == {"small": 1}

        body = _save(client, body, _post(client, body, change(
            {"op": "remove", "path": "/validation_trials/cohort/over/load"},
            {"op": "remove", "path": "/validation_trials/cohort/over/k"},
            {"op": "replace", "path": "/validation_trials/cohort/label", "value": "Plain trial"},
            {"op": "replace", "path": "/validation_trials/cohort/variables/catch", "value": True},
            {"op": "replace", "path": "/validation_trials/cohort/epoch_steps/hold", "value": 7},
        )))
        plain = _post(client, body)
        assert path not in plain.get("families", {})
        trial = plain["task"]["validation_trials"]["cohort"]
        assert trial["label"] == "Plain trial"
        assert "over" not in trial
        assert "combination" not in trial


@pytest.mark.parametrize("role", ["objective", "task"])
def test_family_problem_from_later_combination_uses_authored_path(tmp_path, role):
    raw = document_project(authored_documents())
    if role == "objective":
        owner = raw["documents"]["objective"]["payload"]["terms"]["effort"]
        owner.update(weight="1 + k", over={"k": {
            "passing": 0, "failing": {"value": True, "label": "Later choice"},
        }})
        path = "/objective/terms/effort/weight"
    else:
        raw["documents"]["task"]["payload"]["validation_trials"] = {"cohort": {
            "label": "Trial {k}",
            "over": {"k": {"passing": True, "failing": {"value": 0, "label": "Later choice"}}},
            "variables": {"catch": "k"},
        }}
        path = "/task/validation_trials/cohort/variables/catch"
    app = create_app(projects_dir=tmp_path)
    folders = app.state.project_folders
    _save_project(folders, "families", Project.model_validate(raw))
    body = {"revision": folders.read("families").revision,
            "run": {"variant": "default", "step": "train", "row": "default"}}
    with TestClient(app) as client:
        view = _post(client, body)
    problems = _field(view, path)["problems"]
    assert len(problems) == 1, view["problems"]
    problem = problems[0]
    assert pointer(tuple(problem["path"])) == path
    assert "k=Later choice" in problem["reason"]


def test_validation_trial_draft_problem_starts_at_the_section(tmp_path):
    raw = document_project(authored_documents())
    raw["documents"]["task"]["payload"]["validation_trials"] = {"cohort": {
        "label": "Trial {k}", "over": {"k": "0..1"}, "epoch_steps": {"hold": "1 % k"},
    }}
    app = create_app(projects_dir=tmp_path)
    folders = app.state.project_folders
    _save_project(folders, "families", Project.model_validate(raw))
    body = {"revision": folders.read("families").revision,
            "run": {"variant": "default", "step": "train", "row": "default"}}
    with TestClient(app) as client:
        view = _post(client, body)
    assert any(problem["path"] == ["validation_trials", "cohort", "epoch_steps", "hold"] for problem in view["problems"])
    assert all(problem["path"][:2] != ["task", "validation_trials"] for problem in view["problems"])


def test_task_view_size_tracks_authored_validation_and_objective_families(tmp_path):
    raw = document_project(authored_documents())
    task = raw["documents"]["task"]["payload"]
    term = raw["documents"]["objective"]["payload"]["terms"]["effort"]
    app = create_app(projects_dir=tmp_path)
    folders = app.state.project_folders
    validation = "/task/validation_trials/"
    objective = "/objective/terms/effort"
    replies = []

    with TestClient(app) as client:
        for count in (4, 32, 96):
            rule = f"0..{count - 1}"
            task["validation_trials"] = {
                "reference": {"label": "Reference", "variables": {"catch": False}},
                "grid": {"label": "Trial {k}", "over": {"k": rule}, "variables": {"catch": "k > 0"}},
            }
            term.update(over={"k": rule}, weight="1 + k")
            _save_project(folders, "families", Project.model_validate(raw))
            response = client.post("/api/projects/families/authoring/trials", json={
                "revision": folders.read("families").revision,
                "run": {"variant": "default", "step": "train", "row": "default"},
            })
            assert response.status_code == 200, response.text
            view = response.json()
            assert not view["problems"], view["problems"]
            assert view["families"][validation + "grid"]["member_count"] == count
            assert view["families"][objective]["member_count"] == count
            paths = [field["path"] for field in view["fields"]]
            assert [path for path in paths if path.startswith(validation)] == [
                validation + "reference/variables/catch",
                validation + "grid/over/k",
                validation + "grid/variables/catch",
            ]
            assert all(path.startswith(objective + "/") for path in paths if path.startswith("/objective/terms/"))
            replies.append((len(response.content), len(paths)))

    assert len({fields for _, fields in replies}) == 1, replies
    lengths = [size for size, _ in replies]
    # Only the written range texts and count digits grow here.
    assert max(lengths) - min(lengths) <= 64, replies


def test_task_view_fields_track_authored_set_entries(tmp_path):
    raw = document_project(authored_documents())
    task = raw["documents"]["task"]["payload"]
    task["validation_trials"] = {"reference": {"label": "Reference", "variables": {"catch": False}}}
    differences = {"/interior/wires/task-wire/attachments": {"value": {"pulse": {
        "kind": "intervention", "operation": "Scale",
        "operands": {"factor": {"kind": "task", "variable": "amount"}},
    }}}}
    entry = {
        "label": "Pulse {k}", "differences": differences,
        "task_variables": {"amount": {"label": "Amount", "type": {"base": "real"}, "value": "1 + k"}},
    }
    raw["documents"]["bank"] = {
        "label": "Bank", "schema": SET_DOCUMENT_SCHEMA_ID, "payload": {"entries": {"pulse": entry}},
    }
    raw["steps"]["train"]["set_fields"] = {"pulse": {"axis": {"set": "bank"}, "sampling": {"kind": "listed"}}}
    app = create_app(projects_dir=tmp_path)
    folders = app.state.project_folders
    field_counts = []
    with TestClient(app) as client:
        for count in (2, 12, 24):
            entry["over"] = {"k": f"0..{count - 1}"}
            _save_project(folders, "families", Project.model_validate(raw))
            response = client.post("/api/projects/families/authoring/trials", json={
                "revision": folders.read("families").revision,
                "run": {"variant": "default", "step": "train", "row": "default"},
            })
            assert response.status_code == 200, response.text
            view = response.json()
            assert not view["problems"], view["problems"]
            placed, = view["set_fields"][0]["entries"]
            assert placed["member_count"] == count
            representative, = [value for value in view["variables"] if "set_entry" in value]
            variables = task["variables"].keys() | {representative["id"]}
            assert set(view["task"]["variables"]) == variables
            assert {value["id"] for value in view["variables"]} == variables
            assert set(view["task"]["signals"]) == task["signals"].keys()
            paths = [field["path"] for field in view["fields"]]
            assert {path.split("/")[3] for path in paths if path.startswith("/task/variables/")} == variables
            assert {path.split("/")[3] for path in paths if path.startswith("/task/signals/")} == task["signals"].keys()
            assert _field(view, f"/task/variables/{representative['id']}/value")["text"] == "1 + k"
            field_counts.append(len(paths))

    assert len(set(field_counts)) == 1, field_counts
