import json

import pytest
from fastapi.testclient import TestClient

from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.objective_document import STORED_OBJECTIVE_DOCUMENT_SCHEMA_ID
from feedbax.contracts.stored_model import STORED_MODEL_DOCUMENT_SCHEMA_ID
from feedbax.contracts.task_document import STORED_TASK_DOCUMENT_SCHEMA_ID
from feedbax.web.app import create_app


URL = "/api/projects/study"
MODEL_LINKS = [
    ModelAddress.at_state((), "body", "position").to_text(),
    ModelAddress.at_state((), "body", "velocity").to_text(),
]


@pytest.fixture
def study(tmp_path):
    def write(relative, data):
        path = tmp_path / "study" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data) + "\n")
        return path

    write("study.feedbax", {
        "schema_version": "feedbax.project.v1", "store": "results", "sweep": {}, "variants": ["check"],
    })
    write("variants/check.json", {"label": "Check"})
    write("steps/fit.json", {"kind": "train", "label": "Fit", "base": {"inline": {}}})
    objective = {
        "id": "objective",
        "terms": {
            key: {"measure": {"kind": "literal", "value": value}}
            for key, value in [("first", 1.0), ("second", 2.0), ("last", 3.0)]
        },
    }
    write("documents/loss.json", {
        "label": "Loss", "schema": STORED_OBJECTIVE_DOCUMENT_SCHEMA_ID, "payload": objective,
    })
    write("documents/based.json", {
        "label": "Based", "schema": STORED_OBJECTIVE_DOCUMENT_SCHEMA_ID, "base": {"document": "loss"},
        "differences": {"/terms/first/weight": {"value": {"kind": "literal", "value": 4.0}}},
    })
    write("documents/opaque.json", {
        "label": "Opaque", "schema": "notes", "payload": objective | {"values": [1, 2]},
    })
    write("documents/model.json", {
        "label": "Model", "schema": STORED_MODEL_DOCUMENT_SCHEMA_ID, "payload": {
            "step_seconds": 0.01,
            "ports": {"inputs": {"first": {"label": "First", "type": {}}, "last": {"label": "Last", "type": {}}}},
            "interior": {},
        },
    })
    write("documents/task.json", {
        "label": "Task", "schema": STORED_TASK_DOCUMENT_SCHEMA_ID, "payload": {
            "id": "trials",
            "variables": {
                key: {"label": key, "type": {"base": "real"}, "value": {"kind": "literal", "value": 1.0}}
                for key in ("first", "last")
            },
            "epochs": {"move": {"label": "Move", "end": {"kind": "rest"}}},
            "signals": {},
            "links": dict(zip(MODEL_LINKS, ("first", "last"), strict=True)),
        },
    })
    return tmp_path / "study", TestClient(create_app(projects_dir=tmp_path))


def _snapshot(client):
    response = client.get(URL)
    assert response.status_code == 200, response.text
    return response.json()


def _patch(client, current, kind, name, *edits):
    return client.patch(URL, json={
        "revision": current["revision"], "address": {"kind": kind, "id": name}, "patches": edits,
    })


def _order(path, *keys):
    return {"op": "order", "path": path, "order": list(keys)}


def _file(snapshot, name):
    return next(item for item in snapshot["file_revisions"] if item["address"] == {"kind": "document", "id": name})


def test_an_order_edit_writes_and_replays_the_typed_member_order(study):
    root, client = study
    path = root / "documents/loss.json"
    original = path.read_bytes()
    before = _snapshot(client)

    response = _patch(client, before, "document", "loss", _order("/payload/terms", "last", "no-longer-there"))
    assert response.status_code == 200, response.text
    after = response.json()
    written = path.read_bytes()
    assert list(json.loads(written)["payload"]["terms"]) == ["last", "first", "second"]
    assert list(after["project"]["documents"]["loss"]["payload"]["terms"]) == ["last", "first", "second"]
    assert before["revision"] != after["revision"]

    for current, desired, data in [(after, before, original), (before, after, written)]:
        replay = client.post(f"{URL}/history/replay", json={
            "revision": current["revision"], "files": [{
                "address": _file(current, "loss")["address"],
                "before": _file(current, "loss")["revision"],
                "after": _file(desired, "loss")["data"],
            }],
        })
        assert replay.status_code == 200, replay.text
        assert path.read_bytes() == data
        assert replay.json()["revision"] == desired["revision"]


def test_order_can_name_model_ports_and_address_keyed_task_links(study):
    _, client = study
    model = _patch(client, _snapshot(client), "document", "model", _order("/payload/ports/inputs", "last"))
    assert model.status_code == 200, model.text
    assert list(model.json()["project"]["documents"]["model"]["payload"]["ports"]["inputs"]) == ["last", "first"]

    task = _patch(client, model.json(), "document", "task", _order("/payload/links", MODEL_LINKS[1]))
    assert task.status_code == 200, task.text
    assert list(task.json()["project"]["documents"]["task"]["payload"]["links"]) == MODEL_LINKS[::-1]


@pytest.mark.parametrize(("kind", "name", "edits"), [
    ("document", "loss", [_order("/payload/terms", "last", "last")]),
    ("document", "loss", [_order("/payload/terms", " ")]),
    ("document", "loss", [_order("/payload/terms/first", "measure")]),
    ("document", "loss", [_order("/payloads/terms", "last")]),
    ("document", "model", [_order("/payload/ports/outputs", "missing")]),
    ("document", "based", [_order("/differences", "/terms/first/weight")]),
    ("document", "based", [_order("/payload/terms", "last")]),
    ("document", "opaque", [_order("/payload/terms", "last")]),
    ("document", "opaque", [
        {"op": "replace", "path": "/schema", "value": STORED_OBJECTIVE_DOCUMENT_SCHEMA_ID},
        _order("/payload/terms", "last"),
    ]),
    ("step", "fit", [_order("/payload/terms", "last")]),
    ("variant", "check", [_order("/payload/terms", "last")]),
])
def test_order_refuses_invalid_authored_targets_without_writing(study, kind, name, edits):
    _, client = study
    before = _snapshot(client)
    response = _patch(client, before, kind, name, *edits)
    assert response.status_code == 422, response.text
    assert _snapshot(client)["revision"] == before["revision"]


def test_ordinary_authored_array_insertion_is_still_available(study):
    _, client = study
    response = _patch(client, _snapshot(client), "document", "opaque", {
        "op": "add", "path": "/payload/values/0", "value": 0,
    })
    assert response.status_code == 200, response.text
    assert response.json()["project"]["documents"]["opaque"]["payload"]["values"] == [0, 1, 2]
