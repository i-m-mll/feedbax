"""Rename a project document or step, and duplicate and use a document, through Studio's routes."""

from copy import deepcopy
import json

from fastapi.testclient import TestClient
import pytest

from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.project import Project
from feedbax.web.app import create_app
from tests.test_canonical_evaluate_admission import _save_project
from tests.test_document_train_pair import authored_documents, document_project

PROJECT = "documents"
URL = f"/api/projects/{PROJECT}"
STEPS = ("train", "train-b")
MASS = {
    "composites": [],
    "element": {"kind": "field", "parent": {"kind": "component", "id": "plant"}, "path": ["fields", "mass"]},
}
MASS_TEXT = ModelAddress.model_validate(MASS).to_text()


@pytest.fixture
def studio(tmp_path):
    app = create_app(projects_dir=tmp_path / "projects")
    with TestClient(app) as client:
        folders = app.state.project_folders
        raw = document_project(authored_documents())
        raw["steps"]["train-b"] = deepcopy(raw["steps"]["train"])
        # An attachment batch is named by its author, so a batch that shares a document's name is not a reference.
        raw["steps"]["train-b"]["model_attachments"] = {
            "reach-model": [{"op": "set_field", "address": MASS, "value": 1.4}],
        }
        model = raw["documents"]["reach-model"]["payload"]
        controller = model["interior"]["components"]["wrapper"]["interior"]["components"]["controller"]
        reset = ModelAddress.model_validate({
            "composites": ["wrapper", "controller"],
            "element": {"kind": "port", "end": {
                "owner": "component", "component": "cell", "port": "hidden", "direction": "output",
            }},
        })
        controller["interior"]["wires"]["carry"]["attachments"][0]["initial"] = {
            "kind": "name", "reference": reset.to_text(),
        }
        _save_project(folders, PROJECT, Project.model_validate(raw))
        (folders.root / PROJECT / "presentation.json").write_text(json.dumps({
            "documents": {"reach-model": {"positions": {
                ModelAddress.model_validate({
                    "composites": [], "element": {"kind": "component", "id": "wrapper"},
                }).to_text(): {"x": 1, "y": 2},
            }}},
        }))
        yield client, folders


def _snapshot(client):
    response = client.get(URL)
    assert response.status_code == 200, response.text
    return response.json()


def _command(client, snapshot, command, status=200):
    response = client.post(f"{URL}/members/command", json={
        "revision": snapshot["revision"], "presentation_revision": snapshot["presentation_revision"],
        "command": command,
    })
    assert response.status_code == status, response.text
    return response.json()


def _requests(folders, revision):
    plan = folders.read_plan(PROJECT, revision=revision).plan
    assert not plan.diagnostics, plan.diagnostics
    return {run.key.step: run.request_id for run in plan.runs}


def _files(folders):
    root = folders.root / PROJECT
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*") if path.is_file() and not path.name.startswith(".")
    }


def _replay(client, before, after):
    prior = {(item["address"]["kind"], item["address"]["id"]): item for item in before["file_revisions"]}
    current = {(item["address"]["kind"], item["address"]["id"]): item for item in after["file_revisions"]}
    replay = [
        {"address": {"kind": kind, "id": id}, "before": current[kind, id]["revision"] if (kind, id) in current else None,
         "after": prior[kind, id]["data"] if (kind, id) in prior else None}
        for kind, id in prior.keys() | current.keys()
        if prior.get((kind, id), {}).get("revision") != current.get((kind, id), {}).get("revision")
    ]
    if before["declaration"]["authored_revision"] != after["declaration"]["authored_revision"]:
        replay.append({
            "address": {"kind": "declaration", "id": "declaration"},
            "before": after["declaration"]["revision"], "after": before["declaration"]["data"],
        })
    return client.post(f"{URL}/history/replay", json={
        "revision": after["revision"], "files": replay,
        "presentation": {"before": after["presentation_revision"], "after": before["presentation"]},
    })


def test_rename_preserves_document_bytes_and_requests_through_undo(studio):
    client, folders = studio
    raw = _snapshot(client)["project"]
    # A document whose content mentions another document's name is not a reference to it.
    raw["documents"]["notes"] = {
        "schema": "application/json",
        "payload": {"document": "reach-model", "label": "documents.reach-model.payload.name"},
    }
    raw["sweep"] = {"rows": {"nominal": {"patches": {f"documents.reach-model.{MASS_TEXT}": {"value": 2.0}}}}}
    _save_project(folders, PROJECT, Project.model_validate(raw))
    before = _snapshot(client)
    files = _files(folders)
    requests = _requests(folders, before["revision"])
    assert requests.keys() == set(STEPS)

    result = _command(client, before, {"kind": "rename", "family": "documents", "member": "reach-model", "name": "arm"})
    after = result["snapshot"]
    assert result["member"] == "arm"
    assert "reach-model" not in after["project"]["documents"]
    for step in STEPS:
        trial = after["project"]["steps"][step]["base"]["inline"]["trial_documents"]
        assert trial["model"] == {"document": "arm"} and "model_name" not in trial
    assert set(after["project"]["steps"]["train-b"]["model_attachments"]) == {"reach-model"}
    assert after["project"]["sweep"]["rows"]["nominal"]["patches"] == {f"documents.arm.{MASS_TEXT}": {"value": 2.0}}
    assert after["presentation"]["documents"] == {"arm": before["presentation"]["documents"]["reach-model"]}
    # A document's name is outside request identity, so both Train steps keep their results.
    assert _requests(folders, after["revision"]) == requests

    documents = {name: data for name, data in files.items() if name.startswith("documents/")}
    documents["documents/arm.json"] = documents.pop("documents/reach-model.json")
    assert {name: data for name, data in _files(folders).items() if name.startswith("documents/")} == documents

    # A client's Undo replays the prior bytes of every changed file and the presentation.
    response = _replay(client, before, after)
    assert response.status_code == 200, response.text
    restored = _files(folders)
    assert restored.keys() == files.keys()
    assert all(restored[name] == files[name] for name in files if name != "presentation.json")
    assert response.json()["presentation"] == before["presentation"]


def test_rename_refuses_a_name_already_in_use(studio):
    client, folders = studio
    snapshot = _snapshot(client)
    _command(client, snapshot, {"kind": "rename", "family": "documents", "member": "reach-model", "name": "task"}, 409)
    raw = snapshot["project"]
    raw["steps"]["train-b"]["base"]["inline"]["trial_documents"]["objective"] = {"document": "missing"}
    _save_project(folders, PROJECT, Project.model_validate(raw))
    snapshot = _snapshot(client)
    _command(client, snapshot, {"kind": "rename", "family": "documents", "member": "objective", "name": "missing"}, 422)


def test_a_step_using_a_duplicate_diverges_alone(studio):
    client, folders = studio
    snapshot = _snapshot(client)
    original = _requests(folders, snapshot["revision"])
    result = _command(client, snapshot, {"kind": "duplicate", "family": "documents", "member": "task"})
    copy = result["member"]
    assert copy == "task-2"
    assert result["snapshot"]["project"]["documents"][copy] == snapshot["project"]["documents"]["task"]

    response = client.post(f"{URL}/authoring/trials", json={
        "revision": result["snapshot"]["revision"], "run": {"variant": "default", "step": "train", "row": "default"},
        "change": {"kind": "select_document", "role": "task", "document": copy},
    })
    assert response.status_code == 200, response.text
    view = response.json()
    response = client.patch(URL, json={"revision": view["revision"], "files": view["files"]})
    assert response.status_code == 200, response.text
    used = response.json()
    assert used["project"]["steps"]["train"]["base"]["inline"]["trial_documents"]["task"] == {"document": copy}
    assert used["project"]["steps"]["train-b"] == snapshot["project"]["steps"]["train-b"]

    response = client.patch(URL, json={
        "revision": used["revision"], "address": {"kind": "document", "id": copy},
        "patches": [{"op": "replace", "path": "/payload/variables/0/value/args/0/args/0/value", "value": -0.35}],
    })
    assert response.status_code == 200, response.text
    edited = _requests(folders, response.json()["revision"])
    assert edited["train"] != original["train"]
    assert edited["train-b"] == original["train-b"]


def test_a_step_uses_a_duplicated_model_whose_links_and_measures_resolve(studio):
    client, folders = studio
    before = _snapshot(client)
    requests = _requests(folders, before["revision"])
    result = _command(client, before, {"kind": "duplicate", "family": "documents", "member": "reach-model", "name": "arm"})
    assert result["snapshot"]["presentation"]["documents"]["arm"] == before["presentation"]["documents"]["reach-model"]
    files = _files(folders)
    assert files["documents/arm.json"] == files["documents/reach-model.json"]
    run = {"variant": "default", "step": "train-b", "row": "default"}
    response = client.post(f"{URL}/authoring/trials", json={
        "revision": result["snapshot"]["revision"], "run": run,
        "change": {"kind": "select_document", "role": "model", "document": "arm"},
    })
    assert response.status_code == 200, response.text
    view = response.json()
    response = client.patch(URL, json={"revision": view["revision"], "files": view["files"]})
    assert response.status_code == 200, response.text
    revision = response.json()["revision"]
    trial = response.json()["project"]["steps"]["train-b"]["base"]["inline"]["trial_documents"]
    assert trial["model"] == {"document": "arm"} and "model_name" not in trial
    assert trial["task"] == {"document": "task"} and trial["objective"] == {"document": "objective"}
    assert _requests(folders, revision) == requests
    plan = folders.read_plan(PROJECT, revision=revision).plan
    # Model addresses are relative to their document, so the copy's links and measures resolve as the original's.
    assert {run.key.step for run in plan.runs} == {"train", "train-b"}
    assert not plan.diagnostics, plan.diagnostics

    response = client.post(f"{URL}/authoring/trials", json={"revision": revision, "run": run})
    assert response.status_code == 200, response.text
    authored = response.json()
    assert authored["model"] == "arm" and not authored["problems"]


def test_choosing_a_document_of_another_role_is_refused(studio):
    client, _ = studio
    response = client.post(f"{URL}/authoring/trials", json={
        "revision": _snapshot(client)["revision"], "run": {"variant": "default", "step": "train", "row": "default"},
        "change": {"kind": "select_document", "role": "model", "document": "task"},
    })
    assert response.status_code == 422, response.text



@pytest.fixture
def evaluated(tmp_path):
    app = create_app(projects_dir=tmp_path / "projects")
    with TestClient(app) as client:
        folders = app.state.project_folders
        raw = document_project(authored_documents(), evaluation=True)
        target = "steps.train." + MASS_TEXT
        raw["sweep"] = {"axes": {"mass": {"values": {
            name: {"value": value, "patches": {target: {"value": value}}} for name, value in (("nominal", 1), ("heavy", 2))
        }}}}
        raw["steps"]["train"]["group"] = "fitting"
        _save_project(folders, PROJECT, Project.model_validate(raw))
        (folders.root / PROJECT / "presentation.json").write_text(
            '{"steps": {"train": {"sweep_order": ["mass"], "folded_sweep_id": "mass"}}}\n',
        )
        yield client, folders


def _step_requests(folders, revision):
    plan = folders.read_plan(PROJECT, revision=revision).plan
    assert not plan.diagnostics, plan.diagnostics
    return {(run.key.variant, run.key.step, run.key.row): run.request_id for run in plan.runs}


def test_a_step_rename_rewrites_its_references_and_keeps_every_request(evaluated):
    client, folders = evaluated
    before = _snapshot(client)
    files = _files(folders)
    requests = _step_requests(folders, before["revision"])
    assert {step for _, step, _ in requests} == {"train", "baseline"}

    result = _command(client, before, {"kind": "rename", "family": "steps", "member": "train", "name": "fit"})
    after = result["snapshot"]
    assert result["member"] == "fit"
    steps = after["project"]["steps"]
    assert set(steps) == {"fit", "baseline"} and steps["fit"]["group"] == "fitting"
    assert steps["baseline"]["inputs"]["subject"]["step"] == "fit"
    targets = {target for value in after["project"]["sweep"]["axes"]["mass"]["values"].values() for target in value["patches"]}
    assert targets and all(target.startswith("steps.fit.{") for target in targets)
    assert after["presentation"]["steps"] == {"fit": {"sweep_order": ["mass"], "folded_sweep_id": "mass"}}
    renamed = _files(folders)
    assert "steps/fit.json" in renamed and "steps/train.json" not in renamed
    assert _step_requests(folders, after["revision"]) == {
        (variant, "fit" if step == "train" else step, row): request for (variant, step, row), request in requests.items()
    }

    response = _replay(client, before, after)
    assert response.status_code == 200, response.text
    restored = _files(folders)
    assert restored.keys() == files.keys()
    # Replay re-encodes the declaration it restores.
    assert response.json()["project"] == before["project"]
    assert all(restored[name] == files[name] for name in files if name not in {"presentation.json", "documents.feedbax"})
    assert response.json()["presentation"] == before["presentation"]


def test_a_step_rename_refuses_a_taken_name(evaluated):
    client, folders = evaluated
    snapshot = _snapshot(client)
    _command(client, snapshot, {"kind": "rename", "family": "steps", "member": "train", "name": "baseline"}, 409)
    _command(client, snapshot, {"kind": "rename", "family": "steps", "member": "missing", "name": "other"}, 404)
