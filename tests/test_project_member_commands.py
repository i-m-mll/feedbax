"""Rename a project document or step, and duplicate and use a document, through Studio's routes."""

from copy import deepcopy
import json

from fastapi.testclient import TestClient
import pytest

from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.project import Project
from feedbax.web.app import create_app
from tests.test_canonical_evaluate_admission import _save_project
from tests.support.managed_train_fixture import managed_evaluation_step
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


def test_change_key_preserves_document_bytes_and_requests_through_undo(studio):
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

    result = _command(client, before, {"kind": "change_key", "family": "documents", "member": "reach-model", "key": "arm"})
    after = result["snapshot"]
    assert result["member"] == "arm"
    assert "reach-model" not in after["project"]["documents"]
    for step in STEPS:
        trial = after["project"]["steps"][step]["base"]["inline"]["trial_documents"]
        assert trial["model"] == {"document": "arm"} and "model_name" not in trial
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


def test_change_key_preserves_bound_and_dangling_readers_through_reload_and_undo(studio):
    client, folders = studio
    snapshot = _snapshot(client)
    untouched = _files(folders)
    _command(client, snapshot, {"kind": "change_key", "family": "documents", "member": "reach-model", "key": "task"}, 409)
    assert _files(folders) == untouched
    raw = snapshot["project"]
    raw["steps"]["train-b"]["base"]["inline"]["trial_documents"]["objective"] = {"document": "missing"}
    _save_project(folders, PROJECT, Project.model_validate(raw))
    snapshot = _snapshot(client)
    before = folders.read_plan(PROJECT, revision=None).plan
    result = _command(client, snapshot, {
        "kind": "change_key", "family": "documents", "member": "objective", "key": "missing",
    })
    reopened = _snapshot(client)
    readers = reopened["project"]["steps"]
    assert readers["train"]["base"]["inline"]["trial_documents"]["objective"] == {"document": "missing"}
    assert readers["train-b"]["base"]["inline"]["trial_documents"]["objective"] == {
        "document": "missing", "unresolved": True,
    }
    after = folders.read_plan(PROJECT, revision=None).plan
    assert {run.key: run.request_id for run in after.runs} == {run.key: run.request_id for run in before.runs}
    assert any(
        item["address"] == "steps/train-b.json" and item["path"].endswith("objective")
        for item in result["problems"]
    )
    undone = _replay(client, snapshot, reopened)
    assert undone.status_code == 200, undone.text
    assert _snapshot(client)["project"] == snapshot["project"]


def test_a_step_using_a_duplicate_diverges_alone(studio):
    client, folders = studio
    snapshot = _snapshot(client)
    original = _requests(folders, snapshot["revision"])
    result = _command(client, snapshot, {"kind": "duplicate", "family": "documents", "member": "task"})
    copy = result["member"]
    assert copy == "task-2"
    assert result["snapshot"]["project"]["documents"][copy] == {
        **snapshot["project"]["documents"]["task"], "label": "Task 2",
    }

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
        target = "steps.train.base.inline.training_config.batch_size"
        raw["sweep"] = {"axes": {"batch": {"values": {
            name: {"value": value, "patches": {target: {"value": value}}} for name, value in (("nominal", 4), ("heavy", 8))
        }}}}
        raw["steps"]["train"]["group"] = "fitting"
        _save_project(folders, PROJECT, Project.model_validate(raw))
        (folders.root / PROJECT / "presentation.json").write_text(
            '{"steps": {"train": {"sweep_order": ["batch"]}}}\n',
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

    result = _command(client, before, {"kind": "change_key", "family": "steps", "member": "train", "key": "fit"})
    after = result["snapshot"]
    assert result["member"] == "fit"
    steps = after["project"]["steps"]
    assert set(steps) == {"fit", "baseline"} and steps["fit"]["group"] == "fitting"
    assert steps["baseline"]["inputs"]["subject"]["step"] == "fit"
    targets = {target for value in after["project"]["sweep"]["axes"]["batch"]["values"].values() for target in value["patches"]}
    assert targets and all(target.startswith("steps.fit.") for target in targets)
    assert after["presentation"]["steps"] == {"fit": {"sweep_order": ["batch"]}}
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
    _command(client, snapshot, {"kind": "change_key", "family": "steps", "member": "train", "key": "baseline"}, 409)
    _command(client, snapshot, {"kind": "change_key", "family": "steps", "member": "missing", "key": "other"}, 404)


def test_a_step_duplicate_shares_its_sources_requests_until_it_differs(evaluated):
    client, folders = evaluated
    before = _snapshot(client)
    files = _files(folders)
    requests = _step_requests(folders, before["revision"])

    result = _command(client, before, {"kind": "duplicate", "family": "steps", "member": "train"})
    after = result["snapshot"]
    assert result["member"] == "train-2"
    steps = after["project"]["steps"]
    assert steps["train-2"] == steps["train"] == before["project"]["steps"]["train"]
    assert set(after["project"]["documents"]) == set(before["project"]["documents"])
    assert after["presentation"]["steps"]["train-2"] == before["presentation"]["steps"]["train"]
    duplicated = _step_requests(folders, after["revision"])
    variants = {variant for variant, _, _ in requests}
    assert len(variants) == 2
    # The project's sweep patches the copy as it patches its source, so each variant's copy is the source's request.
    for (variant, step, row), request in requests.items():
        assert duplicated[variant, step, row] == request
        if step == "train":
            assert duplicated[variant, "train-2", row] == request
    assert _files(folders)["steps/train-2.json"] == files["steps/train.json"]

    # One Undo removes the step.
    response = _replay(client, before, after)
    assert response.status_code == 200, response.text
    assert response.json()["project"] == before["project"]
    assert _files(folders).keys() == files.keys()

    again = _command(client, response.json(), {"kind": "duplicate", "family": "steps", "member": "train"})["snapshot"]
    response = client.patch(URL, json={
        "revision": again["revision"], "address": {"kind": "step", "id": "train-2"},
        "patches": [{"op": "replace", "path": "/base/inline/training_config/n_batches", "value": 3}],
    })
    assert response.status_code == 200, response.text
    edited = _step_requests(folders, response.json()["revision"])
    assert {key: edited[key] for key in requests} == requests
    assert all(edited[variant, "train-2", row] != request for (variant, step, row), request in requests.items() if step == "train")


def test_a_duplicate_of_a_continuation_is_named_for_its_kind_and_continues_its_parent(evaluated):
    client, folders = evaluated
    before = _snapshot(client)
    requests = _step_requests(folders, before["revision"])

    result = _command(client, before, {"kind": "duplicate", "family": "steps", "member": "baseline"})
    assert result["member"] == "evaluate"
    assert result["snapshot"]["project"]["steps"]["evaluate"]["inputs"]["subject"] == {"step": "train", "checkpoint": "last"}
    duplicated = _step_requests(folders, result["snapshot"]["revision"])
    assert {key: duplicated[key] for key in requests} == requests
    assert {(variant, row): request for (variant, step, row), request in duplicated.items() if step == "evaluate"} == {
        (variant, row): request for (variant, step, row), request in requests.items() if step == "baseline"
    }


MASS_POINTER = "/interior/components/plant/fields/mass"


@pytest.fixture
def based(tmp_path):
    """`train` reads `reach-model`; `train-b` reads `model-2`, based on it with a heavier plant."""
    app = create_app(projects_dir=tmp_path / "projects")
    with TestClient(app) as client:
        folders = app.state.project_folders
        raw = document_project(authored_documents())
        raw["documents"]["model-2"] = {
            "schema": raw["documents"]["reach-model"]["schema"], "base": {"document": "reach-model"},
            "differences": {MASS_POINTER: {"value": 3.0}},
        }
        raw["steps"]["train-b"] = deepcopy(raw["steps"]["train"])
        raw["steps"]["train-b"]["base"]["inline"]["trial_documents"]["model"] = {"document": "model-2"}
        _save_project(folders, PROJECT, Project.model_validate(raw))
        yield client, folders


def _change_base(client, folders):
    """Change a value of `reach-model` that `model-2` inherits, and return the new revision."""
    raw = _snapshot(client)["project"]
    readout = raw["documents"]["reach-model"]["payload"]["interior"]["components"]["wrapper"]["interior"][
        "components"]["controller"]["interior"]["components"]["readout"]
    readout["fields"]["use_bias"] = False
    _save_project(folders, PROJECT, Project.model_validate(raw))
    return _snapshot(client)["revision"]


def _point(client, snapshot, step, document):
    response = client.patch(URL, json={
        "revision": snapshot["revision"], "address": {"kind": "step", "id": step},
        "patches": [{"op": "replace", "path": "/base/inline/trial_documents/model", "value": {"document": document}}],
    })
    assert response.status_code == 200, response.text
    return response.json()


def test_detach_keeps_requests_and_stops_following_the_base(based):
    client, folders = based
    before = _snapshot(client)
    requests = _requests(folders, before["revision"])
    assert requests["train"] != requests["train-b"]

    after = _command(client, before, {"kind": "detach", "member": "model-2"})["snapshot"]
    detached = after["project"]["documents"]["model-2"]
    assert "base" not in detached and detached["payload"]["interior"]["components"]["plant"]["fields"]["mass"] == 3.0
    assert _requests(folders, after["revision"]) == requests

    edited = _requests(folders, _change_base(client, folders))
    assert edited["train"] != requests["train"] and edited["train-b"] == requests["train-b"]
    _command(client, _snapshot(client), {"kind": "detach", "member": "model-2"}, 422)


def test_duplicate_of_a_based_document_is_a_root_with_its_content(based):
    client, folders = based
    before = _snapshot(client)
    requests = _requests(folders, before["revision"])
    result = _command(client, before, {"kind": "duplicate", "family": "documents", "member": "model-2"})
    assert result["member"] == "model-3"
    copied = result["snapshot"]["project"]["documents"]["model-3"]
    assert "base" not in copied and copied["label"] == "Model 3"
    pointed = _point(client, result["snapshot"], "train-b", "model-3")
    assert _requests(folders, pointed["revision"]) == requests
    # The copy is unrelated to `reach-model`, whose later changes do not reach it.
    edited = _requests(folders, _change_base(client, folders))
    assert edited["train"] != requests["train"] and edited["train-b"] == requests["train-b"]


def test_change_key_of_a_base_rewrites_the_documents_based_on_it(based):
    client, folders = based
    before = _snapshot(client)
    requests = _requests(folders, before["revision"])
    after = _command(client, before, {"kind": "change_key", "family": "documents", "member": "reach-model", "key": "arm"})["snapshot"]
    assert after["project"]["documents"]["model-2"]["base"] == {"document": "arm"}
    assert _requests(folders, after["revision"]) == requests
    _command(client, after, {"kind": "change_key", "family": "documents", "member": "objective", "key": "reach-model"})
    raw = _snapshot(client)["project"]
    raw["documents"]["orphan"] = {"schema": raw["documents"]["arm"]["schema"], "base": {"document": "gone"}}
    _save_project(folders, PROJECT, Project.model_validate(raw))
    _command(client, _snapshot(client), {"kind": "change_key", "family": "documents", "member": "arm", "key": "gone"})
    reopened = _snapshot(client)
    assert reopened["project"]["documents"]["orphan"]["base"] == {"document": "gone", "unresolved": True}
    assert reopened["project"]["documents"]["model-2"]["base"] == {"document": "gone"}
    assert any(item["address"] == "documents/orphan.json" for item in reopened["problems"])


def test_a_document_label_changes_no_key_reference_or_request(based):
    client, folders = based
    before = _snapshot(client)
    requests = _requests(folders, before["revision"])
    files = _files(folders)
    labelled = _command(client, before, {"kind": "rename", "family": "documents", "member": "reach-model", "label": "Arm"})
    after = labelled["snapshot"]
    assert labelled["member"] == "reach-model"
    assert after["project"]["documents"]["reach-model"]["label"] == "Arm"
    assert after["project"]["documents"]["model-2"]["base"] == {"document": "reach-model"}
    assert "label" not in after["project"]["documents"]["model-2"]
    assert _requests(folders, after["revision"]) == requests
    assert {name for name, data in _files(folders).items() if files.get(name) != data} == {"documents/reach-model.json"}

    variant = _command(client, after, {"kind": "rename", "family": "documents", "member": "model-2", "label": "Heavy"})
    detached = _command(client, variant["snapshot"], {"kind": "detach", "member": "model-2"})["snapshot"]
    assert detached["project"]["documents"]["model-2"]["label"] == "Heavy"
    keyed = _command(client, detached, {"kind": "change_key", "family": "documents", "member": "reach-model", "key": "arm"})
    assert keyed["snapshot"]["project"]["documents"]["arm"]["label"] == "Arm"
    cleared = _command(client, keyed["snapshot"], {"kind": "rename", "family": "documents", "member": "arm", "label": None})
    assert "label" not in cleared["snapshot"]["project"]["documents"]["arm"]
    assert _requests(folders, cleared["snapshot"]["revision"]) == requests


def test_detach_stores_a_problem_for_the_evaluation_that_loses_its_model_base(based):
    client, folders = based
    raw = _snapshot(client)["project"]
    raw["steps"]["baseline"] = managed_evaluation_step("train")
    raw["steps"]["baseline"]["base"]["inline"]["trial_documents"] = {"model": {"document": "model-2"}}
    # A based document no evaluation's relation passes through may be detached.
    raw["documents"]["model-4"] = deepcopy(raw["documents"]["model-2"])
    _save_project(folders, PROJECT, Project.model_validate(raw))
    snapshot = _snapshot(client)
    assert not folders.read_plan(PROJECT, revision=snapshot["revision"]).plan.diagnostics
    detached = _command(client, snapshot, {"kind": "detach", "member": "model-2"})
    saved = _snapshot(client)
    assert "base" not in saved["project"]["documents"]["model-2"]
    assert any(
        item["step"] == "baseline" and item["path"] == "payload.trial_documents.model"
        for item in detached["problems"]
    )
    state = client.post(f"{URL}/model/evaluate-model", json={
        "revision": saved["revision"], "scope": {"kind": "step"},
        "run": {"variant": "default", "step": "baseline", "row": "default"},
    })
    assert state.status_code == 200, state.text
    assert state.json().get("resolved_model") is None
    assert any(item["path"] == "payload.trial_documents.model" for item in state.json()["problems"])
    assert any(run.key.step == "train" for run in folders.read_plan(PROJECT, revision=saved["revision"]).plan.runs)
