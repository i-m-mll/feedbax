"""A Model's added, removed and rewired elements against what it differs from, and their Reset."""

import json

from fastapi.testclient import TestClient

from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.stored_model import STORED_MODEL_DOCUMENT_SCHEMA_ID, StoredModelDocument
from feedbax.contracts.project import Project
from feedbax.contracts.project_io import STUDIO_PRESENTATION_FILENAME, load_text_project
from feedbax.contracts.project_resolution import compose_document
from feedbax.contracts.set_document import SET_DOCUMENT_SCHEMA_ID, read_set
from feedbax.web.app import create_app
from feedbax.web.services.project_folder import ProjectFolderService
from tests.test_canonical_evaluate_admission import _save_project
from tests.test_document_train_pair import authored_documents, document_project

_URL = "/api/projects/ghost"
_FORCE = {"owner": "component", "component": "wrapper", "port": "force"}
_SPARE = ModelAddress.model_validate({"composites": [], "element": {"kind": "component", "id": "spare"}})
_PLANT_WIRE = ModelAddress.model_validate({"composites": [], "element": {"kind": "wire", "id": "plant-wire"}})
_SPARE_WIRE = {"source": _FORCE, "target": {"owner": "component", "component": "spare", "port": "force"}, "attachments": {}}
_REWIRED = {"value": {"owner": "component", "component": "other", "port": "force"}}


def _project_with_spare():
    raw = document_project(authored_documents())
    model = raw["documents"]["reach-model"]["payload"]
    for name in ("spare", "other"):
        model["interior"]["components"][name] = {"kind": "leaf", "label": name, "type": "feedbax.PointMass", "fields": {"mass": 1.0}}
    model["interior"]["wires"]["spare-wire"] = {key: end for key, end in _SPARE_WIRE.items() if key != "attachments"}
    return raw


def _project(tmp_path, differences):
    raw = _project_with_spare()
    raw["documents"]["variant"] = {"label": "Variant", "schema": STORED_MODEL_DOCUMENT_SCHEMA_ID, "base": {"document": "reach-model"}, "differences": differences}
    raw["steps"]["train"]["base"]["inline"]["trial_documents"]["model"] = {"document": "variant"}
    folders = ProjectFolderService(tmp_path)
    _save_project(folders, "ghost", Project.model_validate(raw))
    (tmp_path / "ghost" / STUDIO_PRESENTATION_FILENAME).write_text(json.dumps({"documents": {
        "reach-model": {"positions": {_SPARE.to_authored_text(): {"x": 40.0, "y": 80.0}}},
    }}))
    return folders


def _composed(tmp_path):
    return StoredModelDocument.model_validate(
        compose_document(load_text_project(tmp_path / "ghost").project.documents.get, "variant").payload,
    ).lower()


def test_a_removed_component_carries_its_base_wiring_and_its_reset_restores_it(tmp_path):
    removal = {"/interior/components/spare": {"op": "remove"}, "/interior/wires/spare-wire": {"op": "remove"}}
    folders = _project(tmp_path, removal | {"/interior/wires/plant-wire/target": _REWIRED})
    with TestClient(create_app(projects_dir=tmp_path)) as client:
        snapshot = client.get(_URL).json()
        revision = snapshot["revision"]
        (row,) = folders.read_plan("ghost", revision=revision).plan.rows
        response = client.post(f"{_URL}/model/projection", json={
            "revision": revision, "document": "variant",
            "run": {"variant": row.key.variant, "step": row.key.step, "row": row.key.row},
        })
        assert response.status_code == 200, response.text
        differences = response.json()["base_differences"]
        assert [(item["change"], item["address"]) for item in differences] == [
            ("removed", _SPARE.model_dump(mode="json")), ("rewired", _PLANT_WIRE.model_dump(mode="json")),
        ]
        removed, rewired = differences
        assert removed["wires"] == {"spare-wire": _SPARE_WIRE} and removed["position"] == {"x": 40.0, "y": 80.0}
        assert rewired["previous"]["target"] == {"owner": "component", "component": "plant", "port": "force"}

        reply = client.post(f"{_URL}/model/edit", json={
            "revision": revision, "presentation_revision": snapshot["presentation_revision"], "document": "variant",
            "command": {"operation": {"op": "reset_difference", "address": _SPARE.model_dump(mode="json")}},
        })
        assert reply.status_code == 200, reply.text
        assert reply.json()["result"]["status"] == "applied", reply.text
    reset = _composed(tmp_path)
    stored = load_text_project(tmp_path / "ghost").project.documents["variant"].differences
    assert all(path.startswith("/interior/wires/plant-wire/") for path in stored), stored
    _project(tmp_path / "never", {"/interior/wires/plant-wire/target": _REWIRED})
    assert reset == _composed(tmp_path / "never")
    presentation = json.loads((tmp_path / "ghost" / STUDIO_PRESENTATION_FILENAME).read_text())
    assert presentation["documents"]["variant"]["positions"][_SPARE.to_authored_text()] == {"x": 40.0, "y": 80.0}


def test_a_set_reset_restores_a_removed_component_in_base_order(tmp_path):
    raw = _project_with_spare()
    kept = {
        "/interior/wires/plant-wire/target": _REWIRED,
        "/interior/wires/local-wire": {"value": {
            "source": _FORCE, "target": {"owner": "component", "component": "plant", "port": "force"},
        }},
    }
    removal = {"/interior/components/spare": {"op": "remove"}, "/interior/wires/spare-wire": {"op": "remove"}}
    raw["documents"]["arms"] = {"label": "Arms", "schema": SET_DOCUMENT_SCHEMA_ID, "payload": {"entries": {
        "keep": {"label": "Keep"}, "drop": {"label": "Drop", "differences": removal | kept},
    }}}
    raw["steps"]["train"]["sweep"] = {"axes": {"arm": {"set": "arms"}}}
    folders = ProjectFolderService(tmp_path)
    _save_project(folders, "ghost", Project.model_validate(raw))

    with TestClient(create_app(projects_dir=tmp_path)) as client:
        snapshot = client.get(_URL).json()
        before = folders.read_plan("ghost", revision=snapshot["revision"]).plan
        selected = next(row.key for row in before.rows if row.key.row == "arm=drop")
        reply = client.post(f"{_URL}/model/edit", json={
            "revision": snapshot["revision"], "presentation_revision": snapshot["presentation_revision"],
            "document": "reach-model", "step": "train", "rows": [selected.row], "axes": {"arm": ["drop"]},
            "command": {"operation": {"op": "reset_difference", "address": _SPARE.model_dump(mode="json")}},
        })
        assert reply.status_code == 200, reply.text
        result = reply.json()["result"]
        assert result["status"] == "applied", result

    project = load_text_project(tmp_path / "ghost").project
    entry = read_set(project.documents.get, "arms").entries["drop"]
    assert {path: patch.model_dump(mode="json", exclude_unset=True) for path, patch in entry.differences.items()} == kept
    after = folders.read_plan("ghost", revision=result["snapshot"]["revision"], source_runs=(selected,)).plan
    base = project.documents["reach-model"].payload["interior"]
    restored = after.selected_definitions[selected].payload["trial_documents"]["model"]["interior"]
    assert list(restored["components"]) == list(base["components"])
    assert list(restored["wires"]) == [*base["wires"], "local-wire"]
    assert restored["wires"]["spare-wire"] == base["wires"]["spare-wire"]
