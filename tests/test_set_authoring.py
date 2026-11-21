"""Duplicate Run, edits of a selected value, Save Set and Inline Set through the API, as Studio stores them."""

from copy import deepcopy

import pytest
from fastapi.testclient import TestClient

from feedbax.contracts.project import Project
from feedbax.web.app import create_app
from feedbax.web.services.project_folder import ProjectFolderService
from tests.test_canonical_evaluate_admission import _save_project
from tests.test_document_train_pair import authored_documents, document_project

PROJECT = "sets"
BASE = f"/api/projects/{PROJECT}"
GAIN = {
    "kind": "leaf", "name": "Gain", "type": "feedbax.Scale", "revision": None, "fields": {"scale": 0.5},
    "port_counts": {}, "attachments": {"sources": {}, "sinks": {}},
}
INSERT = {"op": "insert_component", "address": {"composites": [], "element": {"kind": "wire", "id": "effector-wire"}},
          "component": GAIN, "input_port": "input", "output_port": "output", "position": {"x": 0, "y": 0}}


def _raw():
    raw = document_project(authored_documents(), evaluation=True)
    raw["steps"]["other"] = deepcopy(raw["steps"]["train"])
    five = deepcopy(raw["steps"]["train"])
    five["sweep"] = {"axes": {"rate": {"values": {
        f"r{index}": {"patches": {"settings.optimizer.learning_rate.value": {"op": "replace", "value": index / 1000}}}
        for index in range(1, 6)
    }}}}
    raw["steps"]["five"] = five
    over = deepcopy(raw["steps"]["baseline"])
    over["inputs"]["subject"]["step"] = "five"
    over["sweep"] = {"axes": {"model": {"input": "subject"}}}
    raw["steps"]["over"] = over
    pair = deepcopy(over)
    pair["inputs"]["subject"]["where"] = {"rows": [
        {"key": key, "variant": "default", "row": row} for key, row in (("a", "rate=r1"), ("b", "rate=r3"))
    ]}
    raw["steps"]["pair"] = pair
    one = deepcopy(raw["steps"]["baseline"])
    one["inputs"]["subject"] |= {"step": "five", "row": "rate=r2"}
    raw["steps"]["one"] = one
    return raw


@pytest.fixture
def studio(tmp_path):
    folders = ProjectFolderService(tmp_path)
    _save_project(folders, PROJECT, Project.model_validate(_raw()))
    with TestClient(create_app(projects_dir=tmp_path)) as client:
        yield folders, client


def _runs(folders, step):
    plan = folders.read_plan(PROJECT, revision=None).plan
    return {run.key.row: run.request_id for run in plan.runs if run.key.step == step}


def _rows(folders, step):
    plan = folders.read_plan(PROJECT, revision=None).plan
    return {row.key.row: row for row in plan.rows if row.key.step == step}


def _set_edit(folders, client, command):
    """Prepare a set edit and store it as Studio does: its documents, then its files."""
    revision = folders.read(PROJECT).revision
    reply = client.post(f"{BASE}/authoring/sets", json={"revision": revision, "command": command})
    assert reply.status_code == 200, reply.text
    edit = reply.json()
    for created in edit.get("created", []):
        made = client.post(f"{BASE}/files", json={"revision": revision, "file": created})
        assert made.status_code == 201, made.text
        revision = made.json()["snapshot"]["revision"]
    if edit["files"]:
        written = client.patch(BASE, json={"revision": revision, "files": edit["files"]})
        assert written.status_code == 200, written.text
    return edit


def _duplicate(folders, client, step, rows=(), axes=None):
    row = next(iter(rows), None) or next(iter(_rows(folders, step)))
    command = {"kind": "duplicate_run", "run": {"variant": "default", "step": step, "row": row}, "rows": list(rows)}
    return _set_edit(folders, client, command | ({"axes": axes} if axes else {}))["selected"]


def _model_edit(folders, client, step, document, operation, rows, axes=None, chosen=()):
    snapshot = folders.read(PROJECT)
    reply = client.post(f"{BASE}/model/edit", json={
        "revision": snapshot.revision, "presentation_revision": snapshot.presentation_revision,
        "document": document, "command": {"operation": operation}, "step": step, "rows": list(rows),
        "chosen_steps": list(chosen),
    } | ({"axes": axes} if axes else {}))
    assert reply.status_code == 200, reply.text
    result = reply.json()["result"]
    assert result["status"] == "applied", result
    return result


@pytest.mark.parametrize(("selected", "expected"), [(1, 6), (5, 10), (2, 7)])
def test_duplicate_run_copies_each_selected_run_of_an_evaluate_over_five_models(studio, selected, expected):
    folders, client = studio
    before = _runs(folders, "over")
    assert len(before) == 5
    rows = sorted(before)[:selected]
    copy = _duplicate(folders, client, "over", rows)
    after = _runs(folders, "over")
    assert len(after) == expected
    assert set(after.values()) == set(before.values())
    assert copy == {"axis": "run", "value": "run-2"}


def test_duplicate_run_selected_by_a_carried_coordinate_is_conditioned_on_it(studio):
    folders, client = studio
    rows = _rows(folders, "over")
    (row,) = (key for key, row in rows.items() if row.coordinates["rate"].value_id == "r3")
    _duplicate(folders, client, "over", [row], {"rate": ["r3"]})
    after = _rows(folders, "over")
    assert len(after) == 6
    assert sorted(key for key, row in after.items() if row.coordinates["rate"].value_id == "r3") == sorted(
        key for key, row in after.items() if "run" in row.coordinates and row.coordinates["rate"].value_id == "r3"
    )


def test_duplicate_run_keeps_the_steps_that_select_its_runs_by_id_on_the_same_runs(studio):
    folders, client = studio
    before = {step: _runs(folders, step) for step in ("pair", "one")}
    assert len(before["pair"]) == 2
    _duplicate(folders, client, "five")
    assert len(_runs(folders, "five")) == 10
    assert {step: _runs(folders, step) for step in ("pair", "one")} == before
    project = folders.read(PROJECT).project
    assert project.steps["one"].inputs["subject"].where.rows[0].model_dump() == {
        "key": "rate=r2", "variant": "default", "row": "rate=r2--run=run",
    }
    assert [row.row for row in project.steps["pair"].inputs["subject"].where.rows] == [
        "rate=r1--run=run", "rate=r3--run=run",
    ]


def test_a_set_made_by_duplicate_run_is_edited_saved_applied_narrowed_and_inlined(studio, tmp_path):
    folders, client = studio
    train_before = _runs(folders, "train")
    (original,) = _runs(folders, "baseline").values()

    assert _duplicate(folders, client, "baseline") == {"axis": "run", "value": "run-2"}
    assert _duplicate(folders, client, "baseline", ["run=run-2"], {"run": ["run-2"]}) == {"axis": "run", "value": "run-3"}
    copies = _runs(folders, "baseline")
    assert sorted(copies) == ["run=run", "run=run-2", "run=run-3"]
    assert set(copies.values()) == {original}

    _model_edit(folders, client, "baseline", "reach-model", INSERT, ["run=run-2"], {"run": ["run-2"]})
    edited = _runs(folders, "baseline")
    assert {row for row in edited if edited[row] != copies[row]} == {"run=run-2"}
    assert _runs(folders, "train") == train_before
    assert folders.read(PROJECT).project.documents["reach-model"] == Project.model_validate(_raw()).documents["reach-model"]

    created = _set_edit(folders, client, {"kind": "save_set", "step": "baseline", "axis": "run"})["created"]
    assert [item["id"] for item in created] == ["set"]
    assert _runs(folders, "baseline") == edited

    revision = folders.read(PROJECT).revision
    offered = client.post(f"{BASE}/authoring/sets-offered", json={"revision": revision, "step": "other"})
    assert offered.json() == {"sets": ["set"]}
    _set_edit(folders, client, {"kind": "apply_set", "step": "other", "set": "set"})
    other = _runs(folders, "other")
    assert sorted(other) == ["set=run", "set=run-2", "set=run-3"]

    field = {"op": "set_field", "address": {"composites": [], "element": {
        "kind": "field", "parent": {"kind": "component", "id": "plant"}, "path": ["fields", "mass"],
    }}, "value": 2.5}
    result = _model_edit(folders, client, "other", "reach-model", field, ["set=run-2"], {"set": ["run-2"]})
    project = folders.read(PROJECT).project
    assert project.documents["set-2"].base.document == "set"
    assert project.steps["other"].sweep.axes["set"].set == "set-2"
    assert project.steps["baseline"].sweep.axes["run"].set == "set"
    assert _runs(folders, "baseline") == edited
    narrowed = _runs(folders, "other")
    assert {row for row in narrowed if narrowed[row] != other[row]} == {"set=run-2"}
    assert result["snapshot"]["revision"] == folders.read(PROJECT).revision

    _set_edit(folders, client, {"kind": "inline_set", "step": "baseline", "axis": "run"})
    assert "set" not in folders.read(PROJECT).project.steps["baseline"].sweep.axes["run"].model_fields_set
    assert _runs(folders, "baseline") == edited
    assert not any("model_attachments" in path.read_text() for path in (tmp_path / PROJECT).rglob("*.json"))


def test_removing_the_last_copy_removes_the_axis(studio):
    folders, client = studio
    before = _runs(folders, "baseline")
    _duplicate(folders, client, "baseline")
    _set_edit(folders, client, {"kind": "remove_value", "step": "baseline", "axis": "run", "value": "run-2"})
    assert folders.read(PROJECT).project.steps["baseline"].sweep is None or not folders.read(
        PROJECT).project.steps["baseline"].sweep.axes
    assert _runs(folders, "baseline") == before
