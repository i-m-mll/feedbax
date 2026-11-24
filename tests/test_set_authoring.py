"""Duplicate Run, edits of a selected value, Save Set and Inline Set through the API, as Studio stores them."""

from copy import deepcopy

import pytest
from fastapi.testclient import TestClient

from feedbax.contracts.project import Project
from feedbax.web.app import create_app
from feedbax.web.services.project_folder import ProjectFolderService
from tests.support.managed_train_fixture import managed_train_project
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


def _set_edit(folders, client, command, *, store=None):
    """Prepare a set edit and store it as Studio does: its documents, then its files."""
    revision = folders.read(PROJECT).revision
    body = {"revision": revision, "command": command}
    reply = client.post(f"{BASE}/authoring/sets", json=body)
    assert reply.status_code == 200, reply.text
    edit = reply.json()
    if store is not None:
        assert edit["files"] == [] and folders.read(PROJECT).revision == revision
        choice = next(option for option in edit["store_choices"] if option["kind"] == store)
        reply = client.post(f"{BASE}/authoring/sets", json=body | {"store_choice": choice["id"]})
        assert reply.status_code == 200, reply.text
        edit = reply.json()
    else:
        assert not edit.get("store_choices"), edit
    if edit.get("created") or edit["files"]:
        written = client.patch(BASE, json={
            "revision": revision, "created": edit.get("created", []), "files": edit["files"],
        })
        assert written.status_code == 200, written.text
    return edit


def _duplicate(folders, client, step, rows=(), axes=None, *, store=None):
    row = next(iter(rows), None) or next(iter(_rows(folders, step)))
    command = {"kind": "duplicate_run", "run": {"variant": "default", "step": step, "row": row}, "rows": list(rows)}
    return _set_edit(folders, client, command | ({"axes": axes} if axes else {}), store=store)["selected"]


def _model_edit(folders, client, step, document, operation, rows, axes=None, chosen=(), *, store=None):
    snapshot = folders.read(PROJECT)
    body = {
        "revision": snapshot.revision, "presentation_revision": snapshot.presentation_revision,
        "document": document, "command": {"operation": operation}, "step": step, "rows": list(rows),
        "chosen_steps": list(chosen),
    } | ({"axes": axes} if axes else {})
    reply = client.post(f"{BASE}/model/edit", json=body)
    assert reply.status_code == 200, reply.text
    result = reply.json()["result"]
    if store is not None:
        assert result["status"] == "asked" and folders.read(PROJECT).revision == snapshot.revision
        choice = next(option for option in result["store_choices"] if option["kind"] == store)
        reply = client.post(f"{BASE}/model/edit", json=body | {"store_choice": choice["id"]})
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
    copies = [key for key, row in after.items() if "run" in row.coordinates]
    assert [after[key].coordinates["rate"].value_id for key in copies] == ["r3"]
    assert row in after and "run" not in after[row].coordinates


def test_duplicate_run_keeps_every_run_and_every_reader_of_it(studio):
    folders, client = studio
    readers = ("pair", "one", "over")
    before = {step: _runs(folders, step) for step in ("five", *readers)}
    authored = {step: folders.read(PROJECT).project.steps[step] for step in readers}
    assert len(before["pair"]) == 2
    _duplicate(folders, client, "five")
    after = {step: _runs(folders, step) for step in ("five", *readers)}
    assert len(after["five"]) == 10 and before["five"].items() <= after["five"].items()
    assert set(after["five"].values()) == set(before["five"].values())
    assert {step: after[step] for step in ("pair", "one")} == {step: before[step] for step in ("pair", "one")}
    assert before["over"].items() <= after["over"].items() and len(after["over"]) == 10
    assert {step: folders.read(PROJECT).project.steps[step] for step in readers} == authored


def test_a_reader_of_every_run_keeps_its_own_exceptions_after_a_duplicate(studio):
    folders, client = studio
    row = sorted(_runs(folders, "over"))[0]
    revision = folders.read(PROJECT).revision
    written = client.patch(BASE, json={"revision": revision, "files": [{
        "address": {"kind": "step", "id": "over"},
        "patches": [{"op": "add", "path": "/sweep/exceptions", "value": {"drop": {"where": {"rows": [row]}, "exclude": True}}}],
    }]})
    assert written.status_code == 200, written.text
    assert len(_runs(folders, "over")) == 4
    _duplicate(folders, client, "five")
    plan = folders.read_plan(PROJECT, revision=None).plan
    assert not [problem for problem in plan.diagnostics if problem.step == "over"]
    assert row not in _runs(folders, "over") and len(_runs(folders, "over")) == 9


def test_a_set_made_by_duplicate_run_is_edited_saved_applied_narrowed_and_inlined(studio, tmp_path):
    folders, client = studio
    train_before = _runs(folders, "train")
    (original,) = _runs(folders, "baseline").values()

    assert _duplicate(folders, client, "baseline") == {"axis": "run", "value": "run-2"}
    assert _duplicate(folders, client, "baseline", ["run=run-2"], {"run": ["run-2"]}) == {"axis": "run", "value": "run-3"}
    copies = _runs(folders, "baseline")
    assert sorted(copies) == ["default", "run=run-2", "run=run-3"]
    assert set(copies.values()) == {original}

    _model_edit(folders, client, "baseline", "reach-model", INSERT, ["run=run-2"], {"run": ["run-2"]})
    edited = _runs(folders, "baseline")
    assert {row for row in edited if edited[row] != copies[row]} == {"run=run-2"}
    assert _runs(folders, "train") == train_before
    assert folders.read(PROJECT).project.documents["reach-model"] == Project.model_validate(_raw()).documents["reach-model"]

    created = _set_edit(folders, client, {"kind": "save_set", "step": "baseline", "axis": "run"})["created"]
    assert [item["id"] for item in created] == ["set"]
    assert _runs(folders, "baseline") == edited
    assert folders.read(PROJECT).project.steps["baseline"].sweep.axes["run"].optional

    revision = folders.read(PROJECT).revision
    offered = client.post(f"{BASE}/authoring/sets-offered", json={"revision": revision, "step": "other"})
    assert offered.json() == {"sets": ["set"]}
    _set_edit(folders, client, {"kind": "apply_set", "step": "other", "set": "set"})
    other = _runs(folders, "other")
    assert sorted(other) == ["set=run-2", "set=run-3"]

    field = {"op": "set_field", "address": {"composites": [], "element": {
        "kind": "field", "parent": {"kind": "component", "id": "plant"}, "path": ["fields", "mass"],
    }}, "value": 2.5}
    result = _model_edit(
        folders, client, "other", "reach-model", field, ["set=run-2"], {"set": ["run-2"]}, store="selected_readers",
    )
    project = folders.read(PROJECT).project
    assert project.documents["set-2"].base.document == "set"
    assert project.steps["other"].sweep.axes["set"].set == "set-2"
    assert project.steps["baseline"].sweep.axes["run"].set == "set"
    assert _runs(folders, "baseline") == edited
    narrowed = _runs(folders, "other")
    assert {row for row in narrowed if narrowed[row] != other[row]} == {"set=run-2"}
    assert result["snapshot"]["revision"] == folders.read(PROJECT).revision

    _set_edit(folders, client, {"kind": "inline_set", "step": "baseline", "axis": "run"})
    inlined = folders.read(PROJECT).project.steps["baseline"].sweep.axes["run"]
    assert "set" not in inlined.model_fields_set and inlined.optional
    assert _runs(folders, "baseline") == edited
    assert not any("model_attachments" in path.read_text() for path in (tmp_path / PROJECT).rglob("*.json"))


def test_a_copied_set_entry_keeps_its_originals_request_id_on_every_step_reading_the_set(studio):
    folders, client = studio
    _duplicate(folders, client, "baseline")
    _model_edit(folders, client, "baseline", "reach-model", INSERT, ["run=run-2"], {"run": ["run-2"]})
    assert _duplicate(folders, client, "baseline", ["default"]) == {"axis": "run", "value": "run-3"}
    _set_edit(folders, client, {"kind": "save_set", "step": "baseline", "axis": "run"})
    _set_edit(folders, client, {"kind": "apply_set", "step": "other", "set": "set"})
    unexcepted = _runs(folders, "other")["set=run-2"]
    revision = folders.read(PROJECT).revision
    written = client.patch(BASE, json={"revision": revision, "files": [{
        "address": {"kind": "step", "id": "other"},
        "patches": [{"op": "add", "path": "/sweep/exceptions", "value": {"slow": {
            "where": {"rows": ["set=run-2"]},
            "patches": {"settings.optimizer.learning_rate.value": {"op": "replace", "value": 0.0005}},
        }}}],
    }]})
    assert written.status_code == 200, written.text
    assert _runs(folders, "other")["set=run-2"] != unexcepted

    copy = _duplicate(folders, client, "other", ["set=run-2"], {"set": ["run-2"]}, store="set_entry")
    assert copy == {"axis": "set", "value": "run-4"}
    other, baseline = _runs(folders, "other"), _runs(folders, "baseline")
    assert other["set=run-4"] == other["set=run-2"] != other["set=run-3"]
    assert baseline["run=run-4"] == baseline["run=run-2"] != baseline["run=run-3"]
    assert sorted(baseline) == ["default", "run=run-2", "run=run-3", "run=run-4"]


def test_removing_the_last_copy_removes_the_axis(studio):
    folders, client = studio
    before = _runs(folders, "baseline")
    _duplicate(folders, client, "baseline")
    _set_edit(folders, client, {"kind": "remove_value", "step": "baseline", "axis": "run", "value": "run-2"})
    assert folders.read(PROJECT).project.steps["baseline"].sweep is None or not folders.read(
        PROJECT).project.steps["baseline"].sweep.axes
    assert _runs(folders, "baseline") == before


def test_to_field_moves_the_set_into_one_train_and_to_sweep_restores_it(studio):
    folders, client = studio
    train_before = _runs(folders, "train")
    for _ in range(3):
        _duplicate(folders, client, "other", ["default"])
    _set_edit(folders, client, {"kind": "save_set", "step": "other", "axis": "run"})
    before = folders.read(PROJECT).project.steps["other"].sweep.axes["run"]
    sweep = _runs(folders, "other")
    assert len(sweep) == 4

    _set_edit(folders, client, {"kind": "set_to_field", "step": "other", "axis": "run"})
    assert list(_runs(folders, "other")) == ["default"]
    revision = folders.read(PROJECT).revision
    reply = client.post(f"{BASE}/authoring/trials", json={
        "revision": revision, "run": {"variant": "default", "step": "other", "row": "default"},
    })
    assert reply.status_code == 200, reply.text
    (field,) = reply.json()["set_fields"]
    assert field["axis"] == "run" and field["set"] == before.set
    assert field["entries"][0]["entry"] is None
    assert all(item["weight"] == 1 for item in field["entries"])
    assert _runs(folders, "train") == train_before

    _set_edit(folders, client, {"kind": "set_to_sweep", "step": "other", "axis": "run"})
    assert _runs(folders, "other") == sweep
    assert not folders.read(PROJECT).project.steps["other"].set_fields


def test_required_set_can_be_emptied_and_repaired(tmp_path):
    raw = managed_train_project().model_dump(mode="json", by_alias=True, exclude_unset=True)
    raw["documents"]["draft"] = {"schema": "feedbax.set_document.v1", "payload": {"entries": {"one": {}}}}
    raw["steps"]["train"]["sweep"] = {"axes": {"draft": {"set": "draft"}}}
    folders = ProjectFolderService(tmp_path)
    _save_project(folders, PROJECT, Project.model_validate(raw))
    with TestClient(create_app(projects_dir=tmp_path)) as client:
        emptied = _set_edit(folders, client, {"kind": "remove_value", "step": "train", "axis": "draft", "value": "one"})
        assert emptied["files"] and emptied["problems"][0]["address"] == "documents/draft.json"
        assert emptied["problems"][0]["authored_path"] == "/payload/entries"
        snapshot = folders.read(PROJECT)
        assert snapshot.project.documents["draft"].payload["entries"] == {}
        assert snapshot.project.steps["train"].sweep.axes["draft"].set == "draft"
        assert any(problem.authored_path == "/payload/entries"
                   for problem in folders.read_plan(PROJECT, revision=None).plan.diagnostics)
        repaired = _set_edit(folders, client, {"kind": "add_set_entry", "step": "train", "axis": "draft"})
        assert repaired["files"] and not repaired["problems"] and _runs(folders, "train")


def test_applying_an_inapplicable_set_stores_the_reference_and_marks_the_difference(tmp_path):
    raw = managed_train_project().model_dump(mode="json", by_alias=True, exclude_unset=True)
    pointer = "/interior/components/removed/fields/mass"
    raw["documents"]["draft"] = {"schema": "feedbax.set_document.v1", "payload": {
        "entries": {"heavy": {"differences": {pointer: {"value": 2.0}}}},
    }}
    folders = ProjectFolderService(tmp_path)
    _save_project(folders, PROJECT, Project.model_validate(raw))
    with TestClient(create_app(projects_dir=tmp_path)) as client:
        applied = _set_edit(folders, client, {"kind": "apply_set", "step": "train", "set": "draft"})
        assert applied["files"] and applied["problems"][0]["address"] == "documents/draft.json"
        assert applied["problems"][0]["authored_path"].startswith("/payload/entries/heavy/differences/")
        assert folders.read(PROJECT).project.steps["train"].sweep.axes["draft"].set == "draft"
        plan = folders.read_plan(PROJECT, revision=None).plan
        assert any(problem.document == "draft" and pointer in problem.detail for problem in plan.diagnostics)

        inlined = _set_edit(folders, client, {"kind": "inline_set", "step": "train", "axis": "draft"})
        assert inlined["files"] and inlined["problems"][0]["address"] == "steps/train.json"
        path = inlined["problems"][0]["authored_path"]
        assert path.startswith("/sweep/axes/draft/values/heavy/patches/")
        assert any(problem.authored_path.startswith(path) for problem in folders.read_plan(PROJECT, revision=None).plan.diagnostics
                   if problem.authored_path is not None)
