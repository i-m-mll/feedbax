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
    pair["inputs"]["subject"]["where"] = {"clauses": [{"rate": [{"id": value, "label": value} for value in ("r1", "r3")]}]}
    raw["steps"]["pair"] = pair
    one = deepcopy(over)
    one["inputs"]["subject"]["where"] = {"clauses": [{"rate": [{"id": "r2", "label": "r2"}]}]}
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


def test_duplicate_run_also_widens_readers_of_its_axis_values(studio):
    folders, client = studio
    readers = ("pair", "one", "over")
    before = {step: _runs(folders, step) for step in ("five", *readers)}
    authored = {step: folders.read(PROJECT).project.steps[step] for step in readers}
    assert len(before["pair"]) == 2
    _duplicate(folders, client, "five")
    after = {step: _runs(folders, step) for step in ("five", *readers)}
    assert len(after["five"]) == 10 and before["five"].items() <= after["five"].items()
    assert set(after["five"].values()) == set(before["five"].values())
    for step in readers:
        assert before[step].items() <= after[step].items()
        assert len(after[step]) == 2 * len(before[step])
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


def test_set_entry_task_edits_reach_only_selected_sweep_and_trial_rows(studio):
    folders, client = studio
    entry = _duplicate(folders, client, "other", ["default"])["value"]
    _set_edit(folders, client, {"kind": "save_set", "step": "other", "axis": "run"})
    variable = {
        "id": "onset", "name": "Onset", "type": {"base": "real", "dimensions": {"time": 1}},
        "value": {"kind": "literal", "value": 0.03, "unit": "s"},
    }
    _set_edit(folders, client, {
        "kind": "set_entry_task", "step": "other", "axis": "run", "entry": entry, "task_variables": [variable],
    })

    def selected(row, change=None):
        revision = folders.read(PROJECT).revision
        response = client.post(f"{BASE}/authoring/trials", json={
            "revision": revision, "run": {"variant": "default", "step": "other", "row": row},
        } | ({"change": change} if change is not None else {}))
        assert response.status_code == 200, response.text
        return response.json()

    def set_expression(row, id, text):
        revision = folders.read(PROJECT).revision
        edited = selected(row, {
            "kind": "expression", "path": f"/task/variables/{id}/value", "edit": {"kind": "text", "text": text},
        })
        assert not edited.get("store_choices") and edited["files"], edited
        response = client.patch(BASE, json={"revision": revision, "files": edited["files"]})
        assert response.status_code == 200, response.text

    field = folders.read(PROJECT).project.steps["other"].sweep.axes["run"]
    origin = {"step": "other", "axis": "run", "set": field.set, "entry": entry, "variable": "onset"}
    for row in _rows(folders, "other"):
        read = selected(row)
        for variables in (read["variables"], read["task"]["variables"]):
            assert any(item["id"] == variable["id"] for item in variables) == (row == f"run={entry}")
        assert [item["set_entry"] for item in read["variables"] if "set_entry" in item] == (
            [origin] if row == f"run={entry}" else []
        )
    set_expression(f"run={entry}", "onset", "0.04 s")
    assert selected(f"run={entry}")["task"]["variables"][-1]["value"]["value"] == 0.04
    _set_edit(folders, client, {"kind": "set_to_field", "step": "other", "axis": "run"})
    read = selected("default")
    for variables in (read["variables"], read["task"]["variables"]):
        assert len([item for item in variables if item["name"] == f"{entry} · Onset"]) == 1
    projected = next(item for item in read["variables"] if item["name"] == f"{entry} · Onset")
    assert projected["set_entry"] == origin
    scoped = projected["id"]
    set_expression("default", scoped, "0.07 s")
    field = folders.read(PROJECT).project.steps["other"].set_fields["run"]
    project = folders.read(PROJECT).project
    source = project.documents[field.axis.set].payload["entries"][entry]
    assert source["task_variables"][0]["id"] == "onset"
    assert source["task_variables"][0]["value"]["value"] == 0.07


def test_a_set_sweep_adds_variables_to_an_evaluates_inherited_task(tmp_path):
    from feedbax.contracts.set_document import SET_DOCUMENT_SCHEMA_ID

    raw = document_project(authored_documents(), evaluation=True)
    variable = {
        "id": "onset", "name": "Onset", "type": {"base": "real", "dimensions": {"time": 1}},
        "value": {"kind": "literal", "value": 0.03, "unit": "s"},
    }
    raw["documents"]["entry-set"] = {
        "schema": SET_DOCUMENT_SCHEMA_ID, "payload": {"entries": {"ordinary": {}, "pulse": {"task_variables": [variable]}}},
    }
    raw["steps"]["baseline"]["sweep"] = {"axes": {"window": {"set": "entry-set"}}}
    with TestClient(create_app(projects_dir=tmp_path)) as client:
        folders = client.app.state.project_folders
        _save_project(folders, PROJECT, Project.model_validate(raw))
        before = _runs(folders, "baseline")
        assert set(before) == {"window=ordinary", "window=pulse"}
        revision = folders.read(PROJECT).revision

        def author(row, change=None):
            body = {"revision": revision, "run": {"variant": "default", "step": "baseline", "row": row}}
            if change is not None:
                body["change"] = change
            response = client.post(f"{BASE}/authoring/trials", json=body)
            assert response.status_code == 200, response.text
            return response.json()

        for row in before:
            read = author(row)
            for projected in (read["variables"], read["task"]["variables"]):
                assert any(item["id"] == "onset" for item in projected) == (row == "window=pulse")
        edit = author("window=pulse", {
            "kind": "expression", "path": "/task/variables/onset/value", "edit": {"kind": "text", "text": "0.04 s"},
        })
        assert edit["files"] and not edit.get("store_choices"), edit
        saved = client.patch(BASE, json={"revision": revision, "files": edit["files"]})
        assert saved.status_code == 200, saved.text
        after = _runs(folders, "baseline")
        assert after["window=ordinary"] == before["window=ordinary"]
        assert after["window=pulse"] != before["window=pulse"]


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
    assert field["condition"] is None
    assert next(item for item in reply.json()["fields"] if item["path"] == "/set_fields/run/condition")["text"] == ""
    assert field["entries"][0]["entry"] is None
    assert all(item["weight"] == 1 for item in field["entries"])
    assert _runs(folders, "train") == train_before

    _set_edit(folders, client, {"kind": "set_to_sweep", "step": "other", "axis": "run"})
    assert _runs(folders, "other") == sweep
    assert not folders.read(PROJECT).project.steps["other"].set_fields


def test_trial_expression_edits_and_removes_a_saved_set_condition(tmp_path):
    raw = document_project(authored_documents())
    raw["documents"]["task"]["payload"]["variables"].append({
        "id": "is_catch", "name": "is_catch", "type": {"base": "boolean"},
        "value": {"kind": "literal", "value": False},
    })
    raw["documents"]["loads"] = {"schema": "feedbax.set_document.v1", "payload": {"entries": {"heavy": {}}}}
    raw["steps"]["train"]["set_fields"] = {
        "load": {"axis": {"set": "loads"}, "sampling": {"kind": "weighted"}},
        "listed": {
            "axis": {"set": "loads"}, "sampling": {"kind": "listed"},
            "condition": {"kind": "literal", "value": True},
        },
    }
    folders = ProjectFolderService(tmp_path)
    _save_project(folders, PROJECT, Project.model_validate(raw))
    run = {"variant": "default", "step": "train", "row": "default"}
    with TestClient(create_app(projects_dir=tmp_path)) as client:
        def author(axis, edit=None, *, save=False):
            path = f"/set_fields/{axis}/condition"
            revision = folders.read(PROJECT).revision
            request = {"run": run, "revision": revision}
            if edit is not None:
                request["change"] = {"kind": "expression", "path": path, "edit": edit}
            response = client.post(f"{BASE}/authoring/trials", json=request)
            assert response.status_code == 200, response.text
            payload = response.json()
            if save:
                (file,) = payload["files"]
                assert file["address"] == {"kind": "step", "id": "train"}
                assert [patch["path"] for patch in file["patches"]] == [path]
                written = client.patch(BASE, json={"revision": revision, "files": payload["files"]})
                assert written.status_code == 200, written.text
            return payload, next(item for item in payload["fields"] if item["path"] == path)

        _, blank = author("load")
        assert blank["text"] == "" and blank["site"]["site"]["expected"]["base"] == "boolean"
        assert "is_catch" in {name["id"] for name in blank["site"]["site"]["names"]}
        _, listed = author("listed")
        assert "Weighted or Co-occurring" in listed["problems"][0]["reason"]
        assert listed["problems"][0]["span"] == {"start": 0, "end": 4}
        author("listed", {"kind": "text", "text": ""}, save=True)
        assert _runs(folders, "train")

        for rejected in ({"kind": "to_list"}, {"kind": "name_value", "value_id": "one", "label": "One"}):
            payload, _ = author("load", rejected)
            assert not payload["files"]
            assert any(problem["path"] == ["set_fields", "load", "condition"] for problem in payload["problems"])
        _, unknown = author("load", {"kind": "text", "text": "not missing"}, save=True)
        assert unknown["problems"][0]["span"] == {"start": 4, "end": 11}
        _, wrong_type = author("load", {"kind": "text", "text": "7"}, save=True)
        assert "Expected boolean" in wrong_type["problems"][0]["reason"]
        assert wrong_type["problems"][0]["span"] == {"start": 0, "end": 1}
        _, valid = author("load", {"kind": "text", "text": "not is_catch"}, save=True)
        assert not valid["problems"] and valid["text"] == "not is_catch"
        saved = folders.read(PROJECT).project.steps["train"].set_fields["load"]
        assert saved.condition.args[0].reference == "is_catch"
        assert _runs(folders, "train")

        _, cleared = author("load", {"kind": "text", "text": " "}, save=True)
        assert cleared["text"] == ""
        assert folders.read(PROJECT).project.steps["train"].set_fields["load"].condition is None


@pytest.mark.parametrize(("sampling", "edits", "weights", "variants"), [
    (
        {"kind": "weighted", "weights": {"nominal": 3}, "none": 4},
        [{"entry": "heavy", "weight": 2}, {"entry": None, "weight": 0}],
        {None: 0, "heavy": 2, "nominal": 3},
        True,
    ),
    (
        {"kind": "cooccurring", "weights": {"heavy": {"off": 3}, "nominal": {"on": 4}}},
        [{"entry": "heavy", "part": "on", "weight": 2}],
        {"heavy": {"on": 2, "off": 3}, "nominal": {"on": 4, "off": 1}},
        False,
    ),
])
def test_set_field_weight_changes_only_the_named_part_of_the_selected_train(tmp_path, sampling, edits, weights, variants):
    raw = document_project(authored_documents())
    raw["documents"]["loads"] = {
        "schema": "feedbax.set_document.v1", "payload": {"entries": {"heavy": {}, "nominal": {}}},
    }
    raw["steps"]["train"]["set_fields"] = {"load": {"axis": {"set": "loads"}, "sampling": sampling}}
    raw["steps"]["other"] = deepcopy(raw["steps"]["train"])
    if variants:
        raw["sweep"] = {"rows": {"base": {}, "compare": {"from": "base"}}}
    folders = ProjectFolderService(tmp_path)
    _save_project(folders, PROJECT, Project.model_validate(raw))
    before = folders.read(PROJECT)
    train, other = (_runs(folders, step) for step in ("train", "other"))
    with TestClient(create_app(projects_dir=tmp_path)) as client:
        for edit in edits:
            result = _set_edit(folders, client, {
                "kind": "set_field_weight", "step": "train", "axis": "load", **edit,
            }, store="whole_step" if variants else None)
            assert [file["address"] for file in result["files"]] == [{"kind": "step", "id": "train"}]
        saved = folders.read(PROJECT)
        reply = client.post(f"{BASE}/authoring/trials", json={
            "revision": saved.revision,
            "run": {"variant": "base" if variants else "default", "step": "train", "row": "default"},
        })
        assert reply.status_code == 200, reply.text
        (field,) = reply.json()["set_fields"]
        assert {item["entry"]: item["weight"] for item in field["entries"]} == weights

    assert saved.project.documents["loads"] == before.project.documents["loads"]
    assert saved.project.steps["other"] == before.project.steps["other"]
    assert _runs(folders, "train") != train
    assert _runs(folders, "other") == other


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
