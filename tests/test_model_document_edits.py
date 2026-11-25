"""Model edits, and an Evaluate step's edits of its Model and Task, are stored by the document rule."""

from copy import deepcopy
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.project import BasedDocument, Project
from feedbax.contracts.project_io import ProjectConfiguration, dump_text_project, load_text_project
from feedbax.planner.project_plan import compile_project
from feedbax.contracts.project_resolution import compose_document
from feedbax.web.app import create_app
from feedbax.web.services.project_plan import row_problems
from tests.support.managed_train_fixture import managed_evaluation_step
from tests.test_evaluate_documents import FEE, MASS, TASK, _based, _data, _model, _with_mass, _with_push

_URL = "/api/projects/reach"
_PLANT = {"composites": [], "element": {"kind": "component", "id": "plant"}}
_MASS_TEXT = ModelAddress.model_validate(MASS).to_text()


class _Project:
    def __init__(self, tmp_path: Path, data: dict):
        self.root = tmp_path / "reach"
        for name, encoded in dump_text_project(
            Project.model_validate(data), ProjectConfiguration(project="reach", store="results"),
        ).items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(encoded)
        self.client = TestClient(create_app(projects_dir=tmp_path))

    def snapshot(self):
        response = self.client.get(_URL)
        assert response.status_code == 200, response.text
        return response.json()

    def project(self) -> Project:
        return load_text_project(self.root).project

    def plan(self):
        return compile_project(self.project())

    def ids(self):
        return {run.key.step: run.request_id for run in self.plan().runs}

    def run(self, step):
        row = next(row for row in self.plan().rows if row.key.step == step)
        return {"variant": row.key.variant, "step": step, "row": row.key.row}

    def reads(self, step, slot="model"):
        row = next(row for row in self.plan().rows if row.key.step == step)
        return row.document_ids[f"payload.trial_documents.{slot}"]

    def composed(self, name):
        return compose_document(self.project().documents.get, name).payload

    def model_edit(self, operation, *, step=None, chosen=(), files=(), document="model", choice=None):
        snapshot = self.snapshot()
        return self.client.post(f"{_URL}/model/edit", json={
            "revision": snapshot["revision"], "presentation_revision": snapshot["presentation_revision"],
            "document": document, "command": {"operation": operation}, "chosen_steps": list(chosen),
            "files": list(files), "store_choice": choice, **({"step": step} if step else {}),
        })

    def edit(self, operation, *, store=None, **scope):
        response = self.model_edit(operation, **scope)
        assert response.status_code == 200, response.text
        result = response.json()["result"]
        if store is not None:
            assert result["status"] == "asked", result
            selected = next(option for option in result["store_choices"] if option["kind"] == store)
            response = self.model_edit(operation, **scope, choice=selected["id"])
            assert response.status_code == 200, response.text
            result = response.json()["result"]
        assert result["status"] == "applied", result
        return result

    def fields(self, step, edit, document="model", **body):
        response = self.client.post(f"{_URL}/model/fields", json={
            "revision": self.snapshot()["revision"], "document": document, "owner": _PLANT, "run": self.run(step),
            "address": MASS, "edit": edit, **body,
        })
        assert response.status_code == 200, response.text
        return response.json()


def _mass(value):
    return {"op": "set_field", "address": MASS, "value": value}


def _two_readers(data):
    data["steps"]["other"] = deepcopy(data["steps"]["fee"])
    data["steps"]["other"]["base"]["inline"]["seed"] = 1
    return data


def test_editing_the_base_for_its_remaining_reader_preserves_an_unselected_copy(tmp_path):
    data = _two_readers(_data())
    del data["steps"]["check"]
    _based(data, "model-2", _with_mass(_model(data), 2.5))
    data["steps"]["fee"]["base"]["inline"]["trial_documents"]["model"] = {"document": "model-2"}
    project = _Project(tmp_path, data)
    with project.client:
        prior = project.ids()
        based = project.project().documents["model-2"]
        result = project.edit(_mass(3.0), step="other")
        assert not result["created"]
        assert project.project().documents["model-2"] == based
        assert project.ids()["fee"] == prior["fee"]
        assert project.ids()["other"] != prior["other"]
        assert project.composed("model") == _with_mass(_model(data), 3.0)


def test_an_evaluation_field_edit_makes_a_document_based_on_its_subjects(tmp_path):
    project = _Project(tmp_path, _data())
    with project.client:
        prior = project.ids()
        reply = project.fields("check", {"kind": "text", "text": "2"}, evaluate_scope={"kind": "step"})
        assert reply["command"]["operation"]["address"] == MASS and not reply.get("files")
        result = project.edit(_mass(2.0), step="check", store="selected_readers")
        assert result["created"] == ["model-2"] and result.get("undo_token") is None
        documents = project.project().documents
        assert isinstance(documents["model-2"], BasedDocument) and documents["model-2"].base.document == "model"
        assert project.reads("check") == "model-2"
        edited = project.ids()
        assert edited["fee"] == prior["fee"] and edited["check"] != prior["check"]
        assert project.composed("model-2")["interior"]["components"]["plant"]["fields"]["mass"] == 2

        data = project.project().model_dump(mode="json", by_alias=True, exclude_unset=True)
        data["steps"]["check-2"] = managed_evaluation_step("fee")
        data["steps"]["check-2"]["base"]["inline"]["trial_documents"] = {"model": {"document": "model-2"}}
        data["steps"]["next"] = deepcopy(data["steps"]["fee"])
        data["steps"]["next"]["base"]["inline"]["trial_documents"]["model"] = {"document": "model-2"}
        planned = {run.key.step: run.request_id for run in compile_project(Project.model_validate(data)).runs}
        assert planned["check-2"] == edited["check"] and "next" in planned


def test_an_evaluation_edit_with_its_subject_chosen_is_written_to_the_subjects_document(tmp_path):
    project = _Project(tmp_path, _data())
    with project.client:
        prior = project.ids()
        result = project.edit(_mass(2.0), step="check", chosen=("fee",))
        assert result["created"] == []
        assert set(project.project().documents) == set(_data()["documents"])
        assert project.composed("model") == _with_mass(_model(_data()), 2.0)
        assert project.reads("check") == "model" and project.ids()["fee"] != prior["fee"]


def test_deleting_a_component_for_one_reader_makes_a_document_without_it(tmp_path):
    project = _Project(tmp_path, _two_readers(_data()))
    with project.client:
        other = project.ids()["other"]
        readout = {"composites": [], "element": {"kind": "component", "id": "readout"}}
        operation = {"op": "delete", "selection": [readout]}
        before = project.snapshot()
        ask = project.model_edit(operation, step="fee")
        assert ask.status_code == 200, ask.text
        assert ask.json()["result"]["status"] == "asked"
        assert project.snapshot()["revision"] == before["revision"]
        choices = {option["kind"]: option for option in ask.json()["result"]["store_choices"]}
        assert {"fee", "other"} <= set(choices["all_readers"]["steps"])
        assert choices["all_readers"]["label"] == f"All {len(choices['all_readers']['steps'])} Steps"
        assert choices["selected_readers"]["steps"] == ["fee"]
        choice = choices["selected_readers"]["id"]
        stale = project.model_edit(operation, step="fee", chosen=["other"], choice=choice)
        assert stale.status_code == 409, stale.text
        assert project.snapshot()["revision"] == before["revision"]
        reply = project.model_edit(operation, step="fee", choice=choice)
        assert reply.status_code == 200, reply.text
        result = reply.json()["result"]
        assert result["created"] == ["model-2"]
        assert "readout" not in project.composed("model-2")["interior"]["components"]
        assert "readout" in project.composed("model")["interior"]["components"]
        assert project.ids()["other"] == other


def test_a_kept_sweep_value_is_stored_in_a_document_based_on_the_shared_one(tmp_path):
    data = _two_readers(_data())
    data["steps"]["fee"]["sweep"] = {"axes": {"mass": {"path": _MASS_TEXT, "values": [1.5, 3.0]}}}
    project = _Project(tmp_path, data)
    with project.client:
        other = project.ids()["other"]
        reply = project.fields("fee", {"kind": "text", "text": "4"})
        assert reply["command"]["operation"]["address"] == MASS and reply["files"]
        project.edit(reply["command"]["operation"], step="fee", files=reply["files"], store="selected_readers")
        fee = project.project().steps["fee"]
        assert fee.sweep is None or all(axis is None for axis in fee.sweep.axes.values())
        assert project.reads("fee") == "model-2"
        assert project.composed("model-2")["interior"]["components"]["plant"]["fields"]["mass"] == 4
        assert project.ids()["other"] == other


@pytest.mark.parametrize("conditioned", [False, True])
def test_model_range_to_list_persists_an_identity_preserving_step_edit(tmp_path, conditioned):
    first, last = 1.000123456789, 2.000987654321
    data = _data()
    del data["steps"]["check"]
    data["steps"]["fee"]["sweep"] = {"axes": {"mass": {
        "path": _MASS_TEXT, "range": {"start": first, "stop": last, "count": 3},
    }}}
    axes = data["steps"]["fee"]["sweep"]["axes"]
    if conditioned:
        axes["scenario"] = {"values": {"simple": {}}}
        axes["mass"]["where"] = {"rule": "scenario=simple"}
    project = _Project(tmp_path, data)

    with project.client:
        revision = project.snapshot()["revision"]
        before = project.plan()
        rows = [(run.key, run.coordinates, run.request_id) for run in before.runs]
        assert len(rows) == 3
        response = project.fields("fee", {"kind": "to_list"}, rows=[], chosen_steps=["fee"])
        assert not response["problems"] and response["store_choices"] is None
        (file,) = response["files"]
        assert file["address"] == {"kind": "step", "id": "fee"}
        saved = project.client.patch(_URL, json={"revision": revision, "files": [file]})
        assert saved.status_code == 200 and saved.json()["revision"] != revision, saved.text

        axis = project.project().steps["fee"].sweep.axes["mass"]
        assert (axis.where.rule if axis.where is not None else None) == ("scenario=simple" if conditioned else None)
        assert list(axis.values) == [coordinates["mass"].value_id for _, coordinates, _ in rows]
        assert [entry.value for entry in axis.values.values()] == [first, (first + last) / 2, last]
        assert [(run.key, run.coordinates, run.request_id) for run in project.plan().runs] == rows
        (field,) = [field for field in project.fields("fee", None)["fields"] if field["address"] == MASS]
        assert field["sweep"]["kind"] == "list"


@pytest.mark.parametrize("shared_variant", [False, True])
def test_model_sweep_edit_reaches_its_authors_and_excludes_result_readers(tmp_path, shared_variant):
    from web.e2e.studio.fixtures.canonical_evaluate import task_pivot_files

    data = {"documents": {}, "steps": {}}
    for file in task_pivot_files():
        family = f"{file['kind']}s"
        data[family][file["id"]] = file[file["kind"]]
    if shared_variant:
        data["sweep"] = {"rows": {"base": {}, "compare": {"from": "base"}}}
    project = _Project(tmp_path, data)

    with project.client:
        before = project.plan()
        variant = before.variants[0]
        selected = [
            row.key.row for row in before.rows if row.key.step == "train" and row.key.variant == variant
            and row.coordinates["learning-rate"].value_id == "Fast"
        ]
        assert len(selected) == 2
        snapshot = project.snapshot()
        payload = {
            "revision": snapshot["revision"], "presentation_revision": snapshot["presentation_revision"],
            "document": "reach-model", "step": "train", "rows": selected, "axes": {"learning-rate": ["Fast"]},
            "command": {"operation": _mass(2)},
        }
        response = project.client.post(f"{_URL}/model/edit", json=payload)
        assert response.status_code == 200, response.text
        result = response.json()["result"]
        if shared_variant:
            assert result["status"] == "asked", result
            (choice,) = result["store_choices"]
            assert choice["steps"] == ["train"]
            assert {(run["variant"], run["row"]) for run in choice["runs"]} == {
                (variant, row) for variant in before.variants for row in selected
            }
            assert project.snapshot()["revision"] == snapshot["revision"]
        else:
            assert result["status"] == "applied", result
            after = project.plan()
            request_ids = {run.key: run.request_id for run in before.runs}
            changed_train = {
                run.key.row for run in after.runs if run.key.step == "train"
                and run.request_id != request_ids[run.key]
            }
            assert changed_train == set(selected)
            assert [row.key for row in after.rows if row.key.step == "evaluate"] == [
                row.key for row in before.rows if row.key.step == "evaluate"
            ]


def test_connection_review_reads_the_topology_on_the_selected_sweep_value(tmp_path):
    data = _data()
    data["steps"]["fee"]["sweep"] = {"axes": {"wiring": {"values": {
        "base": {},
        "expanded": {"patches": {"documents.model.payload.interior.components.added": {
            "op": "add", "value": _model(data)["interior"]["components"]["readout"],
        }}},
    }}}}
    project = _Project(tmp_path, data)
    endpoint = {"owner": "component", "component": "readout", "port": "output"}
    with project.client:
        context = {"revision": project.snapshot()["revision"], "document": "model", "graph": {
            "composites": [], "element": {"kind": "graph"},
        }, "source": endpoint}
        base = project.client.post(f"{_URL}/model/connections", json=context)
        assert base.status_code == 200, base.text
        assert all(target["target"].get("component") != "added" for target in base.json()["targets"])
        selected = project.client.post(f"{_URL}/model/connections", json=context | {
            "run": {"variant": "default", "step": "fee", "row": "wiring=expanded"},
            "rows": ["wiring=expanded"], "axes": {"wiring": ["expanded"]},
        })
        assert selected.status_code == 200, selected.text
        assert any(target["target"].get("component") == "added" for target in selected.json()["targets"])


def test_an_evaluation_task_edit_makes_a_document_based_on_its_subjects(tmp_path):
    project = _Project(tmp_path, _data())
    with project.client:
        task = project.composed(TASK)
        bound = task["variables"][0]["value"]["args"][0]["args"][0]
        path = "/variables/0/value/args/0/args/0/value"
        response = project.client.post(f"{_URL}/authoring/trials", json={
            "revision": project.snapshot()["revision"], "run": project.run("check"),
            "change": {"kind": "task", "edits": [{"op": "replace", "path": path, "value": bound["value"] - 0.1}]},
            "scope": {"kind": "step"},
        })
        assert response.status_code == 200, response.text
        body = response.json()
        assert not body.get("problems"), body
        created, = body["created"]
        assert created["id"] == f"{TASK}-2" and created["document"]["base"] == {"document": TASK}
        made = project.client.post(f"{_URL}/files", json={"revision": project.snapshot()["revision"], "file": created})
        assert made.status_code == 201, made.text
        saved = project.client.patch(_URL, json={"revision": project.snapshot()["revision"], "files": body["files"]})
        assert saved.status_code == 200, saved.text
        assert project.reads("check", "task") == f"{TASK}-2"
        assert project.ids()["fee"] == FEE


def test_a_variant_s_evaluation_task_edit_is_a_document_only_its_row_points_at(tmp_path):
    data = _data()
    data["sweep"] = {"rows": {"control": {}, "treated": {}}}
    project = _Project(tmp_path, data)
    with project.client:
        before = {run.key: run.request_id for run in project.plan().runs}
        steps = project.project().steps
        path = "/variables/0/value/args/0/args/0/value"
        bound = project.composed(TASK)["variables"][0]["value"]["args"][0]["args"][0]
        response = project.client.post(f"{_URL}/authoring/trials", json={
            "revision": project.snapshot()["revision"], "run": project.run("check") | {"variant": "treated"},
            "change": {"kind": "task", "edits": [{"op": "replace", "path": path, "value": bound["value"] - 0.1}]},
            "scope": {"kind": "step"},
        })
        assert response.status_code == 200, response.text
        body = response.json()
        assert not body.get("problems"), body
        created, = body["created"]
        assert created["id"] == f"{TASK}-2" and created["document"]["base"] == {"document": TASK}
        assert project.client.post(f"{_URL}/files", json={
            "revision": project.snapshot()["revision"], "file": created,
        }).status_code == 201
        saved = project.client.patch(_URL, json={"revision": project.snapshot()["revision"], "files": body["files"]})
        assert saved.status_code == 200, saved.text
        assert project.project().steps == steps
        assert {path: write.model_dump(mode="json") for path, write in project.project().sweep.rows["treated"].patches.items()} == {
            "steps.check.base.inline.trial_documents": {"op": "add", "value": {"task": {"document": f"{TASK}-2"}}},
        }
        after = {run.key: run.request_id for run in project.plan().runs}
        assert {key for key in before if before[key] != after[key]} == {
            key for key in before if (key.variant, key.step) == ("treated", "check")
        }


def test_a_variant_writes_a_model_field_of_a_based_document():
    def ids(writes, model_2):
        data = _data()
        data["steps"]["other"] = deepcopy(data["steps"]["fee"])
        data["steps"]["other"]["base"]["inline"]["trial_documents"]["model"] = {"document": "model-2"}
        _based(data, "model-2", model_2(_model(data)))
        data["sweep"] = {"rows": {"default": {}, "changed": {"patches": writes}}}
        plan = compile_project(Project.model_validate(data))
        assert not [problem for problem in plan.diagnostics if problem.variant == "changed"]
        return {
            variant: {run.key.step: run.request_id for run in plan.runs if run.key.variant == variant and run.kind == "train"}
            for variant in ("default", "changed")
        }

    def changed(writes, model_2):
        result = ids(writes, model_2)
        return {step for step in result["default"] if result["default"][step] != result["changed"][step]}

    assert changed({f"documents.model.{_MASS_TEXT}": {"value": 3.0}}, _with_push) == {"fee", "other"}
    assert changed({f"documents.model.{_MASS_TEXT}": {"value": 3.0}}, lambda payload: _with_mass(payload, 2.0)) == {"fee"}
    assert changed({f"documents.model-2.{_MASS_TEXT}": {"value": 3.0}}, _with_push) == {"other"}


def test_reset_of_the_only_changed_value_leaves_the_based_document_without_differences(tmp_path):
    project = _Project(tmp_path, _two_readers(_data()))
    with project.client:
        before = project.ids()
        project.edit(_mass(2.0), step="fee", chosen=("fee", "check"), store="selected_readers")
        assert project.reads("fee") == "model-2" and project.ids()["fee"] != before["fee"]

        def mass(document):
            fields = project.fields("fee", None, document=document)["fields"]
            return next(field for field in fields if field["address"] == MASS)

        assert mass("model-2")["changed"] and not mass("model")["changed"]
        reply = project.fields("fee", {"kind": "reset"}, document="model-2")
        project.edit(reply["command"]["operation"], step="fee", chosen=("fee", "check"), document="model-2")
        assert project.project().documents["model-2"].differences == {}
        assert not mass("model-2")["changed"]
        assert project.ids() == before


def test_an_evaluation_marks_the_values_its_document_changes(tmp_path):
    project = _Project(tmp_path, _data())
    with project.client:
        before = project.ids()
        project.edit(_mass(2.0), step="check", store="selected_readers")
        fields = project.fields("check", None, document="model-2", evaluate_scope={"kind": "step"})["fields"]
        assert [field["address"] == MASS for field in fields] == [field["changed"] for field in fields]
        reply = project.fields("check", {"kind": "reset"}, document="model-2", evaluate_scope={"kind": "step"})
        project.edit(reply["command"]["operation"], step="check", document="model-2")
        assert project.project().documents["model-2"].differences == {}
        assert project.ids() == before


def test_an_evaluation_picks_a_model_document_and_one_unrelated_to_its_subjects_cannot_run(tmp_path):
    data = _data()
    _based(data, "model-2", _with_mass(_model(data), 2.0))
    data["documents"]["unrelated"] = deepcopy(data["documents"]["model"])
    project = _Project(tmp_path, data)

    def pick(document):
        response = project.client.post(f"{_URL}/authoring/trials", json={
            "revision": project.snapshot()["revision"], "run": project.run("check"),
            "change": {"kind": "select_document", "role": "model", "document": document}, "scope": {"kind": "step"},
        })
        assert response.status_code == 200, response.text
        body = response.json()
        saved = project.client.patch(_URL, json={"revision": project.snapshot()["revision"], "files": body["files"]})
        assert saved.status_code == 200, saved.text

    with project.client:
        prior = project.ids()
        pick("model-2")
        edited = project.ids()
        assert project.reads("check") == "model-2"
        assert edited["fee"] == prior["fee"] and edited["check"] != prior["check"]
        pick("unrelated")
        assert "check" not in project.ids()
        problems = [problem for problem in project.plan().diagnostics if problem.step == "check"]
        assert [problem.path for problem in problems] == ["payload.trial_documents.model"]


def test_invalid_model_field_survives_reopen_and_can_be_repaired(tmp_path):
    project = _Project(tmp_path, _data())
    text = " 1 + "
    with project.client:
        recipe = project.fields("fee", {"kind": "text", "text": text})
        assert recipe["problems"][0]["model_address"] == MASS
        before = project.snapshot()["revision"]
        saved = project.edit(recipe["command"]["operation"], files=recipe["files"])
        assert saved["snapshot"]["revision"] != before
        assert project.composed("model")["interior"]["components"]["plant"]["fields"]["mass"] == {
            "kind": "unresolved", "text": text,
        }
        (fee,) = [run for run in project.plan().runs if run.key.step == "fee"]
        assert fee.admission_problem is not None
        assert any(problem.model_address == ModelAddress.model_validate(MASS) for problem in fee.problems)
        fields = project.fields("fee", None)["fields"]
        assert any(field["address"] == MASS and field["text"] == text and field["problems"] for field in fields)
        recipe = project.fields("fee", {"kind": "text", "text": "2"})
        project.edit(recipe["command"]["operation"], files=recipe["files"])
        (fee,) = [run for run in project.plan().runs if run.key.step == "fee"]
        assert fee.admission_problem is None


def test_invalid_swept_field_edits_the_active_member_and_marks_its_axis(tmp_path):
    owner = {"composites": [], "element": {"kind": "component", "id": "cell"}}
    address = ModelAddress.model_validate({"composites": [], "element": {
        "kind": "field", "parent": owner["element"], "path": ["fields", "hidden_size"],
    }})
    path = address.to_text()
    readout_input = ModelAddress.model_validate({"composites": [], "element": {
        "kind": "field", "parent": {"kind": "component", "id": "readout"}, "path": ["fields", "input_size"],
    }}).to_text()
    entries = {
        f"value-{index}": {"value": value, "patches": {path: {"value": value}, readout_input: {"value": value}}}
        for index, value in enumerate((64, 100, 128), start=1)
    }
    data = _data()
    del data["steps"]["check"]
    data["steps"]["fee"]["sweep"] = {"axes": {"hidden-size": {"values": entries}}}
    project = _Project(tmp_path, data)

    with project.client:
        revision = project.snapshot()["revision"]
        before = project.plan()
        assert len(before.runs) == 3 and all(not row.problems for row in before.runs)
        active = next(
            row.key for row in before.rows
            if row.key.step == "fee" and row.coordinates["hidden-size"].value_id == "value-1"
        )
        payload = {
            "revision": revision, "document": "model", "owner": owner,
            "run": {"variant": active.variant, "step": active.step, "row": active.row}, "rows": [],
            "address": address.model_dump(mode="json"), "edit": {"kind": "text", "text": "U(0, 1)"},
        }
        response = project.client.post(f"{_URL}/model/fields", json=payload)
        assert response.status_code == 200, response.text
        recipe = response.json()
        assert recipe.get("command") is None and not recipe["store_choices"], recipe
        (file,) = recipe["files"]
        assert file["address"] == {"kind": "step", "id": "fee"}
        assert recipe["problems"][0]["model_address"] == payload["address"]
        saved = project.client.patch(_URL, json={"revision": revision, "files": [file]})
        assert saved.status_code == 200, saved.text

        reopened = project.project()
        values = reopened.steps["fee"].sweep.axes["hidden-size"].values
        unresolved = {"kind": "unresolved", "text": "U(0, 1)"}
        assert values["value-1"].value == values["value-1"].patches[path].value == unresolved
        assert values["value-1"].patches[readout_input].value == 64
        assert [values[member].value for member in ("value-2", "value-3")] == [100, 128]
        assert reopened.documents["model"].payload == data["documents"]["model"]["payload"]
        after = project.plan()
        (problem,) = row_problems(after)
        assert problem.path == "sweep.axes.hidden-size.values.value-1"
        assert problem.address == "steps/fee.json" and problem.row == active.row
        assert problem.authored_path.startswith("/sweep/axes/hidden-size/values/value-1/")
        assert problem.place.member == address
        assert {run.key: run.request_id for run in before.runs if run.key.step == "fee" and run.key != active} == {
            run.key: run.request_id for run in after.runs if run.key.step == "fee" and run.key != active
        }


def test_removing_a_swept_model_field_keeps_other_fields_readable(tmp_path):
    data = _data()
    data["steps"]["fee"]["sweep"] = {"axes": {"mass": {"path": _MASS_TEXT, "values": [1.5, 3.0]}}}
    project = _Project(tmp_path, data)
    with project.client:
        before = project.snapshot()["revision"]
        result = project.edit({"op": "delete", "selection": [_PLANT]})
        assert result["snapshot"]["revision"] != before
        assert result["sweep_problems"][0]["authored_path"] == "/sweep/axes/mass/path"
        assert "plant" not in project.composed("model")["interior"]["components"]
        assert project.project().steps["fee"].sweep.axes["mass"].path == _MASS_TEXT
        assert any(problem.authored_path and problem.authored_path.startswith("/sweep/axes/mass")
                   for problem in project.plan().diagnostics if problem.step == "fee")
        read = project.fields("fee", None, owner={"composites": [], "element": {"kind": "graph"}})
        (step,) = read["fields"]
        assert step["label"] == "Step" and step["text"] == "0.01 s"


def test_grouping_a_swept_model_field_rewrites_its_target_by_identity(tmp_path):
    data = _data()
    data["steps"]["fee"]["sweep"] = {"axes": {"mass": {"path": _MASS_TEXT, "values": [1.5, 3.0]}}}
    project = _Project(tmp_path, data)
    with project.client:
        result = project.edit({"op": "group", "selection": [_PLANT], "name": "Plant"})
        target = ModelAddress.from_text(project.project().steps["fee"].sweep.axes["mass"].path)
        assert target.composites and target.element == ModelAddress.model_validate(MASS).element
        (group,) = target.composites
        assert project.composed("model")["interior"]["components"][group]["interior"]["components"]["plant"]
        assert not result["sweep_problems"] and "fee" in project.ids()


@pytest.mark.parametrize(("name", "hidden"), [("CDEStandard", 2.0), ("CDEDecay", 1.8)])
def test_place_registered_template_as_an_editable_composite(tmp_path, name, hidden):
    """A placed template keeps its boundary wiring and the registered controller's dynamics."""
    import jax
    import jax.numpy as jnp

    from feedbax.application.composition import load_application
    from feedbax.compiler.graph import GraphDocument, compile_graph
    from feedbax.compiler.model_document import lower_model_document
    from feedbax.contracts.model_document import CompositeComponent, ModelDocument

    data = _data()
    data["steps"] = {}
    data["documents"]["model"]["payload"] = {"step_seconds": 0.01, "ports": {}, "interior": {}}
    project = _Project(tmp_path, data)
    position = {"x": 20, "y": 30}
    address = ModelAddress.model_validate({"composites": [], "element": {"kind": "component", "id": "controller"}})

    with project.client:
        project.edit({
            "op": "add_component", "graph": {"composites": [], "element": {"kind": "graph"}}, "id": "controller",
            "component": {"kind": "definition", "definition": f"feedbax.{name}"}, "position": position,
        })
        assert project.snapshot()["presentation"]["documents"]["model"]["positions"][address.to_text()] == position

    model = ModelDocument.model_validate(project.composed("model"))
    component = model.interior.components["controller"]
    assert isinstance(component, CompositeComponent)
    assert component.name == name
    assert component.interior.components["vf"].kind == "leaf"
    assert [port.id for port in component.ports.inputs] == ["obs", "obs_prev", "h_prev"]
    assert [port.id for port in component.ports.outputs] == ["h_new", "action"]

    registry = load_application().bundle.components
    interior_model = ModelDocument(step_seconds=model.step_seconds, ports=component.ports, interior=component.interior)
    lowered = lower_model_document(interior_model, registry).graph
    prototypes = {bound: jax.ShapeDtypeStruct((1,), jnp.float32) for bound in lowered.input_bindings.values()}
    graph = compile_graph(GraphDocument(graph=lowered), registry, prototypes).graph
    key = jax.random.key(8)
    inputs = {"obs": jnp.ones((1, 1)), "obs_prev": jnp.ones((1, 1)), "h_prev": jnp.full((1, 1), 2.0)}
    outputs, _ = graph(inputs, graph.init_state(key=key), key=key, n_steps=1)
    assert float(outputs["h_new"][0, 0]) == pytest.approx(hidden)


def test_template_zero_carry_preserves_a_recurrent_controller_rollout():
    """Replacing a template carry with a Model Delay must preserve reset and later steps."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    from feedbax.application.composition import load_application
    from feedbax.compiler.graph import GraphDocument, compile_graph
    from feedbax.compiler.model_document import lower_model_document
    from feedbax.compiler.model_templates import instantiate_model_template
    from feedbax.contracts.graph import GraphSpec
    from feedbax.contracts.model_document import ModelDocument

    registry = load_application().bundle.components
    name = "feedbax.RecurrentController"
    original = registry.get(name).template_graph
    assert isinstance(original, GraphSpec)
    placed = instantiate_model_template(name, registry, 0.01)
    model = ModelDocument(step_seconds=0.01, ports=placed.ports, interior=placed.interior)
    restored = lower_model_document(model, registry).graph
    key = jax.random.key(19)
    inputs = {"input": jnp.arange(15, dtype=jnp.float32).reshape((5, 3)) / 10,
              "feedback": jnp.full((5, 3), 0.1)}

    def rollout(spec):
        prototypes = {bound: jax.ShapeDtypeStruct((3,), jnp.float32) for bound in spec.input_bindings.values()}
        graph = compile_graph(GraphDocument(graph=spec), registry, prototypes).graph
        return graph(inputs, graph.init_state(key=key), key=key, n_steps=5)[0]["output"]

    np.testing.assert_allclose(rollout(restored), rollout(original), rtol=1e-6, atol=1e-6)


def test_library_previews_the_same_reason_as_an_unplaceable_template_edit(tmp_path):
    """The Library must warn before offering a drop whose reset cannot be represented."""
    data = _data()
    data["steps"] = {}
    project = _Project(tmp_path, data)
    graph = {"composites": [], "element": {"kind": "graph"}}
    with project.client:
        before = project.snapshot()
        response = project.client.get("/api/components")
        assert response.status_code == 200, response.text
        library = {entry["name"]: entry["placement_reason"] for entry in response.json()["data"]["components"]}
        assert library["feedbax.CDEStandard"] is None
        assert library["feedbax.RecurrentController"] is None
        assert library["feedbax.PointMass"] is None
        reason = library["feedbax.SimpleFeedbackLoop"]
        assert reason
        refused = project.model_edit({
            "op": "add_component", "graph": graph, "position": {"x": 20, "y": 30},
            "component": {"kind": "definition", "definition": "feedbax.SimpleFeedbackLoop"},
        })
        assert refused.status_code == 200, refused.text
        result = refused.json()["result"]
        assert result["status"] == "refused"
        assert result["problem"] == {"address": graph, "detail": reason}
        assert project.snapshot()["revision"] == before["revision"]


def test_insert_registered_template_routes_the_wire_through_its_composite_ports(tmp_path):
    """Inserting a template must splice its real input and output into the targeted wire."""
    data = _data()
    data["steps"] = {}
    source = {"owner": "inputs", "port": "in"}
    target = {"owner": "outputs", "port": "out"}
    data["documents"]["model"]["payload"] = {
        "step_seconds": 0.01,
        "ports": {
            "inputs": [{"id": "in", "name": "Input", "type": {}}],
            "outputs": [{"id": "out", "name": "Output", "type": {}}],
        },
        "interior": {"wires": {"straight": {"source": source, "target": target}}},
    }
    project = _Project(tmp_path, data)
    with project.client:
        project.edit({
            "op": "insert_component", "address": {"composites": [], "element": {"kind": "wire", "id": "straight"}},
            "id": "controller", "component": {"kind": "definition", "definition": "feedbax.CDEDecay"},
            "input_port": "obs", "output_port": "h_new", "position": {"x": 20, "y": 30},
        })

    interior = project.composed("model")["interior"]
    component = interior["components"]["controller"]
    assert component["kind"] == "composite" and "decay" in component["interior"]["components"]
    assert interior["wires"]["straight"]["source"] == {"owner": "component", "component": "controller", "port": "h_new"}
    assert interior["wires"]["straight"]["target"] == target
    (incoming,) = [wire for key, wire in interior["wires"].items() if key != "straight"]
    assert incoming["source"] == source
    assert incoming["target"] == {"owner": "component", "component": "controller", "port": "obs"}


@pytest.mark.parametrize(
    ("domain", "body_type", "ground_type", "spring_type", "damper_type", "effort", "coordinate"),
    [
        ("translational", "Mass", "Ground", "LinearSpring", "LinearDamper", "force", "position"),
        ("rotational", "Inertia", "RotationalGround", "TorsionalSpring", "RotationalDamper", "torque", "angle"),
    ],
)
def test_equations_edits_store_and_run_like_a_direct_document(
    tmp_path, domain, body_type, ground_type, spring_type, damper_type, effort, coordinate,
):
    import jax
    import jax.numpy as jnp
    import numpy as np

    from feedbax.compiler import GraphDocument, compile_graph
    from feedbax.compiler.model_document import lower_model_document
    from feedbax.component_registry import ComponentRegistry
    from feedbax.contracts.model_document import ModelDocument

    def leaf(type_, **fields):
        return {"kind": "leaf", "name": type_, "type": f"feedbax.{type_}", "fields": fields}

    def end(component, port="flange"):
        return {"owner": "component", "component": component, "port": port}

    def boundary(owner, port):
        return {"owner": owner, "port": port}

    def wire(source, target):
        return {"source": source, "target": target}

    def connection(a, b):
        return {"a": a, "b": b}

    root = {"composites": [], "element": {"kind": "graph"}}
    inner = {"composites": ["equations"], "element": {"kind": "graph"}}
    position = {"x": 20, "y": 30}
    model = {
        "step_seconds": 0.01,
        "ports": {
            "inputs": [{"id": "drive", "name": "drive", "type": {"shape": []}}],
            "outputs": [{"id": "read", "name": "read", "type": {"shape": []}}],
        },
        "interior": {},
    }
    shell = {
        "kind": "composite", "name": "Equations",
        "ports": {
            "inputs": [{"id": "in", "name": "Input", "type": {"shape": []}}],
            "outputs": [{"id": "out", "name": "Output", "type": {"shape": []}}],
        },
        "interior": {"physical_domain": domain},
    }
    free = {"id": "free", "name": "Free end", "type": domain}
    flange = boundary("boundary", "free")
    leaves = {
        "wall": leaf(ground_type), "body": leaf(body_type),
        "spring": leaf(spring_type, stiffness=10.0), "damper": leaf(damper_type, damping=2.0),
        "act": leaf("ActuationInput", source_kind=effort), "sense": leaf("SensorOutput", quantity=coordinate),
    }
    root_wires = {
        "drive": wire(boundary("inputs", "drive"), end("equations", "in")),
        "read": wire(end("equations", "out"), boundary("outputs", "read")),
    }
    inner_wires = {
        "drive": wire(boundary("inputs", "in"), end("act", "u")),
        "read": wire(end("sense", "y"), boundary("outputs", "out")),
    }
    physical = {
        "damper-wall": connection(end("wall"), end("damper", "flange_a")),
        "damper-body": connection(end("damper", "flange_b"), end("body")),
        "act-flange": connection(end("act"), end("body")),
        "sense-flange": connection(end("sense"), end("body")),
    }
    data = _data()
    data["steps"] = {}
    data["documents"]["model"]["payload"] = model
    project = _Project(tmp_path, data)
    with project.client:
        library = project.client.get("/api/components")
        assert library.status_code == 200, library.text
        entries = {row["name"]: row for row in library.json()["data"]["components"]}
        assert set(entries["feedbax.AcausalSystem"]["equations_domains"]) == {"translational", "rotational"}
        project.edit({"op": "add_component", "graph": root, "position": position, "id": "equations", "component": shell})
        project.edit({"op": "add_port", "graph": inner, "direction": "physical", "port": free})
        for key, component in leaves.items():
            if key != "spring":
                project.edit({"op": "add_component", "graph": inner, "position": position, "id": key, "component": component})
        for graph, wires in ((root, root_wires), (inner, inner_wires)):
            for item in wires.values():
                project.edit({"op": "connect", "graph": graph, **item})
        for item in physical.values():
            project.edit({"op": "connect_physical", "graph": inner, "connection": item})
        straight = project.edit({
            "op": "connect_physical", "graph": inner, "connection": connection(flange, end("body")),
        })["changed"][0]
        insertion = {
            "op": "insert_physical_component", "address": straight, "component": leaves["spring"],
            "id": "spring", "a_port": "flange_a", "b_port": "flange_b", "position": position, "flipped": True,
        }
        before = project.snapshot()
        duplicate = project.model_edit({
            "op": "connect_physical", "graph": inner, "connection": connection(end("body"), flange),
        })
        assert duplicate.status_code == 200, duplicate.text
        assert duplicate.json()["result"]["status"] == "refused"
        bad = project.model_edit(insertion | {"b_port": "missing"})
        assert bad.status_code == 200, bad.text
        assert bad.json()["result"]["status"] == "refused"
        assert project.snapshot()["revision"] == before["revision"]
        assert project.snapshot()["presentation_revision"] == before["presentation_revision"]
        added = project.edit(insertion)
        spring_ground = connection(end("wall"), end("spring", "flange_a"))
        project.edit({"op": "connect_physical", "graph": inner, "connection": spring_ground})
        projection = project.client.post(f"{_URL}/model/projection", json={
            "revision": project.snapshot()["revision"], "document": "model",
        })
        assert projection.status_code == 200, projection.text
        physical_types = {
            ModelAddress.model_validate(entry["address"]).to_text(): entry["type"]
            for entry in projection.json()["physical_ports"]
        }
        for scope, port in (
            ([], {**end("equations", "free"), "direction": "physical"}),
            (["equations"], boundary("boundary", "free")),
            (["equations"], {**end("act"), "direction": "physical"}),
            (["equations"], {**end("spring", "flange_a"), "direction": "physical"}),
        ):
            address = ModelAddress.model_validate({"composites": scope, "element": {"kind": "port", "end": port}})
            assert physical_types[address.to_text()] == domain

    stored = project.composed("model")
    edited = stored["interior"]["components"]["equations"]["interior"]
    routed = connection(end("spring", "flange_b"), end("body"))
    added_edge = connection(flange, end("spring", "flange_a"))
    assert edited["connections"][straight["element"]["id"]] == routed
    assert added_edge in edited["connections"].values()
    direct = model | {"interior": {"wires": root_wires, "components": {"equations": shell | {
        "ports": shell["ports"] | {"physical": [free]},
        "interior": shell["interior"] | {
            "components": leaves, "wires": inner_wires,
            "connections": physical | {"spring-free": added_edge, "spring-body": routed, "spring-wall": spring_ground},
        },
    }}}}

    def rollout(document):
        registry = ComponentRegistry()
        lowered = lower_model_document(ModelDocument.model_validate(document), registry)
        graph = compile_graph(GraphDocument(graph=lowered.graph), registry, {
            lowered.graph.input_bindings["drive"]: jax.ShapeDtypeStruct((), jnp.float32),
        }).graph
        key = jax.random.key(1)
        return graph({"drive": jnp.ones(8)}, graph.init_state(key=key), key=key, n_steps=8)[0]["read"]

    expected = rollout(direct)
    assert expected[-1] > 0
    np.testing.assert_allclose(rollout(stored), expected, rtol=1e-6)

    with project.client:
        project.edit({"op": "delete", "selection": [added["changed"][0]]})
    remaining = project.composed("model")["interior"]["components"]["equations"]["interior"]
    assert "spring" not in remaining["components"]
    assert remaining["connections"] == {key: edge for key, edge in edited["connections"].items() if edge not in (
        routed, added_edge, spring_ground,
    )}
