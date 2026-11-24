"""Model edits, and an Evaluate step's edits of its Model and Task, are stored by the document rule."""

from copy import deepcopy
from pathlib import Path

from fastapi.testclient import TestClient

from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.project import BasedDocument, Project
from feedbax.contracts.project_io import ProjectConfiguration, dump_text_project, load_text_project
from feedbax.planner.project_plan import compile_project
from feedbax.contracts.project_resolution import compose_document
from feedbax.web.app import create_app
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
        assert choices["selected_readers"]["steps"] == ["check", "fee"]
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
