"""Model edits, and an Evaluate step's edits of its Model and Task, are stored by the document rule."""

from copy import deepcopy
from pathlib import Path

from fastapi.testclient import TestClient

from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.project import BasedDocument, Project
from feedbax.contracts.project_io import ProjectConfiguration, dump_text_project, load_text_project
from feedbax.contracts.project_plan import compile_project
from feedbax.contracts.project_resolution import compose_document
from feedbax.web.app import create_app
from tests.support.managed_train_fixture import managed_evaluation_step
from tests.test_evaluate_documents import FEE, FIELD_CHANGED, MASS, TASK, _based, _data, _model, _with_mass, _with_push

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

    def edit(self, operation, *, step=None, chosen=(), files=(), document="model"):
        snapshot = self.snapshot()
        response = self.client.post(f"{_URL}/model/edit", json={
            "revision": snapshot["revision"], "presentation_revision": snapshot["presentation_revision"],
            "document": document, "command": {"operation": operation}, "chosen_steps": list(chosen),
            "files": list(files), **({"step": step} if step else {}),
        })
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


def test_an_evaluation_field_edit_makes_a_document_based_on_its_subjects(tmp_path):
    project = _Project(tmp_path, _data())
    with project.client:
        reply = project.fields("check", {"kind": "text", "text": "2"}, evaluate_scope={"kind": "step"})
        assert reply["command"]["operation"]["address"] == MASS and not reply.get("files")
        result = project.edit(_mass(2.0), step="check")
        assert result["created"] == ["model-2"] and result.get("undo_token") is None
        documents = project.project().documents
        assert isinstance(documents["model-2"], BasedDocument) and documents["model-2"].base.document == "model"
        assert project.reads("check") == "model-2"
        assert project.ids() == {"fee": FEE, "check": FIELD_CHANGED}

        data = project.project().model_dump(mode="json", by_alias=True, exclude_unset=True)
        data["steps"]["check-2"] = managed_evaluation_step("fee")
        data["steps"]["check-2"]["base"]["inline"]["trial_documents"] = {"model": {"document": "model-2"}}
        data["steps"]["next"] = deepcopy(data["steps"]["fee"])
        data["steps"]["next"]["base"]["inline"]["trial_documents"]["model"] = {"document": "model-2"}
        planned = {run.key.step: run.request_id for run in compile_project(Project.model_validate(data)).runs}
        assert planned["check-2"] == FIELD_CHANGED and "next" in planned


def test_an_evaluation_edit_with_its_subject_chosen_is_written_to_the_subjects_document(tmp_path):
    project = _Project(tmp_path, _data())
    with project.client:
        result = project.edit(_mass(2.0), step="check", chosen=("fee",))
        assert result["created"] == []
        assert set(project.project().documents) == set(_data()["documents"])
        assert project.composed("model") == _with_mass(_model(_data()), 2.0)
        assert project.reads("check") == "model" and project.ids()["fee"] != FEE


def test_deleting_a_component_for_one_reader_makes_a_document_without_it(tmp_path):
    project = _Project(tmp_path, _two_readers(_data()))
    with project.client:
        other = project.ids()["other"]
        readout = {"composites": [], "element": {"kind": "component", "id": "readout"}}
        result = project.edit({"op": "delete", "selection": [readout]}, step="fee")
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
        project.edit(reply["command"]["operation"], step="fee", files=reply["files"])
        fee = project.project().steps["fee"]
        assert fee.sweep is None or all(axis is None for axis in fee.sweep.axes.values())
        assert "model_attachments" not in fee.model_fields_set
        assert project.reads("fee") == "model-2"
        assert project.composed("model-2")["interior"]["components"]["plant"]["fields"]["mass"] == 4
        assert project.ids()["other"] == other


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
        project.edit(_mass(2.0), step="fee", chosen=("fee",))
        assert project.reads("fee") == "model-2" and project.ids()["fee"] != before["fee"]

        def mass(document):
            fields = project.fields("fee", None, document=document)["fields"]
            return next(field for field in fields if field["address"] == MASS)

        assert mass("model-2")["changed"] and not mass("model")["changed"]
        reply = project.fields("fee", {"kind": "reset"}, document="model-2")
        project.edit(reply["command"]["operation"], step="fee", chosen=("fee",), document="model-2")
        assert project.project().documents["model-2"].differences == {}
        assert not mass("model-2")["changed"]
        assert project.ids() == before


def test_an_evaluation_marks_the_values_its_document_changes(tmp_path):
    project = _Project(tmp_path, _data())
    with project.client:
        before = project.ids()
        project.edit(_mass(2.0), step="check")
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
        pick("model-2")
        assert project.reads("check") == "model-2" and project.ids() == {"fee": FEE, "check": FIELD_CHANGED}
        pick("unrelated")
        assert "check" not in project.ids()
        problems = [problem for problem in project.plan().diagnostics if problem.step == "check"]
        assert [problem.path for problem in problems] == ["payload.trial_documents.model"]
