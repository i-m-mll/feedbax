"""Authored invalid values survive an edit and reopen with an addressable Problem."""


from fastapi.testclient import TestClient
import pytest

from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.project import Project
from feedbax.contracts.project_sweep import CHECKPOINT_RULE_TARGET, model_field_path
from feedbax.contracts.stored_model import STORED_MODEL_DOCUMENT_SCHEMA_ID
from feedbax.examples import load_shipped_projects
from feedbax.web.app import create_app
from feedbax.web.services.project_folder import ProjectFolderService
from tests.support.managed_train_fixture import managed_evaluation_step
from tests.test_canonical_evaluate_admission import _save_project
from tests.test_evaluate_documents import MASS, _based, _data, _model, _reads, _with_mass
from tests.test_saved_update_series import definition


_PROJECT = "stored-problems"
_URL = f"/api/projects/{_PROJECT}"
_GRAPH = {"composites": [], "element": {"kind": "graph"}}
_PLANT = {"composites": [], "element": {"kind": "component", "id": "plant"}}
_MASS_TARGET = model_field_path(ModelAddress.model_validate(MASS))


def _json(response):
    assert response.status_code == 200, response.text
    return response.json()


class _Studio:
    def __init__(self, client, folders):
        self.client = client
        self.folders = folders
        self.before = _json(client.get(f"{_URL}/open"))["snapshot"]

    def post(self, path, **body):
        return _json(self.client.post(f"{_URL}{path}", json={"revision": self.before["revision"], **body}))

    def write(self, files):
        assert files and all(file["patches"] for file in files), files
        return _json(self.client.patch(_URL, json={"revision": self.before["revision"], "files": files}))

    def model_edit(self, document, command, *, files=()):
        result = self.post(
            "/model/edit", presentation_revision=self.before["presentation_revision"],
            document=document, command=command, files=list(files),
        )["result"]
        assert result["status"] == "applied" and result["changed"], result
        return result

    def member(self, **command):
        return self.post("/members/command", presentation_revision=self.before["presentation_revision"], command=command)

    def reopen(self, written, kind, member):
        def files(snapshot):
            return {
                (file["address"]["kind"], file["address"]["id"]): file["revision"]
                for file in snapshot["file_revisions"]
            }

        reopened = _json(self.client.get(f"{_URL}/open"))["snapshot"]
        assert reopened["revision"] == written["revision"] != self.before["revision"]
        target = kind, member
        assert files(reopened)[target] == files(written)[target] != files(self.before).get(target)
        return reopened

    def plan_problems(self, revision):
        response = self.client.get(f"{_URL}/train-runs", params={"revision": revision, "window_map": True})
        return _json(response)["window_map"]["problems"]

    def run(self, step):
        plan = self.folders.read_plan(_PROJECT, revision=self.before["revision"]).plan
        row = next(row for row in plan.rows if row.key.step == step)
        return {"variant": row.key.variant, "step": step, "row": row.key.row}


def _graph_project():
    return {"documents": {"model": {"label": "Model", "schema": STORED_MODEL_DOCUMENT_SCHEMA_ID, "payload": {
        "step_seconds": "10 ms",
        "ports": {
            "inputs": {
                "vector": {"label": "Vector", "type": {"shape": [2]}},
                "scalar": {"label": "Scalar", "type": {"shape": []}},
            },
            "outputs": {"out": {"label": "Out", "type": {"shape": [2]}}},
        },
        "interior": {},
    }}}}


def _incompatible_wire(studio):
    source, target = {"owner": "inputs", "port": "scalar"}, {"owner": "outputs", "port": "out"}
    edit = studio.model_edit("model", {"operation": {
        "op": "connect", "graph": _GRAPH, "source": source, "target": target,
    }})
    [address] = edit["changed"]
    assert any(problem["address"] == address for problem in edit["problems"])
    reopened = studio.reopen(edit["snapshot"], "document", "model")
    wire = reopened["project"]["documents"]["model"]["payload"]["interior"]["wires"][address["element"]["id"]]
    assert (wire["source"], wire["target"]) == (source, target)


def _duplicate_ports(studio):
    edit = studio.model_edit("model", {"operation": {
        "op": "rename", "name": "Vector",
        "address": {"composites": [], "element": {"kind": "port", "end": {"owner": "inputs", "port": "scalar"}}},
    }})
    reopened = studio.reopen(edit["snapshot"], "document", "model")
    ports = reopened["project"]["documents"]["model"]["payload"]["ports"]["inputs"]
    assert [(key, port["label"]) for key, port in ports.items()] == [("vector", "Vector"), ("scalar", "Vector")]
    assert {problem["address"]["element"]["end"]["port"] for problem in edit["problems"]
            if problem["address"]["element"]["kind"] == "port"} == {"vector", "scalar"}


def _typed_signal_project():
    project = load_shipped_projects()["point-mass-reaching"].model_dump(mode="json", by_alias=True, exclude_unset=True)
    project["steps"] = {"simple-reaches": project["steps"]["simple-reaches"]}
    project["steps"]["check"] = managed_evaluation_step("simple-reaches")
    model = project["documents"]["point-mass"]["payload"]
    for key, leaf in (("position", "pos"), ("velocity", "vel")):
        model["interior"]["components"][key] = {
            "kind": "leaf", "label": key, "type": "feedbax.SelectState", "fields": {"paths": [leaf]},
        }
        model["interior"]["wires"][f"select-{key}"] = {
            "source": {"owner": "component", "component": "plant", "port": "effector"},
            "target": {"owner": "component", "component": key, "port": "state"},
        }
    model["ports"]["inputs"]["target_pos"]["type"] = {"shape": [2]}
    model["interior"]["components"]["point_receiver"] = {
        "kind": "composite", "label": "Point receiver",
        "ports": {"inputs": {
            "in": {"label": "Point", "type": {"shape": [2], "space": "point"}},
            "free": {"label": "Free", "type": {"shape": [2], "space": "free"}},
        }},
        "interior": {},
    }
    return project


def _incompatible_signal_space(studio):
    document = "point-mass"
    run = studio.run("simple-reaches")
    source = {"owner": "component", "component": "velocity", "port": "output"}
    target = {"owner": "component", "component": "point_receiver", "port": "in"}
    review = studio.post("/model/connections", document=document, run=run, graph=_GRAPH, source=source)
    verdict = next(item for item in review["targets"] if item["target"] == target)
    assert verdict["reason"] is None and "point" in verdict["problem"] and "free" in verdict["problem"]

    root_source = {"owner": "inputs", "port": "target_pos"}
    from_task = studio.post("/model/connections", document=document, run=run, graph=_GRAPH, source=root_source)
    free = next(item for item in from_task["targets"] if item["target"] == {**target, "port": "free"})
    assert free["reason"] is None and "expected free space, received point space" in free["problem"]
    evaluate = studio.post("/model/projection", document=document, run=studio.run("check"), evaluate_scope={"kind": "step"})
    assert any(port["readers"] == ["effector_position"] for port in evaluate["ports"])
    evaluate_trials = studio.post("/authoring/trials", run=studio.run("check"))
    assert ["Point mass", "effector", "position"] in [reset["path"] for reset in evaluate_trials["resets"]]

    edit = studio.model_edit(document, {"operation": {
        "op": "connect", "graph": _GRAPH, "source": source, "target": target,
    }})
    [wire] = edit["changed"]
    [problem] = edit["problems"]
    assert problem["address"] == wire and "point" in problem["reason"] and "free" in problem["reason"]
    saved = studio.reopen(edit["snapshot"], "document", document)
    persisted = saved["project"]["documents"][document]["payload"]["interior"]["wires"][wire["element"]["id"]]
    assert (persisted["source"], persisted["target"]) == (source, target)

    projection = studio.post("/model/projection", revision=saved["revision"], document=document, run=run)
    assert projection["problems"] == [problem]
    ports = {ModelAddress.model_validate(port["address"]).to_text(): port for port in projection["ports"]}

    def port_at(end, *path):
        element = {"kind": "port", "end": end}
        if path:
            element = {"kind": "field", "parent": element, "path": path}
        return ports[ModelAddress.model_validate({"composites": [], "element": element}).to_text()]

    root_effector = {"owner": "outputs", "port": "effector"}
    assert port_at(root_effector)["type"] == {"base": "state"}
    hand = port_at(root_effector, "pos")
    assert hand["type"] == {"base": "real", "shape": [2], "unit": "m", "space": "point"}
    assert hand["readers"] == ["effector_position"]
    effector = {"owner": "component", "component": "plant", "direction": "output", "port": "effector"}
    assert port_at(effector, "pos")["readers"] == ["effector_position"]
    assert port_at(effector, "vel")["readers"] == ["effector_final_velocity"]
    assert port_at(effector)["readers"] == ["effector_position", "effector_final_velocity"]

    current = _Studio(studio.client, studio.folders)
    grown = current.model_edit(document, {"operation": {
        "op": "connect", "graph": _GRAPH, "source": source,
        "target": {"owner": "outputs", "port": "__new_port__"}, "grow_target": True,
    }})
    output = next(address["element"]["end"] for address in grown["changed"] if address["element"]["kind"] == "port")

    def output_type(revision):
        view = studio.post("/model/projection", revision=revision, document=document, run=run)
        return next(port["type"] for port in view["ports"]
                    if port["address"]["element"] == {"kind": "port", "end": output})

    assert output_type(grown["snapshot"]["revision"]) == {"base": "real", "shape": [2], "unit": "m/s", "space": "free"}
    current = _Studio(studio.client, studio.folders)
    rewired = current.model_edit(document, {"operation": {
        "op": "connect", "graph": _GRAPH, "source": {**source, "component": "position"}, "target": output,
    }})
    assert output_type(rewired["snapshot"]["revision"]) == {"base": "real", "shape": [2], "unit": "m", "space": "point"}


def _invalid_model_field(studio):
    text = " 1 + "
    field = {"document": "model", "owner": _PLANT, "address": MASS, "run": studio.run("fee")}
    prepared = studio.post("/model/fields", **field, edit={"kind": "text", "text": text})
    assert any(problem["model_address"] == MASS for problem in prepared["problems"])
    edit = studio.model_edit("model", prepared["command"], files=prepared["files"])
    reopened = studio.reopen(edit["snapshot"], "document", "model")
    fields = reopened["project"]["documents"]["model"]["payload"]["interior"]["components"]["plant"]["fields"]
    assert fields["mass"] == text
    view = studio.post("/model/fields", **field, revision=reopened["revision"])
    [saved] = [item for item in view["fields"] if item["address"] == MASS]
    assert saved["text"] == text
    assert any(problem["model_address"] == MASS for problem in saved["problems"])


def _rule_project():
    return definition({"every": 2})


def _checkpoint_project():
    raw = _rule_project()
    raw["sweep"] = {"rows": {"case": {}}}
    return raw


def _bad_run_rule(studio):
    text = "  seed=  "
    variant = studio.run("baseline")["variant"]
    answer = studio.post("/authoring/run-rule", variant=variant, step="train", rule=text)
    assert answer["result"]["kind"] == "refused"
    path = "/inputs/subject/where"
    written = studio.write([{
        "address": {"kind": "step", "id": "baseline"},
        "patches": [{"op": "add", "path": path, "value": {"rule": text}}],
    }])
    reopened = studio.reopen(written, "step", "baseline")
    assert reopened["project"]["steps"]["baseline"]["inputs"]["subject"]["where"]["rule"] == text
    assert any(
        (problem["address"], problem["path"], problem["authored_path"])
        == ("steps/baseline.json", "inputs.subject.where.rule", f"{path}/rule")
        for problem in studio.plan_problems(reopened["revision"])
    )


def _invalid_checkpoint(studio):
    held = {"kind": "unresolved", "text": "  1, huh  "}
    run = studio.run("baseline")
    prepared = studio.post(
        "/authoring/checkpoint-rule", run=run,
        edit={"kind": "set", "scope": {"kind": "rows", "rows": [run["row"]]}, "rule": held},
    )
    assert any(problem["path"] == "inputs.subject.checkpoint" for problem in prepared["problems"])
    reopened = studio.reopen(studio.write(prepared["files"]), "step", "baseline")
    [exception] = reopened["project"]["steps"]["baseline"]["sweep"]["exceptions"].values()
    assert exception["patches"][CHECKPOINT_RULE_TARGET]["value"] == held["text"]
    assert any(
        (problem["address"], problem["path"]) == ("steps/baseline.json", "inputs.subject.checkpoint")
        and (problem.get("authored_path") or "").endswith("/value")
        for problem in studio.plan_problems(reopened["revision"])
    )


def _sweep_project():
    raw = _data()
    raw["steps"]["fee"]["sweep"] = {"axes": {"mass": {"path": _MASS_TARGET, "values": [1.5, 3.0]}}}
    return raw


def _missing_sweep_target(studio):
    edit = studio.model_edit("model", {"operation": {"op": "delete", "selection": [_PLANT]}})
    reopened = studio.reopen(edit["snapshot"], "document", "model")
    assert "plant" not in reopened["project"]["documents"]["model"]["payload"]["interior"]["components"]
    assert reopened["project"]["steps"]["fee"]["sweep"]["axes"]["mass"]["path"] == _MASS_TARGET
    assert any(
        (problem["address"], problem.get("authored_path")) == ("steps/fee.json", "/sweep/axes/mass/path")
        for problem in edit["sweep_problems"]
    )
    problems = studio.plan_problems(reopened["revision"])
    assert any(
        problem["address"] == "steps/fee.json"
        and (problem.get("authored_path") or "").startswith("/sweep/axes/mass/")
        and (problem.get("place") or {}).get("member") == MASS
        for problem in problems
    ), problems


def _set_project():
    raw = _data()
    raw["documents"]["draft"] = {
        "label": "Draft", "schema": "feedbax.set_document.v2", "payload": {"entries": {"one": {"label": "One"}}},
    }
    raw["steps"]["fee"]["sweep"] = {"axes": {"draft": {"set": "draft"}}}
    return raw


def _empty_required_set(studio):
    prepared = studio.post(
        "/authoring/sets", command={"kind": "remove_value", "step": "fee", "axis": "draft", "value": "one"},
    )
    reopened = studio.reopen(studio.write(prepared["files"]), "document", "draft")
    assert reopened["project"]["documents"]["draft"]["payload"]["entries"] == {}
    assert reopened["project"]["steps"]["fee"]["sweep"]["axes"]["draft"]["set"] == "draft"
    assert any(
        (problem["address"], problem.get("authored_path")) == ("documents/draft.json", "/payload/entries")
        for problem in studio.plan_problems(reopened["revision"])
    )


def _based_project():
    raw = _data()
    _based(raw, "heavy", _with_mass(_model(raw), 3.0))
    _reads(raw, "check", "model", "heavy")
    return raw


def _detach_with_broken_reader(studio):
    run = studio.run("check")
    edit = studio.member(kind="detach", member="heavy")
    reopened = studio.reopen(edit["snapshot"], "document", "heavy")
    document = reopened["project"]["documents"]["heavy"]
    assert "base" not in document and document["payload"]["interior"]["components"]["plant"]["fields"]["mass"] == 3.0
    assert reopened["project"]["steps"]["check"]["base"]["inline"]["trial_documents"]["model"] == {"document": "heavy"}
    assert any(
        (problem["address"], problem["path"]) == ("steps/check.json", "payload.trial_documents.model")
        for problem in edit["problems"]
    )
    view = studio.post("/model/evaluate-model", revision=reopened["revision"], run=run, scope={"kind": "step"})
    assert any(problem["path"] == "payload.trial_documents.model" for problem in view["problems"])


def _change_key(studio):
    edit = studio.member(kind="change_key", family="documents", member="model", key="arm")
    reopened = studio.reopen(edit["snapshot"], "document", "arm")
    authored = reopened["project"]
    assert "model" not in authored["documents"]
    assert authored["documents"]["arm"] == studio.before["project"]["documents"]["model"]
    assert authored["documents"]["heavy"]["base"] == {"document": "arm"}
    assert authored["steps"]["fee"]["base"]["inline"]["trial_documents"]["model"] == {"document": "arm"}
    assert authored["steps"]["check"]["base"]["inline"]["trial_documents"]["model"] == {"document": "heavy"}
    assert not edit["problems"]


@pytest.mark.parametrize(("binding", "document"), [
    ("python_task", "simple-reaches-task"),
    ("task", "simple-reaches-objective"),
    ("objective", "simple-reaches-task"),
])
def test_model_projection_refuses_a_wrong_trial_document_kind(tmp_path, binding, document):
    raw = load_shipped_projects()["point-mass-reaching"].model_dump(mode="json", by_alias=True, exclude_unset=True)
    step = raw["steps"]["simple-reaches"]
    bindings = step["base"]["inline"]["trial_documents"]
    if binding == "python_task":
        del bindings["task"]
    bindings[binding] = {"document": document}
    raw["steps"] = {"simple-reaches": step}
    app = create_app(projects_dir=tmp_path)
    with TestClient(app) as client:
        _save_project(app.state.project_folders, _PROJECT, Project.model_validate(raw))
        studio = _Studio(client, app.state.project_folders)
        response = client.post(f"{_URL}/model/projection", json={
            "revision": studio.before["revision"], "document": "point-mass", "run": studio.run("simple-reaches"),
        })
    assert response.status_code == 422, response.text
    assert f"trial_documents.{binding}" in response.json()["detail"]


@pytest.mark.parametrize(("seed", "witness"), [
    pytest.param(_graph_project, _incompatible_wire, id="incompatible-wire"),
    pytest.param(_typed_signal_project, _incompatible_signal_space, id="incompatible-space"),
    pytest.param(_graph_project, _duplicate_ports, id="duplicate-port-label"),
    pytest.param(_data, _invalid_model_field, id="invalid-model-field"),
    pytest.param(_rule_project, _bad_run_rule, id="bad-run-rule"),
    pytest.param(_checkpoint_project, _invalid_checkpoint, id="invalid-checkpoint"),
    pytest.param(_sweep_project, _missing_sweep_target, id="missing-sweep-target"),
    pytest.param(_set_project, _empty_required_set, id="empty-required-set"),
    pytest.param(_based_project, _detach_with_broken_reader, id="detach-broken-reader"),
    pytest.param(_based_project, _change_key, id="change-key-keeps-references"),
])
def test_studio_edit_is_written_and_reopened_at_its_authored_place(tmp_path, monkeypatch, seed, witness):
    """Catch a command that refuses an editable value, reports only a preview Problem, or loses its place on reopen."""
    folders = ProjectFolderService(tmp_path)
    _save_project(folders, _PROJECT, Project.model_validate(seed()))
    app = create_app(projects_dir=tmp_path, automatic_run="off")
    with TestClient(app) as client:
        monkeypatch.setattr(app.state.job_service.host, "jobs_snapshot", lambda **_: {"cursor": 0, "jobs": []})
        witness(_Studio(client, folders))
