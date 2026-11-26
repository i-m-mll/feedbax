"""Task and Model authoring answer from the same authored Set campaign."""

from copy import deepcopy

import pytest
from fastapi.testclient import TestClient

from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.project import Project
from feedbax.contracts.project_resolution import compose_document
from feedbax.web.app import create_app
from tests.test_canonical_evaluate_admission import _save_project
from web.e2e.studio.fixtures.canonical_train import hold_campaign_files


@pytest.fixture
def campaign(tmp_path):
    files = hold_campaign_files()
    documents = {file["id"]: file["document"] for file in files if file["kind"] == "document"}
    steps = {file["id"]: file["step"] for file in files if file["kind"] == "step"}
    swept = deepcopy(steps["train"])
    swept["label"] = "Swept training"
    placement = swept.pop("set_fields")["kicks"]
    sweep_set = deepcopy(documents[placement["axis"]["set"]])
    sweep_set["label"] = "Swept command kicks"
    for entry in sweep_set["payload"]["entries"].values():
        (path, value), = entry["differences"].items()
        parent, _, attachment = path.rpartition("/")
        entry["differences"] = {parent: {"value": {attachment: value["value"]}}}
    documents["sweep-kicks"] = sweep_set
    swept["sweep"] = {"axes": {"kicks": {"set": "sweep-kicks"}}}
    steps["swept"] = swept
    project = Project.model_validate({"documents": documents, "steps": steps})
    app = create_app(projects_dir=tmp_path)
    folders = app.state.project_folders
    _save_project(folders, "hold-campaign", project)
    snapshot = folders.read("hold-campaign")
    authored_model = compose_document(snapshot.project.documents.get, "heavy-point-mass").payload
    entries = snapshot.project.documents["command-kicks"].payload["entries"]
    unbound = snapshot.project.documents["level-task"].base.document
    with TestClient(app) as client:
        yield client, snapshot.revision, unbound, authored_model, entries


def _post(view, step, campaign, *, row="default", **kwargs):
    client, revision, *_ = campaign
    response = client.post(f"/api/projects/hold-campaign/{view}", json={
        "revision": revision, "run": {"variant": "default", "step": step, "row": row}, **kwargs,
    })
    assert response.status_code == 200, response.text
    return response.json()


@pytest.mark.parametrize("step", ["train", "kick-check"])
def test_task_choices_pair_without_generated_set_inputs(campaign, step):
    view = _post("authoring/trials", step, campaign)
    assert not view["problems"], view["problems"]
    choices = {choice["document"]: choice["problem_indices"] for choice in view["tasks"]}
    assert choices[view["task_document"] or "level-task"] == []
    index, = choices[campaign[2]]
    (problem,) = view["task_problems"]
    assert view["task_problems"][index] == problem
    assert "Level" in problem["reason"]
    assert "set-" not in problem["reason"]
    assert problem["path"][:2] == ["model", "inputs"]
    assert ModelAddress.from_text(problem["path"][-1]).element.end.port == "level"


def _model_keys(model):
    interior = model["interior"]
    branches = {
        key: _model_keys(component)
        for key, component in interior.get("components", {}).items() if component["kind"] == "composite"
    }
    return set(interior.get("components", {})), set(interior.get("wires", {})), branches


@pytest.mark.parametrize("step", ["train", "kick-check"])
def test_task_and_model_project_the_authored_model_members(campaign, step):
    *_prefix, authored, _entries = campaign
    view = _post("authoring/trials", step, campaign)
    inputs = authored["ports"]["inputs"]
    assert [(port["id"], port["name"]) for port in view["inputs"]] == [
        (id, port["label"]) for id, port in inputs.items()
    ]
    assert {output["address"]["element"]["end"]["port"] for output in view["outputs"]} == set(authored["ports"]["outputs"])
    model = _post("model/projection", step, campaign, document="heavy-point-mass")
    assert _model_keys(model["resolved_model"]) == _model_keys(authored)
    ports = [
        place["element"]["end"] for port in model["ports"] if not (place := port["address"]).get("composites")
        and place["element"]["kind"] == "port"
    ]
    assert {port["port"] for port in ports if port["owner"] == "inputs"} == set(inputs)
    assert {bound["element"]["end"]["port"] for bound in model["task_inputs"]} == set(inputs)
    assert {signal["id"] for signal in model["task_signals"]} == set(view["task"]["signals"])
    assert all("set-" not in text for text in model["address_texts"].values())
    assert all(
        "set-" not in name["id"] for site in view["sites"].values() for name in site["names"]
        if "set_entry" not in name
    )


@pytest.mark.parametrize(("step", "row", "local_axis", "origins"), [
    ("train", "default", "kicks", {"kicks": ("train", ("drift", "pulse"))}),
    ("kick-check", "default", "kick-bank", {
        "kicks": ("train", ("drift", "pulse")), "kick-bank": ("kick-check", ("drift", "pulse")),
    }),
    ("swept", "kicks=pulse-x-minus", None, {"kicks": ("swept", ("pulse",))}),
])
def test_set_variables_carry_the_label_of_their_authored_entry(campaign, step, row, local_axis, origins):
    *_prefix, authored = campaign
    view = _post("authoring/trials", step, campaign, row=row)
    assert [placement["axis"] for placement in view["set_fields"]] == ([local_axis] if local_axis else [])
    for placement in view["set_fields"]:
        for entry in placement["entries"]:
            key = entry["entry"]
            assert entry["label"] == (authored[key]["label"] if key is not None else None)
            assert entry["member_count"] == (4 if key == "pulse" else None)
    variables = {}
    for variable in view["variables"]:
        if (origin := variable.get("set_entry")) is None:
            continue
        key = (origin["axis"], origin["entry"], origin["variable"])
        assert key not in variables
        variables[key] = variable["id"]
        entry = authored[origin["entry"]]
        assert origin["step"] == origins[origin["axis"]][0]
        assert variable["name"] == entry["task_variables"][origin["variable"]]["label"]
        assert origin["label"] == entry["label"]
        assert origin["member_count"] == (4 if origin["entry"] == "pulse" else None)
        assert view["task"]["variables"][variable["id"]]["label"] == variable["name"]
    assert variables.keys() == {
        (axis, entry, variable) for axis, (_, entries) in origins.items()
        for entry in entries for variable in authored[entry]["task_variables"]
    }
    if step == "swept":
        model = _post("model/projection", step, campaign, document="heavy-point-mass", row=row)["resolved_model"]
        assert set(model["interior"]["wires"]["drive"]["attachments"]) == {"kick-pulse"}
    for site in view["sites"].values():
        for named in site["names"]:
            if (origin := named.get("set_entry")) is not None:
                key = (origin["axis"], origin["entry"], origin["variable"])
                assert variables[key] == named["id"]
                assert named["name"] == authored[origin["entry"]]["task_variables"][origin["variable"]]["label"]
