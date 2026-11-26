"""Task and Model authoring answer from the same authored Set campaign."""

from copy import deepcopy

import pytest
from fastapi.testclient import TestClient

from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.project import Project
from feedbax.contracts.project_resolution import compose_document
from feedbax.web.app import create_app
from tests.test_canonical_evaluate_admission import _save_project
from web.e2e.studio.fixtures.canonical_train import hold_campaign_files, successor_file


@pytest.fixture
def campaign(tmp_path, request):
    files = hold_campaign_files()
    documents = {file["id"]: file["document"] for file in files if file["kind"] == "document"}
    steps = {file["id"]: file["step"] for file in files if file["kind"] == "step"}
    if getattr(request, "param", None) == "swept":
        swept = deepcopy(steps["train"])
        swept["label"] = "Swept training"
        placement = swept.pop("set_fields")["kicks"]
        swept["sweep"] = {"axes": {"kicks": placement["axis"]}}
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


@pytest.mark.parametrize(("step", "row", "local_axis", "origins", "campaign"), [
    ("train", "default", "kicks", {"kicks": ("train", ("drift", "pulse"))}, None),
    ("kick-check", "default", None, {"kicks": ("train", ("drift", "pulse"))}, None),
    ("swept", "kicks=pulse-x-minus", None, {"kicks": ("swept", ("pulse",))}, "swept"),
], indirect=["campaign"])
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
                variable = next(value for value in view["variables"] if value["id"] == named["id"])
                assert origin == {key: variable["set_entry"][key] for key in (
                    "step", "axis", "set", "entry", "variable",
                )}


def test_editing_an_inherited_set_variable_writes_the_set_entry(campaign):
    view = _post("authoring/trials", "kick-check", campaign)
    inherited = next(variable for variable in view["variables"] if (
        (origin := variable.get("set_entry")) is not None
        and origin["step"] == "train" and origin["entry"] == "pulse"
    ))
    edited = _post("authoring/trials", "kick-check", campaign, change={
        "kind": "expression", "path": f"/task/variables/{inherited['id']}/value",
        "edit": {"kind": "text", "text": "0.7 * sign * axis"},
    })
    (file,) = edited["files"]
    assert file["address"] == {"kind": "document", "id": "command-kicks"}
    client, revision, *_rest, entries = campaign
    response = client.patch("/api/projects/hold-campaign", json={"revision": revision, "files": edited["files"]})
    assert response.status_code == 200, response.text
    saved = client.app.state.project_folders.read("hold-campaign").project.documents["command-kicks"].payload
    expected = deepcopy(entries)
    expected["pulse"]["task_variables"]["kick"]["value"] = "0.7 * sign * axis"
    assert saved["entries"] == expected


@pytest.mark.parametrize("continued", [False, True], ids=["direct", "through-train-checkpoint"])
def test_evaluate_refuses_own_trial_set_when_subject_train_placed_one(tmp_path, continued):
    files = hold_campaign_files()
    documents = {file["id"]: file["document"] for file in files if file["kind"] == "document"}
    steps = {file["id"]: file["step"] for file in files if file["kind"] == "step"}
    evaluate = steps["kick-check"]
    evaluate["set_fields"] = {"kick-bank": {"axis": {"set": "command-kicks"}, "sampling": {"kind": "listed"}}}
    if continued:
        file = successor_file()
        steps[file["id"]] = file["step"]
        evaluate["inputs"]["subject"]["step"] = file["id"]

    app = create_app(projects_dir=tmp_path)
    folders = app.state.project_folders
    _save_project(folders, "hold-campaign", Project.model_validate({"documents": documents, "steps": steps}))
    settled = folders.read_plan("hold-campaign", revision=None)
    assert {row.key.step for row in settled.plan.runs} == ({"train", "successor"} if continued else {"train"})
    url = "/api/projects/hold-campaign/train-runs"
    with TestClient(app) as client:
        response = client.get(url, params={"revision": settled.revision, "window_map": True})
        assert response.status_code == 200, response.text
        window = response.json()["window_map"]
        [problem] = [item for item in window["problems"] if item["step"] == "kick-check"]
        assert problem["path"] == "/set_fields"
        assert problem["detail"] == "Kicked training places Command kicks on its trials, so Kick check cannot place a Set of its own."
        [row] = [item for item in window["rows"] if item["key"]["step"] == "kick-check"]
        assert row["status"] == "cannot_run"
