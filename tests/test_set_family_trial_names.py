"""Studio shows the authored trial labels, including the coordinates of crossed Set families."""

from base64 import b64decode

import pytest
from fastapi.testclient import TestClient

from feedbax.application.project_writer import dump_text_project
from feedbax.contracts.project import Project
from feedbax.contracts.project_io import ProjectConfiguration
from feedbax.contracts.request_identity import decode_request
from feedbax.web.app import create_app
from feedbax.web.services.project_folder import ProjectFolderService
from tests.test_canonical_evaluate_admission import _save_project
from tests.test_synthesis_runtime import synthesis_project
from web.e2e.studio.fixtures.canonical_train import (
    HOLD_CAMPAIGN_LOADED, HOLD_CAMPAIGN_MODEL, hold_campaign_archive, hold_campaign_files, loaded_check_file,
)


_TRIAL_LABELS = (
    "Reach 0, draw 0", "Reach 0, draw 1", "Reach 1, draw 0", "Reach 1, draw 1",
    "Reach 2, draw 0", "Reach 2, draw 1", "Reach 3, draw 0", "Reach 3, draw 1",
)
_ENTRY_LABELS = (
    "Drift", "Command pulse x Minus", "Command pulse x Plus", "Command pulse y Minus", "Command pulse y Plus",
)


def test_evaluate_trial_names_project_authored_labels_without_changing_requests(tmp_path):
    files = hold_campaign_files()
    content = {kind + "s": {file["id"]: file[kind] for file in files if file["kind"] == kind}
               for kind in ("document", "step")}
    del content["steps"]["train"]["set_fields"]
    content["steps"]["kick-check"]["set_fields"] = {
        "kick-bank": {"axis": {"set": "command-kicks"}, "sampling": {"kind": "listed"}},
    }
    root = tmp_path / "campaign"
    config = ProjectConfiguration(project="campaign", store="results")

    def save():
        for name, data in dump_text_project(Project.model_validate(content), config).items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)

    def read(client):
        revision = client.get("/api/projects/campaign").json()["revision"]
        response = client.get("/api/projects/campaign/train-runs", params={"revision": revision, "window_map": True})
        assert response.status_code == 200, response.text
        window = response.json()["window_map"]
        [run] = [row for row in window["rows"] if row["key"]["step"] == "kick-check"]
        [saved] = [row for row in window["evaluations"] if row["key"]["step"] == "kick-check"]
        assert saved["trials"] == run["evaluation_conditions"]
        return saved["trials"]

    save()
    folders = ProjectFolderService(tmp_path)
    before = folders.read_plan("campaign", revision=None).plan
    with TestClient(create_app(projects_dir=tmp_path)) as client:
        trials = read(client)
        names = [trial["name"] for trial in trials]
        assert names == [f"{trial} · {entry}" for trial in _TRIAL_LABELS for entry in _ENTRY_LABELS]
        assert len(set(names)) == len(names)
        assert all(trial["id"] != trial["name"] for trial in trials)
        content["documents"]["command-kicks"]["payload"]["entries"]["drift"]["label"] = "Lateral drift"
        save()
        renamed = read(client)
        assert renamed == [entry | {"name": entry["name"].replace("Drift", "Lateral drift")} for entry in trials]
    after = folders.read_plan("campaign", revision=None).plan
    assert after.requests == before.requests


@pytest.mark.parametrize("subject", ["loaded_model", "synthesize"])
def test_evaluate_condition_domain_matches_the_accepted_trial_rule(tmp_path, subject):
    if subject == "loaded_model":
        files = [*hold_campaign_files(), loaded_check_file()]
        raw = {kind + "s": {file["id"]: file[kind] for file in files if file["kind"] == kind}
               for kind in ("document", "step")}
        step, count = "loaded-check", 40
    else:
        raw = synthesis_project()
        raw["steps"]["baseline"]["base"]["inline"]["trial_documents"] = {
            "task": {"document": "stabilization-task"},
        }
        step, count = "baseline", 4
    raw["steps"][step]["base"]["inline"]["params"]["trial_set"] = {"kind": "every_condition"}
    project = "trial-domain"
    url = f"/api/projects/{project}"
    with TestClient(create_app(projects_dir=tmp_path, automatic_run="off")) as client:
        folders = client.app.state.project_folders
        _save_project(folders, project, Project.model_validate(raw))
        if subject == "loaded_model":
            revision = folders.read(project).revision
            response = client.post(f"{url}/loaded-models/import", params={
                "revision": revision, "name": HOLD_CAMPAIGN_LOADED, "model": HOLD_CAMPAIGN_MODEL,
            }, content=b64decode(hold_campaign_archive()), headers={"Content-Type": "application/octet-stream"})
            assert response.status_code == 201, response.text

        def trials(plan):
            [run] = [row for row in plan.runs if row.key.step == step]
            assert run.admission_problem is None
            spec = decode_request(plan.requests[run.request_id], plan.documents.__getitem__).spec
            return spec.params["trials"]

        every = folders.read_plan(project, revision=None)
        response = client.get(f"{url}/train-runs", params={"revision": every.revision, "window_map": True})
        assert response.status_code == 200, response.text
        [row] = [row for row in response.json()["window_map"]["rows"] if row["key"]["step"] == step]
        offered = [trial["id"] for trial in row["evaluation_conditions"]]
        assert len(offered) == count
        assert offered == trials(every.plan)

        selected = folders.read(project).project.model_copy(deep=True)
        selected.steps[step].base.inline["params"]["trial_set"] = {"kind": "conditions", "ids": offered}
        _save_project(folders, project, selected)
        assert trials(folders.read_plan(project, revision=None).plan) == offered
