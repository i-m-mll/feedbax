"""Studio shows the authored trial labels, including the coordinates of crossed Set families."""

from fastapi.testclient import TestClient

from feedbax.application.project_writer import dump_text_project
from feedbax.contracts.project import Project
from feedbax.contracts.project_io import ProjectConfiguration
from feedbax.web.app import create_app
from feedbax.web.services.project_folder import ProjectFolderService
from web.e2e.studio.fixtures.canonical_train import hold_campaign_files


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
