"""The named origin of a Model field supplied by a Set family."""

from fastapi.testclient import TestClient

from feedbax.application.project_writer import dump_text_project
from feedbax.contracts.project import Project
from feedbax.contracts.project_io import ProjectConfiguration
from feedbax.web.app import create_app
from web.e2e.studio.fixtures.canonical_train import hold_campaign_files


def test_a_swept_model_field_names_the_authored_set_family(tmp_path):
    files = hold_campaign_files()
    raw = {
        f"{kind}s": {file["id"]: file[kind] for file in files if file["kind"] == kind}
        for kind in ("document", "step")
    }
    train = raw["steps"]["train"]
    del train["set_fields"]
    del raw["steps"]["kick-check"]
    axis, axis_name = "disturbance", "Command disturbance"
    train["sweep"] = {"axes": {axis: {"set": "command-kicks", "label": axis_name}}}
    path = "/interior/components/plant/fields/mass"
    entries = raw["documents"]["command-kicks"]["payload"]["entries"]
    entries["drift"]["differences"] = {}
    entries["pulse"]["differences"] = {path: {"value": 3}}

    root = tmp_path / "hold-campaign"
    project = Project.model_validate(raw)
    configuration = ProjectConfiguration(project="hold-campaign", store="results")
    for relative, content in dump_text_project(project, configuration).items():
        file = root / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(content)

    with TestClient(create_app(projects_dir=tmp_path, automatic_run="off")) as client:
        settled = client.app.state.project_folders.read_plan("hold-campaign", revision=None)
        assert not settled.plan.diagnostics, [problem.detail for problem in settled.plan.diagnostics]
        (row,) = (
            row for row in settled.plan.rows if row.coordinates[axis].label == "Command pulse x Minus"
        )
        response = client.post("/api/projects/hold-campaign/model/fields", json={
            "revision": settled.revision,
            "document": "heavy-point-mass",
            "owner": {"composites": [], "element": {"kind": "component", "id": "plant"}},
            "run": {"variant": row.key.variant, "step": row.key.step, "row": row.key.row},
        })
        assert response.status_code == 200, response.text
        (mass,) = (field for field in response.json()["fields"] if field["label"] == "Mass")
        assert mass["sources"] == [{
            "kind": "sweep_value", "key": "command-kicks", "name": "Command pulse {axis} {sign}",
            "axis_name": axis_name, "member_count": 4, "step": "train", "axis": axis, "value": "pulse",
            "path": "/entries/pulse/differences/~1interior~1components~1plant~1fields~1mass/value",
        }]
