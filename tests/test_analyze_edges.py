"""Split an Analyze card along its wires and merge it back, through Studio's member-command route."""

from fastapi.testclient import TestClient
import pytest

from feedbax.web.app import create_app
from tests.test_canonical_evaluate_admission import _save_project
from tests.test_project_report_plan import _report_project

PROJECT = "edges"
URL = f"/api/projects/{PROJECT}"


@pytest.fixture
def studio(tmp_path):
    app = create_app(projects_dir=tmp_path / "projects")
    with TestClient(app) as client:
        yield client, app.state.project_folders


def _snapshot(client):
    response = client.get(URL)
    assert response.status_code == 200, response.text
    return response.json()


def _command(client, command, status=200):
    snapshot = _snapshot(client)
    response = client.post(f"{URL}/members/command", json={
        "revision": snapshot["revision"], "presentation_revision": snapshot["presentation_revision"],
        "command": command,
    })
    assert response.status_code == status, response.text
    return response.json()


def _requests(folders):
    """Every operation, figure and Report request id, and the card holding the figure the Report reads."""
    plan = folders.read_plan(PROJECT, revision=None).plan
    assert not plan.diagnostics, plan.diagnostics
    ids = sorted(element.request_id for element in plan.elements.values())
    (report,) = (run for run in plan.runs if run.kind == "report")
    (figure,) = report.inputs.values()
    assert plan.elements[figure.element].kind == "render"
    return ids, report.request_id, figure.element.run.step


def test_a_card_split_into_one_card_per_element_and_merged_back_recomputes_nothing(studio):
    client, folders = studio
    _save_project(folders, PROJECT, _report_project())
    original = _snapshot(client)["project"]
    ids, report, holder = _requests(folders)
    assert holder == "analyze" and len(ids) == 3

    middle = _command(client, {"kind": "split", "member": "analyze", "elements": ["summarize", "profile"]})["member"]
    last = _command(client, {"kind": "split", "member": middle, "elements": ["profile"]})["member"]
    project = _snapshot(client)["project"]
    assert project["steps"]["report"]["inputs"]["velocity_summary"]["step"] == last
    assert project["steps"][last]["inputs"] == {
        "summary": {"step": middle, "element": "summarize", "product": "summary"},
    }
    assert _requests(folders) == (ids, report, last)

    assert _command(client, {"kind": "merge", "member": "analyze", "others": [middle, last]})["member"] == "analyze"
    assert _snapshot(client)["project"] == original
    assert _requests(folders) == (ids, report, "analyze")


def test_split_and_merge_are_refused_for_a_shared_document(studio):
    client, folders = studio
    project = _report_project()
    project.steps["again"] = project.steps["analyze"].model_copy(deep=True)
    _save_project(folders, PROJECT, project)
    before = _snapshot(client)["project"]
    assert folders.read_plan(PROJECT, revision=None).regroup_refused == {"analyze", "again"}
    refused = _command(client, {"kind": "split", "member": "analyze", "elements": ["profile"]}, status=422)
    assert "shared" in refused["detail"]
    _command(client, {"kind": "merge", "member": "analyze", "others": ["again"]}, status=422)
    assert _snapshot(client)["project"] == before
