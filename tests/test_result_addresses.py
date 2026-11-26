"""Result addresses: labels, element keys held by one document family, and `[card/]element[@row]` lookups."""

import json

from fastapi.testclient import TestClient
import pytest

from feedbax.cli import main
from feedbax.contracts.project import BasedDocument, RootDocument
from feedbax.planner.project_plan import ElementKey, RunKey, compile_project
from feedbax.contracts.project_resolution import ResolutionCode
from feedbax.host import HostClient
from feedbax.web.app import create_app
from tests.test_canonical_evaluate_admission import _save_project
from tests.test_project_analyze_plan import _operations
from tests.test_project_report_plan import _report_project

PROJECT = "reaching"
URL = f"/api/projects/{PROJECT}"
REPORT = RunKey("default", "report", "default")


@pytest.fixture
def studio(tmp_path):
    app = create_app(projects_dir=tmp_path / "projects")
    with TestClient(app) as client:
        folders = app.state.project_folders
        _save_project(folders, PROJECT, _report_project())
        yield client, folders


def _command(client, command, status=200):
    snapshot = client.get(URL).json()
    response = client.post(f"{URL}/members/command", json={
        "revision": snapshot["revision"], "presentation_revision": snapshot["presentation_revision"],
        "command": command,
    })
    assert response.status_code == status, response.text
    return response.json()


def _plan(folders):
    plan = folders.read_plan(PROJECT, revision=None).plan
    assert not plan.diagnostics, plan.diagnostics
    return plan


def _request_ids(plan):
    return {run.key: run.request_id for run in plan.runs} | {
        key: element.request_id for key, element in plan.elements.items()
    } | {row.key: row.request_id for row in plan.rows if hasattr(row, "request_id")}


def test_labels_and_element_key_changes_change_no_request(studio):
    client, folders = studio
    before = _request_ids(_plan(folders))
    _command(client, {"kind": "rename", "family": "steps", "member": "analyze", "label": "Velocity"})
    _command(client, {
        "kind": "rename", "family": "elements", "document": "velocity-analysis", "member": "align", "label": "Aligned",
    })
    project = folders.read_plan(PROJECT, revision=None).plan.variant_projects["default"]
    assert project.steps["analyze"].label == "Velocity"
    assert project.documents["velocity-analysis"].payload["operations"]["align"]["label"] == "Aligned"
    assert _request_ids(_plan(folders)) == before

    _command(client, {
        "kind": "change_key", "family": "elements", "document": "velocity-analysis", "member": "summarize", "key": "summary",
    })
    _command(client, {
        "kind": "change_key", "family": "elements", "document": "velocity-analysis", "member": "profile", "key": "speed",
    })
    after = _plan(folders)
    renamed = {
        ElementKey(key.run, {"summarize": "summary", "profile": "speed"}.get(key.name, key.name))
        if isinstance(key, ElementKey) else key: request
        for key, request in before.items()
    }
    assert _request_ids(after) == renamed
    reference = folders.read_plan(PROJECT, revision=None).plan.variant_projects["default"].steps["report"].inputs["velocity_summary"]
    assert (reference.step, reference.element) == ("analyze", "speed")
    _command(client, {
        "kind": "change_key", "family": "elements", "document": "velocity-analysis", "member": "speed", "key": "align",
    }, 422)


def test_a_card_qualifier_that_no_longer_holds_the_element_resolves_by_element():
    operations = _operations()
    project = _report_project()
    report = compile_project(project, operations=operations).runs
    expected = next(run.request_id for run in report if run.key == REPORT)
    project.steps["summary"] = project.steps.pop("analyze")
    moved = compile_project(project, operations=operations)
    assert not moved.diagnostics
    assert next(run.request_id for run in moved.runs if run.key == REPORT) == expected

    project.steps["copy"] = project.steps["summary"]
    (problem,) = compile_project(project, operations=operations).diagnostics
    assert (problem.code, problem.step, problem.path) == (
        ResolutionCode.BAD_REFERENCE, "report", "inputs.velocity_summary.element",
    )


def test_a_key_held_by_two_unrelated_documents_is_a_diagnostic(studio):
    project = _report_project()
    analysis = project.documents["velocity-analysis"]
    project.documents["velocity-analysis-b"] = BasedDocument.model_validate({
        "label": "Velocity analysis b",
        "schema": analysis.schema_id, "base": {"document": "velocity-analysis"},
    })
    assert not compile_project(project, operations=_operations()).diagnostics
    project.documents["zz-analysis"] = RootDocument.model_validate(analysis.model_dump(by_alias=True))
    problems = compile_project(project, operations=_operations()).diagnostics
    assert {(problem.document, problem.element) for problem in problems} == {
        ("zz-analysis", name) for name in ("align", "summarize", "profile")
    }

    client, folders = studio
    copy = _command(client, {"kind": "duplicate", "family": "documents", "member": "velocity-analysis"})["member"]
    payload = folders.read_plan(PROJECT, revision=None).plan.variant_projects["default"].documents[copy].payload
    assert set(payload["operations"]) == {"align-2", "summarize-2"}
    assert payload["operations"]["summarize-2"]["inputs"]["velocity"]["element"] == "align-2"
    assert set(payload["views"]) == {"profile-2"}
    _plan(folders)


def test_status_by_card_element_and_by_element_agree(tmp_path, monkeypatch, capsys):
    root = tmp_path / PROJECT
    project = _report_project()
    from feedbax.application.project_writer import dump_text_project
    from feedbax.contracts.project_io import ProjectConfiguration

    for relative, data in dump_text_project(project, ProjectConfiguration(project=PROJECT, store="results")).items():
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_bytes(data)
    (root / "steps/analyze.json").rename(root / "steps/summary.json")
    monkeypatch.setattr(
        HostClient, "jobs_snapshot", lambda self, **_: {"cursor": 0, "oldest_cursor": 0, "jobs": []},
    )
    monkeypatch.setenv("FEEDBAX_HOST_ROOT", str(tmp_path / "host"))
    outputs = {}
    for command in ("status", "plan"):
        for address in ("analyze/profile", "profile", "summary/profile@default"):
            assert main([command, str(root), address]) == 0, capsys.readouterr()
            outputs[command, address] = capsys.readouterr().out
        assert len({outputs[command, address] for address in ("analyze/profile", "profile", "summary/profile@default")}) == 1
    (line,) = outputs["status", "profile"].splitlines()[1:]
    assert line.startswith("summary/profile\t")
    (line,) = outputs["plan", "profile"].splitlines()
    assert line.startswith("summary/profile render:")
    assert main(["plan", str(root), line.split()[0]]) == 0
    assert capsys.readouterr().out == outputs["plan", "profile"]
    assert main(["status", str(root), "summary"]) == 0
    assert len(capsys.readouterr().out.splitlines()) == 1 + 4
    assert main(["status", str(root), "nothing"]) == 2
    assert "No card holds 'nothing'." in capsys.readouterr().err
    assert json.loads((root / "steps/report.json").read_bytes())["inputs"]["velocity_summary"]["step"] == "analyze"
