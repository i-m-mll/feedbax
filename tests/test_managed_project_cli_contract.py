"""Require every managed command to honor the shared CLI output contract."""

import pytest
from fastapi.testclient import TestClient

from feedbax.bin.project_commands import PROJECT_COMMANDS
from feedbax.cli import main
from feedbax.contracts.jobs import (
    StudioJobSubmission,
    StudioProjectWindowMapProblem,
    StudioSavedTrainRuns,
)
from feedbax.contracts.project_io import (
    ProjectConfiguration,
    dump_text_project,
    read_text_project_snapshot,
)
from feedbax.contracts.studio_authoring import StudioAuthoringCatalog
from feedbax.web.app import create_app
from tests.support.managed_train_fixture import managed_train_project
from tests.test_local_job_host import admit, job_row, run_job, wait_until
from tests.test_local_job_host import host as host_fixture
from tests.test_text_project_plan_cli import _edit, _write_walkthrough

host = host_fixture


def test_managed_commands_share_http_contracts_and_exit_codes(host, tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("FEEDBAX_STUDIO_EVALUATION_CANDIDATE", raising=False)
    monkeypatch.setenv("FEEDBAX_ACTOR_LABEL", "environment actor")
    monkeypatch.setenv("FEEDBAX_HOST_ROOT", str(host.root))
    projects = tmp_path / "projects"
    walkthrough = projects / "reaching"
    _, files = _write_walkthrough(walkthrough)
    train = projects / "managed checkout"
    declaration = train / "managed.feedbax"
    for relative, data in dump_text_project(
        managed_train_project(), ProjectConfiguration(project="managed", store="results"),
    ).items():
        path = train / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    release = tmp_path / "release"
    host.policy(compute=1)
    gate = run_job(
        "cli-contract-gate",
        f"from pathlib import Path\nimport time\n"
        f"while not Path({str(release)!r}).exists(): time.sleep(.01)",
    )
    admit(host, "gate", gate)
    wait_until(lambda: job_row(host, gate.id, "running"))

    baseline = tmp_path / "review.json"
    cases = {
        "plan": (StudioSavedTrainRuns, (str(walkthrough), "--out", str(baseline))),
        "status": (StudioSavedTrainRuns, (str(walkthrough / "reaching.feedbax"),)),
        "run": (StudioJobSubmission, (str(declaration), "train", "--label", "explicit actor")),
        "describe": (StudioAuthoringCatalog, ()),
    }
    assert cases.keys() == PROJECT_COMMANDS.keys()

    def call(args, expected_exit, response=None):
        assert main(args) == expected_exit
        captured = capsys.readouterr()
        if response is None:
            assert captured.out == ""
            payload = None
        else:
            payload = response.model_validate_json(captured.out)
            assert len(captured.out.splitlines()) == 1
        problems = []
        for line in captured.err.splitlines():
            problem = StudioProjectWindowMapProblem.model_validate_json(line)
            assert problem.code and problem.address and problem.path
            problems.append(problem)
        return payload, problems

    results = {}
    for name in PROJECT_COMMANDS:
        model, args = cases[name]
        with pytest.raises(SystemExit) as help_exit:
            main([name, "--help"])
        assert help_exit.value.code == 0
        help_text = capsys.readouterr().out
        assert all(text in help_text for text in (
            "--json", "0  Success", "1  Review drift", "2  Invalid project",
        ))
        _, problems = call([name, "--not-a-real-flag", "--json"], 2)
        (problem,) = problems
        assert problem.code == "invalid_arguments"
        assert problem.address == problem.path == "--not-a-real-flag"
        assert "suggest" not in problem.detail and "did you mean" not in problem.detail
        results[name], problems = call([name, *args, "--json"], 0, model)
        assert problems == []

    _, problems = call(["pla", "--json"], 2)
    assert [problem.code for problem in problems] == ["unknown_command"]

    plan = results["plan"]
    status = results["status"]
    assert plan.window_map is not None and status.window_map is not None
    assert plan.window_map.requests == status.window_map.requests
    assert {request.request_id.partition(":")[0] for request in plan.window_map.requests} == {
        "train", "evaluate", "analyze", "render", "report",
    }
    assert not (walkthrough / "results").exists()
    submission = results["run"]
    assert submission.actor is not None and submission.actor.client == "cli"
    assert submission.actor.label == "explicit actor"
    assert submission.project_id == train.name
    assert [item.disposition for item in submission.items] == ["admitted"]
    with TestClient(create_app(projects_dir=projects)) as client:
        response = client.get(
            f"/api/projects/{walkthrough.name}/train-runs",
            params={"revision": read_text_project_snapshot(walkthrough).revision, "window_map": True},
        )
        assert response.status_code == 200
        assert StudioSavedTrainRuns.model_validate(response.json()) == status
        response = client.get(f"/api/submissions/{submission.id}")
        assert response.status_code == 200
        assert StudioJobSubmission.model_validate(response.json()) == submission
        response = client.get("/api/projects/authoring/catalog")
        assert response.status_code == 200
        assert StudioAuthoringCatalog.model_validate(response.json()) == results["describe"]

    selected, problems = call(["describe", "step:report", "--json"], 0, StudioAuthoringCatalog)
    assert problems == []
    assert selected.step_payloads == {"report": results["describe"].step_payloads["report"]}
    assert selected.document_families == selected.registries == {}
    _, problems = call(["describe", "step:missing", "--json"], 2)
    assert [(problem.code, problem.address) for problem in problems] == [
        ("unknown_selector", "step:missing"),
    ]

    check_args = ["plan", str(walkthrough), "--json", "--check", str(baseline)]
    unchanged, problems = call(check_args, 0, StudioSavedTrainRuns)
    assert not problems
    assert unchanged.review and {entry.state for entry in unchanged.review} == {"unchanged"}

    _edit(
        walkthrough, files, "velocity-report",
        lambda doc: doc["blocks"][0].update(markdown="Edited interpretation."),
    )
    drift, problems = call(check_args, 1, StudioSavedTrainRuns)
    assert not problems and drift.review
    (changed,) = (entry for entry in drift.review if entry.state == "changed")
    assert changed.key.step == "report" and changed.path == "payload.blocks.0.markdown"

    rejected, problems = call(
        ["run", str(walkthrough), "report", "--json"], 1, StudioJobSubmission,
    )
    (item,) = rejected.items
    assert item.disposition == "rejected"
    assert [(problem.code, problem.address) for problem in problems] == [
        ("admission_rejected", item.run_key),
    ]

    _edit(
        walkthrough, files, "velocity-analysis",
        lambda doc: doc["operations"]["align"].update(operation="feedbax.unknown"),
    )
    project_errors = {}
    for name in PROJECT_COMMANDS.keys() - {"describe"}:
        model = cases[name][0] if name != "run" else None
        result, problems = call([name, str(walkthrough), "--json"], 2, model)
        project_errors[name] = problems
        assert any(
            problem.address == "steps/analyze.json"
            and problem.path == "payload.operations.align.operation"
            for problem in problems
        )
        if result is not None:
            assert result.window_map is not None
            assert result.window_map.problems[:len(problems)] == problems
    assert project_errors["plan"] == project_errors["status"] == project_errors["run"]

    (walkthrough / "steps/report.json").unlink()
    removed, _ = call(check_args, 2, StudioSavedTrainRuns)
    assert removed.review and any(
        entry.state == "removed" and entry.key.step == "report" for entry in removed.review
    )
    release.touch()
