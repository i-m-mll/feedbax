"""Require every managed command to honor the shared CLI output contract."""

import pytest
from fastapi.testclient import TestClient

from feedbax.application.project_writer import dump_text_project
from feedbax.bin.project_commands import PROJECT_COMMANDS
from feedbax.cli import main
from feedbax.contracts.jobs import (
    StudioJobSubmission,
    StudioProjectWindowMapProblem,
    StudioSavedTrainRuns,
)
from feedbax.contracts.objective_document import STORED_OBJECTIVE_DOCUMENT_SCHEMA_ID
from feedbax.contracts.project import Project
from feedbax.contracts.project_io import (
    ProjectConfiguration,
    read_text_project_snapshot,
)
from feedbax.contracts.studio_authoring import StudioAuthoringCatalog
from feedbax.contracts.set_document import SET_DOCUMENT_SCHEMA_ID
from feedbax.contracts.store_check import StoreCheckResult
from feedbax.contracts.task_document import STORED_TASK_DOCUMENT_SCHEMA_ID
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.web.app import create_app
from tests.support.managed_train_fixture import managed_train_project
from tests.support.store_damage import writable_store_object
from tests.support.host import admit, held_until_released, job_row, run_job, token, wait_until
from tests.test_text_project_plan_cli import _edit, _write_walkthrough


def _write_advice_project(root, documents, *, initialize_store=True):
    project = Project.model_validate({"documents": documents})
    files = dump_text_project(project, ProjectConfiguration(project="study", store="results"))
    for name, data in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    if initialize_store:
        ImmutableArtifactBlobProvider(root / "results").store_object(b"family advice check")


def test_check_advises_on_unrolled_families_without_writing_files(tmp_path, capsys):
    def document(schema, label, payload):
        return {"schema": schema, "label": label, "payload": payload}

    def entry(label, value, *, scale=1, target="gain"):
        return {
            "label": label,
            "task_variables": {"load": {"label": "Load", "type": {"base": "real"}, "value": value}},
            "differences": {f"/interior/components/{target}/fields/scale": {"value": scale}},
        }

    def trial(label, position, *, epoch="move"):
        return {"label": label, "variables": {"position": position}, "epoch_steps": {epoch: 5}}

    documents = {
        "cost": document(STORED_OBJECTIVE_DOCUMENT_SCHEMA_ID, "Cost", {"id": "cost", "terms": {
            "early": {"label": "Early", "measure": "x * 2", "weight": 1},
            "late": {"label": "Late", "measure": "x * 4"},
            "other_signal": {"measure": "y * 6"},
            "other_literal_type": {"measure": "x * 6.5"},
            "other_unit": {"measure": "x * 6 s"},
            "other_reduction": {"measure": "x * 6", "reduction": "last"},
            "draft": {"measure": "x + "},
            "family": {"label": "Movement", "measure": "x * k", "over": {"k": "1..10"}},
        }}),
        "bank": document(SET_DOCUMENT_SCHEMA_ID, "Bank", {"entries": {
            "low": entry("Low", "2 * bias"),
            "high": entry("High", "4 * bias"),
            "other_target": entry("Other target", "6 * bias", target="other"),
            "other_difference": entry("Other difference", "6 * bias", scale=2),
            "family": entry("Gain {k}", "k * bias") | {"over": {"k": "1..10"}},
        }}),
        "cohort": document(STORED_TASK_DOCUMENT_SCHEMA_ID, "Cohort", {
            "id": "cohort",
            "variables": {"position": {"label": "Position", "type": {"base": "real", "shape": [2]}, "value": "[0, 0]"}},
            "epochs": {name: {"label": name, "end": {"kind": "length", "length": "5 st"}}
                       for name in ("move", "hold")},
            "signals": {},
            "validation_trials": {
                "near": trial("Near", "[0, 1]"),
                "far": trial("Far", "[3, 5]"),
                "other_epoch": trial("Other epoch", "[4, 5]", epoch="hold"),
                "other_shape": trial("Other shape", "[4, 5, 6]"),
                "family": trial("Trial {k}", "[k, 0]") | {"over": {"k": "1..10"}},
            },
        }),
    }
    _write_advice_project(tmp_path, documents)
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    assert main(["check", str(tmp_path), "--json"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    result = StoreCheckResult.model_validate_json(captured.out)
    assert not result.findings and result.verified_objects == 1
    expected = {
        ("documents/cost.json", "/payload/terms", ("early", "late")),
        ("documents/bank.json", "/payload/entries", ("low", "high")),
        ("documents/cohort.json", "/payload/validation_trials", ("near", "far")),
    }
    assert {(item.address, item.path, item.members) for item in result.advice} == expected
    assert all(item.count == 2 and item.code == "unrolled_member_family" for item in result.advice)
    assert main(["check", str(tmp_path / "study.feedbax")]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert len(captured.out.splitlines()) == 4
    for item in result.advice:
        assert f"Advice {item.address}:{item.path} {item.detail}" in captured.out
    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


def test_check_locates_unrolled_family_in_a_fresh_based_project(tmp_path, capsys):
    schema = STORED_OBJECTIVE_DOCUMENT_SCHEMA_ID
    _write_advice_project(tmp_path, {
        "base": {"schema": schema, "label": "Base", "payload": {
            "id": "cost", "terms": {"first": {"measure": "x * 2"}},
        }},
        "second": {"schema": schema, "label": "Second", "base": {"document": "base"}, "differences": {
            "/terms/second": {"value": {"measure": "x * 4"}},
        }},
        "copy": {"schema": schema, "label": "Copy", "base": {"document": "second"}},
    }, initialize_store=False)

    assert main(["check", str(tmp_path), "--json"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    report = StoreCheckResult.model_validate_json(captured.out)
    assert report.verified_objects == report.verified_bytes == 0 and not report.findings
    (item,) = report.advice
    assert (item.address, item.path, item.members, item.count) == (
        "documents/second.json", "/differences/~1terms~1second", ("first", "second"), 2,
    )
    store = tmp_path / "results"
    assert not store.exists()
    assert main(["check", str(tmp_path), "--attempt", "missing", "--json"]) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and StudioProjectWindowMapProblem.model_validate_json(captured.err).code == "io_error"
    assert not store.exists()
    store.mkdir()
    assert main(["check", str(tmp_path), "--json"]) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and StudioProjectWindowMapProblem.model_validate_json(captured.err).code == "io_error"
    assert not tuple(store.iterdir())


def test_managed_commands_share_http_contracts_and_exit_codes(host, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FEEDBAX_ACTOR_LABEL", "environment actor")
    monkeypatch.setenv("FEEDBAX_HOST_ROOT", str(host.root))
    projects = tmp_path / "projects"
    walkthrough = projects / "reaching"
    _, files = _write_walkthrough(walkthrough)
    train = tmp_path / "external" / "managed checkout"
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
        held_until_released(release),
    )
    admit(host, token("gate"), gate)
    wait_until(lambda: job_row(host, gate.id, "running"))

    baseline = tmp_path / "review.json"
    declared_store = ImmutableArtifactBlobProvider(train / "results")
    payload = declared_store.store_object(b"integrity witness")
    cases = {
        "check": (StoreCheckResult, (str(declaration),)),
        "plan": (StudioSavedTrainRuns, (str(walkthrough), "--out", str(baseline))),
        "status": (StudioSavedTrainRuns, (str(walkthrough / "reaching.feedbax"),)),
        "run": (StudioJobSubmission, (str(declaration), "train", "--label", "explicit actor")),
        "describe": (StudioAuthoringCatalog, ()),
    }
    assert cases.keys() == PROJECT_COMMANDS.keys() - {"store"}

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

    def comparable_status(view):
        # Studio includes trial-condition previews for editing.
        return view.model_dump(exclude={"window_map": {"rows": {"__all__": {"evaluation_conditions"}}}})

    results = {}
    for name in cases:
        model, args = cases[name]
        with pytest.raises(SystemExit) as help_exit:
            main([name, "--help"])
        assert help_exit.value.code == 0
        help_text = capsys.readouterr().out
        nonzero_help = "1  Damaged objects" if name == "check" else "1  Review drift"
        assert all(text in help_text for text in ("--json", "0  Success", nonzero_help, "2  Invalid project"))
        _, problems = call([name, "--not-a-real-flag", "--json"], 2)
        (problem,) = problems
        assert problem.code == "invalid_arguments"
        assert problem.address == problem.path == "--not-a-real-flag"
        assert "suggest" not in problem.detail and "did you mean" not in problem.detail
        results[name], problems = call([name, *args, "--json"], 0, model)
        assert problems == []

    _, problems = call(["pla", "--json"], 2)
    assert [problem.code for problem in problems] == ["unknown_command"]
    assert results["check"].verified_bytes == payload.size_bytes
    assert results["check"].verified_objects == 1
    _, problems = call(["check", str(declaration), "--attempt", "absent", "--json"], 2)
    assert "No committed receipt" in problems[0].detail
    before_damage, problems = call(["check", str(declaration), "--json"], 0, StoreCheckResult)
    assert not problems
    object_path = declared_store.object_path(payload)
    with writable_store_object(object_path):
        original = object_path.read_bytes()
        altered = b"x" * len(original)
        try:
            object_path.write_bytes(altered)
            damaged, problems = call(["check", str(declaration), "--json"], 1, StoreCheckResult)
            assert damaged.verified_objects == before_damage.verified_objects - 1
            assert [problem.address for problem in problems] == [payload.object_id]
            assert object_path.read_bytes() == altered
        finally:
            object_path.write_bytes(original)

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
    assert [item.disposition for item in submission.items] == ["admitted"]
    app = create_app(projects_dir=projects)
    external_status, problems = call(["status", str(declaration), "--json"], 0, StudioSavedTrainRuns)
    assert not problems
    with TestClient(app) as client:
        opened, _ = app.state.project_folders.open_native_document(declaration)
        assert opened.snapshot.project_id == submission.project_id
        response = client.get(
            f"/api/projects/{walkthrough.name}/train-runs",
            params={"revision": read_text_project_snapshot(walkthrough).revision, "window_map": True},
        )
        assert response.status_code == 200
        studio_status = StudioSavedTrainRuns.model_validate(response.json())
        assert comparable_status(studio_status) == comparable_status(status)
        response = client.get(f"/api/submissions/{submission.id}")
        assert response.status_code == 200
        assert StudioJobSubmission.model_validate(response.json()) == submission
        response = client.get(
            f"/api/projects/{opened.snapshot.project_id}/train-runs",
            params={"revision": opened.snapshot.revision, "window_map": True},
        )
        assert response.status_code == 200
        assert comparable_status(StudioSavedTrainRuns.model_validate(response.json())) == comparable_status(external_status)
        (attempt,) = external_status.rows[0].attempts
        assert attempt.job_id == submission.items[0].job_id
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
    refused = [item for item in rejected.items if item.disposition == "rejected"]
    assert refused
    assert [(problem.code, problem.address) for problem in problems] == [
        ("admission_rejected", item.run_key) for item in refused
    ]

    _edit(
        walkthrough, files, "velocity-analysis",
        lambda doc: doc["operations"]["align"].update(operation="feedbax.unknown"),
    )
    separate_store = ImmutableArtifactBlobProvider(walkthrough / "results")
    separate_store.store_object(b"independent of Plan compilation")
    checked, problems = call(["check", str(walkthrough), "--json"], 0, StoreCheckResult)
    assert checked.verified_objects == 1 and problems == []
    project_errors = {}
    for name in cases.keys() - {"describe", "check"}:
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
