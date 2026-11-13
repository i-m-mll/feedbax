"""Compare the text CLI to a Plan compiled from the same Studio authoring."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from feedbax.cli import main
from feedbax.contracts.base import sha256_bytes
from feedbax.contracts.project_io import (
    STUDIO_WRITE_LOCK_FILENAME,
    ProjectConfiguration,
    dump_text_project,
)
from feedbax.contracts.project_plan import PlanRequest, PlanRequests, RunKey, diff_plan_requests
from feedbax.contracts.project_plan_io import dump_plan_requests, load_plan_requests
from feedbax.contracts.request_identity import encode_request
from feedbax.contracts.request_specs import EvaluationRequestSpec
from feedbax.web.services.project_folder import ProjectFolderService
from tests.test_project_analyze_plan import _operations
from tests.test_project_report_plan import _report_project


def _write_walkthrough(root):
    project = _report_project()
    files = dump_text_project(project, ProjectConfiguration(project="reaching", store="results"))
    for relative, data in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return project, files


def _files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _edit(root, files, document, mutate):
    name = f"documents/{document}.json"
    payload = json.loads(files[name])
    mutate(payload["payload"])
    files[name] = (json.dumps(payload, indent=2) + "\n").encode()
    (root / name).write_bytes(files[name])


def test_cli_review_matches_studio_authoring_and_detects_report_drift(
    tmp_path, monkeypatch, capsys
):
    root = tmp_path / "reaching"
    _, files = _write_walkthrough(root)
    monkeypatch.chdir(root)
    assert main(["plan", "--out", "plan.json"]) == 0
    assert {line.split()[1].split(":")[0] for line in capsys.readouterr().out.splitlines()} == {
        "train",
        "evaluate",
        "analyze",
        "render",
        "report",
    }
    saved = root / "plan.json"
    files["plan.json"] = saved.read_bytes()
    assert _files(root) == files
    review, scope = load_plan_requests(files["plan.json"])
    folders = ProjectFolderService(tmp_path, operations=_operations())
    api = folders.read_plan(root.name, revision=None).plan
    assert not api.diagnostics and scope is None
    assert review == PlanRequests.from_plan(api)
    files[STUDIO_WRITE_LOCK_FILENAME] = b""
    assert _files(root) == files
    assert not (root / "results").exists()

    monkeypatch.chdir(root / "steps")
    assert main(["plan", "--check", "../plan.json"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == len(review.entries)
    assert all(line.split()[1] == "unchanged" for line in lines)
    _edit(
        root,
        files,
        "velocity-report",
        lambda document: document["blocks"][0].update(markdown="Edited interpretation."),
    )
    assert main(["plan", "--check", "../plan.json"]) == 1
    changes = [
        line for line in capsys.readouterr().out.splitlines() if line.split()[1] != "unchanged"
    ]
    (change,) = changes
    assert change.startswith("default/report/default changed(payload.blocks.0.markdown) report:")
    assert _files(root) == files
    assert not (root / "results").exists()


def test_saved_plan_keeps_figure_edges_and_reports_removed_requests(tmp_path, capsys):
    root = tmp_path / "reaching"
    _, files = _write_walkthrough(root)
    baseline = tmp_path / "plan.json"
    plan = ["plan", str(root / "steps"), "--variant", "default"]
    assert main([*plan, "--out", str(baseline)]) == 0
    capsys.readouterr()
    initial = baseline.read_bytes()
    _, scope = load_plan_requests(initial)
    assert scope == ("default",)
    _edit(root, files, "velocity-figure", lambda document: document.update(theme="plotly_dark"))
    assert main(["plan", str(root), "--check", str(baseline)]) == 1
    lines = capsys.readouterr().out.splitlines()
    assert any(
        "default/analyze/default/profile changed(payload.views.profile.figure.theme)" in line
        for line in lines
    )
    assert any(
        line.startswith("default/report/default changed(inputs.velocity_summary)")
        and line.endswith("upstream=default/analyze/default/profile")
        for line in lines
    )

    (root / "steps/report.json").unlink()
    del files["steps/report.json"]
    assert main([*plan, "--check", str(baseline)]) == 1
    assert "default/report/default removed report:" in capsys.readouterr().out
    assert baseline.read_bytes() == initial
    assert _files(root) == files
    assert not (root / "results").exists()


def test_cli_refuses_writes_to_the_project_inputs_or_store(tmp_path, capsys):
    root = tmp_path / "reaching"
    _, files = _write_walkthrough(root)
    for path in (
        root / "other.feedbax",
        root / "steps/plan.json",
        root / "results/plan.json",
    ):
        assert main(["plan", str(root), "--out", str(path)]) == 2
        assert "review file" in capsys.readouterr().err
    assert _files(root) == files
    assert not (root / "results").exists()


def test_cli_locates_the_exact_declaration_and_stops_at_ambiguity(tmp_path, capsys, monkeypatch):
    root = tmp_path / "reaching"
    _, files = _write_walkthrough(root)
    declaration, = (root / name for name in files if name.endswith(".feedbax"))
    child = root / "working"
    child.mkdir()
    monkeypatch.chdir(child)
    assert main(["plan"]) == 0
    expected = capsys.readouterr().out
    for selection in (root, declaration):
        assert main(["plan", str(selection)]) == 0
        assert capsys.readouterr().out == expected
    assert main(["plan", str(root / "missing.feedbax")]) == 2
    assert "not the folder's project declaration" in capsys.readouterr().err
    for name in ("one.feedbax", "two.feedbax"):
        (child / name).write_bytes(declaration.read_bytes())
    assert main(["plan"]) == 2
    assert "one.feedbax, two.feedbax" in capsys.readouterr().err
    (child / "two.feedbax").unlink()
    (child / "one.feedbax").rename(child / "bad name.feedbax")
    assert main(["plan"]) == 2
    assert "invalid project name" in capsys.readouterr().err
    declaration.unlink()
    assert main(["plan", str(root)]) == 2
    assert "no *.feedbax" in capsys.readouterr().err


def test_cli_and_managed_plan_share_a_project_relative_figure_pin(tmp_path, capsys):
    root = tmp_path / "reaching"
    project, files = _write_walkthrough(root)
    baseline = tmp_path / "plan.json"
    assert main(["plan", str(root), "--out", str(baseline)]) == 0
    capsys.readouterr()
    shared = tmp_path / "shared-figure.json"
    raw = json.dumps({"payload": project.documents["velocity-figure"].payload}).encode()
    shared.write_bytes(raw)
    digest = sha256_bytes(raw)

    def pin(document):
        document.clear()
        document.update(path="../shared-figure.json", sha256=digest, pointer="/payload")

    _edit(root, files, "velocity-figure", pin)
    assert main(["plan", str(root / "steps"), "--check", str(baseline)]) == 0
    assert "default/analyze/default/profile unchanged render:" in capsys.readouterr().out
    assert main(["plan", str(root), "--out", str(shared)]) == 2
    assert "review file overlaps an authored project file" in capsys.readouterr().err
    assert shared.read_bytes() == raw
    review, _ = load_plan_requests(baseline.read_bytes())
    folders = ProjectFolderService(tmp_path, operations=_operations())
    settled = folders.read_plan("reaching", revision=None)
    assert not settled.plan.diagnostics
    assert PlanRequests.from_plan(settled.plan) == review

    shared.write_bytes(b"{")
    assert main(["plan", str(root), "--check", str(baseline)]) == 2
    captured = capsys.readouterr()
    assert "default/analyze/default/align unchanged analyze:" in captured.out
    assert "documents/velocity-figure.json:payload.sha256" in captured.err
    assert f"[bad_external_content]: expected {digest}, got {sha256_bytes(b'{')}" in captured.err
    refused = folders.read_plan("reaching", revision=settled.revision)
    problem = next(problem for problem in refused.plan.diagnostics if problem.element == "profile")
    assert (problem.document, problem.document_path) == ("velocity-figure", "payload.sha256")
    assert any(key.name == "align" for key in refused.plan.elements)
    assert not any(key.name == "profile" for key in refused.plan.elements)
    assert not any(run.kind == "report" for run in refused.plan.runs)
    assert {name: data for name, data in _files(root).items() if name != STUDIO_WRITE_LOCK_FILENAME} == files


def test_portable_review_factors_nested_fields_and_keeps_null_and_arrays():
    first = encode_request(EvaluationRequestSpec(
        evaluation_type="rollout", params={"nested": {"shared": [None, True], "choice": None}},
    ), {})
    second = encode_request(EvaluationRequestSpec(
        evaluation_type="rollout", params={"nested": {"shared": [None, True], "choice": [1, None]}},
    ), {})
    key = RunKey("default", "evaluate", "default")
    sibling = RunKey("default", "evaluate", "sibling")
    original = PlanRequests(
        {current: PlanRequest(current, item.id, {}) for current, item in ((key, first), (sibling, second))},
        {item.id: item.data for item in (first, second)},
        {},
    )
    encoded = dump_plan_requests(original, ("default",))
    data = json.loads(encoded)
    group, = data["request_groups"]
    assert data["schema_version"] == "feedbax.project-plan.v2"
    assert group["common"]["spec"]["params"]["nested"] == {"shared": [None, True]}
    assert group["requests"][first.id]["spec"]["params"]["nested"] == {"choice": None}
    restored, scope = load_plan_requests(encoded)
    assert restored == original
    assert scope == ("default",)
    assert list(restored.entries) == list(original.entries)
    assert dump_plan_requests(restored, scope) == encoded
    updated = PlanRequests({key: PlanRequest(key, second.id, {})}, {second.id: second.data}, {})
    assert diff_plan_requests(restored, updated) == diff_plan_requests(original, updated)

    shared_override = deepcopy(data)
    residual = shared_override["request_groups"][0]["requests"][first.id]
    residual["spec"]["params"]["nested"]["shared"] = [None, True]
    with pytest.raises(ValueError, match="repeats shared field"):
        load_plan_requests(json.dumps(shared_override).encode())

    changed_identity = deepcopy(data)
    residual = changed_identity["request_groups"][0]["requests"][first.id]
    residual["spec"]["params"]["nested"]["choice"] = False
    with pytest.raises(ValueError, match="does not match"):
        load_plan_requests(json.dumps(changed_identity).encode())

    old = {"schema_version": "feedbax.project-plan.v1", "entries": [], "requests": {}, "documents": {}}
    with pytest.raises(ValueError, match="schema_version"):
        load_plan_requests(json.dumps(old).encode())
