"""Compare the text CLI to a Plan compiled from the same Studio authoring."""

import json
from pathlib import Path

from feedbax.cli import main
from feedbax.contracts.project_io import (
    ProjectConfiguration,
    StudioProjectFile,
    dump_studio_project,
    dump_text_project,
    parse_studio_project,
)
from feedbax.contracts.project_plan import PlanRequests, compile_project
from feedbax.contracts.project_plan_io import load_plan_requests
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
    authored, files = _write_walkthrough(root)
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
    review, scope = load_plan_requests(files["plan.json"])
    studio = parse_studio_project(
        dump_studio_project(StudioProjectFile(project=authored, presentation={}))
    )
    api = compile_project(studio.project, operations=_operations())
    assert not api.diagnostics and scope is None
    assert review == PlanRequests.from_plan(api)
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
        root / "feedbax.project.json",
        root / "steps/plan.json",
        root / "results/plan.json",
    ):
        assert main(["plan", str(root), "--out", str(path)]) == 2
        assert "review file" in capsys.readouterr().err
    assert _files(root) == files
    assert not (root / "results").exists()
