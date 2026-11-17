"""Compare the text CLI to a Plan compiled from the same Studio authoring."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from feedbax.cli import main
from feedbax.contracts.base import sha256_bytes
from feedbax.contracts.project import Project
from feedbax.contracts.project_io import (
    STUDIO_WRITE_LOCK_FILENAME,
    ProjectConfiguration,
    dump_text_project,
)
from feedbax.contracts.project_plan import PlanRequest, PlanRequests, RunKey, compile_project, diff_plan_requests
from feedbax.contracts.project_plan_io import dump_plan_requests, load_plan_requests
from feedbax.contracts.request_identity import encode_request
from feedbax.contracts.request_specs import EvaluationRequestSpec
from feedbax.web.services.project_folder import ProjectFolderService
from tests.support.managed_train_fixture import managed_evaluation_step, managed_train_project
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


def test_cli_review_keeps_managed_source_addresses(tmp_path, capsys):
    root = tmp_path / "mixed"
    base = managed_train_project().model_dump(mode="json", by_alias=True, exclude_unset=True)
    steps = base["steps"]
    train = steps["train"]
    train["base"]["inline"].update(seed=0, on_nan="raise")
    train["sweep"] = {
        "axes": {"seed": {"values": {
            tag: {"patches": {"seed": {"value": value}}} for tag, value in (("a", 1), ("b", 2))
        }}},
        "exceptions": {"restore": {
            "where": {"rows": ["seed=a"]},
            "patches": {"on_nan": {"value": "halt_restore_checkpoint"}},
        }},
    }
    steps["child"] = {"kind": "train", "base": {"step": "train"}}
    steps["evaluate"] = managed_evaluation_step("child")
    steps["evaluate"]["inputs"]["subject"]["row"] = "seed=a"
    variant_path = "steps.train.base.inline.training_config.batch_size"
    base["sweep"] = {"rows": {"control": {}, "variant": {"patches": {
        variant_path: {"value": 8},
    }}}}

    def save(raw):
        for relative, data in dump_text_project(
            Project.model_validate(raw), ProjectConfiguration(project="mixed", store="results"),
        ).items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)

    def source(kind, name, path, **coordinates):
        return {"kind": kind, "name": name, "path": path, **coordinates}

    mass = ("documents", "point-mass", "payload", "interior", "components", "plant", "fields", "mass")
    batch_size = ("steps", "train", "base", "inline", "training_config", "batch_size")
    swept = ("steps", "train", "sweep", "axes", "seed", "values", "b", "patches", "seed", "value")
    exception = ("steps", "train", "sweep", "exceptions", "restore", "patches", "on_nan", "value")
    variant = ("sweep", "rows", "variant", "patches", variant_path, "value")
    subject = ("steps", "evaluate", "inputs", "subject", "row")
    model_source = source("document", "point-mass", "/interior/components/plant/fields/mass")
    step_source = source("step", "train", "/base/inline/training_config/batch_size", step="train")
    sweep_source = source(
        "sweep_value", "b", "/sweep/axes/seed/values/b/patches/seed/value",
        step="train", axis="seed", value="b",
    )
    exception_source = source("exception", "restore", "/sweep/exceptions/restore/patches/on_nan/value", step="train")
    variant_source = source("variant", "variant", "/steps/train/base/inline/training_config/batch_size", variant="variant")
    def upstream(row):
        return source("upstream", "child", "", step="child", variant="control", row=row)

    cases = (
        (mass, 3, RunKey("control", "child", "seed=a"), model_source, model_source),
        (batch_size, 6, RunKey("control", "child", "seed=a"), step_source, step_source),
        (swept, 7, RunKey("control", "child", "seed=b"), sweep_source, sweep_source),
        (exception, "raise", RunKey("control", "child", "seed=a"), exception_source, exception_source),
        (variant, 12, RunKey("variant", "child", "seed=a"), variant_source, variant_source),
        (subject, "seed=b", RunKey("control", "evaluate", "default"), upstream("seed=a"), upstream("seed=b")),
    )
    save(base)
    baseline = tmp_path / "baseline.json"
    assert main(["plan", str(root), "--out", str(baseline)]) == 0
    capsys.readouterr()
    archived, scope = load_plan_requests(baseline.read_bytes())
    assert dump_plan_requests(archived, scope) == baseline.read_bytes()
    assert replace(archived, source_index=None) == PlanRequests.from_plan(compile_project(Project.model_validate(base)))
    for path, value, key, before, after in cases:
        edited = deepcopy(base)
        target = edited
        for part in path[:-1]:
            target = target[part]
        target[path[-1]] = value
        save(edited)
        assert main(["plan", str(root), "--check", str(baseline), "--json"]) == 1
        rows = json.loads(capsys.readouterr().out)["review"]
        row = next(row for row in rows if row["key"] == {"variant": key.variant, "step": key.step, "row": key.row})
        assert row["state"] == "changed"
        assert (row["before_source"], row["after_source"]) == (before, after)
    assert not (root / "results").exists()


def test_saved_source_snapshot_does_not_follow_project_mutations():
    project = managed_train_project()
    original = PlanRequests.from_plan(compile_project(project, capture_sources=True))
    original_bytes = dump_plan_requests(original, None)
    project.documents["point-mass"].payload["step_seconds"] = 0.02
    assert dump_plan_requests(original, None) == original_bytes


def test_cli_review_locates_derived_trials_and_addressed_model_field(tmp_path, capsys):
    from feedbax.contracts.model_address import ModelAddress
    from feedbax.contracts.task_document import TaskDocument

    root = tmp_path / "derived"
    project = managed_train_project().model_dump(mode="json", by_alias=True, exclude_unset=True)
    first, second, *_ = TaskDocument.model_validate(
        project["documents"]["simple-reaches-task"]["payload"]
    ).condition_ids
    evaluate = managed_evaluation_step("train")
    params = evaluate["base"]["inline"]["params"]
    params["trial_set"] = {"kind": "conditions", "ids": [first, second]}
    project["steps"]["evaluate"] = evaluate
    field = ModelAddress.model_validate({
        "document": "point-mass", "composites": [],
        "element": {"kind": "field", "parent": {"kind": "component", "id": "plant"}, "path": ["fields", "mass"]},
    })
    edit = {"op": "set_field", "address": field.model_dump(mode="json"), "value": 2.0}
    project["steps"]["train"]["model_attachments"] = {"physics": [edit]}

    def save():
        for relative, data in dump_text_project(
            Project.model_validate(project), ProjectConfiguration(project="derived", store="results"),
        ).items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)

    save()
    baseline = tmp_path / "baseline.json"
    assert main(["plan", str(root), "--out", str(baseline)]) == 0
    capsys.readouterr()
    params["trial_set"]["ids"] = [second, first]
    edit["value"] = 3.0
    save()
    assert main(["plan", str(root), "--check", str(baseline), "--json"]) == 1
    review = {item["key"]["step"]: item for item in json.loads(capsys.readouterr().out)["review"]}
    train = review["train"]
    assert train["path"].endswith(".plant.fields.mass")
    assert train["before_source"] == train["after_source"] == {
        "kind": "step", "name": "train", "step": "train", "path": "/model_attachments/physics/0/value",
    }
    evaluation = review["evaluate"]
    assert evaluation["path"] == "payload.params.trials.0"
    assert evaluation["before_source"] == evaluation["after_source"] == {
        "kind": "step", "name": "evaluate", "step": "evaluate", "path": "/base/inline/params/trial_set/ids/0",
    }


def test_cli_review_uses_saved_pinned_source_without_reopening_it(tmp_path, capsys):
    root = tmp_path / "reaching"
    project, files = _write_walkthrough(root)
    shared = tmp_path / "shared-figure.json"
    raw = json.dumps({"payload": project.documents["velocity-figure"].payload}, indent=2).encode()
    shared.write_bytes(raw)

    def pin(document):
        document.clear()
        document.update(path="../shared-figure.json", sha256=sha256_bytes(raw), pointer="/payload")

    _edit(root, files, "velocity-figure", pin)
    baseline = tmp_path / "baseline.json"
    assert main(["plan", str(root), "--out", str(baseline)]) == 0
    capsys.readouterr()
    shared.unlink()

    def inline(document):
        document.clear()
        document.update(project.documents["velocity-figure"].payload, theme="plotly_dark")

    _edit(root, files, "velocity-figure", inline)
    assert main(["plan", str(root), "--check", str(baseline), "--json"]) == 1
    reviews = json.loads(capsys.readouterr().out)["review"]
    review = next(item for item in reviews if item.get("element") == "profile")
    assert review["before_source"] == {
        "kind": "document", "name": "../shared-figure.json", "path": "/payload/theme",
    }
    assert review["after_source"] == {"kind": "document", "name": "velocity-figure", "path": "/theme"}
    truncated = json.loads(baseline.read_bytes())
    truncated["sources"]["externals"].clear()
    baseline.write_text(json.dumps(truncated))
    assert main(["plan", str(root), "--check", str(baseline)]) == 2
    assert str(baseline) in capsys.readouterr().err


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
    assert replace(review, source_index=None) == PlanRequests.from_plan(api)
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
    assert "before=document:velocity-report/blocks/0/markdown" in change
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
    assert PlanRequests.from_plan(settled.plan) == replace(review, source_index=None)

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


@pytest.mark.parametrize(("before", "after"), (
    ({"a.b/x": None}, {}),
    ({"a.b/x": True}, {"a.b/x": 1}),
    ({}, {"a.b/x": {"nested": 1}}),
))
def test_review_attributes_opaque_parameter_edits(before, after):
    key = RunKey("default", "evaluate", "default")
    authored = managed_train_project()

    def review(params):
        project = Project.model_validate({
            "steps": {
                "train": authored.steps["train"],
                "evaluate": {
                    "kind": "evaluate",
                    "base": {"inline": {"evaluation_type": "custom", "params": params}},
                    "inputs": {"subject": {"step": "train", "checkpoint": "last"}},
                },
            },
            "documents": authored.documents,
        })
        plan = compile_project(project, capture_sources=True)
        assert not plan.diagnostics
        return load_plan_requests(dump_plan_requests(PlanRequests.from_plan(plan), None))[0]

    result = diff_plan_requests(review(before), review(after))[key]
    assert result.path == "payload.params.a.b/x"
    for params, source in ((before, result.before_source), (after, result.after_source)):
        assert (source.path if source is not None else None) == (
            "/base/inline/params/a.b~1x" if params else None
        )
