"""Authored patches may enter a declared expression but no ordinary value."""

from copy import deepcopy

import pytest

from feedbax.contracts.manifest import OverridePatch
from feedbax.contracts.matrix_core import apply_stored_patch_copy, prepare_stored_patch
from feedbax.contracts.project import Project
from feedbax.contracts.project_resolution import ProjectDiagnostic, ResolvedStep, ResolutionCode, compose_document, compose_step
from feedbax.contracts.project_storage import stored_expression_site, stored_step_expression_site
from feedbax.contracts.project_sweep import expand_sweep
from feedbax.contracts.task_document import STORED_TASK_DOCUMENT_SCHEMA_ID, StoredTaskDocument
from feedbax.planner.project_plan import PlanSourceIndex, compile_project, project_variants, resolve_authored_evaluation_task
from tests.support.managed_train_fixture import managed_train_project


def _task(value="U(1 m, 4 m)"):
    return {
        "id": "task",
        "variables": {"goal": {"label": "Goal", "type": {"base": "real"}, "value": value}},
        "epochs": {"prep": {"label": "Prep", "end": {"kind": "length", "length": "10 st"}}},
        "signals": {},
    }


def _document(payload):
    return {"schema": STORED_TASK_DOCUMENT_SCHEMA_ID, "label": "Task", "payload": payload}


_STEP_PATH = "trial_documents.task.variables.goal.value.args.1.value"
_EXPR_PATH = "/trial_documents/task/variables/goal/value"


def test_raw_patch_only_opens_declarations_and_prints_after_editing_nodes():
    root = {
        "call": "mix(1 m, gain=2 m)",
        "time": "10 st",
        "caption": "mix(1 m, gain=2 m)",
        "matrix": [[1, 2]],
        "fake": {"kind": "call", "function": "mix", "args": [{"kind": "literal", "value": 1}]},
    }

    def site(parts):
        if parts == ("call",):
            return "expression"
        if parts == ("time",):
            return "time"
        return None

    physical, origin = prepare_stored_patch(root, OverridePatch(path="call.kwargs.gain.value", value=3), expression_site=site)
    assert (physical.path, physical.value) == ("call", "mix(1 m, gain=3 m)")
    assert origin is not None
    assert origin.node.path == "kwargs.gain.value"
    changed = apply_stored_patch_copy(root, OverridePatch(path="call.args.0", value="5 m"), expression_site=site)
    assert changed["call"] == "mix(5 m, gain=2 m)"
    assert root["call"] == "mix(1 m, gain=2 m)"
    changed = apply_stored_patch_copy(root, OverridePatch(path="time.value", value=14), expression_site=site)
    assert changed["time"] == "14 st"
    with pytest.raises(ValueError, match="missing segment 'kwargs'"):
        apply_stored_patch_copy(root, OverridePatch(path="caption.kwargs.gain.value", value=3), expression_site=site)
    for path in ("matrix.0.0", "fake.args.0.value"):
        with pytest.raises(ValueError, match="cannot traverse an array"):
            apply_stored_patch_copy(root, OverridePatch(path=path, value=3), expression_site=site)
    with pytest.raises(ValueError, match="non-negative decimal index"):
        apply_stored_patch_copy(root, OverridePatch(path="call.args.00.value", value=3), expression_site=site)


@pytest.mark.parametrize("named", [True, False], ids=["named-task", "inline-task"])
@pytest.mark.parametrize("axis", [{"values": [5, 7]}, {"range": {"start": 5, "stop": 7, "count": 2}}], ids=["list", "range"])
def test_axis_patches_keep_units_and_node_origins(named, axis):
    task = _task()
    project = Project.model_validate({
        "documents": {"task": _document(task)} if named else {},
        "steps": {"fit": {
            "label": "Fit", "kind": "train",
            "base": {"inline": {"trial_documents": {"task": {"document": "task"} if named else task}}},
            "sweep": {"axes": {"goal": {"path": _STEP_PATH, **axis}}},
        }},
    })
    template = compose_step(project, "fit", track_sources=True)
    assert not isinstance(template, ProjectDiagnostic)
    rows = expand_sweep(template.sweep)
    assert len(rows) == 2
    for row, value in zip(rows, (5, 7), strict=True):
        result = template.resolve(row, include_sources=True)
        assert isinstance(result, ResolvedStep)
        assert result.payload["trial_documents"]["task"]["variables"]["goal"]["value"] == f"U(1 m, {value} m)"
        assert result.sources is not None
        assert result.sources[_EXPR_PATH + "/args/1/value"] == template.sweep_source(row, _STEP_PATH)
        field_origin = result.sources[_EXPR_PATH]
        assert field_origin.kind == ("document" if named else "step")
        assert result.sources[_EXPR_PATH + "/args/0/value"] == field_origin
        assert result.sources[_EXPR_PATH + "/args/1/unit"] == field_origin


def test_malformed_stored_text_is_a_problem_at_the_swept_field():
    project = Project.model_validate({
        "documents": {"task": _document(_task("uniform("))},
        "steps": {"fit": {
            "label": "Fit", "kind": "train",
            "base": {"inline": {"trial_documents": {"task": {"document": "task"}}}},
            "sweep": {"axes": {"goal": {"path": _STEP_PATH, "values": [5]}}},
        }},
    })
    template = compose_step(project, "fit", track_sources=True)
    assert not isinstance(template, ProjectDiagnostic)
    (row,) = expand_sweep(template.sweep)
    problem = template.resolve(row, include_sources=True)
    assert isinstance(problem, ProjectDiagnostic)
    assert problem.code is ResolutionCode.INVALID_SWEEP
    assert problem.document == "task"
    assert problem.path == "variables.goal.value.args.1.value"
    assert "Expected a field expression" in problem.detail
    assert problem.authored_file == "steps/fit.json"


def test_authored_evaluate_and_project_variant_paths_change_only_the_literal_number():
    original = _task()
    authored = StoredTaskDocument.model_validate(original)
    changed = resolve_authored_evaluation_task(authored, (
        OverridePatch(path="task.epochs.prep.end.length.value", value=14),
    ))
    assert changed.model_dump(mode="json")["epochs"]["prep"]["end"]["length"] == "14 st"
    assert authored.model_dump(mode="json")["epochs"]["prep"]["end"]["length"] == "10 st"

    axis_path = "documents.derived.payload.variables.goal.value.args.1.value"
    project = Project.model_validate({
        "documents": {
            "task": _document(deepcopy(original)),
            "derived": {"schema": STORED_TASK_DOCUMENT_SCHEMA_ID, "label": "Derived", "base": {"document": "task"}},
        },
        "sweep": {"axes": {"goal": {"path": axis_path, "values": [8]}}},
    })
    (variant,) = [value for value in project_variants(project).values() if value is not project]
    derived = compose_document(variant.documents.get, "derived")
    assert not isinstance(derived, ProjectDiagnostic)
    assert derived.payload["variables"]["goal"]["value"] == "U(1 m, 8 m)"
    assert variant.documents["task"].payload["variables"]["goal"]["value"] == "U(1 m, 4 m)"
    assert stored_expression_site(STORED_TASK_DOCUMENT_SCHEMA_ID, ("variables", "goal", "label")) is None
    assert stored_step_expression_site("synthesize", ("task", "epochs", "prep", "end", "length")) == "expression"



def test_managed_plan_keeps_origins_from_base_and_derived_variant_node_writes():
    raw = managed_train_project().model_dump(mode="json", by_alias=True, exclude_unset=True)
    held = raw["steps"]["train"]["base"]["inline"]["trial_documents"]["task"]
    base = held["document"]
    raw["documents"]["derived-task"] = {
        "schema": STORED_TASK_DOCUMENT_SCHEMA_ID, "label": "Derived Task", "base": {"document": base},
    }
    held["document"] = "derived-task"
    prefix = "variables.start.value.args.0.args"
    low = f"documents.{base}.payload.{prefix}.0.value"
    high = f"documents.derived-task.payload.{prefix}.1.value"
    raw["sweep"] = {"axes": {
        "low": {"path": low, "values": [-0.45]},
        "high": {"path": high, "values": [0.65]},
    }}
    plan = compile_project(Project.model_validate(raw), capture_sources=True)
    assert not plan.diagnostics
    (run,) = (run for run in plan.runs if run.key.variant != "default")
    assert not run.problems
    assert plan.source_snapshot is not None
    index = PlanSourceIndex(plan.source_snapshot)

    def origin(path):
        return index.source(run.key, ("payload", "trial_documents", "task", *path.split(".")))

    assert (origin(prefix + ".0.value").kind, origin(prefix + ".0.value").path) == (
        "variant", "/" + low.replace(".", "/"),
    )
    assert (origin(prefix + ".1.value").kind, origin(prefix + ".1.value").path) == (
        "variant", "/" + high.replace(".", "/"),
    )
    unchanged = origin("variables.start.value.args.1.args.0.value")
    assert (unchanged.kind, unchanged.name, unchanged.path) == ("document", base, "/variables/start/value")
