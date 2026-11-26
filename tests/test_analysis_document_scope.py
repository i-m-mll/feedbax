"""Store Analyze edits of a shared analysis document for the chosen steps, as every document is."""

import pytest

from feedbax.contracts.operation import DatasetPort
from feedbax.contracts.project import BasedDocument, Step
from feedbax.planner.project_plan import ElementChange, ElementKey, PlannedAnalyzeRun, RunKey, compile_project, diff_analyze_elements
from feedbax.contracts.studio_analyze import StudioAnalyzeEdit, StudioAnalyzeRemove, StudioAnalyzeReset
from feedbax.web.services.analyze_authoring import analyze_edit
from feedbax.web.services.trial_scope import changed_from_base, preview_stored
from tests.test_project_analyze_plan import _operations
from tests.test_project_report_plan import _report_project


_ONE = RunKey("default", "analyze", "default")
_TWO = RunKey("default", "other", "default")
_BOUNDS = "/operations/summarize/params/bounds_scale"
_CHANGED = ["/operations", "/operations/summarize", "/operations/summarize/params", _BOUNDS]


def _shared():
    project = _report_project()
    project.steps["other"] = project.steps["analyze"].model_copy(deep=True)
    project.steps["other-report"] = Step.model_validate(
        project.steps["report"].model_dump(mode="json", exclude_unset=True)
        | {"inputs": {"velocity_summary": {"step": "other", "element": "profile", "product": "figure"}}}
    )
    return project


def _compile(project, operations):
    return compile_project(project, operations=operations)


def _states(operations):
    port = operations.resolve("feedbax.aligned_velocity").value.definition.inputs["states"]
    return DatasetPort(schema_ids=(port.schema_ids[0],))


def _analyze(plan, key) -> PlannedAnalyzeRun:
    return next(row for row in plan.rows if row.key == key and isinstance(row, PlannedAnalyzeRun))


def _store(project, key, document, change, chosen, operations):
    stored, _ = analyze_edit(
        project, _compile(project, operations), key, document, change, frozenset(chosen), operations, _states(operations),
    )
    return preview_stored(project, stored), stored


def _edit(path, value):
    return StudioAnalyzeEdit(kind="edit", edits=[{"op": "replace", "path": path, "value": value}])


def test_a_narrowed_parameter_edit_makes_a_based_document_for_the_lit_step():
    operations = _operations()
    project = _shared()
    before = _compile(project, operations)
    edited, stored = _store(project, _ONE, "velocity-analysis", _edit(_BOUNDS, 2.0), {"analyze"}, operations)
    (created,) = stored.created
    assert isinstance(created.document, BasedDocument) and created.document.base.document == "velocity-analysis"
    assert edited.steps["analyze"].base.document == created.id
    assert edited.steps["other"].base.document == "velocity-analysis"
    assert edited.documents["velocity-analysis"] == project.documents["velocity-analysis"]
    after = _compile(edited, operations)
    assert not after.diagnostics
    changes = diff_analyze_elements(before, after)
    assert {key.name: change.state for key, change in changes.items() if key.run == _ONE} == {
        "align": "unchanged", "summarize": "changed", "profile": "changed",
    }
    assert all(change == ElementChange("unchanged") for key, change in changes.items() if key.run == _TWO)
    assert {run.key: run.request_id for run in after.runs if run.key.step in {"other", "other-report"}} == {
        run.key: run.request_id for run in before.runs if run.key.step in {"other", "other-report"}
    }
    assert changed_from_base(edited, created.id) == _CHANGED

    reports = {run.key.step: run for run in after.runs if run.kind == "report"}
    assert reports["report"].inputs["velocity_summary"].element == ElementKey(_ONE, "profile")
    assert reports["other-report"].inputs["velocity_summary"].element == ElementKey(_TWO, "profile")
    assert reports["report"].request_id != reports["other-report"].request_id

    added = {"label": "Operation", "operation": "feedbax.summarize_velocity", "params": {"ddof": 0, "bounds_scale": 1.0},
             "inputs": {"velocity": {"element": "align", "product": "velocity"}}}
    grown, stored = _store(
        edited, _TWO, "velocity-analysis", _edit("/operations/extra", added), {"analyze", "other"}, operations,
    )
    assert not stored.created
    plan = _compile(grown, operations)
    assert _analyze(plan, _ONE).requests("extra") and _analyze(plan, _TWO).requests("extra")
    assert changed_from_base(grown, created.id) == _CHANGED

    reset, stored = _store(grown, _ONE, created.id, StudioAnalyzeReset(kind="reset", paths=[_BOUNDS]), {"analyze"}, operations)
    assert not stored.created and changed_from_base(reset, created.id) == []
    assert _analyze(_compile(reset, operations), _ONE).requests("summarize")[0].request_id == (
        _analyze(plan, _TWO).requests("summarize")[0].request_id
    )


def test_removing_an_operation_unbinds_its_readers_and_a_foreign_step_is_refused():
    operations = _operations()
    project = _shared()
    removed, stored = _store(
        project, _ONE, "velocity-analysis", StudioAnalyzeRemove(kind="remove", element="summarize"), {"analyze"}, operations,
    )
    (created,) = stored.created
    payload = removed.documents[created.id]
    assert payload.differences  # stored as the difference from its base
    plan = _compile(removed, operations)
    assert {key.name for key in _analyze(plan, _ONE).elements} == {"align"}
    assert {key.name for key in _analyze(plan, _TWO).elements} == {"align", "summarize", "profile"}
    with pytest.raises(ValueError, match="chosen steps"):
        _store(project, _ONE, "velocity-analysis", _edit(_BOUNDS, 3.0), {"analyze", "report"}, operations)


def test_a_swept_analyze_step_takes_an_edit_of_its_document():
    operations = _operations()
    project = _shared()
    project.steps["analyze"] = Step.model_validate(
        project.steps["analyze"].model_dump(mode="json", exclude_unset=True) | {"sweep": {"axes": {"scale": {"values": {
            name: {"patches": {"operations.summarize.params.bounds_scale": {"value": value}}}
            for name, value in (("one", 1), ("two", 2))
        }}}}}
    )
    project.steps.pop("report")
    plan = _compile(project, operations)
    key = next(row.key for row in plan.rows if row.key.step == "analyze")
    stored, _ = analyze_edit(
        project, plan, key, "velocity-analysis", _edit("/operations/summarize/params/ddof", 0), frozenset({"analyze"}),
        operations, _states(operations),
    )
    edited = _compile(preview_stored(project, stored), operations)
    assert not edited.diagnostics
    rows = [row for row in edited.rows if row.key.step == "analyze"]
    assert len(rows) == 2 and all(isinstance(row, PlannedAnalyzeRun) for row in rows)
