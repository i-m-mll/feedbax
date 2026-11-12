"""Bind an authored Report to saved figure elements through the pure Plan."""

from copy import deepcopy
import json
from pathlib import Path

from feedbax.contracts.figure_document import PlotlyFigurePort
from feedbax.contracts.project import Document, Project, Step
from feedbax.contracts.project_inputs import StepReference
from feedbax.contracts.project_plan import (
    ElementChange,
    ElementKey,
    ProjectPlanMemo,
    RunKey,
    compile_project,
    diff_analyze_elements,
    diff_plan_requests,
)
from feedbax.contracts.request_identity import decode_request
from feedbax.contracts.request_refs import ProductRef
from tests.test_project_analyze_plan import _operations, _project


_FIXTURES = Path(__file__).resolve().parents[1] / "web/e2e/studio/fixtures"
_REPORT = RunKey("default", "report", "default")
_ANALYZE = RunKey("default", "analyze", "default")
_PROFILE = ElementKey(_ANALYZE, "profile")


def _report_project() -> Project:
    project = _project()
    project.documents["velocity-report"] = Document.model_validate(
        {
            "schema": "feedbax.report_document.v1",
            "payload": json.loads((_FIXTURES / "velocity-summary.report.json").read_bytes()),
        }
    )
    project.steps["report"] = Step.model_validate_json(
        (_FIXTURES / "velocity-report-step.json").read_bytes()
    )
    return project


def _report_run(plan):
    return next(run for run in plan.runs if run.key == _REPORT)


def _compile(project, operations, memo=None):
    return compile_project(project, operations=operations, source_runs=(_REPORT,), memo=memo)


def test_report_binds_the_typed_figure_and_preserves_other_requests():
    project = _report_project()
    operations = _operations()
    upstream = deepcopy(project)
    upstream.steps.pop("report")
    original = compile_project(upstream, operations=operations)
    plan = _compile(project, operations)
    assert not plan.diagnostics
    assert len(plan.requests) == len(original.requests) + 1
    assert all(plan.requests[key] == raw for key, raw in original.requests.items())
    assert diff_analyze_elements(original, plan) == {
        key: ElementChange("unchanged") for key in original.elements
    }
    figure = plan.elements[_PROFILE]
    assert figure.outputs == {"figure": PlotlyFigurePort()}
    assert figure.datasets == {}
    report = _report_run(plan)
    request = decode_request(plan.requests[report.request_id], plan.documents.__getitem__)
    assert request.kind == "report" and request.spec.report_type is None
    assert request.spec.document.completeness == "complete"
    assert request.spec.document.figure_roles == ("velocity_summary",)
    assert request.inputs["velocity_summary"][0].reference == ProductRef(
        request_id=figure.request_id, role="figure"
    )
    assert report.inputs["velocity_summary"].element == _PROFILE
    assert (
        plan.sources[_REPORT]["/inputs/velocity_summary"].path
        == "/elements/profile/products/figure"
    )


def test_report_text_and_upstream_figure_edits_have_distinct_attribution():
    project = _report_project()
    operations = _operations()
    memo = ProjectPlanMemo()
    initial = _compile(project, operations, memo)
    project.documents["velocity-report"].payload["blocks"][0]["markdown"] += (
        "\n\nThe reset sample is included."
    )
    text_edit = _compile(project, operations, memo)
    assert text_edit == _compile(project, operations)
    assert {
        key: change
        for key, change in diff_plan_requests(initial, text_edit).items()
        if change.state != "unchanged"
    } == {_REPORT: ElementChange("changed", "payload.blocks.0.markdown")}

    project.documents["velocity-figure"].payload["theme"] = "plotly_dark"
    figure_edit = _compile(project, operations, memo)
    assert figure_edit == _compile(project, operations)
    assert {
        key: change
        for key, change in diff_plan_requests(text_edit, figure_edit).items()
        if change.state != "unchanged"
    } == {
        _PROFILE: ElementChange("changed", "payload.views.profile.figure.theme"),
        _REPORT: ElementChange("changed", "inputs.velocity_summary", _PROFILE),
    }


def test_second_figure_edit_and_analyze_sibling_diagnostic_keep_precise_authority():
    project = _report_project()
    operations = _operations()
    memo = ProjectPlanMemo()
    analysis = project.documents["velocity-analysis"].payload
    alternate = deepcopy(analysis["views"]["profile"])
    alternate["figure"] = {"document": "alternate-figure"}
    analysis["views"]["alternate"] = alternate
    project.documents["alternate-figure"] = deepcopy(project.documents["velocity-figure"])
    unrelated = deepcopy(analysis["operations"]["align"])
    unrelated["params"]["target"] = 3
    analysis["operations"]["unrelated"] = unrelated
    project.documents["velocity-report"].payload["blocks"].append({"figure": "other"})
    project.steps["report"].inputs["other"] = StepReference(
        step="analyze", row="default", element="alternate", product="figure"
    )
    original = _compile(project, operations, memo)
    assert [item.element for item in original.diagnostics] == ["unrelated"]
    assert _report_run(original).inputs.keys() == {"velocity_summary", "other"}

    project.documents["alternate-figure"].payload["theme"] = "plotly_dark"
    revised = _compile(project, operations, memo)
    assert revised == _compile(project, operations)
    old, new = _report_run(original), _report_run(revised)
    assert old.inputs["velocity_summary"] == new.inputs["velocity_summary"]
    assert old.inputs["other"] != new.inputs["other"]
    assert diff_plan_requests(original, revised)[_REPORT] == ElementChange(
        "changed", "inputs.other", ElementKey(_ANALYZE, "alternate")
    )
    assert revised.sources[_REPORT]["/inputs/other"].path == "/elements/alternate/products/figure"


def test_completeness_is_per_request_and_repeated_placements_share_one_binding():
    project = _report_project()
    operations = _operations()
    initial = _compile(project, operations)
    document = project.documents["velocity-report"].payload
    document["completeness"] = "allow_omissions"
    policy_edit = _compile(project, operations)
    assert diff_plan_requests(initial, policy_edit)[_REPORT] == ElementChange(
        "changed", "payload.completeness"
    )
    document["blocks"].append({"figure": "velocity_summary", "caption": "Again"})
    allowed = _compile(project, operations)
    run = _report_run(allowed)
    request = decode_request(allowed.requests[run.request_id], allowed.documents.__getitem__)
    assert request.spec.document.completeness == "allow_omissions"
    assert len(request.inputs["velocity_summary"]) == 1
    assert run.request_id != _report_run(initial).request_id

    project.steps["report"].inputs.clear()
    missing = _compile(project, operations)
    (problem,) = missing.diagnostics
    assert (problem.step, problem.path, problem.document) == (
        "report",
        "inputs.velocity_summary",
        "velocity-report",
    )
    assert all(run.kind != "report" for run in missing.runs)


def test_wrong_or_ambiguous_figure_ref_is_refused_and_text_report_needs_no_registry():
    project = _report_project()
    operations = _operations()
    source = project.steps["report"].inputs["velocity_summary"]
    project.steps["report"].inputs["velocity_summary"] = source.model_copy(
        update={"element": "summarize"}
    )
    wrong = _compile(project, operations)
    (problem,) = wrong.diagnostics
    assert problem.path == "inputs.velocity_summary.element"
    assert "summarize" in problem.detail and "figure" in problem.detail

    project.steps["report"].inputs["velocity_summary"] = source.model_copy(update={"row": None})
    project.steps["analyze"] = Step.model_validate(
        project.steps["analyze"].model_dump(mode="json", exclude_unset=True)
        | {
            "sweep": {
                "axes": {
                    "scale": {
                        "values": {
                            "one": {
                                "patches": {
                                    "operations.summarize.params.bounds_scale": {"value": 1}
                                }
                            },
                            "two": {
                                "patches": {
                                    "operations.summarize.params.bounds_scale": {"value": 2}
                                }
                            },
                        }
                    }
                }
            }
        }
    )
    ambiguous = _compile(project, operations)
    (problem,) = ambiguous.diagnostics
    assert problem.path == "inputs.velocity_summary"
    assert "scale=one" in problem.detail and "scale=two" in problem.detail

    text = compile_project(
        Project.model_validate(
            {
                "steps": {
                    "report": {
                        "kind": "report",
                        "base": {
                            "inline": {
                                "title": "Notes",
                                "blocks": [{"markdown": "An observation."}],
                            }
                        },
                    }
                }
            }
        )
    )
    assert not text.diagnostics
    request = decode_request(
        text.requests[_report_run(text).request_id], text.documents.__getitem__
    )
    assert request.kind == "report" and request.inputs == {}
