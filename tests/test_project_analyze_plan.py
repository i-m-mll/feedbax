"""Plan the real saved Evaluate and typed velocity figure without result access."""

from copy import deepcopy
import json
from pathlib import Path

from feedbax.analysis.figure_presets import velocity_summary_figure
from feedbax.contracts.project import Document, Project, Step
from feedbax.contracts.project_io import (
    ProjectConfiguration,
    StudioProjectFile,
    dump_studio_project,
    dump_text_project,
    load_text_project,
    parse_studio_project,
)
from feedbax.contracts.project_plan import (
    ElementChange,
    ElementKey,
    PlannedAnalyzeRun,
    ProjectPlanMemo,
    RunKey,
    compile_project,
    diff_analyze_elements,
)
from feedbax.contracts.request_identity import decode_request
from feedbax.contracts.request_refs import ProductRef
from feedbax.plugins.application import new_application_registry_bundle
from tests.test_canonical_evaluate_admission import authored_evaluation_project


_KEY = RunKey("default", "analyze", "default")
_FIXTURES = Path(__file__).resolve().parents[1] / "web/e2e/studio/fixtures"


def _project() -> Project:
    project, _ = authored_evaluation_project()
    project.steps = {name: project.steps[name] for name in ("train", "baseline")}
    project.documents["velocity-analysis"] = Document.model_validate(
        {
            "schema": "feedbax.analysis_document.v1",
            "payload": json.loads((_FIXTURES / "velocity-analysis.json").read_bytes()),
        }
    )
    project.documents["velocity-figure"] = Document.model_validate(
        {
            "schema": "feedbax.figure_document.v1",
            "payload": velocity_summary_figure().model_dump(mode="json", exclude_unset=True),
        }
    )
    project.steps["analyze"] = Step.model_validate(
        {
            "kind": "analyze",
            "base": {"document": "velocity-analysis"},
            "inputs": {"states": {"step": "baseline"}},
        }
    )
    return project


def _operations():
    bundle = new_application_registry_bundle()
    bundle.seal()
    return bundle.operations


def _decode(plan, name):
    element = plan.elements[ElementKey(_KEY, name)]
    return decode_request(plan.requests[element.request_id], plan.documents.__getitem__)


def test_text_and_studio_compile_the_same_saved_evaluate_element_plan(tmp_path):
    project = _project()
    operations = _operations()
    upstream = project.model_copy(deep=True)
    upstream.steps.pop("analyze")
    baseline = compile_project(upstream)
    for name, content in dump_text_project(
        project, ProjectConfiguration(project="reaching", store="results")
    ).items():
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_bytes(content)
    text = load_text_project(tmp_path)
    studio = parse_studio_project(
        dump_studio_project(StudioProjectFile(project=project, presentation={}))
    )
    plan = compile_project(text.project, operations=operations, source_runs=(_KEY,))
    assert plan == compile_project(studio.project, operations=operations, source_runs=(_KEY,))
    assert not plan.diagnostics
    assert plan.runs == baseline.runs
    assert all(plan.requests[key] == raw for key, raw in baseline.requests.items())
    assert all(plan.documents[key] == raw for key, raw in baseline.documents.items())
    (row,) = (item for item in plan.rows if isinstance(item, PlannedAnalyzeRun))
    assert row.key == _KEY and tuple(row.elements) == ("align", "summarize", "profile")
    assert row.document_ids == {
        "payload": "velocity-analysis",
        "payload.views.profile.figure": "velocity-figure",
    }
    align, summary, figure = (_decode(plan, name) for name in row.elements)
    assert [item.kind for item in (align, summary, figure)] == ["analyze", "analyze", "render"]
    evaluate = next(run for run in plan.runs if run.kind == "evaluate")
    assert align.inputs["states"][0].reference == ProductRef(
        request_id=evaluate.request_id, role="states"
    )
    assert summary.inputs["velocity"][0].reference == ProductRef(
        request_id=align.id, role="velocity"
    )
    assert figure.inputs["summary"][0].reference == ProductRef(
        request_id=summary.id, role="summary"
    )
    assert row.elements["profile"].inputs["summary"].element == ElementKey(_KEY, "summarize")
    assert align.spec.evaluation_states_policy == "require_durable"
    assert figure.spec.figure is None and figure.spec.document == velocity_summary_figure()
    source = plan.sources[_KEY]["/payload/operations/summarize/params/bounds_scale"]
    assert (source.kind, source.name, source.path) == (
        "document",
        "velocity-analysis",
        "/operations/summarize/params/bounds_scale",
    )


def test_managed_wire_review_uses_plan_ports_without_writing_a_draft():
    import pytest

    from feedbax.contracts.operation import DatasetPort
    from feedbax.contracts.studio_analyze import StudioAnalyzeWire
    from feedbax.web.services.analyze_authoring import review_wire

    project = _project()
    registry = _operations()
    document = project.documents["velocity-analysis"].payload
    document["views"]["profile"]["figure"] = project.documents["velocity-figure"].payload
    summary = document["operations"]["summarize"]
    summary["params"].pop("ddof")
    summary["inputs"].clear()
    (draft_row,) = (row for row in compile_project(project, operations=registry).rows if isinstance(row, PlannedAnalyzeRun))
    assert draft_row.states.producer == RunKey("default", "baseline", "default")
    states_port = registry.resolve("feedbax.aligned_velocity").definition.inputs["states"]
    states = DatasetPort(schema_ids=(states_port.schema_ids[0],))
    with pytest.raises(ValueError, match="Requires dataset schemas"):
        review_wire(project, _KEY, "velocity-analysis", StudioAnalyzeWire(
            element="summarize", role="velocity", source={"input": "states"},
        ), registry, states)
    result = review_wire(project, _KEY, "velocity-analysis", StudioAnalyzeWire(
        element="summarize", role="velocity", source={"element": "align", "product": "velocity"},
    ), registry, states)
    assert result.model_dump(mode="json") == {
        "address": {"kind": "document", "id": "velocity-analysis"},
        "patches": [{
            "op": "add", "path": "/payload/operations/summarize/inputs/velocity",
            "value": {"element": "align", "product": "velocity"},
        }],
    }
    assert summary["inputs"] == {}
    assert "ddof" not in summary["params"]
    summary["inputs"]["velocity"] = {"element": "align", "product": "velocity"}
    replacement = review_wire(project, _KEY, "velocity-analysis", StudioAnalyzeWire(
        element="summarize", role="velocity", source={"element": "align", "product": "velocity"},
    ), registry, states)
    assert replacement.patches[0].op == "replace"


def test_element_edit_and_upstream_attribution_leave_independent_requests_equal():
    project = _project()
    operations = _operations()
    memo = ProjectPlanMemo()
    original = compile_project(project, operations=operations, memo=memo)
    edited = deepcopy(project)
    edited.documents["velocity-analysis"].payload["operations"]["summarize"]["params"][
        "bounds_scale"
    ] = 2.0
    revised = compile_project(edited, operations=operations, memo=memo)
    assert revised == compile_project(edited, operations=operations)
    assert revised.runs == original.runs
    align, summary, figure = (ElementKey(_KEY, name) for name in ("align", "summarize", "profile"))
    assert diff_analyze_elements(original, revised) == {
        align: ElementChange("unchanged"),
        summary: ElementChange("changed", "payload.operations.summarize.params.bounds_scale"),
        figure: ElementChange("changed", "payload.views.profile.inputs.summary", summary),
    }
    edited.documents["velocity-figure"].payload["theme"] = "plotly_dark"
    figure_edit = compile_project(edited, operations=operations, memo=memo)
    assert diff_analyze_elements(revised, figure_edit) == {
        align: ElementChange("unchanged"),
        summary: ElementChange("unchanged"),
        figure: ElementChange("changed", "payload.views.profile.figure.theme"),
    }


def test_element_aliases_share_identity_and_refusals_name_the_authored_element():
    project = _project()
    operations = _operations()
    original = compile_project(project, operations=operations)
    document = project.documents["velocity-analysis"].payload
    document["operations"]["alias"] = deepcopy(document["operations"]["align"])
    document["operations"]["summarize"]["inputs"]["velocity"]["element"] = "alias"
    changed = compile_project(project, operations=operations)
    assert changed.requests == original.requests
    assert changed.elements[ElementKey(_KEY, "alias")].request_id == _decode(original, "align").id
    assert changed.elements[ElementKey(_KEY, "summarize")].inputs["velocity"].element == ElementKey(
        _KEY, "alias"
    )

    document["operations"]["summarize"]["params"]["ddof"] = -1
    refused = compile_project(project, operations=operations)
    assert {(item.element, item.path, item.document) for item in refused.diagnostics} == {
        ("summarize", "payload.operations.summarize.params.ddof", "velocity-analysis"),
        ("profile", "payload.views.profile.inputs.summary", "velocity-analysis"),
    }
    assert set(refused.elements) == {ElementKey(_KEY, "align"), ElementKey(_KEY, "alias")}


def test_figure_binding_and_unselected_evaluate_rows_fail_closed():
    project = _project()
    operations = _operations()
    project.documents["velocity-figure"].payload["layers"][0]["encode"]["upper"] = "unknown_upper"
    refused = compile_project(project, operations=operations)
    (problem,) = refused.diagnostics
    assert (problem.document, problem.element, problem.path) == (
        "velocity-figure",
        "profile",
        "payload.views.profile.figure.layers.0.encode.upper",
    )
    assert "upper" in problem.detail and "unknown_upper" in problem.detail
    assert len(refused.elements) == 2

    original, _ = authored_evaluation_project()
    project.steps["conditions"] = original.steps["conditions"]
    project.steps["analyze"] = Step.model_validate(
        {
            "kind": "analyze",
            "base": {"document": "velocity-analysis"},
            "inputs": {"states": {"step": "conditions"}},
        }
    )
    ambiguous = compile_project(project, operations=operations)
    (problem,) = ambiguous.diagnostics
    assert problem.path == "inputs.states" and not ambiguous.elements


def test_operation_cycle_keeps_independent_elements_plannable():
    project = _project()
    document = project.documents["velocity-analysis"].payload
    document["operations"]["independent"] = deepcopy(document["operations"]["align"])
    document["operations"]["align"]["inputs"]["states"] = {
        "element": "summarize",
        "product": "summary",
    }
    plan = compile_project(project, operations=_operations())
    assert set(plan.elements) == {ElementKey(_KEY, "independent")}
    cycle = [item for item in plan.diagnostics if "dependency cycle" in item.detail]
    assert {item.element for item in cycle} == {"align", "summarize"}
    assert all(item.document == "velocity-analysis" for item in cycle)
