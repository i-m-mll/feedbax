"""Plan the real saved Evaluate and typed velocity figure without result access."""

from copy import deepcopy
import json
from pathlib import Path

from feedbax.builtins.figure_templates import VelocitySummary, VelocitySummarySettings
from feedbax.contracts.analysis_document import FigureElement
from feedbax.contracts.base import sha256_bytes
from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.project import Document, DocumentReference, Project, Step
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
from feedbax.contracts.project_resolution import BaseReaders, ResolutionCode
from feedbax.contracts.project_sweep import Sweep
from feedbax.contracts.request_identity import decode_request
from feedbax.contracts.request_refs import ProductRef
from feedbax.plugins.application import new_application_registry_bundle
from tests.support.managed_train_fixture import managed_evaluation_step, managed_train_project


_KEY = RunKey("default", "analyze", "default")
_FIXTURES = Path(__file__).resolve().parents[1] / "web/e2e/studio/fixtures"


def _project() -> Project:
    project = managed_train_project()
    project.steps["baseline"] = Step.model_validate(managed_evaluation_step("train"))
    project.documents["velocity-analysis"] = Document.model_validate(
        {
            "schema": "feedbax.analysis_document.v1",
            "payload": json.loads((_FIXTURES / "velocity-analysis.json").read_bytes()) | {
                "inputs": {"states": {"kind": "dataset"}},
            },
        }
    )
    project.documents["velocity-figure"] = Document.model_validate(
        {
            "schema": "feedbax.figure_document.v1",
            "payload": VelocitySummary(VelocitySummarySettings()).model_dump(mode="json", exclude_unset=True),
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


def _conditions() -> Step:
    mass = ModelAddress.model_validate({
        "document": "point-mass", "composites": [],
        "element": {"kind": "field", "parent": {"kind": "component", "id": "plant"}, "path": ["fields", "mass"]},
    }).to_text()
    return Step.model_validate(managed_evaluation_step("train") | {"sweep": {"axes": {"mass": {"values": {
        name: {"value": value, "patches": {mass: {"value": value}}}
        for name, value in (("nominal", 1), ("heavy", 2))
    }}}}})


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
    assert row.elements["profile"].inputs["summary"][0].element == ElementKey(_KEY, "summarize")
    assert figure.spec.document == VelocitySummary(VelocitySummarySettings())
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
    summary = document["operations"]["summarize"]
    ddof = summary["params"].pop("ddof")
    summary["inputs"].clear()
    (draft_row,) = (row for row in compile_project(project, operations=registry).rows if isinstance(row, PlannedAnalyzeRun))
    assert draft_row.inputs["states"][0].producer == RunKey("default", "baseline", "default")
    states_port = registry.resolve("feedbax.aligned_velocity").value.definition.inputs["states"]
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

    summary["params"]["ddof"] = ddof
    view = document["views"]["profile"]
    view["inputs"].clear()
    draft_plan = compile_project(project, operations=registry)
    draft_row = next(row for row in draft_plan.rows if isinstance(row, PlannedAnalyzeRun))
    assert draft_row.figures["profile"].data == ("summary",)
    assert "profile" not in draft_row.elements
    reviewed = review_wire(project, _KEY, "velocity-analysis", StudioAnalyzeWire(
        element="profile", role="summary", source={"element": "summarize", "product": "summary"},
    ), registry, states)
    assert reviewed.patches[0].path == "/payload/views/profile/inputs/summary"
    assert reviewed.patches[0].op == "add"
    assert view == {"figure": {"document": "velocity-figure"}, "inputs": {}}


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


def test_named_figure_matches_inline_and_variant_edit_reaches_both_views():
    project = _project()
    views = project.documents["velocity-analysis"].payload["views"]
    views["second"] = deepcopy(views["profile"])
    views["inline"] = deepcopy(views["profile"])
    views["inline"]["figure"] = deepcopy(project.documents["velocity-figure"].payload)
    assert isinstance(FigureElement.model_validate(views["second"]).figure, DocumentReference)
    assert FigureElement.model_validate(views["inline"]).figure == VelocitySummary(VelocitySummarySettings())
    project.sweep = Sweep.model_validate({"rows": {"base": {}, "dark": {}}})
    operations = _operations()
    memo = ProjectPlanMemo()
    base, dark = (RunKey(variant, "analyze", "default") for variant in ("base", "dark"))
    initial = compile_project(project, operations=operations, source_runs=(base, dark), memo=memo)
    assert not initial.diagnostics
    keys = {ElementKey(row, view) for row in (base, dark) for view in views}
    (request_id,) = {initial.elements[key].request_id for key in keys}
    inline = deepcopy(project)
    inline_views = inline.documents["velocity-analysis"].payload["views"]
    for view in inline_views.values():
        view["figure"] = deepcopy(inline_views["inline"]["figure"])
    reference = compile_project(inline, operations=operations)
    assert initial.requests[request_id] == reference.requests[request_id]
    assert initial.documents == reference.documents

    pinned = deepcopy(project)
    document = pinned.documents["velocity-figure"]
    raw = json.dumps(document.payload).encode()
    document.payload = {"path": "../shared-figure.json", "sha256": sha256_bytes(raw)}
    reads = []

    def read_external(path):
        reads.append(path)
        return raw

    pin_plan = compile_project(
        pinned, operations=operations, source_runs=(base, dark),
        readers=BaseReaders(figure_external=read_external),
    )
    assert not pin_plan.diagnostics
    assert pin_plan.requests == initial.requests and pin_plan.documents == initial.documents
    assert reads == ["../shared-figure.json"]
    pin_source = pin_plan.sources[dark]["/payload/views/second/figure/theme"]
    assert (pin_source.kind, pin_source.name, pin_source.path) == ("document", "../shared-figure.json", "/theme")

    project.sweep = Sweep.model_validate({"rows": {
        "base": {}, "dark": {"patches": {"documents.velocity-figure.payload.theme": {"value": "plotly_dark"}}},
    }})
    revised = compile_project(project, operations=operations, source_runs=(base, dark), memo=memo)
    assert not revised.diagnostics and revised.runs == initial.runs
    assert revised == compile_project(project, operations=operations, source_runs=(base, dark))
    changes = diff_analyze_elements(initial, revised)
    assert {key: value for key, value in changes.items() if value.state != "unchanged"} == {
        ElementKey(dark, view): ElementChange("changed", f"payload.views.{view}.figure.theme")
        for view in ("profile", "second")
    }
    assert revised.requests[request_id] == initial.requests[request_id]
    assert len({revised.elements[ElementKey(dark, view)].request_id for view in ("profile", "second")}) == 1
    for view in ("profile", "second"):
        source = revised.sources[dark][f"/payload/views/{view}/figure/theme"]
        assert (source.kind, source.name, source.path) == ("variant", "dark", "/documents/velocity-figure/payload/theme")


def test_installed_figure_template_shares_identity_with_its_inline_figure():
    project = _project()
    authored = project.documents["velocity-figure"]
    profile = project.documents["velocity-analysis"].payload["views"]["profile"]
    saved_source = profile["figure"]
    profile["figure"] = deepcopy(authored.payload)
    bundle = new_application_registry_bundle()
    bundle.seal()
    memo = ProjectPlanMemo()

    def plan():
        return compile_project(
            project, operations=bundle.operations, figure_templates=bundle.figure_templates, memo=memo,
        )

    inline = _decode(plan(), "profile")
    profile["figure"] = saved_source
    authored.payload = {"template": "feedbax.VelocitySummary"}
    named = _decode(plan(), "profile")
    assert named.id == inline.id
    assert named.spec == inline.spec

    authored.payload["settings"] = {"checkpoint_index": 0}
    assert _decode(plan(), "profile").id == inline.id

    authored.payload["settings"]["checkpoint_index"] = 1
    changed = _decode(plan(), "profile")
    assert changed.id != inline.id
    assert changed.spec.document.layers[0].select["checkpoint"].index == 1

    authored.payload["settings"]["checkpoint_index"] = -1
    invalid = plan()
    (problem,) = invalid.diagnostics
    assert problem.document_path == "payload.settings.checkpoint_index"

    authored.payload["template"] = "missing.Figure"
    unresolved = plan()
    (problem,) = unresolved.diagnostics
    assert (problem.path, problem.document_path) == (
        "payload.views.profile.figure.template", "payload.template",
    )
    assert "'missing.Figure' from package 'missing'" in problem.detail
    assert set(unresolved.elements) == {ElementKey(_KEY, "align"), ElementKey(_KEY, "summarize")}


def test_named_figure_pin_does_not_supply_an_external_train_base():
    project = _project()
    train = project.steps["train"]
    raw = json.dumps(train.base.inline).encode()
    project.documents["train-pin"] = Document.model_validate({
        "schema": "feedbax.figure_document.v1",
        "payload": {"path": "train.json", "sha256": sha256_bytes(raw)},
    })
    project.steps["external"] = train.model_copy(update={"base": DocumentReference(document="train-pin")})
    reads = []

    def read_figure(path):
        reads.append(path)
        return raw

    plan = compile_project(project, operations=_operations(), readers=BaseReaders(figure_external=read_figure))
    assert reads == []
    assert not any(run.key.step == "external" for run in plan.runs)
    assert any(problem.step == "external" for problem in plan.diagnostics)


def test_refuses_named_figure_at_its_authored_field_and_keeps_inline_sibling():
    project = _project()
    operations = _operations()
    figure = project.documents["velocity-figure"]
    views = project.documents["velocity-analysis"].payload["views"]
    views["inline"] = deepcopy(views["profile"])
    views["inline"]["figure"] = deepcopy(figure.payload)
    initial = compile_project(project, operations=operations)
    assert not initial.diagnostics
    figure.schema_id = "example.not-a-figure"
    wrong_schema = compile_project(project, operations=operations)
    (problem,) = wrong_schema.diagnostics
    assert (problem.code, problem.document, problem.document_path, problem.path) == (
        ResolutionCode.INVALID_DOCUMENT, "velocity-figure", "schema", "payload.views.profile.figure",
    )

    figure.schema_id = "feedbax.figure_document.v1"
    figure.payload["layers"][0].setdefault("style", {})["band_alpha"] = -1
    malformed = compile_project(project, operations=operations)
    problem, layers = malformed.diagnostics
    assert (problem.code, problem.document, problem.document_path, problem.path, problem.element) == (
        ResolutionCode.INVALID_DOCUMENT, "velocity-figure", "payload.layers.0.style.band_alpha",
        "payload.views.profile.figure.layers.0.style.band_alpha", "profile",
    )
    assert (layers.code, layers.document, layers.document_path, layers.path, layers.element) == (
        ResolutionCode.INVALID_DOCUMENT, "velocity-figure", "payload.layers",
        "payload.views.profile.figure.layers", "profile",
    )
    for refused in (wrong_schema, malformed):
        assert set(refused.elements) == set(initial.elements) - {ElementKey(_KEY, "profile")}
        assert all(refused.elements[key].request_id == initial.elements[key].request_id for key in refused.elements)


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
    assert changed.elements[ElementKey(_KEY, "summarize")].inputs["velocity"][0].element == ElementKey(
        _KEY, "alias"
    )

    document["operations"]["summarize"]["params"]["ddof"] = -1
    refused = compile_project(project, operations=operations)
    assert {(item.element, item.path, item.document) for item in refused.diagnostics} == {
        ("summarize", "payload.operations.summarize.params.ddof", "velocity-analysis"),
        ("profile", "payload.views.profile.inputs.summary", "velocity-analysis"),
    }
    assert set(refused.elements) == {ElementKey(_KEY, "align"), ElementKey(_KEY, "alias")}


def test_figure_binding_fails_closed():
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

def test_analyze_collects_static_rows_in_each_named_variant():
    project = _project()
    project.steps["conditions"] = _conditions()
    project.sweep = Sweep.model_validate({"rows": {"base": {}, "compare": {"from": "base"}}})
    project.steps["analyze"] = Step.model_validate(
        {
            "kind": "analyze",
            "base": {"document": "velocity-analysis"},
            "inputs": {"states": {"step": "conditions", "where": {"axes": {"mass": ["heavy", "nominal"]}}}},
        }
    )
    operations = _operations()
    plan = compile_project(project, operations=operations)
    assert not plan.diagnostics
    sources = {
        run.key.variant: tuple(binding.producer for binding in run.inputs["states"])
        for run in plan.rows if isinstance(run, PlannedAnalyzeRun)
    }
    assert sources == {
        name: (RunKey(name, "conditions", "mass=nominal"), RunKey(name, "conditions", "mass=heavy"))
        for name in ("base", "compare")
    }
    assert plan.elements[ElementKey(RunKey("base", "analyze", "default"), "align")].request_id == (
        plan.elements[ElementKey(RunKey("compare", "analyze", "default"), "align")].request_id
    )

    project.steps["analyze"].inputs["states"] = project.steps["analyze"].inputs["states"].model_copy(update={"where": None})
    assert compile_project(project, operations=operations).elements == plan.elements


def test_ordered_analyze_subset_is_one_request_unless_the_input_is_expanded():
    project = _project()
    operations = _operations()
    project.steps["conditions"] = _conditions()
    baseline = compile_project(project, operations=operations)
    conditions = {
        run.coordinates["mass"].value_id: run for run in baseline.runs if run.key.step == "conditions"
    }
    selection = [
        {"key": key, "variant": "default", "row": conditions[key].key.row}
        for key in ("heavy", "nominal")
    ]

    def choose(rows):
        project.steps["analyze"] = Step.model_validate({
            "kind": "analyze",
            "base": {"document": "velocity-analysis"},
            "inputs": {"states": {"step": "conditions", "where": {"rows": rows}}},
        })
        return compile_project(project, operations=operations)

    ordered = choose(selection)
    reversed_plan = choose(selection[::-1])
    assert not ordered.diagnostics and not reversed_plan.diagnostics
    first, second = _decode(ordered, "align"), _decode(reversed_plan, "align")
    assert first.inputs["states"] == second.inputs["states"][::-1]
    assert first.id != second.id
    assert len(ordered.elements) == len(reversed_plan.elements) == 3

    project.steps["analyze"].sweep = Sweep.model_validate({"axes": {"source": {"input": "states"}}})
    expanded = compile_project(project, operations=operations)
    assert not expanded.diagnostics
    assert len(expanded.elements) == 6
    assert {
        tuple(binding.member.key for binding in element.inputs["states"])
        for element in expanded.elements.values() if element.key.name == "align"
    } == {("heavy",), ("nominal",)}


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
