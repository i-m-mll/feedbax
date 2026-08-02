"""Fulfilling the two layer products native fulfillment could not reach.

The analysis layer compiles into an analysis *run* or an analysis *bundle*; the
figure layer into a figure or a figure *composition*. Both second members were
absent from the layer table and from the lowering table, so a closure that named
one refused as an unsupported product rather than executing.

What is under test here is the whole path for those two kinds, and nothing else:

* the layer table recognizes both compiled identities, and still refuses one it
  does not enumerate rather than lowering by resemblance;
* lowering produces the node request each compiled identity executes as — a
  bundle drives its own staged plan, a composition renders as the ordinary figure
  it resolves to — and refuses a document that declares the identity but is not a
  member of it;
* a closure carrying either kind fulfills end to end, and the receipts it earns
  are ordinary per-kind manifests;
* a bundle binds its roots by identity and refuses a selection that is not the
  set the plan named, and refuses to hand a single receipt to a consumer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pytest

from feedbax.analysis.evaluation import EvaluationRecipeResult
from feedbax.analysis.figures import RenderedFigure
from feedbax.analysis.fulfillment_adapters import (
    AnalysisBundleNodeRequest,
    FigureNodeRequest,
    FulfillmentEnvironment,
    admit_node,
    execute_node,
)
from feedbax.analysis.fulfillment_derivation import (
    COMPILED_PRODUCT_KINDS,
    UnsupportedCompiledProductError,
    derive_fulfillment_plan,
    read_compiled_outputs,
)
from feedbax.analysis.fulfillment_driver import (
    AmbiguousNodeReceiptError,
    closure_requests,
    fulfill_closure,
    preflight,
    truncated_closure,
)
from feedbax.analysis.fulfillment_lowering import NodeLoweringError, supported_lowerings
from feedbax.analysis.fulfillment_plan import LogicalKey
from feedbax.analysis.specs import AnalysisRecipeResult
from feedbax.contracts.analysis_bundle_composition import ANALYSIS_BUNDLE_SPEC_SCHEMA_ID
from feedbax.contracts.experiment_compile_lock import (
    AnalysisInputBinding,
    FigureRuntimeInputBinding,
    ReportParentBinding,
)
from feedbax.contracts.figures import FIGURE_COMPOSITION_SPEC_SCHEMA_ID
from feedbax.contracts.manifest import load_manifest

from tests.analysis_fixtures import ToyAnalysis, build_toy_analysis_data
from tests.fake_project_experiment.products import (
    CONDENSE_TYPE,
    PROBE_TYPE,
    QuillonOutputs,
    planned,
)


# --------------------------------------------------------------------------
# The smallest registered work that still writes real receipts
# --------------------------------------------------------------------------


class _Calls:
    def __init__(self) -> None:
        self.evaluation = 0
        self.analysis = 0


@pytest.fixture
def calls() -> _Calls:
    return _Calls()


@pytest.fixture
def outputs(tmp_path: Path) -> QuillonOutputs:
    return QuillonOutputs(tmp_path / "repo")


@pytest.fixture
def environment(
    tmp_path: Path, outputs: QuillonOutputs, application_registry_bundle, calls: _Calls
) -> FulfillmentEnvironment:
    def evaluation_recipe(run_spec, _root, _states_path, _execution_context):
        calls.evaluation += 1
        return EvaluationRecipeResult(
            states={"value": np.asarray(1, dtype=np.int32)},
            summary_metrics={"stage": run_spec.params.get("stage", "")},
            metadata={"states_schema": "quillon.states.v1"},
        )

    def analysis_recipe(_spec, _root, _inputs, _execution_context):
        calls.analysis += 1
        return AnalysisRecipeResult(
            analyses={"toy": ToyAnalysis(variant="toy", cache_result=True)},
            data=build_toy_analysis_data(value=1),
        )

    application_registry_bundle.evaluation_recipes.register(PROBE_TYPE, evaluation_recipe)
    application_registry_bundle.analysis_recipes.register(CONDENSE_TYPE, analysis_recipe)
    return FulfillmentEnvironment(
        root=tmp_path / "receipts",
        registries=application_registry_bundle,
        repo_root=outputs.root,
        issues=("b74330d",),
    )


@pytest.fixture
def rendered(monkeypatch: pytest.MonkeyPatch) -> None:
    """Render every figure as one empty plot, so identity is what is under test."""
    from feedbax.analysis import figures as figures_module

    monkeypatch.setattr(
        figures_module,
        "_build_figures",
        lambda *_args: [RenderedFigure(name="only", figure=go.Figure())],
    )


def _closure(outputs: QuillonOutputs, target: str):
    index = read_compiled_outputs(outputs.output_directory)
    return preflight(derive_fulfillment_plan(index, target=target), index)


def _fulfill(outputs: QuillonOutputs, target: str, *, environment):
    return fulfill_closure(_closure(outputs, target), environment=environment)


def _analysis_input(product, *, role_path: str, alias: str, role: str):
    return planned(
        product,
        role_path=role_path,
        consumer=AnalysisInputBinding(alias=alias, role=role),
    )


def _bundle_request(outputs: QuillonOutputs, target: str, *, environment):
    """Fulfil everything upstream of one bundle node and return its request."""
    fulfill_closure(truncated_closure(_closure(outputs, target), 1), environment=environment)
    return closure_requests(
        _closure(outputs, target),
        environment=environment,
        stop_at=LogicalKey("analysis", target),
    )[-1]


def _bound_sheaf(outputs: QuillonOutputs, name: str, **sheaf_kwargs) -> str:
    """A two-node closure: one evaluation, and a bundle bound to its receipt."""
    source = outputs.probe(f"{name}-source")
    outputs.sheaf(
        name,
        references=[
            _analysis_input(
                source, role_path="inputs.states", alias=f"{name}-source", role="observed"
            )
        ],
        **sheaf_kwargs,
    )
    return name


# --------------------------------------------------------------------------
# The layer table: both identities, and still no lowering by resemblance
# --------------------------------------------------------------------------


def test_both_second_layer_products_are_planned_and_lowerable() -> None:
    assert COMPILED_PRODUCT_KINDS[ANALYSIS_BUNDLE_SPEC_SCHEMA_ID].layer == "analysis"
    assert COMPILED_PRODUCT_KINDS[FIGURE_COMPOSITION_SPEC_SCHEMA_ID].layer == "figure"
    assert COMPILED_PRODUCT_KINDS[ANALYSIS_BUNDLE_SPEC_SCHEMA_ID].executable
    assert COMPILED_PRODUCT_KINDS[FIGURE_COMPOSITION_SPEC_SCHEMA_ID].executable
    assert ANALYSIS_BUNDLE_SPEC_SCHEMA_ID in supported_lowerings()
    assert FIGURE_COMPOSITION_SPEC_SCHEMA_ID in supported_lowerings()


def test_a_bundle_and_a_composition_are_derived_into_their_own_layers(
    outputs: QuillonOutputs,
) -> None:
    target = _bound_sheaf(outputs, "sheaf-derived")
    closure = _closure(outputs, target)
    assert closure.order == ("evaluation:sheaf-derived-source", "analysis:sheaf-derived")

    outputs.montage("montage-derived")
    montage = _closure(outputs, "montage-derived")
    assert montage.order == ("figure:montage-derived",)
    assert montage.target == LogicalKey("figure", "montage-derived")


def test_a_schema_id_the_table_does_not_enumerate_still_refuses(
    outputs: QuillonOutputs,
) -> None:
    """Neither new row widens the table into accepting a near-miss identity."""
    outputs.emit(
        "unlisted",
        {
            "schema_id": "feedbax.spec.analysis_bundle_v6",
            "schema_version": "feedbax.spec.analysis_bundle.v6",
            "name": "unlisted",
        },
    )
    index = read_compiled_outputs(outputs.output_directory)
    with pytest.raises(UnsupportedCompiledProductError) as caught:
        derive_fulfillment_plan(index, target="unlisted")
    assert ANALYSIS_BUNDLE_SPEC_SCHEMA_ID in str(caught.value)
    assert FIGURE_COMPOSITION_SPEC_SCHEMA_ID in str(caught.value)


# --------------------------------------------------------------------------
# Lowering: what each compiled identity executes as
# --------------------------------------------------------------------------


def test_a_bundle_lowers_to_a_bundle_node_bound_to_its_root_receipts(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment
) -> None:
    target = _bound_sheaf(outputs, "sheaf-lowered")
    request = _bundle_request(outputs, target, environment=environment)
    assert isinstance(request, AnalysisBundleNodeRequest)
    assert request.node_kind == "analysis"
    assert request.bundle.name == target
    assert [parent.kind for parent in request.root_inputs] == ["EvaluationRunManifest"]
    assert [parent.role for parent in request.root_inputs] == ["observed"]


def test_a_composition_lowers_to_the_ordinary_figure_it_renders_as(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment
) -> None:
    source = outputs.probe("montage-source")
    outputs.montage(
        "montage-lowered",
        references=[
            planned(
                source,
                role_path="runtime.states",
                consumer=FigureRuntimeInputBinding(input_role="observed"),
            )
        ],
    )
    fulfill_closure(_closure(outputs, "montage-source"), environment=environment)
    requests = closure_requests(
        _closure(outputs, "montage-lowered"),
        environment=environment,
        stop_at=LogicalKey("figure", "montage-lowered"),
    )
    request = requests[-1]
    assert isinstance(request, FigureNodeRequest)
    assert request.node_kind == "figure"
    assert request.spec["schema_id"] == FIGURE_COMPOSITION_SPEC_SCHEMA_ID
    assert request.spec == dict(_closure(outputs, "montage-lowered").nodes[-1].document)
    assert [ref.role for ref in request.runtime_inputs] == ["observed"]


def test_a_document_that_is_not_a_member_of_the_identity_it_declares_refuses(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment
) -> None:
    """The table decided which model; the model decides whether this is one."""
    outputs.sheaf(
        "sheaf-malformed",
        stages=[{"name": "condense", "kind": "analysis"}],  # no analysis_type
    )
    with pytest.raises(NodeLoweringError, match="AnalysisBundleSpec"):
        _fulfill(outputs, "sheaf-malformed", environment=environment)

    outputs.emit(
        "montage-malformed",
        {
            "schema_id": FIGURE_COMPOSITION_SPEC_SCHEMA_ID,
            "schema_version": "feedbax.spec.figure_composition.v2",
            "parent": {"ref": "bases/absent.figure.json", "sha256": "a" * 64},
            "deltas": [],
        },
    )
    with pytest.raises(NodeLoweringError, match="FigureCompositionSpec"):
        _fulfill(outputs, "montage-malformed", environment=environment)


# --------------------------------------------------------------------------
# End to end: the closure runs, and the receipts are ordinary
# --------------------------------------------------------------------------


def test_a_bundle_closure_fulfills_end_to_end(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment, calls: _Calls
) -> None:
    target = _bound_sheaf(outputs, "sheaf-run")
    run = _fulfill(outputs, target, environment=environment)

    assert run.execution_order == ("evaluation:sheaf-run-source", "analysis:sheaf-run")
    assert calls.evaluation == 1
    assert calls.analysis == 1
    bundle_result = run.results[-1]
    assert bundle_result.node_kind == "analysis"
    assert bundle_result.disposition == "executed"
    assert bundle_result.receipts
    for receipt in bundle_result.receipts:
        assert receipt.path.is_file()
        assert load_manifest(receipt.path).status == "completed"
    assert [receipt.node_kind for receipt in bundle_result.receipts] == ["analysis"]


def test_a_composition_closure_fulfills_end_to_end(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment, rendered: None
) -> None:
    outputs.montage("montage-run")
    run = _fulfill(outputs, "montage-run", environment=environment)

    result = run.results[-1]
    assert result.node_kind == "figure"
    assert result.disposition == "executed"
    manifest = load_manifest(result.receipt.path)
    assert manifest.status == "completed"
    assert manifest.kind == "FigureManifest"
    # The composition really composed: the receipt carries the ordinary figure
    # the delta produced over the repo-tracked parent, not the parent itself.
    assert manifest.figure_spec.inline["name"] == "montage-run"
    assert manifest.figure_spec.inline["schema_id"] == "feedbax.spec.figure"
    assert _fulfill(outputs, "montage-run", environment=environment).results[-1].disposition == (
        "reused"
    )


# --------------------------------------------------------------------------
# What a bundle node refuses
# --------------------------------------------------------------------------


def test_a_bundle_refuses_a_selection_that_is_not_the_set_the_plan_named(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment, calls: _Calls
) -> None:
    """A predicate that cannot reach the bound receipts describes other work."""
    target = _bound_sheaf(outputs, "sheaf-narrowed", manifest_kind="AnalysisRunManifest")
    with pytest.raises(ValueError, match="binds root receipts"):
        _fulfill(outputs, target, environment=environment)
    assert calls.analysis == 0


def test_a_bundle_hands_no_single_receipt_to_a_consumer(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment
) -> None:
    sheaf = outputs.sheaf(
        "sheaf-read",
        stages=[
            {"name": "one", "kind": "analysis", "analysis_type": CONDENSE_TYPE},
            {"name": "two", "kind": "analysis", "analysis_type": CONDENSE_TYPE},
        ],
    )
    outputs.bulletin(
        "sheaf-reader",
        references=[
            planned(
                sheaf,
                role_path="body.bundle",
                consumer=ReportParentBinding(parent_kind="bundle", parent_id="bundle"),
            )
        ],
    )
    with pytest.raises(AmbiguousNodeReceiptError, match="analysis bundle"):
        closure_requests(_closure(outputs, "sheaf-reader"), environment=environment)


def test_a_bundle_node_is_neither_admitted_nor_executed_as_one_receipt(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment
) -> None:
    target = _bound_sheaf(outputs, "sheaf-single")
    request = _bundle_request(outputs, target, environment=environment)
    with pytest.raises(TypeError, match="no single receipt"):
        admit_node(request, environment=environment)
    with pytest.raises(TypeError, match="not a single execution"):
        execute_node(request, environment=environment)


def test_a_template_bundle_runs_through_its_own_expansion(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment, calls: _Calls
) -> None:
    """The bundle document's own execution shape decides which entrypoint runs."""
    target = _bound_sheaf(
        outputs,
        "sheaf-templates",
        templates=[{"name": "condense", "mode": "grouped", "analysis_type": CONDENSE_TYPE}],
    )
    run = _fulfill(outputs, target, environment=environment)
    result = run.results[-1]
    assert calls.analysis == 1
    assert [receipt.node_kind for receipt in result.receipts] == ["analysis"]
    assert load_manifest(result.receipts[0].path).kind == "AnalysisRunManifest"


def test_a_bundle_never_runs_in_an_environment_it_cannot_carry_bindings_into(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment, calls: _Calls
) -> None:
    """A resolved staged context cannot be turned back into bundle bindings."""
    from dataclasses import replace

    from feedbax.analysis.execution_context import (
        EMPTY_STAGED_EXECUTION_CONTEXT,
        with_staged_repo_root,
    )

    target = _bound_sheaf(outputs, "sheaf-staged")
    staged = replace(
        environment,
        execution_context=with_staged_repo_root(
            EMPTY_STAGED_EXECUTION_CONTEXT, outputs.root
        ),
    )
    with pytest.raises(ValueError, match="staged execution context"):
        _fulfill(outputs, target, environment=staged)
    assert calls.analysis == 0


def test_a_bundle_stating_no_work_refuses(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment
) -> None:
    outputs.sheaf("sheaf-empty", stages=[])
    with pytest.raises(ValueError, match="neither a staged plan nor a template set"):
        _fulfill(outputs, "sheaf-empty", environment=environment)
