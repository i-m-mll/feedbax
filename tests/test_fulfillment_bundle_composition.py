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

import hashlib
import json
from dataclasses import replace
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
    analysis_bundle_root_identities,
    analysis_bundle_root_run_ids,
    execute_analysis_bundle_node,
    execute_node,
)
from feedbax.analysis.fulfillment_adapters import _require_bundle_roots
from feedbax.analysis.bundles import (
    BundleRootVerificationError,
    DuplicateBundleRootError,
)
from feedbax.workflow.derivation import (
    COMPILED_PRODUCT_KINDS,
    UnsupportedCompiledProductError,
    derive_workflow_plan,
    read_compiled_outputs,
)
from feedbax.workflow.execution import (
    AmbiguousNodeReceiptError,
    workflow_requests,
    execute_workflow,
    prepare_workflow,
    truncated_workflow,
)
from feedbax.workflow.operation_execution import NodeLoweringError, supported_lowerings
from feedbax.workflow.plan import LogicalKey
from feedbax.analysis.specs import AnalysisRecipeResult
from feedbax.contracts.analysis_bundle_composition import ANALYSIS_BUNDLE_SPEC_SCHEMA_ID
from feedbax.contracts.experiment_compile_lock import (
    AnalysisInputBinding,
    FigureRuntimeInputBinding,
    ReportParentBinding,
)
from feedbax.contracts.figures import FIGURE_COMPOSITION_SPEC_SCHEMA_ID
from feedbax.contracts.manifest import canonical_manifest_path, load_manifest

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
    return prepare_workflow(derive_workflow_plan(index, target=target), index)


def _fulfill(outputs: QuillonOutputs, target: str, *, environment):
    return execute_workflow(_closure(outputs, target), environment=environment)


def _analysis_input(product, *, role_path: str, alias: str, role: str):
    return planned(
        product,
        role_path=role_path,
        consumer=AnalysisInputBinding(alias=alias, role=role),
    )


def _bundle_request(outputs: QuillonOutputs, target: str, *, environment):
    """Fulfil everything upstream of one bundle node and return its request."""
    execute_workflow(truncated_workflow(_closure(outputs, target), 1), environment=environment)
    return workflow_requests(
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
        derive_workflow_plan(index, target="unlisted")
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


def test_an_unbound_bundle_lowers_with_no_declared_root_set(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment
) -> None:
    """No root reference in the lock is the one case ambient selection stands.

    ``None`` and an empty set are two different statements. ``None`` says the
    lock authenticated nothing to compare a selection against, so the bundle's
    own predicate is the only statement about its roots; an empty declared set
    would say the bundle executes over no receipts, which is not work.
    """
    outputs.sheaf("sheaf-unbound")

    request = workflow_requests(
        _closure(outputs, "sheaf-unbound"),
        environment=environment,
        stop_at=LogicalKey("analysis", "sheaf-unbound"),
    )[-1]

    assert isinstance(request, AnalysisBundleNodeRequest)
    assert request.root_inputs is None
    assert analysis_bundle_root_run_ids(request) is None


def test_a_declared_but_empty_root_set_is_refused_rather_than_read_as_ambient(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment
) -> None:
    outputs.sheaf("sheaf-empty")
    request = workflow_requests(
        _closure(outputs, "sheaf-empty"),
        environment=environment,
        stop_at=LogicalKey("analysis", "sheaf-empty"),
    )[-1]

    with pytest.raises(ValueError, match="declares an empty root set"):
        analysis_bundle_root_run_ids(replace(request, root_inputs=()))


def test_a_bound_bundle_binds_by_exact_manifest_identity(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment
) -> None:
    """A declared root set is the exact set, with no ambient fallback beneath it.

    Identity is kind, id, and the bytes the bound ref authenticates — not the id
    on its own. Ids are unique within a kind, so an id alone cannot say which
    artifact was selected, and it says nothing at all about which bytes.
    """
    target = _bound_sheaf(outputs, "sheaf-exact")
    request = _bundle_request(outputs, target, environment=environment)

    identities = analysis_bundle_root_identities(request)

    assert identities is not None
    assert [identity.kind for identity in identities] == ["EvaluationRunManifest"]
    assert [identity.id for identity in identities] == [parent.id for parent in request.root_inputs]
    assert all(identity.manifest_sha256 is not None for identity in identities)
    # The id-only view still exists, because bundle selection addresses
    # candidates by id, but it is explicitly not what the gate compares.
    assert analysis_bundle_root_run_ids(request) == tuple(identity.id for identity in identities)


def test_a_selected_root_of_another_kind_with_the_same_id_refuses(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment
) -> None:
    """The reviewer's case: same id, different kind, and it must not pass.

    A lock naming a ``TrainingRunManifest`` and a selection producing an
    ``EvaluationRunManifest`` that happens to share its id are two different
    artifacts. Comparing id sets cannot tell them apart; comparing addresses can.
    """
    target = _bound_sheaf(outputs, "sheaf-kind-swap")
    request = _bundle_request(outputs, target, environment=environment)
    identities = analysis_bundle_root_identities(request)
    assert identities is not None
    shared_id = identities[0].id

    with pytest.raises(ValueError) as caught:
        _require_bundle_roots(
            request,
            [replace(identities[0], kind="TrainingRunManifest")],
            [identities[0]],
            stage="selection",
        )

    message = str(caught.value)
    assert "TrainingRunManifest" in message and "EvaluationRunManifest" in message
    assert shared_id in message


def test_a_selected_root_whose_bytes_moved_refuses(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment
) -> None:
    """Same artifact, different bytes: a rerun is not the receipt the plan named."""
    target = _bound_sheaf(outputs, "sheaf-byte-swap")
    request = _bundle_request(outputs, target, environment=environment)
    identities = analysis_bundle_root_identities(request)
    assert identities is not None
    reran = replace(identities[0], manifest_sha256="f" * 64, size_bytes=1)

    with pytest.raises(ValueError) as caught:
        _require_bundle_roots(request, identities, [reran], stage="selection")

    message = str(caught.value)
    assert identities[0].manifest_sha256 in message
    assert "f" * 64 in message


def test_the_exact_root_gate_accepts_the_set_it_bound(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment
) -> None:
    """The gate is exact rather than merely strict: the bound set itself passes."""
    target = _bound_sheaf(outputs, "sheaf-agrees")
    request = _bundle_request(outputs, target, environment=environment)
    identities = analysis_bundle_root_identities(request)
    assert identities is not None

    _require_bundle_roots(request, identities, list(identities), stage="selection")


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
    execute_workflow(_closure(outputs, "montage-source"), environment=environment)
    requests = workflow_requests(
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
        workflow_requests(_closure(outputs, "sheaf-reader"), environment=environment)


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


def test_a_bundle_runs_under_an_already_resolved_staged_context(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment, calls: _Calls
) -> None:
    """A resolved context is handed to the bundle as itself, not turned back into bindings.

    The bundle used to refuse any environment declaring a staged context, on the
    grounds that a resolved context cannot be reconstructed into a descriptor
    plus root bindings. That was true and the wrong conclusion: it does not need
    to be reconstructed, it needs to be *passed*. What the bundle adds on top is
    its own root locations, because a bundle re-roles the manifests it selects
    and a staged context addresses a parent by its complete reference.
    """
    from dataclasses import replace

    from feedbax.analysis.execution_context import (
        EMPTY_STAGED_EXECUTION_CONTEXT,
        with_staged_repo_root,
    )

    target = _bound_sheaf(outputs, "sheaf-staged")
    staged = replace(
        environment,
        execution_context=with_staged_repo_root(EMPTY_STAGED_EXECUTION_CONTEXT, outputs.root),
    )
    run = _fulfill(outputs, target, environment=staged)
    assert calls.analysis == 1
    assert [receipt.node_kind for receipt in run.results[-1].receipts] == ["analysis"]


def test_a_resolved_context_and_raw_bundle_bindings_are_mutually_exclusive(
    application_registry_bundle, tmp_path
) -> None:
    """Two statements of the same bindings, one of which would silently lose."""
    from feedbax.analysis.bundles import execute_staged_analysis_bundle
    from feedbax.analysis.execution_context import (
        EMPTY_STAGED_EXECUTION_CONTEXT,
        StagedArtifactProviderRootBinding,
        StagedExecutionContextError,
    )

    with pytest.raises(StagedExecutionContextError, match="cannot be combined"):
        execute_staged_analysis_bundle(
            {"name": "irrelevant", "stages": []},
            root=tmp_path,
            execution_context=EMPTY_STAGED_EXECUTION_CONTEXT,
            artifact_provider_bindings=[StagedArtifactProviderRootBinding("results", tmp_path)],
            registries=application_registry_bundle,
        )


def test_a_bundle_stating_no_work_refuses(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment
) -> None:
    outputs.sheaf("sheaf-empty", stages=[])
    with pytest.raises(ValueError, match="neither a staged plan nor a template set"):
        _fulfill(outputs, "sheaf-empty", environment=environment)


# --------------------------------------------------------------------------
# The bundle root gate never passes by having nothing to compare
# --------------------------------------------------------------------------


def test_a_selected_root_with_no_byte_profile_refuses_against_a_bound_one(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment
) -> None:
    """The reviewer's case: an address-only selected side is not agreement.

    The bound side carries the digest the lock authenticated. A selected side
    that carries none is not a weaker match — it is a byte comparison that never
    happened, and a gate that passes for want of anything to say is a gate that
    is not there.
    """
    target = _bound_sheaf(outputs, "sheaf-profileless")
    request = _bundle_request(outputs, target, environment=environment)
    identities = analysis_bundle_root_identities(request)
    assert identities is not None
    unprofiled = replace(identities[0], manifest_sha256=None, size_bytes=None)

    with pytest.raises(ValueError) as caught:
        _require_bundle_roots(request, identities, [unprofiled], stage="prepare_workflow")

    message = str(caught.value)
    assert "no byte profile" in message
    assert identities[0].manifest_sha256 in message


def test_no_gate_over_bundle_roots_can_be_told_to_skip_the_byte_comparison(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment
) -> None:
    """The address-only escape hatch is gone from the gate's signature.

    Both gates over a bundle's roots now compare identities settled by the same
    verified read, so neither has anything to skip. A flag that could turn the
    byte comparison off is a flag that would eventually be passed.
    """
    import inspect

    signature = inspect.signature(_require_bundle_roots)
    assert "compare_profiles" not in signature.parameters

    target = _bound_sheaf(outputs, "sheaf-address-only")
    request = _bundle_request(outputs, target, environment=environment)
    identities = analysis_bundle_root_identities(request)
    assert identities is not None
    address_only = replace(identities[0], manifest_sha256=None, size_bytes=None)

    with pytest.raises(ValueError, match="no byte profile"):
        _require_bundle_roots(request, identities, [address_only], stage="selection")


def test_a_pinned_root_that_cannot_be_read_refuses(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment, tmp_path
) -> None:
    """A read failure is a refusal, never a downgrade to address-only identity."""
    from feedbax.analysis.bundles import BundleRootVerificationError, verify_bundle_roots

    target = _bound_sheaf(outputs, "sheaf-unreadable")
    request = _bundle_request(outputs, target, environment=environment)
    identities = analysis_bundle_root_identities(request)
    assert identities is not None

    empty_root = tmp_path / "no-receipts-here"
    empty_root.mkdir()
    with pytest.raises(BundleRootVerificationError) as caught:
        verify_bundle_roots(identities, root=empty_root)

    message = str(caught.value)
    assert identities[0].id in message
    assert "never a comparison that is skipped" in message


# --------------------------------------------------------------------------
# Cross-phase pinning: prepare_workflow and execution consume one proved read
# --------------------------------------------------------------------------


def _root_receipt_path(request: AnalysisBundleNodeRequest, root: Path) -> Path:
    identities = analysis_bundle_root_identities(request)
    assert identities is not None and len(identities) == 1
    return canonical_manifest_path(identities[0].kind, identities[0].id, root=root)


def test_substituting_a_root_between_preflight_and_execution_refuses(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment, calls: _Calls
) -> None:
    """The round-4 reproduction: swap a bound root's bytes, same kind and id.

    Preflight used to read and profile the root, execution used to re-select and
    re-read it, freshly authenticating whatever it found, and the gate between
    the phases compared addresses. So a substitution landing in between was
    consumed and blessed by execution, and the post-execution gate had nothing
    to say about it because the id had not changed.

    Now the root is read once and proved against the lock's pin before anything
    runs. The substitution is caught at that proof, and no stage executes.
    """
    target = _bound_sheaf(outputs, "sheaf-cross-phase")
    request = _bundle_request(outputs, target, environment=environment)
    root_path = _root_receipt_path(request, Path(environment.root))
    original = root_path.read_bytes()
    replacement = json.dumps(
        {**json.loads(original), "metadata": {"rerun": "second-pass"}}
    ).encode()
    assert hashlib.sha256(replacement).hexdigest() != hashlib.sha256(original).hexdigest()
    root_path.write_bytes(replacement)
    # Every addressing fact survives; only the bytes moved.
    substituted = load_manifest(root_path)
    assert substituted.status == "completed"

    before = calls.analysis
    with pytest.raises(BundleRootVerificationError) as caught:
        execute_analysis_bundle_node(request, environment=environment)

    message = str(caught.value)
    assert "is pinned to manifest_sha256" in message
    assert hashlib.sha256(replacement).hexdigest() in message
    assert calls.analysis == before, "no stage runs over a root that failed its pin"


def test_bytes_substituted_after_the_pin_is_proved_never_reach_what_is_emitted(
    outputs: QuillonOutputs,
    environment: FulfillmentEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reads after the proof are addressing, and addressing cannot launder bytes.

    Three reads of a bound root happen across a bundle node, and only the first
    is authentication. The other two are scans — the bundle predicate deciding
    which addresses match, and the executor's tiered lookup deciding where a
    manifest lives. Neither may become a source of identity, and this pins that
    down the only way that means anything: the spy returns the real bytes to the
    proving read and tampered bytes to every scan after it, and the node must
    either refuse or emit products recording the *proved* digest.

    What it must never do is finish while recording the tampered digest, which
    is what happened when execution re-selected and re-authenticated for itself.
    """
    target = _bound_sheaf(outputs, "sheaf-one-read")
    request = _bundle_request(outputs, target, environment=environment)
    root = Path(environment.root)
    root_path = _root_receipt_path(request, root)
    original = root_path.read_bytes()
    proved_digest = hashlib.sha256(original).hexdigest()
    tampered = json.dumps({**json.loads(original), "metadata": {"swapped": True}}).encode()
    tampered_digest = hashlib.sha256(tampered).hexdigest()
    assert tampered_digest != proved_digest

    reads: list[Path] = []
    real_read_bytes = Path.read_bytes

    def spying_read_bytes(self: Path) -> bytes:
        if self == root_path:
            reads.append(self)
            if len(reads) > 1:
                return tampered
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", spying_read_bytes)
    refusal: Exception | None = None
    receipts: tuple = ()
    try:
        receipts = execute_analysis_bundle_node(request, environment=environment)
    except Exception as exc:  # noqa: BLE001 - the refusal's type is not the point
        refusal = exc

    assert len(reads) > 1, "the fixture must actually exercise a post-proof read"
    recorded = {
        parent.metadata.get("manifest_sha256")
        for receipt in receipts
        for parent in load_manifest(receipt.path).provenance.parents
        if parent.id == request.root_inputs[0].id
    }
    # Refusing is a permitted outcome: a scan that disagrees with the proof can
    # only ever narrow what runs. Recording the tampered digest is not.
    assert tampered_digest not in recorded, (
        "a post-proof read became a source of identity: the emitted parents "
        f"record {recorded!r} rather than the proved {proved_digest!r}"
    )
    if refusal is None:
        assert recorded == {proved_digest}


def test_every_product_records_the_root_under_the_digest_the_pin_proved(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment
) -> None:
    """An untampered run binds the proved digest, not one minted during execution."""
    target = _bound_sheaf(outputs, "sheaf-proved-digest")
    request = _bundle_request(outputs, target, environment=environment)
    root = Path(environment.root)
    pinned_digest = hashlib.sha256(_root_receipt_path(request, root).read_bytes()).hexdigest()

    receipts = execute_analysis_bundle_node(request, environment=environment)

    assert receipts
    bound_digests = {
        parent.metadata.get("manifest_sha256")
        for receipt in receipts
        for parent in load_manifest(receipt.path).provenance.parents
        if parent.id == request.root_inputs[0].id
    }
    assert bound_digests == {pinned_digest}


def test_the_post_execution_gate_compares_full_identities(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment
) -> None:
    """The honored-set gate has real digests on both sides, not just addresses."""
    from feedbax.analysis.bundles import verify_bundle_roots

    target = _bound_sheaf(outputs, "sheaf-honored")
    request = _bundle_request(outputs, target, environment=environment)
    identities = analysis_bundle_root_identities(request)
    assert identities is not None

    verified = verify_bundle_roots(identities, root=Path(environment.root))
    assert [item.identity for item in verified] == list(identities)
    assert all(item.manifest_sha256 is not None for item in verified)
    # The gate the post-execution phase runs is the same exact-identity gate,
    # fed the verified records rather than bare ids.
    _require_bundle_roots(
        request, identities, [item.identity for item in verified], stage="selection"
    )


# --------------------------------------------------------------------------
# One address, one root, on every side of the gate
# --------------------------------------------------------------------------


def test_duplicate_selected_roots_refuse_before_anything_keys_them(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment
) -> None:
    """A repeated address executes twice and compares once; that gap is refused."""
    target = _bound_sheaf(outputs, "sheaf-dup-selected")
    request = _bundle_request(outputs, target, environment=environment)
    identities = analysis_bundle_root_identities(request)
    assert identities is not None

    with pytest.raises(DuplicateBundleRootError) as caught:
        _require_bundle_roots(
            request, identities, [identities[0], identities[0]], stage="prepare_workflow"
        )
    assert f"{identities[0].kind}:{identities[0].id}" in str(caught.value)

    with pytest.raises(DuplicateBundleRootError):
        _require_bundle_roots(
            request, [identities[0], identities[0]], list(identities), stage="prepare_workflow"
        )


def test_a_root_scan_that_returns_one_address_twice_refuses(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment
) -> None:
    """Selection is where a corrupt root surfaces, and it refuses there.

    A full scan reaches every manifest file under the root, so two files can
    hold one address. Downstream that expands into duplicated execution while
    every set comparison counts it once, which is why it is refused at the one
    place selection happens rather than at any of the places that consume it.
    """
    from feedbax.analysis.bundles import require_unique_root_addresses

    target = _bound_sheaf(outputs, "sheaf-dup-scan")
    request = _bundle_request(outputs, target, environment=environment)
    identities = analysis_bundle_root_identities(request)
    assert identities is not None
    manifest = load_manifest(_root_receipt_path(request, Path(environment.root)))

    require_unique_root_addresses([manifest], what="a healthy root set")
    with pytest.raises(DuplicateBundleRootError) as caught:
        require_unique_root_addresses([manifest, manifest], what="the scanned root set")
    assert "the scanned root set" in str(caught.value)


def test_verification_refuses_two_pins_at_one_address(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment
) -> None:
    from feedbax.analysis.bundles import verify_bundle_roots

    target = _bound_sheaf(outputs, "sheaf-dup-pins")
    request = _bundle_request(outputs, target, environment=environment)
    identities = analysis_bundle_root_identities(request)
    assert identities is not None

    with pytest.raises(DuplicateBundleRootError):
        verify_bundle_roots([identities[0], identities[0]], root=Path(environment.root))


def test_execution_performs_no_selection_of_its_own_over_a_pinned_root_set(
    outputs: QuillonOutputs,
    environment: FulfillmentEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The decisive structural fact: pinned execution never re-selects.

    The byte-substitution tests can be satisfied by a recipe that happens to
    choke on tampered input, so they prove the outcome without proving the
    mechanism. This proves the mechanism directly. Selection is the step that
    discarded the prepare_workflow proof and went looking for roots again; with a
    verified set in hand, execution must not reach it at all.

    One call is expected and permitted — the predicate gate in
    ``execute_analysis_bundle_node``, which is what proves the plan's root set
    is the set the bundle's own predicate selects. Any call after that one is
    execution re-selecting.
    """
    from feedbax.analysis import bundles as bundles_module

    target = _bound_sheaf(outputs, "sheaf-no-reselect")
    request = _bundle_request(outputs, target, environment=environment)

    calls_to_selection: list[tuple] = []
    real_select = bundles_module.select_bundle_manifests

    def counting_select(*args, **kwargs):
        calls_to_selection.append((args, tuple(sorted(kwargs))))
        return real_select(*args, **kwargs)

    monkeypatch.setattr(bundles_module, "select_bundle_manifests", counting_select)
    monkeypatch.setattr(
        "feedbax.analysis.fulfillment_adapters.select_bundle_manifests", counting_select
    )
    execute_analysis_bundle_node(request, environment=environment)

    assert len(calls_to_selection) == 1, (
        "execution re-selected its roots instead of consuming the verified set; "
        f"selection ran {len(calls_to_selection)} times"
    )
