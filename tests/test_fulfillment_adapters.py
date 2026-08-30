"""Executor adapters, receipt binding, and strictly serial fulfillment semantics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import plotly.graph_objects as go
import pytest

from feedbax.analysis.evaluation import EvaluationRecipeResult
from feedbax.analysis.specs import AnalysisRecipeResult
from feedbax.analysis.figures import RenderedFigure
from feedbax.analysis.fulfillment import FulfillmentAdmissionError
from feedbax.analysis.fulfillment_adapters import (
    AnalysisNodeRequest,
    EvaluationMatrixNodeRequest,
    EvaluationNodeRequest,
    FigureNodeRequest,
    FulfillmentEnvironment,
    ReportNodeRequest,
    canonical_fulfillment_order,
    expand_evaluation_matrix_node,
    fulfill_node,
    fulfill_nodes,
    receipt_exact_parent_entry,
    receipt_parent_ref,
    report_execution_spec,
    staged_exact_parents_from_receipts,
)
from feedbax.analysis.manifest_inputs import is_authenticated_manifest_ref
from feedbax.analysis.reports import (
    REPORT_RENDER_ROLE,
    ReportRecipeResult,
    execute_authored_report_spec,
)
from feedbax.contracts.manifest import (
    AnalysisRunSpec,
    EvaluationRunSpec,
    ReportSpec,
    load_manifest,
    report_manifest_id,
    store_bytes_artifact,
)
from tests.analysis_fixtures import ToyAnalysis, build_toy_analysis_data


AUTHORED_FIGURE = {
    "schema_id": "feedbax.spec.figure",
    "schema_version": "feedbax.spec.figure.v2",
    "name": "adapter-figure",
    "assembler": "feedbax.grid_figure",
}


class _Calls:
    """Counts recipe invocations so reuse can be proven, not assumed."""

    def __init__(self) -> None:
        self.evaluation = 0
        self.report = 0


@pytest.fixture
def calls() -> _Calls:
    return _Calls()


@pytest.fixture
def environment(tmp_path: Path, application_registry_bundle, calls: _Calls):
    def evaluation_recipe(run_spec, root, states_path, execution_context):
        calls.evaluation += 1
        return EvaluationRecipeResult(
            states=None,
            summary_metrics={"rows": run_spec.params.get("rows", 0)},
            metadata={"states_schema": "testpkg.states.v1"},
        )

    def report_recipe(report_spec, root, inputs):
        calls.report += 1
        artifact = store_bytes_artifact(
            f"# {report_spec.report_type}\n".encode(),
            root=root,
            role=REPORT_RENDER_ROLE,
            logical_name="report.md",
            media_type="text/markdown",
        )
        return ReportRecipeResult(artifacts=[artifact], summary={"inputs": len(inputs)})

    application_registry_bundle.evaluation_recipes.register(
        "testpkg.fulfillment", evaluation_recipe
    )
    application_registry_bundle.report_recipes.register("testpkg.fulfillment", report_recipe)
    return FulfillmentEnvironment(
        root=tmp_path / "receipts",
        registries=application_registry_bundle,
        issues=("402ffb1",),
    )


def _evaluation_node(key: str, rows: int = 1) -> EvaluationNodeRequest:
    return EvaluationNodeRequest(
        node_key=key,
        spec=EvaluationRunSpec(evaluation_type="testpkg.fulfillment", params={"rows": rows}),
    )


def test_first_run_executes_and_second_run_reuses(environment, calls: _Calls) -> None:
    node = _evaluation_node("eval/a")

    first = fulfill_node(node, environment=environment)
    assert first.disposition == "executed"
    assert calls.evaluation == 1
    assert first.receipt.path.is_file()

    second = fulfill_node(node, environment=environment)
    assert second.disposition == "reused"
    assert calls.evaluation == 1
    assert second.receipt.path == first.receipt.path
    assert second.receipt.manifest_id == first.receipt.manifest_id


def test_executed_receipt_is_re_admitted_through_the_same_validator(environment) -> None:
    result = fulfill_node(_evaluation_node("eval/a"), environment=environment)
    assert result.admissions[0].codes == ("manifest_absent",)
    assert result.admissions[-1].admitted


def test_failed_admission_refuses_instead_of_re_executing(environment, calls: _Calls) -> None:
    node = _evaluation_node("eval/a")
    receipt = fulfill_node(node, environment=environment).receipt
    assert calls.evaluation == 1

    payload = receipt.path.read_text(encoding="utf-8").replace('"completed"', '"failed"')
    receipt.path.write_text(payload, encoding="utf-8")

    with pytest.raises(FulfillmentAdmissionError) as excinfo:
        fulfill_node(node, environment=environment)
    assert "status_not_completed" in excinfo.value.outcome.codes
    assert calls.evaluation == 1, "a failed record must never be silently re-executed over"
    assert receipt.path.is_file()


def test_missing_required_output_role_refuses_before_execution(environment, calls: _Calls) -> None:
    node = _evaluation_node("eval/a")
    fulfill_node(node, environment=environment)

    demanding = EvaluationNodeRequest(
        node_key="eval/a",
        spec=node.spec,
        required_output_roles=("evaluation_states",),
    )
    with pytest.raises(FulfillmentAdmissionError) as excinfo:
        fulfill_node(demanding, environment=environment)
    assert "output_role_absent" in excinfo.value.outcome.codes
    assert calls.evaluation == 1


def test_serial_run_is_deterministic_and_reuses_on_reinvocation(
    environment, calls: _Calls
) -> None:
    nodes = [_evaluation_node("eval/c", 3), _evaluation_node("eval/a", 1), _evaluation_node(
        "eval/b", 2
    )]

    first = fulfill_nodes(nodes, environment=environment)
    assert first.execution_order == ("eval/c", "eval/a", "eval/b")
    assert first.executed == first.execution_order
    assert calls.evaluation == 3

    second = fulfill_nodes(nodes, environment=environment)
    assert second.execution_order == first.execution_order
    assert second.reused == first.execution_order
    assert second.executed == ()
    assert calls.evaluation == 3, "a fully satisfied plan executes zero nodes"
    assert [r.receipt.manifest_id for r in second.results] == [
        r.receipt.manifest_id for r in first.results
    ]


def test_declared_order_wins_and_ties_break_on_node_key(environment) -> None:
    nodes = [
        EvaluationNodeRequest(node_key="z", spec=_evaluation_node("z").spec, order=0),
        EvaluationNodeRequest(node_key="a", spec=_evaluation_node("a").spec, order=0),
        EvaluationNodeRequest(node_key="m", spec=_evaluation_node("m").spec, order=-1),
    ]
    assert [n.node_key for n in canonical_fulfillment_order(nodes)] == ["m", "a", "z"]


def test_duplicate_logical_keys_refuse(environment) -> None:
    nodes = [_evaluation_node("eval/a"), _evaluation_node("eval/a", 2)]
    with pytest.raises(ValueError, match="duplicate fulfillment node keys"):
        canonical_fulfillment_order(nodes)


def test_receipt_binding_is_mechanical_and_authenticated(environment) -> None:
    receipt = fulfill_node(_evaluation_node("eval/a"), environment=environment).receipt

    ref = receipt_parent_ref(receipt, role="evaluation_run")
    assert is_authenticated_manifest_ref(ref)
    assert ref.id == receipt.manifest_id
    assert ref.kind == "EvaluationRunManifest"
    assert ref.metadata["size_bytes"] == receipt.path.stat().st_size

    exact = staged_exact_parents_from_receipts([(receipt, "evaluation_run")])
    assert len(exact.parents) == 1
    entry = exact.parents[0]
    assert entry.parent == ref
    assert environment.root.joinpath(entry.execution_uri) == receipt.path


def test_report_node_addresses_the_substituted_exact_parent_spec(
    environment, calls: _Calls
) -> None:
    receipt = fulfill_node(_evaluation_node("eval/a"), environment=environment).receipt
    exact = staged_exact_parents_from_receipts([(receipt, "evaluation_run")])
    authored = ReportSpec(
        report_type="testpkg.fulfillment",
        inputs=[entry.parent for entry in exact.parents],
    )
    node = ReportNodeRequest(node_key="report/main", spec=authored, exact_parents=exact)

    expected_id = report_manifest_id(report_execution_spec(node))
    result = fulfill_node(node, environment=environment)
    assert result.disposition == "executed"
    assert result.receipt.manifest_id == expected_id
    assert calls.report == 1

    again = fulfill_node(node, environment=environment)
    assert again.disposition == "reused"
    assert calls.report == 1


def test_report_execution_spec_matches_the_public_authored_executor(
    environment, tmp_path: Path
) -> None:
    """The adapter's identity derivation must equal what the executor really writes."""
    receipt = fulfill_node(_evaluation_node("eval/a"), environment=environment).receipt
    exact = staged_exact_parents_from_receipts([(receipt, "evaluation_run")])
    authored = ReportSpec(
        report_type="testpkg.fulfillment",
        inputs=[entry.parent for entry in exact.parents],
    )
    node = ReportNodeRequest(node_key="report/direct", spec=authored, exact_parents=exact)

    manifest, _path = execute_authored_report_spec(
        authored,
        registry=environment.registries.report_recipes,
        exact_parents=exact,
        root=environment.root,
    )
    assert manifest.id == report_manifest_id(report_execution_spec(node))


def test_figure_node_reuses_only_for_the_same_runtime_binding(
    environment, monkeypatch: pytest.MonkeyPatch, calls: _Calls
) -> None:
    from feedbax.analysis import figures as figures_module

    monkeypatch.setattr(
        figures_module,
        "_build_figures",
        lambda *_args: [RenderedFigure(name="only", figure=go.Figure())],
    )
    receipt = fulfill_node(_evaluation_node("eval/a"), environment=environment).receipt
    bound = receipt_parent_ref(receipt, role="evaluation_run")

    node = FigureNodeRequest(
        node_key="figure/main",
        spec=AUTHORED_FIGURE,
        runtime_inputs=(bound,),
        runtime_metadata={"node": "figure/main"},
    )
    first = fulfill_node(node, environment=environment)
    assert first.disposition == "executed"
    assert fulfill_node(node, environment=environment).disposition == "reused"

    rebound = FigureNodeRequest(
        node_key="figure/main",
        spec=AUTHORED_FIGURE,
        runtime_inputs=(),
        runtime_metadata={"node": "figure/main"},
    )
    with pytest.raises(FulfillmentAdmissionError) as excinfo:
        fulfill_node(rebound, environment=environment)
    assert "runtime_inputs_mismatch" in excinfo.value.outcome.codes


def test_evaluation_matrix_node_expands_to_canonical_ordered_rows(
    environment, calls: _Calls
) -> None:
    matrix = {
        "schema_id": "feedbax.spec.evaluation_run_matrix",
        "schema_version": "feedbax.spec.evaluation_run_matrix.v3",
        "base": {
            "schema_id": "feedbax.spec.evaluation_run",
            "schema_version": "feedbax.spec.evaluation_run.v1",
            "evaluation_type": "testpkg.fulfillment",
            "params": {"rows": 0},
        },
        "rows": [
            {"row_id": "r1", "deltas": [{"path": "params.rows", "value": 1}]},
            {"row_id": "r2", "deltas": [{"path": "params.rows", "value": 2}]},
        ],
    }
    node = EvaluationMatrixNodeRequest(node_key="eval/matrix", matrix=matrix)

    expanded = expand_evaluation_matrix_node(node, environment=environment)
    assert [request.node_key for request in expanded] == ["eval/matrix#r1", "eval/matrix#r2"]

    first = fulfill_node(node, environment=environment)
    assert first.disposition == "executed"
    assert len(first.receipts) == 2
    assert calls.evaluation == 2
    for receipt in first.receipts:
        assert receipt.path.is_file()
        assert load_manifest(receipt.path).status == "completed"

    second = fulfill_node(node, environment=environment)
    assert second.disposition == "reused"
    assert calls.evaluation == 2


def test_unknown_node_request_type_refuses(environment) -> None:
    with pytest.raises(TypeError, match="unknown fulfillment node request type"):
        fulfill_node(object(), environment=environment)  # type: ignore[arg-type]


def test_analysis_node_request_carries_its_node_kind() -> None:
    request = AnalysisNodeRequest(
        node_key="analysis/a",
        spec=AnalysisRunSpec(analysis_type="testpkg.fulfillment"),
    )
    assert request.node_kind == "analysis"


def test_analysis_node_executes_then_reuses(environment) -> None:
    analysis_calls = {"count": 0}

    def analysis_recipe(_spec, _root, _inputs, _execution_context):
        analysis_calls["count"] += 1
        return AnalysisRecipeResult(
            analyses={"toy": ToyAnalysis(variant="toy", cache_result=True)},
            data=build_toy_analysis_data(value=3),
        )

    environment.registries.analysis_recipes.register("testpkg.fulfillment", analysis_recipe)
    node = AnalysisNodeRequest(
        node_key="analysis/main",
        spec=AnalysisRunSpec(analysis_type="testpkg.fulfillment", params={"k": 1}),
    )

    first = fulfill_node(node, environment=environment)
    assert first.disposition == "executed"
    assert analysis_calls["count"] == 1
    assert load_manifest(first.receipt.path).status == "completed"

    second = fulfill_node(node, environment=environment)
    assert second.disposition == "reused"
    assert analysis_calls["count"] == 1
    assert second.receipt.path == first.receipt.path


# --------------------------------------------------------------------------
# A receipt carries the identity admission settled, and never re-derives it
# --------------------------------------------------------------------------


def test_a_receipt_carries_the_profile_of_the_bytes_that_were_admitted(
    environment,
) -> None:
    receipt = fulfill_node(_evaluation_node("eval/profile"), environment=environment).receipt
    raw = receipt.path.read_bytes()

    assert receipt.manifest_sha256 == hashlib.sha256(raw).hexdigest()
    assert receipt.size_bytes == len(raw)
    ref = receipt_parent_ref(receipt, role="evaluation_run")
    assert ref.metadata["manifest_sha256"] == receipt.manifest_sha256
    assert ref.metadata["size_bytes"] == receipt.size_bytes


def test_binding_after_a_byte_substitution_emits_the_admitted_digest(
    environment, tmp_path: Path
) -> None:
    """The reviewer's reproduction: admit, substitute, bind.

    The replacement is the same kind, the same id, and still completed, so every
    addressing fact about it is intact. Only its bytes differ. Binding must
    restate the digest admission authenticated — which no longer matches the
    file, and should not, because the file is no longer what was admitted — and
    must never mint the replacement's own digest, which would bless bytes that
    were never admitted with a profile produced by the authentication step.
    """
    result = fulfill_node(_evaluation_node("eval/substituted"), environment=environment)
    receipt = result.receipt
    admitted_digest = receipt.manifest_sha256
    original = receipt.path.read_bytes()

    replacement = json.dumps(
        {**json.loads(original), "metadata": {"rerun": "second-pass"}}
    ).encode()
    assert hashlib.sha256(replacement).hexdigest() != admitted_digest
    receipt.path.write_bytes(replacement)
    # Every addressing fact survives the substitution; only the bytes changed.
    substituted = load_manifest(receipt.path)
    assert substituted.kind == receipt.manifest_kind
    assert substituted.id == receipt.manifest_id
    assert substituted.status == "completed"

    ref = receipt_parent_ref(receipt, role="evaluation_run")
    entry = receipt_exact_parent_entry(receipt, role="evaluation_run")

    assert ref.metadata["manifest_sha256"] == admitted_digest
    assert ref.metadata["manifest_sha256"] != hashlib.sha256(replacement).hexdigest()
    assert entry.parent == ref

    # Restore, so no other assertion in this module observes the substitution.
    receipt.path.write_bytes(original)


def test_a_receipt_cannot_be_bound_from_an_admission_that_read_nothing() -> None:
    """An in-memory admission authenticates no stored bytes, so it binds none."""
    from feedbax.analysis.fulfillment import AdmittedManifestBytes

    with pytest.raises(RuntimeError, match="no manifest bytes were captured"):
        AdmittedManifestBytes().receipt(node_kind="evaluation", root=Path("/nowhere"))


def test_a_receipt_cannot_be_constructed_with_a_digest_nobody_measured() -> None:
    """Verified construction is enforced, not conventional.

    A receipt built by hand with a plausible-looking digest is indistinguishable
    downstream from one built from real bytes: every forward binding made from
    it is authenticated on the strength of that number. So the constructor
    demands a capability token that only the verified constructors mint, and the
    only way to mint one is to hash bytes that were actually read.
    """
    from feedbax.analysis.fulfillment import FulfillmentReceipt

    with pytest.raises(TypeError, match="built from bytes that were actually read"):
        FulfillmentReceipt(
            node_kind="evaluation",
            manifest=object(),
            path=Path("/nowhere/manifest.json"),
            root=Path("/nowhere"),
            manifest_sha256="a" * 64,
            size_bytes=123,
        )
    # A token is not a bearer credential for any profile: it carries the one it
    # measured, so it cannot be paired with a digest from somewhere else.
    with pytest.raises(TypeError, match="built from bytes that were actually read"):
        FulfillmentReceipt(
            node_kind="evaluation",
            manifest=object(),
            path=Path("/nowhere/manifest.json"),
            root=Path("/nowhere"),
            manifest_sha256="a" * 64,
            size_bytes=123,
            verified=object(),
        )


def test_a_verified_token_cannot_be_paired_with_a_foreign_profile(
    environment,
) -> None:
    """The token names the profile it measured, so the two cannot be mismatched."""
    from dataclasses import replace as dataclass_replace

    from feedbax.analysis.fulfillment import FulfillmentReceipt

    receipt = fulfill_node(_evaluation_node("eval/token"), environment=environment).receipt
    with pytest.raises(ValueError, match="not the profile the verified read settled"):
        dataclass_replace(receipt, manifest_sha256="b" * 64)


# --------------------------------------------------------------------------
# The node's own context outranks the environment's, and rows inherit it
# --------------------------------------------------------------------------


def test_a_request_context_outranks_the_environment_and_absence_falls_back(
    environment, tmp_path: Path
) -> None:
    """A hand-assembled request carries no context and takes the environment's.

    A lowered one carries the context the closure resolved for exactly that
    node, and it is that context — not the run-wide one — that execution,
    admission, rebuild, and repair all see.
    """
    from dataclasses import replace

    from feedbax.analysis.execution_context import (
        EMPTY_STAGED_EXECUTION_CONTEXT,
        with_staged_repo_root,
    )
    from feedbax.analysis.fulfillment_adapters import node_execution_context

    request = EvaluationNodeRequest(
        node_key="evaluation:x",
        spec=EvaluationRunSpec(evaluation_type="testpkg.absent"),
    )
    assert node_execution_context(request, environment) is EMPTY_STAGED_EXECUTION_CONTEXT

    per_node_repo = tmp_path / "per-node-repo"
    per_node_repo.mkdir()
    run_wide = with_staged_repo_root(EMPTY_STAGED_EXECUTION_CONTEXT, tmp_path)
    per_node = with_staged_repo_root(EMPTY_STAGED_EXECUTION_CONTEXT, per_node_repo)
    staged = replace(environment, execution_context=run_wide)
    assert node_execution_context(request, staged) is run_wide
    assert (
        node_execution_context(replace(request, execution_context=per_node), staged)
        is per_node
    )


def test_a_matrix_row_inherits_the_matrix_node_context(environment, tmp_path: Path) -> None:
    """A matrix does not execute; its rows do, under the context it resolved."""
    from feedbax.analysis.execution_context import (
        EMPTY_STAGED_EXECUTION_CONTEXT,
        with_staged_repo_root,
    )

    repo = tmp_path / "matrix-repo"
    repo.mkdir()
    context = with_staged_repo_root(EMPTY_STAGED_EXECUTION_CONTEXT, repo)
    request = EvaluationMatrixNodeRequest(
        node_key="eval/context-matrix",
        matrix={
            "schema_id": "feedbax.spec.evaluation_run_matrix",
            "schema_version": "feedbax.spec.evaluation_run_matrix.v3",
            "base": {
                "schema_id": "feedbax.spec.evaluation_run",
                "schema_version": "feedbax.spec.evaluation_run.v1",
                "evaluation_type": "testpkg.fulfillment",
                "params": {"rows": 0},
            },
            "rows": [
                {"row_id": "r1", "deltas": [{"path": "params.rows", "value": 1}]},
                {"row_id": "r2", "deltas": [{"path": "params.rows", "value": 2}]},
            ],
        },
        execution_context=context,
    )
    rows = expand_evaluation_matrix_node(request, environment=environment)
    assert [row.node_key for row in rows] == [
        "eval/context-matrix#r1",
        "eval/context-matrix#r2",
    ]
    assert all(row.execution_context is context for row in rows)
