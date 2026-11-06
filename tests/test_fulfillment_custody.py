"""Transactional repair and rebuild-as-verification for fulfillment receipts."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from feedbax.analysis.evaluation import EvaluationRecipeResult
from feedbax.analysis.execution_context import (
    StagedParentExecutionLocation,
    with_staged_parent_execution_locations,
)
from feedbax.analysis.manifest_inputs import authenticated_manifest_ref
from feedbax.analysis.fulfillment import (
    FulfillmentAdmissionError,
    artifact_bytes_path,
    receipt_path,
)
from feedbax.analysis.fulfillment_adapters import (
    AnalysisNodeRequest,
    EvaluationNodeRequest,
    FulfillmentEnvironment,
    ReportNodeRequest,
    execute_node,
    fulfill_node,
    staged_exact_parents_from_receipts,
)
from feedbax.analysis.fulfillment_custody import (
    FULFILLMENT_PROJECTION_SCHEMA_ID,
    FULFILLMENT_PROJECTION_SCHEMA_VERSION,
    FULFILLMENT_QUARANTINE_DIRECTORY,
    FULFILLMENT_REPAIR_DIRECTORY,
    FULFILLMENT_REPAIR_SCHEMA_ID,
    FULFILLMENT_REPAIR_SCHEMA_VERSION,
    FulfillmentDriftError,
    FulfillmentRepairError,
    OutputProjection,
    compare_output_projections,
    migrate_output_projection,
    migrate_repair_record,
    output_projection,
    rebuild_node,
    rebuild_nodes,
    repair_node,
    require_no_drift,
    shadow_custody,
)
from feedbax.contracts.base import Provenance
from feedbax.contracts.artifact_store import store_bytes_artifact
from feedbax.contracts.manifest import (
    AnalysisRunSpec,
    EvaluationRunManifest,
    EvaluationRunSpec,
    ReportSpec,
    evaluation_run_manifest_id,
    evaluation_states_cache_path,
    load_manifest,
    spec_payload,
)
from feedbax.analysis.reports import REPORT_RENDER_ROLE, ReportRecipeResult
from feedbax.analysis.specs import AnalysisRecipeResult
from tests.support.analysis_fixtures import ToyAnalysis, build_toy_analysis_data


class _Recipes:
    """Counts recipe invocations and controls the bytes they emit."""

    def __init__(self) -> None:
        self.evaluation = 0
        self.report = 0
        self.analysis = 0
        self.payload = "baseline"
        self.analysis_value = 3


@pytest.fixture
def recipes() -> _Recipes:
    return _Recipes()


@pytest.fixture
def environment(tmp_path: Path, application_registry_bundle, recipes: _Recipes):
    def evaluation_recipe(run_spec, root, states_path, execution_context):
        recipes.evaluation += 1
        artifact = store_bytes_artifact(
            f"{recipes.payload}:{run_spec.params.get('rows', 0)}\n".encode(),
            root=root,
            role="evaluation_states",
            logical_name="states.bin",
        )
        return EvaluationRecipeResult(
            states=None,
            summary_metrics={
                "rows": run_spec.params.get("rows", 0),
                "payload": recipes.payload,
            },
            artifacts=[artifact],
            metadata={"states_schema": "testpkg.states.v1"},
        )

    def report_recipe(report_spec, root, inputs):
        recipes.report += 1
        artifact = store_bytes_artifact(
            f"# {report_spec.report_type}\n".encode(),
            root=root,
            role=REPORT_RENDER_ROLE,
            logical_name="report.md",
            media_type="text/markdown",
        )
        return ReportRecipeResult(artifacts=[artifact], summary={"inputs": len(inputs)})

    def stateful_evaluation_recipe(run_spec, root, states_path, execution_context):
        recipes.evaluation += 1
        return EvaluationRecipeResult(
            states={"trajectory": [1, 2, 3]},
            summary_metrics={
                "rows": run_spec.params.get("rows", 0),
                "payload": recipes.payload,
            },
            metadata={"states_schema": "testpkg.states.v1"},
        )

    def analysis_recipe(_run_spec, _root, _inputs, _execution_context):
        recipes.analysis += 1
        return AnalysisRecipeResult(
            analyses={"toy": ToyAnalysis(variant="toy")},
            data=build_toy_analysis_data(value=recipes.analysis_value),
        )

    application_registry_bundle.evaluation_recipes.register("testpkg.custody", evaluation_recipe)
    application_registry_bundle.evaluation_recipes.register(
        "testpkg.custody.stateful", stateful_evaluation_recipe
    )
    application_registry_bundle.report_recipes.register("testpkg.custody", report_recipe)
    application_registry_bundle.analysis_recipes.register(
        "testpkg.custody", analysis_recipe, computation_revision="test-v1"
    )
    return FulfillmentEnvironment(
        root=tmp_path / "receipts",
        registries=application_registry_bundle,
        issues=("402ffb1",),
    )


def _evaluation_node(key: str = "eval/a", rows: int = 1) -> EvaluationNodeRequest:
    return EvaluationNodeRequest(
        node_key=key,
        spec=EvaluationRunSpec(evaluation_type="testpkg.custody", params={"rows": rows}),
    )


def _stateful_evaluation_node(
    key: str = "eval/stateful",
    *,
    states_custody: str = "cache",
) -> EvaluationNodeRequest:
    """A node whose recipe returns states, so the states cache is populated."""
    return EvaluationNodeRequest(
        node_key=key,
        spec=EvaluationRunSpec(
            evaluation_type="testpkg.custody.stateful",
            params={"rows": 1, "states_custody": states_custody},
        ),
    )


def _analysis_node(key: str = "analysis/a") -> AnalysisNodeRequest:
    return AnalysisNodeRequest(
        node_key=key,
        spec=AnalysisRunSpec(analysis_type="testpkg.custody", params={"k": 1}),
    )


def _report_node(environment, key: str = "report/main") -> ReportNodeRequest:
    receipt = fulfill_node(_evaluation_node(), environment=environment).receipt
    exact = staged_exact_parents_from_receipts([(receipt, "evaluation_run")])
    authored = ReportSpec(
        report_type="testpkg.custody",
        inputs=[entry.parent for entry in exact.parents],
    )
    return ReportNodeRequest(node_key=key, spec=authored, exact_parents=exact)


def _stored_artifact_path(environment, node: EvaluationNodeRequest) -> Path:
    manifest = load_manifest(
        receipt_path(
            "evaluation",
            evaluation_run_manifest_id(node.spec),
            root=environment.root,
        )
    )
    path = artifact_bytes_path(manifest.artifacts[0], root=Path(environment.root))
    assert path is not None
    return path


def _tamper(path: Path, data: bytes) -> None:
    path.chmod(0o600)
    path.write_bytes(data)


# --- Rebuild as verification -------------------------------------------------


def test_rebuild_over_an_intact_tree_reports_zero_drift_and_preserves_receipts(
    environment, recipes: _Recipes
) -> None:
    report = _report_node(environment)
    evaluation = _evaluation_node()
    fulfill_node(report, environment=environment)
    receipts = {
        node.node_key: fulfill_node(node, environment=environment).receipt
        for node in (evaluation, report)
    }
    before = {key: receipt.path.read_bytes() for key, receipt in receipts.items()}
    executions = (recipes.evaluation, recipes.report)

    run = rebuild_nodes([evaluation, report], environment=environment)

    assert run.verification_order == ("eval/a", "report/main")
    assert run.drifted == ()
    assert all(outcome.matched for outcome in run.outcomes)
    assert [outcome.differences for outcome in run.outcomes] == [[], []]
    assert {key: receipt.path.read_bytes() for key, receipt in receipts.items()} == before
    assert (recipes.evaluation, recipes.report) == (
        executions[0] + 1,
        executions[1] + 1,
    ), "a rebuild executes each node exactly once, in shadow custody"


@pytest.mark.parametrize("states_custody", ["cache", "durable"])
def test_intact_evaluation_rebuild_re_executes_past_the_states_cache(
    environment, recipes: _Recipes, states_custody: str
) -> None:
    """An evaluation rebuild runs the recipe again even with states cached.

    ``execute_evaluation_run_spec`` serves cached states by default and then
    takes the previously completed manifest's summary metrics and artifacts
    instead of the recipe's. The shadow root mirrors ``cache/`` along with every
    other readable entry, so that path would reproduce the receipt from itself.
    Both states custody modes are covered: ``durable`` additionally republishes
    the states artifact into the shadow root, which must not collide with the
    mirrored copy.
    """
    node = _stateful_evaluation_node(states_custody=states_custody)
    receipt = fulfill_node(node, environment=environment).receipt
    assert (Path(environment.root) / "cache" / "states").is_dir()
    before = receipt.path.read_bytes()
    executions = recipes.evaluation

    outcome = rebuild_node(node, environment=environment)

    assert recipes.evaluation == executions + 1, (
        "a rebuild must re-execute the evaluation recipe, not replay its states cache"
    )
    assert outcome.matched
    assert outcome.differences == []
    assert receipt.path.read_bytes() == before


def test_perturbed_evaluation_recipe_drifts_despite_a_populated_states_cache(
    environment, recipes: _Recipes
) -> None:
    """A changed evaluation recipe is caught even though its states are cached."""
    node = _stateful_evaluation_node()
    receipt = fulfill_node(node, environment=environment).receipt
    before = receipt.path.read_bytes()

    recipes.payload = "perturbed"
    outcome = rebuild_node(node, environment=environment)

    assert not outcome.matched
    assert outcome.field_paths == ("summaries.summary_metrics.payload",)
    assert outcome.differences[0].expected == "baseline"
    assert outcome.differences[0].observed == "perturbed"
    assert receipt.path.read_bytes() == before


def test_intact_analysis_rebuild_re_executes_the_recipe_and_reports_zero_drift(
    environment, recipes: _Recipes
) -> None:
    """An analysis rebuild really runs the recipe again before comparing.

    ``execute_analysis_run_spec`` short-circuits by default on any completed
    manifest with the same id found beneath its root. A shadow root mirrors
    ``manifests/`` — dependent nodes must resolve their authenticated parents
    there — so that short-circuit would hand the rebuild its own authoritative
    receipt and every analysis node would match vacuously. The recipe counter is
    the proof that this rebuild executed rather than returning the mirror.
    """
    analysis = _analysis_node()
    receipt = fulfill_node(analysis, environment=environment).receipt
    before = receipt.path.read_bytes()
    executions = recipes.analysis

    outcome = rebuild_node(analysis, environment=environment)

    assert recipes.analysis == executions + 1, (
        "a rebuild must re-execute the analysis recipe, not return the mirrored receipt"
    )
    assert outcome.matched
    assert outcome.differences == []
    assert outcome.manifest_id == load_manifest(receipt.path).id
    assert receipt.path.read_bytes() == before


def test_perturbed_analysis_recipe_drifts_and_leaves_the_original_receipt_intact(
    environment, recipes: _Recipes
) -> None:
    """A changed analysis recipe output is reported as drift, naming the fields."""
    analysis = _analysis_node()
    receipt = fulfill_node(analysis, environment=environment).receipt
    before = receipt.path.read_bytes()
    executions = recipes.analysis

    recipes.analysis_value += 1
    outcome = rebuild_node(analysis, environment=environment)

    assert recipes.analysis == executions + 1
    assert not outcome.matched
    assert outcome.field_paths == ("artifacts[0].sha256",)
    difference = outcome.differences[0]
    assert difference.expected == outcome.expected.artifacts[0].sha256
    assert difference.observed == outcome.observed.artifacts[0].sha256
    assert difference.expected != difference.observed
    assert outcome.describe() == (
        "analysis node analysis/a drifted: artifacts[0].sha256"
    )
    assert receipt.path.read_bytes() == before

    with pytest.raises(FulfillmentDriftError) as excinfo:
        rebuild_nodes([analysis], environment=environment)
    assert excinfo.value.drifted[0].field_paths == outcome.field_paths


def test_execute_node_for_an_analysis_never_returns_a_pre_existing_receipt(
    environment, recipes: _Recipes
) -> None:
    """The adapter path executes unconditionally even with a receipt in place.

    This is ``execute_node``'s contract stated directly against the analysis
    adapter: reuse is the caller's decision, taken through ``admit_node``.
    """
    analysis = _analysis_node()
    fulfill_node(analysis, environment=environment)
    executions = recipes.analysis

    _manifest, path = execute_node(analysis, environment=environment)
    assert recipes.analysis == executions + 1

    _again, again_path = execute_node(analysis, environment=environment)
    assert recipes.analysis == executions + 2
    assert again_path == path


@pytest.mark.parametrize("authority", ["local", "external", "system_alias"])
def test_rebuild_stages_original_exact_parent_and_root_local_artifact(
    environment, recipes: _Recipes, tmp_path: Path, authority: str
) -> None:
    import sys

    external = authority == "external"
    parent_root = tmp_path / "retained" if external else environment.root
    if authority == "system_alias":
        private_var = Path("/private/var")
        if sys.platform != "darwin" or not parent_root.is_relative_to(private_var):
            pytest.skip("this case requires a fixture under Darwin's /var system alias")
        parent_root = Path("/var") / parent_root.relative_to(private_var)
    evaluation = _evaluation_node()
    parent = fulfill_node(evaluation, environment=environment).receipt
    local_path = parent.path
    relative = local_path.relative_to(environment.root)
    raw = b"  " + json.dumps(
        json.loads(local_path.read_bytes()), separators=(",", ":"), sort_keys=True
    ).encode() + b"\n\n"
    original = tmp_path / "retained" / relative if external else local_path
    original.parent.mkdir(parents=True, exist_ok=True)
    original.write_bytes(raw)
    artifact_path = _stored_artifact_path(environment, evaluation)
    artifact_relative = artifact_path.relative_to(environment.root)
    if external:
        local_path.unlink()
    ref = authenticated_manifest_ref(parent.manifest, original, "evaluation_run")
    unrelated = environment.root / "unrelated"
    unrelated.write_bytes(b"private unrelated bytes")
    observed_roots: list[Path] = []

    def report_recipe(_spec, root, inputs):
        observed_roots.append(root)
        assert inputs[0].path.read_bytes() == raw
        assert (root / artifact_relative).read_bytes() == b"baseline:1\n"
        if root != environment.root:
            assert not (root / "unrelated").exists()
            assert (root / relative).exists() is not external
            assert inputs[0].path == (original if external else root / relative)
        artifact = store_bytes_artifact(
            b"# Report\n", root=root, role=REPORT_RENDER_ROLE, logical_name="report.md",
            media_type="text/markdown",
        )
        return ReportRecipeResult(
            artifacts=[artifact], summary={"value": inputs[0].manifest.summary_metrics["payload"]}
        )

    environment.registries.report_recipes.register("testpkg.readparent", report_recipe)
    request = ReportNodeRequest(
        node_key="report/original",
        spec=ReportSpec(report_type="testpkg.readparent", inputs=[ref]),
    )
    context = with_staged_parent_execution_locations(
        environment.execution_context,
        [StagedParentExecutionLocation(ref, parent_root, relative.as_posix())],
    )
    request = replace(request, execution_context=context)
    fulfill_node(request, environment=environment)
    recipes.payload = "drifted"

    if external:
        assert rebuild_node(request, environment=environment).matched
    else:
        with pytest.raises(FulfillmentDriftError) as excinfo:
            rebuild_nodes([evaluation, request], environment=environment)
        evaluation_outcome, report_outcome = excinfo.value.outcomes
        assert not evaluation_outcome.matched
        assert report_outcome.matched

    assert observed_roots[0] == environment.root
    assert observed_roots[1] != environment.root
    assert original.read_bytes() == raw
    assert artifact_path.read_bytes() == b"baseline:1\n"


def test_rebuild_refuses_before_opening_workspace(environment, monkeypatch) -> None:
    from feedbax.analysis import fulfillment_custody

    def unexpected_workspace(*args, **kwargs):
        pytest.fail("a refused receipt must not allocate a candidate workspace")

    monkeypatch.setattr(fulfillment_custody, "shadow_custody", unexpected_workspace)
    with pytest.raises(FulfillmentAdmissionError):
        rebuild_node(_evaluation_node(), environment=environment)


@pytest.mark.parametrize("state_source", ["cache", "durable", "recomputed"])
def test_analysis_rebuild_stages_evaluation_states(
    environment, recipes: _Recipes, state_source: str
) -> None:
    import jax.tree_util as jtu

    evaluation = _stateful_evaluation_node(
        states_custody="durable" if state_source == "durable" else "cache"
    )
    parent = fulfill_node(evaluation, environment=environment).receipt
    cache = evaluation_states_cache_path(parent.manifest_id, root=environment.root)
    if state_source == "recomputed":
        cache.unlink()

    def analysis_recipe(_spec, _root, inputs, _context):
        assert list(inputs[0].states["trajectory"]) == [1, 2, 3]
        return AnalysisRecipeResult(
            analyses={"toy": ToyAnalysis(variant="toy")},
            data=build_toy_analysis_data(value=3),
        )

    environment.registries.analysis_recipes.register(
        "testpkg.readstates",
        analysis_recipe,
        computation_revision="test-v1",
        evaluation_states_structure=lambda _manifest: jtu.tree_structure(
            {"trajectory": [1, 2, 3]}
        ),
    )
    request = AnalysisNodeRequest(
        node_key="analysis/stateful",
        spec=AnalysisRunSpec(
            analysis_type="testpkg.readstates",
            inputs=[authenticated_manifest_ref(parent.manifest, parent.path, "evaluation_run")],
            evaluation_states_policy="require_durable" if state_source == "durable" else "recompute",
        ),
    )
    fulfill_node(request, environment=environment)
    if state_source == "recomputed":
        cache.unlink()
    before = parent.path.read_bytes()
    executions = recipes.evaluation

    assert rebuild_node(request, environment=environment).matched
    assert recipes.evaluation == executions + (state_source == "recomputed")
    assert parent.path.read_bytes() == before
    assert cache.is_file() is (state_source != "recomputed")


def test_rebuilt_report_projection_carries_artifacts_and_recorded_summaries(
    environment,
) -> None:
    report = _report_node(environment)
    fulfill_node(report, environment=environment)

    outcome = rebuild_node(report, environment=environment)

    assert outcome.matched
    assert require_no_drift(outcome) is outcome
    assert [artifact.role for artifact in outcome.expected.artifacts] == [REPORT_RENDER_ROLE]
    assert outcome.expected.artifacts[0].media_type == "text/markdown"
    assert outcome.expected.summaries == {"metadata_summary": {"inputs": 1}}
    assert outcome.expected.manifest_id == outcome.observed.manifest_id


@pytest.mark.parametrize("cas", [False, True])
def test_figure_rebuild_stages_declared_root_local_piece(environment, cas: bool) -> None:
    import hashlib

    from feedbax.analysis.fulfillment_adapters import FigureNodeRequest
    from feedbax.contracts.base import ArtifactRef
    from feedbax.contracts.figures import FigurePiece, FigureSpec
    from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
    from feedbax.plot.constructors import register_figure_piece

    root = environment.root
    root.mkdir()
    raw = b'{"x":[0,1,2],"y":[[1,2,3],[2,3,4]]}'
    if cas:
        artifact = ImmutableArtifactBlobProvider(root).store_bytes(
            raw, role="figure_piece", logical_name="piece.json", media_type="application/json"
        )
    else:
        (root / "piece.json").write_bytes(raw)
        artifact = ArtifactRef(
            role="figure_piece",
            logical_name="piece.json",
            media_type="application/json",
            uri="piece.json",
            sha256=hashlib.sha256(raw).hexdigest(),
            size_bytes=len(raw),
        )
    piece = FigurePiece(
        name="testpkg.custody_piece",
        label="Piece",
        description="Root-local rebuild input",
        artifact_ref=artifact,
        constructor="feedbax.profile_band",
    )
    register_figure_piece(piece, registry=environment.registries.figures)
    request = FigureNodeRequest(
        node_key="figure/piece",
        spec=FigureSpec(
            name="custody-piece", assembler="feedbax.grid_figure", pieces=[piece.name]
        ),
    )

    fulfill_node(request, environment=environment)
    assert rebuild_node(request, environment=environment).matched


def test_intact_figure_rebuild_reports_no_drift(
    environment, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An intact figure node rebuilds with zero drift, sidecar bytes included.

    A figure's runtime inputs resolve in the shadow root exactly as they do in
    the authoritative root, and every artifact a figure execution stores is now
    byte-reproducible: the rendered figure, the Plotly JSON, and the
    ``figure_spec`` sidecar. The sidecar became reproducible when
    ``save_figure_with_spec`` stopped stamping it with the wall clock; nothing
    is excluded from the projection to achieve this, so the comparison stays
    sensitive to real changes in the same bytes. The original receipt is
    untouched.
    """
    import plotly.graph_objects as go

    from feedbax.analysis import figures as figures_module
    from feedbax.analysis.figures import FIGURE_RENDER_ROLE, FIGURE_SPEC_ROLE, RenderedFigure
    from feedbax.analysis.fulfillment_adapters import FigureNodeRequest, receipt_parent_ref

    monkeypatch.setattr(
        figures_module,
        "_build_figures",
        lambda *_args: [RenderedFigure(name="only", figure=go.Figure())],
    )
    receipt = fulfill_node(_evaluation_node(), environment=environment).receipt
    node = FigureNodeRequest(
        node_key="figure/main",
        spec={
            "schema_id": "feedbax.spec.figure",
            "schema_version": "feedbax.spec.figure.v2",
            "name": "custody-figure",
            "assembler": "feedbax.grid_figure",
        },
        runtime_inputs=(receipt_parent_ref(receipt, role="evaluation_run"),),
        runtime_metadata={"node": "figure/main"},
    )
    receipt_bytes = fulfill_node(node, environment=environment).receipt.path.read_bytes()
    outcome = rebuild_node(node, environment=environment)

    assert outcome.node_kind == "figure"
    assert outcome.expected.manifest_id == outcome.observed.manifest_id
    assert outcome.matched
    assert outcome.field_paths == ()
    assert outcome.differences == []
    assert require_no_drift(outcome) is outcome
    assert outcome.expected.artifacts[0].role == FIGURE_SPEC_ROLE
    assert outcome.expected.artifacts[1].role == FIGURE_RENDER_ROLE
    assert outcome.expected.artifacts == outcome.observed.artifacts, (
        "every figure artifact reproduces byte-for-byte, sidecar included"
    )
    assert (
        fulfill_node(node, environment=environment).receipt.path.read_bytes() == receipt_bytes
    ), "a verifying rebuild leaves the authoritative receipt untouched"


def test_altered_stored_bytes_refuse_admission_before_any_rebuild(
    environment, recipes: _Recipes
) -> None:
    """Storage corruption is custody loss, not drift: it never reaches comparison."""
    node = _evaluation_node()
    fulfill_node(node, environment=environment)
    _tamper(_stored_artifact_path(environment, node), b"tampered custody bytes\n")
    executions = recipes.evaluation

    with pytest.raises(FulfillmentAdmissionError) as excinfo:
        rebuild_node(node, environment=environment)

    assert "artifact_sha256_mismatch" in excinfo.value.outcome.codes
    assert recipes.evaluation == executions, "a corrupt receipt must not be rebuilt for comparison"


def test_perturbed_executor_output_drifts_with_the_original_receipt_intact(
    environment, recipes: _Recipes
) -> None:
    node = _evaluation_node()
    receipt = fulfill_node(node, environment=environment).receipt
    before = receipt.path.read_bytes()
    stored_bytes = _stored_artifact_path(environment, node).read_bytes()
    recipes.payload = "perturbed"

    with pytest.raises(FulfillmentDriftError) as excinfo:
        rebuild_nodes([node], environment=environment)

    outcome = excinfo.value.drifted[0]
    assert not outcome.matched
    assert outcome.node_key == "eval/a"
    assert set(outcome.field_paths) == {
        "artifacts[0].sha256",
        "artifacts[0].size_bytes",
        "summaries.summary_metrics.payload",
    }
    payload_drift = next(
        d for d in outcome.differences if d.field_path == "summaries.summary_metrics.payload"
    )
    assert (payload_drift.expected, payload_drift.observed) == ("baseline", "perturbed")
    assert receipt.path.read_bytes() == before
    assert _stored_artifact_path(environment, node).read_bytes() == stored_bytes


def test_two_rebuilds_of_the_same_intact_node_produce_identical_projections(
    environment,
) -> None:
    node = _evaluation_node()
    fulfill_node(node, environment=environment)

    first = rebuild_node(node, environment=environment)
    second = rebuild_node(node, environment=environment)

    assert first.expected == second.expected
    assert first.observed == second.observed
    assert first == second


def test_retained_shadow_custody_is_reusable_and_never_authoritative(
    environment, tmp_path: Path
) -> None:
    node = _evaluation_node()
    fulfill_node(node, environment=environment)
    shadow_root = tmp_path / "shadow"

    with shadow_custody(environment, shadow_root=shadow_root, retain=True) as shadow:
        assert list(shadow.root.iterdir()) == []
        assert rebuild_node(node, environment=environment, shadow=shadow).matched
        assert (shadow.root / "manifests").is_dir()
        assert shadow.root == shadow_root
        assert shadow.environment.root == shadow_root
        assert shadow.environment.registries is environment.registries
    assert shadow_root.is_dir(), "an explicitly retained shadow root survives the block"

    with pytest.raises(ValueError, match="must not live inside the authoritative root"):
        with shadow_custody(environment, shadow_root=environment.root / "inner"):
            pass
    with pytest.raises(ValueError, match="must not contain the authoritative root"):
        with shadow_custody(environment, shadow_root=environment.root.parent):
            pass


def test_a_shadow_root_symlinked_into_authoritative_custody_is_refused(
    environment, tmp_path: Path
) -> None:
    """Containment is physical: a sibling-looking symlink into the root is refused.

    A lexical check passes ``tmp_path/shadow-link`` as a sibling of the receipt
    root while every write through it lands inside authoritative custody, so the
    shadow would overwrite the very bytes it exists not to touch.
    """
    node = _evaluation_node()
    fulfill_node(node, environment=environment)
    inside = Path(environment.root) / "captured-shadow"
    inside.mkdir(parents=True)
    link = tmp_path / "shadow-link"
    link.symlink_to(inside, target_is_directory=True)

    with pytest.raises(ValueError, match="must not live inside the authoritative root"):
        with shadow_custody(environment, shadow_root=link):
            pass

    assert sorted(entry.name for entry in inside.iterdir()) == [], (
        "a refused shadow never seeded anything inside authoritative custody"
    )


@pytest.mark.parametrize("operation", ["rebuild", "repair"])
@pytest.mark.parametrize("redirect", ["directory_symlink", "leaf_symlink", "leaf_hardlink"])
def test_retained_shadow_refuses_aliased_outputs_before_execution(
    environment, recipes: _Recipes, tmp_path: Path, operation: str, redirect: str
) -> None:
    node = _evaluation_node()
    receipt = fulfill_node(node, environment=environment).receipt
    if operation == "repair":
        _prepare_status_corruption(environment, node, receipt.path)
    original_root = Path(environment.root)

    def authoritative_bytes() -> dict[Path, bytes]:
        return {path: path.read_bytes() for path in original_root.rglob("*") if path.is_file()}

    original = authoritative_bytes()
    executions = recipes.evaluation

    shadow_root = tmp_path / "retained-shadow"
    shadow_receipt = shadow_root / receipt.path.relative_to(environment.root)
    if redirect == "directory_symlink":
        shadow_root.mkdir()
        (shadow_root / "manifests").symlink_to(
            Path(environment.root) / "manifests", target_is_directory=True
        )
    else:
        shadow_receipt.parent.mkdir(parents=True)
        if redirect == "leaf_symlink":
            shadow_receipt.symlink_to(receipt.path)
        else:
            shadow_receipt.hardlink_to(receipt.path)

    run = rebuild_node if operation == "rebuild" else repair_node
    with shadow_custody(environment, shadow_root=shadow_root, retain=True) as shadow:
        with pytest.raises(ValueError, match="unsafe fulfillment output path") as refusal:
            run(node, environment=environment, shadow=shadow)

    assert str(shadow_root) in str(refusal.value)
    assert recipes.evaluation == executions
    assert authoritative_bytes() == original


def test_a_shadow_root_beside_a_symlinked_authoritative_root_still_opens(
    environment, tmp_path: Path
) -> None:
    """Resolving both sides must not refuse a legitimate sibling shadow.

    Receipt roots are commonly reached through a symlink. Resolving only the
    shadow would compare a physical path against a lexical one and refuse every
    sibling, so the containment fix is checked from the permissive side too.
    """
    node = _evaluation_node()
    fulfill_node(node, environment=environment)
    linked_root = tmp_path / "receipts-link"
    linked_root.symlink_to(Path(environment.root), target_is_directory=True)
    through_link = FulfillmentEnvironment(
        root=linked_root,
        registries=environment.registries,
        issues=environment.issues,
    )

    with shadow_custody(through_link, shadow_root=tmp_path / "sibling-shadow") as shadow:
        assert rebuild_node(node, environment=through_link, shadow=shadow).matched


def test_ephemeral_shadow_custody_is_discarded(environment, tmp_path: Path) -> None:
    node = _evaluation_node()
    fulfill_node(node, environment=environment)

    assert rebuild_node(node, environment=environment).matched

    leftovers = [entry for entry in tmp_path.iterdir() if entry.name.startswith(".fulfillment-")]
    assert leftovers == []


# --- Output projection -------------------------------------------------------


def _manifest_with_created_at(created_at: str) -> EvaluationRunManifest:
    spec = EvaluationRunSpec(evaluation_type="testpkg.custody", params={"rows": 1})
    return EvaluationRunManifest(
        id=evaluation_run_manifest_id(spec),
        status="completed",
        created_at=created_at,
        evaluation_spec=spec_payload(
            "EvaluationRunSpec", spec.model_dump(mode="json", exclude_none=True)
        ),
        provenance=Provenance(parents=[]),
        summary_metrics={"rows": 1},
    )


def test_output_projection_ignores_manifest_bytes_and_timestamps() -> None:
    early = _manifest_with_created_at("2026-01-01T00:00:00Z")
    late = _manifest_with_created_at("2026-08-01T00:00:00Z")
    assert early.model_dump_json() != late.model_dump_json()

    assert output_projection(early, node_kind="evaluation") == output_projection(
        late, node_kind="evaluation"
    )
    assert compare_output_projections(
        output_projection(early, node_kind="evaluation"),
        output_projection(late, node_kind="evaluation"),
    ) == ()


def test_compare_output_projections_names_list_length_and_leaf_drift() -> None:
    base = output_projection(_manifest_with_created_at("2026-01-01T00:00:00Z"), node_kind="evaluation")
    other = base.model_copy(update={"status": "failed"}, deep=True)

    differences = compare_output_projections(base, other)

    assert [(d.field_path, d.expected, d.observed) for d in differences] == [
        ("status", "completed", "failed")
    ]


def test_output_projection_schema_identity_migrates_or_rejects() -> None:
    projection = output_projection(
        _manifest_with_created_at("2026-01-01T00:00:00Z"), node_kind="evaluation"
    )
    payload = projection.model_dump(mode="json")
    assert payload["schema_id"] == FULFILLMENT_PROJECTION_SCHEMA_ID
    assert payload["schema_version"] == FULFILLMENT_PROJECTION_SCHEMA_VERSION
    assert migrate_output_projection(payload) == projection
    assert migrate_output_projection(projection) == projection

    with pytest.raises(ValueError, match="unsupported OutputProjection schema_version"):
        migrate_output_projection({**payload, "schema_version": "feedbax.fulfillment.x.v0"})
    with pytest.raises(ValueError, match="unsupported OutputProjection schema_id"):
        migrate_output_projection({**payload, "schema_id": "other.family"})


def test_rebuild_outcome_refuses_inconsistent_decisions() -> None:
    projection = output_projection(
        _manifest_with_created_at("2026-01-01T00:00:00Z"), node_kind="evaluation"
    )
    from feedbax.analysis.fulfillment_custody import NodeRebuildOutcome

    with pytest.raises(ValueError, match="must name at least one projection difference"):
        NodeRebuildOutcome(
            node_key="eval/a",
            node_kind="evaluation",
            manifest_id=projection.manifest_id,
            matched=False,
            expected=projection,
            observed=projection,
        )


# --- Repair ------------------------------------------------------------------


def _quarantine_files(environment) -> list[Path]:
    directory = Path(environment.root) / FULFILLMENT_QUARANTINE_DIRECTORY
    return sorted(path for path in directory.rglob("*") if path.is_file())


def test_repair_quarantines_failed_bytes_and_promotes_a_validated_replacement(
    environment, recipes: _Recipes
) -> None:
    node = _evaluation_node()
    receipt = fulfill_node(node, environment=environment).receipt
    manifest = load_manifest(receipt.path)
    declared = manifest.artifacts[0].sha256
    stored = _stored_artifact_path(environment, node)
    good_bytes = stored.read_bytes()
    _tamper(stored, b"tampered custody bytes\n")

    result = repair_node(node, environment=environment)

    quarantined = {entry.origin: entry for entry in result.record.quarantined}
    assert set(quarantined) == {"manifest", "artifact"}
    artifact_entry = quarantined["artifact"]
    assert artifact_entry.declared_sha256 == declared
    assert artifact_entry.observed_sha256 != declared
    quarantined_bytes = Path(environment.root) / artifact_entry.quarantine_relative_path
    assert quarantined_bytes.read_bytes() == b"tampered custody bytes\n"
    assert artifact_entry.observed_sha256 in quarantined_bytes.name

    assert stored.read_bytes() == good_bytes, "the authoritative name holds validated bytes again"
    assert result.receipt.path == receipt.path
    assert "artifact_sha256_mismatch" in result.record.triggering_admission.codes
    assert result.record.admission_after_repair.admitted
    assert result.record.replacement_artifacts[0].sha256 == declared

    executions = recipes.evaluation
    assert fulfill_node(node, environment=environment).disposition == "reused"
    assert recipes.evaluation == executions, "the repaired receipt admits without re-executing"


def test_promotion_refuses_shadow_bytes_that_drifted_after_shadow_admission(
    environment, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The read that promotes is the read that must authenticate.

    Shadow admission proves the candidate's bytes, but the promotion reads them
    again afterwards, and the earlier proof says nothing about the second read.
    Tampering in that window is exactly the accidental-corruption shape the
    repair exists to undo, so promotion refuses before touching anything
    authoritative — the corrupt authoritative name is not even unlinked.
    """
    from feedbax.analysis import fulfillment_custody

    node = _evaluation_node()
    receipt = fulfill_node(node, environment=environment).receipt
    manifest_bytes = receipt.path.read_bytes()
    stored = _stored_artifact_path(environment, node)
    _tamper(stored, b"tampered custody bytes\n")

    authoritative_root = Path(environment.root)
    real_admit = fulfillment_custody.admit_node
    tampered_in_shadow: list[Path] = []

    def _admit_then_tamper_the_shadow(request, *, environment, **kwargs):
        outcome = real_admit(request, environment=environment, **kwargs)
        shadow_root = Path(environment.root)
        if shadow_root != authoritative_root and outcome.admitted and not tampered_in_shadow:
            candidate = load_manifest(Path(outcome.manifest_path))
            shadow_bytes = artifact_bytes_path(candidate.artifacts[0], root=shadow_root)
            assert shadow_bytes is not None
            _tamper(shadow_bytes, b"shadow bytes replaced after admission\n")
            tampered_in_shadow.append(shadow_bytes)
        return outcome

    monkeypatch.setattr(fulfillment_custody, "admit_node", _admit_then_tamper_the_shadow)

    with pytest.raises(FulfillmentRepairError, match="not the bytes shadow admission"):
        repair_node(node, environment=environment)

    assert tampered_in_shadow, "the test must really have drifted the shadow bytes"
    assert receipt.path.read_bytes() == manifest_bytes
    assert stored.read_bytes() == b"tampered custody bytes\n", (
        "a refused promotion never unlinks the authoritative name it was going to replace"
    )
    assert not (Path(environment.root) / FULFILLMENT_REPAIR_DIRECTORY).exists()
    assert _quarantine_files(environment), "the failed bytes are preserved before any promotion"


def test_repair_record_is_durable_and_addressed_by_the_failed_bytes(
    environment,
) -> None:
    node = _evaluation_node()
    receipt = fulfill_node(node, environment=environment).receipt
    _prepare_status_corruption(environment, node, receipt.path)
    original = b"  " + receipt.path.read_bytes() + b"\n\n"
    receipt.path.write_bytes(original)
    repaired = repair_node(node, environment=environment)
    quarantined = next(entry for entry in repaired.record.quarantined if entry.origin == "manifest")
    assert (Path(environment.root) / quarantined.quarantine_relative_path).read_bytes() == original
    assert repaired.record.replacement_manifest_sha256 == repaired.receipt.manifest_sha256

    directory = (
        Path(environment.root)
        / FULFILLMENT_REPAIR_DIRECTORY
    )
    records = sorted(directory.rglob("*.json"))
    assert [path.name for path in records] == [f"{quarantined.observed_sha256}.json"]
    loaded = migrate_repair_record(json.loads(records[0].read_text(encoding="utf-8")))
    assert loaded.manifest_id == receipt.manifest_id
    assert "status_not_completed" in loaded.triggering_admission.codes
    assert loaded.admission_after_repair.admitted
    assert loaded.completed_at >= loaded.started_at


def _prepare_status_corruption(environment, node, path: Path):
    """Corrupt a stored receipt's status in place and return the same node request."""
    payload = path.read_text(encoding="utf-8").replace('"completed"', '"failed"')
    path.write_text(payload, encoding="utf-8")
    return node


def test_repair_refuses_a_healthy_or_absent_receipt(environment) -> None:
    node = _evaluation_node()

    with pytest.raises(FulfillmentRepairError, match="not a repair"):
        repair_node(node, environment=environment)

    fulfill_node(node, environment=environment)
    with pytest.raises(FulfillmentRepairError, match="admits cleanly"):
        repair_node(node, environment=environment)


def test_repair_refuses_when_the_candidate_fails_admission(environment) -> None:
    node = _evaluation_node()
    fulfill_node(node, environment=environment)
    demanding = EvaluationNodeRequest(
        node_key=node.node_key,
        spec=node.spec,
        required_output_roles=("never_produced",),
    )
    receipt_bytes = receipt_path(
        "evaluation", evaluation_run_manifest_id(node.spec), root=environment.root
    ).read_bytes()

    with pytest.raises(FulfillmentRepairError) as excinfo:
        repair_node(demanding, environment=environment)

    assert "output_role_absent" in excinfo.value.outcome.codes
    assert "shadow custody" in str(excinfo.value)
    assert (
        receipt_path(
            "evaluation", evaluation_run_manifest_id(node.spec), root=environment.root
        ).read_bytes()
        == receipt_bytes
    ), "a refused repair never touches the authoritative receipt"
    assert not (Path(environment.root) / FULFILLMENT_REPAIR_DIRECTORY).exists()


def test_crash_before_promotion_leaves_the_authoritative_receipt_untouched(
    environment, monkeypatch: pytest.MonkeyPatch
) -> None:
    from feedbax.analysis import fulfillment_custody

    node = _evaluation_node()
    receipt = fulfill_node(node, environment=environment).receipt
    stored = _stored_artifact_path(environment, node)
    _tamper(stored, b"tampered custody bytes\n")
    manifest_bytes = receipt.path.read_bytes()

    def _crash(*_args, **_kwargs):
        raise RuntimeError("simulated crash after candidate execution")

    monkeypatch.setattr(fulfillment_custody, "_promote_artifact_bytes", _crash)
    with pytest.raises(RuntimeError, match="simulated crash"):
        repair_node(node, environment=environment)

    assert receipt.path.read_bytes() == manifest_bytes
    assert stored.read_bytes() == b"tampered custody bytes\n"
    assert not (Path(environment.root) / FULFILLMENT_REPAIR_DIRECTORY).exists()
    assert _quarantine_files(environment), "the failed bytes are preserved before any promotion"


def test_crash_before_publication_never_leaves_a_half_promoted_candidate(
    environment, monkeypatch: pytest.MonkeyPatch
) -> None:
    from feedbax.analysis import fulfillment_custody

    node = _evaluation_node()
    receipt = fulfill_node(node, environment=environment).receipt
    _tamper(_stored_artifact_path(environment, node), b"tampered custody bytes\n")
    manifest_bytes = receipt.path.read_bytes()

    def _crash(*_args, **_kwargs):
        raise RuntimeError("simulated crash before publication")

    monkeypatch.setattr(fulfillment_custody, "_publish_manifest_bytes", _crash)
    with pytest.raises(RuntimeError, match="simulated crash"):
        repair_node(node, environment=environment)

    assert receipt.path.read_bytes() == manifest_bytes, (
        "the authoritative manifest holds the original bytes, never an unvalidated candidate"
    )
    assert not (Path(environment.root) / FULFILLMENT_REPAIR_DIRECTORY).exists()
    assert _quarantine_files(environment)


def test_repair_publisher_refuses_manifest_directory_redirected_after_preflight(
    environment, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from feedbax.analysis import fulfillment_custody

    node = _evaluation_node()
    receipt = fulfill_node(node, environment=environment).receipt
    _prepare_status_corruption(environment, node, receipt.path)
    original = receipt.path.read_bytes()
    redirected = tmp_path / "redirected"
    redirected.mkdir()
    target = redirected / receipt.path.name
    target.write_bytes(original)
    receipt_directory = receipt.path.parent
    moved_directory = receipt_directory.with_name("moved")
    real_admit = fulfillment_custody.admit_node

    def admit_then_redirect(request, *, environment, **kwargs):
        outcome = real_admit(request, environment=environment, **kwargs)
        if kwargs.get("manifest") is not None and outcome.admitted:
            receipt_directory.rename(moved_directory)
            receipt_directory.symlink_to(redirected, target_is_directory=True)
        return outcome

    monkeypatch.setattr(fulfillment_custody, "admit_node", admit_then_redirect)
    with pytest.raises(ValueError, match="unsafe fulfillment output path"):
        repair_node(node, environment=environment)

    assert target.read_bytes() == original
    assert (moved_directory / receipt.path.name).read_bytes() == original
    assert not (Path(environment.root) / FULFILLMENT_REPAIR_DIRECTORY).exists()


def test_repair_record_schema_identity_migrates_or_rejects(environment) -> None:
    node = _evaluation_node()
    fulfill_node(node, environment=environment)
    _tamper(_stored_artifact_path(environment, node), b"tampered custody bytes\n")
    record = repair_node(node, environment=environment).record

    payload = record.model_dump(mode="json")
    assert payload["schema_id"] == FULFILLMENT_REPAIR_SCHEMA_ID
    assert payload["schema_version"] == FULFILLMENT_REPAIR_SCHEMA_VERSION
    assert migrate_repair_record(payload) == record
    assert migrate_repair_record(record) == record

    with pytest.raises(ValueError, match="unsupported RepairRecord schema_version"):
        migrate_repair_record({**payload, "schema_version": "feedbax.fulfillment.repair.v0"})
    with pytest.raises(ValueError, match="unsupported RepairRecord schema_id"):
        migrate_repair_record({**payload, "schema_id": "other.family"})


def test_repair_record_refuses_to_describe_a_repair_that_did_not_happen(
    environment,
) -> None:
    node = _evaluation_node()
    fulfill_node(node, environment=environment)
    _tamper(_stored_artifact_path(environment, node), b"tampered custody bytes\n")
    record = repair_node(node, environment=environment).record
    payload = record.model_dump(mode="json")

    healthy = dict(payload)
    healthy["triggering_admission"] = payload["admission_after_repair"]
    with pytest.raises(ValueError, match="admission failure that triggered it"):
        migrate_repair_record(healthy)

    unrepaired = dict(payload)
    unrepaired["admission_after_repair"] = payload["triggering_admission"]
    with pytest.raises(ValueError, match="only after the replacement admits"):
        migrate_repair_record(unrepaired)


def test_projection_and_repair_families_are_distinct_documents() -> None:
    assert FULFILLMENT_PROJECTION_SCHEMA_ID != FULFILLMENT_REPAIR_SCHEMA_ID
    assert OutputProjection.model_fields["schema_id"].default == FULFILLMENT_PROJECTION_SCHEMA_ID
