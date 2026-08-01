"""The fulfillment driver: preflight, native lowering, pull-forward, custody.

Everything here is stated over ``quillon``'s compiled outputs — real Feedbax
specs under real compile locks, emitted into ``tmp_path`` — and a receipt root
under ``tmp_path``. There is no lowering callback and no payload preparer: the
driver reaches Feedbax's own lowering directly, so the only project vocabulary
in play is the recipe names quillon registers and the roles its bindings state.

Five claims are under test:

* **preflight refuses before anything runs.** A closure naming a boundary node —
  a compiled training run matrix — is refused with every boundary node named,
  and no recipe of any branch runs;
* **the node request follows from the compiled document's schema identity**, and
  the receipts it binds follow from the lock's typed consumer bindings;
* **fulfillment is a pull-forward.** Each node is reused when its receipt admits
  and executed exactly once when it does not, in the plan's dependency order, so
  a second walk over a fulfilled closure executes nothing and an interrupted one
  resumes at the node boundary it stopped at;
* **a receipt that exists but fails admission is a refusal**, never a cache miss;
* **rebuild is verification and repair is transactional**, both over the whole
  closure and both feedbax's, reached through this driver's resolution walk.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from feedbax.analysis.evaluation import EvaluationRecipeResult
from feedbax.analysis.fulfillment import FulfillmentAdmissionError
from feedbax.analysis.fulfillment_adapters import FulfillmentEnvironment
from feedbax.analysis.fulfillment_custody import FulfillmentDriftError
from feedbax.analysis.fulfillment_derivation import (
    derive_fulfillment_plan,
    read_compiled_outputs,
)
from feedbax.analysis.fulfillment_driver import (
    AmbiguousNodeReceiptError,
    ExternalBoundaryError,
    MissingExternalReceiptError,
    PlanDocumentDriftError,
    closure_requests,
    fulfill_closure,
    preflight,
    rebuild_closure,
    repair_closure_node,
    truncated_closure,
)
from feedbax.analysis.fulfillment_lowering import NodeLoweringError
from feedbax.analysis.fulfillment_plan import LogicalKey
from feedbax.analysis.reports import REPORT_RENDER_ROLE, ReportRecipeResult
from feedbax.contracts.experiment_compile_lock import (
    AnalysisInputBinding,
    AuthenticatedReceiptReference,
    CheckpointInitializationBinding,
    EvaluationSubjectBinding,
    FigureRuntimeInputBinding,
    ReceiptLocatorReference,
    ReportParentBinding,
)
from feedbax.contracts.manifest import (
    ReportManifest,
    canonical_manifest_path,
    load_manifest,
    store_bytes_artifact,
)

from tests.fake_project_extension.products import (
    BULLETIN_TYPE,
    CONDENSE_TYPE,
    PROBE_TYPE,
    QuillonOutputs,
    planned,
)

DIGEST = "b" * 64


# --------------------------------------------------------------------------
# Recipes: the smallest registered work that still writes a real receipt
# --------------------------------------------------------------------------


class _Calls:
    """Counts recipe invocations, so reuse is proven rather than assumed."""

    def __init__(self) -> None:
        self.evaluation = 0
        self.report = 0
        self.payload = "baseline"


@pytest.fixture
def calls() -> _Calls:
    return _Calls()


@pytest.fixture
def outputs(tmp_path: Path) -> QuillonOutputs:
    return QuillonOutputs(tmp_path / "repo")


@pytest.fixture
def environment(tmp_path: Path, application_registry_bundle, calls: _Calls):
    def evaluation_recipe(run_spec, root, states_path, execution_context):
        calls.evaluation += 1
        artifact = store_bytes_artifact(
            f"{calls.payload}:{run_spec.params.get('stage', '')}\n".encode(),
            root=root,
            role="evaluation_states",
            logical_name="states.bin",
        )
        return EvaluationRecipeResult(
            states=None,
            summary_metrics={"stage": run_spec.params.get("stage", ""),
                             "payload": calls.payload},
            artifacts=[artifact],
            metadata={"states_schema": "quillon.states.v1"},
        )

    def report_recipe(report_spec, root, inputs):
        calls.report += 1
        artifact = store_bytes_artifact(
            f"# {report_spec.report_type}\n".encode(),
            root=root,
            role=REPORT_RENDER_ROLE,
            logical_name="bulletin.md",
            media_type="text/markdown",
            suffix=".md",
        )
        return ReportRecipeResult(artifacts=[artifact], summary={"inputs": len(inputs)})

    application_registry_bundle.evaluation_recipes.register(PROBE_TYPE, evaluation_recipe)
    application_registry_bundle.evaluation_recipes.register(CONDENSE_TYPE, evaluation_recipe)
    application_registry_bundle.report_recipes.register(BULLETIN_TYPE, report_recipe)
    return FulfillmentEnvironment(
        root=tmp_path / "receipts",
        registries=application_registry_bundle,
        issues=("7be9c5d",),
    )


# --------------------------------------------------------------------------
# Closure helpers
# --------------------------------------------------------------------------


def _closure(outputs: QuillonOutputs, target: str):
    index = read_compiled_outputs(outputs.output_directory)
    return preflight(derive_fulfillment_plan(index, target=target), index)


def _fulfill(outputs: QuillonOutputs, target: str, *, environment):
    return fulfill_closure(_closure(outputs, target), environment=environment)


def _subject(product, *, role_path: str, subject_id: str):
    return planned(
        product, role_path=role_path, consumer=EvaluationSubjectBinding(subject_id=subject_id)
    )


def _chain(outputs: QuillonOutputs) -> str:
    """A three-node closure: probe -> probe -> bulletin, and its target name."""
    root = outputs.probe("chain-root")
    mid = outputs.probe(
        "chain-mid", references=[_subject(root, role_path="body.upstream", subject_id="upstream")]
    )
    outputs.bulletin(
        "chain-leaf",
        references=[
            planned(
                mid,
                role_path="body.middle",
                consumer=ReportParentBinding(parent_kind="probe", parent_id="middle"),
            )
        ],
    )
    return "chain-leaf"


CHAIN_ORDER = ("evaluation:chain-root", "evaluation:chain-mid", "report:chain-leaf")


def _reports_directory(environment: FulfillmentEnvironment) -> Path:
    """Return where a report receipt would be written, so absence is checkable."""
    return canonical_manifest_path("ReportManifest", "probe", root=environment.root).parent


def _mutate(path: Path, **changes: Any) -> None:
    document = json.loads(path.read_text(encoding="utf-8"))
    document.update(changes)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# Preflight: the closure, its order, and the external boundary
# --------------------------------------------------------------------------


def test_preflight_states_the_whole_closure_in_dependency_order(
    outputs: QuillonOutputs,
) -> None:
    closure = _closure(outputs, _chain(outputs))
    assert closure.order == CHAIN_ORDER
    assert closure.target == LogicalKey("report", "chain-leaf")
    for node in closure.nodes:
        assert node.compiled.content_hash == node.plan_node.content_hash
        assert node.document["schema_id"] == node.kind


def test_a_document_that_no_longer_hashes_to_what_was_pinned_refuses(
    outputs: QuillonOutputs,
) -> None:
    target = _chain(outputs)
    index = read_compiled_outputs(outputs.output_directory)
    plan = derive_fulfillment_plan(index, target=target)
    outputs.probe("chain-mid", stage_note="moved")
    with pytest.raises(PlanDocumentDriftError) as caught:
        preflight(plan, read_compiled_outputs(outputs.output_directory))
    assert caught.value.key == LogicalKey("evaluation", "chain-mid")


def test_a_training_matrix_refuses_the_closure_before_any_node_executes(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment, calls: _Calls
) -> None:
    cohort = outputs.cohort("boundary-run")
    outputs.probe(
        "boundary-consumer",
        references=[_subject(cohort, role_path="body.harvested", subject_id="harvested")],
    )

    with pytest.raises(ExternalBoundaryError) as caught:
        _fulfill(outputs, "boundary-consumer", environment=environment)

    record = caught.value.record()
    assert [node["key"] for node in record["boundary_nodes"]] == ["training:boundary-run"]
    node = record["boundary_nodes"][0]
    assert node["source_ref"] == "studies/boundary-run.envelope.json"
    assert node["boundary"] == "feedbax.spec.training_run_matrix"
    assert node["named_by"] == [
        {"consumer": "evaluation:boundary-consumer", "role_path": ["body", "harvested"]}
    ]
    assert node["unblocks"] == ["evaluation:boundary-consumer"]
    assert calls.evaluation == 0
    assert not environment.root.exists()


def test_the_refusal_lists_every_branch_the_boundary_unblocks(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment, calls: _Calls
) -> None:
    """Zero executions on *every* branch, including the ones that could run."""
    cohort = outputs.cohort("shared-run")
    free = outputs.probe("free-branch")
    blocked = outputs.probe(
        "blocked-branch",
        references=[_subject(cohort, role_path="body.harvested", subject_id="harvested")],
    )
    outputs.bulletin(
        "joined",
        references=[
            planned(
                free,
                role_path="body.free",
                consumer=ReportParentBinding(parent_kind="probe", parent_id="free"),
            ),
            planned(
                blocked,
                role_path="body.blocked",
                consumer=ReportParentBinding(parent_kind="probe", parent_id="blocked"),
            ),
        ],
    )
    with pytest.raises(ExternalBoundaryError) as caught:
        _fulfill(outputs, "joined", environment=environment)
    assert caught.value.record()["boundary_nodes"][0]["unblocks"] == [
        "evaluation:blocked-branch",
        "report:joined",
    ]
    assert calls.evaluation == 0


# --------------------------------------------------------------------------
# Native lowering: the schema decides the request, the binding decides the role
# --------------------------------------------------------------------------


def test_the_compiled_schema_decides_the_node_request(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment
) -> None:
    target = _chain(outputs)
    _fulfill(outputs, target, environment=environment)
    requests = closure_requests(_closure(outputs, target), environment=environment)
    assert [request.node_kind for request in requests] == ["evaluation", "evaluation", "report"]
    assert [request.node_key for request in requests] == list(CHAIN_ORDER)
    assert [request.order for request in requests] == [0, 1, 2]


def test_a_consumer_binding_names_the_role_its_receipt_is_bound_under(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment
) -> None:
    root = outputs.probe("role-source")
    outputs.condensate(
        "role-consumer",
        references=[
            planned(
                root,
                role_path="inputs.states",
                consumer=AnalysisInputBinding(alias="role-source", role="observed_states"),
            )
        ],
    )
    outputs.probe("role-source")  # keep the emitted pair identical
    fulfill_closure(
        truncated_closure(_closure(outputs, "role-consumer"), 1), environment=environment
    )
    # No analysis recipe is registered, so the analysis node stops the walk before
    # admission; its resolved request is what the binding decides.
    requests = closure_requests(
        _closure(outputs, "role-consumer"),
        environment=environment,
        stop_at=LogicalKey("analysis", "role-consumer"),
    )
    assert [ref.role for ref in requests[-1].spec.inputs] == ["observed_states"]
    assert requests[-1].spec.inputs[0].kind == "EvaluationRunManifest"


def test_a_figure_binds_its_runtime_input_authority_by_role(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment
) -> None:
    source = outputs.probe("plate-source")
    outputs.plate(
        "plate",
        references=[
            planned(
                source,
                role_path="runtime.states",
                consumer=FigureRuntimeInputBinding(input_role="observed"),
            )
        ],
    )
    fulfill_closure(
        truncated_closure(_closure(outputs, "plate"), 1), environment=environment
    )
    requests = closure_requests(
        _closure(outputs, "plate"),
        environment=environment,
        stop_at=LogicalKey("figure", "plate"),
    )
    figure = requests[-1]
    assert figure.node_kind == "figure"
    assert figure.runtime_inputs is not None
    assert [ref.role for ref in figure.runtime_inputs] == ["observed"]
    assert figure.spec["schema_id"] == "feedbax.spec.figure"


def test_a_checkpoint_initialization_binding_never_binds_an_executable_node(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment
) -> None:
    prior = outputs.probe("prior-run")
    outputs.probe(
        "misbound",
        references=[
            planned(
                prior,
                role_path="body.checkpoint",
                consumer=CheckpointInitializationBinding(
                    mode="continue_from", row_id="misbound"
                ),
            )
        ],
    )
    with pytest.raises(NodeLoweringError, match="training row"):
        _fulfill(outputs, "misbound", environment=environment)


def test_a_compiled_spec_that_already_declares_inputs_refuses(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment
) -> None:
    """A compile plan cannot authenticate an input, so it may not state one."""
    outputs.emit(
        "pre-bound",
        {
            "schema_id": "feedbax.spec.evaluation_run",
            "schema_version": "feedbax.spec.evaluation_run.v1",
            "evaluation_type": PROBE_TYPE,
            "params": {"stage": "pre-bound"},
            "inputs": [{"kind": "EvaluationRunManifest", "id": "feedbax-evaluation-run:x"}],
        },
    )
    with pytest.raises(NodeLoweringError, match="cannot authenticate an input"):
        _fulfill(outputs, "pre-bound", environment=environment)


def test_an_evaluation_matrix_refuses_a_closure_edge_it_cannot_distribute(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment
) -> None:
    source = outputs.probe("matrix-source")
    outputs.probe_matrix(
        "matrix-consumer",
        references=[_subject(source, role_path="body.subject", subject_id="subject")],
    )
    with pytest.raises(NodeLoweringError, match="per row"):
        _fulfill(outputs, "matrix-consumer", environment=environment)


# --------------------------------------------------------------------------
# Pull-forward execution
# --------------------------------------------------------------------------


def test_one_walk_fulfils_the_closure_and_the_second_executes_nothing(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment, calls: _Calls
) -> None:
    target = _chain(outputs)

    first = _fulfill(outputs, target, environment=environment)
    assert first.execution_order == CHAIN_ORDER
    assert first.executed == CHAIN_ORDER
    assert calls.evaluation == 2 and calls.report == 1
    for result in first.results:
        assert load_manifest(result.receipt.path).status == "completed"

    second = _fulfill(outputs, target, environment=environment)
    assert second.executed == ()
    assert second.reused == CHAIN_ORDER
    assert calls.evaluation == 2 and calls.report == 1


def test_two_walks_produce_identical_order_and_listings(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment
) -> None:
    target = _chain(outputs)
    first = _fulfill(outputs, target, environment=environment)
    second = _fulfill(outputs, target, environment=environment)
    assert first.execution_order == second.execution_order
    assert [result.receipt.manifest_id for result in first.results] == [
        result.receipt.manifest_id for result in second.results
    ]


def test_a_receipt_binds_forward_as_the_authenticated_parent_it_is(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment
) -> None:
    run = _fulfill(outputs, _chain(outputs), environment=environment)
    upstream, middle, leaf = run.results
    middle_parents = load_manifest(middle.receipt.path).provenance.parents
    assert [ref.id for ref in middle_parents] == [upstream.receipt.manifest_id]
    assert middle_parents[0].metadata["manifest_sha256"] == hashlib.sha256(
        upstream.receipt.path.read_bytes()
    ).hexdigest()
    assert [ref.role for ref in middle_parents] == ["upstream"]
    leaf_parents = load_manifest(leaf.receipt.path).provenance.parents
    assert [ref.id for ref in leaf_parents] == [middle.receipt.manifest_id]
    assert [ref.role for ref in leaf_parents] == ["middle"]


@pytest.mark.parametrize("boundary", [1, 2])
def test_an_interrupted_walk_resumes_at_the_node_boundary_it_stopped_at(
    outputs: QuillonOutputs,
    environment: FulfillmentEnvironment,
    calls: _Calls,
    boundary: int,
) -> None:
    """A crash after node k leaves k admitted receipts; re-invocation finishes."""
    target = _chain(outputs)
    partial = truncated_closure(_closure(outputs, target), boundary)
    interrupted = fulfill_closure(partial, environment=environment)
    assert interrupted.executed == CHAIN_ORDER[:boundary]
    assert calls.evaluation == boundary

    resumed = _fulfill(outputs, target, environment=environment)
    assert resumed.reused == CHAIN_ORDER[:boundary]
    assert resumed.executed == CHAIN_ORDER[boundary:]
    assert calls.evaluation == 2


def test_deleting_one_producer_receipt_rebinds_and_re_mints_its_consumers(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment, calls: _Calls
) -> None:
    target = _chain(outputs)
    first = _fulfill(outputs, target, environment=environment)
    root_receipt, mid_receipt, leaf_receipt = (result.receipt for result in first.results)
    old_leaf_id = leaf_receipt.manifest_id
    old_leaf_path = leaf_receipt.path
    old_leaf_parents = load_manifest(old_leaf_path).provenance.parents

    mid_receipt.path.unlink()
    calls.payload = "re-executed"  # the re-execution is real, not a byte replay

    second = _fulfill(outputs, target, environment=environment)
    assert second.reused == ("evaluation:chain-root",)
    assert set(second.executed) == {"evaluation:chain-mid", "report:chain-leaf"}
    assert calls.evaluation == 3, "nothing upstream of the deleted receipt re-executed"
    assert load_manifest(root_receipt.path).id == root_receipt.manifest_id

    new_leaf = second.results[-1].receipt
    assert new_leaf.manifest_id != old_leaf_id, "the consumer's identity is recomputed"
    assert old_leaf_path.is_file(), "the old receipt stays a valid record of its old parents"
    assert load_manifest(old_leaf_path).provenance.parents == old_leaf_parents
    assert load_manifest(new_leaf.path).provenance.parents != old_leaf_parents


# --------------------------------------------------------------------------
# Admission failure is refusal, never a cache miss
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("changes", "code"),
    [
        ({"status": "failed"}, "status_not_completed"),
        ({"status": "running"}, "status_not_completed"),
        ({"id": "feedbax-evaluation-run:0000"}, "identity_mismatch"),
        ({"provenance": {"parents": []}}, "parents_mismatch"),
    ],
)
def test_each_admission_failure_refuses_with_its_named_failure(
    outputs: QuillonOutputs,
    environment: FulfillmentEnvironment,
    calls: _Calls,
    changes: dict[str, Any],
    code: str,
) -> None:
    target = _chain(outputs)
    run = _fulfill(outputs, target, environment=environment)
    _mutate(run.results[1].receipt.path, **changes)

    with pytest.raises(FulfillmentAdmissionError) as caught:
        _fulfill(outputs, target, environment=environment)
    assert code in caught.value.outcome.codes
    assert calls.evaluation == 2, "a failed record is never silently re-executed over"


def test_altered_artifact_bytes_refuse_as_custody_loss(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment, calls: _Calls
) -> None:
    target = _chain(outputs)
    run = _fulfill(outputs, target, environment=environment)
    artifact = load_manifest(run.results[0].receipt.path).artifacts[0]
    (environment.root / artifact.metadata["relative_path"]).write_bytes(b"corrupt")

    with pytest.raises(FulfillmentAdmissionError) as caught:
        _fulfill(outputs, target, environment=environment)
    assert "artifact_sha256_mismatch" in caught.value.outcome.codes
    assert calls.evaluation == 2


# --------------------------------------------------------------------------
# Already-produced receipts: quoted, resolved canonically, never inferred
# --------------------------------------------------------------------------


def test_a_receipt_locator_binds_at_its_canonical_location(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment, calls: _Calls
) -> None:
    outputs.probe("prior-source")
    produced = _fulfill(outputs, "prior-source", environment=environment).results[0].receipt
    outputs.bulletin(
        "prior-consumer",
        references=[
            ReceiptLocatorReference(
                manifest_kind="EvaluationRunManifest",
                manifest_id=produced.manifest_id,
                role_path="body.prior",
                consumer=ReportParentBinding(parent_kind="probe", parent_id="prior"),
            )
        ],
    )
    run = _fulfill(outputs, "prior-consumer", environment=environment)
    assert run.execution_order == ("report:prior-consumer",)
    bound = load_manifest(run.results[0].receipt.path).provenance.parents
    assert [ref.id for ref in bound] == [produced.manifest_id]
    assert [ref.role for ref in bound] == ["prior"]
    assert bound[0].metadata["manifest_sha256"] == hashlib.sha256(
        produced.path.read_bytes()
    ).hexdigest()
    assert calls.evaluation == 1


def test_an_authenticated_receipt_reference_binds_the_same_way(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment
) -> None:
    outputs.probe("quoted-source")
    produced = _fulfill(outputs, "quoted-source", environment=environment).results[0].receipt
    raw = produced.path.read_bytes()
    outputs.bulletin(
        "quoted-consumer",
        references=[
            AuthenticatedReceiptReference(
                manifest_kind="EvaluationRunManifest",
                manifest_id=produced.manifest_id,
                manifest_sha256=hashlib.sha256(raw).hexdigest(),
                size_bytes=len(raw),
                role_path="body.quoted",
                consumer=ReportParentBinding(parent_kind="probe", parent_id="quoted"),
            )
        ],
    )
    run = _fulfill(outputs, "quoted-consumer", environment=environment)
    bound = load_manifest(run.results[0].receipt.path).provenance.parents
    assert [ref.id for ref in bound] == [produced.manifest_id]


def test_an_absent_receipt_is_a_refusal_and_never_an_inapplicability(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment, calls: _Calls
) -> None:
    outputs.bulletin(
        "absent-consumer",
        references=[
            ReceiptLocatorReference(
                manifest_kind="EvaluationRunManifest",
                manifest_id="feedbax-evaluation-run:never-produced",
                role_path="body.prior",
                consumer=ReportParentBinding(parent_kind="probe", parent_id="prior"),
            )
        ],
    )
    with pytest.raises(MissingExternalReceiptError) as caught:
        _fulfill(outputs, "absent-consumer", environment=environment)
    assert caught.value.manifest_id == "feedbax-evaluation-run:never-produced"
    assert caught.value.consumer == LogicalKey("report", "absent-consumer")
    assert "never means the input does not apply" in str(caught.value)
    assert calls.evaluation == 0
    assert not _reports_directory(environment).exists()


def test_an_incomplete_receipt_is_refused_rather_than_bound(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment
) -> None:
    outputs.probe("incomplete-source")
    produced = (
        _fulfill(outputs, "incomplete-source", environment=environment).results[0].receipt
    )
    _mutate(produced.path, status="failed")
    outputs.bulletin(
        "incomplete-consumer",
        references=[
            ReceiptLocatorReference(
                manifest_kind="EvaluationRunManifest",
                manifest_id=produced.manifest_id,
                role_path="body.prior",
                consumer=ReportParentBinding(parent_kind="probe", parent_id="prior"),
            )
        ],
    )
    with pytest.raises(MissingExternalReceiptError):
        _fulfill(outputs, "incomplete-consumer", environment=environment)


# --------------------------------------------------------------------------
# Applicability reaches the walk as a decision, never as a missing input
# --------------------------------------------------------------------------


def test_an_inapplicable_role_binds_nothing_and_blocks_nothing(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment
) -> None:
    from feedbax.contracts.experiment_compile_lock import NotApplicableReference

    source = outputs.probe("partial-source")
    outputs.bulletin(
        "partial-bulletin",
        references=[
            planned(
                source,
                role_path="body.nominal",
                consumer=ReportParentBinding(parent_kind="probe", parent_id="nominal"),
            ),
            NotApplicableReference(
                role_path="body.appendix",
                basis="compiler_rule",
                reason="no producer in this closure binds the role",
                rule_id="feedbax.rule.unbound_role.v1",
            ),
        ],
    )
    run = _fulfill(outputs, "partial-bulletin", environment=environment)
    manifest = load_manifest(run.results[-1].receipt.path)
    assert isinstance(manifest, ReportManifest)
    assert [parent.role for parent in manifest.provenance.parents] == ["nominal"]


# --------------------------------------------------------------------------
# Rebuild as verification, repair as recovery
# --------------------------------------------------------------------------


def test_rebuilding_an_intact_closure_reports_no_drift_and_preserves_receipts(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment
) -> None:
    target = _chain(outputs)
    run = _fulfill(outputs, target, environment=environment)
    before = {result.receipt.path: result.receipt.path.read_bytes() for result in run.results}

    rebuilt = rebuild_closure(_closure(outputs, target), environment=environment)
    assert rebuilt.verification_order == CHAIN_ORDER
    assert rebuilt.drifted == ()
    assert {path: path.read_bytes() for path in before} == before


def test_a_receipt_that_disagrees_with_a_clean_re_execution_drifts(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment
) -> None:
    """Drift is reported per node against the defined projection, original intact."""
    outputs.probe("drifting-probe")
    receipt = _fulfill(outputs, "drifting-probe", environment=environment).results[0].receipt
    document = json.loads(receipt.path.read_text(encoding="utf-8"))
    document["summary_metrics"] = {**document.get("summary_metrics", {}), "stage": "tampered"}
    receipt.path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(FulfillmentDriftError) as caught:
        rebuild_closure(_closure(outputs, "drifting-probe"), environment=environment)
    (outcome,) = caught.value.drifted
    assert outcome.node_key == "evaluation:drifting-probe"
    assert json.loads(receipt.path.read_text(encoding="utf-8"))["summary_metrics"]["stage"] == (
        "tampered"
    ), "the authoritative receipt is never written to by a rebuild"


def test_altered_stored_bytes_refuse_before_any_rebuild(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment, calls: _Calls
) -> None:
    target = _chain(outputs)
    run = _fulfill(outputs, target, environment=environment)
    artifact = load_manifest(run.results[0].receipt.path).artifacts[0]
    (environment.root / artifact.metadata["relative_path"]).write_bytes(b"corrupt")

    with pytest.raises(FulfillmentAdmissionError) as caught:
        rebuild_closure(_closure(outputs, target), environment=environment)
    assert "artifact_sha256_mismatch" in caught.value.outcome.codes
    assert calls.evaluation == 2, "corruption is custody loss, not drift, so nothing rebuilt"


def test_repair_promotes_a_revalidated_candidate_and_records_the_custody_event(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment
) -> None:
    target = _chain(outputs)
    run = _fulfill(outputs, target, environment=environment)
    receipt = run.results[1].receipt
    _mutate(receipt.path, status="failed")

    result = repair_closure_node(
        _closure(outputs, target),
        LogicalKey("evaluation", "chain-mid"),
        environment=environment,
    )
    assert result.record.node_key == "evaluation:chain-mid"
    assert result.record.triggering_admission.codes == ("status_not_completed",)
    assert result.record.admission_after_repair.admitted
    assert result.record_path.is_file()
    assert load_manifest(receipt.path).status == "completed"


def test_resolution_refuses_before_repairing_around_a_broken_upstream(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment
) -> None:
    target = _chain(outputs)
    run = _fulfill(outputs, target, environment=environment)
    _mutate(run.results[0].receipt.path, status="failed")
    with pytest.raises(FulfillmentAdmissionError):
        repair_closure_node(
            _closure(outputs, target),
            LogicalKey("report", "chain-leaf"),
            environment=environment,
        )


# --------------------------------------------------------------------------
# What the driver refuses to guess
# --------------------------------------------------------------------------


def test_a_matrix_node_has_no_single_receipt_to_resolve(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment
) -> None:
    """Resolving one receipt for a matrix would mean choosing among its rows."""
    matrix = outputs.probe_matrix("standalone-matrix")
    outputs.bulletin(
        "matrix-reader",
        references=[
            planned(
                matrix,
                role_path="body.rows",
                consumer=ReportParentBinding(parent_kind="matrix", parent_id="rows"),
            )
        ],
    )
    with pytest.raises(AmbiguousNodeReceiptError, match="evaluation matrix"):
        closure_requests(_closure(outputs, "matrix-reader"), environment=environment)
