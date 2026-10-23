"""The fulfillment driver: prepare_workflow, native lowering, pull-forward, custody.

Everything here is stated over ``quillon``'s compiled outputs — real Feedbax
specs under real compile locks, emitted into ``tmp_path`` — and a receipt root
under ``tmp_path``. There is no lowering callback and no payload preparer: the
driver reaches Feedbax's own lowering directly, so the only project vocabulary
in play is the recipe names quillon registers and the roles its bindings state.

Five claims are under test:

* **prepare_workflow refuses before anything runs.** A closure naming an external operation —
  a compiled training run matrix — is refused with every such operation named,
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
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import numpy as np

from feedbax.analysis.evaluation import EvaluationRecipeResult
from feedbax.analysis.specs import AnalysisRecipeResult
from feedbax.analysis.fulfillment import FulfillmentAdmissionError
from feedbax.analysis.fulfillment_adapters import FulfillmentEnvironment
from feedbax.analysis.fulfillment_custody import FulfillmentDriftError
from feedbax.workflow.derivation import (
    ExternalReceiptRecord,
    derive_workflow_plan,
    read_compiled_outputs,
)
from feedbax.workflow.execution import (
    AmbiguousNodeReceiptError,
    ExternalOperationError,
    ExternalReceiptAuthenticationError,
    PlanLockDisagreementError,
    PlanNodeDisagreementError,
    external_parent_ref,
    MissingExternalReceiptError,
    NodeBinding,
    PlanDocumentDriftError,
    UnpinnedPlanNodeError,
    workflow_requests,
    execute_workflow,
    prepare_workflow,
    rebuild_workflow,
    repair_workflow_operation,
    truncated_workflow,
)
from feedbax.workflow.operation_execution import NodeLoweringError
from feedbax.workflow.plan import (
    LogicalKey,
    WorkflowPlan,
    WorkflowPlanIdentityError,
    workflow_plan_from_document,
)
from feedbax.analysis.reports import (
    ORDERED_FIGURE_REPORT_PARAMS_SCHEMA_ID,
    ORDERED_FIGURE_REPORT_PARAMS_SCHEMA_VERSION,
    ORDERED_FIGURE_REPORT_TYPE,
    REPORT_RENDER_ROLE,
    ReportRecipeResult,
)
from feedbax.contracts.experiment_compile_lock import (
    AnalysisInputBinding,
    AnalysisReceiptSetBinding,
    AuthenticatedReceiptReference,
    CheckpointInitializationBinding,
    EvaluationSubjectBinding,
    FigureRuntimeInputBinding,
    ReceiptLocatorReference,
    ReportParentBinding,
)
from feedbax.contracts.figures import FIGURE_SPEC_SCHEMA_ID
from feedbax.contracts.base import (
    canonical_json_bytes,
    sha256_bytes,
)
from feedbax.contracts.artifact_store import store_bytes_artifact
from feedbax.contracts.manifest import (
    canonical_manifest_path,
    load_manifest,
)

from tests.fake_project_experiment.products import (
    BULLETIN_TYPE,
    CONDENSE_TYPE,
    PROBE_TYPE,
    QuillonOutputs,
    planned,
)
from tests.analysis_fixtures import ToyAnalysis, build_toy_analysis_data

DIGEST = "b" * 64


# --------------------------------------------------------------------------
# Recipes: the smallest registered work that still writes a real receipt
# --------------------------------------------------------------------------


class _Calls:
    """Counts recipe invocations, so reuse is proven rather than assumed."""

    def __init__(self) -> None:
        self.evaluation = 0
        self.analysis = 0
        self.report = 0
        self.payload = "baseline"
        self.deterministic_analysis = False


class _DeterministicToyAnalysis(ToyAnalysis):
    """Toy analysis without Plotly's random HTML element IDs."""

    def make_figs(self, data, *, result, **kwargs):
        return {}


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
            role="payload",
            logical_name="states.bin",
        )
        return EvaluationRecipeResult(
            states={"value": np.asarray(1, dtype=np.int32)},
            summary_metrics={"stage": run_spec.params.get("stage", ""), "payload": calls.payload},
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
        )
        return ReportRecipeResult(artifacts=[artifact], summary={"inputs": len(inputs)})

    def analysis_recipe(_spec, _root, inputs, _execution_context):
        calls.analysis += 1
        return AnalysisRecipeResult(
            analyses={
                "toy": (
                    _DeterministicToyAnalysis(variant="toy", cache_result=True)
                    if calls.deterministic_analysis
                    else ToyAnalysis(variant="toy", cache_result=True)
                )
            },
            data=build_toy_analysis_data(value=len(inputs)),
        )

    application_registry_bundle.evaluation_recipes.register(PROBE_TYPE, evaluation_recipe)
    application_registry_bundle.evaluation_recipes.register(CONDENSE_TYPE, evaluation_recipe)
    application_registry_bundle.analysis_recipes.register(CONDENSE_TYPE, analysis_recipe)
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
    return prepare_workflow(derive_workflow_plan(index, target=target), index)


def _fulfill(outputs: QuillonOutputs, target: str, *, environment):
    return execute_workflow(_closure(outputs, target), environment=environment)


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
        assert node.document["schema_id"] == node.compiled_schema_id


def test_a_document_that_no_longer_hashes_to_what_was_pinned_refuses(
    outputs: QuillonOutputs,
) -> None:
    target = _chain(outputs)
    index = read_compiled_outputs(outputs.output_directory)
    plan = derive_workflow_plan(index, target=target)
    outputs.probe("chain-mid", stage_note="moved")
    with pytest.raises(PlanDocumentDriftError) as caught:
        prepare_workflow(plan, read_compiled_outputs(outputs.output_directory))
    assert caught.value.key == LogicalKey("evaluation", "chain-mid")


def test_a_training_matrix_refuses_the_closure_before_any_node_executes(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment, calls: _Calls
) -> None:
    cohort = outputs.cohort("boundary-run")
    outputs.probe(
        "boundary-consumer",
        references=[_subject(cohort, role_path="body.harvested", subject_id="harvested")],
    )

    with pytest.raises(ExternalOperationError) as caught:
        _fulfill(outputs, "boundary-consumer", environment=environment)

    record = caught.value.record()
    assert [node["key"] for node in record["operations"]] == ["campaign:boundary-run"]
    node = record["operations"][0]
    assert node["source_ref"] == "studies/boundary-run.envelope.json"
    assert node["operation_type"] == "feedbax.operation.train"
    assert node["capabilities"] == ["training"]
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
    with pytest.raises(ExternalOperationError) as caught:
        _fulfill(outputs, "joined", environment=environment)
    assert caught.value.record()["operations"][0]["unblocks"] == [
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
    requests = workflow_requests(_closure(outputs, target), environment=environment)
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
    execute_workflow(
        truncated_workflow(_closure(outputs, "role-consumer"), 1), environment=environment
    )
    # No analysis recipe is registered, so the analysis node stops the walk before
    # admission; its resolved request is what the binding decides.
    requests = workflow_requests(
        _closure(outputs, "role-consumer"),
        environment=environment,
        stop_at=LogicalKey("analysis", "role-consumer"),
    )
    assert [ref.role for ref in requests[-1].spec.inputs] == ["observed_states"]
    assert requests[-1].spec.inputs[0].kind == "EvaluationRunManifest"


def test_a_matrix_receipt_set_executes_analysis_in_row_order_and_rebuilds(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment, calls: _Calls
) -> None:
    calls.deterministic_analysis = True
    matrix = outputs.probe_matrix("set-source", rows=2)
    outputs.condensate(
        "set-consumer",
        references=[
            planned(
                matrix,
                role_path="inputs.evaluation",
                consumer=AnalysisReceiptSetBinding(alias="evaluation", role="evaluation"),
            )
        ],
    )
    closure = _closure(outputs, "set-consumer")

    run = execute_workflow(closure, environment=environment)
    produced = run.results[0].receipts
    assert len(produced) == 2
    assert calls.analysis == 1
    analysis_manifest = load_manifest(run.results[-1].receipt.path)
    assert [parent.id for parent in analysis_manifest.inputs] == [
        receipt.manifest_id for receipt in produced
    ]
    assert [parent.role for parent in analysis_manifest.inputs] == [
        "evaluation",
        "evaluation",
    ]

    requests = workflow_requests(closure, environment=environment)
    assert [parent.id for parent in requests[-1].spec.inputs] == [
        receipt.manifest_id for receipt in produced
    ]
    rebuilt = rebuild_workflow(closure, environment=environment)
    assert rebuilt.drifted == ()
    assert rebuilt.verification_order[-1] == ("analysis:set-consumer")


def test_repair_rebuilds_a_set_bound_analysis_from_all_admitted_rows(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment
) -> None:
    matrix = outputs.probe_matrix("repair-set", rows=2)
    outputs.condensate(
        "repair-consumer",
        references=[
            planned(
                matrix,
                role_path="inputs.evaluation",
                consumer=AnalysisReceiptSetBinding(alias="evaluation", role="evaluation"),
            )
        ],
    )
    closure = _closure(outputs, "repair-consumer")
    run = execute_workflow(closure, environment=environment)
    _mutate(run.results[-1].receipt.path, status="failed")

    repaired = repair_workflow_operation(closure, closure.target, environment=environment)
    manifest = load_manifest(repaired.receipt.path)
    assert [parent.id for parent in manifest.inputs] == [
        receipt.manifest_id for receipt in run.results[0].receipts
    ]


def test_a_one_row_matrix_is_still_ambiguous_for_a_singular_edge(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment
) -> None:
    matrix = outputs.probe_matrix("one-row-set", rows=1)
    outputs.condensate(
        "singular-reader",
        references=[
            planned(
                matrix,
                role_path="inputs.evaluation",
                consumer=AnalysisInputBinding(alias="evaluation", role="evaluation"),
            )
        ],
    )
    with pytest.raises(AmbiguousNodeReceiptError, match="cardinality is one"):
        workflow_requests(_closure(outputs, "singular-reader"), environment=environment)


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_a_partial_or_corrupt_matrix_set_never_reaches_analysis(
    outputs: QuillonOutputs,
    environment: FulfillmentEnvironment,
    calls: _Calls,
    damage: str,
) -> None:
    matrix = outputs.probe_matrix(f"{damage}-set", rows=2)
    outputs.condensate(
        f"{damage}-consumer",
        references=[
            planned(
                matrix,
                role_path="inputs.evaluation",
                consumer=AnalysisReceiptSetBinding(alias="evaluation", role="evaluation"),
            )
        ],
    )
    closure = _closure(outputs, f"{damage}-consumer")
    matrix_run = execute_workflow(truncated_workflow(closure, 1), environment=environment)
    damaged = matrix_run.results[0].receipts[1].path
    if damage == "missing":
        damaged.unlink()
    else:
        _mutate(damaged, status="failed")

    with pytest.raises(FulfillmentAdmissionError):
        workflow_requests(closure, environment=environment)
    assert calls.analysis == 0


@pytest.mark.parametrize("shape", ["empty", "duplicate"])
def test_an_empty_or_duplicate_receipt_set_is_refused(
    outputs: QuillonOutputs,
    environment: FulfillmentEnvironment,
    shape: str,
) -> None:
    matrix = outputs.probe_matrix(f"{shape}-set", rows=2)
    outputs.condensate(
        f"{shape}-consumer",
        references=[
            planned(
                matrix,
                role_path="inputs.evaluation",
                consumer=AnalysisReceiptSetBinding(alias="evaluation", role="evaluation"),
            )
        ],
    )
    closure = _closure(outputs, f"{shape}-consumer")
    matrix_run = execute_workflow(truncated_workflow(closure, 1), environment=environment)
    result = matrix_run.results[0]
    receipts = () if shape == "empty" else (result.receipts[0], result.receipts[0])
    binding = NodeBinding(
        closure=closure,
        environment=environment,
        fulfillments={closure.nodes[0].key: replace(result, receipts=receipts)},
    )
    edge = closure.plan.required_edges(closure.target)[0]
    with pytest.raises(AmbiguousNodeReceiptError, match=shape):
        binding.producer_receipts(edge)


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
    execute_workflow(truncated_workflow(_closure(outputs, "plate"), 1), environment=environment)
    requests = workflow_requests(
        _closure(outputs, "plate"),
        environment=environment,
        stop_at=LogicalKey("figure", "plate"),
    )
    figure = requests[-1]
    assert figure.node_kind == "figure"
    assert figure.runtime_inputs is not None
    assert [ref.role for ref in figure.runtime_inputs] == ["observed"]
    assert figure.spec["schema_id"] == "feedbax.spec.figure"
    # No contract, no authority: the input is bound as provenance and read from
    # no artifact, which is a statement an author is entitled to make.
    assert figure.runtime_input_authorities is None


def test_an_ordered_report_binds_a_figure_under_its_authored_role(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment
) -> None:
    figure = outputs.plate("produced-figure")
    outputs.emit(
        "role-report",
        {
            "schema_id": "feedbax.spec.report",
            "schema_version": "feedbax.spec.report.v1",
            "report_type": ORDERED_FIGURE_REPORT_TYPE,
            "params": {
                "schema_id": ORDERED_FIGURE_REPORT_PARAMS_SCHEMA_ID,
                "schema_version": ORDERED_FIGURE_REPORT_PARAMS_SCHEMA_VERSION,
                "title": "Role projection",
                "output_name": "role-projection.md",
                "sections": [
                    {
                        "title": "Velocity",
                        "figures": [
                            {
                                "input_role": "peak_velocity",
                                "figure_spec_sha256": figure.content_hash,
                                "caption": "Peak velocity",
                            }
                        ],
                        "tables": [],
                    }
                ],
            },
        },
        references=[
            planned(
                figure,
                role_path="params.sections.0.figures.0",
                consumer=ReportParentBinding(
                    parent_kind=FIGURE_SPEC_SCHEMA_ID,
                    parent_id="peak_velocity",
                ),
            )
        ],
    )

    run = execute_workflow(_closure(outputs, "role-report"), environment=environment)

    report = load_manifest(run.results[-1].receipt.path)
    assert [parent.role for parent in report.provenance.parents] == ["peak_velocity"]


def test_a_figure_authority_is_built_from_the_lock_contract_and_nothing_else(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment
) -> None:
    """The artifact half of a figure input comes from the lock, addressed by role."""
    source = outputs.probe("contracted-source")
    outputs.plate(
        "contracted-plate",
        references=[
            planned(
                source,
                role_path="runtime.states",
                consumer=FigureRuntimeInputBinding(
                    input_role="observed",
                    contract={
                        "input_role": "observed",
                        "artifact_role": "result",
                        "artifact_provider": "quillon.custody",
                        "payload_name": "observed_summary",
                        "payload_schema_id": "quillon.span_result",
                        "payload_schema_version": "quillon.span_result.v1",
                    },
                ),
            )
        ],
    )
    execute_workflow(
        truncated_workflow(_closure(outputs, "contracted-plate"), 1), environment=environment
    )
    requests = workflow_requests(
        _closure(outputs, "contracted-plate"),
        environment=environment,
        stop_at=LogicalKey("figure", "contracted-plate"),
    )

    figure = requests[-1]
    (authority,) = figure.runtime_input_authorities
    assert authority.input_role == "observed"
    assert authority.resolve_parent(figure.runtime_inputs) == figure.runtime_inputs[0]
    (payload,) = authority.artifact_payloads
    assert payload.name == "observed_summary"
    assert payload.manifest_role == "observed"
    assert payload.artifact_role == "result"
    assert payload.artifact_provider == "quillon.custody"
    assert payload.payload_schema_version == "quillon.span_result.v1"


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
                consumer=CheckpointInitializationBinding(mode="continue_from", row_id="misbound"),
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


def _matrix_request(outputs: QuillonOutputs, target: str, *, environment, upstream: int = 1):
    """Fulfil everything upstream of one matrix node and return its request."""
    closure = _closure(outputs, target)
    execute_workflow(truncated_workflow(closure, upstream), environment=environment)
    requests = workflow_requests(
        _closure(outputs, target),
        environment=environment,
        stop_at=LogicalKey("evaluation", target),
    )
    return requests[-1]


def _base_spec(stage: str, **params: Any) -> dict[str, Any]:
    """The evaluation run spec a content-pinned matrix base holds."""
    return {
        "schema_id": "feedbax.spec.evaluation_run",
        "schema_version": "feedbax.spec.evaluation_run.v1",
        "evaluation_type": PROBE_TYPE,
        "params": {"stage": stage, **params},
    }


def _write_base_spec(outputs: QuillonOutputs, relative: str, payload: dict[str, Any]) -> str:
    """Write one content-pinned base into the repo and return its canonical pin."""
    path = outputs.root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return sha256_bytes(canonical_json_bytes(payload))


def _pinned_matrix(relative: str, sha256: str) -> dict[str, Any]:
    """An axis matrix over a content-pinned base, which no injection may touch."""
    return {
        "schema_id": "feedbax.spec.evaluation_run_matrix",
        "schema_version": "feedbax.spec.evaluation_run_matrix.v3",
        "base": {"ref": relative, "sha256": sha256},
        "axes": [{"id": "stage", "values": [{"id": "one"}, {"id": "two"}]}],
    }


def test_an_evaluation_matrix_binds_a_closure_edge_as_a_staged_parent(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment
) -> None:
    """A matrix binds its required input the way a matrix reaches a row."""
    source = outputs.probe("matrix-source")
    outputs.probe_matrix(
        "matrix-consumer",
        references=[_subject(source, role_path="body.subject", subject_id="subject")],
    )

    request = _matrix_request(outputs, "matrix-consumer", environment=environment)

    staged = request.matrix["staged_parents"]
    assert list(staged) == ["subject"]
    parent = staged["subject"]["parent"]
    assert parent["kind"] == "EvaluationRunManifest"
    assert parent["role"] == "subject"
    assert parent["metadata"]["ref_schema_id"] == "feedbax.ref.authenticated_manifest"
    assert request.matrix["base"]["params"]["staged_prerequisites"] == staged


def test_a_staged_matrix_parent_reaches_every_materialized_row(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment
) -> None:
    """Every row inherits the prerequisite, which is what makes the matrix legal."""
    from feedbax.analysis.evaluation import materialize_evaluation_run_matrix

    source = outputs.probe("row-source")
    outputs.probe_matrix(
        "row-consumer",
        references=[_subject(source, role_path="body.subject", subject_id="subject")],
    )
    request = _matrix_request(outputs, "row-consumer", environment=environment)

    rows = materialize_evaluation_run_matrix(
        request.matrix, registry=environment.registries.evaluation_recipes
    )

    assert [row.row_id for row in rows] == ["row-consumer-0", "row-consumer-1"]
    for row in rows:
        assert row.payload.params["staged_prerequisites"] == request.matrix["staged_parents"]


def test_a_staged_matrix_never_runs_without_a_declared_execution_context(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment
) -> None:
    """Where the bound bytes live is the environment's declaration, never a guess."""
    source = outputs.probe("context-source")
    outputs.probe_matrix(
        "context-consumer",
        references=[_subject(source, role_path="body.subject", subject_id="subject")],
    )
    with pytest.raises(ValueError, match="staged execution context"):
        _fulfill(outputs, "context-consumer", environment=environment)


def test_a_matrix_that_already_states_a_staged_parent_refuses_to_bind_it_twice(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment
) -> None:
    source = outputs.probe("twice-source")
    outputs.emit(
        "twice-consumer",
        {
            "schema_id": "feedbax.spec.evaluation_run_matrix",
            "schema_version": "feedbax.spec.evaluation_run_matrix.v3",
            "base": {
                "schema_id": "feedbax.spec.evaluation_run",
                "schema_version": "feedbax.spec.evaluation_run.v1",
                "evaluation_type": PROBE_TYPE,
                "params": {"stage": "twice-consumer"},
            },
            "rows": [{"row_id": "twice-consumer-0"}],
            "staged_parents": {
                "subject": {
                    "parent": {
                        "kind": "EvaluationRunManifest",
                        "id": "feedbax-evaluation-run:preauthored",
                        "role": "subject",
                        "metadata": {
                            "manifest_sha256": DIGEST,
                            "ref_schema_id": "feedbax.ref.authenticated_manifest",
                            "ref_schema_version": "feedbax.ref.authenticated_manifest.v1",
                            "size_bytes": 1,
                        },
                    }
                }
            },
        },
        references=[_subject(source, role_path="body.subject", subject_id="subject")],
    )
    with pytest.raises(NodeLoweringError, match="cannot authenticate a parent"):
        _fulfill(outputs, "twice-consumer", environment=environment)


def test_a_matrix_binds_every_typed_prerequisite_its_lock_authenticates(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment
) -> None:
    """One matrix binds as many staged parents as its lock has required roles."""
    from feedbax.analysis.evaluation import materialize_evaluation_run_matrix

    first = outputs.probe("multi-first")
    second = outputs.probe("multi-second")
    outputs.probe_matrix(
        "multi-consumer",
        references=[
            _subject(first, role_path="body.first", subject_id="baseline"),
            _subject(second, role_path="body.second", subject_id="perturbed"),
        ],
    )

    request = _matrix_request(outputs, "multi-consumer", environment=environment, upstream=2)

    staged = request.matrix["staged_parents"]
    assert list(staged) == ["baseline", "perturbed"]
    assert [staged[name]["parent"]["role"] for name in staged] == ["baseline", "perturbed"]
    assert staged["baseline"]["parent"]["id"] != staged["perturbed"]["parent"]["id"]
    assert request.matrix["base"]["params"]["staged_prerequisites"] == staged

    rows = materialize_evaluation_run_matrix(
        request.matrix, registry=environment.registries.evaluation_recipes
    )
    for row in rows:
        assert row.payload.params["staged_prerequisites"] == staged


def test_a_matrix_may_restate_the_staged_parent_its_lock_authenticates(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment
) -> None:
    """A restated entry is read, not trusted: its provider survives, its parent must agree."""
    source = outputs.probe("restate-source")
    outputs.probe_matrix(
        "restate-probe",
        references=[_subject(source, role_path="body.subject", subject_id="subject")],
    )
    bound = _matrix_request(outputs, "restate-probe", environment=environment)
    authenticated = bound.matrix["staged_parents"]["subject"]["parent"]

    outputs.emit(
        "restate-consumer",
        {
            "schema_id": "feedbax.spec.evaluation_run_matrix",
            "schema_version": "feedbax.spec.evaluation_run_matrix.v3",
            "base": _base_spec("restate-consumer"),
            "rows": [{"row_id": "restate-consumer-0"}],
            "staged_parents": {
                "subject": {"parent": authenticated, "artifact_provider": "evidence"}
            },
        },
        references=[_subject(source, role_path="body.subject", subject_id="subject")],
    )

    request = _matrix_request(outputs, "restate-consumer", environment=environment)

    staged = request.matrix["staged_parents"]
    assert staged["subject"]["parent"] == authenticated
    assert staged["subject"]["artifact_provider"] == "evidence"
    assert request.matrix["base"]["params"]["staged_prerequisites"] == staged


def test_a_matrix_may_not_state_a_staged_parent_its_lock_never_bound(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment
) -> None:
    """A name only the document states has no authenticated source."""
    source = outputs.probe("unbound-source")
    outputs.emit(
        "unbound-consumer",
        {
            "schema_id": "feedbax.spec.evaluation_run_matrix",
            "schema_version": "feedbax.spec.evaluation_run_matrix.v3",
            "base": _base_spec("unbound-consumer"),
            "rows": [{"row_id": "unbound-consumer-0"}],
            "staged_parents": {
                "extra": {
                    "parent": {
                        "kind": "EvaluationRunManifest",
                        "id": "feedbax-evaluation-run:extra",
                        "role": "extra",
                        "metadata": {
                            "manifest_sha256": DIGEST,
                            "ref_schema_id": "feedbax.ref.authenticated_manifest",
                            "ref_schema_version": "feedbax.ref.authenticated_manifest.v1",
                            "size_bytes": 1,
                        },
                    }
                }
            },
        },
        references=[_subject(source, role_path="body.subject", subject_id="subject")],
    )
    with pytest.raises(NodeLoweringError, match="cannot authenticate a parent"):
        _fulfill(outputs, "unbound-consumer", environment=environment)


def test_a_restated_staged_parent_may_name_the_artifacts_own_role(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment
) -> None:
    """The paired-controller shape: the document's role and the lock's differ.

    A compiled matrix that inherits ``staged_parents`` from a tracked base
    restates the parent as the artifact it is — role ``evaluation_run`` — while
    the binding name the lock owns is what the *consumer* calls that slot, here
    ``trial_bank``. Those are two true statements about different things, and the
    restatement's job is only to agree about which artifact.

    So the bound parent takes the lock's role, and the restatement is checked on
    kind, id, and byte profile alone.
    """
    subject = outputs.probe("paired-subject")
    bank = outputs.probe("paired-bank")
    execute_workflow(_closure(outputs, "paired-subject"), environment=environment)
    produced = _fulfill(outputs, "paired-bank", environment=environment).results[0].receipt
    raw = produced.path.read_bytes()
    outputs.emit(
        "paired-consumer",
        {
            "schema_id": "feedbax.spec.evaluation_run_matrix",
            "schema_version": "feedbax.spec.evaluation_run_matrix.v3",
            "base": _base_spec("paired-consumer"),
            "rows": [{"row_id": "paired-consumer-0"}],
            "staged_parents": {
                "trial_bank": {
                    "parent": {
                        "kind": "EvaluationRunManifest",
                        "id": produced.manifest_id,
                        # The artifact's own role, exactly as the corpus base
                        # states it, and deliberately not the binding name.
                        "role": "evaluation_run",
                        "metadata": {
                            "ref_schema_id": "feedbax.ref.authenticated_manifest",
                            "ref_schema_version": "feedbax.ref.authenticated_manifest.v1",
                            "manifest_sha256": hashlib.sha256(raw).hexdigest(),
                            "size_bytes": len(raw),
                        },
                    },
                    "artifact_provider": "shared",
                }
            },
        },
        references=[
            _subject(subject, role_path="body.subject", subject_id="subject"),
            _subject(bank, role_path="body.trial_bank", subject_id="trial_bank"),
        ],
    )

    request = _matrix_request(outputs, "paired-consumer", environment=environment, upstream=2)

    staged = request.matrix["staged_parents"]
    assert sorted(staged) == ["subject", "trial_bank"]
    # The lock's consumer binding decides the role, so the restated
    # ``evaluation_run`` is not what gets bound.
    assert staged["trial_bank"]["parent"]["role"] == "trial_bank"
    assert staged["trial_bank"]["parent"]["id"] == produced.manifest_id
    assert staged["trial_bank"]["parent"]["metadata"]["manifest_sha256"] == (
        hashlib.sha256(raw).hexdigest()
    )
    # A non-authenticating field the document owns still travels.
    assert staged["trial_bank"]["artifact_provider"] == "shared"
    assert request.matrix["base"]["params"]["staged_prerequisites"] == staged


def test_a_restated_staged_parent_naming_another_artifact_still_refuses(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment
) -> None:
    """Relaxing the role comparison does not relax the identity comparison."""
    subject = outputs.probe("disagree-subject")
    bank = outputs.probe("disagree-bank")
    execute_workflow(_closure(outputs, "disagree-subject"), environment=environment)
    _fulfill(outputs, "disagree-bank", environment=environment)
    outputs.emit(
        "disagree-consumer",
        {
            "schema_id": "feedbax.spec.evaluation_run_matrix",
            "schema_version": "feedbax.spec.evaluation_run_matrix.v3",
            "base": _base_spec("disagree-consumer"),
            "rows": [{"row_id": "disagree-consumer-0"}],
            "staged_parents": {
                "trial_bank": {
                    "parent": {
                        "kind": "EvaluationRunManifest",
                        "id": "feedbax-evaluation-run:some-other-bank",
                        "role": "evaluation_run",
                    }
                }
            },
        },
        references=[
            _subject(subject, role_path="body.subject", subject_id="subject"),
            _subject(bank, role_path="body.trial_bank", subject_id="trial_bank"),
        ],
    )

    with pytest.raises(NodeLoweringError) as caught:
        _matrix_request(outputs, "disagree-consumer", environment=environment, upstream=2)

    message = str(caught.value)
    assert "some-other-bank" in message
    assert "may only restate the artifact the lock binds" in message


def test_a_restated_staged_parent_whose_digest_disagrees_refuses(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment
) -> None:
    """Same artifact, different bytes: the restatement disagrees about the run."""
    subject = outputs.probe("digest-subject")
    bank = outputs.probe("digest-bank")
    execute_workflow(_closure(outputs, "digest-subject"), environment=environment)
    produced = _fulfill(outputs, "digest-bank", environment=environment).results[0].receipt
    outputs.emit(
        "digest-consumer",
        {
            "schema_id": "feedbax.spec.evaluation_run_matrix",
            "schema_version": "feedbax.spec.evaluation_run_matrix.v3",
            "base": _base_spec("digest-consumer"),
            "rows": [{"row_id": "digest-consumer-0"}],
            "staged_parents": {
                "trial_bank": {
                    "parent": {
                        "kind": "EvaluationRunManifest",
                        "id": produced.manifest_id,
                        "role": "evaluation_run",
                        "metadata": {
                            "ref_schema_id": "feedbax.ref.authenticated_manifest",
                            "ref_schema_version": "feedbax.ref.authenticated_manifest.v1",
                            "manifest_sha256": DIGEST,
                            "size_bytes": 1,
                        },
                    }
                }
            },
        },
        references=[
            _subject(subject, role_path="body.subject", subject_id="subject"),
            _subject(bank, role_path="body.trial_bank", subject_id="trial_bank"),
        ],
    )

    with pytest.raises(NodeLoweringError) as caught:
        _matrix_request(outputs, "digest-consumer", environment=environment, upstream=2)

    assert "byte profile" in str(caught.value)


def test_a_pinned_matrix_base_is_bound_without_being_touched(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment
) -> None:
    """Pinned bytes keep their pin; the authenticated parent binds at matrix level."""
    source = outputs.probe("pinned-source")
    pin = _write_base_spec(outputs, "bases/pinned.json", _base_spec("pinned"))
    outputs.emit(
        "pinned-consumer",
        _pinned_matrix("bases/pinned.json", pin),
        references=[_subject(source, role_path="body.subject", subject_id="subject")],
    )

    request = _matrix_request(outputs, "pinned-consumer", environment=environment)

    assert request.matrix["base"] == {"ref": "bases/pinned.json", "sha256": pin}
    staged = request.matrix["staged_parents"]
    assert list(staged) == ["subject"]
    assert staged["subject"]["parent"]["kind"] == "EvaluationRunManifest"


def test_a_pinned_matrix_row_that_ignores_the_bound_parent_refuses(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment
) -> None:
    """The parent is never silently unconsumed: every row must reference it."""
    from feedbax.analysis.evaluation import materialize_evaluation_run_matrix

    source = outputs.probe("ignored-source")
    pin = _write_base_spec(outputs, "bases/ignored.json", _base_spec("ignored"))
    outputs.emit(
        "ignored-consumer",
        _pinned_matrix("bases/ignored.json", pin),
        references=[_subject(source, role_path="body.subject", subject_id="subject")],
    )
    request = _matrix_request(outputs, "ignored-consumer", environment=environment)

    with pytest.raises(ValueError, match="does not reference staged parent"):
        materialize_evaluation_run_matrix(
            request.matrix,
            registry=environment.registries.evaluation_recipes,
            repo_root=outputs.root,
        )


def test_a_pinned_matrix_row_that_states_the_bound_parent_materializes(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment
) -> None:
    """A pinned base that already consumes the parent runs under the lock's binding."""
    from feedbax.analysis.evaluation import materialize_evaluation_run_matrix

    source = outputs.probe("consumed-source")
    first = _write_base_spec(outputs, "bases/consumed.json", _base_spec("consumed"))
    outputs.emit(
        "consumed-consumer",
        _pinned_matrix("bases/consumed.json", first),
        references=[_subject(source, role_path="body.subject", subject_id="subject")],
    )
    staged = _matrix_request(outputs, "consumed-consumer", environment=environment).matrix[
        "staged_parents"
    ]

    consuming = _base_spec("consumed", staged_prerequisites=staged)
    second = _write_base_spec(outputs, "bases/consumed.json", consuming)
    outputs.emit(
        "consumed-consumer",
        _pinned_matrix("bases/consumed.json", second),
        references=[_subject(source, role_path="body.subject", subject_id="subject")],
    )
    request = _matrix_request(outputs, "consumed-consumer", environment=environment)

    rows = materialize_evaluation_run_matrix(
        request.matrix,
        registry=environment.registries.evaluation_recipes,
        repo_root=outputs.root,
    )

    assert [row.row_id for row in rows] == ["stage-one", "stage-two"]
    for row in rows:
        assert row.payload.params["staged_prerequisites"] == staged


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
    assert (
        middle_parents[0].metadata["manifest_sha256"]
        == hashlib.sha256(upstream.receipt.path.read_bytes()).hexdigest()
    )
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
    partial = truncated_workflow(_closure(outputs, target), boundary)
    interrupted = execute_workflow(partial, environment=environment)
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
    assert (
        bound[0].metadata["manifest_sha256"]
        == hashlib.sha256(produced.path.read_bytes()).hexdigest()
    )
    assert calls.evaluation == 1


def test_an_authenticated_receipt_reference_binds_the_bytes_it_quoted(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment
) -> None:
    outputs.probe("quoted-source")
    produced = _fulfill(outputs, "quoted-source", environment=environment).results[0].receipt
    raw = produced.path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    outputs.bulletin(
        "quoted-consumer",
        references=[
            AuthenticatedReceiptReference(
                manifest_kind="EvaluationRunManifest",
                manifest_id=produced.manifest_id,
                manifest_sha256=digest,
                size_bytes=len(raw),
                role_path="body.quoted",
                consumer=ReportParentBinding(parent_kind="probe", parent_id="quoted"),
            )
        ],
    )
    run = _fulfill(outputs, "quoted-consumer", environment=environment)
    bound = load_manifest(run.results[0].receipt.path).provenance.parents
    assert [ref.id for ref in bound] == [produced.manifest_id]
    # The bound ref carries the same profile the lock quoted, so the digest the
    # consumer records is the one the compile authenticated rather than a fresh
    # one minted from whatever happened to be resolvable.
    assert bound[0].metadata["manifest_sha256"] == digest
    assert bound[0].metadata["size_bytes"] == len(raw)


def test_an_authenticated_receipt_refuses_when_the_stored_bytes_disagree(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment, calls: _Calls
) -> None:
    """Same kind, same id, same completed status, different bytes: a refusal.

    Kind, id, and status address a receipt and say it finished; none of them says
    the bytes are the ones the compile read. A rerun that produced different
    results lands at exactly the same canonical location with all three intact,
    so authentication has to be the byte profile the lock quoted or it is not
    authentication at all.
    """
    outputs.probe("substituted-source")
    produced = _fulfill(outputs, "substituted-source", environment=environment).results[0].receipt
    raw = produced.path.read_bytes()
    quoted_digest = hashlib.sha256(raw).hexdigest()
    outputs.bulletin(
        "substituted-consumer",
        references=[
            AuthenticatedReceiptReference(
                manifest_kind="EvaluationRunManifest",
                manifest_id=produced.manifest_id,
                manifest_sha256=quoted_digest,
                size_bytes=len(raw),
                role_path="body.quoted",
                consumer=ReportParentBinding(parent_kind="probe", parent_id="quoted"),
            )
        ],
    )
    # Rewrite the receipt in place, preserving every field the resolver checks.
    _mutate(produced.path, metadata={"rerun": "second-pass"})
    substituted = load_manifest(produced.path)
    assert substituted.kind == "EvaluationRunManifest"
    assert substituted.id == produced.manifest_id
    assert substituted.status == "completed"
    found_raw = produced.path.read_bytes()
    assert hashlib.sha256(found_raw).hexdigest() != quoted_digest

    before = calls.report
    with pytest.raises(ExternalReceiptAuthenticationError) as caught:
        _fulfill(outputs, "substituted-consumer", environment=environment)

    message = str(caught.value)
    assert quoted_digest in message
    assert hashlib.sha256(found_raw).hexdigest() in message
    detail = caught.value.record_detail()
    assert detail["lock_manifest_sha256"] == quoted_digest
    assert detail["found_manifest_sha256"] == hashlib.sha256(found_raw).hexdigest()
    assert detail["consumer"] == "report:substituted-consumer"
    assert detail["role_path"] == ["body", "quoted"]
    assert calls.report == before, "a refused consumer never runs"


def test_a_receipt_locator_binds_without_a_byte_profile_to_check(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment
) -> None:
    """A locator quoted no bytes, so changed bytes at its address still bind.

    This is the honest complement of the refusal above rather than a hole in it:
    a locator is the record of a receipt that did not exist at compile time, and
    the compile therefore has nothing to authenticate it against. What the two
    cases together prove is that the *lock* decides which check applies.
    """
    outputs.probe("locator-source")
    produced = _fulfill(outputs, "locator-source", environment=environment).results[0].receipt
    outputs.bulletin(
        "locator-consumer",
        references=[
            ReceiptLocatorReference(
                manifest_kind="EvaluationRunManifest",
                manifest_id=produced.manifest_id,
                role_path="body.quoted",
                consumer=ReportParentBinding(parent_kind="probe", parent_id="quoted"),
            )
        ],
    )
    _mutate(produced.path, metadata={"rerun": "second-pass"})

    run = _fulfill(outputs, "locator-consumer", environment=environment)

    bound = load_manifest(run.results[0].receipt.path).provenance.parents
    assert (
        bound[0].metadata["manifest_sha256"]
        == hashlib.sha256(produced.path.read_bytes()).hexdigest()
    )


def test_stripping_a_plan_edges_byte_profile_refuses_at_preflight(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment, calls: _Calls
) -> None:
    """The reviewer's reproduction: a durable plan cannot downgrade a lock edge.

    A plan travels apart from the locks it was derived from, so everything
    authenticating it carries is a copy. Removing ``manifest_sha256`` and
    ``size_bytes`` from an ``authenticated_receipt`` edge leaves a document whose
    node hashes all still match, and whose edge now reads as a bare locator —
    which is exactly the downgrade the lock forbids.
    """
    outputs.probe("downgrade-source")
    produced = _fulfill(outputs, "downgrade-source", environment=environment).results[0].receipt
    raw = produced.path.read_bytes()
    outputs.bulletin(
        "downgrade-consumer",
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
    index = read_compiled_outputs(outputs.output_directory)
    document = derive_workflow_plan(index, target="downgrade-consumer").document()
    stripped = 0
    for edge in document["edges"]:
        external = edge.get("external")
        if external and "manifest_sha256" in external:
            external.pop("manifest_sha256")
            external.pop("size_bytes")
            stripped += 1
    assert stripped == 1, "the fixture must carry exactly one authenticated edge"
    before = calls.report
    with pytest.raises(WorkflowPlanIdentityError):
        workflow_plan_from_document(document)
    assert calls.report == before, "a refused closure never runs"


def test_a_plan_edge_naming_a_manifest_the_lock_never_named_refuses(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment
) -> None:
    """Substitution is the same defect as downgrade, and refuses the same way."""
    outputs.probe("substitute-source")
    produced = _fulfill(outputs, "substitute-source", environment=environment).results[0].receipt
    outputs.bulletin(
        "substitute-consumer",
        references=[
            ReceiptLocatorReference(
                manifest_kind="EvaluationRunManifest",
                manifest_id=produced.manifest_id,
                role_path="body.quoted",
                consumer=ReportParentBinding(parent_kind="probe", parent_id="quoted"),
            )
        ],
    )
    index = read_compiled_outputs(outputs.output_directory)
    document = derive_workflow_plan(index, target="substitute-consumer").document()
    for edge in document["edges"]:
        if edge.get("external"):
            edge["external"]["manifest_id"] = "feedbax-evaluation-run:substituted"

    with pytest.raises(WorkflowPlanIdentityError):
        workflow_plan_from_document(document)


def test_a_plan_that_drops_an_input_the_lock_states_refuses(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment
) -> None:
    """Deletion is not a quieter downgrade; the lock states the input either way."""
    outputs.probe("dropped-source")
    produced = _fulfill(outputs, "dropped-source", environment=environment).results[0].receipt
    outputs.bulletin(
        "dropped-consumer",
        references=[
            ReceiptLocatorReference(
                manifest_kind="EvaluationRunManifest",
                manifest_id=produced.manifest_id,
                role_path="body.quoted",
                consumer=ReportParentBinding(parent_kind="probe", parent_id="quoted"),
            )
        ],
    )
    index = read_compiled_outputs(outputs.output_directory)
    document = derive_workflow_plan(index, target="dropped-consumer").document()
    document["edges"] = [edge for edge in document["edges"] if not edge.get("external")]

    with pytest.raises(WorkflowPlanIdentityError):
        workflow_plan_from_document(document)


def test_a_plan_derived_from_its_own_locks_reconciles(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment
) -> None:
    """The reconciliation is exact, so an untouched plan must survive it."""
    target = _chain(outputs)
    index = read_compiled_outputs(outputs.output_directory)
    plan = derive_workflow_plan(index, target=target)

    round_tripped = workflow_plan_from_document(plan.document())

    closure = prepare_workflow(round_tripped, read_compiled_outputs(outputs.output_directory))
    assert closure.order == _closure(outputs, target).order


def test_the_bytes_authenticated_are_the_bytes_bound(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment, monkeypatch
) -> None:
    """No second read exists for a swap to slip into.

    The receipt file is read once. This spy returns the real bytes on that read
    and tampered bytes on any read after it, so a resolver that re-opened the
    path to mint its digest would emit the tampered digest. The bound ref must
    carry the lock's profile, and the spy must see exactly one read.
    """
    outputs.probe("single-read-source")
    produced = _fulfill(outputs, "single-read-source", environment=environment).results[0].receipt
    raw = produced.path.read_bytes()
    lock_digest = hashlib.sha256(raw).hexdigest()
    tampered = json.dumps({**json.loads(raw), "metadata": {"swapped": True}}).encode()
    assert hashlib.sha256(tampered).hexdigest() != lock_digest

    reads: list[Path] = []
    real_read_bytes = Path.read_bytes

    def spying_read_bytes(self: Path) -> bytes:
        if self == produced.path:
            reads.append(self)
            if len(reads) > 1:
                return tampered
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", spying_read_bytes)

    parent, path = external_parent_ref(
        ExternalReceiptRecord(
            manifest_kind="EvaluationRunManifest",
            manifest_id=produced.manifest_id,
            manifest_sha256=lock_digest,
            size_bytes=len(raw),
        ),
        role="quoted",
        root=environment.root,
    )

    assert reads == [produced.path], "the receipt is read exactly once"
    assert path == produced.path
    assert parent.metadata["manifest_sha256"] == lock_digest
    assert parent.metadata["size_bytes"] == len(raw)


def test_a_locator_binds_the_profile_of_the_bytes_it_actually_read(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment
) -> None:
    """A locator quoted nothing, so the single read is the only authority there."""
    outputs.probe("locator-profile-source")
    produced = (
        _fulfill(outputs, "locator-profile-source", environment=environment).results[0].receipt
    )

    parent, _path = external_parent_ref(
        ExternalReceiptRecord(
            manifest_kind="EvaluationRunManifest", manifest_id=produced.manifest_id
        ),
        role="quoted",
        root=environment.root,
    )

    raw = produced.path.read_bytes()
    assert parent.metadata["manifest_sha256"] == hashlib.sha256(raw).hexdigest()
    assert parent.metadata["size_bytes"] == len(raw)


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
    produced = _fulfill(outputs, "incomplete-source", environment=environment).results[0].receipt
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
    """A certified omission sits beside a bound role without blocking it.

    The omission is stated where its rule actually decides — a figure input slot
    row expansion fills per row — because certification evaluates the rule rather
    than looking its id up, and the same decision on a report parent is refused.
    """
    from feedbax.contracts.applicability_rules import (
        PER_ROW_FIGURE_INPUT_RULE,
        certify_not_applicable,
    )

    source = outputs.probe("partial-source")
    outputs.plate(
        "partial-plate",
        references=[
            planned(
                source,
                role_path="runtime.states",
                consumer=FigureRuntimeInputBinding(input_role="nominal"),
            ),
            certify_not_applicable("inputs.observed", PER_ROW_FIGURE_INPUT_RULE),
        ],
    )
    key = LogicalKey("figure", "partial-plate")
    closure = _closure(outputs, "partial-plate")

    assert [edge.role_path for edge in closure.plan.required_edges(key)] == [("runtime", "states")]
    assert [edge.role_path for edge in closure.plan.certified_omissions(consumer=key)] == [
        ("inputs", "observed")
    ]

    run = _fulfill(outputs, "partial-plate", environment=environment)
    figure = run.results[-1]
    assert figure.node_kind == "figure"
    manifest = load_manifest(figure.receipt.path)
    assert [parent.role for parent in manifest.provenance.parents] == ["nominal"]


def test_a_per_row_figure_input_binds_no_single_manifest_edge(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment
) -> None:
    """Row expansion fills the role per row, so the closure waits on nothing for it."""
    from feedbax.contracts.experiment_compile_lock import NotApplicableReference
    from feedbax.envelope.compile import PER_ROW_INPUT_REASON, PER_ROW_INPUT_RULE_ID

    outputs.plate(
        "expanded-plate",
        references=[
            NotApplicableReference(
                role_path="inputs.observed",
                basis="compiler_rule",
                reason=PER_ROW_INPUT_REASON,
                rule_id=PER_ROW_INPUT_RULE_ID,
            )
        ],
    )
    closure = _closure(outputs, "expanded-plate")

    key = LogicalKey("figure", "expanded-plate")
    assert closure.plan.required_edges(key) == ()
    omission = closure.plan.certified_omissions(consumer=key)
    assert [edge.role_path for edge in omission] == [("inputs", "observed")]
    assert omission[0].rule == PER_ROW_INPUT_RULE_ID
    assert omission[0].producer is None and omission[0].external is None
    assert closure.order == ("figure:expanded-plate",)
    requests = workflow_requests(closure, environment=environment, stop_at=key)
    assert requests[-1].node_kind == "figure"
    assert requests[-1].runtime_inputs is None


# --------------------------------------------------------------------------
# Rebuild as verification, repair as recovery
# --------------------------------------------------------------------------


def test_rebuilding_an_intact_closure_reports_no_drift_and_preserves_receipts(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment
) -> None:
    target = _chain(outputs)
    run = _fulfill(outputs, target, environment=environment)
    before = {result.receipt.path: result.receipt.path.read_bytes() for result in run.results}

    rebuilt = rebuild_workflow(_closure(outputs, target), environment=environment)
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
        rebuild_workflow(_closure(outputs, "drifting-probe"), environment=environment)
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
        rebuild_workflow(_closure(outputs, target), environment=environment)
    assert "artifact_sha256_mismatch" in caught.value.outcome.codes
    assert calls.evaluation == 2, "corruption is custody loss, not drift, so nothing rebuilt"


def test_repair_promotes_a_revalidated_candidate_and_records_the_custody_event(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment
) -> None:
    target = _chain(outputs)
    run = _fulfill(outputs, target, environment=environment)
    receipt = run.results[1].receipt
    _mutate(receipt.path, status="failed")

    result = repair_workflow_operation(
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
        repair_workflow_operation(
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
        workflow_requests(_closure(outputs, "matrix-reader"), environment=environment)


# --------------------------------------------------------------------------
# Class invariants: one read, one edge per role, and no comparison that skips
# --------------------------------------------------------------------------


def _external_consumer(
    outputs: QuillonOutputs,
    *,
    layer: str,
    name: str,
    manifest_id: str,
    digest: str,
    size: int,
):
    """Emit one node of *layer* whose only input is an authenticated receipt.

    Every consumer kind binds by its own closed consumer binding, so this is the
    one place the four spellings differ; what is under test is that they all
    reach the same single read.
    """
    reference_kwargs = dict(
        manifest_kind="EvaluationRunManifest",
        manifest_id=manifest_id,
        manifest_sha256=digest,
        size_bytes=size,
        role_path="body.quoted",
    )
    if layer == "evaluation":
        return outputs.probe(
            name,
            references=[
                AuthenticatedReceiptReference(
                    **reference_kwargs,
                    consumer=EvaluationSubjectBinding(subject_id="quoted"),
                )
            ],
        )
    if layer == "analysis":
        return outputs.condensate(
            name,
            references=[
                AuthenticatedReceiptReference(
                    **reference_kwargs,
                    consumer=AnalysisInputBinding(alias="quoted", role="observed_states"),
                )
            ],
        )
    if layer == "figure":
        return outputs.plate(
            name,
            references=[
                AuthenticatedReceiptReference(
                    **reference_kwargs,
                    consumer=FigureRuntimeInputBinding(input_role="observed"),
                )
            ],
        )
    return outputs.bulletin(
        name,
        references=[
            AuthenticatedReceiptReference(
                **reference_kwargs,
                consumer=ReportParentBinding(parent_kind="probe", parent_id="quoted"),
            )
        ],
    )


def _bound_parents_of(request) -> tuple:
    """Return every ``ParentRef`` one lowered request binds, whatever its kind."""
    parents = []
    spec = getattr(request, "spec", None)
    if spec is not None and not isinstance(spec, dict):
        parents.extend(getattr(spec, "inputs", ()) or ())
    runtime = getattr(request, "runtime_inputs", None)
    if runtime:
        parents.extend(runtime)
    exact = getattr(request, "exact_parents", None)
    if exact is not None:
        parents.extend(entry.parent for entry in exact.parents)
    return tuple(parents)


@pytest.mark.parametrize("layer", ["evaluation", "analysis", "figure", "report"])
def test_lowering_reads_each_external_input_exactly_once(
    outputs: QuillonOutputs,
    environment: FulfillmentEnvironment,
    monkeypatch: pytest.MonkeyPatch,
    layer: str,
) -> None:
    """The single-read invariant, over every node kind that binds an input.

    One read exists for each external input, and nothing downstream opens the
    path again. The spy proves both halves at once: it hands back the real bytes
    on the first read of the receipt and tampered bytes on every read after it,
    so any second read anywhere in the lowering — binding a parent, restating a
    profile, resolving an execution locator — would surface as a bound digest
    that is not the one the lock quoted. A report binds both a parent and the
    location it executes from, which is exactly the pair that used to resolve
    twice.
    """
    outputs.probe(f"{layer}-single-read-source")
    produced = (
        _fulfill(outputs, f"{layer}-single-read-source", environment=environment).results[0].receipt
    )
    raw = produced.path.read_bytes()
    lock_digest = hashlib.sha256(raw).hexdigest()
    tampered = json.dumps({**json.loads(raw), "metadata": {"swapped": True}}).encode()
    assert hashlib.sha256(tampered).hexdigest() != lock_digest

    name = f"{layer}-single-read-consumer"
    _external_consumer(
        outputs,
        layer=layer,
        name=name,
        manifest_id=produced.manifest_id,
        digest=lock_digest,
        size=len(raw),
    )
    closure = _closure(outputs, name)
    key = LogicalKey(layer, name)

    reads: list[Path] = []
    real_read_bytes = Path.read_bytes

    def spying_read_bytes(self: Path) -> bytes:
        if self == produced.path:
            reads.append(self)
            if len(reads) > 1:
                return tampered
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", spying_read_bytes)
    requests = workflow_requests(closure, environment=environment, stop_at=key)

    assert reads == [produced.path], "one external input, one read"
    parents = _bound_parents_of(requests[-1])
    assert parents, "the consumer binds the external receipt it declares"
    for parent in parents:
        assert parent.id == produced.manifest_id
        assert parent.metadata["manifest_sha256"] == lock_digest
        assert parent.metadata["size_bytes"] == len(raw)


def test_an_admitted_receipt_binds_the_digest_admission_read(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment, monkeypatch
) -> None:
    """Bytes substituted after admission are never blessed with their own digest.

    The producer here is an in-closure node, so its receipt is admitted rather
    than resolved from a lock quote. Admission is the read; the consumer's
    binding must restate what admission authenticated, so a substitution landing
    between the two is a bound digest that no longer matches the file — which is
    the honest outcome — and never a fresh digest of the replacement.
    """
    target = _chain(outputs)
    _fulfill(outputs, target, environment=environment)
    closure = _closure(outputs, target)
    requests = workflow_requests(closure, environment=environment)
    report_request = requests[-1]
    admitted_digest = report_request.spec.inputs[0].metadata["manifest_sha256"]
    upstream = canonical_manifest_path(
        "EvaluationRunManifest", report_request.spec.inputs[0].id, root=environment.root
    )
    assert hashlib.sha256(upstream.read_bytes()).hexdigest() == admitted_digest

    # Substitute same-kind, same-id, still-completed bytes after admission read
    # them, and prove the binding does not follow the substitution.
    real_read_bytes = Path.read_bytes
    reads: list[Path] = []
    substituted = json.dumps(
        {**json.loads(upstream.read_bytes()), "metadata": {"rerun": "second-pass"}}
    ).encode()

    def spying_read_bytes(self: Path) -> bytes:
        if self == upstream:
            reads.append(self)
            if len(reads) > 1:
                return substituted
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", spying_read_bytes)
    rebound = workflow_requests(closure, environment=environment)[-1]

    assert reads == [upstream], "the admitted receipt is read once per walk"
    assert rebound.spec.inputs[0].metadata["manifest_sha256"] == admitted_digest
    assert (
        rebound.spec.inputs[0].metadata["manifest_sha256"]
        != hashlib.sha256(substituted).hexdigest()
    )


#: Every module reachable while a fulfillment closure executes. This is the
#: security boundary, and it is deliberately not a filename glob: the bundle
#: executors and the analysis-run executor run *inside* the closure, so a
#: check/use race in them is a check/use race in fulfillment. Round 3 drew the
#: boundary at ``fulfillment*.py`` and missed exactly that.
EXECUTION_SURFACE = (
    "analysis/fulfillment.py",
    "analysis/fulfillment_adapters.py",
    "analysis/fulfillment_checkpoint_init.py",
    "analysis/fulfillment_custody.py",
    "workflow/derivation.py",
    "workflow/execution.py",
    "workflow/experiment.py",
    "workflow/operation_execution.py",
    "workflow/plan.py",
    "analysis/fulfillment_row_custody.py",
    "analysis/bundles.py",
    "analysis/specs.py",
)

#: The helper that opens a path it is handed and mints a profile from whatever
#: it finds. Correct as a *first* authentication; a silent override anywhere a
#: profile already exists, which on this surface is everywhere.
FORBIDDEN_HELPER = "authenticated_manifest_ref"


def _package_root() -> Path:
    return Path(__file__).resolve().parents[1] / "feedbax"


def test_no_module_on_the_execution_path_can_reach_a_fresh_digest_minting_helper() -> None:
    """``authenticated_manifest_ref`` is out of reach for the whole closure.

    The helper re-reads the path it is given and mints a digest from the second
    read, which is the check/use defect in one function. Callers outside the
    closure — Studio, the web API — legitimately use it to authenticate bytes
    they have just written and hold no prior proof of. Inside the closure, a
    proof always exists by the time anything wants a ref, so the surface must
    not be able to reach it at all.

    Detection covers the static reaches: importing the name, referring to it,
    and reaching it as an attribute on a module object. It also covers the two
    dynamic spellings that would defeat the first three —
    ``importlib.import_module`` and ``getattr`` — by flagging any ``getattr``
    whose attribute name is a literal match and any dynamic import of the
    module that defines it.
    """
    import ast

    offenders: list[str] = []
    for relative in EXECUTION_SURFACE:
        module = _package_root() / relative
        assert module.is_file(), f"{relative} must exist for the guard to mean anything"
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if FORBIDDEN_HELPER in {alias.name for alias in node.names}:
                    offenders.append(f"{relative}: imports {FORBIDDEN_HELPER}")
            elif isinstance(node, ast.Name) and node.id == FORBIDDEN_HELPER:
                offenders.append(f"{relative}: names {FORBIDDEN_HELPER}")
            elif isinstance(node, ast.Attribute) and node.attr == FORBIDDEN_HELPER:
                offenders.append(f"{relative}: reaches {FORBIDDEN_HELPER} as an attribute")
            elif isinstance(node, ast.Call):
                function = node.func
                name = (
                    function.attr
                    if isinstance(function, ast.Attribute)
                    else function.id
                    if isinstance(function, ast.Name)
                    else ""
                )
                if name == "getattr" and len(node.args) >= 2:
                    target = node.args[1]
                    if isinstance(target, ast.Constant) and target.value == FORBIDDEN_HELPER:
                        offenders.append(f"{relative}: getattr reaches {FORBIDDEN_HELPER}")
                if name == "import_module":
                    literal = node.args[0] if node.args else None
                    if isinstance(literal, ast.Constant) and "manifest_inputs" in str(
                        literal.value
                    ):
                        offenders.append(
                            f"{relative}: dynamically imports the module defining "
                            f"{FORBIDDEN_HELPER}"
                        )
    assert offenders == []


def test_the_execution_surface_guard_actually_detects_each_reach(tmp_path: Path) -> None:
    """The guard is only worth its assertion if it fails on the reaches it names.

    A structural guard that cannot be shown to fire is a comment. Each spelling
    below is parsed the same way the guard parses the real surface.
    """
    import ast

    spellings = {
        "import": "from feedbax.analysis.manifest_inputs import authenticated_manifest_ref\n",
        "name": "ref = authenticated_manifest_ref(m, p, 'r')\n",
        "attribute": "ref = manifest_inputs.authenticated_manifest_ref(m, p, 'r')\n",
        "getattr": "fn = getattr(manifest_inputs, 'authenticated_manifest_ref')\n",
        "import_module": ("mod = importlib.import_module('feedbax.analysis.manifest_inputs')\n"),
    }

    def detects(source: str) -> bool:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if FORBIDDEN_HELPER in {alias.name for alias in node.names}:
                    return True
            elif isinstance(node, ast.Name) and node.id == FORBIDDEN_HELPER:
                return True
            elif isinstance(node, ast.Attribute) and node.attr == FORBIDDEN_HELPER:
                return True
            elif isinstance(node, ast.Call):
                function = node.func
                name = (
                    function.attr
                    if isinstance(function, ast.Attribute)
                    else function.id
                    if isinstance(function, ast.Name)
                    else ""
                )
                if name == "getattr" and len(node.args) >= 2:
                    target = node.args[1]
                    if isinstance(target, ast.Constant) and target.value == FORBIDDEN_HELPER:
                        return True
                if name == "import_module":
                    literal = node.args[0] if node.args else None
                    if isinstance(literal, ast.Constant) and "manifest_inputs" in str(
                        literal.value
                    ):
                        return True
        return False

    for label, source in spellings.items():
        assert detects(source), f"the guard must detect the {label} reach"
    assert not detects("ref = authenticated_manifest_ref_from_read(p, role='r')\n")


def test_a_duplicate_role_edge_refuses_at_reconciliation(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment, calls: _Calls
) -> None:
    """Defense in depth: reconciliation never dict-keys away a duplicate.

    The plan kernel refuses a duplicate, so this plan is assembled around it —
    which is exactly what a hand-written or maliciously constructed plan would
    do. Reconciliation must name the duplicate rather than compare whichever
    copy it keyed last, because the copy it drops is the one that never faces a
    lock.
    """
    from feedbax.workflow.plan import WorkflowPlan, PlanEdge

    outputs.probe("recon-source")
    produced = _fulfill(outputs, "recon-source", environment=environment).results[0].receipt
    outputs.bulletin(
        "recon-consumer",
        references=[
            ReceiptLocatorReference(
                manifest_kind="EvaluationRunManifest",
                manifest_id=produced.manifest_id,
                role_path="body.quoted",
                consumer=ReportParentBinding(parent_kind="probe", parent_id="quoted"),
            )
        ],
    )
    index = read_compiled_outputs(outputs.output_directory)
    plan = derive_workflow_plan(index, target="recon-consumer")
    genuine = next(edge for edge in plan.edges if edge.external is not None)
    injected = PlanEdge(
        consumer=genuine.consumer,
        role_path=genuine.role_path,
        status=genuine.status,
        basis=genuine.basis,
        input_type=genuine.input_type,
        reason=genuine.reason,
        external={**dict(genuine.external), "manifest_id": "feedbax-evaluation-run:injected"},
        external_type=genuine.external_type,
    )
    smuggled = WorkflowPlan(
        target=plan.target,
        nodes=plan.nodes,
        edges=(injected, *plan.edges),
        origin=plan.origin,
    )

    before = calls.report
    with pytest.raises(PlanLockDisagreementError) as caught:
        prepare_workflow(smuggled, read_compiled_outputs(outputs.output_directory))

    differences = caught.value.record()["differences"]
    assert any("stated 2 times by the plan" in difference for difference in differences)
    assert calls.report == before, "a refused closure never runs"


def test_a_plan_node_with_no_content_hash_refuses_rather_than_skipping_the_pin(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment, calls: _Calls
) -> None:
    """A pin that is absent is not a pin that matched."""
    target = _chain(outputs)
    index = read_compiled_outputs(outputs.output_directory)
    plan = derive_workflow_plan(index, target=target)
    unpinned = WorkflowPlan(
        target=plan.target,
        nodes=tuple(
            replace(node, content_hash=None)
            if node.key == LogicalKey("evaluation", "chain-mid")
            else node
            for node in plan.nodes
        ),
        edges=plan.edges,
        origin=plan.origin,
    )

    before = calls.evaluation
    with pytest.raises(UnpinnedPlanNodeError) as caught:
        prepare_workflow(
            unpinned,
            read_compiled_outputs(outputs.output_directory),
        )
    assert caught.value.record()["key"] == "evaluation:chain-mid"
    assert "never a skipped check" in str(caught.value)
    assert calls.evaluation == before


@pytest.mark.parametrize(
    ("field", "substituted"),
    [
        ("compiled_schema_id", "feedbax.spec.report"),
        ("execution_identity", "e" * 64),
        ("key", "analysis:chain-mid"),
    ],
)
def test_a_node_fact_the_lock_does_not_determine_refuses_at_preflight(
    outputs: QuillonOutputs,
    environment: FulfillmentEnvironment,
    calls: _Calls,
    field: str,
    substituted: str,
) -> None:
    """The document pin proves the document, and nothing else the node claims.

    A plan is a derived record of the compile locks, so every fact it carries
    about a node is a copy. Copying is fine; carrying an intact ``content_hash``
    beside a substituted schema id, execution identity, or logical key is a plan
    describing a node the compile never emitted, and prepare_workflow is where the
    copies are checked against the authority they were copied from.
    """
    target = _chain(outputs)
    index = read_compiled_outputs(outputs.output_directory)
    plan = derive_workflow_plan(index, target=target)
    original_key = LogicalKey("evaluation", "chain-mid")
    replacement_key = LogicalKey.parse(substituted) if field == "key" else original_key
    nodes = []
    for node in plan.nodes:
        if node.key != original_key:
            nodes.append(node)
        elif field == "compiled_schema_id":
            nodes.append(
                replace(
                    node,
                    operation=replace(
                        node.operation,
                        parameters={
                            **node.operation.parameters,
                            "compiled_schema_id": substituted,
                        },
                    ),
                )
            )
        elif field == "execution_identity":
            nodes.append(replace(node, execution_identity=substituted))
        else:
            nodes.append(replace(node, key=replacement_key))
    edges = tuple(
        replace(
            edge,
            consumer=replacement_key if edge.consumer == original_key else edge.consumer,
            producer=replacement_key if edge.producer == original_key else edge.producer,
        )
        for edge in plan.edges
    )
    substituted_plan = WorkflowPlan(
        target=plan.target,
        nodes=tuple(nodes),
        edges=edges,
        origin=plan.origin,
    )

    before = calls.evaluation
    with pytest.raises(PlanNodeDisagreementError) as caught:
        prepare_workflow(
            substituted_plan,
            read_compiled_outputs(outputs.output_directory),
        )

    differences = caught.value.record()["differences"]
    assert any(difference.startswith(f"{field}:") for difference in differences)
    assert substituted in str(caught.value)
    assert calls.evaluation == before, "a refused closure never runs"


def test_an_honest_derived_plan_still_reconciles_every_node_fact(
    outputs: QuillonOutputs,
) -> None:
    """The node gate refuses substitutions, not the plans derivation produces."""
    closure = _closure(outputs, _chain(outputs))
    assert closure.order == CHAIN_ORDER
    for node in closure.nodes:
        assert node.plan_node.operation.parameters["compiled_schema_id"] == node.compiled.schema_id
        assert node.plan_node.execution_identity == node.compiled.execution_identity
        assert node.plan_node.key == node.compiled.key


def test_a_half_stated_restated_profile_is_refused_rather_than_dropped(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment
) -> None:
    """A partial authentication claim is unreadable, not absent.

    A document that restates no byte profile is entitled to: it cannot
    authenticate a parent, so it asserts nothing about bytes and there is
    nothing to disagree with. A document that restates *half* a profile is a
    different thing — something stated an authentication and got it wrong — and
    reading it as "states nothing" drops the malformed claim out of the very
    comparison it should have been subject to.
    """
    subject = outputs.probe("half-profile-subject")
    bank = outputs.probe("half-profile-bank")
    execute_workflow(_closure(outputs, "half-profile-subject"), environment=environment)
    produced = _fulfill(outputs, "half-profile-bank", environment=environment).results[0].receipt
    raw = produced.path.read_bytes()
    outputs.emit(
        "half-profile-consumer",
        {
            "schema_id": "feedbax.spec.evaluation_run_matrix",
            "schema_version": "feedbax.spec.evaluation_run_matrix.v3",
            "base": _base_spec("half-profile-consumer"),
            "rows": [{"row_id": "half-profile-consumer-0"}],
            "staged_parents": {
                "trial_bank": {
                    "parent": {
                        "kind": "EvaluationRunManifest",
                        "id": produced.manifest_id,
                        "role": "evaluation_run",
                        "metadata": {
                            "ref_schema_id": "feedbax.ref.authenticated_manifest",
                            "ref_schema_version": "feedbax.ref.authenticated_manifest.v1",
                            # Half a profile: the digest without the size.
                            "manifest_sha256": hashlib.sha256(raw).hexdigest(),
                        },
                    }
                }
            },
        },
        references=[
            _subject(subject, role_path="body.subject", subject_id="subject"),
            _subject(bank, role_path="body.trial_bank", subject_id="trial_bank"),
        ],
    )

    with pytest.raises(NodeLoweringError) as caught:
        _matrix_request(outputs, "half-profile-consumer", environment=environment, upstream=2)
    assert "cannot read" in str(caught.value)


def test_restated_parent_differences_reports_an_unreadable_profile_on_either_side() -> None:
    """Neither side may drop out of the comparison by being malformed."""
    from feedbax.workflow.operation_execution import restated_parent_differences
    from feedbax.contracts.base import ParentRef

    complete = {
        "ref_schema_id": "feedbax.ref.authenticated_manifest",
        "ref_schema_version": "feedbax.ref.authenticated_manifest.v1",
        "manifest_sha256": DIGEST,
        "size_bytes": 12,
    }
    half = {key: value for key, value in complete.items() if key != "size_bytes"}
    good = ParentRef(kind="EvaluationRunManifest", id="x", role="a", metadata=complete)
    partial = ParentRef(kind="EvaluationRunManifest", id="x", role="b", metadata=half)

    assert restated_parent_differences(good, good) == ()
    stated_defect = restated_parent_differences(partial, good)
    assert any("the document states an authentication profile" in item for item in stated_defect)
    bound_defect = restated_parent_differences(good, partial)
    assert any("the bound parent states an authentication profile" in item for item in bound_defect)
    # A document that states nothing about bytes still says nothing to refuse.
    silent = ParentRef(kind="EvaluationRunManifest", id="x", role="b", metadata={})
    assert restated_parent_differences(silent, good) == ()


# --------------------------------------------------------------------------
# The staged surface: several declared authorities, exactly one of them holding
# --------------------------------------------------------------------------


def _retained_context(*roots: Path):
    """A staged context declaring nothing but the given retained manifest stores."""
    from feedbax.analysis.execution_context import (
        StagedManifestRootBinding,
        resolve_staged_execution_context,
    )
    from feedbax.contracts.staged_execution import (
        STAGED_EXECUTION_DESCRIPTOR_SCHEMA_ID,
        STAGED_EXECUTION_DESCRIPTOR_SCHEMA_VERSION,
        StagedExecutionDescriptor,
    )

    for root in roots:
        root.mkdir(parents=True, exist_ok=True)
    return resolve_staged_execution_context(
        StagedExecutionDescriptor(
            schema_id=STAGED_EXECUTION_DESCRIPTOR_SCHEMA_ID,
            schema_version=STAGED_EXECUTION_DESCRIPTOR_SCHEMA_VERSION,
            artifact_providers={},
            checkpoint_custody={},
        ),
        manifest_root_bindings=[
            StagedManifestRootBinding(f"retained-{index}", root) for index, root in enumerate(roots)
        ],
    )


def _retain(receipt, root: Path) -> Path:
    """Copy one produced receipt to its canonical location under another root."""
    path = canonical_manifest_path(receipt.manifest_kind, receipt.manifest_id, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(receipt.path.read_bytes())
    return path


def _quoted_consumer(outputs: QuillonOutputs, name: str, receipt) -> str:
    raw = receipt.path.read_bytes()
    outputs.bulletin(
        name,
        references=[
            AuthenticatedReceiptReference(
                manifest_kind=receipt.manifest_kind,
                manifest_id=receipt.manifest_id,
                manifest_sha256=hashlib.sha256(raw).hexdigest(),
                size_bytes=len(raw),
                role_path="body.quoted",
                consumer=ReportParentBinding(parent_kind="probe", parent_id="quoted"),
            )
        ],
    )
    return name


def test_a_parent_only_a_retained_store_holds_resolves_from_it(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment, tmp_path: Path
) -> None:
    """The receipt root is one authority among the declared ones, not the only one.

    The reference's bytes were produced elsewhere and retained in a manifest
    store the caller bound by name. Nothing was copied under the receipt root,
    and nothing needed to be: the store is where the reference resolves, and the
    node executes with a context that says so.
    """
    from dataclasses import replace

    outputs.probe("retained-source")
    produced = _fulfill(outputs, "retained-source", environment=environment).results[0].receipt
    retained = tmp_path / "retained"
    retained_path = _retain(produced, retained)
    produced.path.unlink()
    target = _quoted_consumer(outputs, "retained-consumer", _Receipt(produced, retained_path))

    staged = replace(environment, execution_context=_retained_context(retained))
    run = _fulfill(outputs, target, environment=staged)

    bound = load_manifest(run.results[0].receipt.path).provenance.parents
    assert [ref.id for ref in bound] == [produced.manifest_id]
    request = workflow_requests(_closure(outputs, target), environment=staged)[0]
    location = request.execution_context.parent_execution_location(bound[0])
    assert location.root == retained
    assert location.execution_uri == retained_path.relative_to(retained).as_posix()


class _Receipt:
    """A receipt-shaped view whose bytes live somewhere other than the receipt root."""

    def __init__(self, receipt, path: Path) -> None:
        self.manifest_kind = receipt.manifest_kind
        self.manifest_id = receipt.manifest_id
        self.path = path


def test_a_parent_two_declared_authorities_hold_refuses_before_any_effect(
    outputs: QuillonOutputs,
    environment: FulfillmentEnvironment,
    tmp_path: Path,
    calls: _Calls,
) -> None:
    """No precedence: a reference held twice is a custody question, not a tie.

    The receipt root and a retained store both hold a completed manifest of the
    named kind and id. Preferring either would silently pick a custody domain
    nobody chose, and the two can differ in exactly the bytes an authenticated
    reference exists to pin. So the closure refuses, and the consumer never runs.
    """
    from dataclasses import replace

    from feedbax.workflow.execution import AmbiguousExternalReceiptError

    outputs.probe("doubled-source")
    produced = _fulfill(outputs, "doubled-source", environment=environment).results[0].receipt
    retained = tmp_path / "retained"
    _retain(produced, retained)
    target = _quoted_consumer(outputs, "doubled-consumer", produced)

    staged = replace(environment, execution_context=_retained_context(retained))
    before = calls.report
    with pytest.raises(AmbiguousExternalReceiptError) as caught:
        _fulfill(outputs, target, environment=staged)

    detail = caught.value.record_detail()
    assert detail["manifest_id"] == produced.manifest_id
    assert detail["consumer"] == "report:doubled-consumer"
    assert detail["authorities"] == ["receipt root", "manifest root 'retained-0'"]
    assert "no precedence" in str(caught.value)
    assert calls.report == before, "an ambiguous reference never reaches execution"
    assert not _reports_directory(environment).exists()


def test_a_missing_parent_names_every_authority_that_was_searched(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment, tmp_path: Path
) -> None:
    """A refusal describes the search that happened, not one root of it."""
    from dataclasses import replace

    outputs.bulletin(
        "searched-consumer",
        references=[
            ReceiptLocatorReference(
                manifest_kind="EvaluationRunManifest",
                manifest_id="feedbax-evaluation-run:nowhere",
                role_path="body.prior",
                consumer=ReportParentBinding(parent_kind="probe", parent_id="prior"),
            )
        ],
    )
    staged = replace(environment, execution_context=_retained_context(tmp_path / "retained"))
    with pytest.raises(MissingExternalReceiptError) as caught:
        _fulfill(outputs, "searched-consumer", environment=staged)
    assert caught.value.searched == ("receipt root", "manifest root 'retained-0'")
    assert "manifest root 'retained-0'" in str(caught.value)


# --------------------------------------------------------------------------
# Per-node contexts are the request's, so every operation sees the same one
# --------------------------------------------------------------------------


def test_every_operation_over_a_closure_reconstructs_the_same_node_contexts(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment, tmp_path: Path
) -> None:
    """Fulfillment, admission, rebuild, and repair all lower the node again.

    A context assembled during the walk and thrown away after it would leave
    rebuild-as-verification and repair resolving parents some other way. The
    context lives on the request, so each of those operations gets the one the
    lowering settled.
    """
    from dataclasses import replace

    outputs.probe("persisted-source")
    produced = _fulfill(outputs, "persisted-source", environment=environment).results[0].receipt
    retained = tmp_path / "retained"
    retained_path = _retain(produced, retained)
    produced.path.unlink()
    target = _quoted_consumer(outputs, "persisted-consumer", _Receipt(produced, retained_path))

    staged = replace(environment, execution_context=_retained_context(retained))
    closure = _closure(outputs, target)

    fulfilled = execute_workflow(closure, environment=staged)
    assert fulfilled.executed == ("report:persisted-consumer",)

    # Reuse: the second walk admits rather than executes, and the request it
    # admits against carries the same context.
    again = execute_workflow(closure, environment=staged)
    assert again.reused == ("report:persisted-consumer",)

    requests = workflow_requests(closure, environment=staged)
    assert len(requests) == 1
    context = requests[0].execution_context
    assert context is not None
    assert [location.root for location in context.parent_execution_locations] == [retained]

    rebuild = rebuild_workflow(closure, environment=staged)
    assert rebuild.drifted == ()

    # Repair executes the node again into shadow custody, which means resolving
    # its parents again. A repair that lost the node's context would look for
    # the parent beneath the receipt root, where it is not.
    _mutate(fulfilled.results[0].receipt.path, status="failed")
    repaired = repair_workflow_operation(
        closure, LogicalKey("report", "persisted-consumer"), environment=staged
    )
    assert repaired.record.node_key == "report:persisted-consumer"
    assert repaired.record.admission_after_repair.admitted


def test_a_run_declaring_no_staged_bindings_lowers_no_context(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment
) -> None:
    """Cold start is left exactly as it was, not routed through an empty context."""
    target = _chain(outputs)
    closure = _closure(outputs, target)
    execute_workflow(closure, environment=environment)
    assert [
        request.execution_context for request in workflow_requests(closure, environment=environment)
    ] == [None, None, None]


def test_declaring_a_staged_surface_changes_no_produced_receipt(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment, tmp_path: Path
) -> None:
    """A declared authority nothing resolves through leaves identity untouched.

    Manifest identity is derived from the spec, so an equal id across the two
    runs is the statement that declaring a staged surface did not change what
    was built — only where a parent may be looked for. The subject is a node
    with no parents on purpose: a node that binds one binds the digest of a
    *previous run's* bytes, which differ between two independent runs for
    reasons that have nothing to do with staged inputs.
    """
    from dataclasses import replace

    outputs.probe("surface-regression")
    cold = _fulfill(outputs, "surface-regression", environment=environment)
    staged = replace(
        environment,
        root=tmp_path / "staged-receipts",
        execution_context=_retained_context(tmp_path / "unused-retained"),
    )
    declared = execute_workflow(_closure(outputs, "surface-regression"), environment=staged)

    assert declared.executed == cold.executed
    assert declared.results[0].receipt.manifest_id == cold.results[0].receipt.manifest_id
    assert (
        load_manifest(declared.results[0].receipt.path).summary_metrics
        == load_manifest(cold.results[0].receipt.path).summary_metrics
    )


def _provider_context(root: Path, name: str = "results"):
    """A staged context declaring exactly one immutable artifact provider."""
    from feedbax.analysis.execution_context import (
        StagedArtifactProviderRootBinding,
        resolve_staged_execution_context,
    )
    from feedbax.contracts.artifact_custody import ImmutableArtifactBlobProviderSpec
    from feedbax.contracts.staged_execution import (
        STAGED_EXECUTION_DESCRIPTOR_SCHEMA_ID,
        STAGED_EXECUTION_DESCRIPTOR_SCHEMA_VERSION,
        StagedExecutionDescriptor,
    )

    root.mkdir(parents=True, exist_ok=True)
    return resolve_staged_execution_context(
        StagedExecutionDescriptor(
            schema_id=STAGED_EXECUTION_DESCRIPTOR_SCHEMA_ID,
            schema_version=STAGED_EXECUTION_DESCRIPTOR_SCHEMA_VERSION,
            artifact_providers={name: ImmutableArtifactBlobProviderSpec()},
            checkpoint_custody={},
        ),
        artifact_provider_bindings=[StagedArtifactProviderRootBinding(name, root)],
    )


def test_a_parent_only_an_artifact_provider_holds_resolves_and_binds_its_alias(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment, tmp_path: Path
) -> None:
    """A provider is addressed by the digest the reference quotes, and by nothing else.

    A locator states no digest and therefore reaches no provider — there is no
    other way to address content-addressed bytes. This reference is an
    authenticated one, so the provider is a candidate, it is the only authority
    holding the manifest, and the node executes with the provider bound both as
    the parent's location and as the authored alias of the same name.
    """
    from dataclasses import replace

    from feedbax.contracts.artifact_custody import ImmutableArtifactBlobProviderSpec
    from feedbax.persistence.artifact_custody import open_immutable_artifact_blob_provider

    outputs.probe("provider-source")
    produced = _fulfill(outputs, "provider-source", environment=environment).results[0].receipt
    raw = produced.path.read_bytes()

    provider_root = tmp_path / "provider"
    provider_root.mkdir()
    provider = open_immutable_artifact_blob_provider(
        ImmutableArtifactBlobProviderSpec(), explicit_root=provider_root
    )
    provider.store_bytes(raw, role="manifest", logical_name="receipt.json")
    produced.path.unlink()

    outputs.bulletin(
        "provider-consumer",
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

    staged = replace(environment, execution_context=_provider_context(provider_root))
    run = _fulfill(outputs, "provider-consumer", environment=staged)

    bound = load_manifest(run.results[0].receipt.path).provenance.parents
    assert [ref.id for ref in bound] == [produced.manifest_id]
    context = workflow_requests(_closure(outputs, "provider-consumer"), environment=staged)[
        0
    ].execution_context
    location = context.parent_execution_location(bound[0])
    assert location.artifact_provider == "results"
    assert location.root == provider_root
    assert [binding.authored_provider for binding in context.parent_artifact_provider_bindings] == [
        "results"
    ]


def test_a_locator_reference_reaches_no_artifact_provider(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment, tmp_path: Path
) -> None:
    """Without a quoted digest there is no content address, so no provider is searched."""
    from dataclasses import replace

    from feedbax.contracts.artifact_custody import ImmutableArtifactBlobProviderSpec
    from feedbax.persistence.artifact_custody import open_immutable_artifact_blob_provider

    outputs.probe("locator-provider-source")
    produced = (
        _fulfill(outputs, "locator-provider-source", environment=environment).results[0].receipt
    )
    raw = produced.path.read_bytes()
    provider_root = tmp_path / "provider"
    provider_root.mkdir()
    open_immutable_artifact_blob_provider(
        ImmutableArtifactBlobProviderSpec(), explicit_root=provider_root
    ).store_bytes(raw, role="manifest", logical_name="receipt.json")
    produced.path.unlink()

    outputs.bulletin(
        "locator-provider-consumer",
        references=[
            ReceiptLocatorReference(
                manifest_kind="EvaluationRunManifest",
                manifest_id=produced.manifest_id,
                role_path="body.prior",
                consumer=ReportParentBinding(parent_kind="probe", parent_id="prior"),
            )
        ],
    )
    staged = replace(environment, execution_context=_provider_context(provider_root))
    with pytest.raises(MissingExternalReceiptError) as caught:
        _fulfill(outputs, "locator-provider-consumer", environment=staged)
    assert caught.value.searched == ("receipt root",)
