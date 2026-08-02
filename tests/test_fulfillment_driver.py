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
    ExternalReceiptAuthenticationError,
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
    canonical_json_bytes,
    canonical_manifest_path,
    load_manifest,
    sha256_bytes,
    store_bytes_artifact,
)

from tests.fake_project_experiment.products import (
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


def _matrix_request(outputs: QuillonOutputs, target: str, *, environment, upstream: int = 1):
    """Fulfil everything upstream of one matrix node and return its request."""
    closure = _closure(outputs, target)
    fulfill_closure(truncated_closure(closure, upstream), environment=environment)
    requests = closure_requests(
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
        assert (
            row.payload.params["staged_prerequisites"]
            == request.matrix["staged_parents"]
        )


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
    staged = _matrix_request(
        outputs, "consumed-consumer", environment=environment
    ).matrix["staged_parents"]

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
    produced = (
        _fulfill(outputs, "substituted-source", environment=environment).results[0].receipt
    )
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
    assert bound[0].metadata["manifest_sha256"] == hashlib.sha256(
        produced.path.read_bytes()
    ).hexdigest()


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

    assert [edge.role_path for edge in closure.plan.required_edges(key)] == [
        ("runtime", "states")
    ]
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
    requests = closure_requests(closure, environment=environment, stop_at=key)
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
