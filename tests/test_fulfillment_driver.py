"""The generic fulfillment driver: preflight, pull-forward execution, custody.

Everything here is stated against an invented corpus — its layers, node kinds,
external-record vocabulary, and rule names are this module's own — and a receipt
root under ``tmp_path``. The lowering from a node kind to a feedbax node request
is supplied per test, exactly as a project supplies it, so the driver is
exercised without any project vocabulary reaching it. Four claims are under test:

* **preflight refuses before anything runs.** A closure naming a boundary node
  is refused with every boundary node named, and no recipe of any branch runs;
* **fulfillment is a pull-forward.** Each node is reused when its receipt admits
  and executed exactly once when it does not, in the plan's dependency order, so
  a second walk over a fulfilled closure executes nothing and an interrupted one
  resumes at the node boundary it stopped at;
* **a receipt that exists but fails admission is a refusal**, never a cache
  miss, and the named admission failure reaches the caller;
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
from feedbax.analysis.exact_parents import (
    STAGED_EXACT_PARENTS_SCHEMA_ID,
    STAGED_EXACT_PARENTS_SCHEMA_VERSION,
    StagedExactParents,
)
from feedbax.analysis.fulfillment import FulfillmentAdmissionError
from feedbax.analysis.fulfillment_adapters import (
    EvaluationMatrixNodeRequest,
    EvaluationNodeRequest,
    FulfillmentEnvironment,
    ReportNodeRequest,
)
from feedbax.analysis.fulfillment_custody import FulfillmentDriftError
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
from feedbax.analysis.fulfillment_plan import (
    CertifiedOmissionsPendingError,
    EdgeDeclaration,
    LogicalKey,
    NodeDeclaration,
    PlanNode,
    expand_fulfillment_plan,
)
from feedbax.analysis.reports import REPORT_RENDER_ROLE, ReportRecipeResult
from feedbax.contracts.manifest import (
    EvaluationRunSpec,
    ReportSpec,
    canonical_json_bytes,
    canonical_manifest_path,
    load_manifest,
    sha256_bytes,
    store_bytes_artifact,
)


#: The invented corpus's vocabulary. None of it is known to the driver.
SAMPLE_LAYER = "sample"
DIGEST_LAYER = "digest"
BULLETIN_LAYER = "bulletin"
HARVEST_LAYER = "harvest"

SAMPLE_KIND = "toybox.sample"
DIGEST_KIND = "toybox.digest"
BULLETIN_KIND = "toybox.bulletin"
HARVEST_BOUNDARY = "harvest"

TOY_EVALUATION_TYPE = "testpkg.toybox_probe"
TOY_REPORT_TYPE = "testpkg.toybox_bulletin"

UNBOUND_SLOT_RULE = "unbound_inherited_slot"


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
            metadata={"states_schema": "testpkg.states.v1"},
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

    application_registry_bundle.evaluation_recipes.register(
        TOY_EVALUATION_TYPE, evaluation_recipe
    )
    application_registry_bundle.report_recipes.register(TOY_REPORT_TYPE, report_recipe)
    return FulfillmentEnvironment(
        root=tmp_path / "receipts",
        registries=application_registry_bundle,
        issues=("4390d11",),
    )


# --------------------------------------------------------------------------
# The invented corpus, its expander, and its lowering
# --------------------------------------------------------------------------


class _Corpus:
    """A tiny declaration store standing in for a project's compiler."""

    def __init__(self) -> None:
        self.documents: dict[str, dict[str, Any]] = {}
        self.declarations: dict[str, NodeDeclaration] = {}

    def declare(
        self,
        name: str,
        *,
        layer: str,
        kind: str,
        params: dict[str, Any] | None = None,
        edges: tuple[EdgeDeclaration, ...] = (),
        boundary: str | None = None,
    ) -> str:
        ref = f"{name}.decl"
        document = {"name": name, "kind": kind, "params": params or {"stage": name}}
        self.documents[ref] = document
        self.declarations[ref] = NodeDeclaration(
            node=PlanNode(
                key=LogicalKey(layer, name),
                source_ref=ref,
                kind=kind,
                content_hash=content_hash(document),
                execution_identity=f"identity-of-{name}",
                boundary=boundary,
            ),
            edges=edges,
        )
        return ref

    def expand(self, source_ref: str) -> NodeDeclaration:
        return self.declarations[source_ref]

    def prepare(self, node: PlanNode) -> dict[str, Any]:
        return json.loads(json.dumps(self.documents[node.source_ref]))


def content_hash(payload: Any) -> str:
    return sha256_bytes(canonical_json_bytes(payload))


@pytest.fixture
def corpus() -> _Corpus:
    return _Corpus()


def _external_binding(edge) -> dict[str, str]:
    """Read this corpus's external-record vocabulary into a (kind, id) pair."""
    if edge.producer is not None:
        return {}
    external = edge.external or {}
    return {"kind": external["manifest_kind"], "manifest_id": external["manifest_id"]}


def lower(node, *, binding):
    """The project-supplied lowering: one node kind to one feedbax node request."""
    edges = binding.required_edges(node.key)
    if node.kind in (SAMPLE_KIND, DIGEST_KIND):
        inputs = [
            binding.parent_ref(edge, role=edge.role_path[-1], **_external_binding(edge))
            for edge in edges
        ]
        return EvaluationNodeRequest(
            node_key=node.key.text,
            spec=EvaluationRunSpec(
                evaluation_type=TOY_EVALUATION_TYPE,
                params=node.payload["params"],
                inputs=inputs,
            ),
            order=node.order,
        )
    if node.kind == BULLETIN_KIND:
        entries = [
            binding.exact_parent_entry(edge, role=edge.role_path[-1], **_external_binding(edge))
            for edge in edges
        ]
        exact = StagedExactParents(
            schema_id=STAGED_EXACT_PARENTS_SCHEMA_ID,
            schema_version=STAGED_EXACT_PARENTS_SCHEMA_VERSION,
            parents=entries,
        )
        return ReportNodeRequest(
            node_key=node.key.text,
            spec=ReportSpec(
                report_type=TOY_REPORT_TYPE,
                inputs=[entry.parent for entry in entries],
                params=node.payload["params"],
            ),
            exact_parents=exact,
            order=node.order,
        )
    raise AssertionError(f"the corpus declares no lowering for {node.kind!r}")


def _sample(corpus: _Corpus, name: str, **params: Any) -> str:
    return corpus.declare(
        name, layer=SAMPLE_LAYER, kind=SAMPLE_KIND, params={"stage": name, **params}
    )


def _digest(corpus: _Corpus, name: str, **roles: str) -> str:
    return corpus.declare(
        name,
        layer=DIGEST_LAYER,
        kind=DIGEST_KIND,
        edges=tuple(
            EdgeDeclaration(role_path=(role,), producer_ref=ref) for role, ref in roles.items()
        ),
    )


def _chain(corpus: _Corpus) -> str:
    """A three-node closure: root -> mid -> leaf, and its target ref."""
    _sample(corpus, "chain-root")
    corpus.declare(
        "chain-mid",
        layer=DIGEST_LAYER,
        kind=DIGEST_KIND,
        edges=(EdgeDeclaration(role_path=("upstream",), producer_ref="chain-root.decl"),),
    )
    return corpus.declare(
        "chain-leaf",
        layer=DIGEST_LAYER,
        kind=DIGEST_KIND,
        edges=(EdgeDeclaration(role_path=("middle",), producer_ref="chain-mid.decl"),),
    )


CHAIN_EXECUTION_ORDER = ("sample:chain-root", "digest:chain-mid", "digest:chain-leaf")


def _closure(corpus: _Corpus, target_ref: str, **kwargs: Any):
    plan = expand_fulfillment_plan(target_ref, expand=corpus.expand)
    return preflight(plan, prepare=corpus.prepare, content_hash=content_hash, **kwargs)


def _fulfill(corpus: _Corpus, target_ref: str, *, environment, **kwargs: Any):
    return fulfill_closure(
        _closure(corpus, target_ref, **kwargs), environment=environment, lower=lower
    )


def _reports_directory(environment: FulfillmentEnvironment) -> Path:
    """Return where a report receipt would be written, so absence is checkable."""
    return canonical_manifest_path("ReportManifest", "probe", root=environment.root).parent


# --------------------------------------------------------------------------
# Preflight: the closure, its order, and the external boundary
# --------------------------------------------------------------------------


def test_preflight_states_the_whole_closure_in_dependency_order(corpus: _Corpus) -> None:
    closure = _closure(corpus, _chain(corpus))
    assert closure.order == CHAIN_EXECUTION_ORDER
    assert closure.target == LogicalKey(DIGEST_LAYER, "chain-leaf")
    for node in closure.nodes:
        assert content_hash(node.payload) == node.plan_node.content_hash
        assert node.kind in (SAMPLE_KIND, DIGEST_KIND)


def test_a_payload_that_no_longer_hashes_to_what_was_pinned_refuses(corpus: _Corpus) -> None:
    target = _chain(corpus)
    plan = expand_fulfillment_plan(target, expand=corpus.expand)
    corpus.documents["chain-mid.decl"]["params"]["stage"] = "moved"
    with pytest.raises(PlanDocumentDriftError) as caught:
        preflight(plan, prepare=corpus.prepare, content_hash=content_hash)
    assert caught.value.key == LogicalKey(DIGEST_LAYER, "chain-mid")


def test_an_absent_boundary_receipt_refuses_before_any_node_executes(
    corpus: _Corpus, environment: FulfillmentEnvironment, calls: _Calls
) -> None:
    corpus.declare(
        "boundary-run", layer=HARVEST_LAYER, kind=SAMPLE_KIND, boundary=HARVEST_BOUNDARY
    )
    target = _digest(corpus, "boundary-consumer", harvested="boundary-run.decl")

    with pytest.raises(ExternalBoundaryError) as caught:
        _fulfill(corpus, target, environment=environment)

    record = caught.value.record()
    assert [node["key"] for node in record["boundary_nodes"]] == ["harvest:boundary-run"]
    node = record["boundary_nodes"][0]
    assert node["source_ref"] == "boundary-run.decl"
    assert node["boundary"] == HARVEST_BOUNDARY
    assert node["named_by"] == [
        {"consumer": "digest:boundary-consumer", "role_path": ["harvested"]}
    ]
    assert node["unblocks"] == ["digest:boundary-consumer"]
    assert calls.evaluation == 0
    assert not environment.root.exists()


def test_the_refusal_lists_every_branch_the_boundary_unblocks(
    corpus: _Corpus, environment: FulfillmentEnvironment, calls: _Calls
) -> None:
    """Zero executions on *every* branch, including the ones that could run."""
    corpus.declare(
        "shared-run", layer=HARVEST_LAYER, kind=SAMPLE_KIND, boundary=HARVEST_BOUNDARY
    )
    _sample(corpus, "free-branch")
    _digest(corpus, "blocked-branch", harvested="shared-run.decl")
    target = _digest(
        corpus, "joined", free="free-branch.decl", blocked="blocked-branch.decl"
    )
    with pytest.raises(ExternalBoundaryError) as caught:
        _fulfill(corpus, target, environment=environment)
    assert caught.value.record()["boundary_nodes"][0]["unblocks"] == [
        "digest:blocked-branch",
        "digest:joined",
    ]
    assert calls.evaluation == 0


# --------------------------------------------------------------------------
# Pull-forward execution
# --------------------------------------------------------------------------


def test_one_walk_fulfils_the_closure_and_the_second_executes_nothing(
    corpus: _Corpus, environment: FulfillmentEnvironment, calls: _Calls
) -> None:
    target = _chain(corpus)

    first = _fulfill(corpus, target, environment=environment)
    assert first.execution_order == CHAIN_EXECUTION_ORDER
    assert first.executed == CHAIN_EXECUTION_ORDER
    assert calls.evaluation == 3
    for result in first.results:
        assert load_manifest(result.receipt.path).status == "completed"

    second = _fulfill(corpus, target, environment=environment)
    assert second.execution_order == CHAIN_EXECUTION_ORDER
    assert second.executed == ()
    assert second.reused == CHAIN_EXECUTION_ORDER
    assert calls.evaluation == 3


def test_two_walks_produce_identical_order_and_listings(
    corpus: _Corpus, environment: FulfillmentEnvironment
) -> None:
    target = _chain(corpus)
    first = _fulfill(corpus, target, environment=environment)
    second = _fulfill(corpus, target, environment=environment)
    assert first.execution_order == second.execution_order
    assert [result.receipt.manifest_id for result in first.results] == [
        result.receipt.manifest_id for result in second.results
    ]
    assert [result.node_kind for result in first.results] == [
        result.node_kind for result in second.results
    ]


def test_a_receipt_binds_forward_as_the_authenticated_parent_it_is(
    corpus: _Corpus, environment: FulfillmentEnvironment
) -> None:
    target = _chain(corpus)
    run = _fulfill(corpus, target, environment=environment)
    upstream, middle, leaf = run.results
    middle_parents = load_manifest(middle.receipt.path).provenance.parents
    assert [ref.id for ref in middle_parents] == [upstream.receipt.manifest_id]
    assert middle_parents[0].metadata["manifest_sha256"] == hashlib.sha256(
        upstream.receipt.path.read_bytes()
    ).hexdigest()
    assert [ref.role for ref in middle_parents] == ["upstream"]
    leaf_parents = load_manifest(leaf.receipt.path).provenance.parents
    assert [ref.id for ref in leaf_parents] == [middle.receipt.manifest_id]


@pytest.mark.parametrize("boundary", [1, 2])
def test_an_interrupted_walk_resumes_at_the_node_boundary_it_stopped_at(
    corpus: _Corpus, environment: FulfillmentEnvironment, calls: _Calls, boundary: int
) -> None:
    """A crash after node k leaves k admitted receipts; re-invocation finishes."""
    target = _chain(corpus)
    partial = truncated_closure(_closure(corpus, target), boundary)
    interrupted = fulfill_closure(partial, environment=environment, lower=lower)
    assert interrupted.executed == CHAIN_EXECUTION_ORDER[:boundary]
    assert calls.evaluation == boundary

    resumed = _fulfill(corpus, target, environment=environment)
    assert resumed.reused == CHAIN_EXECUTION_ORDER[:boundary]
    assert resumed.executed == CHAIN_EXECUTION_ORDER[boundary:]
    assert calls.evaluation == 3


def test_deleting_one_producer_receipt_rebinds_and_re_mints_its_consumers(
    corpus: _Corpus, environment: FulfillmentEnvironment, calls: _Calls
) -> None:
    target = _chain(corpus)
    first = _fulfill(corpus, target, environment=environment)
    root_receipt, mid_receipt, leaf_receipt = (result.receipt for result in first.results)
    old_leaf_id = leaf_receipt.manifest_id
    old_leaf_path = leaf_receipt.path
    old_leaf_parents = load_manifest(old_leaf_path).provenance.parents

    mid_receipt.path.unlink()
    calls.payload = "re-executed"  # the re-execution is real, not a byte replay

    second = _fulfill(corpus, target, environment=environment)
    assert second.reused == ("sample:chain-root",)
    assert set(second.executed) == {"digest:chain-mid", "digest:chain-leaf"}
    assert calls.evaluation == 5, "nothing upstream of the deleted receipt re-executed"
    assert load_manifest(root_receipt.path).id == root_receipt.manifest_id

    new_leaf = second.results[-1].receipt
    assert new_leaf.manifest_id != old_leaf_id, "the consumer's identity is recomputed"
    assert new_leaf.path != old_leaf_path
    assert old_leaf_path.is_file(), "the old receipt stays a valid record of its old parents"
    assert load_manifest(old_leaf_path).provenance.parents == old_leaf_parents
    assert load_manifest(new_leaf.path).provenance.parents != old_leaf_parents


# --------------------------------------------------------------------------
# Admission failure is refusal, never a cache miss
# --------------------------------------------------------------------------


def _mutate(path: Path, **changes: Any) -> None:
    document = json.loads(path.read_text(encoding="utf-8"))
    document.update(changes)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


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
    corpus: _Corpus,
    environment: FulfillmentEnvironment,
    calls: _Calls,
    changes: dict[str, Any],
    code: str,
) -> None:
    target = _chain(corpus)
    run = _fulfill(corpus, target, environment=environment)
    _mutate(run.results[1].receipt.path, **changes)

    with pytest.raises(FulfillmentAdmissionError) as caught:
        _fulfill(corpus, target, environment=environment)
    assert code in caught.value.outcome.codes
    assert calls.evaluation == 3, "a failed record is never silently re-executed over"


def test_altered_artifact_bytes_refuse_as_custody_loss(
    corpus: _Corpus, environment: FulfillmentEnvironment, calls: _Calls
) -> None:
    target = _chain(corpus)
    run = _fulfill(corpus, target, environment=environment)
    artifact = load_manifest(run.results[0].receipt.path).artifacts[0]
    (environment.root / artifact.metadata["relative_path"]).write_bytes(b"corrupt")

    with pytest.raises(FulfillmentAdmissionError) as caught:
        _fulfill(corpus, target, environment=environment)
    assert "artifact_sha256_mismatch" in caught.value.outcome.codes
    assert calls.evaluation == 3


# --------------------------------------------------------------------------
# External receipts: quoted, resolved canonically, never inferred
# --------------------------------------------------------------------------


def _external_consumer(corpus: _Corpus, name: str, manifest_id: str, kind: str) -> str:
    return corpus.declare(
        name,
        layer=DIGEST_LAYER,
        kind=DIGEST_KIND,
        edges=(
            EdgeDeclaration(
                role_path=("prior",),
                external={"manifest_kind": kind, "manifest_id": manifest_id},
            ),
        ),
    )


def test_an_external_receipt_binds_at_its_canonical_location(
    corpus: _Corpus, environment: FulfillmentEnvironment, calls: _Calls
) -> None:
    produced = _fulfill(
        corpus, _sample(corpus, "prior-source"), environment=environment
    ).results[0].receipt
    target = _external_consumer(
        corpus, "prior-consumer", produced.manifest_id, "EvaluationRunManifest"
    )
    run = _fulfill(corpus, target, environment=environment)
    assert run.execution_order == ("digest:prior-consumer",)
    bound = load_manifest(run.results[0].receipt.path).provenance.parents
    assert [ref.id for ref in bound] == [produced.manifest_id]
    assert bound[0].metadata["manifest_sha256"] == hashlib.sha256(
        produced.path.read_bytes()
    ).hexdigest()
    assert calls.evaluation == 2


def test_an_absent_external_receipt_is_a_refusal_and_never_an_inapplicability(
    corpus: _Corpus, environment: FulfillmentEnvironment, calls: _Calls
) -> None:
    target = _external_consumer(
        corpus,
        "absent-consumer",
        "feedbax-evaluation-run:never-produced",
        "EvaluationRunManifest",
    )
    with pytest.raises(MissingExternalReceiptError) as caught:
        _fulfill(corpus, target, environment=environment)
    assert caught.value.manifest_id == "feedbax-evaluation-run:never-produced"
    assert caught.value.consumer == LogicalKey(DIGEST_LAYER, "absent-consumer")
    assert "never means the input does not apply" in str(caught.value)
    assert calls.evaluation == 0


def test_an_incomplete_external_receipt_is_refused_rather_than_bound(
    corpus: _Corpus, environment: FulfillmentEnvironment
) -> None:
    produced = _fulfill(
        corpus, _sample(corpus, "incomplete-source"), environment=environment
    ).results[0].receipt
    _mutate(produced.path, status="failed")
    target = _external_consumer(
        corpus, "incomplete-consumer", produced.manifest_id, "EvaluationRunManifest"
    )
    with pytest.raises(MissingExternalReceiptError):
        _fulfill(corpus, target, environment=environment)


def test_an_external_edge_the_lowering_does_not_address_refuses(
    corpus: _Corpus, environment: FulfillmentEnvironment
) -> None:
    """Reading an external record is the project's job, and it is never guessed."""
    target = _external_consumer(
        corpus, "unaddressed", "feedbax-evaluation-run:whatever", "EvaluationRunManifest"
    )
    closure = _closure(corpus, target)

    def bare_lowering(node, *, binding):
        edges = binding.required_edges(node.key)
        return EvaluationNodeRequest(
            node_key=node.key.text,
            spec=EvaluationRunSpec(
                evaluation_type=TOY_EVALUATION_TYPE,
                params=node.payload["params"],
                inputs=[binding.parent_ref(edge, role="prior") for edge in edges],
            ),
            order=node.order,
        )

    with pytest.raises(AmbiguousNodeReceiptError, match="only the lowering project"):
        fulfill_closure(closure, environment=environment, lower=bare_lowering)


# --------------------------------------------------------------------------
# A node kind whose inputs are exact staged parents
# --------------------------------------------------------------------------


def _bulletin(corpus: _Corpus, name: str, **slots: str) -> str:
    return corpus.declare(
        name,
        layer=BULLETIN_LAYER,
        kind=BULLETIN_KIND,
        edges=tuple(
            EdgeDeclaration(role_path=("Nominal", role), producer_ref=ref)
            for role, ref in slots.items()
        ),
    )


def test_a_bulletin_binds_its_slots_from_the_receipts_they_name(
    corpus: _Corpus, environment: FulfillmentEnvironment, calls: _Calls
) -> None:
    _sample(corpus, "k1-source")
    target = _bulletin(corpus, "bound-bulletin", k1="k1-source.decl")

    run = _fulfill(corpus, target, environment=environment)
    assert run.execution_order == ("sample:k1-source", "bulletin:bound-bulletin")
    assert run.executed == ("sample:k1-source", "bulletin:bound-bulletin")
    manifest = load_manifest(run.results[-1].receipt.path)
    assert [parent.role for parent in manifest.provenance.parents] == ["k1"]
    assert manifest.provenance.parents[0].metadata["manifest_sha256"] == hashlib.sha256(
        run.results[0].receipt.path.read_bytes()
    ).hexdigest()
    assert calls.report == 1

    assert _fulfill(corpus, target, environment=environment).executed == ()
    assert calls.report == 1


def test_a_bulletin_whose_external_slot_is_absent_refuses_before_it_renders(
    corpus: _Corpus, environment: FulfillmentEnvironment, calls: _Calls
) -> None:
    target = corpus.declare(
        "unbound-bulletin",
        layer=BULLETIN_LAYER,
        kind=BULLETIN_KIND,
        edges=(
            EdgeDeclaration(
                role_path=("Nominal", "k1"),
                external={
                    "manifest_kind": "EvaluationRunManifest",
                    "manifest_id": "feedbax-evaluation-run:never-produced",
                },
            ),
        ),
    )
    with pytest.raises(MissingExternalReceiptError):
        _fulfill(corpus, target, environment=environment)
    assert calls.report == 0
    assert not _reports_directory(environment).exists()


# --------------------------------------------------------------------------
# Certified omissions at preflight
# --------------------------------------------------------------------------


def _underspecified_bulletin(corpus: _Corpus, name: str) -> str:
    """A bulletin binding `k1` and leaving `s1` certified-unbound."""
    _sample(corpus, "slot-source")
    return corpus.declare(
        name,
        layer=BULLETIN_LAYER,
        kind=BULLETIN_KIND,
        edges=(
            EdgeDeclaration(role_path=("Nominal", "k1"), producer_ref="slot-source.decl"),
            EdgeDeclaration(
                role_path=("Appendix", "s1"),
                status="not_applicable",
                basis="compiler_rule",
                reason=f"no producer in this closure binds the slot ({UNBOUND_SLOT_RULE})",
                rule=UNBOUND_SLOT_RULE,
            ),
            EdgeDeclaration(
                role_path=("Nominal", "k2"),
                status="not_applicable",
                basis="authored",
                reason="this target has no second controller",
            ),
        ),
    )


def _omit(payload: dict[str, Any], edges) -> dict[str, Any]:
    """This corpus's way of saying a slot was materialized as inapplicable."""
    payload["params"] = {
        **payload["params"],
        "omitted": sorted("/".join(edge.role_path) for edge in edges),
    }
    return payload


def test_without_the_flag_a_certified_omission_refuses_at_preflight(
    corpus: _Corpus, environment: FulfillmentEnvironment, calls: _Calls
) -> None:
    target = _underspecified_bulletin(corpus, "underspecified-bulletin")
    with pytest.raises(CertifiedOmissionsPendingError) as caught:
        _fulfill(corpus, target, environment=environment)
    assert [record.role_path for record in caught.value.omissions] == [("Appendix", "s1")]
    assert calls.evaluation == 0
    assert not environment.root.exists()


def test_the_flag_materializes_the_certified_omissions_and_mints_a_new_identity(
    corpus: _Corpus, environment: FulfillmentEnvironment
) -> None:
    target = _underspecified_bulletin(corpus, "omitted-bulletin")
    closure = _closure(corpus, target, omit_certified=True, apply_omission=_omit)
    assert [record.role_path for record in closure.omissions] == [("Appendix", "s1")]
    assert all(record.rule == UNBOUND_SLOT_RULE for record in closure.omissions)
    assert all(record.basis == "compiler_rule" for record in closure.omissions)

    run = fulfill_closure(closure, environment=environment, lower=lower)
    omitted_id = run.results[-1].receipt.manifest_id
    omitted_manifest = load_manifest(run.results[-1].receipt.path)
    assert omitted_manifest.report_spec.inline["params"]["omitted"] == ["Appendix/s1"]

    unomitted = fulfill_closure(
        _closure(
            corpus,
            _bulletin(corpus, "specified-bulletin", k1="slot-source.decl"),
        ),
        environment=environment,
        lower=lower,
    )
    assert unomitted.results[-1].receipt.manifest_id != omitted_id, (
        "an omitted document is a different document, so it has a different identity"
    )


def test_an_uncertified_absence_is_never_omitted(corpus: _Corpus) -> None:
    """The flag materializes only what a closed rule certified."""
    _sample(corpus, "certified-source")
    target = _bulletin(corpus, "certified-only-bulletin", k1="certified-source.decl")
    closure = _closure(corpus, target, omit_certified=True, apply_omission=_omit)
    assert closure.omissions == ()
    plain = _closure(corpus, target)
    assert [node.payload for node in closure.nodes] == [node.payload for node in plain.nodes]


def test_the_flag_needs_a_way_for_a_payload_to_say_a_decision_was_materialized(
    corpus: _Corpus,
) -> None:
    target = _underspecified_bulletin(corpus, "no-applier-bulletin")
    with pytest.raises(ValueError, match="apply_omission"):
        _closure(corpus, target, omit_certified=True)


# --------------------------------------------------------------------------
# Rebuild as verification, repair as recovery
# --------------------------------------------------------------------------


def test_closure_requests_resolve_in_the_plan_order(
    corpus: _Corpus, environment: FulfillmentEnvironment
) -> None:
    target = _chain(corpus)
    _fulfill(corpus, target, environment=environment)
    requests = closure_requests(_closure(corpus, target), environment=environment, lower=lower)
    assert [request.node_key for request in requests] == list(CHAIN_EXECUTION_ORDER)
    assert [request.order for request in requests] == [0, 1, 2]
    assert all(request.required_output_roles == () for request in requests)


def test_rebuilding_an_intact_closure_reports_no_drift_and_preserves_receipts(
    corpus: _Corpus, environment: FulfillmentEnvironment
) -> None:
    target = _chain(corpus)
    run = _fulfill(corpus, target, environment=environment)
    before = {result.receipt.path: result.receipt.path.read_bytes() for result in run.results}

    rebuilt = rebuild_closure(_closure(corpus, target), environment=environment, lower=lower)
    assert rebuilt.verification_order == CHAIN_EXECUTION_ORDER
    assert rebuilt.drifted == ()
    assert {path: path.read_bytes() for path in before} == before


def test_a_receipt_that_disagrees_with_a_clean_re_execution_drifts(
    corpus: _Corpus, environment: FulfillmentEnvironment
) -> None:
    """Drift is reported per node against the defined projection, original intact."""
    target = _sample(corpus, "drifting-sample")
    receipt = _fulfill(corpus, target, environment=environment).results[0].receipt
    document = json.loads(receipt.path.read_text(encoding="utf-8"))
    document["summary_metrics"] = {**document.get("summary_metrics", {}), "stage": "tampered"}
    receipt.path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(FulfillmentDriftError) as caught:
        rebuild_closure(_closure(corpus, target), environment=environment, lower=lower)
    (outcome,) = caught.value.drifted
    assert outcome.node_key == "sample:drifting-sample"
    assert json.loads(receipt.path.read_text(encoding="utf-8"))["summary_metrics"]["stage"] == (
        "tampered"
    ), "the authoritative receipt is never written to by a rebuild"


def test_altered_stored_bytes_refuse_before_any_rebuild(
    corpus: _Corpus, environment: FulfillmentEnvironment, calls: _Calls
) -> None:
    target = _chain(corpus)
    run = _fulfill(corpus, target, environment=environment)
    artifact = load_manifest(run.results[0].receipt.path).artifacts[0]
    (environment.root / artifact.metadata["relative_path"]).write_bytes(b"corrupt")

    with pytest.raises(FulfillmentAdmissionError) as caught:
        rebuild_closure(_closure(corpus, target), environment=environment, lower=lower)
    assert "artifact_sha256_mismatch" in caught.value.outcome.codes
    assert calls.evaluation == 3, "corruption is custody loss, not drift, so nothing rebuilt"


def test_repair_promotes_a_revalidated_candidate_and_records_the_custody_event(
    corpus: _Corpus, environment: FulfillmentEnvironment
) -> None:
    target = _chain(corpus)
    run = _fulfill(corpus, target, environment=environment)
    receipt = run.results[1].receipt
    _mutate(receipt.path, status="failed")

    result = repair_closure_node(
        _closure(corpus, target),
        LogicalKey(DIGEST_LAYER, "chain-mid"),
        environment=environment,
        lower=lower,
    )
    assert result.record.node_key == "digest:chain-mid"
    assert result.record.triggering_admission.codes == ("status_not_completed",)
    assert result.record.admission_after_repair.admitted
    assert result.record_path.is_file()
    assert load_manifest(receipt.path).status == "completed"


def test_resolution_refuses_before_repairing_around_a_broken_upstream(
    corpus: _Corpus, environment: FulfillmentEnvironment
) -> None:
    target = _chain(corpus)
    run = _fulfill(corpus, target, environment=environment)
    _mutate(run.results[0].receipt.path, status="failed")
    with pytest.raises(FulfillmentAdmissionError):
        repair_closure_node(
            _closure(corpus, target),
            LogicalKey(DIGEST_LAYER, "chain-leaf"),
            environment=environment,
            lower=lower,
        )


# --------------------------------------------------------------------------
# What the driver refuses to guess
# --------------------------------------------------------------------------


def test_a_lowering_that_returns_a_foreign_node_key_refuses(
    corpus: _Corpus, environment: FulfillmentEnvironment
) -> None:
    closure = _closure(corpus, _sample(corpus, "keyed"))

    def mislabelled(node, *, binding):
        request = lower(node, binding=binding)
        return EvaluationNodeRequest(
            node_key="somebody-elses-key", spec=request.spec, order=request.order
        )

    with pytest.raises(AmbiguousNodeReceiptError, match="carries the logical key"):
        fulfill_closure(closure, environment=environment, lower=mislabelled)


def test_a_matrix_node_has_no_single_receipt_to_resolve(
    corpus: _Corpus, environment: FulfillmentEnvironment
) -> None:
    """Resolving one receipt for a matrix would mean choosing among its rows."""
    closure = _closure(corpus, _sample(corpus, "matrix-node"))

    def matrix_lowering(node, *, binding):
        return EvaluationMatrixNodeRequest(
            node_key=node.key.text, matrix={"rows": []}, order=node.order
        )

    with pytest.raises(AmbiguousNodeReceiptError, match="evaluation matrix"):
        closure_requests(closure, environment=environment, lower=matrix_lowering)
