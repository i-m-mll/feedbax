"""The fulfillment plan kernel: closure, logical addressing, typed applicability.

Everything here is stated against an invented corpus whose layers, node kinds,
and rule names are this module's own, so the rules under test are the kernel's
rather than any project's. Three claims are under test:

* a plan is the *closure* of one target, derived from declarations alone — no
  receipt root is read and no path is inferred;
* a node is addressed by a *logical key*, so one declaration is one node no
  matter how many referrers reach it, and two refs that claim one key refuse;
* every input edge carries a typed applicability decision, and only a decision a
  closed rule certified may be materialized.
"""

from __future__ import annotations

from typing import Any

import pytest

from feedbax.analysis.fulfillment_plan import (
    APPLICABILITY_BASES,
    APPLICABILITY_STATUSES,
    FULFILLMENT_PLAN_SCHEMA_ID,
    FULFILLMENT_PLAN_SCHEMA_VERSION_V1,
    FULFILLMENT_PLAN_SUPPORTED_SCHEMA_VERSIONS,
    CertifiedOmissionsPendingError,
    DuplicateLogicalKeyError,
    EdgeDeclaration,
    LogicalKey,
    NodeDeclaration,
    PlanCycleError,
    PlanEdge,
    PlanNode,
    UnresolvedPlanReferenceError,
    UnsupportedFulfillmentPlanVersionError,
    apply_certified_omissions,
    build_fulfillment_plan,
    expand_fulfillment_plan,
    fulfillment_plan_from_document,
    read_fulfillment_plan_document,
    require_no_certified_omissions,
)
from feedbax.contracts.manifest import canonical_json_bytes


#: The invented corpus's layers, node kinds, and the one rule it certifies with.
SAMPLE_LAYER = "sample"
DIGEST_LAYER = "digest"
BULLETIN_LAYER = "bulletin"
HARVEST_LAYER = "harvest"

SAMPLE_KIND = "toybox.sample"
DIGEST_KIND = "toybox.digest"
BULLETIN_KIND = "toybox.bulletin"

UNBOUND_SLOT_RULE = "unbound_inherited_slot"


class _Corpus:
    """A tiny declaration store standing in for a project's compiler."""

    def __init__(self) -> None:
        self.declarations: dict[str, NodeDeclaration] = {}
        self.expansions: list[str] = []

    def declare(
        self,
        ref: str,
        key: LogicalKey,
        *,
        kind: str,
        edges: tuple[EdgeDeclaration, ...] = (),
        boundary: str | None = None,
        content_hash: str | None = None,
    ) -> str:
        self.declarations[ref] = NodeDeclaration(
            node=PlanNode(
                key=key,
                source_ref=ref,
                kind=kind,
                content_hash=content_hash,
                execution_identity=f"identity-of-{ref}",
                boundary=boundary,
            ),
            edges=edges,
        )
        return ref

    def expand(self, ref: str) -> NodeDeclaration:
        self.expansions.append(ref)
        if ref not in self.declarations:
            raise KeyError(f"{ref} is not declared in this corpus")
        return self.declarations[ref]


@pytest.fixture
def corpus() -> _Corpus:
    return _Corpus()


def _sample(corpus: _Corpus, name: str) -> str:
    return corpus.declare(f"{name}.decl", LogicalKey(SAMPLE_LAYER, name), kind=SAMPLE_KIND)


def _digest(corpus: _Corpus, name: str, **roles: str) -> str:
    return corpus.declare(
        f"{name}.decl",
        LogicalKey(DIGEST_LAYER, name),
        kind=DIGEST_KIND,
        edges=tuple(
            EdgeDeclaration(role_path=(role,), producer_ref=ref) for role, ref in roles.items()
        ),
    )


# --- the family states its own identity ------------------------------------


def test_the_plan_family_enumerates_its_supported_versions() -> None:
    assert FULFILLMENT_PLAN_SUPPORTED_SCHEMA_VERSIONS == (FULFILLMENT_PLAN_SCHEMA_VERSION_V1,)
    assert FULFILLMENT_PLAN_SCHEMA_VERSION_V1.startswith(f"{FULFILLMENT_PLAN_SCHEMA_ID}.")


def test_the_applicability_vocabulary_is_closed() -> None:
    assert APPLICABILITY_STATUSES == ("required", "not_applicable")
    assert APPLICABILITY_BASES == ("authored", "compiler_rule")
    key = LogicalKey(BULLETIN_LAYER, "any-bulletin")
    with pytest.raises(ValueError):
        PlanEdge(key, ("x",), "maybe", "authored")
    with pytest.raises(ValueError):
        PlanEdge(key, ("x",), "required", "guessed")
    with pytest.raises(ValueError):
        PlanEdge(key, ("x",), "not_applicable", "authored", producer=key)


def test_a_rule_based_decision_names_the_rule_that_certified_it() -> None:
    key = LogicalKey(BULLETIN_LAYER, "any-bulletin")
    with pytest.raises(ValueError, match="names the closed rule"):
        PlanEdge(key, ("x",), "not_applicable", "compiler_rule", reason="because")
    with pytest.raises(ValueError, match="names no rule"):
        PlanEdge(key, ("x",), "required", "authored", rule=UNBOUND_SLOT_RULE)
    certified = PlanEdge(
        key, ("x",), "not_applicable", "compiler_rule", reason="because", rule=UNBOUND_SLOT_RULE
    )
    assert certified.certified


def test_an_input_binds_one_thing_only() -> None:
    key = LogicalKey(DIGEST_LAYER, "d")
    with pytest.raises(ValueError, match="binds one thing"):
        PlanEdge(
            key,
            ("x",),
            "required",
            "authored",
            producer=LogicalKey(SAMPLE_LAYER, "s"),
            external={"pin": "abc"},
        )


def test_a_logical_key_cannot_be_forged_by_a_name() -> None:
    """The canonical text quotes its parts, so no name can spell a separator."""
    assert LogicalKey("figure", "a:b/c").text == "figure:a%3Ab%2Fc"
    assert LogicalKey("figure", "a:b/c").text != LogicalKey("figure", "a").text
    assert LogicalKey.parse(LogicalKey("figure", "a:b/c").text) == LogicalKey("figure", "a:b/c")
    with pytest.raises(ValueError, match="not a layer name"):
        LogicalKey("figure:inner", "a")
    with pytest.raises(ValueError):
        LogicalKey("figure", "")


def test_an_emitted_plan_is_admitted_by_its_own_reader(corpus: _Corpus) -> None:
    plan = expand_fulfillment_plan(_sample(corpus, "solo"), expand=corpus.expand)
    document = plan.document()
    assert (
        read_fulfillment_plan_document(document)["schema_version"]
        == FULFILLMENT_PLAN_SCHEMA_VERSION_V1
    )


def test_an_unknown_plan_version_fails_closed(corpus: _Corpus) -> None:
    document = expand_fulfillment_plan(_sample(corpus, "solo"), expand=corpus.expand).document()
    document["schema_version"] = "feedbax.fulfillment.plan.v9"
    with pytest.raises(UnsupportedFulfillmentPlanVersionError) as caught:
        read_fulfillment_plan_document(document)
    assert FULFILLMENT_PLAN_SCHEMA_VERSION_V1 in str(caught.value)
    assert "no migration" in str(caught.value)


def test_a_foreign_schema_id_is_not_read_as_a_plan(corpus: _Corpus) -> None:
    document = expand_fulfillment_plan(_sample(corpus, "solo"), expand=corpus.expand).document()
    document["schema_id"] = "feedbax.spec.analysis_bundle"
    with pytest.raises(UnsupportedFulfillmentPlanVersionError) as caught:
        read_fulfillment_plan_document(document)
    assert FULFILLMENT_PLAN_SCHEMA_ID in str(caught.value)


def test_a_document_that_is_not_a_mapping_is_refused() -> None:
    with pytest.raises(UnsupportedFulfillmentPlanVersionError):
        read_fulfillment_plan_document(["not", "a", "plan"])


def test_a_plan_round_trips_through_its_document(corpus: _Corpus) -> None:
    _sample(corpus, "left")
    _sample(corpus, "right")
    target = _digest(corpus, "joined", left="left.decl", right="right.decl")
    plan = expand_fulfillment_plan(
        target, expand=corpus.expand, origin={"compiler": "toybox", "version": 3}
    )
    restored = fulfillment_plan_from_document(plan.document())
    assert restored.document() == plan.document()
    assert restored.target == plan.target
    assert restored.origin == {"compiler": "toybox", "version": 3}


def test_two_builds_of_one_plan_emit_identical_documents(corpus: _Corpus) -> None:
    _sample(corpus, "alpha")
    _sample(corpus, "beta")
    target = _digest(corpus, "stable", beta="beta.decl", alpha="alpha.decl")
    first = expand_fulfillment_plan(target, expand=corpus.expand).document()
    second = expand_fulfillment_plan(target, expand=corpus.expand).document()
    assert canonical_json_bytes(first) == canonical_json_bytes(second)


# --- the closure is derived from declarations, not from the filesystem ------


def test_a_plan_reaches_every_declaration_the_target_names(corpus: _Corpus) -> None:
    _sample(corpus, "subject-one")
    _sample(corpus, "subject-two")
    target = _digest(
        corpus, "two-subject", left="subject-one.decl", right="subject-two.decl"
    )
    plan = expand_fulfillment_plan(target, expand=corpus.expand)

    assert plan.target == LogicalKey(DIGEST_LAYER, "two-subject")
    assert {node.key for node in plan.nodes} == {
        LogicalKey(DIGEST_LAYER, "two-subject"),
        LogicalKey(SAMPLE_LAYER, "subject-one"),
        LogicalKey(SAMPLE_LAYER, "subject-two"),
    }
    producers = {edge.role_path: edge.producer for edge in plan.edges}
    assert producers[("left",)] == LogicalKey(SAMPLE_LAYER, "subject-one")
    assert producers[("right",)] == LogicalKey(SAMPLE_LAYER, "subject-two")
    assert all(edge.status == "required" and edge.basis == "authored" for edge in plan.edges)


def test_a_node_carries_exactly_what_its_declaration_pinned(corpus: _Corpus) -> None:
    corpus.declare(
        "pinned.decl",
        LogicalKey(SAMPLE_LAYER, "pinned"),
        kind=SAMPLE_KIND,
        content_hash="hash-of-pinned",
    )
    target = _digest(corpus, "pinned-consumer", only="pinned.decl")
    plan = expand_fulfillment_plan(target, expand=corpus.expand)
    node = plan.node(LogicalKey(SAMPLE_LAYER, "pinned"))
    assert node.content_hash == "hash-of-pinned"
    assert node.execution_identity == "identity-of-pinned.decl"
    assert node.source_ref == "pinned.decl"
    assert node.kind == SAMPLE_KIND


def test_producers_precede_their_consumers_in_the_emitted_order(corpus: _Corpus) -> None:
    _sample(corpus, "ordered-a")
    _sample(corpus, "ordered-b")
    target = _digest(corpus, "ordered", a="ordered-a.decl", b="ordered-b.decl")
    plan = expand_fulfillment_plan(target, expand=corpus.expand)
    assert [node.key for node in plan.nodes] == [
        LogicalKey(SAMPLE_LAYER, "ordered-a"),
        LogicalKey(SAMPLE_LAYER, "ordered-b"),
        LogicalKey(DIGEST_LAYER, "ordered"),
    ]


def test_a_deep_chain_orders_every_producer_before_its_consumer(corpus: _Corpus) -> None:
    _sample(corpus, "root")
    _digest(corpus, "mid", up="root.decl")
    target = corpus.declare(
        "leaf.decl",
        LogicalKey(BULLETIN_LAYER, "leaf"),
        kind=BULLETIN_KIND,
        edges=(EdgeDeclaration(role_path=("middle",), producer_ref="mid.decl"),),
    )
    plan = expand_fulfillment_plan(target, expand=corpus.expand)
    assert [node.key.text for node in plan.nodes] == [
        "sample:root",
        "digest:mid",
        "bulletin:leaf",
    ]


def test_one_declaration_reached_twice_is_one_node(corpus: _Corpus) -> None:
    _sample(corpus, "shared")
    target = _digest(corpus, "diamond", left="shared.decl", right="shared.decl")
    plan = expand_fulfillment_plan(target, expand=corpus.expand)
    assert len(plan.nodes) == 2
    assert len([edge for edge in plan.edges if edge.producer is not None]) == 2
    assert corpus.expansions.count("shared.decl") == 1, "one declaration is expanded once"


def test_two_refs_claiming_one_logical_key_refuse(corpus: _Corpus) -> None:
    corpus.declare("collided-first.decl", LogicalKey(SAMPLE_LAYER, "collided"), kind=SAMPLE_KIND)
    corpus.declare("collided-second.decl", LogicalKey(SAMPLE_LAYER, "collided"), kind=SAMPLE_KIND)
    target = _digest(
        corpus, "collision", first="collided-first.decl", second="collided-second.decl"
    )
    with pytest.raises(DuplicateLogicalKeyError) as caught:
        expand_fulfillment_plan(target, expand=corpus.expand)
    assert caught.value.key == LogicalKey(SAMPLE_LAYER, "collided")
    assert "sample:collided" in str(caught.value)


def test_a_reference_cycle_is_a_structured_refusal(corpus: _Corpus) -> None:
    corpus.declare(
        "cyclic.decl",
        LogicalKey(DIGEST_LAYER, "cyclic"),
        kind=DIGEST_KIND,
        edges=(EdgeDeclaration(role_path=("self",), producer_ref="cyclic.decl"),),
    )
    with pytest.raises(PlanCycleError) as caught:
        expand_fulfillment_plan("cyclic.decl", expand=corpus.expand)
    assert caught.value.unplaced == ("digest:cyclic",)
    assert "cyclic" in str(caught.value)


def test_a_two_node_cycle_is_named_rather_than_truncating_the_plan(corpus: _Corpus) -> None:
    corpus.declare(
        "a.decl",
        LogicalKey(DIGEST_LAYER, "a"),
        kind=DIGEST_KIND,
        edges=(EdgeDeclaration(role_path=("b",), producer_ref="b.decl"),),
    )
    corpus.declare(
        "b.decl",
        LogicalKey(DIGEST_LAYER, "b"),
        kind=DIGEST_KIND,
        edges=(EdgeDeclaration(role_path=("a",), producer_ref="a.decl"),),
    )
    with pytest.raises(PlanCycleError) as caught:
        expand_fulfillment_plan("a.decl", expand=corpus.expand)
    assert caught.value.unplaced == ("digest:a", "digest:b")


def test_an_edge_naming_an_undeclared_producer_refuses() -> None:
    consumer = PlanNode(
        key=LogicalKey(DIGEST_LAYER, "orphaned"), source_ref="orphaned.decl", kind=DIGEST_KIND
    )
    edge = PlanEdge(
        consumer.key, ("gone",), "required", "authored", producer=LogicalKey(SAMPLE_LAYER, "gone")
    )
    with pytest.raises(UnresolvedPlanReferenceError) as caught:
        build_fulfillment_plan(consumer.key, (consumer,), (edge,))
    assert caught.value.missing == LogicalKey(SAMPLE_LAYER, "gone")


def test_a_target_no_declaration_provides_refuses() -> None:
    node = PlanNode(
        key=LogicalKey(SAMPLE_LAYER, "only"), source_ref="only.decl", kind=SAMPLE_KIND
    )
    with pytest.raises(UnresolvedPlanReferenceError):
        build_fulfillment_plan(LogicalKey(DIGEST_LAYER, "absent"), (node,), ())


def test_a_declaration_the_target_does_not_reach_is_not_in_the_closure() -> None:
    """A plan is one target's closure, not every declaration somebody handed it."""
    reached = PlanNode(
        key=LogicalKey(SAMPLE_LAYER, "reached"), source_ref="reached.decl", kind=SAMPLE_KIND
    )
    target = PlanNode(
        key=LogicalKey(DIGEST_LAYER, "target"), source_ref="target.decl", kind=DIGEST_KIND
    )
    stray = PlanNode(
        key=LogicalKey(SAMPLE_LAYER, "stray"), source_ref="stray.decl", kind=SAMPLE_KIND
    )
    edges = (
        PlanEdge(target.key, ("only",), "required", "authored", producer=reached.key),
        PlanEdge(stray.key, ("nothing",), "required", "authored", external={"pin": "x"}),
    )
    plan = build_fulfillment_plan(target.key, (target, reached, stray), edges)
    assert [node.key.text for node in plan.nodes] == ["sample:reached", "digest:target"]
    assert [edge.consumer.text for edge in plan.edges] == ["digest:target"]


def test_an_external_input_is_an_edge_and_never_a_node(corpus: _Corpus) -> None:
    target = corpus.declare(
        "external-consumer.decl",
        LogicalKey(DIGEST_LAYER, "external-consumer"),
        kind=DIGEST_KIND,
        edges=(
            EdgeDeclaration(
                role_path=("old",), external={"kind": "prior_run", "run_id": "run-1234"}
            ),
        ),
    )
    plan = expand_fulfillment_plan(target, expand=corpus.expand)
    assert [node.key for node in plan.nodes] == [LogicalKey(DIGEST_LAYER, "external-consumer")]
    edge = next(edge for edge in plan.edges if edge.role_path == ("old",))
    assert edge.producer is None
    assert edge.external == {"kind": "prior_run", "run_id": "run-1234"}
    assert edge.status == "required"


def test_a_missing_receipt_is_never_read_as_inapplicability(corpus: _Corpus) -> None:
    """Nothing in the plan consults custody, so absence cannot certify anything.

    The bound input names a receipt no root in this test holds; the plan still
    calls it required, because a plan states what must exist rather than what
    does.
    """
    target = corpus.declare(
        "absent-consumer.decl",
        LogicalKey(DIGEST_LAYER, "absent-consumer"),
        kind=DIGEST_KIND,
        edges=(
            EdgeDeclaration(
                role_path=("old",),
                external={"kind": "prior_run", "run_id": "never-produced"},
            ),
        ),
    )
    plan = expand_fulfillment_plan(target, expand=corpus.expand)
    edge = next(edge for edge in plan.edges if edge.role_path == ("old",))
    assert (edge.status, edge.basis) == ("required", "authored")
    assert plan.certified_omissions() == ()


def test_a_boundary_node_is_carried_as_a_node_the_lowerer_marked(corpus: _Corpus) -> None:
    corpus.declare(
        "harvested.decl",
        LogicalKey(HARVEST_LAYER, "harvested"),
        kind=SAMPLE_KIND,
        boundary="harvest",
    )
    target = _digest(corpus, "needs-harvest", picked="harvested.decl")
    plan = expand_fulfillment_plan(target, expand=corpus.expand)
    (boundary,) = plan.boundary_nodes()
    assert boundary.key == LogicalKey(HARVEST_LAYER, "harvested")
    assert boundary.boundary == "harvest"
    assert plan.descendants(boundary.key) == ("digest:needs-harvest",)


def test_descendants_lists_the_whole_subtree_one_node_unblocks(corpus: _Corpus) -> None:
    _sample(corpus, "seed")
    _digest(corpus, "near", from_seed="seed.decl")
    target = corpus.declare(
        "far.decl",
        LogicalKey(BULLETIN_LAYER, "far"),
        kind=BULLETIN_KIND,
        edges=(EdgeDeclaration(role_path=("near",), producer_ref="near.decl"),),
    )
    plan = expand_fulfillment_plan(target, expand=corpus.expand)
    assert plan.descendants(LogicalKey(SAMPLE_LAYER, "seed")) == ("bulletin:far", "digest:near")


# --- typed applicability and certified omissions ----------------------------


def _bulletin_with_slots(corpus: _Corpus) -> str:
    """One bulletin binding `k1`, authoring `k2` away, and leaving `s1` unbound."""
    _sample(corpus, "k1-source")
    return corpus.declare(
        "bulletin.decl",
        LogicalKey(BULLETIN_LAYER, "bulletin"),
        kind=BULLETIN_KIND,
        edges=(
            EdgeDeclaration(role_path=("Nominal", "k1"), producer_ref="k1-source.decl"),
            EdgeDeclaration(
                role_path=("Nominal", "k2"),
                status="not_applicable",
                basis="authored",
                reason="this target has no second controller",
            ),
            EdgeDeclaration(
                role_path=("Appendix", "s1"),
                status="not_applicable",
                basis="compiler_rule",
                reason=f"no producer in this closure binds the slot ({UNBOUND_SLOT_RULE})",
                rule=UNBOUND_SLOT_RULE,
            ),
        ),
    )


def test_certified_omissions_are_only_the_rule_based_ones(corpus: _Corpus) -> None:
    plan = expand_fulfillment_plan(_bulletin_with_slots(corpus), expand=corpus.expand)
    certified = {edge.role_path: edge for edge in plan.certified_omissions()}
    assert set(certified) == {("Appendix", "s1")}
    for edge in certified.values():
        assert edge.status == "not_applicable"
        assert edge.producer is None and edge.external is None
        assert edge.rule == UNBOUND_SLOT_RULE
    authored = [
        edge for edge in plan.edges if edge.status == "not_applicable" and edge.basis == "authored"
    ]
    assert [edge.reason for edge in authored] == ["this target has no second controller"]


def test_required_edges_are_the_inputs_something_must_bind(corpus: _Corpus) -> None:
    plan = expand_fulfillment_plan(_bulletin_with_slots(corpus), expand=corpus.expand)
    required = plan.required_edges(plan.target)
    assert [edge.role_path for edge in required] == [("Nominal", "k1")]


def test_without_the_flag_a_certified_omission_refuses_with_its_records(
    corpus: _Corpus,
) -> None:
    plan = expand_fulfillment_plan(_bulletin_with_slots(corpus), expand=corpus.expand)
    with pytest.raises(CertifiedOmissionsPendingError) as caught:
        require_no_certified_omissions(plan)
    (record,) = caught.value.omissions
    assert record.consumer == plan.target
    assert record.role_path == ("Appendix", "s1")
    assert record.rule == UNBOUND_SLOT_RULE
    assert record.basis == "compiler_rule"
    assert UNBOUND_SLOT_RULE in str(caught.value)


def test_a_fully_specified_target_needs_no_refusal(corpus: _Corpus) -> None:
    _sample(corpus, "bound-source")
    target = corpus.declare(
        "specified.decl",
        LogicalKey(BULLETIN_LAYER, "specified"),
        kind=BULLETIN_KIND,
        edges=(
            EdgeDeclaration(role_path=("Nominal", "k1"), producer_ref="bound-source.decl"),
            EdgeDeclaration(
                role_path=("Nominal", "k2"),
                status="not_applicable",
                basis="authored",
                reason="no second controller in this target",
            ),
        ),
    )
    plan = expand_fulfillment_plan(target, expand=corpus.expand)
    assert plan.certified_omissions() == ()
    require_no_certified_omissions(plan)


def test_apply_certified_omissions_passes_only_certified_edges(corpus: _Corpus) -> None:
    """The never-omit-uncertified invariant: authored and required never reach it."""
    plan = expand_fulfillment_plan(_bulletin_with_slots(corpus), expand=corpus.expand)
    seen: list[tuple[str, ...]] = []

    def apply(payload: dict[str, Any], edges) -> dict[str, Any]:
        seen.extend(edge.role_path for edge in edges)
        omitted = dict(payload)
        omitted["omitted"] = sorted("/".join(edge.role_path) for edge in edges)
        return omitted

    document = {"slots": ["Nominal/k1", "Nominal/k2", "Appendix/s1"]}
    result, records = apply_certified_omissions(
        document, plan, consumer=plan.target, apply=apply
    )
    assert seen == [("Appendix", "s1")]
    assert result["omitted"] == ["Appendix/s1"]
    assert [record.role_path for record in records] == [("Appendix", "s1")]
    assert all(record.rule == UNBOUND_SLOT_RULE for record in records)
    assert all(record.basis == "compiler_rule" for record in records)
    assert document == {"slots": ["Nominal/k1", "Nominal/k2", "Appendix/s1"]}, (
        "the original document is never mutated in place"
    )


def test_a_target_with_nothing_certified_is_returned_unchanged(corpus: _Corpus) -> None:
    _sample(corpus, "only-source")
    target = _digest(corpus, "nothing-certified", only="only-source.decl")
    plan = expand_fulfillment_plan(target, expand=corpus.expand)

    def apply(payload, edges):  # pragma: no cover - must never be called
        raise AssertionError("nothing was certified, so nothing may be materialized")

    document = {"slots": ["only"]}
    result, records = apply_certified_omissions(
        document, plan, consumer=plan.target, apply=apply
    )
    assert records == ()
    assert result == document and result is not document


def test_an_omission_record_serializes_deterministically(corpus: _Corpus) -> None:
    plan = expand_fulfillment_plan(_bulletin_with_slots(corpus), expand=corpus.expand)
    with pytest.raises(CertifiedOmissionsPendingError) as caught:
        require_no_certified_omissions(plan)
    assert caught.value.omissions[0].record() == {
        "consumer": "bulletin:bulletin",
        "role_path": ["Appendix", "s1"],
        "reason": f"no producer in this closure binds the slot ({UNBOUND_SLOT_RULE})",
        "basis": "compiler_rule",
        "rule": UNBOUND_SLOT_RULE,
    }
