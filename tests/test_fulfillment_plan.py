"""The fulfillment plan kernel: closure, logical addressing, typed applicability.

Everything here is stated against an invented corpus whose layers, node kinds,
and rule names are this module's own, so the rules under test are the kernel's
rather than any project's. Three claims are under test:

* a plan is the *closure* of one target, derived from declarations alone — no
  receipt root is read and no path is inferred;
* a node is addressed by a *logical key*, so one declaration is one node no
  matter how many referrers reach it, and two refs that claim one key refuse;
* every input edge carries a typed applicability decision, and the two bases it
  may be reached on stay distinguishable on the plan.
"""

from __future__ import annotations

import pytest

from feedbax.analysis.fulfillment_plan import (
    APPLICABILITY_BASES,
    APPLICABILITY_STATUSES,
    FULFILLMENT_PLAN_SCHEMA_ID,
    FULFILLMENT_PLAN_SCHEMA_VERSION_V1,
    FULFILLMENT_PLAN_SUPPORTED_SCHEMA_VERSIONS,
    ConflictingNodeDeclarationError,
    DuplicateInputEdgeError,
    DuplicateLogicalKeyError,
    EdgeDeclaration,
    LogicalKey,
    NodeDeclaration,
    PlanCycleError,
    PlanEdge,
    PlanNode,
    UnresolvedPlanReferenceError,
    UnsupportedFulfillmentPlanVersionError,
    build_fulfillment_plan,
    expand_fulfillment_plan,
    fulfillment_plan_from_document,
    read_fulfillment_plan_document,
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


def test_two_declarations_of_one_node_that_disagree_refuse_rather_than_first_winning() -> None:
    """A repeat states the same node twice; a conflict states two nodes at one address.

    Two entries at one logical key sharing a source ref used to be admitted
    without comparing anything else they said, so a second declaration could
    restate the node's pinned document, the schema it is lowered by, its
    execution identity, or its boundary, and be discarded in favour of whichever
    the loader happened to see first. Nothing downstream could report the
    discarded claim, because nothing downstream ever saw it.
    """
    key = LogicalKey(DIGEST_LAYER, "restated")
    honest = PlanNode(
        key=key,
        source_ref="restated.decl",
        kind=DIGEST_KIND,
        content_hash="a" * 64,
        execution_identity="b" * 64,
    )
    restated = PlanNode(
        key=key,
        source_ref="restated.decl",
        kind=DIGEST_KIND,
        content_hash="c" * 64,
        execution_identity="b" * 64,
    )

    with pytest.raises(ConflictingNodeDeclarationError) as caught:
        build_fulfillment_plan(key, (honest, restated), ())

    differences = caught.value.differences
    assert any(difference.startswith("content_hash:") for difference in differences)
    assert "a" * 64 in str(caught.value) and "c" * 64 in str(caught.value)
    assert caught.value.key == key


def test_a_genuinely_repeated_node_declaration_is_still_one_node() -> None:
    """Stating the same node twice says one thing twice, and is admitted."""
    key = LogicalKey(DIGEST_LAYER, "repeated")
    node = PlanNode(
        key=key, source_ref="repeated.decl", kind=DIGEST_KIND, content_hash="d" * 64
    )
    plan = build_fulfillment_plan(key, (node, node), ())
    assert len(plan.nodes) == 1
    assert plan.nodes[0].content_hash == "d" * 64


def test_a_restated_boundary_that_disagrees_refuses_at_plan_construction() -> None:
    """Erasing a boundary on a second declaration is a disagreement, not a repeat."""
    key = LogicalKey(DIGEST_LAYER, "bounded")
    bounded = PlanNode(
        key=key, source_ref="bounded.decl", kind=DIGEST_KIND, boundary="some.boundary"
    )
    unbounded = PlanNode(key=key, source_ref="bounded.decl", kind=DIGEST_KIND)

    with pytest.raises(ConflictingNodeDeclarationError) as caught:
        build_fulfillment_plan(key, (bounded, unbounded), ())
    assert any(
        difference.startswith("boundary:") for difference in caught.value.differences
    )


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


def test_a_fully_bound_target_certifies_nothing(corpus: _Corpus) -> None:
    """An authored inapplicability is not a rule-certified one."""
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
    assert [edge.role_path for edge in plan.input_edges(plan.target, status="not_applicable")] == [
        ("Nominal", "k2")
    ]


# --- one role path, one edge -----------------------------------------------


def test_two_edges_at_one_role_path_refuse(corpus: _Corpus) -> None:
    """The reviewer's reproduction, at the kernel that admits the plan.

    Reachability walks every edge; every later consumer of the plan addresses an
    edge by ``(consumer, role_path)`` and therefore keys a dict. A duplicate is
    live for the first and invisible to the second, so an injected producer can
    enter the execution order without any lock ever being asked about it. One
    role, one edge, refused where plans are built.
    """
    genuine = _sample(corpus, "genuine-source")
    injected = _sample(corpus, "injected-extra")
    consumer = LogicalKey(DIGEST_LAYER, "duplicated")
    with pytest.raises(DuplicateInputEdgeError) as caught:
        build_fulfillment_plan(
            consumer,
            (
                corpus.declarations[genuine].node,
                corpus.declarations[injected].node,
                PlanNode(key=consumer, source_ref="duplicated.decl", kind=DIGEST_KIND),
            ),
            (
                PlanEdge(
                    consumer,
                    ("subjects", "paired_trial_bank"),
                    "required",
                    "authored",
                    producer=LogicalKey(SAMPLE_LAYER, "injected-extra"),
                ),
                PlanEdge(
                    consumer,
                    ("subjects", "paired_trial_bank"),
                    "required",
                    "authored",
                    producer=LogicalKey(SAMPLE_LAYER, "genuine-source"),
                ),
            ),
        )
    assert caught.value.consumer == consumer
    assert caught.value.role_path == ("subjects", "paired_trial_bank")
    assert "injected-extra" in str(caught.value)
    assert "genuine-source" in str(caught.value)


def test_a_duplicate_role_edge_refuses_at_plan_load(corpus: _Corpus) -> None:
    """A plan document carrying the duplicate is refused when it is read back."""
    source = _sample(corpus, "loaded-source")
    injected = _sample(corpus, "loaded-extra")
    target = _digest(corpus, "loaded-consumer", subject=source)
    document = expand_fulfillment_plan(target, expand=corpus.expand).document()
    # The injected node is a real node of the document, exactly as it would be
    # if somebody edited a plan to smuggle one into the closure: the duplicate
    # edge is what makes it reachable, and dropping duplicates silently is what
    # would keep reconciliation from ever seeing it.
    document["nodes"].insert(0, corpus.declarations[injected].node.record())
    original = next(edge for edge in document["edges"] if edge["role_path"] == ["subject"])
    document["edges"].insert(0, {**original, "producer": "sample:loaded-extra"})

    with pytest.raises(DuplicateInputEdgeError) as caught:
        fulfillment_plan_from_document(document)
    assert caught.value.role_path == ("subject",)


def test_a_duplicate_outside_the_closure_still_refuses(corpus: _Corpus) -> None:
    """Duplicates are refused over every declared edge, not only reached ones."""
    source = _sample(corpus, "unreached-source")
    unreached = LogicalKey(DIGEST_LAYER, "unreached")
    target = LogicalKey(SAMPLE_LAYER, "unreached-source")
    with pytest.raises(DuplicateInputEdgeError):
        build_fulfillment_plan(
            target,
            (
                corpus.declarations[source].node,
                PlanNode(key=unreached, source_ref="unreached.decl", kind=DIGEST_KIND),
            ),
            (
                PlanEdge(unreached, ("x",), "required", "authored", producer=target),
                PlanEdge(unreached, ("x",), "required", "authored", producer=target),
            ),
        )


def test_an_external_duplicate_names_both_receipts_it_binds(corpus: _Corpus) -> None:
    source = _sample(corpus, "external-consumer-source")
    consumer = LogicalKey(DIGEST_LAYER, "external-duplicated")
    with pytest.raises(DuplicateInputEdgeError) as caught:
        build_fulfillment_plan(
            consumer,
            (
                corpus.declarations[source].node,
                PlanNode(key=consumer, source_ref="external.decl", kind=DIGEST_KIND),
            ),
            (
                PlanEdge(
                    consumer,
                    ("prior",),
                    "required",
                    "authored",
                    external={"manifest_kind": "SampleManifest", "manifest_id": "first"},
                ),
                PlanEdge(
                    consumer,
                    ("prior",),
                    "required",
                    "authored",
                    external={"manifest_kind": "SampleManifest", "manifest_id": "second"},
                ),
            ),
        )
    assert "SampleManifest:first" in str(caught.value)
    assert "SampleManifest:second" in str(caught.value)
