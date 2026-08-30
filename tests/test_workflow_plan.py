"""Contract tests for the sole finite workflow representation."""

from __future__ import annotations

from dataclasses import replace

import pytest

from feedbax.workflow.plan import (
    APPLICABILITY_STATUSES,
    LEGACY_FULFILLMENT_PLAN_SCHEMA_ID,
    WORKFLOW_PLAN_SCHEMA_ID,
    WORKFLOW_PLAN_SCHEMA_VERSION_V1,
    ConflictingNodeDeclarationError,
    DuplicateInputEdgeError,
    EdgeDeclaration,
    GuardPredicate,
    LogicalKey,
    NodeDeclaration,
    Operation,
    PlanCycleError,
    PlanEdge,
    PlanGuard,
    PlanNode,
    UnresolvedGuardOutcomeError,
    UnsupportedWorkflowPlanVersionError,
    WorkflowPlanIdentityError,
    WorkflowTypeMismatchError,
    build_workflow_plan,
    expand_workflow_plan,
    read_workflow_plan_document,
    workflow_plan_from_document,
)


PRODUCT = "toy.product"
DECISION = "toy.decision"


def _operation(
    type_id: str,
    *,
    inputs: dict[str, str] | None = None,
    outputs: dict[str, str] | None = None,
) -> Operation:
    return Operation(
        type_id=type_id,
        parameters={"semantic": type_id},
        input_types=inputs or {},
        output_types=outputs or {"primary": PRODUCT},
    )


def _node(
    layer: str,
    name: str,
    *,
    inputs: dict[str, str] | None = None,
    outputs: dict[str, str] | None = None,
    operation_type: str = "toy.operation",
) -> PlanNode:
    return PlanNode(
        key=LogicalKey(layer, name),
        source_ref=f"{name}.json",
        operation=_operation(operation_type, inputs=inputs, outputs=outputs),
        content_hash=name * 2,
        execution_identity=f"execution-{name}",
    )


def _producer_edge(
    consumer: PlanNode,
    role: str,
    producer: PlanNode,
    *,
    status: str = "required",
    guard: PlanGuard | None = None,
) -> PlanEdge:
    return PlanEdge(
        consumer=consumer.key,
        role_path=tuple(role.split(".")),
        status=status,
        basis="authored",
        input_type=producer.operation.output_types["primary"],
        producer=producer.key,
        producer_output="primary",
        guard=guard,
    )


def test_schema_family_is_workflow_and_rejects_the_predecessor_explicitly() -> None:
    node = _node("analysis", "target")
    document = build_workflow_plan(node.key, (node,), ()).document()
    assert document["schema_id"] == WORKFLOW_PLAN_SCHEMA_ID
    assert document["schema_version"] == WORKFLOW_PLAN_SCHEMA_VERSION_V1

    document["schema_id"] = LEGACY_FULFILLMENT_PLAN_SCHEMA_ID
    with pytest.raises(UnsupportedWorkflowPlanVersionError, match="explicitly rejected"):
        read_workflow_plan_document(document)


def test_unknown_workflow_version_fails_closed() -> None:
    node = _node("analysis", "target")
    document = build_workflow_plan(node.key, (node,), ()).document()
    document["schema_version"] = "feedbax.workflow.plan.v9"
    with pytest.raises(UnsupportedWorkflowPlanVersionError, match="no migration"):
        read_workflow_plan_document(document)


def test_round_trip_preserves_identity_and_origin_does_not_change_it() -> None:
    node = _node("analysis", "target")
    first = build_workflow_plan(node.key, (node,), (), origin={"compiler": "one"})
    second = build_workflow_plan(node.key, (node,), (), origin={"compiler": "two"})
    assert first.identity == second.identity
    assert workflow_plan_from_document(first.document()).document() == first.document()


def test_physical_external_location_does_not_change_semantic_identity() -> None:
    target = _node("analysis", "target", inputs={"subject": PRODUCT})
    edge = PlanEdge(
        consumer=target.key,
        role_path=("subject",),
        status="required",
        basis="authored",
        input_type=PRODUCT,
        external={
            "manifest_id": "exact-receipt",
            "manifest_sha256": "a" * 64,
            "execution_uri": "file:///first/provider/location",
        },
        external_type=PRODUCT,
    )
    first = build_workflow_plan(target.key, (target,), (edge,))
    second = build_workflow_plan(
        target.key,
        (target,),
        (replace(edge, external={**edge.external, "execution_uri": "s3://moved"}),),
    )
    assert first.identity == second.identity
    assert first.document()["edges"] != second.document()["edges"]


def test_semantic_authoring_refuses_realization_state() -> None:
    with pytest.raises(ValueError, match="provider, custody, attempt"):
        Operation(type_id="toy.operation", parameters={"provider_id": "gpu-1"})
    with pytest.raises(ValueError, match="provider, custody, attempt"):
        replace(_node("analysis", "target"), metadata={"custody_root": "/tmp"})


def test_document_tampering_is_diagnosed_by_identity() -> None:
    node = _node("analysis", "target")
    document = build_workflow_plan(node.key, (node,), ()).document()
    document["nodes"][0]["operation"]["parameters"]["semantic"] = "changed"
    with pytest.raises(WorkflowPlanIdentityError, match="canonical semantic content"):
        workflow_plan_from_document(document)


def test_dependency_closure_is_canonical_and_deduplicates_a_diamond() -> None:
    source = _node("campaign", "source")
    left = _node("analysis", "left", inputs={"source": PRODUCT})
    right = _node("evaluation", "right", inputs={"source": PRODUCT})
    target = _node("report", "target", inputs={"left": PRODUCT, "right": PRODUCT})
    edges = (
        _producer_edge(left, "source", source),
        _producer_edge(right, "source", source),
        _producer_edge(target, "left", left),
        _producer_edge(target, "right", right),
    )
    plan = build_workflow_plan(target.key, (target, right, source, left), edges)
    assert [node.key.text for node in plan.nodes] == [
        "campaign:source",
        "analysis:left",
        "evaluation:right",
        "report:target",
    ]


def test_unreachable_declarations_are_not_part_of_the_plan() -> None:
    source = _node("analysis", "source")
    target = _node("report", "target", inputs={"source": PRODUCT})
    stray = _node("analysis", "stray")
    plan = build_workflow_plan(
        target.key,
        (stray, target, source),
        (_producer_edge(target, "source", source),),
    )
    assert [node.key for node in plan.nodes] == [source.key, target.key]


def test_cycles_and_duplicate_roles_are_structured_refusals() -> None:
    left = _node("analysis", "left", inputs={"right": PRODUCT})
    right = _node("analysis", "right", inputs={"left": PRODUCT})
    with pytest.raises(PlanCycleError):
        build_workflow_plan(
            left.key,
            (left, right),
            (_producer_edge(left, "right", right), _producer_edge(right, "left", left)),
        )

    target = _node("report", "target", inputs={"source": PRODUCT})
    edge = _producer_edge(target, "source", left)
    with pytest.raises(DuplicateInputEdgeError):
        build_workflow_plan(target.key, (target, left), (edge, edge))


def test_conflicting_node_declarations_do_not_first_win() -> None:
    first = _node("analysis", "same")
    second = replace(first, operation=_operation("different.operation"))
    with pytest.raises(ConflictingNodeDeclarationError, match="operation"):
        build_workflow_plan(first.key, (first, second), ())


def test_producer_and_external_bindings_are_type_checked() -> None:
    producer = _node("analysis", "producer", outputs={"primary": PRODUCT})
    consumer = _node("report", "consumer", inputs={"subject": DECISION})
    edge = PlanEdge(
        consumer=consumer.key,
        role_path=("subject",),
        status="required",
        basis="authored",
        input_type=DECISION,
        producer=producer.key,
        producer_output="primary",
    )
    with pytest.raises(WorkflowTypeMismatchError, match="produces"):
        build_workflow_plan(consumer.key, (producer, consumer), (edge,))

    external = replace(
        edge,
        producer=None,
        producer_output=None,
        external={"artifact_id": "exact"},
        external_type=PRODUCT,
    )
    with pytest.raises(WorkflowTypeMismatchError, match="exact external"):
        build_workflow_plan(consumer.key, (consumer,), (external,))


def test_certified_omission_is_preserved_and_binds_nothing() -> None:
    target = _node("analysis", "target", inputs={"optional": PRODUCT})
    omission = PlanEdge(
        consumer=target.key,
        role_path=("optional",),
        status="not_applicable",
        basis="compiler_rule",
        input_type=PRODUCT,
        reason="shape excludes it",
        rule="toy.rule.v1",
    )
    plan = build_workflow_plan(target.key, (target,), (omission,))
    assert plan.certified_omissions() == (omission,)


def test_guarded_binding_is_finite_typed_and_ordered_after_its_outcome() -> None:
    decision = _node("evaluation", "decision", outputs={"decision": DECISION, "primary": PRODUCT})
    source = _node("analysis", "source")
    target = _node("report", "target", inputs={"subject": PRODUCT})
    guard = PlanGuard(
        outcome=decision.key,
        output_role="decision",
        output_type=DECISION,
        predicate=GuardPredicate("equals", "accepted"),
    )
    edge = _producer_edge(target, "subject", source, status="guarded", guard=guard)
    plan = build_workflow_plan(target.key, (target, source, decision), (edge,))
    assert [node.key for node in plan.nodes] == [source.key, decision.key, target.key]
    assert plan.active_edges({(decision.key, "decision"): "accepted"}) == (edge,)
    assert plan.active_edges({(decision.key, "decision"): "rejected"}) == ()


def test_guard_refuses_missing_outcomes_and_type_disagreement() -> None:
    decision = _node("evaluation", "decision", outputs={"decision": DECISION})
    source = _node("analysis", "source")
    target = _node("report", "target", inputs={"subject": PRODUCT})
    bad_guard = PlanGuard(
        outcome=decision.key,
        output_role="decision",
        output_type=PRODUCT,
        predicate=GuardPredicate("in", ("yes", "no")),
    )
    edge = _producer_edge(target, "subject", source, status="guarded", guard=bad_guard)
    with pytest.raises(WorkflowTypeMismatchError, match="guard"):
        build_workflow_plan(target.key, (target, source, decision), (edge,))

    guard = replace(bad_guard, output_type=DECISION)
    plan = build_workflow_plan(
        target.key,
        (target, source, decision),
        (replace(edge, guard=guard),),
    )
    with pytest.raises(UnresolvedGuardOutcomeError):
        plan.active_edges({})


def test_expander_preserves_typed_declarations() -> None:
    source = _node("analysis", "source")
    target = _node("report", "target", inputs={"subject": PRODUCT})
    corpus = {
        "source": NodeDeclaration(source),
        "target": NodeDeclaration(
            target,
            (
                EdgeDeclaration(
                    role_path=("subject",),
                    input_type=PRODUCT,
                    producer_ref="source",
                    producer_output="primary",
                ),
            ),
        ),
    }
    plan = expand_workflow_plan("target", expand=corpus.__getitem__)
    assert [node.key for node in plan.nodes] == [source.key, target.key]


def test_guard_and_operation_vocabularies_are_closed() -> None:
    assert APPLICABILITY_STATUSES == ("required", "guarded", "not_applicable")
    with pytest.raises(ValueError):
        GuardPredicate("poll_until", True)
    with pytest.raises(ValueError):
        Operation(type_id="x", determinism="ambient")
