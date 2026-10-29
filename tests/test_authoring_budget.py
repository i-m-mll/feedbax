"""Canonical intent budgets stay independent of the source JSON spelling."""

from dataclasses import replace
import json

import pytest

from feedbax.contracts.authored_canonical import canonical_bytes
from feedbax.contracts.authoring_budget import (
    AUTHORING_BUDGET_SCHEMA_VERSION,
    AUTHORING_BUDGET_SCHEMA_VERSION_V1,
    AuthoringBudgets,
    load_authoring_budget_document,
)
from feedbax.contracts.experiment_envelope import ExperimentEnvelopeRejection
from feedbax.contracts.migrations import UnsupportedSpecVersion, default_spec_registry
from feedbax.envelope.authoring import (
    enforce_budget,
    enforce_assertion_budget,
    read_authored_document,
)
from feedbax.governance.project_init import default_budget_document


def _policy(**caps: int) -> AuthoringBudgets:
    document = default_budget_document("example")
    document["layers"]["evaluation"].update(caps)
    parsed = load_authoring_budget_document(json.dumps(document).encode(), field="budget.json")
    return AuthoringBudgets.from_document(
        parsed, field="budget.json", declared_layers=document["layers"]
    )


def _spellings(document: dict) -> list[bytes]:
    return [
        json.dumps(document, separators=(",", ":")).encode(),
        (json.dumps(document, indent=2, ensure_ascii=False) + "\n").encode(),
        json.dumps(dict(reversed(document.items())), indent=16, sort_keys=True).encode(),
    ]


def test_canonical_byte_boundary_ignores_whitespace_key_order_and_source_unicode() -> None:
    document = {"z": {"y": "é", "x": [1, 2]}, "a": "intent"}
    size = len(canonical_bytes(document))
    for raw in _spellings(document):
        assert (
            read_authored_document(
                raw, _policy(max_bytes=size), field="intent.json", layer_of=lambda _: "evaluation"
            )
            == document
        )
        with pytest.raises(ExperimentEnvelopeRejection, match=rf"{size} canonical_json_v1 bytes"):
            read_authored_document(
                raw,
                _policy(max_bytes=size - 1),
                field="intent.json",
                layer_of=lambda _: "evaluation",
            )


def test_budget_v1_requires_explicit_policy_review() -> None:
    document = default_budget_document("example")
    document["schema_version"] = AUTHORING_BUDGET_SCHEMA_VERSION_V1
    for section in document["layers"].values():
        section["max_lines"] = 80
    with pytest.raises(ExperimentEnvelopeRejection) as caught:
        load_authoring_budget_document(json.dumps(document).encode(), field="budget.json")
    message = str(caught.value)
    assert "remove max_lines from every layer" in message
    assert "review max_bytes under canonical_json_v1" in message
    assert AUTHORING_BUDGET_SCHEMA_VERSION in message


def test_v2_refuses_a_leftover_line_cap() -> None:
    with pytest.raises(ExperimentEnvelopeRejection, match="expected exactly the caps"):
        _policy(max_lines=80)


@pytest.mark.parametrize(
    ("cap", "value", "document", "reason"),
    [
        ("max_scalar_bytes", 3, {"a": "long"}, "scalar"),
        ("max_depth", 2, {"a": {"b": 1}}, "nesting depth"),
        ("max_items", 1, {"a": [1, 2]}, "items"),
        ("max_object_keys", 1, {"a": 1, "b": 2}, "keys"),
        ("max_ast_nodes", 2, {"a": 1}, "syntax nodes"),
    ],
)
def test_structural_bounds_still_refuse(cap: str, value: int, document: dict, reason: str) -> None:
    budget = replace(_policy().for_layer("evaluation"), **{cap: value})
    with pytest.raises(ExperimentEnvelopeRejection, match=reason):
        enforce_budget(document, budget, field="intent.json")


def test_registry_and_loader_agree_on_v1_rejection() -> None:
    document = default_budget_document("example")
    registry = default_spec_registry
    assert registry.migrate("AuthoringBudget", document).payload == document
    document["schema_version"] = AUTHORING_BUDGET_SCHEMA_VERSION_V1
    with pytest.raises(UnsupportedSpecVersion, match="remove max_lines"):
        registry.migrate("AuthoringBudget", document)


def test_evaluation_assertions_remain_bounded() -> None:
    budget = _policy().for_layer("evaluation")
    enforce_assertion_budget(4, budget, field="intent.json")
    with pytest.raises(ExperimentEnvelopeRejection, match="5 assertions"):
        enforce_assertion_budget(5, budget, field="intent.json")


def test_emitted_budget_is_independent_of_other_emissions() -> None:
    document = default_budget_document("example")
    document["layers"]["evaluation"]["max_bytes"] = 1
    assert default_budget_document("example")["layers"]["evaluation"]["max_bytes"] == 8192
