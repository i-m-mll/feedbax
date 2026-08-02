"""The closed structural rule table: one producer, one certifier, no free strings.

A ``compiler_rule`` omission is honored on the strength of the rule it quotes.
Before this table there was nothing that could produce such a decision and
nothing that could tell a rule this build owns from a string somebody wrote, so
the two claims under test are:

* a compile *produces* a structural omission by naming a rule, and the rule — not
  the call site — supplies both the id the lock quotes and the reason it states;
* fulfillment *certifies* one before honoring it, and a closure quoting a rule
  this build does not own is refused at preflight rather than executed around.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from feedbax.analysis.fulfillment_derivation import (
    derive_fulfillment_plan,
    read_compiled_outputs,
)
from feedbax.analysis.fulfillment_driver import (
    UncertifiedApplicabilityError,
    preflight,
    require_certified_applicability,
)
from feedbax.contracts.applicability_rules import (
    PER_ROW_FIGURE_INPUT_RULE,
    STRUCTURAL_APPLICABILITY_RULES,
    StructuralApplicabilityRule,
    UnknownStructuralApplicabilityRuleError,
    certify_not_applicable,
    require_structural_applicability_rule,
)
from feedbax.contracts.experiment_compile_lock import NotApplicableReference

from tests.fake_project_experiment.products import QuillonOutputs


@pytest.fixture
def outputs(tmp_path: Path) -> QuillonOutputs:
    return QuillonOutputs(tmp_path / "repo")


# -- the table itself -------------------------------------------------------


def test_every_rule_is_keyed_by_its_own_versioned_id() -> None:
    for rule_id, rule in STRUCTURAL_APPLICABILITY_RULES.items():
        assert rule_id == rule.rule_id
        assert rule_id.rsplit(".", 1)[-1].startswith("v")


def test_an_unversioned_rule_id_is_not_a_rule() -> None:
    """Tightening a rule is a new rule, which only a version segment allows."""
    with pytest.raises(ValueError, match="must end with a version segment"):
        StructuralApplicabilityRule(
            rule_id="feedbax.rule.unbound_role", reason="because", summary="because"
        )


def test_a_rule_states_a_reason_and_a_summary() -> None:
    with pytest.raises(ValueError, match="states a reason"):
        StructuralApplicabilityRule(
            rule_id="feedbax.rule.example.v1", reason="  ", summary="what it proves"
        )


# -- the producer -----------------------------------------------------------


def test_certifying_a_role_quotes_the_rule_that_decided_it() -> None:
    reference = certify_not_applicable("inputs.observed", PER_ROW_FIGURE_INPUT_RULE)

    assert isinstance(reference, NotApplicableReference)
    assert reference.role_path == "inputs.observed"
    assert reference.basis == "compiler_rule"
    assert reference.rule_id == PER_ROW_FIGURE_INPUT_RULE.rule_id
    assert reference.reason == PER_ROW_FIGURE_INPUT_RULE.reason


def test_the_producer_cannot_mint_a_rule_the_build_does_not_own() -> None:
    """A rule object is not enough: the table is what makes a rule a rule."""
    foreign = StructuralApplicabilityRule(
        rule_id="feedbax.rule.invented_here.v1",
        reason="a call site decided this",
        summary="nothing owns this",
    )

    with pytest.raises(UnknownStructuralApplicabilityRuleError, match="does not own"):
        certify_not_applicable("inputs.observed", foreign)


def test_the_compiler_emits_its_per_row_omission_through_the_table(tmp_path: Path) -> None:
    from feedbax.envelope import kernel_for

    from tests.fake_project_experiment import (
        PROJECT_DECLARATION,
        envelope_path,
        write_repo,
    )

    write_repo(tmp_path)
    outcome = kernel_for(PROJECT_DECLARATION).compile_envelope_file(
        envelope_path(tmp_path, "widened-plot"), repo_root=tmp_path
    )

    reference = next(
        item for item in outcome.compile_lock["references"] if item["kind"] == "not_applicable"
    )
    rule = require_structural_applicability_rule(reference["rule_id"])
    assert rule is PER_ROW_FIGURE_INPUT_RULE
    assert reference["reason"] == rule.reason


# -- the certifier ----------------------------------------------------------


@pytest.mark.parametrize("rule_id", ["feedbax.rule.unbound_role.v1", "", None, 7])
def test_an_unowned_rule_id_certifies_nothing(rule_id: object) -> None:
    with pytest.raises(UnknownStructuralApplicabilityRuleError) as caught:
        require_structural_applicability_rule(rule_id, ref="widened-plot")

    assert caught.value.rule_id == rule_id
    assert "widened-plot" in str(caught.value)
    assert PER_ROW_FIGURE_INPUT_RULE.rule_id in str(caught.value)


def test_preflight_refuses_a_closure_that_omits_under_an_unowned_rule(
    outputs: QuillonOutputs,
) -> None:
    outputs.plate(
        "foreign-plate",
        references=[
            NotApplicableReference(
                role_path="inputs.observed",
                basis="compiler_rule",
                reason="a rule nobody owns decided this",
                rule_id="feedbax.rule.unbound_role.v1",
            )
        ],
    )
    index = read_compiled_outputs(outputs.output_directory)
    plan = derive_fulfillment_plan(index, target="foreign-plate")

    with pytest.raises(UncertifiedApplicabilityError) as caught:
        preflight(plan, index)

    record = caught.value.record()
    assert record["target"] == "figure:foreign-plate"
    assert [item["role_path"] for item in record["uncertified"]] == [["inputs", "observed"]]
    assert record["uncertified"][0]["rule"] == "feedbax.rule.unbound_role.v1"


def test_preflight_admits_a_closure_that_omits_under_an_owned_rule(
    outputs: QuillonOutputs,
) -> None:
    outputs.plate(
        "owned-plate",
        references=[certify_not_applicable("inputs.observed", PER_ROW_FIGURE_INPUT_RULE)],
    )
    index = read_compiled_outputs(outputs.output_directory)
    plan = derive_fulfillment_plan(index, target="owned-plate")

    require_certified_applicability(plan)
    closure = preflight(plan, index)

    assert closure.order == ("figure:owned-plate",)


def test_every_uncertified_decision_is_named_at_once(outputs: QuillonOutputs) -> None:
    """One refusal states the whole problem, in the plan's canonical edge order."""
    outputs.plate(
        "two-foreign",
        references=[
            NotApplicableReference(
                role_path="inputs.observed",
                basis="compiler_rule",
                reason="a rule nobody owns decided this",
                rule_id="feedbax.rule.unbound_role.v1",
            ),
            NotApplicableReference(
                role_path="inputs.expected",
                basis="compiler_rule",
                reason="a second rule nobody owns decided this",
                rule_id="feedbax.rule.other_missing.v2",
            ),
        ],
    )
    index = read_compiled_outputs(outputs.output_directory)
    plan = derive_fulfillment_plan(index, target="two-foreign")

    with pytest.raises(UncertifiedApplicabilityError) as caught:
        preflight(plan, index)

    assert [item["role_path"] for item in caught.value.record()["uncertified"]] == [
        ["inputs", "expected"],
        ["inputs", "observed"],
    ]
    assert [item["rule"] for item in caught.value.record()["uncertified"]] == [
        "feedbax.rule.other_missing.v2",
        "feedbax.rule.unbound_role.v1",
    ]


def test_an_authored_omission_needs_no_rule(outputs: QuillonOutputs) -> None:
    """The certifier is about ``compiler_rule`` only; a human is their own authority."""
    outputs.plate(
        "authored-plate",
        references=[
            NotApplicableReference(
                role_path="inputs.observed",
                basis="authored",
                reason="the widened survey has no comparison arm to plot",
            )
        ],
    )
    index = read_compiled_outputs(outputs.output_directory)
    plan = derive_fulfillment_plan(index, target="authored-plate")

    require_certified_applicability(plan)

    assert plan.certified_omissions() == ()
