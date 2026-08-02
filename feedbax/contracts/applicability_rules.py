"""The closed set of Feedbax structural rules that certify an input inapplicable.

An input a declaration leaves unfilled is decided on exactly one of two bases
(:data:`~feedbax.analysis.fulfillment_plan.APPLICABILITY_BASES`): ``authored``,
where a human stated it in the envelope, and ``compiler_rule``, where a closed
versioned Feedbax structural rule proves that the target provides no such input.

The ``authored`` basis has always had a producer — the envelope says it. The
``compiler_rule`` basis had none: a rule id was a string a caller could write,
with nothing that owned it, nothing that could produce one, and nothing that
could tell a rule this build owns from a rule it has never heard of. This module
is that missing half, and it is deliberately two functions over one closed table:

* :func:`certify_not_applicable` is the **producer**. A compile states a
  structural omission by naming a rule from this table, never by assembling a
  reference and a rule string separately, so the reason and the rule id can never
  drift apart and no compile can mint a rule id of its own.
* :func:`require_structural_applicability_rule` is the **certifier**. Anything
  reading a decision back — plan derivation, preflight, an audit of an emitted
  lock — asks this table whether the named rule is one this build owns, and a
  rule it does not own fails closed rather than being honored on the strength of
  looking like a rule id.

A rule is versioned in its own id, so tightening what a rule certifies is a new
rule rather than a silent redefinition of the old one, and a lock that quoted the
old one still says exactly what it meant.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import re

from feedbax.contracts.experiment_compile_lock import NotApplicableReference

#: A rule id ends in a version segment, matching the compile lock's own guard.
_VERSIONED_RULE_RE = re.compile(r".+\.v[0-9]+$")


class UnknownStructuralApplicabilityRuleError(ValueError):
    """A decision names a structural rule this build does not own.

    A rule id is not a free-form label: it names a closed, versioned rule whose
    meaning this build must be able to state. An unrecognized id is refused
    rather than honored, because honoring it would mean accepting an omission
    certified by nothing.
    """

    def __init__(self, rule_id: object, *, ref: str | None = None) -> None:
        self.rule_id = rule_id
        self.ref = ref
        origin = f"{ref} " if ref else ""
        super().__init__(
            f"{origin}certifies an input as not applicable under the structural rule "
            f"{rule_id!r}, which this build does not own; the closed rule set is "
            f"{sorted(STRUCTURAL_APPLICABILITY_RULES)}. A compiler_rule decision quotes a "
            "Feedbax rule, so a rule this build cannot state is refused rather than honored."
        )


@dataclass(frozen=True)
class StructuralApplicabilityRule:
    """One closed, versioned reason a Feedbax structure fills no such input.

    Attributes:
        rule_id: The rule's durable versioned identity, quoted on every decision
            it certifies.
        reason: The one sentence every decision under this rule states. It is
            fixed by the rule rather than written per call site, so two compiles
            certifying under one rule say the same thing.
        summary: What the rule proves, for a reader of the rule set itself.
    """

    rule_id: str
    reason: str
    summary: str

    def __post_init__(self) -> None:
        if not _VERSIONED_RULE_RE.fullmatch(self.rule_id):
            raise ValueError(
                f"structural applicability rule id {self.rule_id!r} must end with a version "
                "segment such as '.v1'; a rule is versioned so tightening it is a new rule"
            )
        for name in ("reason", "summary"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"structural applicability rule {self.rule_id!r} states a {name}")


#: A ``per_row`` figure input role has no single runtime locator: row expansion
#: fills it once per expanded row from the row index's own custody. The role is
#: not unbound — the per-row profile and its closed artifact contract are recorded
#: in the lock's ``figure_row_expansion`` identity contribution — so what is not
#: applicable is the single-locator reference slot, and stating that is different
#: from leaving the role silent.
PER_ROW_FIGURE_INPUT_RULE = StructuralApplicabilityRule(
    rule_id="feedbax.experiment_envelope.per_row_figure_input.v1",
    reason=(
        "row expansion fills this role once per expanded row from the row index's custody, "
        "so no single locator addresses it; the per-row profile and its artifact contract "
        "are recorded in the figure_row_expansion identity contribution"
    ),
    summary=(
        "The single-locator reference slot of a per-row figure input role, which row "
        "expansion fills once per expanded row from the row index's custody bindings."
    ),
)


#: Every structural rule this build owns, keyed by its versioned id. Both sides
#: are Feedbax's, so the table is Feedbax code: a project contributes no row, and
#: a decision naming an id outside it is refused rather than honored.
STRUCTURAL_APPLICABILITY_RULES: Mapping[str, StructuralApplicabilityRule] = {
    rule.rule_id: rule for rule in (PER_ROW_FIGURE_INPUT_RULE,)
}


def require_structural_applicability_rule(
    rule_id: object, *, ref: str | None = None
) -> StructuralApplicabilityRule:
    """Return the closed rule one decision names, or refuse the decision.

    This is the certifier half. A reader of an already-emitted decision — a plan
    derivation, a preflight, an audit — proves that the rule the decision quotes
    is one this build owns before treating the input as legitimately omitted.

    Raises:
        UnknownStructuralApplicabilityRuleError: The id is absent from the closed
            rule table, including when it is not a string at all.
    """
    if not isinstance(rule_id, str) or rule_id not in STRUCTURAL_APPLICABILITY_RULES:
        raise UnknownStructuralApplicabilityRuleError(rule_id, ref=ref)
    return STRUCTURAL_APPLICABILITY_RULES[rule_id]


def certify_not_applicable(
    role_path: str, rule: StructuralApplicabilityRule
) -> NotApplicableReference:
    """Return the lock reference one structural rule certifies for *role_path*.

    This is the producer half: a compile states a structural omission by naming a
    rule, and the rule supplies both the id the lock quotes and the reason it
    states. A caller therefore cannot emit a reason that disagrees with the rule
    it claims to be applying, and cannot mint a rule id this build does not own.
    """
    require_structural_applicability_rule(rule.rule_id)
    return NotApplicableReference(
        role_path=role_path,
        basis="compiler_rule",
        reason=rule.reason,
        rule_id=rule.rule_id,
    )


__all__ = [
    "PER_ROW_FIGURE_INPUT_RULE",
    "STRUCTURAL_APPLICABILITY_RULES",
    "StructuralApplicabilityRule",
    "UnknownStructuralApplicabilityRuleError",
    "certify_not_applicable",
    "require_structural_applicability_rule",
]
