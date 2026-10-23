"""The closed set of Feedbax structural rules that certify an input inapplicable.

An input a declaration leaves unfilled is decided on exactly one of two bases
(:data:`~feedbax.workflow.plan.APPLICABILITY_BASES`): ``authored``,
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
* :func:`certify_structural_applicability` is the **certifier**. Anything
  reading a decision back — plan derivation, preflight, an audit of an emitted
  lock — asks this table whether the named rule is one this build owns *and
  whether it decides the consumer and role the decision was made on*. A rule
  this build does not own fails closed, and so does a rule it owns quoted
  somewhere the rule proves nothing: a rule id is a claim about a structure, not
  a token that excuses an input. :func:`require_structural_applicability_rule`
  is the ownership half of that check, useful on its own only when there is no
  decision context to evaluate against.

A rule is versioned in its own id, so tightening what a rule certifies is a new
rule rather than a silent redefinition of the old one, and a lock that quoted the
old one still says exactly what it meant.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import re

from feedbax.contracts.experiment_compile_lock import NotApplicableReference

#: A rule id ends in a version segment, matching the compile lock's own guard.
_VERSIONED_RULE_RE = re.compile(r".+\.v[0-9]+$")


class StructuralApplicabilityRuleMismatchError(ValueError):
    """A decision names a rule this build owns, applied where it does not hold.

    Owning a rule id is not the same as the rule applying. Every rule in this
    table proves something about a *particular* structure — which artifact layer
    consumes the input, and which role slot of it the structure declines to fill
    — so honoring the id wherever it appears would let a lock omit any input at
    all under the name of a rule about something else. A rule quoted outside the
    structure it decides is refused, and the refusal says which part did not
    hold.
    """

    def __init__(
        self,
        rule: "StructuralApplicabilityRule",
        detail: str,
        *,
        ref: str | None = None,
    ) -> None:
        self.rule = rule
        self.rule_id = rule.rule_id
        self.detail = detail
        self.ref = ref
        origin = f"{ref} " if ref else ""
        super().__init__(
            f"{origin}certifies an input as not applicable under the structural rule "
            f"{rule.rule_id!r}, which does not apply here: {detail}. That rule proves "
            f"{rule.summary} A rule is honored where it decides and refused everywhere "
            "else, so an omission it does not cover is refused rather than executed around."
        )


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

    A rule is not only an id and a sentence: it is a *predicate*, because what it
    proves is a fact about one structure. :attr:`consumer_layers` and the role
    path shape state which structure that is, so the rule can be evaluated
    against the consumer and role a decision was actually made on rather than
    honored on the strength of being named.

    Attributes:
        rule_id: The rule's durable versioned identity, quoted on every decision
            it certifies.
        reason: The one sentence every decision under this rule states. It is
            fixed by the rule rather than written per call site, so two compiles
            certifying under one rule say the same thing.
        summary: What the rule proves, for a reader of the rule set itself.
        consumer_layers: The Feedbax artifact layers whose nodes this rule can
            decide anything about. A consumer in any other layer is a different
            structure, and this rule proves nothing about it.
        role_path_prefix: The leading role-path segments the decided slot lives
            under, such as ``("inputs",)`` for a figure input role.
        role_path_length: How many segments the decided role path has in total.
            A rule about one role slot does not reach a nested path beneath it.
    """

    rule_id: str
    reason: str
    summary: str
    consumer_layers: tuple[str, ...]
    role_path_prefix: tuple[str, ...]
    role_path_length: int

    def __post_init__(self) -> None:
        if not _VERSIONED_RULE_RE.fullmatch(self.rule_id):
            raise ValueError(
                f"structural applicability rule id {self.rule_id!r} must end with a version "
                "segment such as '.v1'; a rule is versioned so tightening it is a new rule"
            )
        for name in ("reason", "summary"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"structural applicability rule {self.rule_id!r} states a {name}")
        if not self.consumer_layers:
            raise ValueError(
                f"structural applicability rule {self.rule_id!r} names the artifact layers it "
                "decides; a rule that decides every layer decides nothing"
            )
        if self.role_path_length < len(self.role_path_prefix) + 1:
            raise ValueError(
                f"structural applicability rule {self.rule_id!r} must decide a role path longer "
                "than the prefix it lives under, or it names no slot at all"
            )

    def role_path_mismatch(self, role_path: Sequence[str]) -> str | None:
        """Return why this rule does not decide the slot *role_path* addresses."""
        parts = tuple(role_path)
        if parts[: len(self.role_path_prefix)] != self.role_path_prefix:
            return (
                f"it decides a role path under {list(self.role_path_prefix)}, and this input "
                f"is {list(parts)}"
            )
        if len(parts) != self.role_path_length:
            return (
                f"it decides a {self.role_path_length}-segment role path, and this input is "
                f"{list(parts)}"
            )
        return None

    def consumer_mismatch(self, consumer_layer: str) -> str | None:
        """Return why this rule decides nothing about a *consumer_layer* node."""
        if consumer_layer in self.consumer_layers:
            return None
        return (
            f"it decides {list(self.consumer_layers)} nodes, and this consumer is a "
            f"{consumer_layer!r} node"
        )

    def mismatch(self, *, consumer_layer: str, role_path: Sequence[str]) -> str | None:
        """Return why this rule does not decide *role_path* on *consumer_layer*.

        ``None`` means the rule structurally applies. Both halves are reported
        together rather than the first failing one, because a caller told only
        that the layer is wrong would fix the layer and be refused again on the
        path.
        """
        failures = [
            detail
            for detail in (
                self.consumer_mismatch(consumer_layer),
                self.role_path_mismatch(role_path),
            )
            if detail is not None
        ]
        return "; ".join(failures) if failures else None

    def reason_mismatch(self, reason: object) -> str | None:
        """Return why *reason* is not the sentence this rule states, if it is not.

        The rule owns its reason, so a decision quoting the rule and stating
        something else is two claims about one omission. Only one of them can be
        this build's, and it is not the authored one.
        """
        if reason == self.reason:
            return None
        return (
            f"it states one fixed reason, and this decision states {reason!r} instead of "
            f"{self.reason!r}"
        )


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
    consumer_layers=("figure",),
    role_path_prefix=("inputs",),
    role_path_length=2,
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


def certify_structural_applicability(
    rule_id: object,
    *,
    consumer_layer: str,
    role_path: Sequence[str],
    reason: object,
    ref: str | None = None,
) -> StructuralApplicabilityRule:
    """Return the rule one decision names, proved to decide that decision.

    This is the whole certifier: it resolves the named rule from the closed
    table and then *evaluates* it against the consumer and role path the
    decision was actually made on, plus the reason the decision states. A rule
    id alone certifies nothing — the sole rule this build owns is about one
    figure input slot, and honoring its name on an evaluation, report, or
    analysis prerequisite would let a malformed lock omit that prerequisite
    under the authority of a rule about something else.

    Raises:
        UnknownStructuralApplicabilityRuleError: The id is not in the closed
            rule table.
        StructuralApplicabilityRuleMismatchError: The rule is one this build
            owns, but it does not decide this consumer, this role path, or the
            reason the decision states.
    """
    rule = require_structural_applicability_rule(rule_id, ref=ref)
    detail = rule.mismatch(consumer_layer=consumer_layer, role_path=role_path)
    if detail is None:
        detail = rule.reason_mismatch(reason)
    if detail is not None:
        raise StructuralApplicabilityRuleMismatchError(rule, detail, ref=ref)
    return rule


def certify_not_applicable(
    role_path: str, rule: StructuralApplicabilityRule
) -> NotApplicableReference:
    """Return the lock reference one structural rule certifies for *role_path*.

    This is the producer half: a compile states a structural omission by naming a
    rule, and the rule supplies both the id the lock quotes and the reason it
    states. A caller therefore cannot emit a reason that disagrees with the rule
    it claims to be applying, and cannot mint a rule id this build does not own.

    The rule's role-path shape is evaluated here too, so a compile cannot emit a
    decision the certifier would later refuse. The consumer layer is not checked
    at this end: a compile states the omission while lowering one layer's own
    envelope, and the layer a decision reaches a consumer as is a fact of the
    derived plan rather than of this call.

    Raises:
        UnknownStructuralApplicabilityRuleError: The rule is not one this build
            owns.
        StructuralApplicabilityRuleMismatchError: The rule does not decide a
            role path of this shape.
    """
    require_structural_applicability_rule(rule.rule_id)
    detail = rule.role_path_mismatch(tuple(part for part in role_path.split(".") if part))
    if detail is not None:
        raise StructuralApplicabilityRuleMismatchError(rule, detail)
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
    "StructuralApplicabilityRuleMismatchError",
    "UnknownStructuralApplicabilityRuleError",
    "certify_not_applicable",
    "certify_structural_applicability",
    "require_structural_applicability_rule",
]
