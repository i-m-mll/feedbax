"""The run rule: which of a producer's runs a selection, a sweep exception or an axis condition chooses.

A rule is the text its author wrote, such as ``seed=0..3 lr>=0.01 lowest 1 loss per lr``. It is resolved against the
runs it reads each time a plan is made: terms on an axis at once, and a lowest or highest term once the runs have run.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from collections import Counter
from dataclasses import dataclass
import operator
from typing import Any, Protocol

from feedbax.contracts.expression_values import CallValue, Expression, LiteralValue, NameValue, Problem, TextSpan
from feedbax.contracts.expression_text import RULE_NAME, RuleBest, RuleChoice, RuleCondition, RuleInterval, RuleTerm, parse_rule


class SweepCoordinate(Protocol):
    """A run's value on an axis: its name, and the value itself when it has one."""

    @property
    def value_id(self) -> str: ...

    @property
    def value(self) -> Any: ...


class RuleRefusal(ValueError):
    """A refusal at a path relative to the ``where`` that holds the rule (``rule`` or ``rows``)."""

    def __init__(self, path: str, detail: str, span: TextSpan | None = None) -> None:
        self.path = path
        self.detail = detail
        self.span = span
        super().__init__(f"{path}: {detail}" if path else detail)


@dataclass(frozen=True)
class Chosen[Row]:
    """The rows a rule chooses, and its lowest or highest term, which waits on the rows' results."""

    rows: tuple[Row, ...]
    best: RuleBest | None = None


_FUNCTIONS: Mapping[str, Callable[..., Any]] = {
    "add": operator.add, "sub": operator.sub, "mul": operator.mul, "div": operator.truediv, "mod": operator.mod,
    "lt": operator.lt, "le": operator.le, "gt": operator.gt, "ge": operator.ge, "eq": operator.eq, "ne": operator.ne,
    "and": lambda left, right: left and right, "or": lambda left, right: left or right, "not": operator.not_,
    "neg": operator.neg, "pos": operator.pos, "abs": abs, "min": min, "max": max,
}


def value_text(coordinate: SweepCoordinate) -> str:
    """How a rule writes a run's value on an axis: the value itself when it is a number or a word, else its name."""
    value = _scalar(coordinate)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    name = coordinate.value_id if value is None else value
    return name if RULE_NAME.fullmatch(name) and name not in _RESERVED else '"' + name.replace("\\", "\\\\").replace('"', '\\"') + '"'


_RESERVED = frozenset({"true", "false", "and", "or", "not", "lowest", "highest", "per"})


def _scalar(coordinate: SweepCoordinate) -> bool | int | float | str | None:
    value = coordinate.value
    return value if isinstance(value, bool | int | float | str) else None


def _text(rule: str, span: TextSpan) -> str:
    return rule[span.start:span.end]


def _names(expression: Expression) -> set[str]:
    match expression:
        case NameValue(reference=name):
            return {name}
        case CallValue(args=args, kwargs=kwargs):
            return set().union(*map(_names, (*args, *kwargs.values())))
    return set()


def _matches(coordinate: SweepCoordinate, value: LiteralValue | NameValue) -> bool:
    if isinstance(value, NameValue):
        return value.reference in (coordinate.value_id, _scalar(coordinate))
    scalar = _scalar(coordinate)
    return scalar is value.value if type(value.value) is bool else (
        type(scalar) is not bool and isinstance(scalar, int | float) and scalar == value.value
    )


def _evaluate(rule: str, term: RuleCondition, values: Mapping[str, Any]) -> bool:
    def visit(node: Expression) -> Any:
        match node:
            case LiteralValue(value=value):
                return value
            case NameValue(reference=name):
                value = values[name]
                if value is None:
                    raise RuleRefusal(
                        "rule", f"{_text(rule, term.span)}: {name} has named values; choose them with {name}=…",
                        term.span,
                    )
                return value
            case CallValue(function=function, args=args, kwargs=kwargs) if function in _FUNCTIONS and not kwargs:
                return _FUNCTIONS[function](*map(visit, args))
        raise RuleRefusal("rule", f"{_text(rule, term.span)}: a rule compares values", term.span)

    try:
        result = visit(term.expression)
    except (TypeError, ZeroDivisionError) as error:
        raise RuleRefusal("rule", f"{_text(rule, term.span)}: {error}", term.span) from error
    if type(result) is not bool:
        raise RuleRefusal("rule", f"{_text(rule, term.span)} is not a condition", term.span)
    return result


def _term_names(term: RuleTerm) -> set[str]:
    match term:
        case RuleChoice(name=name) | RuleInterval(name=name):
            return {name}
        case RuleCondition(expression=expression):
            return _names(expression)
        case RuleBest(metric=metric):
            return {metric}


def parse(rule: str) -> tuple[RuleTerm, ...]:
    """The rule's terms, or a refusal at ``rule`` with the text it could not read."""
    terms = parse_rule(rule)
    if isinstance(terms, Problem):
        at = _text(rule, terms.span) if terms.span is not None else ""
        raise RuleRefusal("rule", f"{terms.reason} ({at!r})" if at else terms.reason, terms.span)
    return terms


def rule_names(rule: str) -> frozenset[str]:
    """The axis and metric names the rule's terms mention, or a refusal when it cannot be read."""
    return frozenset(name for term in parse(rule) for name in (*_term_names(term), *getattr(term, "per", ())))


def choose[Row](
    rule: str,
    rows: Sequence[Row],
    *,
    coordinates: Callable[[Row], Mapping[str, SweepCoordinate]],
    producer: str,
    ranked: bool = True,
) -> Chosen[Row]:
    """The rows ``rule`` chooses, in their order, and its lowest or highest term, which ranks them once they have run.

    Every name but a ranked metric is an axis the rows of ``producer`` carry. ``ranked=False`` refuses a lowest or
    highest term, as a sweep exception and an axis condition are resolved when the plan is made.
    """
    terms = parse(rule)
    if not terms:
        return Chosen(tuple(rows))
    by_row = tuple(map(coordinates, rows))
    axes: dict[str, list[SweepCoordinate]] = {}
    for row in by_row:
        for axis, coordinate in row.items():
            axes.setdefault(axis, []).append(coordinate)
    best: list[RuleBest] = []
    for term in terms:
        text = _text(rule, term.span)
        if isinstance(term, RuleBest):
            if not ranked:
                raise RuleRefusal("rule", f"{text}: a sweep is planned before its runs have results", term.span)
            names = set(term.per)
            best.append(term)
        else:
            names = _term_names(term)
        if unswept := sorted(names - axes.keys()):
            reason = f"{text}: {', '.join(unswept)} is not swept in {producer}"
            if isinstance(term, RuleCondition):
                reason += "; a rule cannot yet choose runs by a result's value"
            raise RuleRefusal("rule", reason, term.span)
        if isinstance(term, RuleChoice):
            for value in term.values:
                if not any(_matches(coordinate, value) for coordinate in axes[term.name]):
                    spelled = value.reference if isinstance(value, NameValue) else str(value.value).lower()
                    raise RuleRefusal("rule", f"{text}: {term.name} has no value {spelled}", term.span)
    if len(best) > 1:
        raise RuleRefusal("rule", f"{_text(rule, best[1].span)}: a rule ranks by one lowest or highest", best[1].span)
    picked = tuple(
        row for row, values in zip(rows, by_row, strict=True)
        if all(_holds(rule, term, values) for term in terms if not isinstance(term, RuleBest))
    )
    if not picked:
        raise RuleRefusal(
            "rule", f"{rule.strip()} chooses no run", TextSpan(start=terms[0].span.start, end=terms[-1].span.end),
        )
    ranking = best[0] if best else None
    if ranking is not None:
        text = _text(rule, ranking.span)
        if len(set(ranking.per)) != len(ranking.per):
            raise RuleRefusal("rule", f"{text}: per names an axis twice.", ranking.span)
        groups: Counter[tuple[str, ...]] = Counter()
        for row in picked:
            values = coordinates(row)
            if missing := next((axis for axis in ranking.per if axis not in values), None):
                raise RuleRefusal("rule", f"{text}: {missing} is missing on a chosen run.", ranking.span)
            groups[tuple(values[axis].value_id for axis in ranking.per)] += 1
        for group, count in groups.items():
            if count < ranking.count:
                target = f" in group {dict(zip(ranking.per, group, strict=True))!r}" if group else ""
                raise RuleRefusal("rule", f"{text}: asks for {ranking.count} of {count} runs{target}.", ranking.span)
    return Chosen(picked, ranking)


def _values(row: Mapping[str, SweepCoordinate]) -> dict[str, Any]:
    return {axis: _scalar(coordinate) for axis, coordinate in row.items()}


def _holds(rule: str, term: RuleTerm, row: Mapping[str, SweepCoordinate]) -> bool:
    match term:
        case RuleChoice(name=name, values=values):
            return name in row and any(_matches(row[name], value) for value in values)
        case RuleInterval(name=name, first=first, last=last):
            if name not in row:
                return False
            scalar = _scalar(row[name])
            if type(scalar) is bool or not isinstance(scalar, int | float):
                raise RuleRefusal("rule", f"{_text(rule, term.span)}: {name} has named values", term.span)
            return min(first.value, last.value) <= scalar <= max(first.value, last.value)
        case RuleCondition():
            return all(name in row for name in _names(term.expression)) and _evaluate(rule, term, _values(row))
    raise AssertionError(term)
