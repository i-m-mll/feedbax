"""The run rule's written form and the rows it chooses."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from feedbax.contracts.expression_values import (
    CallValue, Level, LiteralValue, NameValue, Problem, Site, ValueBase, ValueType,
)
from feedbax.contracts.project import Project
from feedbax.planner.project_plan import compile_project
from feedbax.contracts.request_refs import BestRank
from feedbax.contracts.run_rule import choose
from feedbax.contracts.expression_text import RuleBest, RuleChoice, RuleCondition, RuleInterval, parse, parse_rule
from tests.test_saved_update_series import definition


def test_rule_forms_parse_into_separate_terms() -> None:
    """A rule's terms stay separate (catches the grammar joining ``1 lr`` or reading ``0..9`` as numbers)."""
    terms = parse_rule("seed % 2 == 1 lr>=0.01 seed=0,2,4 seed=0..9 and noise=low lowest 3 loss per lr, seed")
    assert not isinstance(terms, Problem)
    parity, rate, listed, ranged, noise, best = terms
    assert isinstance(parity, RuleCondition) and isinstance(rate, RuleCondition)
    assert parity.expression == CallValue(function="eq", args=(
        CallValue(function="mod", args=(NameValue(reference="seed"), LiteralValue(value=2))), LiteralValue(value=1),
    ))
    assert rate.expression == CallValue(function="ge", args=(NameValue(reference="lr"), LiteralValue(value=0.01)))
    assert listed == RuleChoice("seed", tuple(LiteralValue(value=v) for v in (0, 2, 4)), listed.span)
    assert ranged == RuleInterval("seed", LiteralValue(value=0), LiteralValue(value=9), ranged.span)
    assert noise == RuleChoice("noise", (NameValue(reference="low"),), noise.span)
    assert best == RuleBest("min", 3, "loss", ("lr", "seed"), best.span)
    assert parse_rule("") == ()


def test_units_still_read_in_fields() -> None:
    """Rules take no units, but a field's ``3 ms`` keeps its unit (catches the two grammars sharing one lexer)."""
    site = Site(expected=ValueType(base=ValueBase.REAL), finest=Level.RUN)
    parsed = parse("3 ms", site)
    assert not isinstance(parsed, tuple)
    assert parsed.value == LiteralValue(value=3, unit="ms")


@dataclass(frozen=True)
class _Coordinate:
    value_id: str
    value: object


_RATES = (0.001, 0.003, 0.01, 0.03)


def _grid(seeds: int = 4) -> list[dict[str, _Coordinate]]:
    return [
        {"seed": _Coordinate(f"value-{seed}", seed), "lr": _Coordinate(f"rate-{index}", rate)}
        for seed in range(seeds) for index, rate in enumerate(_RATES)
    ]


def test_axis_terms_intersect_by_value_not_by_value_id() -> None:
    """``seed=0..2 lr>=0.01`` over a 4 x 4 sweep chooses 3 x 2 runs (catches terms joined by or, or matched by ID)."""
    chosen = choose("seed=0..2 lr>=0.01", _grid(), coordinates=lambda row: row, producer="train")
    assert sorted((row["seed"].value, row["lr"].value) for row in chosen.rows) == [
        (seed, rate) for seed in range(3) for rate in (0.01, 0.03)
    ]
    assert len(choose("seed % 2 == 1 lr>=0.01", _grid(), coordinates=lambda row: row, producer="train").rows) == 4


def _swept(seeds: range, rule: str) -> dict:
    raw = definition("last")
    train = raw["steps"]["train"]
    train["base"]["inline"]["seed"] = 0
    rate = "settings.optimizer.learning_rate.value"
    train["sweep"] = {"axes": {
        "seed": {"values": {f"s{seed}": {"value": seed, "patches": {"seed": {"value": seed}}} for seed in seeds}},
        "lr": {"values": {f"r{i}": {"value": v, "patches": {rate: {"value": v}}} for i, v in enumerate(_RATES)}},
    }}
    raw["steps"]["baseline"]["inputs"]["subject"]["where"] = {"rule": rule}
    raw["steps"]["baseline"]["sweep"] = {"axes": {"subject": {"input": "subject"}}}
    return raw


def _evaluations(raw: dict) -> list:
    plan = compile_project(Project.model_validate(raw))
    assert not plan.diagnostics, plan.diagnostics
    return [run for run in plan.runs if run.kind == "evaluate"]


def test_a_stored_rule_is_resolved_when_the_plan_is_made() -> None:
    """Adding a seed to the producer adds the runs ``seed>=2`` chooses (catches a rule resolved when written)."""
    assert len(_evaluations(_swept(range(4), "seed>=2 lr=0.01"))) == 2
    assert len(_evaluations(_swept(range(5), "seed>=2 lr=0.01"))) == 3


def test_best_per_axis_plans_one_ranked_row_per_group() -> None:
    """``lowest 1 train_loss per lr`` plans one row per learning rate, ranked once the Train runs finish."""
    runs = _evaluations(_swept(range(3), "lowest 1 train_loss per lr"))
    ranks = [run.inputs["subject"].member.reference for run in runs]
    assert all(isinstance(rank, BestRank) and rank.metric == "train_loss" and rank.order == "min" for rank in ranks)
    assert sorted(rank.group["lr"] for rank in ranks) == sorted(f"r{i}" for i in range(4))
    assert all(len(rank.candidates) == 3 for rank in ranks)


@pytest.mark.parametrize(("rule", "named"), [
    ("sed=0", "sed is not swept in"),
    ("train_loss<0.1", "train_loss<0.1"),
    ("seed=9", "seed has no value 9"),
    ("seed>9", "chooses no run"),
])
def test_a_rule_is_refused_on_its_field_with_its_text(rule: str, named: str) -> None:
    """A mistyped axis, a metric comparison, a value the axis lacks and an empty choice are refused at ``where.rule``."""
    plan = compile_project(Project.model_validate(_swept(range(2), rule)))
    [problem] = [problem for problem in plan.diagnostics if problem.step == "baseline"]
    assert problem.path == "inputs.subject.where.rule"
    assert named in problem.detail
