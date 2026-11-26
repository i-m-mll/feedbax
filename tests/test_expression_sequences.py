"""Stored sequence rules have an ordered, statically counted meaning."""

import pytest

from feedbax.contracts.expression_sequence import sequence_definition, structural_integer
from feedbax.contracts.expression_text import ParsedField, parse_sequence_rule, parse_stored_expression, print_stored_expression
from feedbax.contracts.expression_values import SequenceValue


@pytest.mark.parametrize(("text", "members", "unit"), [
    ("-2..2", [-2, -1, 0, 1, 2], None),
    ("1 st..3 st", [1, 2, 3], "st"),
    ("linspace(0, 1, 5)", [0, 0.25, 0.5, 0.75, 1], None),
    ("linspace(0°, 315°, 8)", [0, 45, 90, 135, 180, 225, 270, 315], "°"),
    ("geomspace(1, 100, 3)", [1, 10, 100], None),
    ("linspace(0, 5, 1)", [0], None),
])
def test_sequence_rule_members_and_canonical_text(text, members, unit):
    parsed = parse_sequence_rule(text)
    assert isinstance(parsed, ParsedField)
    assert print_stored_expression(parsed.value) == text
    assert parse_sequence_rule(print_stored_expression(parsed.value)).value == parsed.value
    rule = sequence_definition(parsed.value)
    assert rule.count == len(members)
    assert [(item.value, item.unit) for item in rule.members()] == [(value, unit) for value in members]


@pytest.mark.parametrize("text", ["(0..5)", "a == 0..5", "0..1..2"])
def test_double_dot_is_a_sequence_production(text):
    assert isinstance(parse_sequence_rule(text), tuple)


def test_structural_family_bounds_keep_the_expression():
    parsed = parse_sequence_rule("0..-k * -2 - 1")
    assert isinstance(parsed, ParsedField)
    assert isinstance(parsed.value, SequenceValue)
    assert structural_integer(parsed.value.last, {"k": 3}) == 5
    assert sequence_definition(parsed.value, {"k": 3}).count == 6
    assert print_stored_expression(parsed.value) == "0..-k * -2 - 1"
    bad = parse_stored_expression("k % 0")
    assert isinstance(bad, ParsedField)
    with pytest.raises(ValueError, match="remainder by zero"):
        structural_integer(bad.value, {"k": 3})
