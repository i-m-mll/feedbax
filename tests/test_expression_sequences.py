"""Stored sequence rules have an ordered, statically counted meaning."""

import jax
import jax.numpy as jnp
import pytest

from feedbax.contracts.expression_check import Checked, check
from feedbax.contracts.expression_registry import FunctionRegistry
from feedbax.contracts.expression_sequence import sequence_definition, structural_integer
from feedbax.contracts.expression_site import read_field
from feedbax.contracts.expression_text import (
    ParsedField, parse, parse_sequence_rule, parse_stored_expression, print_stored_expression,
)
from feedbax.contracts.expression_values import (
    TIME, CallValue, Level, ProblemCode, ScopedValue, SequenceValue, Site, ValueBase, ValueType,
)
from feedbax.expression_builtins import register_builtins
from feedbax.expression_compile import Compiled, Inputs, compile_expression


@pytest.fixture
def functions():
    registry = FunctionRegistry()
    register_builtins(registry)
    registry.seal()
    return registry


def _stored(text):
    parsed = parse_stored_expression(text)
    assert isinstance(parsed, ParsedField), parsed
    return parsed.value


def _compile(text, site, functions, **options):
    code = compile_expression(_stored(text), site, functions, field_address=("sequence",), **options)
    assert isinstance(code, Compiled), code
    return code


def _evaluate(code, values=None, defined=None):
    key = jax.random.key(8)
    inputs = Inputs(code.parameters, values or {}, key, key, defined=defined)
    state = code.with_trial(inputs, code.evaluate_run(inputs))
    return code.evaluate_step(inputs, state)[0]


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


def test_sequence_size_is_independent_of_its_member_count():
    small, large = (_stored(f"choice(linspace(0, 1, {count}))") for count in (10, 1000))
    assert len(large.model_dump_json()) - len(small.model_dump_json()) == 2
    assert len(print_stored_expression(large)) - len(print_stored_expression(small)) == 2
    assert sequence_definition(large.args[0]).count == 1000


def test_weighted_choice_compiles_one_run_array(functions):
    text = "choice(0..2, 10, per=run, weights=[0, 0, 1, 0])"
    site = Site(expected=ValueType(base=ValueBase.INTEGER), finest=Level.RUN)
    code = _compile(text, site, functions)
    assert print_stored_expression(code.checked.tree) == text
    assert isinstance(code.checked.tree.args[0], SequenceValue)
    assert tuple(value.shape for value in code.parameters) == ((3,), (), (4,))
    assert len(code.checked.nodes) == 3
    result = _evaluate(code)
    assert bool(result.defined) and int(result.value) == 2
    nominal = _compile(text, site, functions, nominal=True)
    assert int(_evaluate(nominal).value) == 2


def test_vector_flattens_sequences_and_index_selects_rows(functions):
    real = ValueType(base=ValueBase.REAL)
    vector = Site(expected=real.model_copy(update={"shape": (4,)}), finest=Level.RUN)
    code = _compile("[0..1, linspace(2, 3, 1 + 1)]", vector, functions)
    assert _evaluate(code).value.tolist() == [0, 1, 2, 3]

    name = ScopedValue(id="items", name="List", level=Level.TRIAL, type=real, list_count=4)
    sliced = Site(expected=real.model_copy(update={"shape": (2,)}), finest=Level.TRIAL, names=(name,))
    code = _compile("items[1..2]", sliced, functions)
    result = _evaluate(code, {"items": jnp.arange(4)}, {"items": jnp.asarray([True, True, False, True])})
    assert result.value.tolist() == [1, 2]
    assert result.defined.tolist() == [True, False]
    scalar = _compile("items[3 - 1]", sliced.model_copy(update={"expected": real}), functions)
    assert int(_evaluate(scalar, {"items": jnp.arange(4)}).value) == 2
    negative = _compile("[5, 6, 7][-2..-1]", sliced, functions)
    assert _evaluate(negative).value.tolist() == [6, 7]


def test_sequences_refuse_wrong_positions_units_and_function_spelling(functions):
    site = Site(expected=ValueType(base=ValueBase.REAL), finest=Level.TRIAL)
    (problem,) = check(_stored("choice(polar(0.15 m, linspace(0°, 315°, 8)))"), site, functions)
    assert problem.code == ProblemCode.STRUCTURE and problem.path == ("args", 0, "args", 1)
    (problem,) = check(_stored("choice(linspace(0 st, 1 st, 3))"), site, functions)
    assert problem.code == ProblemCode.STRUCTURE and "whole number" in problem.reason
    (problem,) = check(_stored("range(6)"), site, functions)
    assert problem.code == ProblemCode.UNKNOWN_FUNCTION and "0..5" in problem.reason
    (problem,) = parse_sequence_rule("range(6)")
    assert problem.code == ProblemCode.UNKNOWN_FUNCTION and "0..5" in problem.reason
    assert isinstance(parse_stored_expression("choice(0..2, weights=[1, 1])"), tuple)


def test_field_range_and_structural_lag_and_difference(functions):
    real = ValueType(base=ValueBase.REAL)
    signal = ScopedValue(id="x", name="Signal", level=Level.STEP, type=real)
    count = ScopedValue(id="n", name="Count", level=Level.RUN, type=ValueType(base=ValueBase.INTEGER))
    site = Site(expected=real, finest=Level.STEP, names=(signal, count))
    for text in ("lag(x, 2 + 1)", "diff(x, n=2 * 2)"):
        code = _compile(text, site, functions)
        assert isinstance(code.checked.tree.args[0], type(_stored("x")))
        assert code.initial_state.slots
    derivative = site.model_copy(update={"expected": real.model_copy(update={"dimensions": TIME.raised(-2)})})
    assert isinstance(_compile("ddt(x, 1 + 1)", derivative, functions, period_seconds=0.1), Compiled)
    (problem,) = check(_stored("lag(x, n)"), site, functions)
    assert problem.code == ProblemCode.STRUCTURE and "non-negative whole number" in problem.reason
    zero = _compile("lag(x, 1 - 1)", site, functions)
    assert not zero.initial_state.slots and float(_evaluate(zero, {"x": 4.0}).value) == 4.0
    assert parse("lag(Signal, 0)", site).value == _stored("x")

    shown = read_field("linspace(0, 1, 2 + 2)", site, functions)
    assert isinstance(shown, ParsedField) and isinstance(shown.value, SequenceValue)
    assert sequence_definition(shown.value).count == 4
    assert isinstance(shown.value.count, CallValue)
