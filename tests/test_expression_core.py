"""Contract checks for authoring text and the core numerical evaluator."""

import asyncio
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

from feedbax.contracts.expression_values import (
    LENGTH,
    TIME,
    CallValue,
    Level,
    LiteralValue,
    NameValue,
    ProblemCode,
    ScopedValue,
    Site,
    ValueBase,
    ValueType,
)
from feedbax.expression_builtins import register_builtins
from feedbax.expression_check import check, dependency_order
from feedbax.expression_compile import Compiled, Inputs, compile_expression
from feedbax.expression_registry import Function, FunctionClass, FunctionInfo, FunctionRegistry
from feedbax.expression_site import FieldKind, field_kind, project_site, read_field
from feedbax.expression_text import ParsedField, parse, print_field
from feedbax.plugins.application import EXPRESSION_FUNCTIONS
from feedbax.plugins.bootstrap import (
    DOWNSTREAM_PROTOCOL_CURRENT,
    FamilyRequirement,
    PluginDeclaration,
    PluginRegistration,
)
from feedbax.plugins.composition import compose_application
from feedbax.studio.authoring import authoring_surface


@pytest.fixture
def functions():
    registry = FunctionRegistry()
    register_builtins(registry)
    registry.seal()
    return registry


def site(base=ValueBase.REAL, *, shape=(), level=Level.RUN, names=(), dimensions=None):
    value = ValueType(
        base=base, shape=shape, **({} if dimensions is None else {"dimensions": dimensions})
    )
    return Site(expected=value, finest=level, names=names)


def parsed(text, expected):
    result = parse(text, expected)
    assert isinstance(result, ParsedField), result
    return result.value


def compiled(text, expected, registry, *, field=("task", "target"), period=None):
    result = compile_expression(
        parsed(text, expected), expected, registry, field_address=field, period_seconds=period
    )
    assert isinstance(result, Compiled), result
    return result


@pytest.mark.parametrize(
    "text",
    [
        "0.5",
        "0.3 s",
        "30 st",
        "45°",
        "[0.1, 0.2]",
        "1, 2, 4",
        "1,",
        "[0, 0], [0.1, 0]",
        "linspace(0, 1, 5)",
        "geomspace(1e-4, 1e-2, 5)",
        "sample(U(0, 1), 8)",
        "U(-0.3, 1, per=run)",
        "(1 + 2) * -3",
        "true and not (1 < 2)",
        "[1,2][1]",
        "hold([1, 2])",
    ],
)
def test_text_round_trip_without_edit_history(text):
    expected = site()
    first = parsed(text, expected)
    assert parsed(print_field(first, expected), expected) == first
    assert '"id"' not in first.model_dump_json()


def test_scope_ids_survive_rename_and_duplicate_names_report_second():
    variable = ScopedValue(
        id="variable-id", name="start", type=ValueType(base=ValueBase.REAL), level=Level.TRIAL
    )
    original = site(level=Level.TRIAL, names=(variable,))
    revised = original.model_copy(
        update={"names": (variable.model_copy(update={"name": "initial"}),)}
    )
    tree = parsed("start + 1", original)
    assert parsed(print_field(tree, revised), revised) == tree
    duplicate = original.model_copy(
        update={"names": (variable, variable.model_copy(update={"id": "other"}))}
    )
    (problem,) = parse("start", duplicate)
    assert (problem.code, problem.path) == (ProblemCode.DUPLICATE_NAME, ("names", 1))
    for name in ("hand velocity", "true"):
        unusual = original.model_copy(
            update={"names": (variable.model_copy(update={"name": name}),)}
        )
        assert parsed(print_field(tree, unusual), unusual) == tree


def test_editor_value_sweep_and_structural_integer_refusal(functions):
    size = site(ValueBase.INTEGER).model_copy(update={"structural": True})
    assert project_site(size, functions).kinds == (
        FieldKind.VALUE,
        FieldKind.SWEEP,
        FieldKind.FORMULA,
    )
    assert read_field("64", size, functions).value == LiteralValue(value=64)
    (problem,) = read_field("U(0, 1)", size, functions)
    assert problem.reason == "A structural field takes Value, Sweep, Formula."
    vector = site(shape=(2,))
    assert field_kind(read_field("[0, 0]", vector, functions).value, functions) == FieldKind.VALUE
    assert (
        field_kind(read_field("[0, 0], [0.1, 0]", vector, functions).value, functions)
        == FieldKind.SWEEP
    )
    sample = read_field("sample(U(0, 1), 8)", site(), functions).value
    assert sample.count == 8 and sample.draw.family == "uniform"
    assert set(sample.model_dump()) == {"kind", "count", "draw"}


def test_bad_documents_return_path_addressed_problems(functions):
    period = ScopedValue(
        id="period",
        name="T",
        type=ValueType(base=ValueBase.REAL, dimensions=TIME),
        level=Level.STEP,
    )
    start = ScopedValue(
        id="start",
        name="start",
        type=ValueType(base=ValueBase.REAL, shape=(2,), dimensions=LENGTH),
        level=Level.TRIAL,
    )
    expected = site(shape=(2,), dimensions=LENGTH, level=Level.STEP, names=(period, start))
    assert isinstance(read_field("start + [0.1, 0]", expected, functions), ParsedField)
    assert isinstance(
        read_field("0.5 * T", site(dimensions=TIME, level=Level.STEP, names=(period,)), functions),
        ParsedField,
    )
    assert read_field("T", site(level=Level.STEP), functions)[0].code == ProblemCode.OUT_OF_SCOPE
    assert read_field("nonesuch", expected, functions)[0].code == ProblemCode.UNKNOWN_NAME
    assert (
        read_field("0.3 s + 1 m", site(dimensions=TIME), functions)[0].code
        == ProblemCode.UNIT_MISMATCH
    )
    assert read_field("true", expected, functions)[0].code == ProblemCode.TYPE_MISMATCH
    assert check(parsed("U(0,1)", site()), site(), functions)[0].code == ProblemCode.LEVEL_TOO_FINE
    (cycle,) = dependency_order({"a": NameValue(reference="b"), "b": NameValue(reference="a")})
    assert (cycle.code, cycle.path) == (ProblemCode.CYCLE, ("variables", "b", "value"))


def evaluate(implementation, inputs):
    run = implementation.evaluate_run(inputs)
    trial = implementation.with_trial(inputs, run)
    return implementation.evaluate_step(inputs, trial)[0]


def test_literal_edits_keep_real_operands_and_do_not_retrace(functions):
    expected = site()
    first = compiled("1 + 2", expected, functions)
    edited = compiled("0.5 + 3", expected, functions)
    key = jax.random.key(0)
    traces = []

    @jax.jit
    def step(parameters):
        traces.append(True)
        return evaluate(first, Inputs(parameters, {}, key, key))

    assert float(step(first.parameters)) == 3.0
    assert float(step(edited.parameters)) == 3.5
    assert len(traces) == 1


def test_function_declarations_contextualize_nested_literals(functions):
    for text, expected in (
        ("abs(1)", site(dimensions=LENGTH)),
        ("polar(1, 45°)", site(shape=(2,), dimensions=LENGTH)),
        ("lowpass(1, 0.1 s)", site(dimensions=LENGTH, level=Level.STEP)),
    ):
        assert isinstance(read_field(text, expected, functions), ParsedField), text
    for before, after in (("where(true, 1, 2)", "where(true, 0.5, 2)"), ("norm(1)", "norm(0.5)")):
        original = compiled(before, site(), functions)
        edited = compiled(after, site(), functions)
        assert tuple(arg.dtype for arg in original.parameters) == tuple(
            arg.dtype for arg in edited.parameters
        )
    boolean = site(ValueBase.BOOLEAN, level=Level.TRIAL)
    key = jax.random.key(4)
    for value in (0, 1):
        draw = compiled(f"B({value})", boolean, functions)
        assert bool(evaluate(draw, Inputs(draw.parameters, {}, key, key))) == bool(value)


def test_plugin_state_is_a_pytree_and_run_randomness_uses_the_run_key():
    registry = FunctionRegistry()
    real = ValueType(base=ValueBase.REAL)
    registry.register(
        Function(
            FunctionInfo(
                id="demo.counter",
                signature="counter()",
                display_class=FunctionClass.MATH,
                min_args=0,
                max_args=0,
                level=Level.STEP,
                stateful=True,
            ),
            lambda *_: real,
            lambda ctx: (ctx.state[0], (ctx.state[0] + 1, True)),
            initial_state=lambda _: (0, False),
        )
    )
    registry.register(
        Function(
            FunctionInfo(
                id="demo.run_random",
                signature="run_random()",
                display_class=FunctionClass.MATH,
                min_args=0,
                max_args=0,
                random=True,
            ),
            lambda *_: real,
            lambda ctx: jax.random.uniform(ctx.key),
        )
    )
    counter = compiled("demo.counter()", site(level=Level.STEP), registry)
    run, a, b = jax.random.split(jax.random.key(8), 3)
    inputs = Inputs(counter.parameters, {}, run, a)
    step = jax.jit(counter.evaluate_step)
    first, state = step(inputs, counter.initial_state)
    second, _ = step(inputs, state)
    assert (float(first), float(second)) == (0.0, 1.0)
    run_random = compiled("demo.run_random()", site(), registry)
    assert evaluate(run_random, Inputs((), {}, run, a)) == evaluate(
        run_random, Inputs((), {}, run, b)
    )


def test_draws_pair_by_field_and_written_order(functions):
    expected = site(shape=(2,), level=Level.TRIAL)
    first = compiled("[U(0, 1), U(0, 1)]", expected, functions)
    edited = compiled("[U(10, 11), U(10, 11)]", expected, functions)
    sibling = compiled("U(0, 1)", site(level=Level.TRIAL), functions, field=("task", "sibling"))
    run, trial = jax.random.split(jax.random.key(112))
    before = evaluate(first, Inputs(first.parameters, {}, run, trial, 0))
    evaluate(sibling, Inputs(sibling.parameters, {}, run, trial, 0))
    after = evaluate(edited, Inputs(edited.parameters, {}, run, trial, 0))
    assert before[0] != before[1]
    assert jnp.allclose(before, after - 10, atol=1e-6)
    assert jnp.array_equal(before, evaluate(first, Inputs(first.parameters, {}, run, trial, 0)))
    assert not jnp.array_equal(before, evaluate(first, Inputs(first.parameters, {}, run, trial, 1)))


def test_held_waits_for_elapsed_time_and_resets(functions):
    boolean = ValueType(base=ValueBase.BOOLEAN)
    names = (
        ScopedValue(
            id="clock",
            name="t",
            type=ValueType(base=ValueBase.REAL, dimensions=TIME),
            level=Level.STEP,
        ),
        ScopedValue(id="condition", name="near", type=boolean, level=Level.STEP),
    )
    expected = site(ValueBase.BOOLEAN, level=Level.STEP, names=names)
    implementation = compiled("held(near, 0.1 s)", expected, functions)
    key = jax.random.key(0)
    state = implementation.initial_state
    observed = []
    for clock, near in ((0.0, True), (0.05, True), (0.1, True), (0.15, False), (0.2, True)):
        value, state = implementation.evaluate_step(
            Inputs(implementation.parameters, {"clock": clock, "condition": near}, key, key),
            state,
        )
        observed.append(bool(value))
    assert observed == [False, False, True, False, False]


def test_plugin_function_uses_application_provenance():
    definition = FunctionInfo(
        id="example.twice",
        signature="twice(x)",
        display_class=FunctionClass.MATH,
        min_args=1,
        max_args=1,
    )

    def register(context):
        context.registry(EXPRESSION_FUNCTIONS).register(
            Function(
                definition,
                lambda args, _: args[0],
                lambda _context, value: 2 * value,
            )
        )

    registration = PluginRegistration(
        PluginDeclaration(
            plugin_id="example",
            version="1",
            downstream_protocol_version=DOWNSTREAM_PROTOCOL_CURRENT,
            families=(FamilyRequirement(EXPRESSION_FUNCTIONS.family, "1"),),
        ),
        register,
    )
    entry = SimpleNamespace(
        name="example",
        value="example:registration",
        dist=SimpleNamespace(name="example-dist", version="1"),
        load=lambda: registration,
    )
    state = asyncio.run(compose_application(entry_points=(entry,)))
    projection = authoring_surface(state, catalog=True).registries[EXPRESSION_FUNCTIONS.family]
    assert definition in projection.entries
    assert projection.origins[definition.id].distribution == "example-dist"
    implementation = compiled("example.twice(3)", site(), state.bundle.expression_functions)
    key = jax.random.key(0)
    assert evaluate(implementation, Inputs(implementation.parameters, {}, key, key)) == 6
    absent = asyncio.run(compose_application(entry_points=()))
    tree = CallValue(function=definition.id, args=(LiteralValue(value=3),))
    assert (
        check(tree, site(), absent.bundle.expression_functions)[0].code
        == ProblemCode.PLUGIN_MISSING
    )
