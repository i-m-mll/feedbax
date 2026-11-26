"""Contract checks for authoring text and the core numerical evaluator."""


from hypothesis import given, strategies as st
import jax
import jax.numpy as jnp
import pytest
from pydantic import TypeAdapter, ValidationError

from feedbax.contracts.expression_values import (
    CallValue,
    DrawValue,
    LENGTH,
    TIME,
    Level,
    LiteralValue,
    NameValue,
    ProblemCode,
    ScopedValue,
    Site,
    UnresolvedValue,
    ValueBase,
    ValueType,
)
from feedbax.expression_builtins import register_builtins
from feedbax.contracts.expression_check import Checked, check, dependency_order
from feedbax.expression_compile import Compiled, Inputs, compile_expression
from feedbax.contracts.expression_registry import Function, FunctionClass, FunctionInfo, FunctionRegistry
from feedbax.contracts.expression_site import FieldKind, field_kind, project_site, read_field
from feedbax.contracts.expression_storage import StoredExpression, StoredTime, lower_expression, store_expression
from feedbax.contracts.expression_text import ParsedField, parse, parse_stored_expression, print_field, print_stored_expression
from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.stored_members import scoped_set_member_id
from feedbax.contracts.task_document import VALUE_TYPES
from feedbax.contracts.studio_field import StudioFieldView
from feedbax.contracts.time_units import TimeValue
from feedbax.studio.field_sites import FieldSiteIndex
from feedbax.studio.value_structure import value_structure


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


def test_point_expression_accepts_offset_but_marks_free_vectors(functions):
    types = dict(VALUE_TYPES)
    expected = Site(
        expected=types["Point"],
        finest=Level.TRIAL,
        names=tuple(
            ScopedValue(id=name, name=name, type=types[kind], level=Level.TRIAL)
            for name, kind in (("start", "Point"), ("offset", "Displacement"), ("velocity", "Velocity"))
        ),
    )
    result = check(parsed("start + offset", expected), expected, functions)
    assert isinstance(result, Checked)
    assert result.root.type == types["Point"]
    path = ("variables", "target", "value")
    for name, code in (("offset", ProblemCode.TYPE_MISMATCH), ("velocity", ProblemCode.UNIT_MISMATCH)):
        (problem,) = check(parsed(name, expected), expected, functions, path=path)
        assert (problem.code, problem.path) == (code, path)


def parsed(text, expected):
    result = parse(text, expected)
    assert isinstance(result, ParsedField), result
    return result.value


def compiled(
    text, expected, registry, *, field=("task", "target"), period=None, reset_on_epoch=False
):
    result = compile_expression(
        parsed(text, expected),
        expected,
        registry,
        field_address=field,
        period_seconds=period,
        reset_on_epoch=reset_on_epoch,
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
        "-2 ** 3 ** 2",
        "(2 ** 3) ** 2",
        "2 ** -2",
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


_OPERANDS = tuple(
    ScopedValue(id=f"{name}-id", name=name, type=ValueType(base=ValueBase.REAL), level=Level.RUN) for name in "abc"
)
_LEAVES = st.one_of(
    st.sampled_from([NameValue(reference=item.id) for item in _OPERANDS]),
    st.builds(
        LiteralValue,
        value=st.one_of(st.booleans(), st.integers(-5, 5), st.floats(-10, 10, allow_nan=False)),
    ),
    st.builds(LiteralValue, value=st.integers(-5, 5), unit=st.sampled_from(["s", "mm", "°"])),
)
_TREES = st.recursive(
    _LEAVES,
    lambda child: st.one_of(
        st.builds(
            lambda function, left, right: CallValue(function=function, args=(left, right)),
            st.sampled_from(["add", "sub", "mul", "div", "mod", "pow", "eq", "lt", "ge", "and", "or", "index"]),
            child, child,
        ),
        st.builds(
            lambda function, operand: CallValue(function=function, args=(operand,)),
            st.sampled_from(["neg", "pos", "not", "hold"]), child,
        ),
        st.builds(lambda items: CallValue(function="vector", args=tuple(items)), st.lists(child, min_size=1, max_size=3)),
    ),
    max_leaves=12,
)


@given(_TREES)
def test_printed_text_reads_back_as_the_same_value(tree):
    """Catches a printer whose operators or parentheses change what the text means when it is read back."""
    names = site(names=_OPERANDS)
    assert parsed(print_field(tree, names), names) == tree


def test_authored_choice_round_trips_weights_and_quoted_member_keys():
    text = r"choice('fore-arm', 'controller/o\'clock.\\axis', weights=[3, 1])"
    result = parse_stored_expression(text)
    assert isinstance(result, ParsedField)
    assert result.value == DrawValue(
        family="choice", args=(
            NameValue(reference="fore-arm"), NameValue(reference="controller/o'clock.\\axis"),
        ), weights=(3, 1),
    )
    assert print_stored_expression(result.value) == text
    assert ("weights", 1) in result.spans


@pytest.mark.parametrize("text", (
    "choice(1, 2, weights=[1])",
    "choice(1, 2, weights=[0, 0])",
    "choice(1, 2, weights=[1 s, 3])",
    "sample(B(0.5), 20)",
    "1, 2",
))
def test_authored_expression_refuses_invalid_weights_and_field_gestures(text):
    assert isinstance(parse_stored_expression(text), tuple)


def test_authored_expression_normalizes_signs_and_zero_lag():
    result = parse_stored_expression("lag((x + 2.0), 0)")
    assert isinstance(result, ParsedField)
    assert print_stored_expression(result.value) == "x + 2"
    assert ("args", 1) in result.spans
    assert print_stored_expression(LiteralValue(value=2.0)) == 2


def test_stored_expression_lowers_qualified_keys_without_label_lookup():
    stored = TypeAdapter(StoredExpression)
    text = "f(epoch.hold.condition, outputs.hand.position, set.load.heavy.force)"
    authored = stored.validate_python(text)
    assert stored.dump_python(authored, mode="json") == text
    lowered = lower_expression(authored)
    assert isinstance(lowered, CallValue)
    assert tuple(argument.reference for argument in lowered.args) == (
        "epoch:hold:condition",
        ModelAddress.from_authored_text("outputs.hand.position").to_text(),
        scoped_set_member_id("load", "heavy", "force"),
    )
    assert stored.dump_python(store_expression(lowered, scopes={
        lowered.args[-1].reference: "set.load.heavy.force",
    }), mode="json") == text


def test_stored_expression_and_time_hold_editable_text_without_tree_payloads():
    stored = TypeAdapter(StoredExpression)
    assert stored.validate_python("broken(") == UnresolvedValue(text="broken(")
    assert stored.dump_python(stored.validate_python("'later-key'"), mode="json") == "'later-key'"
    with pytest.raises(ValidationError, match="scalar or text"):
        stored.validate_python({"kind": "literal", "value": 2})
    time = TypeAdapter(StoredTime)
    assert time.validate_python("50 ms") == TimeValue(value=0.05, unit="s")
    assert time.dump_python(time.validate_python("30 st"), mode="json") == "30 st"
    assert time.validate_python("broken(") == UnresolvedValue(text="broken(")
    with pytest.raises(ValidationError, match="literal text"):
        time.validate_python({"value": 1, "unit": "s"})


@pytest.mark.parametrize(
    ("written", "shown"),
    [
        ("a + b", "a + b"),
        ("a+(b*c)", "a + b * c"),
        ("(a + b) * c", "(a + b) * c"),
        ("a - (b - c)", "a - (b - c)"),
        ("a ** (b ** c)", "a ** b ** c"),
        ("(a ** b) ** c", "(a ** b) ** c"),
        ("(-a) ** (-b)", "(-a) ** -b"),
        ("(a - b) - c", "a - b - c"),
        ("-(a + b)[0]", "-(a + b)[0]"),
        ("not (a < b and c)", "not (a < b and c)"),
        ("a + polar(b, c)", "a + polar(b, c)"),
        ("add(a, b)", "a + b"),
    ],
)
def test_text_is_shown_back_as_written_with_operators_infix(written, shown):
    """Catches text stored as written ``a + b`` and shown back as ``add(a, b)``, or with parentheses it never needed."""
    names = site(names=_OPERANDS)
    assert print_field(parsed(written, names), names) == shown


@pytest.mark.parametrize(
    ("written", "shown"),
    [
        ("1.0", "1"),
        ("-0.0", "-0.0"),
        ("1.0 st", "1 st"),
        ("45.0°", "45°"),
        ("1.0E-07", "1e-7"),
        ("1.0E+20", "1e+20"),
        ("1.0E+21", "1e+21"),
        ("0.15000000596046448", "0.15000000596046448"),
        ("(-2.0)**(1.0 + 0.1)", "(-2.0) ** (1.0 + 0.1)"),
        ("true and not false", "true and not false"),
    ],
)
def test_editor_prints_canonical_numbers_without_rewriting_the_tree(written, shown):
    expected = site()
    tree = parsed(written, expected)
    assert print_field(tree, expected) == shown
    assert parsed(shown, expected) == tree


def test_editor_keeps_a_float_list_index_and_its_type_problem(functions):
    names = (ScopedValue(id="items", name="items", type=ValueType(base=ValueBase.REAL), level=Level.RUN, list_count=2),)
    expected = site(names=names)
    tree = parsed("items[1.0]", expected)
    shown = print_field(tree, expected)
    assert shown == "items[1.0]"
    assert value_structure(tree, expected, functions).arguments == ["items", "1.0"]
    (problem,) = check(parsed(shown, expected), expected, functions)
    assert (problem.code, problem.path) == (ProblemCode.TYPE_MISMATCH, ("args", 1))


def test_editor_types_an_integer_spelling_from_its_real_root_site(functions):
    expected = site()
    original = parsed("1.0", expected)
    shown = print_field(original, expected)
    assert shown == "1"
    for tree in (original, parsed(shown, expected)):
        typed = check(tree, expected, functions)
        assert isinstance(typed, Checked)
        assert typed.root.type.base == ValueBase.REAL
    expected_boolean = site(ValueBase.BOOLEAN)
    assert print_field(original, expected_boolean) == "1.0"


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
    order, cycles = dependency_order({
        "a": NameValue(reference="b"), "b": NameValue(reference="c"), "c": NameValue(reference="a"),
        "d": NameValue(reference="a"), "e": LiteralValue(value=1),
    })
    assert order == ("d", "e")
    assert [(cycle.code, cycle.path[1], cycle.reason) for cycle in cycles] == [
        (ProblemCode.CYCLE, "a", "a → b → c → a"),
        (ProblemCode.CYCLE, "b", "b → c → a → b"),
        (ProblemCode.CYCLE, "c", "c → a → b → c"),
    ]


def evaluate(implementation, inputs):
    run = implementation.evaluate_run(inputs)
    trial = implementation.with_trial(inputs, run)
    return implementation.evaluate_step(inputs, trial)[0].value


def test_huber_replaces_squared_norm_and_uses_a_canonical_delta(functions):
    """Catch a penalty switch that loses its vector argument or changes with display units."""
    measure = site(dimensions=LENGTH.raised(2))
    authored = parsed("sqnorm([500 mm, 2 m] - [0 m, 0 m])", measure)
    assert "huber" in project_site(measure, functions, authored).parts[0].functions
    key = jax.random.key(0)
    for text, expected in (
        ("huber([0.5, 2])", 1.625),
        ("huber([500 mm, 2 m])", 1.625),
        ("huber([0.5 m, 2 m], 500 mm)", 1.0),
        ("huber([0.5 m, 2 m], 0.5)", 1.0),
    ):
        expression = compiled(text, measure, functions)
        value = evaluate(expression, Inputs(expression.parameters, {}, key, key))
        assert value.shape == ()
        assert float(value) == pytest.approx(expected)


def test_huber_requires_a_positive_scalar_delta_in_the_residual_units(functions):
    """Catch a zero threshold silently showing up as a valid zero penalty."""
    residual_type = ValueType(base=ValueBase.REAL, dimensions=LENGTH)
    delta = ScopedValue(id="threshold", name="delta", type=residual_type, level=Level.RUN)
    measure = site(dimensions=LENGTH.raised(2), names=(delta,))
    expression = compiled("huber(2 m, delta)", measure, functions)
    key = jax.random.key(0)
    for threshold, present in ((0.0, True), (-1.0, True), (1.0, False)):
        inputs = Inputs(
            expression.parameters, {"threshold": threshold}, key, key, defined={"threshold": present},
        )
        trial = expression.with_trial(inputs, expression.evaluate_run(inputs))
        result, _ = expression.evaluate_step(inputs, trial)
        assert not bool(result.defined)
    assert read_field("huber(2 m, [1 m, 2 m])", measure, functions)[0].code == ProblemCode.TYPE_MISMATCH
    assert read_field("huber(2 m, 1 s)", measure, functions)[0].code == ProblemCode.UNIT_MISMATCH
    problem = read_field("huber(2 m, 0 m)", measure, functions)[0]
    assert (problem.code, problem.path) == (ProblemCode.TYPE_MISMATCH, ("args", 1))


def test_studio_draw_cadence_follows_the_field_site(functions):
    index = FieldSiteIndex()
    value = parsed("1 + 2", site())
    for level, structural, expected_cadence in (
        (Level.RUN, False, "run"), (Level.TRIAL, False, "trial"),
        (Level.STEP, False, "trial"), (Level.RUN, True, None),
    ):
        expected = site(level=level).model_copy(update={"structural": structural})
        projection = project_site(expected, functions, value)
        ref = index.reference(projection)
        field = StudioFieldView(label="Value", text="1 + 2", site=ref)
        wire = index.tables((field,))["sites"][ref].model_dump(mode="json")
        assert ("uniform" in projection.parts[0].functions) == (expected_cadence is not None)
        if expected_cadence is None:
            assert "draw_per" not in wire
        else:
            assert wire["draw_per"] == expected_cadence

    expected = site(level=Level.TRIAL)
    for text, cadence in (("U(0, 1)", "trial"), ("U(0, 1, per=run)", "run")):
        expression = parsed(text, expected)
        structure = value_structure(expression, expected, functions)
        assert structure.model_dump(mode="json", exclude_unset=True)["value"]["per"] == cadence


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
    assert (float(first.value), float(second.value)) == (0.0, 1.0)
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
    implementation = compiled("held(near, 0.1 s)", expected, functions, period=0.05)
    key = jax.random.key(0)
    state = implementation.initial_state
    observed = []
    for step, near in enumerate((True, True, True, False, True)):
        value, state = implementation.evaluate_step(
            Inputs(implementation.parameters, {"condition": near}, key, key, step=step),
            state,
        )
        observed.append(bool(value.value))
    assert observed == [False, False, True, False, False]


def scan_expression(implementation, values, entries):
    key = jax.random.key(0)
    count = len(entries)

    def run(values):
        def step(state, data):
            k, environment, epoch_entry = data
            result, state = implementation.evaluate_step(
                Inputs(
                    implementation.parameters,
                    environment,
                    key,
                    key,
                    step=k,
                    epoch_entry=epoch_entry,
                ),
                state,
            )
            return state, result

        inputs = Inputs(implementation.parameters, {}, key, key)
        state = implementation.with_trial(inputs, implementation.evaluate_run(inputs))
        return jax.lax.scan(step, state, (jnp.arange(count), values, entries))[1]

    return jax.jit(run)(values)


def test_backward_difference_has_trial_history_and_time_dimensions(functions):
    vector = ValueType(base=ValueBase.REAL, shape=(2,), dimensions=LENGTH)
    names = (ScopedValue(id="source", name="x", type=vector, level=Level.STEP),)
    x = jnp.asarray([[0.0, 1.0], [1.0, 4.0], [4.0, 9.0], [9.0, 16.0], [16.0, 25.0]])
    epoch_entries = jnp.asarray([True, False, True, False, False])
    for spelling, dimensions, scale in (
        ("diff(x, 2)", LENGTH, 1.0),
        ("ddt(x, n=2)", LENGTH.combine(TIME, -2), 4.0),
    ):
        expected = site(shape=(2,), dimensions=dimensions, level=Level.STEP, names=names)
        form = compiled(spelling, expected, functions, period=0.5, reset_on_epoch=True)
        result = scan_expression(form, {"source": x}, epoch_entries)
        assert result.defined.tolist() == [False, False, True, True, True]
        assert jnp.allclose(result.value[result.defined], scale * jnp.diff(x, n=2, axis=0))
        assert form.checked.root.type.dimensions == dimensions
        empty = scan_expression(form, {"source": x[:0]}, epoch_entries[:0])
        assert empty.defined.shape == (0,)
    assert read_field("diff(x, 0)", expected, functions)[0].path == ("args", 1)
    assert read_field("diff(x, n=x)", expected, functions)[0].path == ("kwargs", "n")


def test_selected_branch_and_chained_temporal_validity(functions):
    numeric = ValueType(base=ValueBase.REAL)
    names = (ScopedValue(id="source", name="x", type=numeric, level=Level.STEP),)
    expected = site(level=Level.STEP, names=names)
    x = jnp.arange(4.0)
    entries = jnp.asarray([True, False, False, False])
    form = compiled("where(x == 0, 7, diff(diff(x)))", expected, functions)
    result = scan_expression(form, {"source": x}, entries)
    assert result.defined.tolist() == [True, False, True, True]
    assert result.value[result.defined].tolist() == [7.0, 0.0, 0.0]


def test_held_restarts_on_epoch_entry_only_at_epoch_sites(functions):
    names = (
        ScopedValue(
            id="condition", name="near", type=ValueType(base=ValueBase.BOOLEAN), level=Level.STEP
        ),
    )
    expected = site(ValueBase.BOOLEAN, level=Level.STEP, names=names)
    entries = jnp.asarray([True, False, False, True, False, False])
    for reset_on_epoch, values in (
        (True, [False, False, True, False, False, True]),
        (False, [False, False, True, True, True, True]),
    ):
        form = compiled(
            "held(near, 2 st)", expected, functions, period=0.05, reset_on_epoch=reset_on_epoch
        )
        result = scan_expression(form, {"condition": jnp.ones(6, dtype=bool)}, entries)
        assert result.value.tolist() == values


def test_shapes_on_the_local_axis(functions):
    clock = ValueType(base=ValueBase.REAL, dimensions=TIME)
    names = (
        ScopedValue(id="local", name="t", type=clock, level=Level.STEP),
        ScopedValue(id="duration", name="T", type=clock, level=Level.TRIAL),
    )
    expected = site(level=Level.STEP, names=names)
    timeline = jnp.arange(9) * 0.1
    env = {"local": timeline, "duration": jnp.ones_like(timeline)}
    entries = jnp.zeros(9, dtype=bool)
    repeating = compiled("pulse(1, 8, 0.19 s, 0.21 s, every=4 st)", expected, functions, period=0.1)
    assert scan_expression(repeating, env, entries).value.tolist() == [1, 1, 8, 8, 1, 1, 8, 8, 1]
    once = compiled("pulse(1, 8, 0.19 s, 0.21 s)", expected, functions, period=0.1)
    assert scan_expression(once, env, entries).value.tolist() == [1, 1, 8, 8, 1, 1, 1, 1, 1]

    for mode, value in (("hold", 0.0), ("linear", 2.5), ("cubic", 1.796875)):
        text = f"points([0.2, 0.4, 0.6], [0, 10, 20], interp={mode})"
        form = compiled(text, expected, functions)
        result = scan_expression(form, env, entries)
        assert result.defined.all()
        assert jnp.allclose(result.value[jnp.asarray([0, 1, 7, 8])], jnp.asarray([0, 0, 20, 20]))
        sample_env = {"local": jnp.asarray([0.25]), "duration": jnp.ones(1)}
        sample = scan_expression(form, sample_env, entries[:1])
        assert float(sample.value[0]) == pytest.approx(value)
        tree = parsed(text, expected)
        assert parsed(print_field(tree, expected), expected) == tree

    bad = read_field("points([0, 1], [1])", expected, functions)
    assert bad[0].code == ProblemCode.LIST_COUNT
    bad = read_field("points([0.5, 0.5], [1, 2])", expected, functions)
    assert bad[0].path == ("args", 0)
    bad = read_field("points([0, 1], [1, 2], interp=other)", expected, functions)
    assert bad[0].path == ("kwargs", "interp")
    assert "data" not in functions.keys()

    sines = compiled("sines(2, 3, [1 Hz], [90°])", expected, functions)
    assert jnp.allclose(
        scan_expression(sines, env, entries).value, 2 + 3 * jnp.cos(2 * jnp.pi * timeline)
    )
    noise = compiled("noise(0, 1)", expected, functions)
    observed = scan_expression(noise, env, entries).value
    assert len(jnp.unique(observed)) == len(timeline)
    assert jnp.array_equal(scan_expression(noise, env, entries).value, observed)


@pytest.mark.parametrize("text", [
    "choice(point(1 m, 2 m), point(3 m, 4 m))",
    "lag(goal + offset, 2)",
    "lag(goal + offset, 0)",
    "index(goals, i)",
    "lag(index(goals, 1) + offset, 2)",
])
def test_projected_replacements_agree_with_whole_expression_check(functions, text):
    from feedbax.contracts.expression_site import _replace_part, _with_function
    from feedbax.contracts.expression_values import walk

    types = dict(VALUE_TYPES)
    point = types["Point"]
    scope = Site(
        expected=point, finest=Level.STEP, repeat_count=2,
        names=(
            ScopedValue(id="goal", name="goal", type=point, level=Level.RUN),
            ScopedValue(id="next_goal", name="next_goal", type=point, level=Level.RUN),
            ScopedValue(id="goals", name="goals", type=point, level=Level.RUN, list_count=2),
            ScopedValue(id="offset", name="offset", type=types["Displacement"], level=Level.TRIAL),
            ScopedValue(id="other_offset", name="other_offset", type=types["Displacement"], level=Level.TRIAL),
            ScopedValue(id="length", name="length", type=ValueType(base=ValueBase.REAL, dimensions=LENGTH),
                        level=Level.TRIAL),
            ScopedValue(id="i", name="i", type=ValueType(base=ValueBase.INTEGER), level=Level.RUN),
            ScopedValue(id="j", name="j", type=ValueType(base=ValueBase.INTEGER), level=Level.RUN),
        ),
    )
    tree = parsed(text, scope)
    baseline = check(tree, scope, functions)
    projection = project_site(scope, functions, tree)

    def eligible(path, replacement):
        candidate = check(_replace_part(tree, path, replacement), scope, functions)
        return isinstance(candidate, Checked) or (
            isinstance(baseline, tuple)
            and all(item in baseline and item.path[:len(path)] != path for item in candidate)
        )

    for part, (path, node) in zip(projection.parts, walk(tree), strict=True):
        assert part.path == path
        assert part.functions == tuple(
            info.id for info in projection.functions
            if (replacement := _with_function(node, info, scope.finest)) is not None
            and eligible(path, replacement)
        )
        assert part.names == tuple(
            name.id for name in scope.names if eligible(path, NameValue(reference=name.id))
        )
