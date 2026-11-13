"""The first expression function declarations and their numerical operations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import partial
from typing import Any

from feedbax.contracts.expression_values import (
    ANGLE,
    DIMENSIONLESS,
    TIME,
    CallValue,
    Dimensions,
    Level,
    LiteralValue,
    ProblemCode,
    ValueBase,
    ValueType,
)
from feedbax.expression_registry import (
    Evaluation,
    Function,
    FunctionClass,
    FunctionInfo,
    FunctionRegistry,
    OperandHint,
    SignatureError,
    StateReset,
)


_BOOL = ValueBase.BOOLEAN
_INT = ValueBase.INTEGER
_REAL = ValueBase.REAL
_SHAPE = FunctionClass.SHAPE
_DRAW = FunctionClass.DRAW
_PENALTY = FunctionClass.PENALTY


@dataclass(frozen=True)
class CallContext:
    axis: Mapping[str, Any]
    key: Any
    state: Any
    period_seconds: Any
    defined: tuple[Any, ...]
    step: Any
    discrete_uniform: bool


def _type(reason: str) -> SignatureError:
    return SignatureError(ProblemCode.TYPE_MISMATCH, reason)


def _unit(reason: str) -> SignatureError:
    return SignatureError(ProblemCode.UNIT_MISMATCH, reason)


def _elementwise(
    args: tuple[ValueType, ...],
    _kwargs: Mapping[str, ValueType],
    *,
    base: ValueBase | None = None,
    dimensions: str = "same",
    boolean: bool = False,
) -> ValueType | SignatureError:
    if any((arg.base == _BOOL) != boolean for arg in args):
        return _type("Expected boolean arguments." if boolean else "Expected numeric arguments.")
    shapes = {arg.shape for arg in args if arg.shape}
    if len(shapes) > 1:
        return _type("Argument shapes differ.")
    unit = args[0].dimensions
    if dimensions == "same" and any(arg.dimensions != unit for arg in args[1:]):
        return _unit("Arguments require equal dimensions.")
    if dimensions == "dimensionless" and any(arg.dimensions != DIMENSIONLESS for arg in args):
        return _unit("Expected dimensionless arguments.")
    if dimensions in ("product", "quotient"):
        unit = unit.combine(args[1].dimensions, 1 if dimensions == "product" else -1)
    actual_base = base or (
        _BOOL if boolean else _REAL if any(arg.base == _REAL for arg in args) else _INT
    )
    return ValueType(
        base=actual_base,
        shape=next(iter(shapes), ()),
        dimensions=DIMENSIONLESS if actual_base == _BOOL else unit,
    )


def _vector(
    args: tuple[ValueType, ...], kwargs: Mapping[str, ValueType]
) -> ValueType | SignatureError:
    if any(arg.shape != args[0].shape for arg in args):
        return _type("Vector members must have the same shape.")
    result = _elementwise(args, kwargs, boolean=all(arg.base == _BOOL for arg in args))
    if isinstance(result, SignatureError):
        return result
    return ValueType(
        base=result.base, dimensions=result.dimensions, shape=(len(args), *result.shape)
    )


def _index(
    args: tuple[ValueType, ...], _kwargs: Mapping[str, ValueType]
) -> ValueType | SignatureError:
    source, index = args
    if not source.shape or index != ValueType(base=_INT):
        return _type("Indexing needs an array and a scalar integer index.")
    return ValueType(base=source.base, shape=source.shape[1:], dimensions=source.dimensions)


def _where(
    args: tuple[ValueType, ...], kwargs: Mapping[str, ValueType]
) -> ValueType | SignatureError:
    condition, *branches = args
    if condition.base != _BOOL:
        return _type("Where needs a boolean condition.")
    branches = tuple(branches)
    result = _elementwise(branches, kwargs, boolean=all(arg.base == _BOOL for arg in branches))
    if isinstance(result, SignatureError):
        return result
    if condition.shape and result.shape and condition.shape != result.shape:
        return _type("Condition and branch shapes differ.")
    return result.model_copy(update={"shape": result.shape or condition.shape})


def _equality(
    args: tuple[ValueType, ...], kwargs: Mapping[str, ValueType]
) -> ValueType | SignatureError:
    return _elementwise(args, kwargs, base=_BOOL, boolean=all(arg.base == _BOOL for arg in args))


def _norm(
    args: tuple[ValueType, ...],
    kwargs: Mapping[str, ValueType],
    *,
    power: int,
) -> ValueType | SignatureError:
    result = _elementwise(args, kwargs, base=_REAL)
    if isinstance(result, SignatureError):
        return result
    return ValueType(base=_REAL, dimensions=result.dimensions.raised(power))


def _polar(
    args: tuple[ValueType, ...], _kwargs: Mapping[str, ValueType]
) -> ValueType | SignatureError:
    radius, angle = args
    if radius.shape or angle.shape or _BOOL in (radius.base, angle.base):
        return _type("Polar requires numeric scalar radius and angle.")
    if angle.dimensions != ANGLE:
        return _unit("Polar requires an angle.")
    return ValueType(base=_REAL, shape=(2,), dimensions=radius.dimensions)


def _duration(value: ValueType) -> SignatureError | None:
    if value.shape or value.base == _BOOL:
        return _type("Duration must be a numeric scalar.")
    if value.dimensions != TIME:
        return _unit("Duration requires time dimensions.")
    return None


def _endpoints(
    args: tuple[ValueType, ...], kwargs: Mapping[str, ValueType]
) -> ValueType | SignatureError:
    if "over" in kwargs and (error := _duration(kwargs["over"])) is not None:
        return error
    return _elementwise(args, kwargs, base=_REAL)


def _held(
    args: tuple[ValueType, ...], _kwargs: Mapping[str, ValueType]
) -> ValueType | SignatureError:
    condition, duration = args
    if condition != ValueType(base=_BOOL):
        return _type("Held needs a scalar boolean condition.")
    return _duration(duration) or condition


def _filter(
    args: tuple[ValueType, ...], _kwargs: Mapping[str, ValueType]
) -> ValueType | SignatureError:
    value, duration = args
    if value.base == _BOOL:
        return _type("A filter requires numeric values.")
    return _duration(duration) or ValueType(
        base=_REAL, shape=value.shape, dimensions=value.dimensions
    )


def _bernoulli(
    args: tuple[ValueType, ...], kwargs: Mapping[str, ValueType]
) -> ValueType | SignatureError:
    return _elementwise(args, kwargs, base=_BOOL, dimensions="dimensionless")


def _choice(
    args: tuple[ValueType, ...], kwargs: Mapping[str, ValueType]
) -> ValueType | SignatureError:
    if any(arg.shape != args[0].shape for arg in args):
        return _type("Choice members must have the same shape.")
    return _elementwise(args, kwargs, boolean=all(arg.base == _BOOL for arg in args))


def _region(
    args: tuple[ValueType, ...], _kwargs: Mapping[str, ValueType]
) -> ValueType | SignatureError:
    center, radius = args
    if center.shape != (2,) or radius.shape or _BOOL in (center.base, radius.base):
        return _type("A disk requires a numeric 2-vector centre and scalar radius.")
    if center.dimensions != radius.dimensions:
        return _unit("A disk requires equal dimensions for its centre and radius.")
    return ValueType(base=_REAL, shape=(2,), dimensions=center.dimensions)


def _operation(op: str, _context: CallContext, *args, **kwargs):
    import jax.numpy as jnp

    return getattr(jnp, op)(*args, **kwargs)


def _stack(_context: CallContext, *args):
    import jax.numpy as jnp

    return jnp.stack(args)


def _at(_context: CallContext, array, index):
    import jax.numpy as jnp

    return jnp.take(array, index, axis=0)


def _polar_value(_context: CallContext, radius, angle):
    import jax.numpy as jnp

    return radius * jnp.stack([jnp.cos(angle), jnp.sin(angle)])


def _norm_value(power: int, _context: CallContext, value, other=None):
    import jax.numpy as jnp

    vector = value if other is None else value - other
    squared = jnp.sum(jnp.square(vector))
    return squared if power == 2 else jnp.sqrt(squared)


def _huber(_context: CallContext, value, delta):
    import jax.numpy as jnp

    absolute = jnp.abs(value)
    return jnp.where(absolute <= delta, 0.5 * jnp.square(value), delta * (absolute - 0.5 * delta))


def _hold(_context: CallContext, value):
    return value


def _interpolate(minimum_jerk: bool, context: CallContext, start, stop, *, over=None):
    import jax.numpy as jnp

    duration = context.axis["T"] if over is None else over
    fraction = jnp.clip(context.axis["t"] / duration, 0, 1)
    if minimum_jerk:
        fraction = fraction**3 * (10 + fraction * (-15 + 6 * fraction))
    return start + (stop - start) * fraction


def _sample(family: str, context: CallContext, *args):
    import jax.numpy as jnp
    import jax.random as random

    key = context.key
    match family:
        case "uniform" if context.discrete_uniform:
            low, high = (jnp.rint(arg / context.period_seconds).astype(int) for arg in args)
            return random.randint(key, (), low, high + 1) * context.period_seconds
        case "uniform" | "loguniform":
            lo, hi = args
            shape = jnp.broadcast_shapes(jnp.shape(lo), jnp.shape(hi))
            if family == "uniform":
                return random.uniform(key, shape, minval=lo, maxval=hi)
            return jnp.exp(random.uniform(key, shape, minval=jnp.log(lo), maxval=jnp.log(hi)))
        case "normal":
            center, sd = args
            return center + sd * random.normal(
                key, jnp.broadcast_shapes(jnp.shape(center), jnp.shape(sd))
            )
        case "bernoulli":
            return random.bernoulli(key, args[0])
        case "choice":
            return jnp.stack(args)[random.randint(key, (), 0, len(args))]
        case "truncexp":
            mean, lo, hi = args
            shape = jnp.broadcast_shapes(*(jnp.shape(arg) for arg in args))
            width = -jnp.expm1(-(hi - lo) / mean)
            return lo - mean * jnp.log1p(-random.uniform(key, shape) * width)
        case "uniform_region":
            center, radius = args
            radial_key, angular_key = random.split(key)
            angle = 2 * jnp.pi * random.uniform(angular_key)
            return center + _polar_value(
                context, radius * jnp.sqrt(random.uniform(radial_key)), angle
            )
    raise AssertionError(family)


def _choose(context: CallContext, condition, yes, no):
    import jax.numpy as jnp

    test_defined, yes_defined, no_defined = context.defined
    return Evaluation(
        jnp.where(condition, yes, no),
        test_defined & jnp.all(jnp.where(condition, yes_defined, no_defined)),
    )


def _held_value(context: CallContext, condition, duration):
    import jax.numpy as jnp

    condition = condition & context.defined[0] & context.defined[1]
    start = jnp.where(condition, jnp.where(context.state < 0, context.step, context.state), -1)
    elapsed = (context.step - start) * context.period_seconds
    return condition & (elapsed >= duration), start


def _filtered(ou: bool, context: CallContext, value, tau):
    import jax.numpy as jnp
    import jax.random as random

    previous = jnp.broadcast_to(context.state, value.shape)
    mix = -jnp.expm1(-context.period_seconds / tau)
    if ou:
        result = (1 - mix) * previous + value * jnp.sqrt(mix * (2 - mix)) * random.normal(
            context.key, value.shape
        )
    else:
        result = previous + mix * (value - previous)
    return result, jnp.where(context.defined[0] & context.defined[1], result, previous)


def _difference_type(order, derivative, args, _kwargs):
    (value,) = args
    if value.base == _BOOL or len(value.shape) > 1:
        return _type("A backward difference needs a numeric scalar or vector.")
    if not derivative:
        return value
    return ValueType(
        base=_REAL, shape=value.shape, dimensions=value.dimensions.combine(TIME, -order)
    )


def _difference_state(order, value):
    import jax.numpy as jnp

    return (
        jnp.zeros((order, *value.shape), dtype=int if value.base == _INT else float),
        jnp.zeros(order, dtype=bool),
    )


def _difference(order, derivative, context, value):
    import jax.lax as lax
    import jax.numpy as jnp

    history, present = context.state

    def subtract(current, predecessor):
        value, defined = current
        previous, available = predecessor
        return (value - previous, defined & available), current

    current = (jnp.asarray(value, dtype=history.dtype), context.defined[0])
    (value, defined), history = lax.scan(subtract, current, (history, present))
    if derivative:
        value = value / context.period_seconds**order
    return Evaluation(value, defined), history


def _difference_context(order, derivative, expected, _known, _keywords):
    dimensions = expected.dimensions
    if derivative and dimensions is not None:
        dimensions = dimensions.combine(TIME, order)
    return (OperandHint(dimensions, expected.base),), {}


def _difference_dimensions(order, derivative, known):
    (dimension,) = known
    if derivative and dimension is not None:
        return dimension.combine(TIME, -order)
    return dimension


def _bind_difference(derivative, function, node):
    if len(node.args) == 2 and "n" in node.kwargs:
        return SignatureError(
            ProblemCode.ARITY, "Difference order was supplied twice.", ("kwargs", "n")
        )
    order_path = ("args", 1) if len(node.args) == 2 else ("kwargs", "n")
    size = node.args[1] if len(node.args) == 2 else node.kwargs.get("n", LiteralValue(value=1))
    if (
        not isinstance(size, LiteralValue)
        or type(size.value) is not int
        or size.value < 1
        or size.unit is not None
    ):
        return SignatureError(
            ProblemCode.STRUCTURE,
            "Difference order must be a positive integer literal.",
            order_path,
        )
    order = size.value
    return replace(
        function,
        infer=partial(_difference_type, order, derivative),
        evaluate=partial(_difference, order, derivative),
        initial_state=partial(_difference_state, order),
        argument_context=partial(_difference_context, order, derivative),
        result_dimensions=partial(_difference_dimensions, order, derivative),
    ), node.model_copy(update={"args": node.args[:1], "kwargs": {}})


def _pulse_type(args, kwargs):
    low, high, at, width = args
    durations = (at, width, *kwargs.values())
    for duration in durations:
        if error := _duration(duration):
            return error
    return _elementwise((low, high), {}, boolean=low.base == _BOOL)


def _pulse(context, low, high, at, width, *, every=None):
    import jax.numpy as jnp

    def steps(seconds):
        return jnp.rint(seconds / context.period_seconds)

    offset = steps(context.axis["t"]) - steps(at)
    local = offset
    rounded_width = steps(width)
    valid = (width == 0) | (rounded_width > 0)
    if every is not None:
        interval = steps(every)
        valid = valid & (interval > 0)
        local = jnp.mod(offset, jnp.maximum(interval, 1))
    high_now = (offset >= 0) & (local < rounded_width)
    return Evaluation(jnp.where(high_now, high, low), valid & _all_defined(context))


def _points_type(mode, args, kwargs):
    at, values = args
    if len(at.shape) != 1 or at.base == _BOOL or at.dimensions != DIMENSIONLESS:
        return _type("Point positions must be a vector of dimensionless numbers.")
    if not values.shape:
        return _type("Point values must be a list.")
    if at.shape[0] != values.shape[0]:
        return SignatureError(
            ProblemCode.LIST_COUNT, "Point positions and values need the same count."
        )
    if "over" in kwargs and (error := _duration(kwargs["over"])):
        return error
    if mode != "hold" and values.base == _BOOL:
        return _type("Interpolated point values must be numeric.")
    return ValueType(
        base=values.base if mode == "hold" else _REAL,
        shape=values.shape[1:],
        dimensions=values.dimensions,
    )


POINT_INTERPOLATIONS = ("linear", "hold", "cubic")


def _bind_points(function, node):
    mode = node.kwargs.get("interp", CallValue(function="linear"))
    if (
        not isinstance(mode, CallValue)
        or mode.function not in POINT_INTERPOLATIONS
        or mode.args
        or mode.kwargs
    ):
        return SignatureError(
            ProblemCode.STRUCTURE,
            "Point interpolation must be linear, hold, or cubic.",
            ("kwargs", "interp"),
        )
    positions = node.args[0]
    if (
        isinstance(positions, CallValue)
        and positions.function == "vector"
        and all(
            isinstance(item, LiteralValue) and item.unit is None and type(item.value) is not bool
            for item in positions.args
        )
    ):
        values = [item.value for item in positions.args]
        if any(left >= right for left, right in zip(values, values[1:])):
            return SignatureError(
                ProblemCode.STRUCTURE, "Point positions must be strictly increasing.", ("args", 0)
            )
    return replace(
        function,
        infer=partial(_points_type, mode.function),
        evaluate=partial(_points, mode.function),
    ), node.model_copy(
        update={"kwargs": {key: value for key, value in node.kwargs.items() if key != "interp"}}
    )


def _all_defined(context):
    valid = True
    for available in context.defined:
        valid = valid & available
    return valid


def _points(mode, context, at, values, *, over=None):
    import jax.numpy as jnp

    from feedbax._interpolation import catmull_rom

    duration = context.axis["T"] if over is None else over
    u = jnp.clip(context.axis["t"] / duration, at[0], at[-1])
    count = at.shape[0]
    valid = (duration > 0) & jnp.all(at[1:] > at[:-1]) & _all_defined(context)
    if count == 1:
        return Evaluation(values[0], valid)
    left = jnp.searchsorted(at, u, side="right") - 1
    if mode == "hold":
        return Evaluation(values[left], valid)
    left = jnp.minimum(left, count - 2)
    right = left + 1
    interval = at[right] - at[left]
    fraction = (u - at[left]) / jnp.where(interval > 0, interval, 1)
    start, stop = values[left], values[right]
    if mode == "linear":
        value = start + (stop - start) * fraction
    else:
        value = catmull_rom(
            values[jnp.maximum(left - 1, 0)],
            start,
            stop,
            values[jnp.minimum(right + 1, count - 1)],
            fraction,
        )
    return Evaluation(value, valid)


def _sines_type(args, _kwargs):
    center, amp, frequencies, phases = args
    if any(arg.base == _BOOL for arg in args):
        return _type("Sines requires numeric arguments.")
    if len(frequencies.shape) != 1 or amp.shape not in ((), center.shape):
        return _type("Sines needs a frequency vector and an amplitude compatible with the centre.")
    if phases.shape not in (frequencies.shape, frequencies.shape + center.shape):
        return _type("Phase must have one value or one array per frequency.")
    if (
        amp.dimensions != center.dimensions
        or frequencies.dimensions != TIME.raised(-1)
        or phases.dimensions != ANGLE
    ):
        return _unit(
            "Amplitude must match the centre's units, with frequencies in Hz and phases as angles."
        )
    return ValueType(base=_REAL, shape=center.shape, dimensions=center.dimensions)


def _sines(context, center, amp, frequencies, phases):
    import jax.numpy as jnp

    phase = 2 * jnp.pi * frequencies * context.axis["t"]
    phase = phase.reshape(phase.shape + (1,) * (phases.ndim - 1))
    return center + amp * jnp.sum(jnp.sin(phase + phases), axis=0)


def _noise(context, mean, sd):
    import jax.numpy as jnp
    import jax.random as random

    shape = jnp.broadcast_shapes(mean.shape, sd.shape)
    return mean + sd * random.normal(context.key, shape)


def _filter_state(value: ValueType):
    import jax.numpy as jnp

    return jnp.zeros(value.shape, dtype=float)


def _huber_type(
    args: tuple[ValueType, ...], kwargs: Mapping[str, ValueType]
) -> ValueType | SignatureError:
    result = _elementwise(args, kwargs, base=_REAL)
    if isinstance(result, SignatureError):
        return result
    return result.model_copy(update={"dimensions": result.dimensions.raised(2)})


def _operand_context(
    expected: OperandHint,
    known: tuple[Dimensions | None, ...],
    keywords: tuple[str, ...],
    *,
    start: int = 0,
    stop: int | None = None,
    fixed: Mapping[int, OperandHint] | None = None,
    base: ValueBase | None = None,
    dimensionless: bool = False,
):
    indices = range(start, len(known) if stop is None else stop)
    anchor = (
        DIMENSIONLESS
        if dimensionless
        else next(
            (known[i] for i in indices if known[i] is not None),
            expected.dimensions,
        )
    )
    hints = tuple(
        OperandHint(anchor, base or expected.base) if i in indices else OperandHint()
        for i in range(len(known))
    )
    return tuple((fixed or {}).get(i, hint) for i, hint in enumerate(hints)), {
        key: OperandHint(TIME, _REAL) for key in keywords if key in ("over", "every")
    }


def _arithmetic_context(expected: OperandHint, known, _keywords):
    left = expected.dimensions if all(unit is None for unit in known) else None
    return (OperandHint(left, expected.base), OperandHint(base=expected.base)), {}


def _known_result(known, *, start=0, stop=None, power=1):
    selected = next((unit for unit in known[start:stop] if unit is not None), None)
    return selected.raised(power) if selected is not None else None


def _known_arithmetic(known, *, power):
    if all(unit is None for unit in known):
        return None
    left, right = (unit or DIMENSIONLESS for unit in known)
    return left.combine(right, power)


def register_builtins(registry: FunctionRegistry) -> None:
    """Register one typed implementation per supported function."""

    def add(
        name,
        signature,
        count,
        infer,
        evaluate,
        *,
        display_class=FunctionClass.MATH,
        keywords=(),
        implicit_names=None,
        level=Level.RUN,
        random=False,
        initial_state=None,
        requires_period=False,
        argument_context=_operand_context,
        result_dimensions=_known_result,
        reset=StateReset.TRIAL,
        bind=None,
        positional_names=(),
    ):
        minimum, maximum = count if isinstance(count, tuple) else (count, count)
        registry.register(
            Function(
                FunctionInfo(
                    id=name,
                    signature=f"{name}({', '.join(positional_names) if positional_names else signature})",
                    display_class=display_class,
                    min_args=minimum,
                    max_args=maximum,
                    keywords=keywords,
                    implicit_names=implicit_names or {},
                    level=level,
                    random=random,
                    stateful=initial_state is not None,
                    requires_period=requires_period,
                    reset=reset,
                ),
                infer,
                evaluate,
                initial_state,
                argument_context,
                result_dimensions,
                bind,
                positional_names,
            )
        )

    for name, operator, dimensions, output in (
        ("add", "add", "same", None),
        ("sub", "subtract", "same", None),
        ("mul", "multiply", "product", None),
        ("div", "true_divide", "quotient", _REAL),
        ("lt", "less", "same", _BOOL),
        ("le", "less_equal", "same", _BOOL),
        ("gt", "greater", "same", _BOOL),
        ("ge", "greater_equal", "same", _BOOL),
        ("min", "minimum", "same", None),
        ("max", "maximum", "same", None),
    ):
        add(
            name,
            "x, y",
            2,
            partial(_elementwise, base=output, dimensions=dimensions),
            partial(_operation, operator),
            argument_context=_arithmetic_context
            if dimensions in ("product", "quotient")
            else _operand_context,
            result_dimensions=(
                partial(_known_arithmetic, power=1 if dimensions == "product" else -1)
                if dimensions in ("product", "quotient")
                else (lambda _: DIMENSIONLESS)
                if output == _BOOL
                else _known_result
            ),
        )
    for name, operator in (("eq", "equal"), ("ne", "not_equal")):
        add(
            name,
            "x, y",
            2,
            _equality,
            partial(_operation, operator),
            result_dimensions=lambda _: DIMENSIONLESS,
        )
    for name, operator, count in (
        ("and", "logical_and", 2),
        ("or", "logical_or", 2),
        ("not", "logical_not", 1),
    ):
        add(
            name,
            "condition" if count == 1 else "left, right",
            count,
            partial(_elementwise, boolean=True),
            partial(_operation, operator),
            argument_context=partial(_operand_context, dimensionless=True),
            result_dimensions=lambda _: DIMENSIONLESS,
        )
    for name, operator in (("neg", "negative"), ("pos", "positive"), ("abs", "abs")):
        add(
            name,
            "x",
            1,
            _elementwise,
            partial(_operation, operator),
            display_class=_PENALTY if name == "abs" else FunctionClass.MATH,
        )
    add("clip", "x, low, high", 3, _elementwise, partial(_operation, "clip"))
    add(
        "where",
        "condition, yes, no",
        3,
        _where,
        _choose,
        argument_context=partial(_operand_context, start=1),
        result_dimensions=partial(_known_result, start=1),
    )
    add("vector", "values…", (1, None), _vector, _stack)
    add(
        "index",
        "array, index",
        2,
        _index,
        _at,
        argument_context=partial(
            _operand_context, stop=1, fixed={1: OperandHint(DIMENSIONLESS, _INT)}
        ),
        result_dimensions=partial(_known_result, stop=1),
    )
    add(
        "polar",
        "radius, angle",
        2,
        _polar,
        _polar_value,
        argument_context=partial(
            _operand_context, stop=1, base=_REAL, fixed={1: OperandHint(ANGLE, _REAL)}
        ),
        result_dimensions=partial(_known_result, stop=1),
    )
    for name, power, count in (("norm", 1, 1), ("sqnorm", 2, 1), ("dist", 1, 2)):
        add(
            name,
            "x" if count == 1 else "x, y",
            count,
            partial(_norm, power=power),
            partial(_norm_value, power),
            display_class=_PENALTY if count == 1 else FunctionClass.MATH,
            argument_context=partial(_operand_context, base=_REAL),
            result_dimensions=partial(_known_result, power=power),
        )
    add(
        "huber",
        "x, delta",
        2,
        _huber_type,
        _huber,
        display_class=_PENALTY,
        argument_context=partial(_operand_context, base=_REAL),
        result_dimensions=partial(_known_result, power=2),
    )
    for name in ("sigmoid", "softplus"):

        def nonlinear(context, value, *, name=name):
            from jax import nn

            return getattr(nn, name)(value)

        add(
            name,
            "x",
            1,
            partial(_elementwise, base=_REAL, dimensions="dimensionless"),
            nonlinear,
            argument_context=partial(_operand_context, base=_REAL, dimensionless=True),
            result_dimensions=lambda _: DIMENSIONLESS,
        )
    for name in ("diff", "ddt"):
        derivative = name == "ddt"
        add(
            name,
            "x, n=1",
            (1, 2),
            partial(_difference_type, 1, derivative),
            partial(_difference, 1, derivative),
            keywords=("n",),
            level=Level.STEP,
            initial_state=partial(_difference_state, 1),
            requires_period=derivative,
            bind=partial(_bind_difference, derivative),
        )
    time = ValueType(base=_REAL, dimensions=TIME)
    add(
        "pulse",
        "low, high, at, width, every=none",
        4,
        _pulse_type,
        _pulse,
        keywords=("every",),
        implicit_names={"t": time},
        level=Level.STEP,
        display_class=_SHAPE,
        requires_period=True,
        result_dimensions=partial(_known_result, stop=2),
        argument_context=partial(
            _operand_context,
            stop=2,
            fixed={2: OperandHint(TIME, _REAL), 3: OperandHint(TIME, _REAL)},
        ),
    )
    add(
        "points",
        "at, values, interp=linear, over=T",
        2,
        partial(_points_type, "linear"),
        partial(_points, "linear"),
        keywords=("interp", "over"),
        implicit_names={"t": time, "T": time},
        display_class=_SHAPE,
        level=Level.STEP,
        bind=_bind_points,
        argument_context=partial(
            _operand_context, start=1, fixed={0: OperandHint(DIMENSIONLESS, _REAL)}
        ),
        result_dimensions=partial(_known_result, start=1),
    )
    add(
        "sines",
        "center, amp, frequencies, phases",
        4,
        _sines_type,
        _sines,
        implicit_names={"t": time},
        display_class=_SHAPE,
        level=Level.STEP,
        argument_context=partial(
            _operand_context,
            stop=2,
            base=_REAL,
            fixed={2: OperandHint(TIME.raised(-1), _REAL), 3: OperandHint(ANGLE, _REAL)},
        ),
        result_dimensions=partial(_known_result, stop=2),
    )
    add(
        "noise",
        "mean, sd",
        2,
        partial(_elementwise, base=_REAL),
        _noise,
        display_class=_SHAPE,
        level=Level.STEP,
        random=True,
        argument_context=partial(_operand_context, base=_REAL),
    )
    add("hold", "value", 1, lambda args, _: args[0], _hold, display_class=_SHAPE, level=Level.STEP)
    for name in ("ramp", "minjerk"):
        add(
            name,
            "start, stop, over=T",
            2,
            _endpoints,
            partial(_interpolate, name == "minjerk"),
            display_class=_SHAPE,
            keywords=("over",),
            implicit_names={name: ValueType(base=_REAL, dimensions=TIME) for name in ("t", "T")},
            level=Level.STEP,
            argument_context=partial(_operand_context, base=_REAL),
        )
    for family, names, infer in (
        ("uniform", ("low", "high"), partial(_elementwise, base=_REAL)),
        ("loguniform", ("low", "high"), partial(_elementwise, base=_REAL)),
        ("normal", ("mean", "sd"), partial(_elementwise, base=_REAL)),
        ("bernoulli", ("probability",), _bernoulli),
        ("choice", (), _choice),
        ("truncexp", ("mean", "low", "high"), partial(_elementwise, base=_REAL)),
        ("uniform_region", ("centre", "radius"), _region),
    ):
        add(
            family,
            "values…",
            len(names) if names else (1, None),
            infer,
            partial(_sample, family),
            display_class=_DRAW,
            positional_names=names,
            random=True,
            argument_context=(
                partial(_operand_context, base=_REAL, dimensionless=True)
                if family == "bernoulli"
                else _operand_context
                if family == "choice"
                else partial(_operand_context, base=_REAL)
            ),
            result_dimensions=(lambda _: DIMENSIONLESS) if family == "bernoulli" else _known_result,
        )
    add(
        "held",
        "condition, duration",
        2,
        _held,
        _held_value,
        level=Level.STEP,
        initial_state=lambda _: -1,
        requires_period=True,
        reset=StateReset.EPOCH,
        argument_context=partial(_operand_context, start=2, fixed={1: OperandHint(TIME, _REAL)}),
        result_dimensions=lambda _: DIMENSIONLESS,
    )
    for name, value in (("lowpass", "x"), ("ou", "sd")):
        add(
            name,
            f"{value}, tau",
            2,
            _filter,
            partial(_filtered, name == "ou"),
            level=Level.STEP,
            random=name == "ou",
            reset=StateReset.EPOCH if name == "ou" else StateReset.TRIAL,
            initial_state=_filter_state,
            requires_period=True,
            argument_context=partial(
                _operand_context, stop=1, base=_REAL, fixed={1: OperandHint(TIME, _REAL)}
            ),
            result_dimensions=partial(_known_result, stop=1),
        )
