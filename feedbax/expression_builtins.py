"""The first expression function declarations and their numerical operations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any

from feedbax.contracts.expression_values import (
    ANGLE,
    DIMENSIONLESS,
    TIME,
    Dimensions,
    Level,
    ProblemCode,
    ValueBase,
    ValueType,
)
from feedbax.expression_registry import (
    Function,
    FunctionClass,
    FunctionInfo,
    FunctionRegistry,
    OperandHint,
    SignatureError,
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


def _held_value(context: CallContext, condition, duration):
    import jax.numpy as jnp

    start = jnp.where(
        condition, jnp.where(context.state < 0, context.axis["t"], context.state), -1.0
    )
    return condition & (context.axis["t"] >= start + duration), start


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
    return result, result


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
        key: OperandHint(TIME, _REAL) for key in keywords if key == "over"
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
    ):
        minimum, maximum = count if isinstance(count, tuple) else (count, count)
        registry.register(
            Function(
                FunctionInfo(
                    id=name,
                    signature=f"{name}({signature})",
                    display_class=display_class,
                    min_args=minimum,
                    max_args=maximum,
                    keywords=keywords,
                    implicit_names=implicit_names or {},
                    level=level,
                    random=random,
                    stateful=initial_state is not None,
                    requires_period=requires_period,
                ),
                infer,
                evaluate,
                initial_state,
                argument_context,
                result_dimensions,
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
        partial(_operation, "where"),
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
    for family, signature, count, infer in (
        ("uniform", "low, high", 2, partial(_elementwise, base=_REAL)),
        ("loguniform", "low, high", 2, partial(_elementwise, base=_REAL)),
        ("normal", "mean, sd", 2, partial(_elementwise, base=_REAL)),
        ("bernoulli", "probability", 1, _bernoulli),
        ("choice", "values…", (1, None), _choice),
        ("truncexp", "mean, low, high", 3, partial(_elementwise, base=_REAL)),
        ("uniform_region", "centre, radius", 2, _region),
    ):
        add(
            family,
            signature,
            count,
            infer,
            partial(_sample, family),
            display_class=_DRAW,
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
        implicit_names={"t": ValueType(base=_REAL, dimensions=TIME)},
        level=Level.STEP,
        initial_state=lambda _: -1.0,
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
            initial_state=_filter_state,
            requires_period=True,
            argument_context=partial(
                _operand_context, stop=1, base=_REAL, fixed={1: OperandHint(TIME, _REAL)}
            ),
            result_dimensions=partial(_known_result, stop=1),
        )
