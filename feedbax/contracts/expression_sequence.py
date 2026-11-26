"""Structural integer expressions and shared inclusive range evaluation."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from fractions import Fraction
import math
import operator

from feedbax.contracts.expression_values import (
    CallValue, Expression, LiteralValue, NameValue, SequenceValue, normal_literal_number,
)
from feedbax.contracts.expression_units import UNITS


_INTEGER_OPERATORS = {"add": operator.add, "sub": operator.sub, "mul": operator.mul, "mod": operator.mod}


def structural_integer(value: Expression, indices: Mapping[str, int] | None = None) -> int:
    """Read an integer expression composed of literals, family indices and integer operations."""
    match value:
        case LiteralValue(value=number, unit=None) if type(number) is int:
            return number
        case NameValue(reference=name) if indices is not None and type(number := indices.get(name)) is int:
            return number
        case CallValue(function=("pos" | "neg") as sign, args=(child,), kwargs=kwargs) if not kwargs:
            number = structural_integer(child, indices)
            return -number if sign == "neg" else number
        case CallValue(function=key, args=(left, right), kwargs=kwargs) if key in _INTEGER_OPERATORS and not kwargs:
            a, b = structural_integer(left, indices), structural_integer(right, indices)
            if key == "mod" and b == 0:
                raise ValueError("An integer expression cannot take a remainder by zero.")
            return _INTEGER_OPERATORS[key](a, b)
        case _:
            raise ValueError("Expected a whole number or an integer expression of family indices.")


def range_refusal(args: tuple[Expression, ...]) -> str:
    """Suggest an inclusive sequence for an unrecognized range call."""
    try:
        end = structural_integer(args[0]) - 1 if len(args) == 1 else -1
    except ValueError:
        end = -1
    return f"Expression function 'range' is unavailable. Write an inclusive sequence such as 0..{end if end >= 0 else 5}."


def real_range(first: float, last: float, count: int, *, logarithmic: bool) -> Iterator[float]:
    """The real members shared by an expression sequence and a sweep range."""
    if not math.isfinite(first) or not math.isfinite(last):
        raise ValueError("range bounds must be finite")
    yield first
    if count == 1:
        return
    if logarithmic:
        start, stop = math.log10(first), math.log10(last)
        for index in range(1, count - 1):
            yield 10 ** (start + (stop - start) * index / (count - 1))
    else:
        for index in range(1, count - 1):
            fraction = index / (count - 1)
            yield first * (1 - fraction) + last * fraction
    yield last


@dataclass(frozen=True)
class SequenceDefinition:
    rule: str
    first: int | float
    last: int | float
    count: int
    unit: str | None

    @property
    def whole(self) -> bool:
        """Whether the rule's members are whole without generating them."""
        first, last = normal_literal_number(self.first), normal_literal_number(self.last)
        if self.count == 1:
            return type(first) is int
        if type(first) is not int or type(last) is not int:
            return False
        if self.rule == "span":
            return True
        steps = self.count - 1
        if self.rule == "linspace":
            return (last - first) % steps == 0
        # In lowest terms, the endpoint ratio must be a power of a rational step ratio.
        ratio = Fraction(last, first)
        return _is_power(ratio.numerator, steps) and _is_power(ratio.denominator, steps)

    def numbers(self) -> Iterator[int | float]:
        """Generate normalized magnitudes without allocating expression nodes."""
        if self.rule == "span":
            first, last = self.first, self.last
            assert type(first) is int and type(last) is int
            values = range(first, last + 1)
        else:
            values = real_range(float(self.first), float(self.last), self.count, logarithmic=self.rule == "geomspace")
        for value in values:
            value = normal_literal_number(value)
            if self.unit == "st" and type(value) is not int:
                raise ValueError("A sequence member in steps must be a whole number.")
            yield value

    def members(self) -> Iterator[LiteralValue]:
        """Evaluate the rule once, keeping every bound and the declared unit."""
        return (LiteralValue(value=value, unit=self.unit) for value in self.numbers())


def _is_power(number: int, exponent: int) -> bool:
    """An exact integer power test bounded by the size of the number."""
    if number == 1 or exponent == 1:
        return True
    bits = number.bit_length()
    if exponent >= bits:
        return False
    low, high = 2, 1 << ((bits + exponent - 1) // exponent)
    while low <= high:
        middle = (low + high) // 2
        power = middle ** exponent
        if power == number:
            return True
        if power < number:
            low = middle + 1
        else:
            high = middle - 1
    return False


def _integer_bound(node: Expression, indices: Mapping[str, int] | None) -> tuple[int, str | None]:
    if isinstance(node, LiteralValue) and type(node.value) is int:
        return node.value, node.unit
    return structural_integer(node, indices), None


def _real_bound(node: Expression) -> tuple[int | float, str | None]:
    if not isinstance(node, LiteralValue) or type(node.value) not in (int, float):
        raise ValueError("Range bounds must be numeric literals.")
    try:
        finite = math.isfinite(node.value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError("Range bounds must be finite.")
    return node.value, node.unit


def sequence_definition(value: SequenceValue, indices: Mapping[str, int] | None = None) -> SequenceDefinition:
    """Resolve a rule's bounds and count without expanding its members."""
    if value.rule == "span":
        if value.count is not None:
            raise ValueError("An inclusive span takes two bounds.")
        first, unit = _integer_bound(value.first, indices)
        last, end_unit = _integer_bound(value.last, indices)
        if first > last:
            raise ValueError("The first span bound must be at most the last.")
        count = last - first + 1
    else:
        if value.count is None or (count := structural_integer(value.count, indices)) < 1:
            raise ValueError("Range count must be a positive whole number.")
        first, unit = _real_bound(value.first)
        last, end_unit = _real_bound(value.last)
        if value.rule == "geomspace" and min(first, last) <= 0:
            raise ValueError("Geometric range bounds must be positive.")
    if unit != end_unit:
        raise ValueError("Sequence bounds must have the same unit.")
    if unit is not None and unit not in UNITS:
        raise ValueError(f"Invalid sequence unit {unit!r}.")
    return SequenceDefinition(value.rule, first, last, count, unit)
