"""Authored time values and their projection onto a single Model clock."""

from fractions import Fraction
from math import isfinite
from typing import Annotated, Literal

from pydantic import Field

from feedbax.contracts.base import StrictModel
from feedbax.contracts.expression_units import UNITS
from feedbax.contracts.expression_values import LiteralValue

TimeUnit = Literal["s", "st"]
TimeUse = Literal["constant", "length", "offset"]


class TimeValue(StrictModel):
    """A time in the authored unit. An offset may be zero or negative."""

    value: Annotated[float, Field(allow_inf_nan=False)]
    unit: TimeUnit


def literal_time(value: LiteralValue) -> TimeValue:
    """Normalize a checked scalar duration, including authored units such as ms."""
    if value.unit == "st":
        return TimeValue(value=value.value, unit="st")
    scale = UNITS[value.unit].scale if value.unit is not None else 1
    return TimeValue(value=value.value * scale, unit="s")


def time_in_seconds(time: TimeValue, step_seconds: float) -> float:
    """Convert a time to seconds without rounding a physical time constant."""
    value = time.value * step_seconds if time.unit == "st" else time.value
    if not isfinite(value):
        raise ValueError("Time in seconds must be finite.")
    return value


def time_steps(time: TimeValue, step_seconds: float) -> int:
    """Pair a sampled time with the Model clock. Seconds round to the nearest step."""
    magnitude = Fraction(str(time.value))
    if time.unit == "st":
        if magnitude.denominator != 1:
            raise ValueError("Time given in steps must be a whole number.")
        return magnitude.numerator
    count = magnitude / Fraction(str(step_seconds))
    steps, remainder = divmod(abs(count.numerator), count.denominator)
    steps += 2 * remainder >= count.denominator
    return -steps if count < 0 else steps


def length_steps(time: TimeValue, step_seconds: float) -> int:
    """Pair a length with the Model clock, preserving zero and refusing a vanished length."""
    steps = time_steps(time, step_seconds)
    if time.value < 0 or (time.value != 0 and steps == 0):
        raise ValueError("Length must be zero or round to at least one step.")
    return steps


def paired_time(time: TimeValue, step_seconds: float, use: TimeUse) -> TimeValue:
    """Show the other unit, rounding sampled lengths and keeping physical constants exact."""
    if use != "constant":
        steps = length_steps(time, step_seconds) if use == "length" else time_steps(time, step_seconds)
        if time.unit == "s":
            return TimeValue(value=steps, unit="st")
    if time.unit == "s":
        return TimeValue(value=time.value / step_seconds, unit="st")
    return TimeValue(value=time_in_seconds(time, step_seconds), unit="s")
