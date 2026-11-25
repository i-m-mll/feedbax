"""Authored time values and their projection onto a single Model clock."""

from math import isfinite
from typing import Annotated, Literal

from pydantic import Field

from feedbax.contracts.base import StrictModel

TimeUnit = Literal["s", "st"]


class TimeValue(StrictModel):
    """A time in the authored unit. An offset may be zero or negative."""

    value: Annotated[float, Field(allow_inf_nan=False)]
    unit: TimeUnit


def time_in_seconds(time: TimeValue, step_seconds: float) -> float:
    """Convert a time to seconds without rounding a physical time constant."""
    value = time.value * step_seconds if time.unit == "st" else time.value
    if not isfinite(value):
        raise ValueError("Time in seconds must be finite.")
    return value


def time_steps(time: TimeValue, step_seconds: float) -> int:
    """Pair a sampled time with the Model clock. Seconds round to the nearest step."""
    count = time.value / step_seconds if time.unit == "s" else time.value
    if not isfinite(count):
        raise ValueError("Time must resolve to a finite step count.")
    steps = round(count)
    if time.unit == "st" and count != steps:
        raise ValueError("Time given in steps must be a whole number.")
    return steps


def length_steps(time: TimeValue, step_seconds: float) -> int:
    """Pair a length with the Model clock, preserving zero and refusing a vanished length."""
    steps = time_steps(time, step_seconds)
    if time.value < 0 or (time.value != 0 and steps == 0):
        raise ValueError("Length must be zero or round to at least one step.")
    return steps
