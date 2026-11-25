"""Authored selections on the trial's post-step sample axis."""

from typing import Annotated, Literal

from pydantic import Field

from feedbax.contracts.base import StrictModel
from feedbax.contracts.expression_units import UNITS
from feedbax.contracts.expression_values import (
    TIME, CallValue, DrawValue, Expression, Level, LiteralValue, NameValue, ScopedValue, Site,
    UnresolvedValue, ValueBase, ValueType,
)
from feedbax.contracts.time_units import TimeValue, literal_time

WindowEdge = Literal["start", "end"]
WindowOffset = TimeValue | Annotated[
    CallValue | DrawValue | NameValue | UnresolvedValue, Field(discriminator="kind")
]


class WindowEpoch(StrictModel):
    """The samples in every instance of an epoch, or one zero-based occurrence."""

    kind: Literal["epoch"]
    epoch: str
    occurrence: int | None = Field(default=None, ge=0)


class WindowTrialAnchor(StrictModel):
    kind: Literal["trial"]
    edge: WindowEdge


class WindowEpochAnchor(WindowEpoch):
    edge: WindowEdge


WindowAnchor = Annotated[WindowTrialAnchor | WindowEpochAnchor, Field(discriminator="kind")]


class WindowPosition(StrictModel):
    anchor: WindowAnchor
    offset: WindowOffset | None = None


class WindowRelativePosition(StrictModel):
    offset: WindowOffset


class WindowInterval(StrictModel):
    kind: Literal["interval"]
    start: WindowPosition
    end: WindowPosition | WindowRelativePosition


class WindowMoment(StrictModel):
    kind: Literal["moment"]
    at: WindowPosition


WindowSpan = Annotated[WindowEpoch | WindowInterval | WindowMoment, Field(discriminator="kind")]


class Window(StrictModel):
    """The union of half-open intervals, whole epochs and individual samples."""

    spans: tuple[WindowSpan, ...]


def window_offset_site(names: tuple[ScopedValue, ...]) -> Site:
    """An offset is a scalar time known before the rollout starts."""
    return Site(
        expected=ValueType(base=ValueBase.REAL, dimensions=TIME),
        finest=Level.TRIAL,
        names=tuple(name for name in names if name.level <= Level.TRIAL),
    )


def window_offset_sites(window: Window | None, names: tuple[ScopedValue, ...]) -> dict[tuple[str, ...], Site]:
    """The editable offsets, including implicit zero offsets on existing anchors."""
    if window is None:
        return {}
    site = window_offset_site(names)
    return {
        ("spans", str(index), part, "offset"): site
        for index, span in enumerate(window.spans)
        for part in (("start", "end") if isinstance(span, WindowInterval)
                     else ("at",) if isinstance(span, WindowMoment) else ())
    }


def window_offset_expression(offset: WindowOffset | None) -> Expression:
    """Read a Window offset in the expression field grammar."""
    if offset is None:
        return LiteralValue(value=0, unit="st")
    if isinstance(offset, TimeValue):
        return LiteralValue(value=offset.value, unit=offset.unit)
    return offset


def store_window_offset(expression: Expression) -> WindowOffset:
    """Store a numeric expression as a time value and keep other expressions authored."""
    if not isinstance(expression, LiteralValue):
        return expression
    unit = expression.unit or "s"
    if type(expression.value) not in (float, int) or unit not in UNITS or UNITS[unit].dimensions != TIME:
        raise ValueError("A Window offset requires a time in seconds or steps.")
    return literal_time(expression)
