"""Authored selections on the trial's post-step sample axis."""

from collections.abc import Mapping
from functools import partial
from typing import Annotated, Literal

from pydantic import Field

from feedbax.contracts.base import StrictModel
from feedbax.contracts.expression_storage import StoredExpression, lower_expression, store_expression
from feedbax.contracts.expression_text import print_field
from feedbax.contracts.expression_units import UNITS
from feedbax.contracts.expression_values import (
    TIME, CallValue, DrawValue, Expression, Level, LiteralValue, NameValue, ScopedValue, Site,
    UnresolvedValue, ValueBase, ValueType,
)
from feedbax.contracts.time_units import TimeValue, literal_time
from feedbax.contracts.stored_members import project_member

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


class _WindowPosition[OffsetT](StrictModel):
    anchor: WindowAnchor
    offset: OffsetT | None = None


class WindowPosition(_WindowPosition[WindowOffset]):
    pass


class StoredWindowPosition(_WindowPosition[StoredExpression]):
    pass


class _WindowRelativePosition[OffsetT](StrictModel):
    offset: OffsetT


class WindowRelativePosition(_WindowRelativePosition[WindowOffset]):
    pass


class StoredWindowRelativePosition(_WindowRelativePosition[StoredExpression]):
    pass


class _WindowInterval[PositionT, RelativeT](StrictModel):
    kind: Literal["interval"]
    start: PositionT
    end: PositionT | RelativeT


class WindowInterval(_WindowInterval[WindowPosition, WindowRelativePosition]):
    pass


class StoredWindowInterval(_WindowInterval[StoredWindowPosition, StoredWindowRelativePosition]):
    pass


class _WindowMoment[PositionT](StrictModel):
    kind: Literal["moment"]
    at: PositionT


class WindowMoment(_WindowMoment[WindowPosition]):
    pass


class StoredWindowMoment(_WindowMoment[StoredWindowPosition]):
    pass


WindowSpan = Annotated[WindowEpoch | WindowInterval | WindowMoment, Field(discriminator="kind")]
StoredWindowSpan = Annotated[WindowEpoch | StoredWindowInterval | StoredWindowMoment, Field(discriminator="kind")]


class Window(StrictModel):
    """The union of half-open intervals, whole epochs and individual samples."""

    spans: tuple[WindowSpan, ...]


class StoredWindow(StrictModel):
    """Window spans keep their order and store offsets in the expression field grammar."""

    spans: tuple[StoredWindowSpan, ...]

    def lower(self) -> Window:
        return Window(spans=tuple(_lower_span(span) for span in self.spans))

    @classmethod
    def from_lowered(cls, window: Window, *, scopes: Mapping[str, str] | None = None) -> "StoredWindow":
        offset = partial(stored_window_offset, scopes=scopes)
        return cls(spans=tuple(_stored_span(span, offset) for span in window.spans))


def lower_window_offset(value: StoredExpression | None) -> WindowOffset | None:
    """Lower a persisted offset with its runtime name and literal time types."""
    if value is None:
        return None
    expression = lower_expression(value)
    try:
        return store_window_offset(expression)
    except ValueError:
        return UnresolvedValue(text=print_field(expression, window_offset_site(())))


def stored_window_offset(
    value: WindowOffset | None, *, scopes: Mapping[str, str] | None = None,
) -> StoredExpression | None:
    """Project a runtime offset into its authored expression type."""
    return store_expression(window_offset_expression(value), scopes=scopes) if value is not None else None


def _lower_position(position: StoredWindowPosition) -> WindowPosition:
    return project_member(position, WindowPosition, {"offset": lower_window_offset})


def _lower_span(span: StoredWindowSpan) -> WindowSpan:
    if isinstance(span, StoredWindowInterval):
        end = _lower_position(span.end) if isinstance(span.end, StoredWindowPosition) else project_member(
            span.end, WindowRelativePosition, {"offset": lower_window_offset},
        )
        return project_member(span, WindowInterval, start=_lower_position(span.start), end=end)
    if isinstance(span, StoredWindowMoment):
        return project_member(span, WindowMoment, at=_lower_position(span.at))
    return span


def _stored_span(span: WindowSpan, offset) -> StoredWindowSpan:
    def position(value: WindowPosition) -> StoredWindowPosition:
        return project_member(value, StoredWindowPosition, {"offset": offset})

    if isinstance(span, WindowInterval):
        end = position(span.end) if isinstance(span.end, WindowPosition) else project_member(
            span.end, StoredWindowRelativePosition, {"offset": offset},
        )
        return project_member(span, StoredWindowInterval, start=position(span.start), end=end)
    if isinstance(span, WindowMoment):
        return project_member(span, StoredWindowMoment, at=position(span.at))
    return span


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
