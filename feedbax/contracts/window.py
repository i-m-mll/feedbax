"""Authored selections on the trial's post-step sample axis."""

from typing import Annotated, Literal

from pydantic import Field

from feedbax.contracts.base import StrictModel
from feedbax.contracts.expression_values import CallValue, DrawValue, NameValue, UnresolvedValue
from feedbax.contracts.time_units import TimeValue

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
