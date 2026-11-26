"""The selection and summaries shared by recorded-trial views."""

from pydantic import Field, JsonValue

from feedbax.contracts.base import StrictModel
from feedbax.contracts.expression_values import Problem


class TrialMeanSelection(StrictModel):
    by: list[str] | None = None
    """Stable IDs of discrete Task variables. Null selects all, and [] pools all trials."""
    align: str | None = None
    """Stable ID of the epoch whose first actual start is zero. Null means trial start."""


class TrialClockChoice(StrictModel):
    id: str
    label: str


class TrialMeanOptions(StrictModel):
    by: list[TrialClockChoice]
    align: list[TrialClockChoice]


class TrialClockSpan(StrictModel):
    start: float
    end: float


class TrialClockAlignment(StrictModel):
    origin: float
    """Recorded time in seconds at the alignment origin."""
    span: TrialClockSpan
    """Extent of this trial in aligned seconds."""


class TrialSeriesSummary(StrictModel):
    """Pointwise summary in the original variable's units, with one row per time."""

    mean: list[list[float]]
    lower: list[list[float]]
    upper: list[list[float]]


class TrialBoundarySummary(StrictModel):
    mean: float
    lower: float
    upper: float


class TrialMeanEpoch(StrictModel):
    epoch: str
    label: str
    occurrence: int
    """Zero-based occurrence of this epoch within a trial."""
    start: TrialBoundarySummary
    end: TrialBoundarySummary


class TrialMeanCondition(StrictModel):
    values: dict[str, JsonValue]
    """Condition values keyed by selected Task variable ID."""
    trials: dict[str, TrialClockAlignment]
    """Participating trials by trial ID, with individual alignment and extent."""
    span: TrialClockSpan | None
    """Common extent in aligned seconds, if the participating trials overlap."""
    time: list[float]
    """Sample times in aligned seconds inside the common extent."""
    states: dict[str, TrialSeriesSummary]
    """World-frame path summaries keyed by Model state address."""
    epochs: list[TrialMeanEpoch]


class TrialMeanTermShare(StrictModel):
    steps: list[float]
    problems: list[Problem]


class TrialMeanShelf(StrictModel):
    """Timeline shared by all trials in the Mean view, on seconds from its alignment origin."""

    times: list[float]
    """Interval edges, with one more edge than the number of shares when nonempty."""
    epochs: list[TrialMeanEpoch]
    signals: dict[str, list[float]]
    terms: dict[str, TrialMeanTermShare]
    interventions: dict[str, list[float]] = Field(default_factory=dict, exclude_if=lambda value: not value)
    """Intervention shares by Model attachment address on the same aligned intervals."""


class TrialMean(StrictModel):
    selection: TrialMeanSelection
    conditions: list[TrialMeanCondition]
    shelf: TrialMeanShelf


class StudioSavedFigureMeanRequest(StrictModel):
    """Read a Mean figure for one retained figure job at its saved revision."""

    revision: str = Field(pattern=r"^[a-f0-9]{64}$")
    job_id: str
    mean: TrialMeanSelection
