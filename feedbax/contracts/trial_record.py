"""Trial records shared by preview and recorded playback."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, JsonValue

from feedbax.contracts.expression_values import Problem
from feedbax.contracts.jobs import JobReceipt


class TaskTimelineCue(BaseModel):
    epoch: str
    label: str
    start: int
    end: int
    outcome: int


class TrialTerm(BaseModel):
    steps: list[bool]
    """Whether the term applies at each step."""
    covered: bool
    values: list[float] | None = None
    """The weighted measure at each step, when recorded."""


class TrialStateTrack(BaseModel):
    samples: list[list[float]]
    units: str | None
    frame: str


class TrialRecord(BaseModel):
    id: str
    inits: dict[str, JsonValue]
    inputs: dict[str, JsonValue]
    present: dict[str, JsonValue]
    variables: dict[str, JsonValue]
    variable_defined: dict[str, JsonValue]
    timeline: list[TaskTimelineCue]
    terms: dict[str, TrialTerm] = Field(default_factory=dict)
    states: dict[str, TrialStateTrack] = Field(default_factory=dict)
    """Recorded world-frame state tracks, keyed by state Model address."""
    configurations: dict[str, JsonValue] = Field(default_factory=dict)
    """Recorded configuration state histories, keyed by state Model address."""


class TrialRecords(BaseModel):
    kind: Literal["trials"]
    trials: list[TrialRecord]


class TaskEpochShare(BaseModel):
    """Per step, the share of trials that are in the epoch."""

    epoch: str
    label: str
    share: list[float]


class TaskSignalAggregate(BaseModel):
    presence: list[float]
    """Per step, the share of trials in which the signal is present."""
    percentiles: list[list[list[float]] | None] | None
    """Per step and numeric component, percentiles over the trials in which the signal is present."""


class TaskVariableRange(BaseModel):
    """The least and greatest drawn value of each component, in the value's own shape."""

    least: JsonValue
    greatest: JsonValue


class TrialTermAggregate(BaseModel):
    steps: list[float]
    """At each step, the share of trials on which the term applies."""
    covered: float
    """The share of trials the term covers."""


class TrialAggregate(BaseModel):
    kind: Literal["aggregate"]
    epochs: list[TaskEpochShare]
    signals: dict[str, TaskSignalAggregate]
    variables: dict[str, TaskVariableRange]
    terms: dict[str, TrialTermAggregate] = Field(default_factory=dict)


class TrialUnavailable(BaseModel):
    kind: Literal["problems"]
    problems: list[Problem]


type TrialDisplay = Annotated[TrialRecords | TrialAggregate | TrialUnavailable, Field(discriminator="kind")]


class StudioTrialReplayResponse(BaseModel):
    n_steps: int
    state_time: list[float]
    """Recorded state times in seconds, including the reset state at time zero."""
    position: str | None
    """Model address of the declared primary position, if available."""
    receipt: JobReceipt
    drawn: Annotated[TrialRecords | TrialAggregate, Field(discriminator="kind")]
