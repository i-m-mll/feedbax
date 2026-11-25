"""Stored task document values."""

from enum import IntEnum
from typing import Annotated, Literal

from pydantic import Field, JsonValue

from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.expression_values import (
    ANGLE, LENGTH, TIME, Expression, LiteralValue, Scalar, ValueBase, ValueRecord, ValueType,
)

from feedbax.contracts.time_units import TimeValue


TASK_DOCUMENT_SCHEMA_ID = "feedbax.spec.task_document"


class SetVariableChoices(ValueRecord):
    """Entry identities indexed by the variable's recorded integer or Boolean value."""

    set: str
    entries: tuple[str | None, ...]


class Variable(ValueRecord):
    id: str
    name: str
    type: ValueType
    value: Expression
    count: Expression | None = Field(default=None, exclude_if=lambda value: value is None)
    set_choices: SetVariableChoices | None = Field(default=None, exclude_if=lambda value: value is None)


class Length(ValueRecord):
    """The epoch lasts its ``length``; a length that varies is bounded by ``max_steps``."""

    kind: Literal["length"] = "length"
    length: Expression
    max_steps: int | None = Field(default=None, ge=0)


class Rest(ValueRecord):
    """The epoch lasts what the trial's fixed length leaves."""

    kind: Literal["rest"] = "rest"


class Until(ValueRecord):
    """Test a Boolean condition on entry and each step, up to a required time limit."""

    kind: Literal["until"] = "until"
    condition: Expression
    limit: TimeValue


type End = Annotated[Length | Rest | Until, Field(discriminator="kind")]


class EpochEnd(IntEnum):
    UNREACHED = 0
    LENGTH = 1
    CONDITION = 2
    LIMIT = 3


class Epoch(ValueRecord):
    id: str
    name: str
    end: End


class Repeat(ValueRecord):
    """Repeat the inclusive span of authored epochs a fixed total number of times."""

    id: str
    first_epoch: str
    last_epoch: str
    count: Expression


class EpochSteps(ValueRecord):
    """The fewest, the nominal and the most steps an epoch of a compiled task lasts."""

    least: int = Field(ge=0)
    nominal: int = Field(ge=0)
    greatest: int = Field(ge=0)


class Cell(ValueRecord):
    id: str
    first_epoch: str
    last_epoch: str
    value: Expression


class Signal(ValueRecord):
    id: str
    name: str
    type: ValueType
    cells: tuple[Cell, ...]
    absent: Expression | None = None


class Link(ValueRecord):
    model_field: str
    variable: str


type ValidationValue = Scalar | tuple[ValidationValue, ...]


class ValidationTrial(ValueRecord):
    variables: dict[str, ValidationValue] = Field(default_factory=dict)
    epoch_steps: dict[str, int] = Field(default_factory=dict)


class TaskDocument(ValueRecord):
    id: str
    variables: tuple[Variable, ...]
    epochs: tuple[Epoch, ...] = Field(min_length=1)
    signals: tuple[Signal, ...]
    repeats: tuple[Repeat, ...] = Field(default=(), exclude_if=lambda value: not value)
    links: tuple[Link, ...] = ()
    fixed_steps: int | None = Field(default=None, ge=0)
    validation_trials: tuple[ValidationTrial, ...] = Field(
        default=(ValidationTrial(),), min_length=1
    )

    @property
    def condition_ids(self) -> tuple[str, ...]:
        """Identify a fixed cohort by the variables that define each condition."""
        rows = self.validation_trials
        if len(rows) == 1 and not rows[0].variables and not rows[0].epoch_steps:
            return ()
        return tuple(canonical_json_v2_bytes(row.model_dump(mode="json")["variables"]).decode() for row in rows)


BLANK_TASK = TaskDocument(
    id="task", variables=(), signals=(),
    epochs=(Epoch(id="epoch", name="epoch", end=Length(length=LiteralValue(value=1, unit="s"))),),
)
"""What New makes of a task: one epoch, and nothing else."""


VALUE_TYPES: tuple[tuple[str, ValueType], ...] = (
    ("Number", ValueType(base=ValueBase.REAL)),
    ("Integer", ValueType(base=ValueBase.INTEGER)),
    ("Boolean", ValueType(base=ValueBase.BOOLEAN)),
    ("Duration", ValueType(base=ValueBase.REAL, dimensions=TIME)),
    ("Distance", ValueType(base=ValueBase.REAL, dimensions=LENGTH)),
    ("Angle", ValueType(base=ValueBase.REAL, dimensions=ANGLE)),
    ("Point", ValueType(base=ValueBase.REAL, shape=(2,), dimensions=LENGTH, space="point")),
    ("Displacement", ValueType(base=ValueBase.REAL, shape=(2,), dimensions=LENGTH, space="free")),
    ("Velocity", ValueType(base=ValueBase.REAL, shape=(2,), dimensions=LENGTH.combine(TIME, -1), space="free")),
    ("Vector", ValueType(base=ValueBase.REAL, shape=(2,))),
)
"""The types a variable or a signal can have, each by its name."""


PYTHON_TASK_SCHEMA_ID = "feedbax.spec.python_task"


class PythonTaskSelection(ValueRecord):
    type: str
    params: dict[str, JsonValue] = Field(default_factory=dict)
