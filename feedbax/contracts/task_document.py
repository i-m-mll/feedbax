"""Stored task document values."""

from pydantic import Field

from feedbax.contracts.expression_values import Expression, Scalar, ValueRecord, ValueType


TASK_DOCUMENT_SCHEMA_ID = "feedbax.spec.task_document"


class Variable(ValueRecord):
    id: str
    name: str
    type: ValueType
    value: Expression


class Epoch(ValueRecord):
    id: str
    name: str
    length: Expression | None = None
    max_steps: int | None = Field(default=None, ge=0)


class Cell(ValueRecord):
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
    links: tuple[Link, ...] = ()
    fixed_steps: int | None = Field(default=None, ge=0)
    validation_trials: tuple[ValidationTrial, ...] = Field(
        default=(ValidationTrial(),), min_length=1
    )
