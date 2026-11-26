"""Authored Task documents and their lowering for execution."""

from collections.abc import Iterable
from enum import IntEnum
from typing import Annotated, Literal

from pydantic import Field, JsonValue, model_validator

from feedbax.contracts.base import StrictModel
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.expression_storage import (
    EXPRESSION_MEMBER_RESERVED_KEYS, StoredExpression, StoredTime, lower_expression, store_expression,
)
from feedbax.contracts.expression_values import (
    ANGLE, LENGTH, TIME, Expression, LiteralValue, Scalar, UnresolvedValue, ValueBase, ValueRecord, ValueType,
)
from feedbax.contracts.model_address import ModelAddress

from feedbax.contracts.stored_members import (
    StoredAddressedMembers,
    StoredMemberKey,
    StoredMembers,
    StoredModelAddressKey,
    keyed_members,
    project_member,
)
from feedbax.contracts.time_units import TimeValue
from feedbax.contracts.window import StoredWindow, Window

TASK_DOCUMENT_SCHEMA_ID = "feedbax.spec.task_document"
STORED_TASK_DOCUMENT_SCHEMA_ID = "feedbax.spec.task_document.v2"


class SetVariableChoices(ValueRecord):
    """Entry identities indexed by the variable's recorded integer or Boolean value."""

    set: str
    entries: tuple[str | None, ...]


class Marker(ValueRecord):
    """The Scene mark for a Task point."""

    shape: Literal["circle", "square", "triangle"]
    filled: bool


class _TaskMember(ValueRecord):
    id: str


class _NamedTaskMember(_TaskMember):
    name: str


class _LabelledTaskMember(ValueRecord):
    label: str = Field(min_length=1)


def _lower_optional(value: StoredExpression | None) -> Expression | None:
    return lower_expression(value) if value is not None else None


def _store_optional(value: Expression | None) -> StoredExpression | None:
    return store_expression(value) if value is not None else None


def _check_expression_keys(keys: Iterable[str]) -> None:
    if reserved := EXPRESSION_MEMBER_RESERVED_KEYS.intersection(keys):
        raise ValueError(f"These Task keys are reserved in expressions: {', '.join(sorted(reserved))}.")


class _VariableFields[ExpressionT](ValueRecord):
    type: ValueType
    value: ExpressionT
    count: ExpressionT | None = Field(default=None, exclude_if=lambda value: value is None)
    set_choices: SetVariableChoices | None = Field(default=None, exclude_if=lambda value: value is None)
    marker: Marker | None = Field(default=None, exclude_if=lambda value: value is None)


class Variable(_VariableFields[Expression], _NamedTaskMember):
    pass


class StoredVariable(_VariableFields[StoredExpression], _LabelledTaskMember):
    def lower(self, key: str) -> Variable:
        return project_member(self, Variable, {"value": lower_expression, "count": _lower_optional},
                              id=key, name=self.label)

    @classmethod
    def from_lowered(cls, variable: Variable) -> "StoredVariable":
        return project_member(variable, cls, {"value": store_expression, "count": _store_optional},
                              label=variable.name)


class _LengthFields[ExpressionT](ValueRecord):
    """The epoch lasts its ``length``; a length that varies is bounded by ``max_steps``."""

    kind: Literal["length"] = "length"
    length: ExpressionT
    max_steps: int | None = Field(default=None, ge=0)


class Length(_LengthFields[Expression]):
    pass


class StoredLength(_LengthFields[StoredExpression]):
    pass


class Rest(ValueRecord):
    """The epoch lasts what the trial's fixed length leaves."""

    kind: Literal["rest"] = "rest"


class _UntilFields[ExpressionT, TimeT](ValueRecord):
    """Test a Boolean condition on entry and each step, up to a required time limit."""

    kind: Literal["until"] = "until"
    condition: ExpressionT
    limit: TimeT


class Until(_UntilFields[Expression, TimeValue]):
    pass


class StoredUntil(_UntilFields[StoredExpression, StoredTime]):
    pass


type End = Annotated[Length | Rest | Until, Field(discriminator="kind")]
type StoredEnd = Annotated[StoredLength | Rest | StoredUntil, Field(discriminator="kind")]


class StoredTaskDraftError(ValueError):
    """A saved Task field whose draft value has no executable representation."""

    def __init__(self, path: tuple[str, ...], text: str, detail: str) -> None:
        super().__init__(detail)
        self.path = path
        self.text = text


def _lower_end(epoch: str, end: StoredEnd) -> End:
    if isinstance(end, StoredLength):
        return project_member(end, Length, {"length": lower_expression})
    if isinstance(end, StoredUntil):
        if isinstance(end.limit, UnresolvedValue):
            raise StoredTaskDraftError(
                ("epochs", epoch, "end", "limit"), end.limit.text, "An Until limit needs a literal time in seconds or steps.",
            )
        return project_member(end, Until, {"condition": lower_expression})
    return end


def _stored_end(end: End) -> StoredEnd:
    if isinstance(end, Length):
        return project_member(end, StoredLength, {"length": store_expression})
    if isinstance(end, Until):
        return project_member(end, StoredUntil, {"condition": store_expression})
    return end


class EpochEnd(IntEnum):
    UNREACHED = 0
    LENGTH = 1
    CONDITION = 2
    LIMIT = 3


class _EpochFields[EndT](ValueRecord):
    end: EndT


class Epoch(_EpochFields[End], _NamedTaskMember):
    pass


class StoredEpoch(_EpochFields[StoredEnd], _LabelledTaskMember):
    pass


class _RepeatFields[ExpressionT](ValueRecord):
    """Repeat the inclusive span of authored epochs a fixed total number of times."""

    first_epoch: str
    last_epoch: str
    count: ExpressionT


class Repeat(_RepeatFields[Expression], _TaskMember):
    pass


class StoredRepeat(_RepeatFields[StoredExpression]):
    pass


class EpochSteps(ValueRecord):
    """The fewest, the nominal and the most steps an epoch of a compiled task lasts."""

    least: int = Field(ge=0)
    nominal: int = Field(ge=0)
    greatest: int = Field(ge=0)


class _CellFields[ExpressionT](ValueRecord):
    first_epoch: str
    last_epoch: str
    value: ExpressionT


class Cell(_CellFields[Expression], _TaskMember):
    pass


class StoredCell(_CellFields[StoredExpression]):
    pass


class _SignalFields[CellsT, ExpressionT, WindowT, AddressT](ValueRecord):
    type: ValueType
    cells: CellsT
    absent: ExpressionT | None = None
    window: WindowT | None = Field(default=None, exclude_if=lambda value: value is None)
    intervention: AddressT | None = Field(default=None, exclude_if=lambda value: value is None)
    """The canonical Model address of the intervention this generated signal gates."""


class Signal(_SignalFields[tuple[Cell, ...], Expression, Window, str], _NamedTaskMember):
    pass


class StoredSignal(_SignalFields[StoredMembers[StoredCell], StoredExpression, StoredWindow, StoredModelAddressKey],
                   _LabelledTaskMember):
    pass


class Link(ValueRecord):
    model_field: str
    variable: str


class TaskAdditions(StrictModel):
    """Task variables and links that are in effect with a Set entry."""

    task_variables: tuple[Variable, ...] = Field(default=(), exclude_if=lambda value: not value)
    task_links: tuple[Link, ...] = Field(default=(), exclude_if=lambda value: not value)


def _lower_variables(variables: dict[str, StoredVariable]) -> tuple[Variable, ...]:
    return tuple(variable.lower(key) for key, variable in variables.items())


def _stored_variables(variables: tuple[Variable, ...]) -> dict[str, StoredVariable]:
    return keyed_members((variable.id, StoredVariable.from_lowered(variable)) for variable in variables)


def _lower_links(links: dict[str, str]) -> tuple[Link, ...]:
    return tuple(Link(model_field=ModelAddress.from_authored_text(address).to_text(), variable=key)
                 for address, key in links.items())


def _stored_links(links: tuple[Link, ...]) -> dict[str, str]:
    return keyed_members((ModelAddress.from_text(link.model_field).to_authored_text(), link.variable) for link in links)


class StoredTaskAdditions(StrictModel):
    """Variables and Model-address links contributed by a stored Set entry or sweep value."""

    task_variables: StoredMembers[StoredVariable] = Field(default_factory=dict, exclude_if=lambda value: not value)
    task_links: StoredAddressedMembers[StoredMemberKey] = Field(
        default_factory=dict, exclude_if=lambda value: not value,
    )

    @model_validator(mode="after")
    def _valid_keys(self) -> "StoredTaskAdditions":
        _check_expression_keys(self.task_variables)
        return self

    def lower(self) -> TaskAdditions:
        return project_member(self, TaskAdditions, {
            "task_variables": _lower_variables,
            "task_links": _lower_links,
        })

    @classmethod
    def from_lowered(cls, additions: TaskAdditions) -> "StoredTaskAdditions":
        return project_member(additions, cls, {
            "task_variables": _stored_variables,
            "task_links": _stored_links,
        })


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

    def without_markers(self) -> "TaskDocument":
        """The computational Task, with its Scene-only metadata removed."""
        if not any(variable.marker is not None for variable in self.variables):
            return self
        return self.model_copy(update={"variables": tuple(
            variable if variable.marker is None else variable.model_copy(update={"marker": None})
            for variable in self.variables
        )})

    def with_member_keys_as_names(self) -> "TaskDocument":
        """Use stable Task variable, epoch, and signal keys in scientific requests."""
        if all(member.name == member.id for group in (self.variables, self.epochs, self.signals) for member in group):
            return self
        return self.model_copy(update={
            field: tuple(member if member.name == member.id else member.model_copy(update={"name": member.id})
                         for member in group)
            for field, group in (("variables", self.variables), ("epochs", self.epochs), ("signals", self.signals))
        })

    def for_execution(self) -> "TaskDocument":
        """Project a Task onto the fields and stable names used in scientific requests."""
        return self.without_markers().with_member_keys_as_names()

    @property
    def condition_ids(self) -> tuple[str, ...]:
        """Identify a fixed cohort by the variables that define each condition."""
        rows = self.validation_trials
        if len(rows) == 1 and not rows[0].variables and not rows[0].epoch_steps:
            return ()
        return tuple(canonical_json_v2_bytes(row.model_dump(mode="json")["variables"]).decode() for row in rows)


def _lower_signal(key: str, signal: StoredSignal) -> Signal:
    cells = tuple(project_member(cell, Cell, {"value": lower_expression}, id=cell_key)
                  for cell_key, cell in signal.cells.items())
    return project_member(signal, Signal, {
        "absent": _lower_optional,
        "window": lambda window: window.lower() if window is not None else None,
        "intervention": lambda address: ModelAddress.from_authored_text(address).to_text() if address is not None else None,
    }, id=key, name=signal.label, cells=cells)


def _stored_signal(signal: Signal) -> StoredSignal:
    cells = keyed_members((cell.id, project_member(cell, StoredCell, {"value": store_expression}))
                          for cell in signal.cells)
    return project_member(signal, StoredSignal, {
        "absent": _store_optional,
        "window": lambda window: StoredWindow.from_lowered(window) if window is not None else None,
        "intervention": lambda address: ModelAddress.from_text(address).to_authored_text() if address is not None else None,
    }, label=signal.name, cells=cells)


def _stored_validation_trials(
    trials: tuple[ValidationTrial, ...], keys: tuple[str, ...] | None = None,
) -> dict[str, ValidationTrial]:
    if not keys and trials == (ValidationTrial(),):
        return {}
    if keys is None:
        keys = tuple(f"trial-{index}" for index in range(1, len(trials) + 1))
    return keyed_members(zip(keys, trials, strict=True))


class StoredTaskDocument(ValueRecord):
    """Task payload in authored order, with keys separate from presentation labels."""

    id: str
    variables: StoredMembers[StoredVariable]
    epochs: StoredMembers[StoredEpoch] = Field(min_length=1)
    signals: StoredMembers[StoredSignal]
    repeats: StoredMembers[StoredRepeat] = Field(default_factory=dict, exclude_if=lambda value: not value)
    links: StoredAddressedMembers[StoredMemberKey] = Field(default_factory=dict)
    fixed_steps: int | None = Field(default=None, ge=0)
    validation_trials: StoredMembers[ValidationTrial] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _unique_expression_keys(self) -> "StoredTaskDocument":
        overlap = self.variables.keys() & self.signals.keys()
        if overlap:
            raise ValueError(f"Task variable and signal keys must be disjoint: {', '.join(sorted(overlap))}.")
        _check_expression_keys((*self.variables, *self.signals))
        return self

    def lower(self) -> TaskDocument:
        return project_member(self, TaskDocument, {
            "variables": _lower_variables,
            "epochs": lambda epochs: tuple(
                project_member(epoch, Epoch, id=key, name=epoch.label, end=_lower_end(key, epoch.end))
                for key, epoch in epochs.items()
            ),
            "signals": lambda signals: tuple(_lower_signal(key, signal) for key, signal in signals.items()),
            "repeats": lambda repeats: tuple(
                project_member(repeat, Repeat, {"count": lower_expression}, id=key) for key, repeat in repeats.items()
            ),
            "links": _lower_links,
            "validation_trials": lambda trials: tuple(trials.values()) or (ValidationTrial(),),
        })

    @classmethod
    def from_lowered(
        cls, task: TaskDocument, *, trial_keys: Iterable[str] | None = None,
    ) -> "StoredTaskDocument":
        """Project a producer's Task, retaining supplied validation row keys when available."""
        overrides = {}
        if trial_keys is not None:
            overrides["validation_trials"] = _stored_validation_trials(task.validation_trials, tuple(trial_keys))
        return project_member(task, cls, {
            "variables": _stored_variables,
            "epochs": lambda epochs: keyed_members(
                (epoch.id, project_member(epoch, StoredEpoch, {"end": _stored_end}, label=epoch.name)) for epoch in epochs
            ),
            "signals": lambda signals: keyed_members((signal.id, _stored_signal(signal)) for signal in signals),
            "repeats": lambda repeats: keyed_members(
                (repeat.id, project_member(repeat, StoredRepeat, {"count": store_expression})) for repeat in repeats
            ),
            "links": _stored_links,
            "validation_trials": _stored_validation_trials,
        }, **overrides)


def add_to_task(task: TaskDocument, additions: Iterable[TaskAdditions]) -> TaskDocument:
    """Add selected entry variables and links once, rejecting competing definitions."""
    variables = {variable.id: variable for variable in task.variables}
    names = {variable.name: variable.id for variable in task.variables}
    links = {link.model_field: link for link in task.links}
    for entry in additions:
        for variable in entry.task_variables:
            if (
                variable.id in variables and variables[variable.id] != variable
                or variable.name in names and names[variable.name] != variable.id
            ):
                raise ValueError(f"Set Task variable {variable.name!r} has a competing definition.")
            variables[variable.id] = variable
            names[variable.name] = variable.id
        for link in entry.task_links:
            if link.model_field in links and links[link.model_field] != link:
                raise ValueError(f"Set links compete for Model field {link.model_field}.")
            links[link.model_field] = link
    return task.model_copy(update={"variables": tuple(variables.values()), "links": tuple(links.values())})


BLANK_TASK = TaskDocument(
    id="task", variables=(), signals=(),
    epochs=(Epoch(id="epoch", name="epoch", end=Length(length=LiteralValue(value=1, unit="s"))),),
)
"""What New makes of a task: one epoch, and nothing else."""

BLANK_STORED_TASK = StoredTaskDocument.from_lowered(BLANK_TASK)


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
