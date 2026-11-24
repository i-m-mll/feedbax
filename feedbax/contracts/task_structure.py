"""Changes to a task document's structure: its members, their types, its cells' spans and its links."""

from typing import Annotated, Literal

from pydantic import Field

from feedbax.contracts.base import StrictModel
from feedbax.contracts.expression_values import ValueType

type Member = Literal["variable", "epoch", "signal"]


class AddMember(StrictModel):
    """A new member named ``name`` at ``index`` (the end when None); a variable or signal has ``type``, and a variable
    initializes the Model fields of ``initializes``."""

    op: Literal["add"]
    member: Member
    name: str = Field(min_length=1)
    index: int | None = Field(default=None, ge=0)
    type: ValueType | None = None
    initializes: tuple[str, ...] = ()


class RemoveMember(StrictModel):
    op: Literal["remove"]
    member: Member
    id: str


class RenameMember(StrictModel):
    op: Literal["rename"]
    member: Member
    id: str
    name: str = Field(min_length=1)


class MoveMember(StrictModel):
    op: Literal["move"]
    member: Member
    id: str
    index: int = Field(ge=0)


class SetType(StrictModel):
    """The member takes ``type``, and its values become that type's zero."""

    op: Literal["set_type"]
    member: Literal["variable", "signal"]
    id: str
    type: ValueType


class SetSpan(StrictModel):
    """The cell spans ``first_epoch`` through ``last_epoch``; the signal's other cells give up the epochs it takes."""

    op: Literal["set_span"]
    signal: str
    cell: str
    first_epoch: str
    last_epoch: str


class MakeAbsent(StrictModel):
    """The cell is removed: the epochs it spanned read the signal's absent value."""

    op: Literal["absent"]
    signal: str
    cell: str


class MakePresent(StrictModel):
    """A cell over ``epoch``, which no cell of the signal spans."""

    op: Literal["present"]
    signal: str
    epoch: str


class SetLink(StrictModel):
    """``model_field`` takes ``variable``'s value, or no variable's when None."""

    op: Literal["link"]
    model_field: str
    variable: str | None


type TaskStructureChange = Annotated[
    AddMember | RemoveMember | RenameMember | MoveMember | SetType | SetSpan | MakeAbsent | MakePresent | SetLink,
    Field(discriminator="op"),
]
