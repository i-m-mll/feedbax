"""Stored values and types shared by expression sites and model interfaces."""

from __future__ import annotations

from collections.abc import Iterator
from enum import IntEnum, StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, StrictBool, StrictInt, model_validator

from feedbax.contracts.model_address import ModelAddress


class ValueRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ValueBase(StrEnum):
    REAL = "real"
    INTEGER = "integer"
    BOOLEAN = "boolean"


class Level(IntEnum):
    RUN = 0
    TRIAL = 1
    STEP = 2


class Dimensions(ValueRecord):
    """Exponents of SI bases and plane angle."""

    length: int = 0
    mass: int = 0
    time: int = 0
    current: int = 0
    temperature: int = 0
    amount: int = 0
    luminosity: int = 0
    angle: int = 0

    def combine(self, other: Dimensions, power: int = 1) -> Dimensions:
        return Dimensions(
            **{
                name: getattr(self, name) + power * getattr(other, name)
                for name in type(self).model_fields
            }
        )

    def raised(self, power: int) -> Dimensions:
        return Dimensions().combine(self, power) if power else Dimensions()


DIMENSIONLESS = Dimensions()
TIME = Dimensions(time=1)
ANGLE = Dimensions(angle=1)
LENGTH = Dimensions(length=1)


type SignalSpace = Literal["point", "free"]


class ValueType(ValueRecord):
    base: ValueBase
    shape: tuple[Annotated[StrictInt, Field(gt=0)], ...] = ()
    dimensions: Dimensions = DIMENSIONLESS
    space: SignalSpace | None = Field(default=None, exclude_if=lambda value: value is None)

    def accepts(self, actual: ValueType) -> bool:
        return (
            self.accepts_shape_and_base(actual)
            and self.dimensions == actual.dimensions
            and self.accepts_space(actual)
        )

    def accepts_space(self, actual: ValueType) -> bool:
        return self.space is None or actual.space is None or self.space == actual.space

    def accepts_shape_and_base(self, actual: ValueType) -> bool:
        return self.shape == actual.shape and (
            self.base == actual.base
            or self.base == ValueBase.REAL
            and actual.base == ValueBase.INTEGER
        )


class SetVariableOrigin(ValueRecord):
    """The entry and original variable behind a projected Task variable."""

    step: str
    axis: str
    set: str
    entry: str
    variable: str


class ScopedValue(ValueRecord):
    id: Annotated[str, Field(min_length=1)]
    name: Annotated[str, Field(min_length=1)]
    type: ValueType
    level: Level
    list_count: Annotated[StrictInt, Field(gt=0)] | None = None
    set_entry: SetVariableOrigin | None = Field(default=None, exclude_if=lambda value: value is None)


class Site(ValueRecord):
    expected: ValueType
    finest: Level
    names: tuple[ScopedValue, ...] = ()
    structural: bool = False
    list_count: Annotated[StrictInt, Field(gt=0)] | None = None
    repeat_count: Annotated[StrictInt, Field(gt=0)] | None = None


class TextSpan(ValueRecord):
    """A half-open range of Unicode character offsets in the editor text."""

    start: int
    end: int


class ProblemCode(StrEnum):
    PARSE = "parse"
    UNKNOWN_NAME = "unknown_name"
    DUPLICATE_NAME = "duplicate_name"
    UNKNOWN_FUNCTION = "unknown_function"
    PLUGIN_MISSING = "plugin_missing"
    ARITY = "arity"
    TYPE_MISMATCH = "type_mismatch"
    UNIT_MISMATCH = "unit_mismatch"
    LEVEL_TOO_FINE = "level_too_fine"
    OUT_OF_SCOPE = "out_of_scope"
    CYCLE = "cycle"
    UNBOUNDED_LENGTH = "unbounded_length"
    LENGTH_BUDGET = "length_budget"
    MISSING_LIMIT = "missing_limit"
    LIST_COUNT = "list_count"
    STRUCTURE = "structure"


class Problem(ValueRecord):
    code: ProblemCode
    reason: str
    path: tuple[str | int, ...] = ()
    span: TextSpan | None = None
    model_address: ModelAddress | None = Field(default=None, exclude_if=lambda value: value is None)
    variable: str | None = Field(default=None, exclude_if=lambda value: value is None)
    """The id of the Task variable whose member at ``path`` the problem is in, such as the variable a link initializes."""


type Scalar = StrictBool | StrictInt | FiniteFloat


def normal_literal_number(value: Scalar) -> Scalar:
    """Use an integer for a whole, representable numeric literal."""
    if type(value) is float and value.is_integer() and abs(value) < 2**53:
        return int(value)
    return value


class LiteralValue(ValueRecord):
    kind: Literal["literal"] = "literal"
    value: Scalar
    unit: str | None = None


class NameValue(ValueRecord):
    kind: Literal["name"] = "name"
    reference: str


class CallValue(ValueRecord):
    kind: Literal["call"] = "call"
    function: str
    args: tuple[Expression, ...] = ()
    kwargs: dict[str, Expression] = Field(default_factory=dict)


type WeightPart = Annotated[FiniteFloat, Field(ge=0)]


class DrawValue(ValueRecord):
    kind: Literal["draw"] = "draw"
    family: str
    args: tuple[Expression, ...]
    per: Literal["run", "trial"] = "trial"
    weights: tuple[WeightPart, ...] | None = Field(default=None, exclude_if=lambda value: value is None)


class UnresolvedValue(ValueRecord):
    kind: Literal["unresolved"] = "unresolved"
    text: str


class SequenceValue(ValueRecord):
    kind: Literal["sequence"] = "sequence"
    rule: Literal["span", "linspace", "geomspace"]
    first: Expression
    last: Expression
    count: Expression | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def _count_matches_rule(self) -> SequenceValue:
        if (self.count is None) != (self.rule == "span"):
            raise ValueError("Only linspace and geomspace take a count.")
        return self


type Expression = Annotated[
    LiteralValue | NameValue | CallValue | DrawValue | SequenceValue | UnresolvedValue, Field(discriminator="kind")
]


class SweepList(ValueRecord):
    kind: Literal["list"] = "list"
    members: tuple[Expression, ...] = Field(min_length=1)


class SweepSample(ValueRecord):
    kind: Literal["sample"] = "sample"
    draw: DrawValue
    count: Annotated[StrictInt, Field(gt=0)]


type FieldValue = Expression | SweepList | SweepSample
type NodePath = tuple[str | int, ...]


class LocatedSpan(ValueRecord):
    path: NodePath
    span: TextSpan


class FieldTextResult(ValueRecord):
    value: FieldValue
    spans: tuple[LocatedSpan, ...]


def scope_problems(site: Site, path: NodePath = ()) -> tuple[Problem, ...]:
    names = set()
    ids = set()
    problems = []
    for index, item in enumerate(site.names):
        if item.name in names or item.id in ids:
            problems.append(
                Problem(
                    code=ProblemCode.DUPLICATE_NAME,
                    path=(*path, "names", index),
                    reason=f"Duplicate scoped name or reference {item.name!r}.",
                )
            )
        names.add(item.name)
        ids.add(item.id)
    return tuple(problems)


RESERVED_NAMES = frozenset({"time", "t", "T", "u", "i"})


def children(node: Expression) -> tuple[tuple[NodePath, Expression], ...]:
    if isinstance(node, CallValue):
        return tuple((("args", i), child) for i, child in enumerate(node.args)) + tuple(
            (("kwargs", key), node.kwargs[key]) for key in sorted(node.kwargs)
        )
    if isinstance(node, DrawValue):
        return tuple((("args", i), child) for i, child in enumerate(node.args))
    if isinstance(node, SequenceValue):
        bounds = ((("first",), node.first), (("last",), node.last))
        return bounds if node.count is None else (*bounds, (("count",), node.count))
    return ()


def walk(
    tree: Expression, at: NodePath = (), *, include_sequence_bounds: bool = True,
) -> Iterator[tuple[NodePath, Expression]]:
    """Every node of ``tree`` with its path, the root first."""
    yield at, tree
    if not include_sequence_bounds and isinstance(tree, SequenceValue):
        return
    for step, child in children(tree):
        yield from walk(child, (*at, *step), include_sequence_bounds=include_sequence_bounds)
