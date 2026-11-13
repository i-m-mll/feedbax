"""Function declarations for checking, lowering, and editor projection."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, NamedTuple

from pydantic import Field

from feedbax.contracts.expression_values import (
    CallValue,
    Dimensions,
    Level,
    NodePath,
    ProblemCode,
    ValueBase,
    ValueRecord,
    ValueType,
)


class FunctionClass(StrEnum):
    MATH = "math"
    SHAPE = "shape"
    DRAW = "draw"
    PENALTY = "penalty"
    SWEEP_RULE = "sweep_rule"


class Evaluation(NamedTuple):
    value: Any
    defined: Any


class StateReset(StrEnum):
    TRIAL = "trial"
    EPOCH = "epoch"


class FunctionInfo(ValueRecord):
    id: str
    signature: str
    display_class: FunctionClass
    min_args: int
    max_args: int | None
    keywords: tuple[str, ...] = ()
    implicit_names: dict[str, ValueType] = Field(default_factory=dict)
    level: Level = Level.RUN
    requires_period: bool = False
    random: bool = False
    stateful: bool = False
    reset: StateReset = StateReset.TRIAL


@dataclass(frozen=True)
class SignatureError:
    code: ProblemCode
    reason: str
    path: NodePath = ()


type Signature = Callable[
    [tuple[ValueType, ...], Mapping[str, ValueType]], ValueType | SignatureError
]


@dataclass(frozen=True)
class OperandHint:
    dimensions: Dimensions | None = None
    base: ValueBase | None = None


type ArgumentContext = Callable[
    [OperandHint, tuple[Dimensions | None, ...], tuple[str, ...]],
    tuple[tuple[OperandHint, ...], Mapping[str, OperandHint]],
]
type ResultDimensions = Callable[[tuple[Dimensions | None, ...]], Dimensions | None]


@dataclass(frozen=True)
class Function:
    info: FunctionInfo
    infer: Signature
    evaluate: Callable[..., Any]
    initial_state: Callable[[ValueType], Any] | None = None
    argument_context: ArgumentContext | None = None
    result_dimensions: ResultDimensions | None = None
    bind: Callable[[Function, CallValue], tuple[Function, CallValue] | SignatureError] | None = None


class FunctionRegistry:
    def __init__(self) -> None:
        self._entries: dict[str, Function] = {}
        self._sealed = False

    def register(self, entry: Function) -> None:
        info = entry.info
        if self._sealed:
            raise ValueError("The expression function registry is sealed.")
        if info.id in self._entries:
            raise ValueError(f"Expression function {info.id!r} is already registered.")
        if info.stateful != (entry.initial_state is not None):
            raise ValueError(f"Expression function {info.id!r} needs an initial-state declaration.")
        if info.stateful and info.level != Level.STEP:
            raise ValueError(f"Stateful expression function {info.id!r} must run per step.")
        if (
            not info.id
            or info.min_args < 0
            or (info.max_args is not None and info.max_args < info.min_args)
        ):
            raise ValueError("Invalid expression function declaration.")
        self._entries[info.id] = entry

    def resolve(self, key: str) -> Function | None:
        return self._entries.get(key)

    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))

    def definitions(self) -> tuple[FunctionInfo, ...]:
        return tuple(self._entries[key].info for key in self.keys())

    def seal(self) -> None:
        self._sealed = True
