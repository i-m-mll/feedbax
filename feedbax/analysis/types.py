"""Analysis-owned data containers and spec enums."""

from collections.abc import Callable
from enum import Enum
from typing import Any, NamedTuple, Optional

import equinox as eqx
from equinox import Module
from jax_cookbook import LDict
from jaxtyping import Array, ArrayLike, PyTree

from feedbax.config.namespace import TreeNamespace
from feedbax.tasks import AbstractTask


class AnalysisInputData(Module):
    models: PyTree[Module]
    tasks: PyTree[Module]
    states: PyTree[Module]
    hps: PyTree[TreeNamespace]
    extras: PyTree[TreeNamespace]


class Labels(NamedTuple):
    full: PyTree[str]
    medium: PyTree[str]
    short: PyTree[str]


class VarSpec(eqx.Module):
    where: Callable[[AnalysisInputData], Array]
    labels: Labels
    time_axis: int = -2
    vec_axis: int = -1
    origin: Optional[ArrayLike | Callable[[AbstractTask], ArrayLike]] = None


class Polar(NamedTuple):
    angle: Array
    radius: Array


class Direction(str, Enum):
    """Available directions for vector components."""

    PARALLEL = "parallel"
    LATERAL = "lateral"


class ResponseVar(str, Enum):
    """Variables available in response state."""

    POSITION = "pos"
    VELOCITY = "vel"
    COMMAND = "command"
    FORCE = "force"


type LDictTree[T] = T | LDict[Any, LDictTree[T]]
