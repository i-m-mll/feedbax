"""Model-parameter producers declared by installed Feedbax packages."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from feedbax.contracts.operation import DatasetPort, JaxOperation
from feedbax.contracts.synthesis import SynthesisMethodExecution
from feedbax.plugins.declarations import CallableDeclaration, DeclarationRegistry, DistributionSource, mark_callable

if TYPE_CHECKING:
    from jax import Array
    from pydantic import BaseModel

    from feedbax.contracts.expression_values import Problem
    from feedbax.contracts.model_address import ModelAddress
    from feedbax.dataset import Dataset
    from feedbax.objectives.document import ObjectiveExecutor
    from feedbax.runtime.graph import Graph
    from feedbax.tasks.document import TrialSource


@dataclass(frozen=True)
class ParameterSlot:
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class SynthesisProblem:
    graph: Graph
    task: TrialSource
    objective: ObjectiveExecutor
    slots: Mapping[str, ParameterSlot]
    """Parameter layouts by canonical ModelAddress text."""
    target: ModelAddress


@dataclass(frozen=True)
class Synthesis:
    parameters: Mapping[str, Array]
    """Arrays by canonical ModelAddress text."""
    datasets: Mapping[str, Dataset] = field(default_factory=dict)


type SynthesisMethod[S: BaseModel] = Callable[[SynthesisProblem, S], Synthesis | tuple[Problem, ...]]


@dataclass(frozen=True, kw_only=True)
class SynthesisMethodDeclaration(CallableDeclaration):
    numerical: JaxOperation
    outputs: Mapping[str, DatasetPort]


def synthesis_method[S: BaseModel](
    *, settings: type[S], numerical: JaxOperation, key: str | None = None, label: str | None = None,
    renamed_from: tuple[str, ...] = (), outputs: Mapping[str, DatasetPort] | None = None,
) -> Callable[[SynthesisMethod[S]], SynthesisMethod[S]]:
    """Declare a model-parameter producer and any additional Dataset roles."""
    if numerical.floating_point_bits != 64:
        raise ValueError("A synthesis method requires 64-bit floating point.")
    if outputs and "model" in outputs:
        raise ValueError("Synthesis reserves the model product role.")
    declaration = SynthesisMethodDeclaration(
        settings_model=settings, key=key, label=label, renamed_from=renamed_from, numerical=numerical,
        outputs=dict(outputs or {}),
    )
    return lambda function: mark_callable(function, declaration)


@dataclass(frozen=True)
class RegisteredSynthesisMethod:
    key: str
    source: DistributionSource
    method: SynthesisMethod[Any]
    declaration: SynthesisMethodDeclaration

    @property
    def definition(self) -> SynthesisMethodExecution:
        return SynthesisMethodExecution(
            key=self.key, numerical=self.declaration.numerical, outputs=dict(self.declaration.outputs),
        )


class SynthesisMethodRegistry:
    def __init__(self) -> None:
        self._declarations = DeclarationRegistry[RegisteredSynthesisMethod]("synthesis method")

    def register_declaration(
        self, function: SynthesisMethod[Any], declaration: SynthesisMethodDeclaration,
        *, key: str, source: DistributionSource,
    ) -> None:
        entry = RegisteredSynthesisMethod(key, source, function, declaration)
        self._declarations.add(key, source, function, entry, renamed_from=declaration.renamed_from)

    def keys(self) -> tuple[str, ...]:
        return self._declarations.keys()

    def declaration_record(self, key: str):
        return self._declarations.resolve(key)

    def resolve(self, key: str) -> RegisteredSynthesisMethod:
        return self.declaration_record(key).value

    def seal(self) -> None:
        self._declarations.seal()
