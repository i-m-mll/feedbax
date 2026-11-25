"""Declare one-replica training methods for package discovery."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from inspect import Parameter, signature
from typing import TYPE_CHECKING, Any, NamedTuple

from feedbax.plugins.declarations import (
    CallableDeclaration,
    DeclarationRegistry,
    DistributionSource,
    ResolvedDeclaration,
    mark_callable,
)

if TYPE_CHECKING:
    from jax import Array
    from jaxtyping import PyTree
    from pydantic import BaseModel

    from feedbax.contracts.training import DocumentTrainingRequestSpec
    from feedbax.runtime.graph import Graph
    from feedbax.training.method_runtime import TrainingContext


class TrainingState(NamedTuple):
    """The graph and the method's remaining persistent array PyTree."""

    model: Graph
    extra: PyTree[Array] = ()


type TrainingMetrics = Mapping[str, Array | TrainingMetrics]
type MethodMetricNames = Sequence[str] | Callable[[DocumentTrainingRequestSpec], Sequence[str]]
type MethodInit[S: BaseModel] = Callable[[S, Graph, Array], TrainingState]
type MethodUpdate[S: BaseModel] = Callable[
    [TrainingState, TrainingContext[S]], tuple[TrainingState, TrainingMetrics]
]


@dataclass(frozen=True, kw_only=True)
class TrainingMethodDeclaration(CallableDeclaration):
    init: MethodInit[Any]
    metrics: MethodMetricNames = ()


def training_method[S: BaseModel](
    *, settings: type[S], init: MethodInit[S], key: str | None = None,
    label: str | None = None, renamed_from: tuple[str, ...] = (), metrics: MethodMetricNames = (),
) -> Callable[[MethodUpdate[S]], MethodUpdate[S]]:
    """Mark a scalar update for package discovery.

    Declare scalar final metric names directly, or derive them from the normalized request.
    """
    declaration = TrainingMethodDeclaration(
        settings_model=settings, init=init, key=key, label=label,
        renamed_from=renamed_from, metrics=metrics,
    )
    arguments = signature(init).parameters.values()
    parameter = next(iter(arguments), None)
    if parameter is None:
        raise TypeError("Training init must accept settings, model, and key")
    annotation = parameter.annotation
    if isinstance(annotation, str):
        annotation = getattr(init, "__globals__", {}).get(annotation, annotation)
    if annotation not in (Parameter.empty, settings, settings.__name__, settings.__qualname__):
        raise TypeError("Training init settings annotation differs from the declared settings model")

    def decorate(update: MethodUpdate[S]) -> MethodUpdate[S]:
        return mark_callable(update, declaration)

    return decorate


@dataclass(frozen=True, kw_only=True)
class RegisteredTrainingMethod:
    key: str
    source: DistributionSource
    settings_model: type[BaseModel]
    label: str | None
    renamed_from: tuple[str, ...]
    init: MethodInit[Any]
    update: MethodUpdate[Any]
    metrics: MethodMetricNames

    def metric_names(self, request: DocumentTrainingRequestSpec) -> frozenset[str]:
        names = self.metrics(request) if callable(self.metrics) else self.metrics
        return frozenset(names)


class TrainingMethodRegistry:
    def __init__(self) -> None:
        self._declarations = DeclarationRegistry[RegisteredTrainingMethod]("training method")

    def register_declaration(
        self, function: MethodUpdate[Any], declaration: TrainingMethodDeclaration,
        *, key: str, source: DistributionSource,
    ) -> None:
        entry = RegisteredTrainingMethod(
            key=key, source=source, update=function, init=declaration.init,
            settings_model=declaration.settings_model, label=declaration.label,
            renamed_from=declaration.renamed_from, metrics=declaration.metrics,
        )
        self._declarations.add(key, source, function, entry, renamed_from=declaration.renamed_from)

    def keys(self) -> tuple[str, ...]:
        return self._declarations.keys()

    def declaration_record(self, key: str) -> ResolvedDeclaration[RegisteredTrainingMethod]:
        return self._declarations.resolve(key)

    def resolve(self, key: str) -> RegisteredTrainingMethod:
        return self.declaration_record(key).value

    def seal(self) -> None:
        self._declarations.seal()
