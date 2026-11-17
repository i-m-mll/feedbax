"""Declare one-replica training methods for package discovery."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from inspect import Parameter, signature
from typing import TYPE_CHECKING, Any, NamedTuple

from feedbax.plugins.declarations import (
    CallableDeclaration,
    DistributionSource,
    mark_callable,
    validate_declaration_key,
)

if TYPE_CHECKING:
    from jax import Array
    from jaxtyping import PyTree
    from pydantic import BaseModel

    from feedbax.runtime.graph import Graph
    from feedbax.training.method_runtime import TrainingContext


class TrainingState(NamedTuple):
    """The graph and the method's remaining persistent array PyTree."""

    model: Graph
    extra: PyTree[Array] = ()


type TrainingMetrics = Mapping[str, Array | TrainingMetrics]
type MethodInit[S: BaseModel] = Callable[[S, Graph, Array], TrainingState]
type MethodUpdate[S: BaseModel] = Callable[
    [TrainingState, TrainingContext[S]], tuple[TrainingState, TrainingMetrics]
]


@dataclass(frozen=True, kw_only=True)
class TrainingMethodDeclaration(CallableDeclaration):
    init: MethodInit[Any]


def training_method[S: BaseModel](
    *, settings: type[S], init: MethodInit[S], key: str | None = None,
    label: str | None = None, revision: int = 1, renamed_from: tuple[str, ...] = (),
) -> Callable[[MethodUpdate[S]], MethodUpdate[S]]:
    """Mark a scalar update for installation through a package entry point."""
    declaration = TrainingMethodDeclaration(
        settings_model=settings, init=init, key=key, label=label,
        revision=revision, renamed_from=renamed_from,
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
class RegisteredTrainingMethod(TrainingMethodDeclaration):
    key: str
    source: DistributionSource
    update: MethodUpdate[Any]


class TrainingMethodRegistry:
    def __init__(self) -> None:
        self._entries: dict[str, RegisteredTrainingMethod] = {}
        self._renames: dict[str, str] = {}
        self._sealed = False

    def register_declaration(
        self, function: MethodUpdate[Any], declaration: TrainingMethodDeclaration,
        *, key: str, source: DistributionSource,
    ) -> None:
        if self._sealed:
            raise RuntimeError("Training method registry is sealed")
        key = validate_declaration_key(key)
        entry = RegisteredTrainingMethod(
            key=key, source=source, update=function, init=declaration.init,
            settings_model=declaration.settings_model, label=declaration.label,
            revision=declaration.revision, renamed_from=declaration.renamed_from,
        )
        for name in (key, *entry.renamed_from):
            if name in self._entries or name in self._renames:
                raise ValueError(f"Training method key {name!r} is already registered")
        self._entries[key] = entry
        self._renames.update(dict.fromkeys(entry.renamed_from, key))

    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))

    def resolve(self, key: str) -> RegisteredTrainingMethod:
        try:
            return self._entries[self._renames.get(key, key)]
        except KeyError:
            raise KeyError(f"Training method {key!r} was not supplied by package {key.rpartition('.')[0]!r}") from None

    def seal(self) -> None:
        self._sealed = True
