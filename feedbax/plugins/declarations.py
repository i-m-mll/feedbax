"""Metadata shared by discoverable Feedbax declarations."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass, replace
from typing import Any

from pydantic import BaseModel

from feedbax.registry_errors import RegistryCollisionError


@dataclass(frozen=True)
class DistributionSource:
    package: str
    distribution: str
    version: str


@dataclass(frozen=True)
class ResolvedDeclaration[T]:
    key: str
    source: DistributionSource
    defining_object: Callable[..., object]
    value: T


type DeclarationObserver = Callable[[str, str, ResolvedDeclaration[Any]], None]
_observer: DeclarationObserver | None = None


@contextmanager
def observe_declarations(observer: DeclarationObserver) -> Iterator[None]:
    """Scope the resolution observer to one delivery in a managed process."""
    global _observer
    previous, _observer = _observer, observer
    try:
        yield
    finally:
        _observer = previous


class DeclarationRegistry[T]:
    """Store declaration authority and mediate every executable lookup."""

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._entries: dict[str, ResolvedDeclaration[T]] = {}
        self._renames: dict[str, str] = {}
        self._sealed = False

    def add(
        self, key: str, source: DistributionSource, defining_object: Callable[..., object], value: T,
        *, renamed_from: tuple[str, ...] = (),
    ) -> None:
        self.require_mutable()
        names = (key, *renamed_from)
        if len(set(names)) != len(names):
            raise RegistryCollisionError(f"{self._kind} {key!r} repeats a key in its rename history")
        for name in names:
            validate_declaration_key(name)
            if name in self._entries or name in self._renames:
                raise RegistryCollisionError(f"{self._kind} {name!r} is already registered")
        self._entries[key] = ResolvedDeclaration(key, source, defining_object, value)
        self._renames.update(dict.fromkeys(renamed_from, key))

    def resolve(self, key: str) -> ResolvedDeclaration[T]:
        declaration = self._entries.get(self._renames.get(key, key))
        if declaration is None:
            raise KeyError(missing_declaration(self._kind, key))
        if _observer is not None:
            _observer(self._kind, key, declaration)
        return declaration

    def bindings(self) -> Iterator[tuple[str, str, ResolvedDeclaration[T]]]:
        for key, declaration in sorted(self._entries.items()):
            yield self._kind, key, declaration
        for alias, key in sorted(self._renames.items()):
            yield self._kind, alias, self._entries[key]

    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))

    def __contains__(self, key: str) -> bool:
        return key in self._entries

    def values(self) -> Iterator[T]:
        return (record.value for record in self._entries.values())

    def items(self) -> Iterator[tuple[str, T]]:
        return ((key, record.value) for key, record in self._entries.items())

    def copy(self) -> DeclarationRegistry[T]:
        registry = copy(self)
        registry._entries = self._entries.copy()
        registry._renames = self._renames.copy()
        return registry

    def replace_value(self, key: str, value: T) -> None:
        self.require_mutable()
        self._entries[key] = replace(self._entries[key], value=value)

    def require_mutable(self) -> None:
        if self._sealed:
            raise RuntimeError(f"{self._kind} registry is sealed")

    def seal(self) -> None:
        self._sealed = True


@dataclass(frozen=True, kw_only=True)
class CallableDeclaration:
    settings_model: type[BaseModel]
    key: str | None = None
    label: str | None = None
    renamed_from: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.settings_model, type) or not issubclass(self.settings_model, BaseModel):
            raise TypeError("Feedbax settings must be a Pydantic model")
        for key in (self.key, *self.renamed_from):
            if key is not None:
                validate_declaration_key(key)
        if self.key in self.renamed_from or len(set(self.renamed_from)) != len(self.renamed_from):
            raise ValueError("Rename history contains a repeated key")


def validate_declaration_key(key: str) -> str:
    if "." not in key or any(not segment.isidentifier() for segment in key.split(".")):
        raise ValueError(f"Expected a dotted Feedbax declaration key, got {key!r}")
    return key


def declaration_key(name: str, key: str | None, source: DistributionSource) -> str:
    return validate_declaration_key(
        f"{source.package.partition('.')[0]}.{name}" if key is None else key
    )


def missing_declaration(kind: str, key: str) -> str:
    return f"Unknown {kind} {key!r}. No installed package provides it (package {key.split('.', 1)[0]!r})."


def mark_callable[F: Callable[..., Any]](function: F, declaration: CallableDeclaration) -> F:
    if callable_declaration(function) is not None:
        raise ValueError(f"{function.__qualname__} already has a Feedbax declaration")
    setattr(function, "__feedbax_declaration__", declaration)
    return function


def callable_declaration(function: object) -> CallableDeclaration | None:
    declaration = getattr(function, "__feedbax_declaration__", None)
    return declaration if isinstance(declaration, CallableDeclaration) else None
