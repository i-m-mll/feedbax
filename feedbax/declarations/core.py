"""Neutral extension declarations and layer-local facet composition."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, TypeVar


class DeclarationCompositionError(ValueError):
    """A declaration set cannot be composed without ambiguity."""


@dataclass(frozen=True)
class Declaration:
    """Facts shared by every layer that consumes one scientific concept."""

    kind: str
    type_id: str
    schema_id: str
    schema_version: str
    capabilities: frozenset[str]
    runtime_protocol: type[Any]
    owner: str

    def __post_init__(self) -> None:
        identities = (
            self.kind,
            self.type_id,
            self.schema_id,
            self.schema_version,
            self.owner,
        )
        if any(not isinstance(value, str) or not value.strip() for value in identities):
            raise DeclarationCompositionError("declaration identities and owner must be non-empty")
        if not isinstance(self.capabilities, frozenset) or any(
            not isinstance(value, str) or not value for value in self.capabilities
        ):
            raise DeclarationCompositionError("declaration capabilities must be non-empty strings")
        if not isinstance(self.runtime_protocol, type):
            raise DeclarationCompositionError("declaration runtime_protocol must be a type")


@dataclass(frozen=True)
class Facet:
    """One layer-owned projection keyed to a neutral declaration identity."""

    kind: str
    type_id: str
    layer: str
    schema_version: str
    value: object

    def __post_init__(self) -> None:
        identities = (self.kind, self.type_id, self.layer, self.schema_version)
        if any(not isinstance(item, str) or not item.strip() for item in identities):
            raise DeclarationCompositionError("facet identities must be non-empty")


T = TypeVar("T")


class DeclarationCatalog:
    """Isolated composition root for declarations and their layer facets."""

    def __init__(self) -> None:
        self._sealed = False
        self._declarations: dict[tuple[str, str], Declaration] = {}
        self._facets: dict[tuple[str, str, str], Facet] = {}

    def declare(self, declaration: Declaration) -> None:
        self._require_mutable()
        key = (declaration.kind, declaration.type_id)
        if key in self._declarations:
            raise DeclarationCompositionError(f"duplicate declaration identity {key!r}")
        self._declarations[key] = declaration

    def add_facet(self, facet: Facet) -> None:
        self._require_mutable()
        declaration = self._declarations.get((facet.kind, facet.type_id))
        if declaration is None:
            raise DeclarationCompositionError(
                f"facet {facet.layer!r} has no declaration for {(facet.kind, facet.type_id)!r}"
            )
        if facet.schema_version != declaration.schema_version:
            raise DeclarationCompositionError(
                f"facet {facet.layer!r} for {facet.type_id!r} uses schema version "
                f"{facet.schema_version!r}; declaration uses {declaration.schema_version!r}"
            )
        key = (facet.kind, facet.type_id, facet.layer)
        if key in self._facets:
            raise DeclarationCompositionError(f"duplicate facet identity {key!r}")
        self._facets[key] = facet

    def register(self, declaration: Declaration, facets: Iterable[Facet] = ()) -> None:
        """Atomically register one declaration and all supplied facets."""
        self._require_mutable()
        facets = tuple(facets)
        trial = DeclarationCatalog()
        trial._declarations = dict(self._declarations)
        trial._facets = dict(self._facets)
        trial.declare(declaration)
        for facet in facets:
            trial.add_facet(facet)
        self._declarations = trial._declarations
        self._facets = trial._facets

    def declaration(self, kind: str, type_id: str) -> Declaration:
        try:
            return self._declarations[(kind, type_id)]
        except KeyError as exc:
            available = sorted(key for key in self._declarations if key[0] == kind)
            raise DeclarationCompositionError(
                f"unknown {kind} declaration {type_id!r}; available={available!r}"
            ) from exc

    def facet(self, kind: str, type_id: str, layer: str, *, required: bool = True) -> object | None:
        entry = self._facets.get((kind, type_id, layer))
        if entry is not None:
            return entry.value
        if required:
            raise DeclarationCompositionError(
                f"{kind} declaration {type_id!r} has no required {layer!r} facet"
            )
        return None

    def compose(
        self,
        kind: str,
        type_id: str,
        *,
        required_layers: Iterable[str] = (),
    ) -> tuple[Declaration, Mapping[str, object]]:
        declaration = self.declaration(kind, type_id)
        required = tuple(required_layers)
        missing = [layer for layer in required if (kind, type_id, layer) not in self._facets]
        if missing:
            raise DeclarationCompositionError(
                f"{kind} declaration {type_id!r} is missing required facets={missing!r}"
            )
        values = {
            layer: facet.value
            for (facet_kind, facet_type_id, layer), facet in self._facets.items()
            if facet_kind == kind and facet_type_id == type_id
        }
        return declaration, MappingProxyType(values)

    def identities(self, kind: str | None = None) -> tuple[tuple[str, str], ...]:
        keys = self._declarations
        return tuple(sorted(key for key in keys if kind is None or key[0] == kind))

    def seal(self) -> None:
        self._sealed = True

    def _require_mutable(self) -> None:
        if self._sealed:
            raise RuntimeError("declaration catalog is sealed")
