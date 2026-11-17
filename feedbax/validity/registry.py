"""Normalize declaration identities from the live mediated registries."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import TYPE_CHECKING

from packaging.utils import canonicalize_name

from feedbax.contracts.code_validity import RegistryBinding

if TYPE_CHECKING:
    from feedbax.plugins.application import ApplicationRegistryBundle
    from feedbax.plugins.declarations import ResolvedDeclaration
    from feedbax.validity.source import SourceCatalog


def registry_binding(
    family: str, requested: str, resolved: ResolvedDeclaration, catalog: SourceCatalog,
) -> RegistryBinding:
    definition = resolved.defining_object
    module = sys.modules.get(definition.__module__)
    filename = vars(module).get("__file__") if module is not None else None
    return RegistryBinding(
        family=family, requested_key=requested, key=resolved.key,
        distribution=canonicalize_name(resolved.source.distribution), version=resolved.source.version,
        module=definition.__module__, qualified_name=definition.__qualname__,
        source_path=catalog.locate(Path(filename)) if isinstance(filename, str) else None,
    )


def snapshot_registry(
    registries: ApplicationRegistryBundle, catalog: SourceCatalog,
) -> tuple[RegistryBinding, ...]:
    """Project an already materialized registry, without invoking definitions."""
    return tuple(
        registry_binding(family, key, entry, catalog)
        for registry in registries.declaration_registries()
        for family, key, entry in registry.bindings()
    )
