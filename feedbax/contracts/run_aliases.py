"""Explicit compile-time aliases for authenticated run-manifest references."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any, Literal

from pydantic import Field, model_validator

from feedbax.contracts.manifest import (
    ParentRef,
    StrictModel,
    authenticated_manifest_ref_profile,
)

RUN_ALIAS_CATALOG_SCHEMA_ID = "feedbax.spec.run_alias_catalog"
RUN_ALIAS_CATALOG_SCHEMA_VERSION = "feedbax.spec.run_alias_catalog.v1"
RUN_ALIAS_REF_SCHEMA_ID = "feedbax.ref.run_alias"
RUN_ALIAS_REF_SCHEMA_VERSION = "feedbax.ref.run_alias.v1"


class RunAliasRef(StrictModel):
    """Branded authored reference expanded before a run spec is validated."""

    schema_id: Literal["feedbax.ref.run_alias"] = RUN_ALIAS_REF_SCHEMA_ID
    schema_version: Literal["feedbax.ref.run_alias.v1"] = RUN_ALIAS_REF_SCHEMA_VERSION
    alias: str

    @model_validator(mode="after")
    def _validate_alias(self) -> "RunAliasRef":
        _validate_alias_name(self.alias)
        return self


class RunAliasDeclaration(StrictModel):
    """One symbolic name bound to a pin or another catalog-local alias."""

    alias: str
    target: ParentRef | RunAliasRef

    @model_validator(mode="after")
    def _validate_alias(self) -> "RunAliasDeclaration":
        _validate_alias_name(self.alias)
        return self


class RunAliasCatalog(StrictModel):
    """Versioned explicit catalog supplied to one compilation operation."""

    schema_id: Literal["feedbax.spec.run_alias_catalog"] = RUN_ALIAS_CATALOG_SCHEMA_ID
    schema_version: Literal[
        "feedbax.spec.run_alias_catalog.v1"
    ] = RUN_ALIAS_CATALOG_SCHEMA_VERSION
    aliases: list[RunAliasDeclaration] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_unique_aliases(self) -> "RunAliasCatalog":
        names = [declaration.alias for declaration in self.aliases]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"run alias catalog contains ambiguous aliases: {duplicates!r}")
        return self


def resolve_run_aliases(
    payload: Mapping[str, Any],
    catalogs: Sequence[RunAliasCatalog | Mapping[str, Any]],
) -> dict[str, Any]:
    """Expand branded alias refs to authenticated, content-pinned ``ParentRef`` values.

    Catalogs are operation inputs, not process-global state. The returned document
    contains only ordinary ``ParentRef`` mappings; neither aliases nor catalog
    declarations can reach execution or durable run identity.
    """

    declarations: dict[str, RunAliasDeclaration] = {}
    for raw_catalog in catalogs:
        catalog = (
            raw_catalog
            if isinstance(raw_catalog, RunAliasCatalog)
            else RunAliasCatalog.model_validate(raw_catalog)
        )
        for declaration in catalog.aliases:
            if declaration.alias in declarations:
                raise ValueError(
                    f"run alias {declaration.alias!r} is ambiguous across supplied catalogs"
                )
            declarations[declaration.alias] = declaration

    resolved: dict[str, ParentRef] = {}
    resolving: list[str] = []

    def resolve_alias(alias: str) -> ParentRef:
        if alias in resolved:
            return resolved[alias]
        if alias in resolving:
            cycle = " -> ".join((*resolving, alias))
            raise ValueError(f"run alias cycle detected: {cycle}")
        try:
            declaration = declarations[alias]
        except KeyError as exc:
            raise ValueError(f"run alias {alias!r} is not declared") from exc
        resolving.append(alias)
        try:
            target = declaration.target
            if isinstance(target, RunAliasRef):
                parent = resolve_alias(target.alias)
            else:
                parent = target
                if authenticated_manifest_ref_profile(parent) is None:
                    raise ValueError(
                        f"run alias {alias!r} target must be an authenticated manifest ParentRef"
                    )
            resolved[alias] = parent
            return parent
        finally:
            resolving.pop()

    for alias in sorted(declarations):
        resolve_alias(alias)

    def expand(value: Any) -> Any:
        if isinstance(value, Mapping):
            if value.get("schema_id") == RUN_ALIAS_REF_SCHEMA_ID:
                alias_ref = RunAliasRef.model_validate(value)
                return resolve_alias(alias_ref.alias).model_dump(mode="json", exclude_none=True)
            return {key: expand(item) for key, item in value.items()}
        if isinstance(value, list):
            return [expand(item) for item in value]
        if isinstance(value, tuple):
            return [expand(item) for item in value]
        return deepcopy(value)

    return expand(payload)


def _validate_alias_name(value: str) -> None:
    if not value or value != value.strip():
        raise ValueError("run alias names must be non-empty and have no surrounding whitespace")


__all__ = [
    "RUN_ALIAS_CATALOG_SCHEMA_ID",
    "RUN_ALIAS_CATALOG_SCHEMA_VERSION",
    "RUN_ALIAS_REF_SCHEMA_ID",
    "RUN_ALIAS_REF_SCHEMA_VERSION",
    "RunAliasCatalog",
    "RunAliasDeclaration",
    "RunAliasRef",
    "resolve_run_aliases",
]
