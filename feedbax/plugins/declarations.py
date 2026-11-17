"""Metadata shared by discoverable Feedbax declarations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from feedbax.contracts.graph import ParamSchema


@dataclass(frozen=True)
class DistributionSource:
    package: str
    distribution: str
    version: str


@dataclass(frozen=True, kw_only=True)
class CallableDeclaration:
    settings_model: type[BaseModel]
    key: str | None = None
    label: str | None = None
    revision: int = 1
    renamed_from: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.settings_model, type) or not issubclass(self.settings_model, BaseModel):
            raise TypeError("Feedbax settings must be a Pydantic model")
        if type(self.revision) is not int or self.revision < 1:
            raise ValueError("A computation revision must be a positive integer")
        for key in (self.key, *self.renamed_from):
            if key is not None:
                validate_declaration_key(key)
        if self.key in self.renamed_from or len(set(self.renamed_from)) != len(self.renamed_from):
            raise ValueError("Rename history contains a repeated key")


def validate_declaration_key(key: str) -> str:
    if "." not in key or any(not segment.isidentifier() for segment in key.split(".")):
        raise ValueError(f"Expected a dotted Feedbax declaration key, got {key!r}")
    return key


def mark_callable[F: Callable[..., Any]](function: F, declaration: CallableDeclaration) -> F:
    if callable_declaration(function) is not None:
        raise ValueError(f"{function.__qualname__} already has a Feedbax declaration")
    setattr(function, "__feedbax_declaration__", declaration)
    return function


def callable_declaration(function: object) -> CallableDeclaration | None:
    declaration = getattr(function, "__feedbax_declaration__", None)
    return declaration if isinstance(declaration, CallableDeclaration) else None


@dataclass(frozen=True)
class SettingsSchema:
    param_schema: tuple[ParamSchema, ...]
    parameter_json_schema: dict[str, Any]
    static_parameters: tuple[str, ...]


def settings_schema(model: type[BaseModel]) -> SettingsSchema:
    from feedbax.component_registry.declarations import parameter_schema_from_model

    schema = model.model_json_schema()
    return SettingsSchema(
        param_schema=parameter_schema_from_model(model),
        parameter_json_schema=schema,
        static_parameters=tuple(
            name
            for name, field in model.model_fields.items()
            if isinstance(field.json_schema_extra, Mapping)
            and field.json_schema_extra.get("feedbax", {}).get("static") is True
        ),
    )
