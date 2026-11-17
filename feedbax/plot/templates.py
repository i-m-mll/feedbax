"""Declarations for reusable figures over labelled datasets."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from inspect import Parameter, signature
from typing import Literal, get_type_hints

from pydantic import BaseModel

from feedbax.contracts.figure_document import FigureDocument
from feedbax.plugins.declarations import (
    CallableDeclaration,
    DeclarationRegistry,
    DistributionSource,
    ResolvedDeclaration,
    mark_callable,
)


@dataclass(frozen=True, kw_only=True)
class FigureTemplateDeclaration(CallableDeclaration):
    family: Literal["figure_templates"] = "figure_templates"


def figure_template(
    *,
    settings: type[BaseModel],
    key: str | None = None,
    label: str | None = None,
    renamed_from: tuple[str, ...] = (),
):
    """Declare a pure function that builds a figure document from typed settings."""
    declaration = FigureTemplateDeclaration(
        key=key,
        label=label,
        settings_model=settings,
        renamed_from=renamed_from,
    )
    return lambda function: mark_callable(function, declaration)


@dataclass(frozen=True)
class RegisteredFigureTemplate:
    key: str
    label: str
    description: str
    function: Callable[[BaseModel], FigureDocument]
    settings_model: type[BaseModel]
    renamed_from: tuple[str, ...]
    source: DistributionSource


class FigureTemplateRegistry:
    """Scanned figure templates and their distribution provenance."""

    def __init__(self) -> None:
        self._declarations = DeclarationRegistry[RegisteredFigureTemplate]("figure template")

    def register_declaration(
        self,
        function: Callable[[BaseModel], FigureDocument],
        declaration: FigureTemplateDeclaration,
        *,
        key: str,
        source: DistributionSource,
    ) -> None:
        self._declarations.require_mutable()
        parameters = tuple(signature(function).parameters.values())
        annotations = get_type_hints(function)
        if (
            len(parameters) != 1
            or parameters[0].kind not in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
            or parameters[0].default is not Parameter.empty
            or annotations.get(parameters[0].name) is not declaration.settings_model
            or annotations.get("return") is not FigureDocument
        ):
            raise TypeError("A figure template must accept its declared settings and return FigureDocument")

        entry = RegisteredFigureTemplate(
            key=key,
            label=declaration.label or function.__name__,
            description=function.__doc__ or "",
            function=function,
            settings_model=declaration.settings_model,
            renamed_from=declaration.renamed_from,
            source=source,
        )
        self._declarations.add(key, source, function, entry, renamed_from=entry.renamed_from)

    def keys(self) -> tuple[str, ...]:
        return self._declarations.keys()

    def declaration_record(self, key: str) -> ResolvedDeclaration[RegisteredFigureTemplate]:
        return self._declarations.resolve(key)

    def resolve(self, key: str) -> RegisteredFigureTemplate:
        return self.declaration_record(key).value

    def seal(self) -> None:
        self._declarations.seal()
