"""Typed operation registration within the application registry."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from inspect import Parameter, signature
from typing import Annotated, Literal, get_args, get_origin, get_type_hints

from pydantic import BaseModel

from feedbax.contracts.operation import (
    DatasetInputPort, DatasetPort, JaxOperation, ModelPort, OperationDefinition, OperationInputPort, TaskPort,
)
from feedbax.dataset import Dataset
from feedbax.plugins.declarations import (
    CallableDeclaration, DeclarationRegistry, DistributionSource, ResolvedDeclaration, mark_callable,
    settings_schema,
)


@dataclass(frozen=True)
class DatasetOutput:
    name: str
    port: DatasetPort


@dataclass(frozen=True, kw_only=True)
class OperationDeclaration(CallableDeclaration):
    family: Literal["operations"] = "operations"
    numerical: JaxOperation | None = None


def operation(
    *,
    settings: type[BaseModel],
    key: str | None = None,
    label: str | None = None,
    renamed_from: tuple[str, ...] = (),
    numerical: JaxOperation | None = None,
):
    """Declare a dataset-producing function for discovery by the package scanner."""
    declaration = OperationDeclaration(
        key=key, label=label, settings_model=settings,
        renamed_from=renamed_from, numerical=numerical,
    )
    return lambda function: mark_callable(function, declaration)


@dataclass(frozen=True)
class RegisteredOperation:
    parameter_model: type[BaseModel]
    definition: OperationDefinition


def _port_annotation(annotation: object) -> tuple[object, object]:
    if get_origin(annotation) is Annotated and len(parts := get_args(annotation)) == 2:
        return parts[0], parts[1]
    raise TypeError("expected a type annotation with operation port metadata")


def _input_port(annotation: object) -> OperationInputPort:
    value_type, metadata = _port_annotation(annotation)
    if value_type is Dataset and isinstance(metadata, DatasetPort):
        return DatasetInputPort(**metadata.model_dump(exclude={"kind"}))
    if isinstance(metadata, (ModelPort, TaskPort)):
        return metadata
    raise TypeError("expected a Dataset, ModelPort, or TaskPort annotation")


class OperationRegistry:
    """Namespaced pure functions and their one authoring projection."""

    def __init__(self) -> None:
        self._entries = DeclarationRegistry[RegisteredOperation]("operation")

    def register_declaration(
        self, function: Callable[..., Dataset], declaration: OperationDeclaration, *,
        key: str, source: DistributionSource,
    ) -> None:
        self._entries.require_mutable()
        annotations = get_type_hints(function, include_extras=True)
        parameters = signature(function).parameters
        model = declaration.settings_model
        if "params" not in parameters or annotations.get("params") is not model:
            raise TypeError("operation params annotation must match its declared settings model")
        inputs = {}
        for name, parameter in parameters.items():
            if (
                parameter.kind not in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)
                or parameter.default is not Parameter.empty
            ):
                raise TypeError("operation arguments must be required and callable by name")
            if name != "params":
                inputs[name] = _input_port(annotations.get(name))
        if not inputs:
            raise TypeError("operation requires a named input")
        if sum(isinstance(port, ModelPort) for port in inputs.values()) > 1:
            raise TypeError("operation accepts at most one model input")
        if sum(isinstance(port, TaskPort) for port in inputs.values()) > 1:
            raise TypeError("operation accepts at most one task input")
        if declaration.numerical is None and any(isinstance(port, ModelPort) for port in inputs.values()):
            raise TypeError("a model-consuming operation requires a numerical worker declaration")
        value_type, output = _port_annotation(annotations.get("return"))
        if value_type is not Dataset or not isinstance(output, DatasetOutput):
            raise TypeError("expected a Dataset return annotation with DatasetOutput metadata")
        for port in inputs.values():
            if isinstance(port, DatasetInputPort):
                if missing := port.selected_variables.keys() - model.model_fields.keys():
                    raise TypeError(f"dataset selectors are missing from params: {sorted(missing)}")
        schema = settings_schema(model)
        definition = OperationDefinition(
            key=key,
            owner=source.distribution,
            owner_version=source.version,
            renamed_from=declaration.renamed_from,
            label=declaration.label or function.__name__,
            description=function.__doc__ or "",
            param_schema=list(schema.param_schema),
            parameter_json_schema=schema.parameter_json_schema,
            static_parameters=schema.static_parameters,
            inputs=inputs,
            outputs={output.name: output.port},
            numerical=declaration.numerical,
        )
        self._entries.add(
            key, source, function, RegisteredOperation(model, definition),
            renamed_from=declaration.renamed_from,
        )

    def keys(self) -> tuple[str, ...]:
        return self._entries.keys()

    def resolve(self, key: str) -> ResolvedDeclaration[RegisteredOperation]:
        entry = self._entries.resolve(key)
        return replace(entry, value=replace(entry.value, definition=entry.value.definition.model_copy(deep=True)))

    def definitions(self) -> tuple[OperationDefinition, ...]:
        return tuple(self.resolve(key).value.definition for key in self.keys())

    def seal(self) -> None:
        self._entries.seal()
