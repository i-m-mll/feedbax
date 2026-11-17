"""Typed operation registration within the application registry."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from inspect import Parameter, signature
from typing import Annotated, Literal, get_args, get_origin, get_type_hints

from pydantic import BaseModel

from feedbax.contracts.operation import (
    DatasetInputPort, DatasetPort, JaxOperation, ModelPort, OperationDefinition, OperationInputPort, TaskPort,
)
from feedbax.dataset import Dataset
from feedbax.contracts.task_document import TaskDocument
from feedbax.plugins.declarations import (
    CallableDeclaration, DistributionSource, mark_callable, settings_schema, validate_declaration_key,
)
from feedbax.tasks.registry import PythonTaskInstance


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
    revision: int = 1,
    renamed_from: tuple[str, ...] = (),
    numerical: JaxOperation | None = None,
):
    """Declare a dataset-producing function for discovery by the package scanner."""
    declaration = OperationDeclaration(
        key=key, label=label, settings_model=settings, revision=revision,
        renamed_from=renamed_from, numerical=numerical,
    )
    return lambda function: mark_callable(function, declaration)


@dataclass(frozen=True)
class RegisteredOperation:
    function: Callable[..., Dataset]
    parameter_model: type[BaseModel]
    definition: OperationDefinition
    task_parameters: type[BaseModel] | None


def _port_annotation(annotation: object) -> tuple[object, object]:
    if get_origin(annotation) is Annotated and len(parts := get_args(annotation)) == 2:
        return parts[0], parts[1]
    raise TypeError("expected a type annotation with operation port metadata")


def _input_port(annotation: object) -> tuple[OperationInputPort, type[BaseModel] | None]:
    value_type, metadata = _port_annotation(annotation)
    if value_type is Dataset and isinstance(metadata, DatasetPort):
        return DatasetInputPort(**metadata.model_dump(exclude={"kind"})), None
    if isinstance(metadata, ModelPort):
        from feedbax.runtime.graph import Graph

        if value_type is Graph:
            return metadata, None
        raise TypeError("a model input must take a Feedbax Graph")
    if isinstance(metadata, TaskPort):
        if metadata.task_key is None and value_type is TaskDocument:
            return metadata, None
        if metadata.task_key is not None and get_origin(value_type) is PythonTaskInstance:
            validate_declaration_key(metadata.task_key)
            (settings,) = get_args(value_type)
            if isinstance(settings, type) and issubclass(settings, BaseModel):
                return metadata, settings
        raise TypeError("a task input must take a TaskDocument or a declared PythonTaskInstance")
    raise TypeError("expected a Dataset, ModelPort, or TaskPort annotation")


class OperationRegistry:
    """Namespaced pure functions and their one authoring projection."""

    def __init__(self) -> None:
        self._entries: dict[str, RegisteredOperation] = {}
        self._sealed = False

    def register_declaration(
        self, function: Callable[..., Dataset], declaration: OperationDeclaration, *,
        key: str, source: DistributionSource,
    ) -> None:
        if self._sealed:
            raise RuntimeError("Operation registry is sealed")
        validate_declaration_key(key)
        if key in self._entries:
            raise ValueError(f"Operation {key!r} is already registered")
        annotations = get_type_hints(function, include_extras=True)
        parameters = signature(function).parameters
        model = declaration.settings_model
        if "params" not in parameters or annotations.get("params") is not model:
            raise TypeError("operation params annotation must match its declared settings model")
        inputs = {}
        task_parameters = None
        for name, parameter in parameters.items():
            if (
                parameter.kind not in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)
                or parameter.default is not Parameter.empty
            ):
                raise TypeError("operation arguments must be required and callable by name")
            if name != "params":
                port, settings = _input_port(annotations.get(name))
                inputs[name] = port
                if settings is not None:
                    task_parameters = settings
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
            revision=declaration.revision,
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
        self._entries[key] = RegisteredOperation(
            function, model, definition.model_copy(deep=True), task_parameters,
        )

    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))

    def resolve(self, key: str) -> RegisteredOperation:
        entry = self._entries[key]
        return RegisteredOperation(
            entry.function, entry.parameter_model, entry.definition.model_copy(deep=True), entry.task_parameters,
        )

    def definitions(self) -> tuple[OperationDefinition, ...]:
        return tuple(self.resolve(key).definition for key in self.keys())

    def seal(self) -> None:
        self._sealed = True
