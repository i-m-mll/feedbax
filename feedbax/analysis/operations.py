"""Typed operation registration within the application registry."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from inspect import Parameter, signature
from typing import Annotated, get_args, get_origin, get_type_hints

from pydantic import BaseModel

from feedbax.analysis.validation import validate_namespaced_type_key
from feedbax.component_registry.declarations import parameter_schema_from_model
from feedbax.contracts.operation import DatasetPort, OperationDefinition
from feedbax.dataset import Dataset


@dataclass(frozen=True)
class DatasetOutput:
    name: str
    port: DatasetPort


@dataclass(frozen=True)
class RegisteredOperation:
    function: Callable[..., Dataset]
    parameter_model: type[BaseModel]
    definition: OperationDefinition


def _dataset_metadata(annotation: object, metadata_type: type):
    if get_origin(annotation) is Annotated:
        value_type, *metadata = get_args(annotation)
        if value_type is Dataset and len(metadata) == 1 and isinstance(metadata[0], metadata_type):
            return metadata[0]
    raise TypeError(f"expected a Dataset annotation with {metadata_type.__name__} metadata")


class OperationRegistry:
    """Namespaced pure functions and their one authoring projection."""

    def __init__(self) -> None:
        self._entries: dict[str, RegisteredOperation] = {}
        self._sealed = False

    def register(
        self, key: str, function: Callable[..., Dataset], *, label: str, owner: str,
    ) -> None:
        if self._sealed:
            raise RuntimeError("Operation registry is sealed")
        validate_namespaced_type_key(key, field="operation key")
        if key in self._entries:
            raise ValueError(f"Operation {key!r} is already registered")
        annotations = get_type_hints(function, include_extras=True)
        parameters = signature(function).parameters
        model = annotations.get("params")
        if "params" not in parameters or not isinstance(model, type) or not issubclass(model, BaseModel):
            raise TypeError("operation requires a Pydantic model annotation for params")
        inputs = {}
        for name, parameter in parameters.items():
            if (
                parameter.kind not in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)
                or parameter.default is not Parameter.empty
            ):
                raise TypeError("operation arguments must be required and callable by name")
            if name != "params":
                inputs[name] = _dataset_metadata(annotations.get(name), DatasetPort)
        if not inputs:
            raise TypeError("operation requires a named dataset input")
        output = _dataset_metadata(annotations.get("return"), DatasetOutput)
        for port in inputs.values():
            if missing := port.selected_variables.keys() - model.model_fields.keys():
                raise TypeError(f"dataset selectors are missing from params: {sorted(missing)}")
        static_parameters = tuple(
            name for name, field in model.model_fields.items()
            if (field.json_schema_extra or {}).get("feedbax", {}).get("static") is True
        )
        definition = OperationDefinition(
            key=key,
            owner=owner,
            label=label,
            description=function.__doc__ or "",
            param_schema=list(parameter_schema_from_model(model)),
            parameter_json_schema=model.model_json_schema(),
            static_parameters=static_parameters,
            inputs=inputs,
            outputs={output.name: output.port},
        )
        self._entries[key] = RegisteredOperation(function, model, definition.model_copy(deep=True))

    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))

    def resolve(self, key: str) -> RegisteredOperation:
        entry = self._entries[key]
        return RegisteredOperation(
            entry.function, entry.parameter_model, entry.definition.model_copy(deep=True)
        )

    def definitions(self) -> tuple[OperationDefinition, ...]:
        return tuple(self.resolve(key).definition for key in self.keys())

    def seal(self) -> None:
        self._sealed = True
