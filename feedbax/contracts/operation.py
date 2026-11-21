"""Portable discovery of operations and their typed ports."""

from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import Field, JsonValue

from feedbax.contracts.base import StrictModel
from feedbax.contracts.dataset import FieldDescriptor
from feedbax.contracts.graph import ParamSchema


class DatasetPort(StrictModel):
    """Dataset schemas and fields accepted or produced by a port."""

    schema_ids: tuple[str, ...] = Field(min_length=1)
    variables: dict[str, FieldDescriptor] = Field(default_factory=dict)
    selected_variables: dict[str, FieldDescriptor] = Field(default_factory=dict)
    coordinates: dict[str, FieldDescriptor] = Field(default_factory=dict)


class WholeDimensions(StrictModel):
    """The operation parameter whose value lists the label dimensions the port takes whole."""

    parameter: str = Field(min_length=1)


type Takes = Literal["all", "one"] | WholeDimensions
"""What a dataset port takes of each label dimension of what is bound: the whole stack (``all``), one member
(``one``), or the dimensions a parameter lists. The operation is one request per combination of the labels it takes
one element of."""


class DatasetInputPort(DatasetPort):
    kind: Literal["dataset"] = "dataset"
    takes: Takes = "all"


class ModelPort(StrictModel):
    kind: Literal["model"] = "model"


class TaskPort(StrictModel):
    kind: Literal["task"] = "task"
    task_key: str | None = None


OperationInputPort = Annotated[DatasetInputPort | ModelPort | TaskPort, Field(discriminator="kind")]


class JaxOperation(StrictModel):
    """The numerical worker required to execute an operation."""

    backend: Literal["cpu", "gpu"]
    floating_point_bits: Literal[32, 64]

    def environment(self) -> dict[str, str]:
        return {
            "JAX_PLATFORMS": "cuda" if self.backend == "gpu" else "cpu",
            "JAX_ENABLE_X64": str(int(self.floating_point_bits == 64)),
        }


class OperationDefinition(StrictModel):
    """Data needed to discover, wire, and author a registered operation."""

    key: str
    owner: str
    owner_version: str
    renamed_from: tuple[str, ...] = ()
    label: str
    description: str
    param_schema: list[ParamSchema]
    parameter_json_schema: dict[str, JsonValue]
    static_parameters: tuple[str, ...]
    inputs: dict[str, OperationInputPort]
    outputs: dict[str, DatasetPort]
    numerical: JaxOperation | None = None

def require_dataset_port(actual: DatasetPort, expected: DatasetPort, *, fields: bool = True) -> None:
    if not set(actual.schema_ids) <= set(expected.schema_ids):
        raise ValueError(f"Requires dataset schemas {expected.schema_ids}, got {actual.schema_ids}.")
    if fields:
        require_dataset_fields(actual.variables, expected.variables)
        require_dataset_fields(actual.coordinates, expected.coordinates)


def require_dataset_fields(
    actual: Mapping[str, FieldDescriptor],
    expected: Mapping[str, FieldDescriptor],
) -> None:
    """Require declared fields with their dimensions and scientific metadata."""
    for name, required in expected.items():
        field = actual.get(name)
        if (
            field is None
            or field.dims != required.dims
            or (required.units is not None and field.units != required.units)
            or (required.frame is not None and field.frame != required.frame)
        ):
            raise ValueError(
                f"Dataset field {name!r} requires {required.dims} with the declared units and frame."
            )
