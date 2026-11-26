"""Portable discovery of operations and their typed ports."""

from collections.abc import Collection, Mapping
from typing import Annotated, Literal

from pydantic import Field, JsonValue

from feedbax.contracts.base import StrictModel
from feedbax.contracts.dataset import REPLICATE_DIM, FieldDescriptor, replicate_axis_index
from feedbax.contracts.graph import ParamSchema


class DatasetPort(StrictModel):
    """Dataset schemas and fields accepted or produced by a port."""

    schema_ids: tuple[str, ...] = Field(min_length=1)
    variables: dict[str, FieldDescriptor] = Field(default_factory=dict)
    selected_variables: dict[str, FieldDescriptor] = Field(default_factory=dict)
    coordinates: dict[str, FieldDescriptor] = Field(default_factory=dict)
    optional_dimensions: tuple[str, ...] = Field(default=(), exclude_if=lambda value: not value)
    """Declared scientific axes preserved when present in the bound dataset."""

    @property
    def dimensions(self) -> frozenset[str]:
        return frozenset(
            dimension for fields in (self.variables, self.coordinates)
            for field in fields.values() for dimension in field.dims
        )

    @property
    def uses_replicate_axis(self) -> bool:
        return REPLICATE_DIM in self.dimensions or any(
            REPLICATE_DIM in field.dims for field in self.selected_variables.values()
        )

    def with_replicate_axis(
        self, coordinate: FieldDescriptor, *, after: str | None = None,
    ) -> "DatasetPort":
        if self.uses_replicate_axis or REPLICATE_DIM in self.variables or REPLICATE_DIM in self.coordinates:
            raise ValueError("Mapped results must describe a single replicate")

        def replicate(field: FieldDescriptor) -> FieldDescriptor:
            dims = field.dims
            axis = replicate_axis_index(dims, after)
            return field.model_copy(update={
                "dims": (*dims[:axis], REPLICATE_DIM, *dims[axis:]), "inline_values": None,
            })

        return self.model_copy(update={
            "variables": {name: replicate(field) for name, field in self.variables.items()},
            "coordinates": {REPLICATE_DIM: coordinate, **self.coordinates},
        })

    def for_dimensions(self, dimensions: Collection[str]) -> "DatasetPort":
        return self._without_dimensions(set(self.optional_dimensions) - set(dimensions))

    def _without_dimensions(self, absent: set[str]) -> "DatasetPort":
        if not absent:
            return self

        def fitted(fields: dict[str, FieldDescriptor]) -> dict[str, FieldDescriptor]:
            return {
                name: field.model_copy(update={
                    "dims": tuple(dim for dim in field.dims if dim not in absent), "inline_values": None,
                }) if absent.intersection(field.dims) else field
                for name, field in fields.items()
            }

        return self.model_copy(update={
            "variables": fitted(self.variables),
            "selected_variables": fitted(self.selected_variables),
            "coordinates": fitted({
                name: field for name, field in self.coordinates.items()
                if not field.dims or not absent.issuperset(field.dims)
            }),
        })


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


class DatasetOutputPort(DatasetPort):
    carries: str | None = Field(default=None, exclude_if=lambda value: value is None)
    """The input role whose labelled entries the output carries, when roles it takes whole carry different ones."""


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
    outputs: dict[str, DatasetOutputPort]
    numerical: JaxOperation | None = None

def operation_input_port(actual: DatasetPort, expected: DatasetPort) -> DatasetPort:
    """The input visible to one invocation after any implicit replica mapping."""
    if not actual.uses_replicate_axis or expected.uses_replicate_axis:
        return actual
    return actual._without_dimensions({REPLICATE_DIM})


def require_dataset_port(actual: DatasetPort, expected: DatasetPort, *, fields: bool = True) -> DatasetPort:
    if not set(actual.schema_ids) <= set(expected.schema_ids):
        raise ValueError(f"Requires dataset schemas {expected.schema_ids}, got {actual.schema_ids}.")
    actual = operation_input_port(actual, expected)
    if fields:
        expected = expected.for_dimensions(actual.dimensions)
        require_dataset_fields(actual.variables, expected.variables)
        require_dataset_fields(actual.coordinates, expected.coordinates)
    return actual


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
