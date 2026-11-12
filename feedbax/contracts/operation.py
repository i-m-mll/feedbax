"""Portable discovery of pure operations and their labelled dataset ports."""

from collections.abc import Mapping

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


class OperationDefinition(StrictModel):
    """Data needed to discover, wire, and author a registered operation."""

    key: str
    owner: str
    label: str
    description: str
    param_schema: list[ParamSchema]
    parameter_json_schema: dict[str, JsonValue]
    static_parameters: tuple[str, ...]
    inputs: dict[str, DatasetPort]
    outputs: dict[str, DatasetPort]


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
