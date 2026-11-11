"""Semantic metadata for labelled datasets carried in array containers."""

from __future__ import annotations

from math import prod
from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    ConfigDict,
    Field,
    JsonValue,
    StrictStr,
    field_validator,
    model_validator,
)

from feedbax.contracts.base import StrictModel
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.strict_json import strict_json_loads


def _valid_name(value: str) -> str:
    if not value or value.strip() != value or not value.isprintable():
        raise ValueError("dataset names must be nonempty printable strings without edge whitespace")
    return value


type Name = Annotated[StrictStr, AfterValidator(_valid_name)]
type DimensionSize = Annotated[int, Field(ge=0, strict=True)]


class FieldDescriptor(StrictModel):
    """A field's labelled dimensions and scientific metadata."""

    model_config = ConfigDict(frozen=True)

    dims: tuple[Name, ...]
    units: StrictStr | None = None
    quantity: StrictStr | None = None
    frame: StrictStr | None = None
    inline_values: tuple[JsonValue, ...] | None = None

    @field_validator("inline_values", mode="before")
    @classmethod
    def _validate_inline_json(cls, value: object) -> object:
        if isinstance(value, (tuple, list)):
            canonical_json_v2_bytes(list(value))
        return value

    @field_validator("dims")
    @classmethod
    def _distinct_dims(cls, dims: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(dims)) != len(dims):
            raise ValueError("a field cannot use the same dimension twice")
        return dims


class DatasetDescriptor(StrictModel):
    """Versioned semantic fields and sizes, independent of physical tensor headers."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal["feedbax.dataset.v1"] = "feedbax.dataset.v1"
    dimensions: dict[Name, DimensionSize]
    variables: dict[Name, FieldDescriptor]
    coords: dict[Name, FieldDescriptor] = Field(default_factory=dict)
    attrs: dict[str, JsonValue] = Field(default_factory=dict)
    schema_id: Name | None = None

    @field_validator("attrs", mode="before")
    @classmethod
    def _validate_attrs_json(cls, value: object) -> object:
        canonical_json_v2_bytes(value)
        return value

    @model_validator(mode="after")
    def _validate_fields(self) -> DatasetDescriptor:
        if overlap := self.variables.keys() & self.coords.keys():
            raise ValueError(f"fields are both variables and coordinates: {sorted(overlap)}")
        for name, field in (*self.variables.items(), *self.coords.items()):
            if unknown := set(field.dims) - self.dimensions.keys():
                raise ValueError(f"field {name!r} uses undeclared dimensions: {sorted(unknown)}")
            if field.inline_values is not None and len(field.inline_values) != prod(
                self.shape(name)
            ):
                raise ValueError(f"inline values for field {name!r} do not match its dimensions")
        return self

    def field(self, name: str) -> FieldDescriptor:
        """Look up a variable or coordinate by its unambiguous name."""
        if name in self.variables:
            return self.variables[name]
        return self.coords[name]

    def shape(self, name: str) -> tuple[int, ...]:
        """Derive a field shape from the sizes of its labelled dimensions."""
        return tuple(self.dimensions[dim] for dim in self.field(name).dims)

    @property
    def tensor_names(self) -> frozenset[str]:
        """The fields whose values are supplied by a numerical tensor source."""
        return frozenset(
            name
            for name, field in (*self.variables.items(), *self.coords.items())
            if field.inline_values is None
        )

    def json_value(self) -> dict[str, JsonValue]:
        """The semantic metadata value embedded by the array container."""
        return self.model_dump(mode="json", exclude_none=True)

    def to_json(self) -> bytes:
        return canonical_json_v2_bytes(self.json_value())

    @classmethod
    def from_json_value(cls, value: JsonValue) -> DatasetDescriptor:
        if not isinstance(value, dict) or "schema_version" not in value:
            raise ValueError("dataset descriptor requires a schema_version")
        return cls.model_validate(value)

    @classmethod
    def from_json(cls, payload: bytes | str) -> DatasetDescriptor:
        return cls.from_json_value(strict_json_loads(payload))
