"""Built-in array declarations for ``ComponentSpec.params`` values."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral, Real
from typing import Annotated, Literal, TypeAlias

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt
from pydantic import TypeAdapter, model_validator


ARRAY_VALUE_SCHEMA_ID = "feedbax.spec.component_param.array_value"
ARRAY_VALUE_SCHEMA_VERSION = f"{ARRAY_VALUE_SCHEMA_ID}.v1"

CanonicalArrayDType: TypeAlias = Literal[
    "bool",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "float32",
    "float64",
]
NonfinitePolicy: TypeAlias = Literal["forbid", "allow"]
NonfiniteToken: TypeAlias = Literal["nan", "+inf", "-inf"]
ArrayScalar: TypeAlias = StrictBool | StrictInt | StrictFloat | NonfiniteToken
SparseCoordinateIndex: TypeAlias = Annotated[StrictInt, Field(ge=0)]
ArrayDimension: TypeAlias = Annotated[StrictInt, Field(gt=0)]


class SparseCooEntrySpec(BaseModel):
    """One coordinate/value pair in a canonical sparse COO declaration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    coordinate: Annotated[tuple[SparseCoordinateIndex, ...], Field(min_length=1)]
    value: ArrayScalar


class _ArrayValueSpecBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["feedbax.spec.component_param.array_value"]
    schema_version: Literal["feedbax.spec.component_param.array_value.v1"]
    shape: Annotated[tuple[ArrayDimension, ...], Field(min_length=1)]
    dtype: CanonicalArrayDType
    nonfinite: NonfinitePolicy


class SparseCooArrayValueSpec(_ArrayValueSpecBase):
    """Versioned sparse COO declaration with an explicit fill value."""

    encoding: Literal["sparse_coo"]
    fill: ArrayScalar
    entries: tuple[SparseCooEntrySpec, ...]

    @model_validator(mode="after")
    def validate_entries(self) -> "SparseCooArrayValueSpec":
        _validate_scalar(self.fill, dtype=self.dtype, nonfinite=self.nonfinite, field="fill")
        occupied: set[tuple[int, ...]] = set()
        for index, entry in enumerate(self.entries):
            coordinate = tuple(entry.coordinate)
            if len(coordinate) != len(self.shape):
                raise ValueError(
                    f"sparse COO entries[{index}] coordinate rank {len(coordinate)} "
                    f"does not match shape rank {len(self.shape)}"
                )
            if any(
                axis_index >= axis_size for axis_index, axis_size in zip(coordinate, self.shape)
            ):
                raise ValueError(
                    f"sparse COO entries[{index}] coordinate {coordinate!r} is outside "
                    f"shape {self.shape!r}"
                )
            if coordinate in occupied:
                raise ValueError(f"sparse COO coordinate {coordinate!r} is duplicated")
            _validate_scalar(
                entry.value,
                dtype=self.dtype,
                nonfinite=self.nonfinite,
                field=f"entries[{index}].value",
            )
            occupied.add(coordinate)
        canonical = tuple(sorted(self.entries, key=lambda entry: tuple(entry.coordinate)))
        if canonical != self.entries:
            object.__setattr__(self, "entries", canonical)
        return self


class ConstantArrayValueSpec(_ArrayValueSpecBase):
    """Versioned declaration broadcasting one scalar over an array shape."""

    encoding: Literal["constant"]
    value: ArrayScalar

    @model_validator(mode="after")
    def validate_value(self) -> "ConstantArrayValueSpec":
        _validate_scalar(self.value, dtype=self.dtype, nonfinite=self.nonfinite, field="value")
        return self


ArrayValueSpec: TypeAlias = Annotated[
    SparseCooArrayValueSpec | ConstantArrayValueSpec,
    Field(discriminator="encoding"),
]

_ARRAY_VALUE_SPEC_ADAPTER = TypeAdapter(ArrayValueSpec)


def _parse_array_value_payload(value: object) -> ArrayValueSpec | None:
    """Validate a claimed array envelope or return None for an ordinary value."""
    if isinstance(value, SparseCooArrayValueSpec | ConstantArrayValueSpec):
        return value
    if not isinstance(value, Mapping):
        return None
    schema_id = value.get("schema_id")
    schema_version = value.get("schema_version")
    claims_id = schema_id == ARRAY_VALUE_SCHEMA_ID
    claims_version = isinstance(schema_version, str) and schema_version.startswith(
        f"{ARRAY_VALUE_SCHEMA_ID}."
    )
    if not claims_id and not claims_version:
        return None
    if schema_id != ARRAY_VALUE_SCHEMA_ID or schema_version != ARRAY_VALUE_SCHEMA_VERSION:
        raise ValueError(
            "array value declarations require both exact reserved tags: "
            f"schema_id={ARRAY_VALUE_SCHEMA_ID!r}, "
            f"schema_version={ARRAY_VALUE_SCHEMA_VERSION!r}; observed "
            f"schema_id={schema_id!r}, schema_version={schema_version!r}"
        )
    return _ARRAY_VALUE_SPEC_ADAPTER.validate_python(value)


def materialize_array_value(spec: ArrayValueSpec) -> np.ndarray:
    """Materialize a built-in declaration as a dense C-order NumPy array.

    Dense versus sparse runtime layout is deliberately a later consumer
    decision. This pure built-in materializer performs no I/O or plugin work.
    """
    validated = _ARRAY_VALUE_SPEC_ADAPTER.validate_python(spec)
    dtype = np.dtype(validated.dtype)
    if isinstance(validated, ConstantArrayValueSpec):
        return np.full(
            validated.shape,
            _materialized_scalar(validated.value),
            dtype=dtype,
            order="C",
        )
    result = np.full(
        validated.shape,
        _materialized_scalar(validated.fill),
        dtype=dtype,
        order="C",
    )
    for entry in validated.entries:
        result[entry.coordinate] = _materialized_scalar(entry.value)
    return result


def _validate_scalar(
    value: ArrayScalar,
    *,
    dtype: CanonicalArrayDType,
    nonfinite: NonfinitePolicy,
    field: str,
) -> None:
    if isinstance(value, str):
        if nonfinite != "allow":
            raise ValueError(f"{field} uses a non-finite token but nonfinite='forbid'")
        if not dtype.startswith("float"):
            raise ValueError(f"{field} non-finite tokens require a floating dtype")
        return
    if isinstance(value, bool):
        if dtype != "bool":
            raise ValueError(f"{field} boolean values require dtype='bool'")
    elif dtype == "bool":
        raise ValueError(f"{field} for dtype='bool' must be a boolean")
    elif dtype.startswith("int") or dtype.startswith("uint"):
        if not isinstance(value, Integral):
            raise ValueError(f"{field} for integer dtype {dtype!r} must be an integer")
        limits = np.iinfo(dtype)
        if not limits.min <= int(value) <= limits.max:
            raise ValueError(f"{field} is outside dtype {dtype!r} range")
    elif not isinstance(value, Real):
        raise ValueError(f"{field} must be numeric")

    if dtype.startswith("float"):
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ValueError(f"{field} must use an explicit non-finite token")
        with np.errstate(over="ignore"):
            converted = np.asarray(numeric, dtype=dtype)
        if not np.isfinite(converted).item():
            raise ValueError(f"{field} is outside dtype {dtype!r} finite range")


def _materialized_scalar(value: ArrayScalar) -> bool | int | float:
    if value == "nan":
        return float("nan")
    if value == "+inf":
        return float("inf")
    if value == "-inf":
        return float("-inf")
    return value


__all__ = [
    "ARRAY_VALUE_SCHEMA_ID",
    "ARRAY_VALUE_SCHEMA_VERSION",
    "ArrayValueSpec",
    "ConstantArrayValueSpec",
    "SparseCooArrayValueSpec",
    "SparseCooEntrySpec",
    "materialize_array_value",
]
