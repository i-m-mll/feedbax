"""Labelled arrays and host-side projection onto numerical JAX inputs."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field

import jax
import numpy as np
from pydantic import JsonValue

from feedbax.contracts.dataset import DatasetDescriptor, FieldDescriptor


type Array = jax.Array | np.ndarray
type Dimensions = tuple[str, ...]
type Index = tuple[int | slice, ...]
type TensorLoader = Callable[[str, Index], Array]
type ProjectionField = tuple[str, Dimensions]
type Label = str | int | float | bool | None


@dataclass(frozen=True, eq=False)
class Variable:
    """A resident field with named dimensions and optional scientific metadata."""

    data: Array
    dims: Dimensions
    units: str | None = field(default=None, kw_only=True)
    quantity: str | None = field(default=None, kw_only=True)
    frame: str | None = field(default=None, kw_only=True)


@jax.tree_util.register_dataclass
@dataclass(frozen=True, eq=False)
class Projection:
    """Only numerical leaves and their static field signatures enter JAX."""

    arrays: tuple[Array, ...]
    fields: tuple[ProjectionField, ...] = field(metadata={"static": True})

    def data(self, name: str) -> Array:
        for (field_name, _), array in zip(self.fields, self.arrays, strict=True):
            if field_name == name:
                return array
        raise KeyError(name)


def _is_numeric(data: Array) -> bool:
    return jax.dtypes.issubdtype(data.dtype, np.number) or data.dtype == np.dtype(np.bool_)


def _selected_shape(shape: tuple[int, ...], index: Index) -> tuple[int, ...]:
    if len(index) > len(shape):
        raise IndexError(f"too many indices for shape {shape}")
    selected: list[int] = []
    for dim, item in enumerate(index):
        size = shape[dim]
        if isinstance(item, slice):
            selected.append(len(range(*item.indices(size))))
        elif type(item) is not int:
            raise TypeError("dataset indices must be integers or slices")
        elif not -size <= item < size:
            raise IndexError(f"index {item} out of range for axis {dim} with size {size}")
    return (*selected, *shape[len(index) :])


def _select(data: Array, index: Index) -> Array:
    if not index:
        return data
    value = data[index]
    if isinstance(data, np.ndarray) and not isinstance(value, np.ndarray):
        scalar = np.empty((), dtype=data.dtype)
        scalar[()] = value
        return scalar
    return value


def _inline_array(values: tuple[JsonValue, ...], shape: tuple[int, ...]) -> np.ndarray:
    if all(isinstance(item, str) for item in values):
        return np.asarray(values, dtype=str).reshape(shape)
    result = np.empty(len(values), dtype=object)
    result[:] = values
    return result.reshape(shape)


@dataclass(frozen=True, eq=False)
class Dataset:
    """A semantic descriptor bound to resident or lazy numerical arrays."""

    _descriptor: DatasetDescriptor
    _load_tensor: TensorLoader

    @classmethod
    def from_arrays(
        cls,
        variables: Mapping[str, Variable],
        *,
        coords: Mapping[str, Variable] | None = None,
        attrs: Mapping[str, JsonValue] | None = None,
        schema_id: str | None = None,
    ) -> Dataset:
        """Describe resident arrays, keeping numeric array data by reference."""
        coordinate_fields = coords if coords is not None else {}
        if overlap := variables.keys() & coordinate_fields.keys():
            raise ValueError(f"fields are both variables and coordinates: {sorted(overlap)}")

        dimensions: dict[str, int] = {}
        descriptions: dict[str, FieldDescriptor] = {}
        tensors: dict[str, Array] = {}
        for name, variable in (*variables.items(), *coordinate_fields.items()):
            data = variable.data
            if isinstance(data, np.ma.MaskedArray):
                raise ValueError(f"field {name!r} needs a separate validity variable")
            if len(variable.dims) != data.ndim:
                raise ValueError(
                    f"field {name!r} has dimensions {variable.dims!r} for shape {data.shape}"
                )
            for dim, size in zip(variable.dims, data.shape, strict=True):
                if dimensions.setdefault(dim, size) != size:
                    raise ValueError(
                        f"field {name!r} has an inconsistent size for dimension {dim!r}"
                    )
            inline_values = None
            if isinstance(data, np.ndarray) and data.dtype.kind in ("O", "U"):
                inline_values = tuple(data.ravel().tolist())
            elif _is_numeric(data):
                tensors[name] = data
            else:
                raise ValueError(f"field {name!r} has unsupported dtype {data.dtype}")
            descriptions[name] = FieldDescriptor(
                dims=variable.dims,
                units=variable.units,
                quantity=variable.quantity,
                frame=variable.frame,
                inline_values=inline_values,
            )

        descriptor = DatasetDescriptor(
            dimensions=dimensions,
            variables={name: descriptions[name] for name in variables},
            coords={name: descriptions[name] for name in coordinate_fields},
            attrs={} if attrs is None else dict(attrs),
            schema_id=schema_id,
        )
        return cls(descriptor, lambda name, index: _select(tensors[name], index))

    @classmethod
    def bind(
        cls,
        descriptor: DatasetDescriptor,
        *,
        tensor_shapes: Mapping[str, Sequence[int]],
        load_tensor: TensorLoader,
    ) -> Dataset:
        """Bind authenticated tensor headers and defer reads to the supplied loader."""
        descriptor = DatasetDescriptor.model_validate(descriptor.model_dump(mode="python"))
        descriptor.validate_tensor_shapes(tensor_shapes)
        return cls(descriptor, load_tensor)

    @property
    def descriptor(self) -> DatasetDescriptor:
        """Return a metadata snapshot for inspection or serialization."""
        return self._descriptor.model_copy(deep=True)

    def data(self, name: str, index: Index = ()) -> Array:
        """Read a field or a leading slice, forwarding the selection to its loader."""
        description = self._descriptor.field(name)
        shape = self._descriptor.shape(name)
        selected_shape = _selected_shape(shape, index)
        if description.inline_values is not None:
            return _select(_inline_array(deepcopy(description.inline_values), shape), index)

        data = self._load_tensor(name, index)
        if (
            isinstance(data, np.ma.MaskedArray)
            or data.shape != selected_shape
            or not _is_numeric(data)
        ):
            raise ValueError(f"loader returned an incompatible value for field {name!r}")
        return data

    def select(self, name: str, *, labels: Mapping[str, Label]) -> Array:
        """Resolve one matching index per dimension using one-dimensional coordinates."""
        dims = self._descriptor.field(name).dims
        candidates: dict[int, set[int]] = {}
        for coordinate, label in labels.items():
            coordinate_dims = self._descriptor.coords[coordinate].dims
            if len(coordinate_dims) != 1 or coordinate_dims[0] not in dims:
                raise ValueError(f"coordinate {coordinate!r} cannot select a dimension of {name!r}")
            axis = dims.index(coordinate_dims[0])
            matches = {
                index
                for index, value in enumerate(np.asarray(self.data(coordinate)).tolist())
                if value == label and (type(value) is bool) == (type(label) is bool)
            }
            candidates.setdefault(axis, matches).intersection_update(matches)

        index: list[int | slice] = [slice(None)] * (max(candidates, default=-1) + 1)
        for axis, matches in candidates.items():
            if len(matches) != 1:
                raise ValueError(
                    f"labels select {len(matches)} positions on dimension {dims[axis]!r}"
                )
            index[axis] = next(iter(matches))
        return self.data(name, tuple(index))

    def project(self, *names: str, coordinates: Sequence[str] = ()) -> Projection:
        """Load the declared numerical variables and coordinates for a JAX call."""
        fields = [(name, self._descriptor.variables[name]) for name in names]
        fields.extend((name, self._descriptor.coords[name]) for name in coordinates)
        if len({name for name, _ in fields}) != len(fields):
            raise ValueError("a projection cannot contain the same field twice")
        for name, description in fields:
            if description.inline_values is not None:
                raise ValueError(f"host-only field {name!r} cannot enter a JAX projection")
        return Projection(
            arrays=tuple(self.data(name) for name, _ in fields),
            fields=tuple((name, description.dims) for name, description in fields),
        )
