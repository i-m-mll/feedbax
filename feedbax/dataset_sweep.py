"""Stack bound dataset members along the authored project sweep coordinates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import product
from math import prod

import numpy as np
from pydantic import JsonValue

from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.dataset import DatasetDescriptor, FieldDescriptor
from feedbax.contracts.project_sweep import (
    SweepCoordinate,
    SweepInputMember,
    sweep_value_identity_bytes,
)
from feedbax.dataset import Dataset, Index, Variable


CONDITION_PROVENANCE = "condition_provenance"
CONDITION_MEMBER = "member"
_CONDITION_ATTRIBUTES = ("effective_graph_sha256", "effective_task_sha256", "variation_sha256")


def _host_values(values: Sequence[JsonValue]) -> np.ndarray:
    array = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        array[index] = value
    return array


def _axis_fields(
    name: str, dim: str, coordinates: Sequence[SweepCoordinate | None]
) -> dict[str, Variable]:
    ids = [coordinate.value_id if coordinate is not None else None for coordinate in coordinates]
    labels = [coordinate.label if coordinate is not None else None for coordinate in coordinates]
    values = [coordinate.value if coordinate is not None else None for coordinate in coordinates]
    fields = {name: Variable(_host_values(ids), (dim,))}
    if any(label is not None for label in labels):
        fields[f"{name}.label"] = Variable(_host_values(labels), (dim,))
    if any(value is not None for value in values):
        types = {type(value) for value in values}
        numeric = None
        if types == {int} and all(0 <= value < 2**64 for value in values):
            if any(value >= 2**63 for value in values):
                numeric = np.asarray(values, dtype=np.uint64)
        if numeric is None and (types == {bool} or types <= {int, float}):
            numeric = np.asarray(values)
        if numeric is None or numeric.dtype.kind not in "biuf" or numeric.tolist() != values:
            numeric = _host_values(values)
        fields[f"{name}.value"] = Variable(numeric, (dim,))
    return fields


def _sweep_layout(
    members: Sequence[SweepInputMember],
) -> tuple[dict[str, int], dict[str, Variable]]:
    by_axis: dict[str, dict[str, SweepCoordinate]] = {}
    for member in members:
        for name, coordinate in member.coordinates.items():
            if name == "member":
                raise ValueError("Sweep axis 'member' is reserved for the bound member dimension")
            values = by_axis.setdefault(name, {})
            prior = values.setdefault(coordinate.value_id, coordinate)
            if prior is not coordinate and (
                prior.label != coordinate.label
                or sweep_value_identity_bytes(prior.value)
                != sweep_value_identity_bytes(coordinate.value)
            ):
                raise ValueError(
                    f"Sweep axis {name!r} gives {coordinate.value_id!r} conflicting values"
                )

    axes = tuple(by_axis)
    rows = [member.coordinates for member in members]
    if (
        axes
        and all(row.keys() == by_axis.keys() for row in rows)
        and len(rows) == prod(len(values) for values in by_axis.values())
        and all(
            tuple(row[axis].value_id for axis in axes) == ids
            for row, ids in zip(rows, product(*by_axis.values()), strict=True)
        )
    ):
        dimensions = {axis: len(by_axis[axis]) for axis in axes}
        coords = {
            "member": Variable(
                np.asarray([member.key for member in members]).reshape(tuple(dimensions.values())),
                axes,
            )
        }
        for axis in axes:
            coords.update(_axis_fields(axis, axis, tuple(by_axis[axis].values())))
        return dimensions, coords

    coords = {"member": Variable(np.asarray([member.key for member in members]), ("member",))}
    for axis in axes:
        coords.update(_axis_fields(axis, "member", [row.get(axis) for row in rows]))
    return {"member": len(members)}, coords


def _stack_fields(
    fields: Sequence[Mapping[str, FieldDescriptor]], axes: tuple[str, ...]
) -> dict[str, FieldDescriptor]:
    baseline = fields[0]
    if any(member.keys() != baseline.keys() for member in fields[1:]):
        raise ValueError("Dataset sweep members have different field names")
    stacked = {}
    for name, prototype in baseline.items():
        for member in fields[1:]:
            other = member[name]
            if (prototype.inline_values is None) != (other.inline_values is None) or (
                prototype.dims,
                prototype.units,
                prototype.quantity,
                prototype.frame,
            ) != (other.dims, other.units, other.quantity, other.frame):
                raise ValueError(f"Dataset sweep members disagree on field {name!r}")
        inline_values = (
            tuple(value for member in fields for value in member[name].inline_values or ())
            if prototype.inline_values is not None
            else None
        )
        stacked[name] = FieldDescriptor(
            dims=(*axes, *prototype.dims),
            units=prototype.units,
            quantity=prototype.quantity,
            frame=prototype.frame,
            inline_values=inline_values,
        )
    return stacked


def join_rollout_conditions(inputs: Sequence[tuple[str, Dataset]]) -> Dataset:
    """Join ordered singleton rollouts on their existing condition dimension."""
    if len(inputs) < 2:
        raise ValueError("Joining rollout conditions requires at least two members")
    keys, datasets = zip(*inputs, strict=True)
    descriptors = [dataset.descriptor for dataset in datasets]
    base = descriptors[0]
    axis = "condition"
    if (
        len(set(keys)) != len(keys)
        or CONDITION_MEMBER in base.variables or CONDITION_MEMBER in base.coords
        or CONDITION_PROVENANCE in base.attrs
        or base.dimensions.get(axis) != 1
        or axis not in base.coords or base.coords[axis].dims != (axis,)
    ):
        raise ValueError("Selected rollout states require distinct member keys and singleton condition coordinates")
    shared = []
    provenance = {}
    for key, item in zip(keys, descriptors, strict=True):
        if (
            item.dimensions != base.dimensions or item.schema_id != base.schema_id
            or item.variables.keys() != base.variables.keys() or item.coords.keys() != base.coords.keys()
        ):
            raise ValueError(f"Selected rollout member {key!r} has an incompatible dataset layout")
        attrs = item.attrs.copy()
        provenance[key] = {name: attrs.pop(name) for name in _CONDITION_ATTRIBUTES if name in attrs}
        shared.append(attrs)
    if any(canonical_json_v2_bytes(attrs) != canonical_json_v2_bytes(shared[0]) for attrs in shared[1:]):
        raise ValueError("Selected rollout members have incompatible shared dataset attributes")

    def field(name: str, prototype: FieldDescriptor) -> FieldDescriptor:
        varies = axis in prototype.dims
        for key, item, dataset in zip(keys[1:], descriptors[1:], datasets[1:], strict=True):
            other = item.field(name)
            normalized = other.model_copy(update={"inline_values": prototype.inline_values}) if varies else other
            if normalized != prototype or (other.inline_values is None) != (prototype.inline_values is None):
                raise ValueError(f"Selected rollout member {key!r} disagrees on dataset field {name!r}")
            if name in base.coords and not varies and prototype.inline_values is None:
                first = np.asarray(datasets[0].data(name))
                actual = np.asarray(dataset.data(name))
                if first.dtype != actual.dtype or not np.array_equal(first, actual):
                    raise ValueError(f"Selected rollout member {key!r} disagrees on coordinate {name!r}")
        if not varies or prototype.inline_values is None:
            return prototype
        values = np.concatenate(
            [np.asarray(dataset.data(name)) for dataset in datasets], axis=prototype.dims.index(axis)
        )
        return prototype.model_copy(update={"inline_values": tuple(values.ravel().tolist())})

    coords = {name: field(name, prototype) for name, prototype in base.coords.items()}
    conditions = coords[axis].inline_values
    if conditions is None or any(not isinstance(value, str) for value in conditions) or len(set(conditions)) != len(keys):
        raise ValueError("Selected rollouts must supply distinct condition IDs")
    coords[CONDITION_MEMBER] = FieldDescriptor(dims=(axis,), inline_values=keys)
    descriptor = DatasetDescriptor(
        dimensions={**base.dimensions, axis: len(keys)},
        variables={name: field(name, prototype) for name, prototype in base.variables.items()},
        coords=coords,
        schema_id=base.schema_id,
        attrs={**shared[0], CONDITION_PROVENANCE: provenance},
    )

    def load(name: str, index: Index) -> np.ndarray:
        dims = base.field(name).dims
        if axis not in dims:
            first = np.asarray(datasets[0].data(name, index))
            if name in base.variables:
                for key, dataset in zip(keys[1:], datasets[1:], strict=True):
                    actual = np.asarray(dataset.data(name, index))
                    if first.dtype != actual.dtype or not np.array_equal(first, actual):
                        raise ValueError(f"Selected rollout member {key!r} disagrees on variable {name!r}")
            return first
        position = dims.index(axis)
        padded = (*index, *(slice(None),) * max(0, position + 1 - len(index)))
        selected = range(len(datasets))[padded[position]]
        before, after = padded[:position], padded[position + 1:]
        if isinstance(selected, int):
            return np.asarray(datasets[selected].data(name, (*before, 0, *after)))
        if not selected:
            return np.asarray(datasets[0].data(name, (*before, slice(0, 0), *after)))
        arrays = [np.asarray(datasets[i].data(name, (*before, slice(None), *after))) for i in selected]
        if any(array.dtype != arrays[0].dtype for array in arrays[1:]):
            raise ValueError(f"Selected rollout members have different dtypes for {name!r}")
        stacked_axis = sum(isinstance(item, slice) for item in before)
        return np.concatenate(arrays, axis=stacked_axis)

    return Dataset.bind(
        descriptor,
        tensor_shapes={name: descriptor.shape(name) for name in descriptor.tensor_names},
        load_tensor=load,
    )


def stack_dataset_members(
    inputs: Sequence[tuple[SweepInputMember, Dataset]],
) -> Dataset:
    """Assemble equal-layout members without reading their numerical fields.

    An ordered complete product has one dimension per sweep axis. Other
    selections retain their original order on the member dimension.
    """
    if not inputs:
        raise ValueError("A dataset sweep needs at least one member")
    members, datasets = zip(*inputs, strict=True)
    if len({member.key for member in members}) != len(members):
        raise ValueError("A dataset sweep cannot bind the same member key twice")

    dimensions, coords = _sweep_layout(members)
    axes = tuple(dimensions)
    coordinate_data = Dataset.from_arrays({}, coords=coords)
    descriptors = [dataset.descriptor for dataset in datasets]
    base = descriptors[0]
    collisions = (set(dimensions) | coords.keys()) & (
        base.dimensions.keys() | base.variables.keys() | base.coords.keys()
    )
    if collisions:
        raise ValueError(f"Sweep coordinates collide with dataset names: {sorted(collisions)}")
    for descriptor in descriptors[1:]:
        if descriptor.dimensions != base.dimensions or descriptor.schema_id != base.schema_id:
            raise ValueError("Dataset sweep members have different dimensions or schema IDs")
        if canonical_json_v2_bytes(descriptor.attrs) != canonical_json_v2_bytes(base.attrs):
            raise ValueError("Dataset sweep metadata differs; declare varying values as fields")

    descriptor = DatasetDescriptor(
        dimensions={**dimensions, **base.dimensions},
        variables=_stack_fields([item.variables for item in descriptors], axes),
        coords={
            **coordinate_data.descriptor.coords,
            **_stack_fields([item.coords for item in descriptors], axes),
        },
        schema_id=base.schema_id,
        attrs=base.attrs,
    )
    positions = np.arange(len(datasets)).reshape(tuple(dimensions.values()))

    def load(name: str, index: tuple[int | slice, ...]) -> np.ndarray:
        if name in coords:
            return np.asarray(coordinate_data.data(name, index))
        selected = positions[index[: len(axes)]]
        inner_index = index[len(axes) :]
        if np.ndim(selected) == 0:
            return np.asarray(datasets[int(selected)].data(name, inner_index))
        if selected.size == 0:
            example = np.asarray(datasets[0].data(name, inner_index))
            return np.empty((*selected.shape, *example.shape), dtype=example.dtype)
        arrays = [
            np.asarray(datasets[int(position)].data(name, inner_index))
            for position in selected.flat
        ]
        if any(array.dtype != arrays[0].dtype for array in arrays[1:]):
            raise ValueError(f"Dataset sweep members have different dtypes for {name!r}")
        return np.stack(arrays).reshape((*selected.shape, *arrays[0].shape))

    return Dataset.bind(
        descriptor,
        tensor_shapes={name: descriptor.shape(name) for name in descriptor.tensor_names},
        load_tensor=load,
    )
