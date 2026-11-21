"""Stack gathered dataset members along their labels."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from pydantic import JsonValue

from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.dataset import DatasetDescriptor, FieldDescriptor
from feedbax.contracts.member_stack import MEMBER, StackLayout, stack_layout
from feedbax.contracts.project_sweep import SweepCoordinate, SweepInputMember
from feedbax.dataset import Dataset, Variable


def _host_values(values: Sequence[JsonValue], shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        array[index] = value
    return array if shape is None else array.reshape(shape)


def _scalar(value: JsonValue) -> np.ndarray:
    array = np.empty((), dtype=object)
    array[()] = value
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


def _label_coords(layout: StackLayout) -> dict[str, Variable]:
    keys = [member.key for member in layout.members]
    if not layout.dimensions:
        coords = {MEMBER: Variable(_scalar(keys[0]), ())}
    else:
        coords = {MEMBER: Variable(np.asarray(keys).reshape(tuple(layout.dimensions.values())), layout.dims(MEMBER))}
    for name in layout.varying:
        if layout.values:
            coords.update(_axis_fields(name, name, layout.values[name]))
        else:
            coords.update(_axis_fields(name, MEMBER, [member.coordinates.get(name) for member in layout.members]))
    for name, coordinate in layout.constant.items():
        coords[name] = Variable(_scalar(coordinate.value_id), ())
        if coordinate.label is not None:
            coords[f"{name}.label"] = Variable(_scalar(coordinate.label), ())
        if coordinate.value is not None:
            coords[f"{name}.value"] = Variable(_scalar(coordinate.value), ())
    return coords


def _same_field(left: FieldDescriptor, right: FieldDescriptor) -> bool:
    return (left.dims, left.units, left.quantity, left.frame) == (
        right.dims, right.units, right.quantity, right.frame,
    ) and (left.inline_values is None) == (right.inline_values is None)


def _shared_coordinate(name: str, datasets: Sequence[Dataset]) -> bool:
    """Whether a dataset coordinate is equal in every member, so that it is not stacked."""
    base = datasets[0].descriptor.coords[name]
    if base.inline_values is not None:
        return all(dataset.descriptor.coords[name] == base for dataset in datasets[1:])
    first = np.asarray(datasets[0].data(name))
    for dataset in datasets[1:]:
        actual = np.asarray(dataset.data(name))
        if actual.dtype != first.dtype or not np.array_equal(actual, first):
            return False
    return True


def _stacked(fields: Sequence[FieldDescriptor], axes: tuple[str, ...], name: str) -> FieldDescriptor:
    prototype = fields[0]
    if any(not _same_field(prototype, other) for other in fields[1:]):
        raise ValueError(f"Dataset members disagree on field {name!r}")
    return FieldDescriptor(
        dims=(*axes, *prototype.dims),
        units=prototype.units,
        quantity=prototype.quantity,
        frame=prototype.frame,
        inline_values=(
            tuple(value for field in fields for value in field.inline_values or ())
            if prototype.inline_values is not None
            else None
        ),
    )


def stack_dataset_members(
    inputs: Sequence[tuple[SweepInputMember, Dataset]],
) -> Dataset:
    """Stack equal-layout members along their labels without reading their variables.

    Labels that vary over the members are the leading dimensions: one per label when
    the members are their complete product in order, else one ``member`` dimension.
    Labels equal in every member are scalar coordinates. Dataset coordinates equal in
    every member are kept once; the others, and the attributes that differ, are stacked.
    """
    if not inputs:
        raise ValueError("A dataset stack needs at least one member")
    members, datasets = zip(*inputs, strict=True)
    layout = stack_layout(members)
    dimensions = dict(layout.dimensions)
    added = _label_coords(layout)
    axes = tuple(dimensions)
    shape = tuple(dimensions.values())
    descriptors = [dataset.descriptor for dataset in datasets]
    base = descriptors[0]
    for descriptor in descriptors[1:]:
        if (
            descriptor.dimensions != base.dimensions or descriptor.schema_id != base.schema_id
            or descriptor.variables.keys() != base.variables.keys()
            or descriptor.coords.keys() != base.coords.keys()
        ):
            raise ValueError("Dataset members have different layouts or schema IDs")

    attrs = {}
    for name in dict.fromkeys(name for descriptor in descriptors for name in descriptor.attrs):
        values = [descriptor.attrs.get(name) for descriptor in descriptors]
        if all(canonical_json_v2_bytes(value) == canonical_json_v2_bytes(values[0]) for value in values[1:]):
            attrs[name] = values[0]
        else:
            added[name] = Variable(_host_values(values, shape), axes)
    collisions = (set(dimensions) | added.keys()) & (
        base.dimensions.keys() | base.variables.keys() | base.coords.keys()
    )
    if collisions:
        raise ValueError(f"Member labels collide with dataset names: {sorted(collisions)}")
    label_data = Dataset.from_arrays({}, coords=added)
    shared = {name for name in base.coords if _shared_coordinate(name, datasets)}

    descriptor = DatasetDescriptor(
        dimensions={**dimensions, **base.dimensions},
        variables={
            name: _stacked([item.variables[name] for item in descriptors], axes, name)
            for name in base.variables
        },
        coords={
            **label_data.descriptor.coords,
            **{
                name: base.coords[name] if name in shared
                else _stacked([item.coords[name] for item in descriptors], axes, name)
                for name in base.coords
            },
        },
        schema_id=base.schema_id,
        attrs=attrs,
    )
    positions = np.arange(len(datasets)).reshape(shape)

    def load(name: str, index: tuple[int | slice, ...]) -> np.ndarray:
        if name in added:
            return np.asarray(label_data.data(name, index))
        if name in shared:
            return np.asarray(datasets[0].data(name, index))
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
            raise ValueError(f"Dataset members have different dtypes for {name!r}")
        return np.stack(arrays).reshape((*selected.shape, *arrays[0].shape))

    return Dataset.bind(
        descriptor,
        tensor_shapes={name: descriptor.shape(name) for name in descriptor.tensor_names},
        load_tensor=load,
    )
