"""Stack gathered dataset members along their labels."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

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
    return left.model_copy(update={"inline_values": None}) == right.model_copy(
        update={"inline_values": None},
    ) and (left.inline_values is None) == (right.inline_values is None)


def _spanning(values: np.ndarray, axes: tuple[str, ...]) -> tuple[np.ndarray, tuple[str, ...]]:
    """Values stacked on ``axes``, keeping only the stacking axes along which they vary."""
    kept = []
    for axis, name in reversed(tuple(enumerate(axes))):
        first = np.take(values, [0], axis=axis)
        if np.all(values == first):
            values = np.take(values, 0, axis=axis)
        else:
            kept.append(name)
    return values, tuple(reversed(kept))


def _member_coordinate(
    name: str, datasets: Sequence[Dataset], shape: tuple[int, ...], axes: tuple[str, ...],
) -> Variable | None:
    """A dataset coordinate stacked on the stacking axes it varies along, or None when every member shares it."""
    fields = [dataset.descriptor.coords[name] for dataset in datasets]
    base = fields[0]
    if any(not _same_field(base, other) for other in fields[1:]):
        raise ValueError(f"Dataset members disagree on field {name!r}")
    arrays = [np.asarray(dataset.data(name)) for dataset in datasets]
    if all(array.dtype == arrays[0].dtype and np.array_equal(array, arrays[0]) for array in arrays[1:]):
        return None
    values, kept = _spanning(np.stack(arrays).reshape((*shape, *arrays[0].shape)), axes)
    return Variable.from_field(values, base, (*kept, *base.dims))


def _stacked(fields: Sequence[FieldDescriptor], axes: tuple[str, ...], name: str) -> FieldDescriptor:
    prototype = fields[0]
    if any(not _same_field(prototype, other) for other in fields[1:]):
        raise ValueError(f"Dataset members disagree on field {name!r}")
    return prototype.model_copy(update={
        "dims": (*axes, *prototype.dims),
        "inline_values": (
            tuple(value for field in fields for value in field.inline_values or ())
            if prototype.inline_values is not None
            else None
        ),
    })


def stack_dataset_members(
    inputs: Sequence[tuple[SweepInputMember, Dataset]],
) -> Dataset:
    """Stack equal-layout members along their labels without reading their variables.

    Labels that vary over the members are the leading dimensions: one per label when
    the members are their complete product in order, else one ``member`` dimension.
    Labels equal in every member are scalar coordinates. Dataset coordinates equal in
    every member are kept once; the others, and the attributes that differ, are stacked
    on only the stacking dimensions along which they vary.
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
            added[name] = Variable(*_spanning(_host_values(values, shape), axes))
    collisions = (set(dimensions) | added.keys()) & (
        base.dimensions.keys() | base.variables.keys() | base.coords.keys()
    )
    if collisions:
        raise ValueError(f"Member labels collide with dataset names: {sorted(collisions)}")
    for name in base.coords:
        if (stacked := _member_coordinate(name, datasets, shape, axes)) is not None:
            added[name] = stacked
    label_data = Dataset.from_arrays({}, coords=added)
    shared = base.coords.keys() - added.keys()

    descriptor = DatasetDescriptor(
        dimensions={**dimensions, **base.dimensions},
        variables={
            name: _stacked([item.variables[name] for item in descriptors], axes, name)
            for name in base.variables
        },
        coords={
            **label_data.descriptor.coords,
            **{name: base.coords[name] for name in base.coords if name in shared},
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


def _is_label(name: str, field: FieldDescriptor, axes: tuple[str, ...]) -> bool:
    return "." not in name and set(field.dims) <= set(axes) and field.inline_values is not None and all(
        value is None or isinstance(value, str) for value in field.inline_values
    )


def dataset_entries(
    member: SweepInputMember, dataset: Dataset,
) -> list[tuple[SweepInputMember, Dataset]]:
    """The labelled entries of a stacked dataset, each keyed by its inner key qualified by ``member`` and carrying
    its inner labels and ``member``'s; a dataset that is no stack is the one entry ``member``."""
    descriptor = dataset.descriptor
    if MEMBER not in descriptor.coords:
        return [(member, dataset)]
    axes = descriptor.coords[MEMBER].dims
    labels = {MEMBER} | {
        name for name, field in descriptor.coords.items() if _is_label(name, field, axes)
    }
    labels |= {f"{label}.{part}" for label in labels for part in ("label", "value")} & descriptor.coords.keys()
    fields = {**descriptor.variables, **descriptor.coords}

    def leading(name: str) -> tuple[str, ...]:
        stacked = tuple(dim for dim in fields[name].dims if dim in axes)
        if fields[name].dims[: len(stacked)] != stacked:
            raise ValueError(f"Field {name!r} does not lead with its stacking dimensions")
        return stacked

    entries = []
    for position in np.ndindex(*(descriptor.dimensions[axis] for axis in axes)):
        at = dict(zip(axes, position, strict=True))

        def prefix(name: str) -> tuple[int, ...]:
            return tuple(at[dim] for dim in leading(name))

        def value(name: str) -> JsonValue:
            return np.asarray(dataset.data(name, prefix(name))).tolist()

        inner = {
            name: SweepCoordinate(
                str(value(name)),
                cast(str | None, value(f"{name}.label")) if f"{name}.label" in labels else None,
                value(f"{name}.value") if f"{name}.value" in labels else None,
            )
            for name in labels if name != MEMBER and "." not in name and value(name) is not None
        }
        if shared := sorted(inner.keys() & member.coordinates.keys()):
            raise ValueError(f"Member {member.key!r} already carries the labels {shared!r}")
        kept = {}
        for name, field in fields.items():
            if name in labels:
                continue
            inline = field.inline_values
            if inline is not None and leading(name):
                inline = tuple(np.asarray(dataset.data(name, prefix(name))).ravel().tolist())
            kept[name] = field.model_copy(update={"dims": field.dims[len(leading(name)):], "inline_values": inline})
        view = DatasetDescriptor(
            dimensions={dim: size for dim, size in descriptor.dimensions.items() if dim not in axes},
            variables={name: kept[name] for name in descriptor.variables},
            coords={name: kept[name] for name in descriptor.coords if name in kept},
            schema_id=descriptor.schema_id,
            attrs=descriptor.attrs,
        )

        def load(name: str, index: tuple[int | slice, ...], prefix=prefix) -> np.ndarray:
            return np.asarray(dataset.data(name, (*prefix(name), *index)))

        entries.append((
            SweepInputMember(f"{member.key}/{value(MEMBER)}", {**member.coordinates, **inner}),
            Dataset.bind(view, tensor_shapes={name: view.shape(name) for name in view.tensor_names}, load_tensor=load),
        ))
    return entries
