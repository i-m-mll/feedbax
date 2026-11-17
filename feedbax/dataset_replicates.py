"""Read and write the labelled replicate axis shared by numerical products."""

from collections.abc import Sequence

import numpy as np

from feedbax.contracts.dataset import DatasetDescriptor
from feedbax.dataset import Dataset, Index, Variable


REPLICATE_DIM = "replicate"


def replicate_ids(dataset: Dataset) -> tuple[str, ...] | None:
    descriptor = dataset.descriptor
    if REPLICATE_DIM not in descriptor.dimensions:
        return None
    coordinate = descriptor.coords.get(REPLICATE_DIM)
    if coordinate is None or coordinate.dims != (REPLICATE_DIM,):
        raise ValueError("Replicated data requires a replicate coordinate with one ID per member")
    values = tuple(np.asarray(dataset.data(REPLICATE_DIM)).tolist())
    if not values or any(not isinstance(value, str) or not value for value in values) or len(set(values)) != len(values):
        raise ValueError("Dataset replicate IDs must be nonempty distinct strings")
    return values


def select_replicate(dataset: Dataset, position: int) -> Dataset:
    source = dataset.descriptor
    dimensions = dict(source.dimensions)
    size = dimensions.pop(REPLICATE_DIM)
    if not 0 <= position < size:
        raise IndexError(f"Dataset replicate position {position} exceeds {size} members")

    def project(name, field):
        if REPLICATE_DIM not in field.dims:
            return field
        axis = field.dims.index(REPLICATE_DIM)
        inline = field.inline_values
        if inline is not None:
            inline = tuple(np.take(dataset.data(name), position, axis=axis).ravel().tolist())
        return field.model_copy(update={"dims": field.dims[:axis] + field.dims[axis + 1 :], "inline_values": inline})

    descriptor = DatasetDescriptor(
        dimensions=dimensions,
        variables={name: project(name, field) for name, field in source.variables.items()},
        coords={name: project(name, field) for name, field in source.coords.items() if name != REPLICATE_DIM},
        attrs=source.attrs,
        schema_id=source.schema_id,
    )

    def load(name: str, index: Index):
        dims = source.field(name).dims
        if REPLICATE_DIM not in dims:
            return dataset.data(name, index)
        axis = dims.index(REPLICATE_DIM)
        padded = (*index, *(slice(None),) * max(0, axis - len(index)))
        return dataset.data(name, (*padded[:axis], position, *padded[axis:]))

    return Dataset.bind(
        descriptor, tensor_shapes={name: descriptor.shape(name) for name in descriptor.tensor_names},
        load_tensor=load,
    )


def stack_replicates(ids: tuple[str, ...], datasets: Sequence[Dataset]) -> Dataset:
    if not datasets or len(ids) != len(datasets) or len(set(ids)) != len(ids):
        raise ValueError("Replica IDs must name every dataset once")
    descriptors = [dataset.descriptor for dataset in datasets]
    first = descriptors[0]
    if REPLICATE_DIM in first.dimensions or REPLICATE_DIM in first.variables or REPLICATE_DIM in first.coords:
        raise ValueError("Mapped results must describe a single replicate")
    for descriptor in descriptors[1:]:
        if (
            descriptor.dimensions != first.dimensions or descriptor.attrs != first.attrs
            or descriptor.schema_id != first.schema_id or descriptor.variables.keys() != first.variables.keys()
            or descriptor.coords.keys() != first.coords.keys()
        ):
            raise ValueError("Replicate results have different dataset layouts or attributes")

    def collect(name, field, *, coordinate=False):
        arrays = [np.asarray(dataset.data(name)) for dataset in datasets]
        metadata = field.model_dump(exclude={"dims", "inline_values"})
        if any(
            descriptor.field(name).dims != field.dims
            or descriptor.field(name).model_dump(exclude={"dims", "inline_values"}) != metadata
            or array.dtype != arrays[0].dtype
            for descriptor, array in zip(descriptors[1:], arrays[1:], strict=True)
        ):
            raise ValueError(f"Replicate results have inconsistent metadata or dtype for {name!r}")
        if coordinate and all(np.array_equal(arrays[0], array) for array in arrays[1:]):
            return Variable(arrays[0], field.dims, **metadata)
        return Variable(np.stack(arrays), (REPLICATE_DIM, *field.dims), **metadata)

    return Dataset.from_arrays(
        {name: collect(name, field) for name, field in first.variables.items()},
        coords={
            REPLICATE_DIM: Variable(np.asarray(ids), (REPLICATE_DIM,)),
            **{name: collect(name, field, coordinate=True) for name, field in first.coords.items()},
        },
        attrs=first.attrs, schema_id=first.schema_id,
    )
