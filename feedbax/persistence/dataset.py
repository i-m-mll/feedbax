"""Persistence for labelled datasets through the shared array container."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from feedbax.contracts.dataset import DatasetDescriptor
from feedbax.dataset import Dataset, Index
from feedbax.persistence.array_container import (
    EncodedArrayContainer,
    encode_array_container,
    open_array_container,
)


def encode_dataset(dataset: Dataset) -> EncodedArrayContainer:
    """Encode the semantic descriptor and all numerical fields as one array object."""
    descriptor = dataset.descriptor
    return encode_array_container(
        {name: np.asarray(dataset.data(name)) for name in descriptor.tensor_names},
        metadata=descriptor.json_value(),
    )


@contextmanager
def open_dataset(path: str | Path, *, expected_descriptor_sha256: str) -> Iterator[Dataset]:
    """Read selected tensors using a descriptor digest from the product binding.

    The reader authenticates each selected tensor independently. Indexing
    applies after that check. A range from an unverified file cannot be checked
    using a whole-tensor digest alone.
    """
    with open_array_container(
        path, expected_descriptor_sha256=expected_descriptor_sha256
    ) as reader:
        descriptor = DatasetDescriptor.from_json_value(reader.descriptor.metadata)

        def load(name: str, index: Index) -> np.ndarray:
            array = reader.read_tensor(name)
            return np.asarray(array[index]) if index else array

        yield Dataset.bind(
            descriptor,
            tensor_shapes={
                name: tensor.shape for name, tensor in reader.descriptor.tensors.items()
            },
            load_tensor=load,
        )
