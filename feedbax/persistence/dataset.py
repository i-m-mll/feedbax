"""Persistence for labelled datasets through the shared array container."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import numpy as np

from feedbax.contracts.dataset import DatasetDescriptor, DatasetProductRef
from feedbax.contracts.request_refs import ContentRef
from feedbax.dataset import Dataset, Index, TensorLoader
from feedbax.persistence.array_container import (
    ArrayContainerDescriptor,
    EncodedArrayContainer,
    encode_array_container,
    open_array_container,
)
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider


def encode_dataset(dataset: Dataset) -> EncodedArrayContainer:
    """Encode the semantic descriptor and all numerical fields as one array object."""
    descriptor = dataset.descriptor
    return encode_array_container(
        {name: np.asarray(dataset.data(name)) for name in descriptor.tensor_names},
        metadata=descriptor.json_value(),
    )


def publish_dataset(store: ImmutableArtifactBlobProvider, dataset: Dataset) -> DatasetProductRef:
    """Encode a dataset locally and publish its exact bytes in immutable custody."""
    encoded = encode_dataset(dataset)
    stored = store.store_object(
        encoded.data,
        validate=partial(_verify_dataset_file, descriptor_sha256=encoded.descriptor_sha256),
    )
    return DatasetProductRef(
        content=ContentRef(sha256=stored.sha256, size=stored.size_bytes),
        descriptor_sha256=encoded.descriptor_sha256,
    )


def _tensor_shapes(container: ArrayContainerDescriptor) -> dict[str, tuple[int, ...]]:
    return {name: tensor.shape for name, tensor in container.tensors.items()}


def import_dataset(
    store: ImmutableArtifactBlobProvider,
    data: bytes,
    expected: DatasetProductRef,
) -> DatasetProductRef:
    """Validate an incoming dataset against its owning binding before publication."""
    store.store_object(
        data,
        expected=expected.object_ref,
        validate=partial(_verify_dataset_file, descriptor_sha256=expected.descriptor_sha256),
    )
    return expected


def _verify_dataset_file(path: Path, *, descriptor_sha256: str) -> None:
    with open_array_container(path, expected_descriptor_sha256=descriptor_sha256) as reader:
        descriptor = DatasetDescriptor.from_json_value(reader.descriptor.metadata)
        descriptor.validate_tensor_shapes(_tensor_shapes(reader.descriptor))
        reader.verify_tensors()


def _bind_dataset(container: ArrayContainerDescriptor, load: TensorLoader) -> Dataset:
    return Dataset.bind(
        DatasetDescriptor.from_json_value(container.metadata),
        tensor_shapes=_tensor_shapes(container),
        load_tensor=load,
    )


@contextmanager
def open_dataset(path: str | Path, *, expected_descriptor_sha256: str) -> Iterator[Dataset]:
    """Authenticate each selected tensor from a file before applying any slice."""
    with _open_dataset(path, descriptor_sha256=expected_descriptor_sha256, verify=True) as dataset:
        yield dataset


@contextmanager
def _open_dataset(path: str | Path, *, descriptor_sha256: str, verify: bool) -> Iterator[Dataset]:
    with open_array_container(
        path, expected_descriptor_sha256=descriptor_sha256
    ) as reader:

        def load(name: str, index: Index) -> np.ndarray:
            if not verify:
                return np.asarray(reader.read_unverified_slice(name, index))
            array = reader.read_tensor(name)
            return np.asarray(array[index]) if index else array

        yield _bind_dataset(reader.descriptor, load)


@contextmanager
def open_stored_dataset(
    store: ImmutableArtifactBlobProvider,
    ref: DatasetProductRef,
    *,
    verify: bool = False,
) -> Iterator[Dataset]:
    """Open a dataset bound by authenticated owning metadata in immutable custody.

    Header and semantic shape checks happen on open. Numeric reads use the
    pinned object and the codec's slice path. With verify=True, authenticate
    each selected whole tensor before slicing. Use the store's verify_object
    method with ref.object_ref to scrub the entire object.
    """
    with (
        store.open_owned_object(ref.object_ref) as path,
        _open_dataset(path, descriptor_sha256=ref.descriptor_sha256, verify=verify) as dataset,
    ):
        yield dataset
