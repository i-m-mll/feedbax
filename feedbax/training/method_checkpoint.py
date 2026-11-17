"""Encode and restore a Train method's model and auxiliary array state."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields, is_dataclass
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING
from uuid import uuid4

import jax
import jax.numpy as jnp
from jax._src.lax.lax import _convert_element_type
from jaxtyping import PyTree
import numpy as np

from feedbax._atomic import sync_parent_directories, write_bytes_atomic
from feedbax.contracts.artifact_custody import CONTENT_OBJECT_REF_PREFIX, ContentObjectRef
from feedbax.contracts.base import sha256_bytes
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.method_checkpoint import (
    CheckpointLatestPointer,
    MethodCheckpointClock,
    MethodCheckpointExtra,
    MethodCheckpointIdentity,
    MethodCheckpointManifest,
    method_checkpoint_extra_path,
    method_checkpoint_manifest_path,
)
from feedbax.persistence.array_container import (
    SUPPORTED_DTYPES,
    ArrayContainerError,
    EncodedArrayContainer,
    decode_array_container,
    encode_array_container,
)
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.method_checkpoint import (
    LATEST_NAME,
    bind_method_checkpoint_content,
    method_checkpoint_pointer,
    read_checkpoint_file,
    read_method_checkpoint_manifest,
)
from feedbax.persistence.model_tree import MODEL_CHECKPOINT_STORE
from feedbax.persistence.semantic_fingerprint import UnsupportedSemanticValue, fingerprint_semantic_values
from feedbax.training.checkpoint_errors import CheckpointCustodyError
from feedbax.training.model_checkpoint import SemanticModelSlot

if TYPE_CHECKING:
    from feedbax.component_registry import ComponentRegistry
    from feedbax.contracts.artifact_custody import ContentBinding
    from feedbax.runtime.graph import Graph
    from feedbax.training.methods import TrainingState
    from feedbax.training.model_checkpoint import SemanticModelWriter


_EXTRA_SCHEMA = "feedbax.method_extra.v1"


def _is_key(leaf: jax.Array) -> bool:
    return jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.prng_key)


def _extra_layout(extra: PyTree, count: int):
    pairs, structure = jax.tree.flatten_with_path(extra)
    named = []
    for path, leaf in pairs:
        name = "$" + jax.tree_util.keystr(path)
        if not isinstance(leaf, jax.Array) or not leaf.is_fully_addressable:
            raise CheckpointCustodyError(f"{name}: method extra state must be addressable JAX arrays")
        if leaf.ndim < 1 or leaf.shape[0] != count:
            raise CheckpointCustodyError(f"{name}: method extra state does not carry {count} replicas")
        if not _is_key(leaf) and (leaf.dtype.hasobject or leaf.dtype.fields or leaf.dtype.itemsize == 0):
            raise CheckpointCustodyError(f"{name}: cannot persist method state dtype {leaf.dtype}")
        named.append((name, leaf))
    if len({name for name, _ in named}) != len(named):
        raise CheckpointCustodyError("Method extra state has ambiguous leaf paths")
    specs = jax.tree.unflatten(
        structure,
        [
            jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, weak_type=getattr(leaf, "weak_type", False))
            for _, leaf in named
        ],
    )
    try:
        fingerprint = fingerprint_semantic_values(
            specs, header=(_EXTRA_SCHEMA,), project_leaf=_checkpoint_metadata,
        )
    except UnsupportedSemanticValue as error:
        raise CheckpointCustodyError(f"Method extra state structure has no durable identity: {error}") from error
    return named, structure, {"schema": _EXTRA_SCHEMA, "structure_sha256": fingerprint}


def _checkpoint_metadata(value):
    if isinstance(value, jax.ShapeDtypeStruct):
        return value.shape, str(value.dtype), value.weak_type
    if isinstance(value, type) and "<locals>" not in value.__qualname__:
        return value.__module__, value.__qualname__
    if isinstance(value, jax.tree_util.GetAttrKey):
        return value.name
    if isinstance(value, jax.tree_util.SequenceKey):
        return value.idx
    if isinstance(value, (jax.tree_util.DictKey, jax.tree_util.FlattenedIndexKey)):
        return value.key
    if is_dataclass(value) and not isinstance(value, type):
        return tuple((field.name, getattr(value, field.name)) for field in fields(value) if field.compare)
    raise UnsupportedSemanticValue(f"{type(value).__module__}.{type(value).__qualname__}")


def validate_method_extra(extra: PyTree, *, replicas: int) -> None:
    """Check initial persistent state before numerical updates begin."""
    _extra_layout(extra, replicas)


def _encode_extra(extra: PyTree, count: int) -> EncodedArrayContainer:
    named, _, metadata = _extra_layout(extra, count)
    tensors = {}
    for name, leaf in named:
        array = np.asarray(jax.random.key_data(leaf) if _is_key(leaf) else leaf)
        if array.dtype.name not in SUPPORTED_DTYPES:
            array = np.asarray(array, dtype=array.dtype.newbyteorder("<"), order="C").reshape(-1).view(np.uint8)
        tensors[name] = array
    try:
        return encode_array_container(tensors, metadata=metadata)
    except ArrayContainerError as error:
        raise CheckpointCustodyError(f"Cannot encode method extra state: {error}") from error


def _decode_extra(
    raw: bytes, binding: MethodCheckpointExtra, template: PyTree, count: int,
) -> PyTree:
    named, structure, metadata = _extra_layout(template, count)
    try:
        decoded = decode_array_container(raw, expected_descriptor_sha256=binding.descriptor_sha256)
    except ArrayContainerError as error:
        raise CheckpointCustodyError(f"Cannot decode method extra state: {error}") from error
    if decoded.descriptor.metadata != metadata or decoded.tensors.keys() != {name for name, _ in named}:
        raise CheckpointCustodyError("Checkpoint method extra state differs from the initialized structure")
    restored = []
    for name, leaf in named:
        array = decoded.tensors[name]
        key = _is_key(leaf)
        expected = jax.random.key_data(leaf) if key else leaf
        dtype = np.dtype(expected.dtype)
        if dtype.name not in SUPPORTED_DTYPES:
            size = expected.size * dtype.itemsize
            if array.shape != (size,) or array.dtype != np.uint8:
                raise CheckpointCustodyError(f"{name}: checkpoint storage differs from method state dtype")
            array = array.view(dtype.newbyteorder("<")).reshape(expected.shape)
        if array.dtype.name != dtype.name or array.shape != expected.shape:
            raise CheckpointCustodyError(f"{name}: checkpoint tensor does not match method state dtype and shape")
        value = jnp.asarray(array)
        if key:
            value = jax.random.wrap_key_data(value, impl=jax.random.key_impl(leaf))
        elif leaf.weak_type:
            # JAX's public cast strengthens the value, changing promotion after Resume.
            value = _convert_element_type(value, leaf.dtype, weak_type=True)
        restored.append(value)
    return jax.tree.unflatten(structure, restored)


@dataclass(frozen=True)
class MethodCheckpointWrite:
    root: Path
    manifest_path: Path
    latest_pointer: CheckpointLatestPointer
    manifest: MethodCheckpointManifest
    content_binding: ContentBinding | None


@dataclass(frozen=True)
class MethodCheckpoint:
    manifest_path: Path
    manifest: MethodCheckpointManifest
    model: SemanticModelSlot

    @property
    def identity(self) -> MethodCheckpointIdentity:
        return self.manifest.identity

    @property
    def clock(self) -> MethodCheckpointClock:
        return self.manifest.clock

    def restore_extra(self, template: PyTree) -> PyTree:
        extra = self.manifest.extra
        path = self.manifest_path.parent / method_checkpoint_extra_path(extra)
        return _decode_extra(
            read_checkpoint_file(path, extra.object), extra, template, len(self.model.replica_ids),
        )


def write_method_checkpoint(
    root: Path, *, run_id: str, identity: MethodCheckpointIdentity, clock: MethodCheckpointClock,
    state: TrainingState, model_writer: SemanticModelWriter,
    model_templates: Sequence[Graph], replica_ids: tuple[str, ...], final: bool = False,
    content_provider: ImmutableArtifactBlobProvider | None = None,
) -> MethodCheckpointWrite:
    from feedbax.training.method_runtime import replica_value

    transaction_id = "tx-" + uuid4().hex
    relative = method_checkpoint_manifest_path(transaction_id)
    manifest_path = root / relative
    transactions = manifest_path.parent.parent
    transactions.mkdir(parents=True, exist_ok=True)
    encoded = _encode_extra(state.extra, len(replica_ids))
    digest = sha256_bytes(encoded.data)
    extra = MethodCheckpointExtra(
        object=ContentObjectRef(
            object_id=CONTENT_OBJECT_REF_PREFIX + digest, sha256=digest, size_bytes=len(encoded.data),
        ),
        descriptor_sha256=encoded.descriptor_sha256,
    )
    with TemporaryDirectory(prefix=f".{transaction_id}.", dir=transactions) as scratch:
        staged = Path(scratch)
        models = tuple(replica_value(state.model, index) for index in range(len(replica_ids)))
        model = model_writer.publish(
            staged, models, templates=model_templates, replica_ids=replica_ids,
        )
        write_bytes_atomic(staged / method_checkpoint_extra_path(extra), encoded.data)
        manifest = MethodCheckpointManifest(
            transaction_id=transaction_id, run_id=run_id, final=final,
            identity=identity, clock=clock, model=model, extra=extra,
        )
        raw = canonical_json_v2_bytes(manifest.model_dump(mode="json"))
        write_bytes_atomic(staged / manifest_path.name, raw)
        os.rename(staged, manifest_path.parent)
    sync_parent_directories(manifest_path)
    pointer = method_checkpoint_pointer(manifest, raw)
    binding = None
    if content_provider is not None:
        binding = bind_method_checkpoint_content(root, pointer, content_provider)
    write_bytes_atomic(root / LATEST_NAME, canonical_json_v2_bytes(pointer.model_dump(mode="json")))
    return MethodCheckpointWrite(root, manifest_path, pointer, manifest, binding)


def read_method_checkpoint(
    root: Path, pointer: CheckpointLatestPointer, *, components: ComponentRegistry,
) -> MethodCheckpoint:
    manifest = read_method_checkpoint_manifest(root, pointer)
    manifest_path = root / pointer.manifest_relative_path
    provider = ImmutableArtifactBlobProvider(manifest_path.parent / MODEL_CHECKPOINT_STORE)
    model = SemanticModelSlot.open_store(provider, manifest.model.tree, manifest.model.encoding, components=components)
    return MethodCheckpoint(manifest_path, manifest, model)
