"""Typed continuation slots for the registered standard Studio Train method."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from feedbax.contracts.checkpoint_projection import (
    STANDARD_TRAIN_STATE_MEDIA_TYPE as STANDARD_TRAIN_STATE_MEDIA_TYPE,
)
from feedbax.contracts.checkpoint_initialization import checkpoint_leaf_path
from feedbax.contracts.checkpoints import (
    CheckpointSemanticAbi,
    CheckpointSlotBlobRef,
    CheckpointTransactionManifest,
)
from feedbax.persistence.array_container import (
    ArrayContainerError,
    EncodedArrayContainer,
    decode_array_container,
    encode_array_container,
)
from feedbax.training.checkpoint_errors import CheckpointCustodyError

if TYPE_CHECKING:
    from feedbax.runtime.graph import Graph


TrainStateCodec = Literal["legacy", "standard_train"]
STANDARD_TRAIN_STATE_CODEC_ID = "feedbax.training.standard_train_state"
STANDARD_TRAIN_STATE_CODEC_VERSION = f"{STANDARD_TRAIN_STATE_CODEC_ID}.v1"
STANDARD_TRAIN_SLOTS = {"optimizer": "optimizer", "prng": "prng", "batch_counter": "auxiliary"}
_PRNG_IMPL = "threefry2x32"


class StandardTrainStateError(CheckpointCustodyError):
    def __init__(self, slot: str, message: str):
        self.path = f"/slots/{slot}"
        super().__init__(f"{self.path}: {message}")


def _metadata(slot: str, abi: CheckpointSemanticAbi) -> dict[str, str]:
    metadata = {
        "schema": STANDARD_TRAIN_STATE_CODEC_VERSION,
        "slot": slot,
        "semantic_abi_sha256": abi.fingerprint_sha256,
    }
    if slot == "prng":
        metadata["prng_impl"] = _PRNG_IMPL
    return metadata


def _named_leaves(slot: str, value: Any, abi: CheckpointSemanticAbi):
    pairs, structure = jax.tree.flatten_with_path(value)
    identities = {leaf.raw_path: leaf.identity for leaf in abi.leaves}
    paths = [checkpoint_leaf_path(path) for path, _ in pairs]
    if set(paths) != identities.keys() or len(paths) != len(identities):
        raise StandardTrainStateError(slot, "prepared values disagree with the named slot ABI")
    leaves = []
    for path, (_, leaf) in zip(paths, pairs, strict=True):
        if not isinstance(leaf, jax.Array) or leaf.weak_type:
            raise StandardTrainStateError(slot, f"{path} must be a strongly typed JAX array")
        leaves.append((identities[path], np.asarray(leaf)))
    return leaves, structure


def _validate_clock(slot: str, value: Any, completed_updates: int | None) -> None:
    if slot == "prng" and (
        value.shape != (2,) or value.dtype != jnp.uint32 or str(jr.key_impl(value)) != _PRNG_IMPL
    ):
        raise StandardTrainStateError(slot, "expected the standard raw Threefry update key")
    if slot == "batch_counter" and (
        value.shape != () or value.dtype != jnp.int32 or int(value) != completed_updates
    ):
        raise StandardTrainStateError(slot, "update counter differs from the authenticated coordinate")


def encode_standard_train_slot(
    slot: str, value: Any, abi: CheckpointSemanticAbi, *, completed_updates: int | None,
) -> EncodedArrayContainer:
    leaves, _ = _named_leaves(slot, value, abi)
    _validate_clock(slot, value, completed_updates)
    try:
        return encode_array_container(dict(leaves), metadata=_metadata(slot, abi))
    except ArrayContainerError as exc:
        raise StandardTrainStateError(slot, str(exc)) from exc


def decode_standard_train_slot(
    slot: CheckpointSlotBlobRef, data: bytes, *, template: Any, completed_updates: int | None,
) -> Any:
    name = slot.slot
    abi = slot.semantic_abi
    if abi is None:
        raise StandardTrainStateError(name, "missing named slot ABI")
    digest = slot.metadata.get("array_descriptor_sha256")
    if not isinstance(digest, str):
        raise StandardTrainStateError(name, "missing array descriptor digest")
    try:
        decoded = decode_array_container(data, expected_descriptor_sha256=digest)
    except ArrayContainerError as exc:
        raise StandardTrainStateError(name, str(exc)) from exc
    leaves, structure = _named_leaves(name, template, abi)
    if decoded.descriptor.metadata != _metadata(name, abi):
        raise StandardTrainStateError(name, "array descriptor disagrees with the slot ABI or key implementation")
    if decoded.tensors.keys() != {identity for identity, _ in leaves}:
        raise StandardTrainStateError(name, "array members disagree with the named slot ABI")
    arrays = []
    for identity, expected in leaves:
        array = decoded.tensors[identity]
        if array.shape != expected.shape or array.dtype != expected.dtype:
            raise StandardTrainStateError(name, f"{identity} shape or dtype differs from the template")
        arrays.append(jnp.asarray(array))
    value = jax.tree.unflatten(structure, arrays)
    _validate_clock(name, value, completed_updates)
    return value


def standard_train_scalar_template(slot: str) -> jax.Array:
    if slot == "prng":
        return jr.PRNGKey(0)
    if slot == "batch_counter":
        return jnp.asarray(0, dtype=jnp.int32)
    raise StandardTrainStateError(slot, "missing optimizer template")


def standard_train_optimizer_template(
    manifest: CheckpointTransactionManifest, transaction_dir: Path, model: Graph | None,
) -> Any:
    from feedbax.component_registry import ComponentRegistry
    from feedbax.contracts.graph import GraphSpec
    from feedbax.contracts.studio_training import STUDIO_GRAPH_METHOD, StudioGraphMethodPayload
    from feedbax.contracts.training import TrainingRunSpec
    from feedbax.training.model_checkpoint import MODEL_TREE_MEDIA_TYPE, SemanticModelSlot
    from feedbax.training.studio_executor import standard_train_optimizer
    from feedbax.compiler.trainability import compile_trainability

    projection = manifest.run_contract_binding.canonical_projection
    if projection is None or "training_run_spec" not in projection:
        raise StandardTrainStateError("optimizer", "missing bound standard Train method")
    spec = TrainingRunSpec.model_validate(projection["training_run_spec"])
    if spec.method_ref != STUDIO_GRAPH_METHOD or spec.graph.inline is None:
        raise StandardTrainStateError("optimizer", "expected the registered standard Train method")
    model_record = next((slot for slot in manifest.slots if slot.slot == "model"), None)
    if model_record is None or model_record.media_type != MODEL_TREE_MEDIA_TYPE:
        raise StandardTrainStateError("optimizer", "missing the matching semantic model layout")
    if model is None:
        model = SemanticModelSlot.open(transaction_dir, model_record).template()
    graph = GraphSpec.model_validate(spec.graph.inline)
    _, filter_spec = compile_trainability(graph, model, ComponentRegistry())
    payload = StudioGraphMethodPayload.model_validate(spec.method_payload.payload)
    optimizer = standard_train_optimizer(payload, gradient_clip=spec.training_config.grad_clip)
    return optimizer.init(eqx.filter(model, filter_spec))
