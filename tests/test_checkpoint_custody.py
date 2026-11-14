"""Typed Train checkpoint custody and state identity."""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from types import FunctionType

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from equinox.nn import StateIndex

import feedbax.training.checkpoint_custody as custody_module
from feedbax.contracts.checkpoints import (
    CheckpointContinuationRequest,
    CheckpointForkBarrierMapping,
    SlotLeafFingerprint,
    StructuralAbiFingerprint,
    structural_abi_content_projection,
    structural_abi_content_sha256,
)
from feedbax.contracts.checkpoint_initialization import checkpoint_leaf_path
from feedbax.contracts.graph import GraphSpec
from feedbax.contracts.worker import MaterializedSlotAxisBinding, ProgressCoordinate
from feedbax.training.checkpoint_custody import (
    CheckpointIntegrityError,
    CheckpointReferenceResolutionError,
    CheckpointSemanticStateUnsupported,
    authenticate_published_checkpoint_custody,
    load_checkpoint_latest_pointer_json,
    load_latest_checkpoint,
    resolve_checkpoint_custody_ref,
    write_checkpoint_transaction,
)
from feedbax.training.managed_studio import prepare_managed_training
from feedbax.training.model_checkpoint import SemanticModelWriter
from feedbax.runtime.components import identity_activation
from tests.support.managed_train_fixture import managed_train_spec


def _write_resolver_checkpoint(root):
    spec, registries = managed_train_spec(batches=2)
    prepared = prepare_managed_training(spec, registries)
    state = prepared.kernel_context["studio_graph_update"].reported_update(prepared.initial_slots)
    return write_checkpoint_transaction(
        root,
        run_spec=spec,
        phase_program=spec.worker_execution.effective_phase.phase_program,
        barrier_name="after_train_batch",
        coordinate=ProgressCoordinate(
            run_id="custody", phase="train_batch", program_step=1,
            completed_barrier="after_train_batch",
        ),
        slots=state,
        completed_training_batches=1,
        semantic_model=SemanticModelWriter.for_graph(GraphSpec.model_validate(spec.graph.inline), None),
    )


def _resolver_parent_ref(checkpoint):
    return authenticate_published_checkpoint_custody(checkpoint.root).parent_ref


def test_checkpoint_transaction_derives_slots_and_loads_multi_slot_state(tmp_path):
    written = _write_resolver_checkpoint(tmp_path)
    resolved = resolve_checkpoint_custody_ref(_resolver_parent_ref(written), allowed_root=tmp_path)
    assert set(resolved.slots) == {"model", "optimizer", "prng", "batch_counter"}
    assert int(resolved.slots["batch_counter"]) == 1
    spec, registries = managed_train_spec(batches=2)
    prepared = prepare_managed_training(spec, registries)
    resumed = load_latest_checkpoint(
        tmp_path,
        expected_run_spec=spec,
        expected_phase_program=spec.worker_execution.effective_phase.phase_program,
        expected_slots=prepared.initial_slots,
        published_pointer=written.latest_pointer,
    )
    for name in ("optimizer", "prng", "batch_counter"):
        assert eqx.tree_equal(resolved.slots[name], resumed.slots[name])
    assert resolved.manifest.run_contract_binding.canonical_projection_sha256


def test_checkpoint_resolver_rejects_modified_typed_slot(tmp_path):
    written = _write_resolver_checkpoint(tmp_path)
    parent = _resolver_parent_ref(written)
    counter = next(slot for slot in written.manifest.slots if slot.slot == "batch_counter")
    path = written.transaction_dir / counter.relative_path
    path.write_bytes(path.read_bytes()[:-1] + b"!")
    with pytest.raises(CheckpointReferenceResolutionError, match="hash mismatch"):
        resolve_checkpoint_custody_ref(parent, allowed_root=tmp_path, slot_names=["batch_counter"])


def test_checkpoint_latest_reader_rejects_historical_schema(tmp_path):
    written = _write_resolver_checkpoint(tmp_path)
    payload = written.latest_pointer.model_dump(mode="json")
    payload["schema_version"] = "feedbax.manifest.training_checkpoint.latest.v1"
    with pytest.raises(CheckpointIntegrityError, match="unsupported checkpoint latest schema_version"):
        load_checkpoint_latest_pointer_json(json.dumps(payload))


def _different_stable_activation(value: object) -> object:
    return value


def _closure_activation():
    captured = object()

    def activation(value: object) -> object:
        return (captured, value)[1]

    return activation


def _state_index_with_marker(marker: str) -> StateIndex:
    index = StateIndex(jnp.zeros(1, dtype=jnp.float32))
    object.__setattr__(index, "marker", marker)
    return index


def test_structural_abi_digest_binds_content_and_ignores_placement() -> None:
    fingerprint = StructuralAbiFingerprint(
        treedef="PyTreeDef(*)",
        leaf_count=1,
        leaves=[SlotLeafFingerprint(path="[0]", leaf_type="jax.Array", shape=(2,), dtype="float32")],
        fingerprint_sha256="not-part-of-content",
    )
    relocated = fingerprint.model_copy(update={
        "leaves": [fingerprint.leaves[0].model_copy(update={
            "sharding": "other device", "layout": "other layout",
        })],
        "metadata": {"operator_note": "relocated"},
        "fingerprint_sha256": "also-not-content",
    })
    reshaped = fingerprint.model_copy(update={
        "leaves": [fingerprint.leaves[0].model_copy(update={"shape": (3,)})],
    })

    assert structural_abi_content_sha256(fingerprint) == structural_abi_content_sha256(relocated)
    assert structural_abi_content_sha256(fingerprint) != structural_abi_content_sha256(reshaped)


def test_structural_abi_content_projection_preserves_tolerant_migration_shape() -> None:
    raw = {
        "fingerprint_algorithm_version": ("feedbax.training_checkpoint.structural_abi.content.v2"),
        "treedef": "PyTreeDef(*)",
        "leaf_count": 1,
        "leaves": [
            "ignored malformed historical leaf",
            {
                "path": "[0]",
                "leaf_type": "jax.Array",
                "shape": None,
                "dtype": "float32",
                "weak_type": False,
                "sharding": "not-content",
                "layout": "not-content",
            },
        ],
        "environment_provenance": {"python_version": "3", "x64_enabled": False},
        "metadata": {"note": "not-content"},
    }

    projection = structural_abi_content_projection(raw)
    assert projection["leaves"] == [
        {
            "path": "[0]",
            "leaf_type": "jax.Array",
            "dtype": "float32",
            "weak_type": False,
        }
    ]
    assert structural_abi_content_sha256(raw) == custody_module._canonical_hash(projection)


def _legacy_leaf_content_digests(value: object) -> list:
    """Leaf digests recomputed with the original whole-tree host-copy formula.

    Leaf hashing used to take one ``jax.device_get`` over the whole slot and then
    a ``tobytes()`` copy per leaf. Those digests are content-addressed durable
    checkpoint state, so hashing an array buffer in place must reproduce exactly
    the same bytes, sizes, and hashes for every dtype and memory layout.
    """
    pairs, _ = custody_module.jt.flatten_with_path(value)
    host_leaves = custody_module.jt.leaves(jax.device_get(value))
    digests = []
    for (path, leaf), host_leaf in zip(pairs, host_leaves, strict=True):
        if custody_module.eqx.is_array(leaf):
            data = np.asarray(host_leaf).tobytes()
        else:
            data = pickle.dumps(leaf, protocol=pickle.HIGHEST_PROTOCOL)
        digests.append(
            custody_module.SlotLeafContentDigest(
                path=checkpoint_leaf_path(path),
                sha256=custody_module.sha256_bytes(data),
                size_bytes=len(data),
            )
        )
    return digests


def _dtype_and_layout_rich_value() -> dict[str, object]:
    """A slot value spanning the dtypes and buffer layouts leaf hashing must handle."""
    return {
        "f32": jnp.arange(12, dtype=jnp.float32).reshape(3, 4),
        "f32_scalar": jnp.array(1.5, dtype=jnp.float32),
        "bf16": jnp.arange(6, dtype=jnp.bfloat16),
        "f16": jnp.arange(6, dtype=jnp.float16),
        "i32": jnp.arange(5, dtype=jnp.int32),
        "i8": jnp.arange(5, dtype=jnp.int8),
        "u32": jnp.array([11, 22], dtype=jnp.uint32),
        "bool": jnp.array([True, False, True]),
        "c64": jnp.array([1 + 2j, 3 - 4j], dtype=jnp.complex64),
        "empty": jnp.zeros((0, 3), dtype=jnp.float32),
        "np_c_order": np.arange(9, dtype=np.float64).reshape(3, 3),
        "np_f_order": np.asfortranarray(np.arange(9, dtype=np.float32).reshape(3, 3)),
        "np_strided": np.arange(24, dtype=np.float32).reshape(4, 6)[:, ::2],
        "np_0d": np.array(7.5, dtype=np.float32),
        "np_bf16": np.arange(4, dtype=jnp.bfloat16.dtype),
        "static_tuple": ("kept", 3),
        "static_str": "hello",
        "static_float": 0.125,
        "static_none": None,
    }


def test_slot_leaf_digests_match_host_copy_bytes_for_every_dtype_and_layout() -> None:
    value = _dtype_and_layout_rich_value()

    integrity = custody_module._slot_integrity_records(value)

    assert integrity.leaf_digests == _legacy_leaf_content_digests(value)
    # Fortran-ordered and strided buffers must still hash their C-order bytes.
    by_path = {digest.path: digest for digest in integrity.leaf_digests}
    assert by_path["/np_f_order"].size_bytes == 36
    assert by_path["/np_strided"].size_bytes == 48
    assert by_path["/np_f_order"].sha256 == custody_module.sha256_bytes(
        np.asarray(value["np_f_order"]).tobytes()
    )
    assert by_path["/np_strided"].sha256 == custody_module.sha256_bytes(
        np.asarray(value["np_strided"]).tobytes()
    )


def test_semantic_state_root_ignores_pickle_memo_identity_but_preserves_semantics() -> None:
    def equal_string() -> str:
        return "".join(("semantic checkpoint ", "identity"))

    shared = equal_string()
    shared_value = {"left": shared, "right": shared}
    unshared_value = {"left": equal_string(), "right": equal_string()}
    assert shared_value == unshared_value
    assert shared_value["left"] is shared_value["right"]
    assert unshared_value["left"] is not unshared_value["right"]
    shared_blob = pickle.dumps(shared_value, protocol=pickle.HIGHEST_PROTOCOL)
    unshared_blob = pickle.dumps(unshared_value, protocol=pickle.HIGHEST_PROTOCOL)
    assert hashlib.sha256(shared_blob).digest() != hashlib.sha256(unshared_blob).digest()

    def root(value: object) -> str:
        return custody_module.semantic_state_root_sha256(
            slot="model",
            role="model",
            value=value,
            structural_abi_fingerprint=custody_module.structural_abi_fingerprint(value),
            materialized_axes=None,
            population=None,
        )

    assert root(shared_value) == root(unshared_value)
    assert root({"left": shared, "right": shared + " changed"}) != root(shared_value)
    assert root({"left": "a", "right": "b"}) != root({"left": "b", "right": "a"})


def _semantic_model_root(value: object) -> str:
    return custody_module.semantic_state_root_sha256(
        slot="model",
        role="model",
        value=value,
        structural_abi_fingerprint=custody_module.structural_abi_fingerprint(value),
        materialized_axes=None,
        population=None,
    )


def test_semantic_state_root_retains_prior_supported_value_projection() -> None:
    value = {
        "array": jnp.asarray([0.5, 1.0], dtype=jnp.float32),
        "count": 3,
        "label": "retained",
        "flag": True,
        "none": None,
        "tuple": (2.5, b"x"),
    }
    assert _semantic_model_root(value) == (
        "8ed77d8c455e9f631b8a94c18c904127567569f71c04fe9c3b210fd0f4a77797"
    )


def test_global_function_leaf_identity_survives_pickle_and_discriminates_functions() -> None:
    restored = pickle.loads(pickle.dumps(identity_activation))
    baseline = custody_module.structural_abi_fingerprint(identity_activation)
    assert custody_module.structural_abi_fingerprint(restored) == baseline
    assert _semantic_model_root(restored) == _semantic_model_root(identity_activation)
    assert _semantic_model_root(_different_stable_activation) != _semantic_model_root(restored)
    assert custody_module.structural_abi_fingerprint(_different_stable_activation) != baseline


@pytest.mark.parametrize(
    "value",
    [
        _closure_activation(),
        lambda value: value,
        FunctionType(identity_activation.__code__, identity_activation.__globals__),
    ],
)
def test_ambiguous_function_leaf_remains_unsupported(value) -> None:
    with pytest.raises(CheckpointSemanticStateUnsupported, match="unsupported or ambiguous"):
        _semantic_model_root(value)


def test_semantic_state_root_projects_nested_typed_dataclass_state() -> None:
    @dataclass(frozen=True, slots=True)
    class ClampState:
        value: float
        count: int

    @dataclass(frozen=True, slots=True)
    class MethodState:
        clamp: ClampState
        enabled: bool

    def root(value: object) -> str:
        return custody_module.semantic_state_root_sha256(
            slot="method",
            role="method",
            value=value,
            structural_abi_fingerprint=custody_module.structural_abi_fingerprint(value),
            materialized_axes=None,
            population=None,
        )

    baseline = MethodState(clamp=ClampState(value=0.5, count=3), enabled=True)
    equal = MethodState(clamp=ClampState(value=0.5, count=3), enabled=True)
    changed = MethodState(clamp=ClampState(value=0.5, count=4), enabled=True)

    assert root(equal) == root(baseline)
    assert root(changed) != root(baseline)


def test_semantic_state_root_rejects_typed_dataclass_cycles() -> None:
    @dataclass(frozen=True, slots=True)
    class RecursiveState:
        child: object | None = None

    value = RecursiveState()
    object.__setattr__(value, "child", value)

    with pytest.raises(CheckpointSemanticStateUnsupported, match="dataclass cycle"):
        custody_module.semantic_state_root_sha256(
            slot="method",
            role="method",
            value=value,
            structural_abi_fingerprint=custody_module.structural_abi_fingerprint(value),
            materialized_axes=None,
            population=None,
        )


def test_semantic_state_root_normalizes_only_anonymous_state_index_markers() -> None:
    anonymous_left = StateIndex(jnp.zeros(1, dtype=jnp.float32))
    anonymous_right = StateIndex(jnp.zeros(1, dtype=jnp.float32))
    named_left = _state_index_with_marker("stable-left")
    named_right = _state_index_with_marker("stable-right")

    def root(value: object) -> str:
        return custody_module.semantic_state_root_sha256(
            slot="model",
            role="model",
            value=value,
            structural_abi_fingerprint=custody_module.structural_abi_fingerprint(value),
            materialized_axes=None,
            population=None,
        )

    assert root(anonymous_left) == root(anonymous_right)
    assert root(named_left) != root(named_right)


def test_semantic_state_root_binds_role_axes_population_and_rejects_ambiguous_leaves() -> None:
    value = jnp.array([1.0, 2.0], dtype=jnp.float32)
    fingerprint = custody_module.structural_abi_fingerprint(value)
    mapped = MaterializedSlotAxisBinding(
        axis="member",
        role="member",
        size=2,
        level=0,
        mode="mapped",
        array_axis=0,
    )
    population = custody_module.PopulationIdentityRecord(
        slot="model",
        length=2,
        member_ids=["member-a", "member-b"],
    )

    def root(*, role: str = "model", axes=(), population_record=None) -> str:
        return custody_module.semantic_state_root_sha256(
            slot="model",
            role=role,
            value=value,
            structural_abi_fingerprint=fingerprint,
            materialized_axes=axes,
            population=population_record,
        )

    baseline = root()
    assert root(role="optimizer") != baseline
    assert root(axes=(mapped,)) != baseline
    assert root(population_record=population) != baseline
    with pytest.raises(CheckpointSemanticStateUnsupported, match="unsupported or ambiguous"):
        custody_module.semantic_state_root_sha256(
            slot="model",
            role="model",
            value={"leaf": object()},
            structural_abi_fingerprint=custody_module.structural_abi_fingerprint(
                {"leaf": object()}
            ),
            materialized_axes=None,
            population=None,
        )


def test_checkpoint_continuation_rejects_unknown_schema_version() -> None:
    with pytest.raises(ValueError, match="migration_intentionally_absent=yes"):
        CheckpointContinuationRequest.model_validate(
            {
                "schema_version": "feedbax.spec.training_checkpoint_continuation.v0",
                "source_completed_batches": 12000,
                "additional_batches": 200,
            }
        )


def test_checkpoint_continuation_v1_is_explicitly_rejected() -> None:
    with pytest.raises(ValueError, match="migration_intentionally_absent=yes"):
        CheckpointContinuationRequest.model_validate(
            {
                "schema_version": "feedbax.spec.training_checkpoint_continuation.v1",
                "source_completed_batches": 12_000,
                "additional_batches": 200,
            }
        )


def test_distinct_barrier_mapping_requires_explicit_coordinate_provenance() -> None:
    with pytest.raises(
        ValueError, match="require explicit target_coordinate and coordinate_mapping"
    ):
        CheckpointForkBarrierMapping(
            source_barrier="after_train_chunk",
            target_barrier="after_adaptive_epsilon_train_chunk",
        )
