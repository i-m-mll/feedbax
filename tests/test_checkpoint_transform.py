from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
import pytest

from feedbax.contracts.checkpoint_initialization import (
    CheckpointInitializationRequest,
    CheckpointSlotStructure,
    CheckpointStructure,
    derive_checkpoint_semantic_abis,
    lower_checkpoint_initialization,
)
from feedbax.contracts.base import ParentRef, authenticated_manifest_ref_metadata
from feedbax.contracts.worker import OptimizerTargetBinding
from feedbax.training.checkpoint_custody import structural_abi_fingerprint
from feedbax.training.checkpoint_transform import (
    AddedParameterAxis,
    CheckpointTransformErrorCode,
    CheckpointTransformRefused,
    apply_semantic_checkpoint_initialization,
    derive_and_apply_semantic_checkpoint_transform,
)


def _values(input_channels: int, *, initialized_value: float = 0.0):
    model = {
        "nodes": {
            "controller": {
                "input": {
                    "weight": jnp.full((2, input_channels), initialized_value, dtype=jnp.float32),
                    "bias": jnp.full((2,), initialized_value, dtype=jnp.float32),
                }
            }
        }
    }
    optimizer = optax.adam(1e-3)
    return model, optimizer, optimizer.init(model)


def _structure(values):
    model, _, optimizer_state = values
    slot_values = {"model": model, "optimizer": optimizer_state}
    fingerprints = {slot: structural_abi_fingerprint(value) for slot, value in slot_values.items()}
    semantic_abis = derive_checkpoint_semantic_abis(
        fingerprints,
        slot_roles={"model": "model", "optimizer": "optimizer"},
        optimizer_bindings=[
            OptimizerTargetBinding(
                name="optimizer_to_model",
                optimizer_slot="optimizer",
                target_slot="model",
            )
        ],
    )
    return (
        CheckpointStructure(
            slots=[
                CheckpointSlotStructure(
                    slot=slot,
                    role="model" if slot == "model" else "optimizer",
                    treedef=fingerprints[slot].treedef,
                    leaves=fingerprints[slot].leaves,
                    semantic_abi=semantic_abis[slot],
                )
                for slot in ("model", "optimizer")
            ]
        ),
        slot_values,
    )


def test_added_input_channel_copies_model_and_adam_state_with_complete_accounting() -> None:
    source_model, optimizer, source_optimizer = _values(3)
    source_model["nodes"]["controller"]["input"]["weight"] = jnp.arange(
        6, dtype=jnp.float32
    ).reshape(2, 3)
    gradients = jax.tree.map(jnp.ones_like, source_model)
    _, source_optimizer = optimizer.update(gradients, source_optimizer, source_model)
    target_model, _, target_optimizer = _values(4)

    source_structure, source_slots = _structure((source_model, optimizer, source_optimizer))
    target_structure, target_slots = _structure((target_model, optimizer, target_optimizer))
    result = derive_and_apply_semantic_checkpoint_transform(
        source_structure=source_structure,
        target_structure=target_structure,
        source_values=source_slots,
        target_values=target_slots,
        added_axes=[
            AddedParameterAxis(
                component_slot="controller",
                parameter_role="input/weight",
                axis=1,
            )
        ],
        optimizer_policy="preserve",
        learning_rate_schedule_policy="continue",
    )

    source_weight = source_model["nodes"]["controller"]["input"]["weight"]
    target_weight = result.values["model"]["nodes"]["controller"]["input"]["weight"]
    assert target_weight[:, :3].tobytes() == source_weight.tobytes()
    assert jnp.array_equal(target_weight[:, 3], jnp.zeros((2,), dtype=jnp.float32))

    source_adam = source_optimizer[0]
    target_adam = result.values["optimizer"][0]
    assert target_adam.count.tobytes() == source_adam.count.tobytes()
    for source_moment, target_moment in (
        (source_adam.mu, target_adam.mu),
        (source_adam.nu, target_adam.nu),
    ):
        source_leaf = source_moment["nodes"]["controller"]["input"]["weight"]
        target_leaf = target_moment["nodes"]["controller"]["input"]["weight"]
        assert target_leaf[:, :3].tobytes() == source_leaf.tobytes()
        assert jnp.array_equal(target_leaf[:, 3], jnp.zeros((2,), dtype=jnp.float32))

    assert {record["identity"] for record in result.accounting_records()} == {
        leaf.identity for slot in source_structure.slots for leaf in slot.semantic_abi.leaves
    }
    assert sum(item.disposition == "extended" for item in result.accounting) == 3
    assert result.optimizer_policy == "preserve"
    assert result.learning_rate_schedule_policy == "continue"

    continued = derive_and_apply_semantic_checkpoint_transform(
        source_structure=source_structure,
        target_structure=source_structure,
        source_values=source_slots,
        target_values=source_slots,
        added_axes=[],
        optimizer_policy="preserve",
        learning_rate_schedule_policy="restart",
    )
    assert all(item.disposition == "copied" for item in continued.accounting)
    assert continued.values["optimizer"][0].count.tobytes() == source_adam.count.tobytes()


def test_duplicate_identity_across_slots_is_ambiguous() -> None:
    values = _values(3)
    structure, slots = _structure(values)
    duplicate_slot = structure.slots[0].model_copy(
        update={"slot": "duplicate", "role": "auxiliary"}
    )
    ambiguous = structure.model_copy(update={"slots": [*structure.slots, duplicate_slot]})
    with pytest.raises(CheckpointTransformRefused) as excinfo:
        derive_and_apply_semantic_checkpoint_transform(
            source_structure=ambiguous,
            target_structure=structure,
            source_values={**slots, "duplicate": slots["model"]},
            target_values=slots,
            added_axes=[],
            optimizer_policy="preserve",
            learning_rate_schedule_policy="continue",
        )
    assert excinfo.value.code is CheckpointTransformErrorCode.AMBIGUOUS_IDENTITY


def test_initialization_plan_copies_by_semantic_identity_across_treedef_change() -> None:
    source_values = {"model": {"nodes": {"controller": {"weight": jnp.array([3.0])}}}}
    target_values = {
        "model": {"wrapper": {"nodes": {"controller": {"weight": jnp.array([0.0])}}}}
    }

    def structure(values):
        fingerprint = structural_abi_fingerprint(values["model"])
        semantic = derive_checkpoint_semantic_abis(
            {"model": fingerprint}, slot_roles={"model": "model"}
        )["model"]
        return CheckpointStructure(
            slots=[
                CheckpointSlotStructure(
                    slot="model",
                    role="model",
                    treedef=fingerprint.treedef,
                    leaves=fingerprint.leaves,
                    semantic_abi=semantic,
                )
            ]
        )

    source_structure = structure(source_values)
    target_structure = structure(target_values)
    plan = lower_checkpoint_initialization(
        CheckpointInitializationRequest(
            mode="continue_from",
            source=ParentRef(
                kind="TrainingCheckpointTransactionManifest",
                id="tx",
                role="training_checkpoint_custody",
                metadata=authenticated_manifest_ref_metadata("a" * 64, 1),
            ),
        ),
        source=source_structure,
        target=target_structure,
    )
    restored = apply_semantic_checkpoint_initialization(
        plan,
        source_structure=source_structure,
        target_structure=target_structure,
        source_values=source_values,
        target_values=target_values,
    )
    assert restored["model"]["wrapper"]["nodes"]["controller"]["weight"].tolist() == [3.0]


@pytest.mark.parametrize(
    ("target", "code"),
    [
        (
            jnp.zeros((3, 3), dtype=jnp.float32),
            CheckpointTransformErrorCode.UNSUPPORTED_SHAPE_CHANGE,
        ),
        (
            jnp.zeros((2, 4), dtype=jnp.int32),
            CheckpointTransformErrorCode.DTYPE_MISMATCH,
        ),
    ],
)
def test_incompatible_added_input_change_is_refused(target, code) -> None:
    source_values = _values(3)
    target_values = _values(4)
    target_values[0]["nodes"]["controller"]["input"]["weight"] = target
    source_structure, source_slots = _structure(source_values)
    target_structure, target_slots = _structure(target_values)
    with pytest.raises(CheckpointTransformRefused) as excinfo:
        derive_and_apply_semantic_checkpoint_transform(
            source_structure=source_structure,
            target_structure=target_structure,
            source_values=source_slots,
            target_values=target_slots,
            added_axes=[
                AddedParameterAxis(
                    component_slot="controller",
                    parameter_role="input/weight",
                    axis=1,
                )
            ],
            optimizer_policy="preserve",
            learning_rate_schedule_policy="continue",
        )
    assert excinfo.value.code is code
