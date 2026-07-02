from __future__ import annotations

import pytest
from pydantic import ValidationError

from feedbax.contracts.worker import (
    CONSISTENCY_PREDICATE_GENERATOR_HASH,
    CONSISTENCY_PREDICATE_SCHEMA_VERSION,
    PPO_MAPPING_TABLE,
    AxisReducerSpec,
    CheckpointSlotManifest,
    CheckpointSlotRecord,
    MethodContractSpec,
    ReducerRequirement,
    StateSlotSpec,
    UpdateKernelSpec,
    supervised_task_trainer_mapping,
    toy_minimax_method_contract,
)
from feedbax.training.phase_executor import InMemoryCheckpointStore, PhaseProgramExecutor
from feedbax.training.worker_validation import (
    DryRunShapeCheckResult,
    WorkerContractValidationError,
    WorkerExecutabilityEnvironment,
    validate_checkpoint_population_payload,
    validate_per_trial_bindings,
    validate_resume_checkpoint_pairing,
    validate_worker_contract,
)


def _warmup_update(slots, coordinate, context):
    return {
        "controller": slots["controller"] + 1,
        "controller_optimizer": slots["controller_optimizer"] + 10,
        "rng": slots["rng"] + 1,
        "loss": 1.0 + coordinate.global_step,
    }


def _adversary_update(slots, coordinate, context):
    return {
        "adversary_population": [
            value + slots["controller"] for value in slots["adversary_population"]
        ],
        "adversary_optimizer": slots["adversary_optimizer"] + 100,
        "rng": slots["rng"] + 1,
        "loss": 2.0 + coordinate.global_step,
    }


def _toy_kernels():
    return {
        "toy_minimax.warmup_update": _warmup_update,
        "toy_minimax.adversary_update": _adversary_update,
    }


def test_task_trainer_and_ppo_map_to_worker_vocabulary() -> None:
    supervised = supervised_task_trainer_mapping()

    assert supervised[0].phase == "train_batch"
    assert "batch" in supervised[0].axes
    assert "task_optimizer_to_model" in supervised[0].optimizer_bindings
    assert "checkpoint" in (supervised[0].checkpoint_transaction or "")

    ppo_phases = {row.phase for row in PPO_MAPPING_TABLE}
    assert {
        "collect_rollout",
        "compute_gae",
        "ppo_epoch_minibatch",
        "batched_body_collect_and_update",
    } <= ppo_phases
    ppo_axes = {axis for row in PPO_MAPPING_TABLE for axis in row.axes}
    assert {"environment", "rollout", "epoch", "minibatch", "replicate"} <= ppo_axes
    assert any("observation_norm" in row.state_slots for row in PPO_MAPPING_TABLE)


def test_toy_minimax_contract_validates_and_emits_governed_predicate() -> None:
    contract = toy_minimax_method_contract()

    effective = validate_worker_contract(
        contract,
        update_kernels=_toy_kernels(),
        dry_run_shape_check=lambda _contract: DryRunShapeCheckResult(passed=True),
    )

    predicate = effective.consistency_predicate
    assert predicate.schema_version == CONSISTENCY_PREDICATE_SCHEMA_VERSION
    assert predicate.generator_hash == CONSISTENCY_PREDICATE_GENERATOR_HASH
    assert predicate.rules
    assert effective.phase_program.initial_phase == "warmup"


def test_method_contract_rejects_method_owned_runner_refs() -> None:
    with pytest.raises(ValidationError, match="method-owned training runner"):
        UpdateKernelSpec(kernel_ref="downstream_method.run_minimax")

    payload = toy_minimax_method_contract().model_dump(mode="json")
    payload["method_ref"] = "downstream_method.run_minimax"
    with pytest.raises(ValidationError, match="method-owned runner"):
        MethodContractSpec.model_validate(payload)


def test_unsatisfiable_declaration_is_rejected_with_path() -> None:
    contract = toy_minimax_method_contract()
    contract.state_slots.append(
        StateSlotSpec(name="bad_population", role="population", axis="missing_axis")
    )

    with pytest.raises(WorkerContractValidationError) as exc:
        validate_worker_contract(contract, update_kernels=_toy_kernels())

    assert "/state_slots/6/axis" in str(exc.value)
    assert "missing_axis" in str(exc.value)


def test_double_reduction_on_one_axis_is_rejected() -> None:
    contract = toy_minimax_method_contract()
    contract.axes[0].reducer = AxisReducerSpec(
        owner="worker",
        reduction="mean",
        path="/axes/0/reducer",
    )
    contract.objective_reducers.append(
        ReducerRequirement(axis="batch", owner="objective", path="/objective/reductions/0")
    )

    with pytest.raises(WorkerContractValidationError, match="more than one reducer"):
        validate_worker_contract(contract, update_kernels=_toy_kernels())


def test_per_trial_binding_validation_uses_current_bindable_role_set() -> None:
    exposed_data = [
        {
            "id": "target",
            "role": "target",
            "bindable": True,
            "dtype": "vector",
            "expected_shape": [2],
        }
    ]
    bindings = [
        {
            "source_data_id": "target",
            "target_node_id": "network",
            "target_port": "input",
            "role": "target",
            "target_dtype": "vector",
            "target_shape": [2],
        }
    ]

    with pytest.raises(WorkerContractValidationError) as exc:
        validate_per_trial_bindings(exposed_data=exposed_data, bindings=bindings)

    assert "/task_binding_spec/bindings/0/role" in str(exc.value)
    assert "graph-bindable" in str(exc.value)


def test_toy_minimax_executes_and_resumes_at_warmup_barrier() -> None:
    contract = toy_minimax_method_contract()
    validate_worker_contract(
        contract,
        update_kernels=_toy_kernels(),
        dry_run_shape_check=lambda _contract: DryRunShapeCheckResult(passed=True),
    )
    store = InMemoryCheckpointStore()
    executor = PhaseProgramExecutor(
        contract.phase_program,
        _toy_kernels(),
        checkpoint_store=store,
    )
    slots = {
        "controller": 0,
        "controller_optimizer": 0,
        "adversary_population": [1, 2],
        "adversary_optimizer": 0,
        "rng": 0,
    }

    warmup = executor.run(
        slots,
        run_id="run-1",
        stop_after_barrier="after_warmup",
    )
    assert warmup.slots["controller"] == 1
    assert warmup.slots["rng"] == 1
    assert warmup.coordinate.completed_barrier == "after_warmup"
    assert warmup.checkpoints["after_warmup"].slots["rng"] == 1

    resumed = executor.run(
        {},
        run_id="run-1",
        resume_from_barrier="after_warmup",
    )

    assert resumed.coordinate.phase == "adversarial"
    assert resumed.slots["adversary_population"] == [2, 3]
    assert resumed.slots["rng"] == 2
    assert resumed.coordinate.completed_barrier == "after_adversarial"


def test_population_length_mismatch_is_rejected_at_load() -> None:
    contract = toy_minimax_method_contract()

    with pytest.raises(WorkerContractValidationError) as exc:
        validate_checkpoint_population_payload(
            contract,
            {
                "adversary_population": [1],
            },
        )

    assert "/checkpoint_payload/adversary_population" in str(exc.value)
    assert "expected 2, found 1" in str(exc.value)


def test_cross_slot_checkpoint_pairing_is_rejected() -> None:
    manifest = CheckpointSlotManifest(
        slots=[
            CheckpointSlotRecord(slot="controller", barrier="after_adversarial", global_step=2),
            CheckpointSlotRecord(slot="adversary_population", barrier="after_adversarial", global_step=1),
        ]
    )

    with pytest.raises(WorkerContractValidationError) as exc:
        validate_resume_checkpoint_pairing(
            manifest,
            required_slots=["controller", "adversary_population"],
        )

    assert "/checkpoint_slots" in str(exc.value)
    assert "cross global steps" in str(exc.value)


def test_dry_run_shape_check_must_exercise_declared_resource_fields() -> None:
    contract = toy_minimax_method_contract()
    contract.phase_program.update_steps[0].kernel.resource_significant_payload_fields.append(
        "n_adversaries"
    )

    with pytest.raises(WorkerContractValidationError) as exc:
        validate_worker_contract(
            contract,
            update_kernels=_toy_kernels(),
            dry_run_shape_check=lambda _contract: DryRunShapeCheckResult(
                passed=True,
                exercised_payload_fields=[],
            ),
        )

    assert "/dry_run_shape_check/exercised_payload_fields" in str(exc.value)
    assert "n_adversaries" in str(exc.value)


def test_unsupported_axis_fails_closed() -> None:
    contract = toy_minimax_method_contract()
    env = WorkerExecutabilityEnvironment(supported_axes={"batch"})

    with pytest.raises(WorkerContractValidationError) as exc:
        validate_worker_contract(contract, environment=env, update_kernels=_toy_kernels())

    assert "/axes/1/name" in str(exc.value)
    assert "adversary_member" in str(exc.value)
