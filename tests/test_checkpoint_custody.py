from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jax.numpy as jnp
import pytest

from feedbax.contracts.manifest import ParentRef, TrainingRunManifest, load_manifest, spec_payload
from feedbax.contracts.migrations import default_spec_registry
from feedbax.contracts.training import (
    STANDARD_SUPERVISED_METHOD_PAYLOAD_SCHEMA_VERSION,
    LossTermSpec,
    ObjectiveSlotSpec,
    TaskSpec,
    TrainingConfig,
    TrainingRunSpec,
    WorkerExecutionSpec,
    standard_supervised_effective_phase_spec,
    standard_supervised_method_contract,
    standard_supervised_method_payload,
    standard_supervised_method_ref,
)
from feedbax.contracts.worker import (
    EffectivePhaseSpec,
    ProgressCoordinate,
    derive_consistency_predicate,
    toy_minimax_method_contract,
)
from feedbax.training.checkpoint_custody import (
    CheckpointCompatibilityError,
    CheckpointConsistencyError,
    CheckpointContractBindingError,
    CheckpointIntegrityError,
    checkpoint_slot_names,
    load_latest_checkpoint,
    write_checkpoint_transaction,
)


def _minimal_graph() -> dict[str, object]:
    return {
        "nodes": {
            "gain": {
                "type": "Gain",
                "params": {"gain": 1.0},
                "input_ports": ["input"],
                "output_ports": ["output"],
            }
        },
        "wires": [],
        "input_ports": ["input"],
        "output_ports": ["output"],
        "input_bindings": {"input": ("gain", "input")},
        "output_bindings": {"output": ("gain", "output")},
    }


def _run_spec(*, minimax: bool = False) -> TrainingRunSpec:
    if minimax:
        contract = toy_minimax_method_contract()
        program = contract.phase_program.model_copy(deep=True)
        program.checkpoint_barriers[0].metadata["consistency_mode"] = "population-barrier"
        method_contract = contract.model_copy(
            update={
                "method_ref": "feedbax/standard_supervised/v1",
                "method_payload_schema_version": (
                    STANDARD_SUPERVISED_METHOD_PAYLOAD_SCHEMA_VERSION
                ),
                "phase_program": program,
            }
        )
        effective_phase = EffectivePhaseSpec(
            method_ref="feedbax/standard_supervised/v1",
            axes=method_contract.axes,
            state_slots=method_contract.state_slots,
            phase_program=program,
            consistency_predicate=derive_consistency_predicate(program),
        )
    else:
        method_contract = standard_supervised_method_contract()
        effective_phase = standard_supervised_effective_phase_spec()

    return TrainingRunSpec(
        graph={"inline": _minimal_graph()},
        task=TaskSpec(type="ReachingTask", params={"n_steps": 4}),
        training_config=TrainingConfig(n_batches=4, batch_size=3),
        objective=ObjectiveSlotSpec(
            loss=LossTermSpec(type="target_state", label="target", selector="output")
        ),
        method_ref=standard_supervised_method_ref(),
        method_payload=standard_supervised_method_payload(),
        worker_execution=WorkerExecutionSpec(
            method_contract=method_contract,
            effective_phase=effective_phase,
        ),
    )


def _coordinate(step: int = 1) -> ProgressCoordinate:
    return ProgressCoordinate(
        run_id="run-1",
        phase="warmup",
        global_step=step,
        completed_barrier="after_warmup",
    )


def _minimax_slots() -> dict[str, object]:
    return {
        "controller": jnp.array([1.0, 2.0]),
        "controller_optimizer": {"count": jnp.array(1)},
        "adversary_population": [jnp.array([0.1, 0.2]), jnp.array([0.3, 0.4])],
        "adversary_optimizer": {"count": jnp.array([1, 1])},
        "rng": jnp.array([11, 22], dtype=jnp.uint32),
        "loss": [0.5],
    }


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2))


def test_checkpoint_transaction_derives_slots_and_loads_multi_slot_state(tmp_path: Path) -> None:
    run_spec = _run_spec(minimax=True)
    program = run_spec.worker_execution.method_contract.phase_program

    result = write_checkpoint_transaction(
        tmp_path,
        run_spec=run_spec,
        phase_program=program,
        barrier_name="after_warmup",
        coordinate=_coordinate(),
        slots=_minimax_slots(),
        population_member_ids={"adversary_population": ["adv-a", "adv-b"]},
        history_availability={"loss": True},
    )

    assert checkpoint_slot_names(program, "after_warmup") == (
        "controller",
        "controller_optimizer",
        "adversary_population",
        "adversary_optimizer",
        "rng",
    )
    assert result.latest_pointer_path.is_file()
    assert {slot.slot: slot.role for slot in result.manifest.slots} == {
        "controller": "model",
        "controller_optimizer": "optimizer",
        "adversary_population": "population",
        "adversary_optimizer": "optimizer",
        "rng": "prng",
    }

    loaded = load_latest_checkpoint(
        tmp_path,
        expected_run_spec=run_spec,
        expected_phase_program=program,
        expected_slots=_minimax_slots(),
        expected_population_member_ids={"adversary_population": ["adv-a", "adv-b"]},
    )

    assert loaded.manifest.transaction_id == result.manifest.transaction_id
    assert loaded.slots["controller"].tolist() == [1.0, 2.0]
    assert loaded.slots["rng"].dtype == jnp.uint32


def test_training_run_manifest_links_checkpoint_custody_ref() -> None:
    ref = ParentRef(
        kind="TrainingCheckpointTransactionManifest",
        id="tx-demo",
        role="training_checkpoint_custody",
        uri="repo://checkpoints/tx-demo/manifest.json",
    )
    manifest = TrainingRunManifest(
        id="feedbax-training-run:demo",
        training_spec=spec_payload("TrainingRunSpec", _run_spec().model_dump(mode="json")),
        checkpoint_custody=[ref],
    )

    assert manifest.checkpoint_custody == [ref]


def test_checkpoint_transaction_schema_family_is_registered() -> None:
    families = {family.kind: family for family in default_spec_registry.families()}

    family = families["TrainingCheckpointTransactionManifest"]
    assert family.identity == "feedbax.manifest.training_checkpoint_transaction"
    assert family.current_version == "feedbax.manifest.training_checkpoint_transaction.v1"
    assert family.policy is not None
    assert family.policy.owner_module == "feedbax.contracts.checkpoints"


def test_manifest_loader_accepts_training_checkpoint_transaction(tmp_path: Path) -> None:
    run_spec = _run_spec(minimax=True)
    program = run_spec.worker_execution.method_contract.phase_program
    result = write_checkpoint_transaction(
        tmp_path,
        run_spec=run_spec,
        phase_program=program,
        barrier_name="after_warmup",
        coordinate=_coordinate(),
        slots=_minimax_slots(),
    )

    loaded = load_manifest(result.manifest_path)

    assert loaded.kind == "TrainingCheckpointTransactionManifest"
    assert loaded.transaction_id == result.manifest.transaction_id


def test_resume_rejects_structural_abi_mismatch_before_returning_slots(
    tmp_path: Path,
) -> None:
    run_spec = _run_spec(minimax=True)
    program = run_spec.worker_execution.method_contract.phase_program
    write_checkpoint_transaction(
        tmp_path,
        run_spec=run_spec,
        phase_program=program,
        barrier_name="after_warmup",
        coordinate=_coordinate(),
        slots=_minimax_slots(),
    )
    incompatible = _minimax_slots()
    incompatible["controller"] = jnp.array([1.0, 2.0, 3.0])

    with pytest.raises(CheckpointCompatibilityError, match="structural ABI mismatch"):
        load_latest_checkpoint(
            tmp_path,
            expected_run_spec=run_spec,
            expected_phase_program=program,
            expected_slots=incompatible,
        )


def test_resume_slot_transform_runs_before_structural_abi_validation(
    tmp_path: Path,
) -> None:
    run_spec = _run_spec(minimax=True)
    program = run_spec.worker_execution.method_contract.phase_program
    write_checkpoint_transaction(
        tmp_path,
        run_spec=run_spec,
        phase_program=program,
        barrier_name="after_warmup",
        coordinate=_coordinate(),
        slots=_minimax_slots(),
    )
    expected = _minimax_slots()
    expected["controller"] = jnp.array([1.0, 2.0, 0.0])

    def resize_controller(slots):
        transformed = dict(slots)
        transformed["controller"] = jnp.pad(transformed["controller"], (0, 1))
        return transformed

    loaded = load_latest_checkpoint(
        tmp_path,
        expected_run_spec=run_spec,
        expected_phase_program=program,
        expected_slots=expected,
        resume_slot_transform=resize_controller,
    )

    assert loaded.slots["controller"].tolist() == [1.0, 2.0, 0.0]


def test_resume_slot_transform_that_drops_required_slot_fails_closed(
    tmp_path: Path,
) -> None:
    run_spec = _run_spec(minimax=True)
    program = run_spec.worker_execution.method_contract.phase_program
    write_checkpoint_transaction(
        tmp_path,
        run_spec=run_spec,
        phase_program=program,
        barrier_name="after_warmup",
        coordinate=_coordinate(),
        slots=_minimax_slots(),
    )

    def drop_controller(slots):
        transformed = dict(slots)
        del transformed["controller"]
        return transformed

    with pytest.raises(CheckpointCompatibilityError, match="missing required checkpoint slots"):
        load_latest_checkpoint(
            tmp_path,
            expected_run_spec=run_spec,
            expected_phase_program=program,
            expected_slots=_minimax_slots(),
            resume_slot_transform=drop_controller,
        )


def test_resume_slot_transform_structural_mismatch_fails_closed(
    tmp_path: Path,
) -> None:
    run_spec = _run_spec(minimax=True)
    program = run_spec.worker_execution.method_contract.phase_program
    write_checkpoint_transaction(
        tmp_path,
        run_spec=run_spec,
        phase_program=program,
        barrier_name="after_warmup",
        coordinate=_coordinate(),
        slots=_minimax_slots(),
    )

    def resize_controller(slots):
        transformed = dict(slots)
        transformed["controller"] = jnp.pad(transformed["controller"], (0, 1))
        return transformed

    with pytest.raises(CheckpointCompatibilityError, match="structural ABI mismatch"):
        load_latest_checkpoint(
            tmp_path,
            expected_run_spec=run_spec,
            expected_phase_program=program,
            expected_slots=_minimax_slots(),
            resume_slot_transform=resize_controller,
        )


def test_resume_slot_transform_exception_fails_closed(
    tmp_path: Path,
) -> None:
    run_spec = _run_spec(minimax=True)
    program = run_spec.worker_execution.method_contract.phase_program
    write_checkpoint_transaction(
        tmp_path,
        run_spec=run_spec,
        phase_program=program,
        barrier_name="after_warmup",
        coordinate=_coordinate(),
        slots=_minimax_slots(),
    )

    def fail_transform(slots):
        del slots
        raise RuntimeError("resize failed")

    with pytest.raises(CheckpointCompatibilityError, match="resume_slot_transform failed"):
        load_latest_checkpoint(
            tmp_path,
            expected_run_spec=run_spec,
            expected_phase_program=program,
            expected_slots=_minimax_slots(),
            resume_slot_transform=fail_transform,
        )


def test_population_coordinate_mismatch_under_population_predicate_rejects_on_resume(
    tmp_path: Path,
) -> None:
    run_spec = _run_spec(minimax=True)
    program = run_spec.worker_execution.method_contract.phase_program
    result = write_checkpoint_transaction(
        tmp_path,
        run_spec=run_spec,
        phase_program=program,
        barrier_name="after_warmup",
        coordinate=_coordinate(step=2),
        slots=_minimax_slots(),
    )
    manifest_payload = json.loads(result.manifest_path.read_text())
    for slot in manifest_payload["slots"]:
        if slot["slot"] == "adversary_population":
            slot["coordinate"]["global_step"] = 1
    _write_json(result.manifest_path, manifest_payload)
    latest_payload = json.loads(result.latest_pointer_path.read_text())
    latest_payload["manifest_sha256"] = _sha256_file(result.manifest_path)
    _write_json(result.latest_pointer_path, latest_payload)

    with pytest.raises(CheckpointConsistencyError, match="mismatched_slots"):
        load_latest_checkpoint(
            tmp_path,
            expected_run_spec=run_spec,
            expected_phase_program=program,
            expected_slots=_minimax_slots(),
        )


def test_population_length_and_member_identity_mismatch_rejects_resume(
    tmp_path: Path,
) -> None:
    run_spec = _run_spec(minimax=True)
    program = run_spec.worker_execution.method_contract.phase_program
    write_checkpoint_transaction(
        tmp_path,
        run_spec=run_spec,
        phase_program=program,
        barrier_name="after_warmup",
        coordinate=_coordinate(),
        slots=_minimax_slots(),
        population_member_ids={"adversary_population": ["adv-a", "adv-b"]},
    )

    with pytest.raises(CheckpointCompatibilityError, match="population identity mismatch"):
        load_latest_checkpoint(
            tmp_path,
            expected_run_spec=run_spec,
            expected_phase_program=program,
            expected_slots=_minimax_slots(),
            expected_population_member_ids={
                "adversary_population": ["adv-a", "adv-b", "adv-c"]
            },
        )


def test_latest_pointer_missing_corrupt_and_stale_cases_fail_closed(
    tmp_path: Path,
) -> None:
    run_spec = _run_spec(minimax=True)
    program = run_spec.worker_execution.method_contract.phase_program

    with pytest.raises(CheckpointIntegrityError, match="latest pointer is missing"):
        load_latest_checkpoint(
            tmp_path,
            expected_run_spec=run_spec,
            expected_phase_program=program,
            expected_slots=_minimax_slots(),
        )

    (tmp_path / "latest.json").write_text("{not-json")
    with pytest.raises(CheckpointIntegrityError, match="latest pointer is corrupt"):
        load_latest_checkpoint(
            tmp_path,
            expected_run_spec=run_spec,
            expected_phase_program=program,
            expected_slots=_minimax_slots(),
        )

    result = write_checkpoint_transaction(
        tmp_path,
        run_spec=run_spec,
        phase_program=program,
        barrier_name="after_warmup",
        coordinate=_coordinate(),
        slots=_minimax_slots(),
    )
    latest_payload = json.loads(result.latest_pointer_path.read_text())
    latest_payload["transaction_root_sha256"] = "0" * 64
    _write_json(result.latest_pointer_path, latest_payload)
    with pytest.raises(CheckpointIntegrityError, match="transaction root is stale"):
        load_latest_checkpoint(
            tmp_path,
            expected_run_spec=run_spec,
            expected_phase_program=program,
            expected_slots=_minimax_slots(),
        )


def test_changed_method_payload_fails_closed_unless_new_lineage_override(
    tmp_path: Path,
) -> None:
    run_spec = _run_spec(minimax=True)
    program = run_spec.worker_execution.method_contract.phase_program
    write_checkpoint_transaction(
        tmp_path,
        run_spec=run_spec,
        phase_program=program,
        barrier_name="after_warmup",
        coordinate=_coordinate(),
        slots=_minimax_slots(),
    )
    changed = run_spec.model_copy(deep=True)
    changed.method_payload.payload["learning_rate"] = 0.5

    with pytest.raises(CheckpointContractBindingError, match="content binding"):
        load_latest_checkpoint(
            tmp_path,
            expected_run_spec=changed,
            expected_phase_program=program,
            expected_slots=_minimax_slots(),
        )

    loaded = load_latest_checkpoint(
        tmp_path,
        expected_run_spec=changed,
        expected_phase_program=program,
        expected_slots=_minimax_slots(),
        allow_new_lineage_override=True,
    )

    assert loaded.new_lineage_required
    assert loaded.previous_transaction_id == loaded.manifest.transaction_id


def test_interrupted_toy_resume_matches_uninterrupted(tmp_path: Path) -> None:
    run_spec = _run_spec()
    program = run_spec.worker_execution.method_contract.phase_program

    def step(state: dict[str, object]) -> dict[str, object]:
        return {
            "model": state["model"] + state["optimizer"]["count"],
            "optimizer": {"count": state["optimizer"]["count"] + 1},
            "prng": state["prng"] + 17,
        }

    continuous = {
        "model": jnp.array([0.0]),
        "optimizer": {"count": jnp.array(1.0)},
        "prng": jnp.array([0, 1], dtype=jnp.uint32),
    }
    interrupted = dict(continuous)
    for _ in range(4):
        continuous = step(continuous)
    for _ in range(2):
        interrupted = step(interrupted)

    coordinate = ProgressCoordinate(
        run_id="run-2",
        phase="train_batch",
        global_step=2,
        completed_barrier="after_train_batch",
    )
    write_checkpoint_transaction(
        tmp_path,
        run_spec=run_spec,
        phase_program=program,
        barrier_name="after_train_batch",
        coordinate=coordinate,
        slots=interrupted,
    )
    loaded = load_latest_checkpoint(
        tmp_path,
        expected_run_spec=run_spec,
        expected_phase_program=program,
        expected_slots=interrupted,
    )
    resumed = loaded.slots
    for _ in range(2):
        resumed = step(resumed)

    assert resumed["model"].tolist() == continuous["model"].tolist()
    assert resumed["optimizer"]["count"].tolist() == continuous["optimizer"]["count"].tolist()
    assert resumed["prng"].tolist() == continuous["prng"].tolist()


def test_legacy_task_trainer_checkpoint_files_reject_with_clear_error(
    tmp_path: Path,
) -> None:
    (tmp_path / "last_batch.txt").write_text("7")
    (tmp_path / "ckpt_7.eqx").write_bytes(b"legacy eqx.tree_serialise_leaves payload")
    run_spec = _run_spec()
    program = run_spec.worker_execution.method_contract.phase_program

    with pytest.raises(CheckpointCompatibilityError, match="schema identity"):
        load_latest_checkpoint(
            tmp_path,
            expected_run_spec=run_spec,
            expected_phase_program=program,
            expected_slots=_minimax_slots(),
        )
