from __future__ import annotations

import hashlib
import json
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

import feedbax.training.checkpoint_custody as custody_module
from feedbax.contracts.checkpoints import (
    BatchHistory,
    CheckpointContinuationRequest,
    CheckpointForkBarrierMapping,
    Granularity,
    TRAINING_CHECKPOINT_LATEST_POINTER_SCHEMA_VERSION,
    TRAINING_CHECKPOINT_LATEST_POINTER_SCHEMA_VERSION_V2,
    TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION,
    TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION_V1,
    TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION_V2,
    TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION_V3,
    TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION_V4,
    TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION_V5,
    TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION_V6,
)
from feedbax.contracts.manifest import ParentRef, TrainingRunManifest, load_manifest, spec_payload
from feedbax.contracts.migrations import default_spec_registry
from feedbax.contracts.migrations import migrate_structured_spec_payload
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
    TrainingBatchProgressSpec,
    derive_consistency_predicate,
    toy_minimax_method_contract,
)
from feedbax.training.checkpoint_custody import (
    LEGACY_CHECKPOINT_ADOPTION_DOCS,
    LEGACY_CHECKPOINT_ADOPTION_ENTRYPOINT,
    CheckpointCompatibilityError,
    CheckpointConsistencyError,
    CheckpointContractBindingError,
    CheckpointIntegrityError,
    checkpoint_slot_names,
    detect_known_legacy_checkpoint_layout,
    fork_checkpoint_transaction,
    load_latest_checkpoint,
    load_checkpoint_custody_documents,
    load_checkpoint_latest_pointer_json,
    load_checkpoint_transaction_manifest_file,
    write_checkpoint_transaction,
    concatenate_checkpoint_histories,
    materialize_concatenated_checkpoint_histories,
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
        program_step=step,
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


def _rewrite_manifest_and_latest(result, payload: dict[str, object]) -> None:
    _write_json(result.manifest_path, payload)
    latest_payload = json.loads(result.latest_pointer_path.read_text())
    latest_payload["manifest_sha256"] = _sha256_file(result.manifest_path)
    _write_json(result.latest_pointer_path, latest_payload)


def _manifest_blob_path(manifest_path: Path, relative_path: str) -> Path:
    return manifest_path.parent / relative_path


def _slot_blob_path(manifest_path: Path, slot_name: str) -> Path:
    payload = json.loads(manifest_path.read_text())
    for slot in payload["slots"]:
        if slot["slot"] == slot_name:
            return _manifest_blob_path(manifest_path, slot["relative_path"])
    raise AssertionError(f"slot not found in manifest: {slot_name}")


def _write_run_spec(path: Path, run_spec: TrainingRunSpec) -> None:
    _write_json(path, run_spec.model_dump(mode="json", exclude_none=True))


def _incompatible_slot_run_spec(run_spec: TrainingRunSpec) -> TrainingRunSpec:
    changed = run_spec.model_copy(deep=True)
    barrier = changed.worker_execution.method_contract.phase_program.checkpoint_barriers[0]
    barrier.slots[0].slot = "controller_v2"
    return changed


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


def test_load_reuses_manifest_structural_fingerprints_for_loaded_slots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    )
    calls = 0
    original_fingerprint = custody_module.structural_abi_fingerprint

    def counting_fingerprint(value):
        nonlocal calls
        calls += 1
        return original_fingerprint(value)

    monkeypatch.setattr(custody_module, "structural_abi_fingerprint", counting_fingerprint)

    load_latest_checkpoint(
        tmp_path,
        expected_run_spec=run_spec,
        expected_phase_program=program,
        expected_slots=_minimax_slots(),
        expected_population_member_ids={"adversary_population": ["adv-a", "adv-b"]},
    )

    assert calls == len(result.manifest.slots) * 2


def test_slot_integrity_records_preserve_leaf_digest_and_fingerprint_values() -> None:
    value = {
        "array": jnp.array([1.0, 2.0], dtype=jnp.float32),
        "static": ("kept", 3),
    }
    legacy_leaf_digests = []
    for path, leaf in custody_module.jt.leaves_with_path(value):
        if custody_module.eqx.is_array(leaf):
            data = bytes(jnp.asarray(leaf).tobytes())
        else:
            data = pickle.dumps(leaf, protocol=pickle.HIGHEST_PROTOCOL)
        legacy_leaf_digests.append(
            custody_module.SlotLeafContentDigest(
                path=custody_module._key_path_to_text(path),
                sha256=custody_module.sha256_bytes(data),
                size_bytes=len(data),
            )
        )

    integrity = custody_module._slot_integrity_records(value)

    assert integrity.leaf_digests == legacy_leaf_digests
    assert integrity.structural_abi_fingerprint == custody_module.structural_abi_fingerprint(value)


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
    assert family.current_version == TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION
    assert family.policy is not None
    assert family.policy.owner_module == "feedbax.contracts.checkpoints"
    assert family.policy.supported_old_versions == (
        TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION_V1,
        TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION_V2,
        TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION_V3,
        TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION_V4,
            TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION_V5,
            TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION_V6,
    )


def test_checkpoint_transaction_manifest_v1_migrates_to_current_portable_custody(
    tmp_path: Path,
) -> None:
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
    payload = result.manifest.model_dump(mode="json", exclude_none=True)
    payload["schema_version"] = TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION_V1
    payload.pop("fork_provenance", None)

    migrated = migrate_structured_spec_payload(
        "TrainingCheckpointTransactionManifest",
        payload,
        path="checkpoint_manifest",
    )

    assert migrated.source_version == TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION_V1
    assert migrated.target_version == TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION
    assert migrated.payload["fork_provenance"] is None
    assert [record.migration_id for record in migrated.migration_records] == [
        "training-checkpoint-transaction-v1-to-v2-fork-provenance",
        "training-checkpoint-transaction-v2-to-v3-portable-custody",
        "training-checkpoint-transaction-v3-to-v4-batch-progress",
        "training-checkpoint-transaction-v4-to-v5-program-coordinate",
            "training-checkpoint-transaction-v5-to-v6-batch-history",
            "training-checkpoint-transaction-v6-to-v7-segment-lineage",
    ]
    assert migrated.payload["metadata"]["batch_history_tree_migration"] == (
        "declared_paths_v5_to_v6"
    )


def test_checkpoint_transaction_manifest_v2_migrates_structural_and_binding_contracts(
    tmp_path: Path,
) -> None:
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
    payload = result.manifest.model_dump(mode="json", exclude_none=True)
    payload["schema_version"] = TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION_V2
    fingerprint = payload["slots"][0]["structural_abi_fingerprint"]
    fingerprint["schema_version"] = "feedbax.manifest.training_checkpoint.structural_abi.v1"
    fingerprint["serializer_versions"] = fingerprint.pop("environment_provenance")
    fingerprint.pop("fingerprint_algorithm_version")
    fingerprint.pop("provenance_status")
    fingerprint["fingerprint_sha256"] = "legacy-mixed-serializer-hash"
    binding = payload["run_contract_binding"]
    binding["schema_version"] = "feedbax.manifest.training_checkpoint.run_contract_binding.v1"
    binding.pop("algorithm_version")
    binding["canonical_projection"]["training_run_spec"]["schema_version"] = (
        "feedbax.spec.training_run.v1"
    )
    binding["canonical_projection"]["training_run_spec"].pop("on_nan")
    binding["canonical_projection_sha256"] = "legacy-projection-hash"

    migrated = migrate_structured_spec_payload(
        "TrainingCheckpointTransactionManifest",
        payload,
        path="checkpoint_manifest",
    )

    assert migrated.source_version == TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION_V2
    assert migrated.target_version == TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION
    migrated_fingerprint = migrated.payload["slots"][0]["structural_abi_fingerprint"]
    assert migrated_fingerprint["fingerprint_algorithm_version"].endswith(".content.v2")
    assert migrated_fingerprint["environment_provenance"]["serializer"].startswith(
        "feedbax.training.checkpoint_custody.pickle"
    )
    assert migrated_fingerprint["fingerprint_sha256"] != "legacy-mixed-serializer-hash"
    migrated_binding = migrated.payload["run_contract_binding"]
    assert migrated_binding["algorithm_version"].endswith(".run_contract_binding.v2")
    assert (
        migrated_binding["canonical_projection"]["training_run_spec"]["schema_version"]
        == run_spec.schema_version
    )
    assert migrated_binding["canonical_projection"]["training_run_spec"]["on_nan"] == "raise"


def test_checkpoint_transaction_manifest_v3_migrates_batch_progress_metadata(
    tmp_path: Path,
) -> None:
    run_spec = _run_spec(minimax=True)
    program = run_spec.worker_execution.method_contract.phase_program
    result = write_checkpoint_transaction(
        tmp_path,
        run_spec=run_spec,
        phase_program=program,
        barrier_name="after_warmup",
        coordinate=_coordinate(step=12009),
        slots=_minimax_slots(),
        metadata={"completed_batches": 16500},
    )
    payload = result.manifest.model_dump(mode="json", exclude_none=True)
    payload["schema_version"] = TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION_V3
    payload.pop("completed_training_batches")
    payload.pop("completed_coordinate_semantics")

    migrated = migrate_structured_spec_payload(
        "TrainingCheckpointTransactionManifest",
        payload,
        path="checkpoint_manifest",
    )

    assert migrated.source_version == TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION_V3
    assert migrated.target_version == TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION
    assert migrated.payload["completed_training_batches"] == 16500
    assert "not the primary training-batch" in migrated.payload["completed_coordinate_semantics"]


def test_checkpoint_latest_pointer_prefers_explicit_batches_over_coordinate(
    tmp_path: Path,
) -> None:
    run_spec = _run_spec(minimax=True)
    program = run_spec.worker_execution.method_contract.phase_program
    result = write_checkpoint_transaction(
        tmp_path,
        run_spec=run_spec,
        phase_program=program,
        barrier_name="after_warmup",
        coordinate=_coordinate(step=12009),
        slots=_minimax_slots(),
        completed_training_batches=16500,
    )

    latest = json.loads(result.latest_pointer_path.read_text())

    assert result.manifest.completed_coordinate.program_step == 12009
    assert result.manifest.completed_training_batches == 16500
    assert result.latest_pointer.completed_training_batches == 16500
    assert latest["completed_training_batches"] == 16500
    assert latest["completed_coordinate"]["program_step"] == 12009
    assert "not the primary training-batch" in latest["completed_coordinate_semantics"]


def _batch_counter_program_and_spec() -> tuple[TrainingRunSpec, object]:
    """Build a method declaration whose chunks each cover 500 batches."""
    spec = _run_spec()
    contract = spec.worker_execution.method_contract.model_copy(deep=True)
    program = contract.phase_program.model_copy(
        update={
            "batch_progress": TrainingBatchProgressSpec(slot="batch_counter"),
        }
    )
    contract = contract.model_copy(
        update={
            "phase_program": program,
            "state_slots": contract.state_slots,
        }
    )
    effective = spec.worker_execution.effective_phase.model_copy(
        update={"phase_program": program, "state_slots": contract.state_slots}
    )
    return (
        spec.model_copy(
            update={
                "training_config": spec.training_config.model_copy(update={"n_batches": 12000}),
                "worker_execution": spec.worker_execution.model_copy(
                    update={"method_contract": contract, "effective_phase": effective}
                ),
            }
        ),
        program,
    )


def test_chunked_custody_records_program_and_batch_progress_independently(
    tmp_path: Path,
) -> None:
    spec, program = _batch_counter_program_and_spec()
    slots = {"model": 0, "optimizer": {"count": 0}, "prng": [0, 1], "batch_counter": 0}
    result = None
    for chunk in range(1, 25):
        slots["batch_counter"] = chunk * 500
        result = write_checkpoint_transaction(
            tmp_path,
            run_spec=spec,
            phase_program=program,
            barrier_name="after_train_batch",
            coordinate=ProgressCoordinate(
                run_id="chunked",
                phase="train_batch",
                program_step=chunk,
                completed_barrier="after_train_batch",
            ),
            slots=slots,
            metadata={"barrier_visit_ordinal": chunk},
        )

    assert result is not None
    assert result.manifest.completed_coordinate.program_step == 24
    assert result.manifest.metadata["barrier_visit_ordinal"] == 24
    assert result.manifest.completed_training_batches == 12000
    latest = json.loads(result.latest_pointer_path.read_text())
    assert latest["completed_coordinate"]["program_step"] == 24
    assert latest["completed_training_batches"] == 12000


def test_custody_rejects_batch_total_seeded_as_program_step(tmp_path: Path) -> None:
    spec, program = _batch_counter_program_and_spec()
    with pytest.raises(CheckpointConsistencyError, match="not a training-batch total"):
        write_checkpoint_transaction(
            tmp_path,
            run_spec=spec,
            phase_program=program,
            barrier_name="after_train_batch",
            coordinate=ProgressCoordinate(
                run_id="poisoned-units",
                phase="train_batch",
                program_step=12000,
                completed_barrier="after_train_batch",
            ),
            slots={
                "model": 0,
                "optimizer": {"count": 0},
                "prng": [0, 1],
                "batch_counter": 12000,
            },
            metadata={"barrier_visit_ordinal": 24},
        )


def test_custody_accepts_one_based_program_step_for_zero_based_barrier_visit(
    tmp_path: Path,
) -> None:
    spec, program = _batch_counter_program_and_spec()
    result = write_checkpoint_transaction(
        tmp_path,
        run_spec=spec,
        phase_program=program,
        barrier_name="after_train_batch",
        coordinate=ProgressCoordinate(
            run_id="correct-units",
            phase="train_batch",
            program_step=24,
            completed_barrier="after_train_batch",
        ),
        slots={
            "model": 0,
            "optimizer": {"count": 0},
            "prng": [0, 1],
            "batch_counter": 12000,
        },
        metadata={"barrier_visit_ordinal": 23},
    )
    assert result.manifest.completed_coordinate.program_step == 24


def test_custody_rejects_declared_batches_that_disagree_with_bookkeeping_slot(
    tmp_path: Path,
) -> None:
    spec, program = _batch_counter_program_and_spec()
    with pytest.raises(CheckpointConsistencyError) as excinfo:
        write_checkpoint_transaction(
            tmp_path,
            run_spec=spec,
            phase_program=program,
            barrier_name="after_train_batch",
            coordinate=ProgressCoordinate(run_id="mismatch", phase="train_batch", program_step=24),
            slots={"model": 0, "optimizer": {"count": 0}, "prng": [0, 1], "batch_counter": 12000},
            completed_training_batches=24,
        )
    message = str(excinfo.value)
    assert "/completed_training_batches=24" in message
    assert "/phase_program/batch_progress=12000" in message


def test_legacy_global_step_is_migrated_only_as_a_program_coordinate() -> None:
    coordinate = ProgressCoordinate.model_validate(
        {"run_id": "legacy", "phase": "train", "global_step": 24}
    )
    assert coordinate.program_step == 24
    assert "global_step" not in coordinate.model_dump(mode="json")


def test_legacy_latest_pointer_migrates_global_step_without_creating_batches(
    tmp_path: Path,
) -> None:
    spec, program = _batch_counter_program_and_spec()
    result = write_checkpoint_transaction(
        tmp_path,
        run_spec=spec,
        phase_program=program,
        barrier_name="after_train_batch",
        coordinate=ProgressCoordinate(
            run_id="legacy-pointer", phase="train_batch", program_step=24
        ),
        slots={
            "model": 0,
            "optimizer": {"count": 0},
            "prng": [0, 1],
            "batch_counter": jnp.array(12000, dtype=jnp.int32),
        },
    )
    pointer = json.loads(result.latest_pointer_path.read_text())
    pointer["schema_version"] = "feedbax.manifest.training_checkpoint_latest_pointer.v2"
    pointer["completed_coordinate"]["global_step"] = pointer["completed_coordinate"].pop(
        "program_step"
    )
    _write_json(result.latest_pointer_path, pointer)

    loaded = load_latest_checkpoint(
        tmp_path,
        expected_run_spec=spec,
        expected_phase_program=program,
        expected_slots={
            "model": 0,
            "optimizer": {"count": 0},
            "prng": [0, 1],
            "batch_counter": jnp.array(0, dtype=jnp.int32),
        },
    )
    assert loaded.manifest.completed_coordinate.program_step == 24
    assert loaded.manifest.completed_training_batches == 12000


def test_public_checkpoint_document_loaders_migrate_and_preserve_provenance(
    tmp_path: Path,
) -> None:
    spec, program = _batch_counter_program_and_spec()
    result = write_checkpoint_transaction(
        tmp_path,
        run_spec=spec,
        phase_program=program,
        barrier_name="after_train_batch",
        coordinate=ProgressCoordinate(
            run_id="public-loader", phase="train_batch", program_step=24
        ),
        slots={
            "model": 0,
            "optimizer": {"count": 0},
            "prng": [0, 1],
            "batch_counter": jnp.array(12000, dtype=jnp.int32),
        },
    )
    pointer = json.loads(result.latest_pointer_path.read_text())
    pointer["schema_version"] = TRAINING_CHECKPOINT_LATEST_POINTER_SCHEMA_VERSION_V2
    pointer["completed_coordinate"]["global_step"] = pointer["completed_coordinate"].pop(
        "program_step"
    )

    migrated_pointer = load_checkpoint_latest_pointer_json(json.dumps(pointer).encode())
    current_manifest = load_checkpoint_transaction_manifest_file(result.manifest_path)
    documents = load_checkpoint_custody_documents(tmp_path)

    assert migrated_pointer.migrated
    assert migrated_pointer.source_version == TRAINING_CHECKPOINT_LATEST_POINTER_SCHEMA_VERSION_V2
    assert migrated_pointer.target_version == TRAINING_CHECKPOINT_LATEST_POINTER_SCHEMA_VERSION
    assert migrated_pointer.document.completed_coordinate.program_step == 24
    assert migrated_pointer.document.completed_training_batches == 12000
    assert not current_manifest.migrated
    assert current_manifest.source_version == TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION
    assert current_manifest.document.transaction_id == result.manifest.transaction_id
    assert documents.latest_pointer.document.transaction_id == result.manifest.transaction_id
    assert documents.manifest.document.transaction_id == result.manifest.transaction_id


def test_public_custody_document_loader_rejects_escaped_manifest_path(tmp_path: Path) -> None:
    spec, program = _batch_counter_program_and_spec()
    result = write_checkpoint_transaction(
        tmp_path,
        run_spec=spec,
        phase_program=program,
        barrier_name="after_train_batch",
        coordinate=ProgressCoordinate(
            run_id="escaped-manifest", phase="train_batch", program_step=1
        ),
        slots={
            "model": 0,
            "optimizer": {"count": 0},
            "prng": [0, 1],
            "batch_counter": jnp.array(1, dtype=jnp.int32),
        },
    )
    pointer = json.loads(result.latest_pointer_path.read_text())
    pointer["manifest_relative_path"] = "../outside/manifest.json"
    _write_json(result.latest_pointer_path, pointer)

    with pytest.raises(CheckpointIntegrityError, match="escapes custody root"):
        load_checkpoint_custody_documents(tmp_path)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"{not-json", "not valid JSON"),
        (
            json.dumps(
                {
                    "schema_id": "wrong.pointer.schema",
                    "schema_version": TRAINING_CHECKPOINT_LATEST_POINTER_SCHEMA_VERSION,
                }
            ).encode(),
            "schema_id",
        ),
        (
            json.dumps(
                {
                    "schema_id": "feedbax.manifest.training_checkpoint_latest_pointer",
                    "schema_version": "feedbax.manifest.training_checkpoint_latest_pointer.v99",
                }
            ).encode(),
            "source_version",
        ),
    ],
)
def test_public_latest_pointer_loader_fails_closed_on_invalid_documents(
    payload: bytes,
    message: str,
) -> None:
    with pytest.raises(CheckpointIntegrityError, match=message):
        load_checkpoint_latest_pointer_json(payload)


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


def test_checkpoint_transaction_manifest_persists_binding_projection(
    tmp_path: Path,
) -> None:
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

    payload = json.loads(result.manifest_path.read_text())
    binding = payload["run_contract_binding"]

    assert binding["canonical_projection"]["training_run_spec"]["schema_id"]
    assert binding["canonical_projection"]["phase_program"]["checkpoint_barriers"]
    assert binding["canonical_projection_sha256"]


def test_checkpoint_fork_hardlinks_three_targets_and_survives_source_quarantine(
    tmp_path: Path,
) -> None:
    run_spec = _run_spec(minimax=True)
    program = run_spec.worker_execution.method_contract.phase_program
    source_root = tmp_path / "source"
    source = write_checkpoint_transaction(
        source_root,
        run_spec=run_spec,
        phase_program=program,
        barrier_name="after_warmup",
        coordinate=_coordinate(),
        slots=_minimax_slots(),
        population_member_ids={"adversary_population": ["adv-a", "adv-b"]},
    )

    targets = []
    for index in range(3):
        result = fork_checkpoint_transaction(
            source_root,
            tmp_path / f"target-{index}",
            target_run_spec=run_spec,
            target_phase_program=program,
            expected_slots=_minimax_slots(),
            expected_population_member_ids={"adversary_population": ["adv-a", "adv-b"]},
        )
        targets.append(result)
        assert set(result.slot_transfer_modes.values()) == {"hardlink"}
        loaded = load_latest_checkpoint(
            result.root,
            expected_run_spec=run_spec,
            expected_phase_program=program,
            expected_slots=_minimax_slots(),
            expected_population_member_ids={"adversary_population": ["adv-a", "adv-b"]},
        )
        assert loaded.slots["controller"].tolist() == [1.0, 2.0]

    for source_slot in source.manifest.slots:
        source_blob = _manifest_blob_path(source.manifest_path, source_slot.relative_path)
        assert source_blob.stat().st_nlink > 1
        source_stat = source_blob.stat()
        for target in targets:
            target_blob = _slot_blob_path(target.manifest_path, source_slot.slot)
            target_stat = target_blob.stat()
            assert (target_stat.st_dev, target_stat.st_ino) == (
                source_stat.st_dev,
                source_stat.st_ino,
            )

    quarantined = tmp_path / "source-quarantined"
    source_root.rename(quarantined)
    shutil.rmtree(quarantined)
    for target in targets:
        loaded = load_latest_checkpoint(
            target.root,
            expected_run_spec=run_spec,
            expected_phase_program=program,
            expected_slots=_minimax_slots(),
            expected_population_member_ids={"adversary_population": ["adv-a", "adv-b"]},
        )
        assert loaded.slots["rng"].tolist() == [11, 22]


def test_checkpoint_fork_transform_rewrites_only_transformed_slot(
    tmp_path: Path,
) -> None:
    run_spec = _run_spec(minimax=True)
    program = run_spec.worker_execution.method_contract.phase_program
    source_root = tmp_path / "source"
    source = write_checkpoint_transaction(
        source_root,
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

    forked = fork_checkpoint_transaction(
        source_root,
        tmp_path / "target",
        target_run_spec=run_spec,
        target_phase_program=program,
        expected_slots=expected,
        slot_transforms={"controller": resize_controller},
        transform_metadata={
            "controller": {
                "identity": "test:resize_controller",
                "parameters": {"from": 2, "to": 3},
            }
        },
    )

    assert forked.slot_transfer_modes["controller"] == "serialized"
    assert {mode for slot, mode in forked.slot_transfer_modes.items() if slot != "controller"} == {
        "hardlink"
    }
    source_controller = _slot_blob_path(source.manifest_path, "controller")
    target_controller = _slot_blob_path(forked.manifest_path, "controller")
    assert source_controller.stat().st_ino != target_controller.stat().st_ino
    for slot in ("controller_optimizer", "adversary_population", "adversary_optimizer", "rng"):
        assert _slot_blob_path(source.manifest_path, slot).stat().st_ino == (
            _slot_blob_path(forked.manifest_path, slot).stat().st_ino
        )
    loaded = load_latest_checkpoint(
        forked.root,
        expected_run_spec=run_spec,
        expected_phase_program=program,
        expected_slots=expected,
    )
    assert loaded.slots["controller"].tolist() == [1.0, 2.0, 0.0]
    assert forked.manifest.fork_provenance is not None
    controller_provenance = {slot.slot: slot for slot in forked.manifest.fork_provenance.slots}[
        "controller"
    ]
    source_controller_slot = {slot.slot: slot for slot in source.manifest.slots}["controller"]
    assert controller_provenance.source_sha256 == source_controller_slot.sha256
    assert controller_provenance.target_sha256 != source_controller_slot.sha256
    assert controller_provenance.transform is not None
    assert controller_provenance.transform.identity == "test:resize_controller"
    assert controller_provenance.transform.parameters == {"from": 2, "to": 3}


def test_checkpoint_fork_fails_closed_on_source_blob_hash_mismatch(
    tmp_path: Path,
) -> None:
    run_spec = _run_spec(minimax=True)
    program = run_spec.worker_execution.method_contract.phase_program
    source_root = tmp_path / "source"
    source = write_checkpoint_transaction(
        source_root,
        run_spec=run_spec,
        phase_program=program,
        barrier_name="after_warmup",
        coordinate=_coordinate(),
        slots=_minimax_slots(),
    )
    controller_blob = _slot_blob_path(source.manifest_path, "controller")
    original = controller_blob.read_bytes()
    controller_blob.write_bytes(bytes([original[0] ^ 0xFF]) + original[1:])
    target_root = tmp_path / "target"

    with pytest.raises(CheckpointIntegrityError, match="hash mismatch"):
        fork_checkpoint_transaction(
            source_root,
            target_root,
            target_run_spec=run_spec,
            target_phase_program=program,
            expected_slots=_minimax_slots(),
        )

    assert not (target_root / "latest.json").exists()


def test_checkpoint_fork_incompatible_target_and_copy_fallback(
    tmp_path: Path,
) -> None:
    run_spec = _run_spec(minimax=True)
    program = run_spec.worker_execution.method_contract.phase_program
    source_root = tmp_path / "source"
    source = write_checkpoint_transaction(
        source_root,
        run_spec=run_spec,
        phase_program=program,
        barrier_name="after_warmup",
        coordinate=_coordinate(),
        slots=_minimax_slots(),
    )
    incompatible = _incompatible_slot_run_spec(run_spec)
    target_root = tmp_path / "bad-target"

    with pytest.raises(CheckpointCompatibilityError, match="controller_v2"):
        fork_checkpoint_transaction(
            source_root,
            target_root,
            target_run_spec=incompatible,
            target_phase_program=incompatible.worker_execution.method_contract.phase_program,
            expected_slots=_minimax_slots(),
        )
    assert not (target_root / "latest.json").exists()

    def copy_strategy(source_blob: Path, target_blob: Path) -> str:
        shutil.copy2(source_blob, target_blob)
        return "copy"

    copied = fork_checkpoint_transaction(
        source_root,
        tmp_path / "copied-target",
        target_run_spec=run_spec,
        target_phase_program=program,
        expected_slots=_minimax_slots(),
        link_strategy=copy_strategy,
    )

    assert set(copied.slot_transfer_modes.values()) == {"copy"}
    assert _slot_blob_path(copied.manifest_path, "controller").stat().st_ino != (
        _slot_blob_path(source.manifest_path, "controller").stat().st_ino
    )


def test_environment_provenance_drift_warns_but_load_and_fork_succeed(
    tmp_path: Path,
) -> None:
    run_spec = _run_spec(minimax=True)
    program = run_spec.worker_execution.method_contract.phase_program
    source_root = tmp_path / "source"
    result = write_checkpoint_transaction(
        source_root,
        run_spec=run_spec,
        phase_program=program,
        barrier_name="after_warmup",
        coordinate=_coordinate(),
        slots=_minimax_slots(),
    )
    payload = json.loads(result.manifest_path.read_text())
    for slot in payload["slots"]:
        provenance = slot["structural_abi_fingerprint"]["environment_provenance"]
        provenance["jax_version"] = "drifted-test-jax"
    _rewrite_manifest_and_latest(result, payload)

    loaded = load_latest_checkpoint(
        source_root,
        expected_run_spec=run_spec,
        expected_phase_program=program,
        expected_slots=_minimax_slots(),
    )

    assert loaded.slots["controller"].tolist() == [1.0, 2.0]
    assert {notice.code for notice in loaded.provenance_notices} == {
        "environment_provenance_mismatch"
    }

    forked = fork_checkpoint_transaction(
        source_root,
        tmp_path / "target",
        target_run_spec=run_spec,
        target_phase_program=program,
        expected_slots=_minimax_slots(),
    )

    assert forked.source_provenance_notices
    assert forked.source_provenance_notices[0].code == "environment_provenance_mismatch"


def test_defaulted_legacy_projection_migrates_and_resumes(
    tmp_path: Path,
) -> None:
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
    payload = json.loads(result.manifest_path.read_text())
    payload["schema_version"] = TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION_V2
    binding = payload["run_contract_binding"]
    binding["canonical_projection"]["training_run_spec"]["schema_version"] = (
        "feedbax.spec.training_run.v1"
    )
    binding["canonical_projection"]["training_run_spec"].pop("on_nan")
    binding["canonical_projection_sha256"] = "legacy-v1-projection"
    _rewrite_manifest_and_latest(result, payload)

    loaded = load_latest_checkpoint(
        tmp_path,
        expected_run_spec=run_spec,
        expected_phase_program=program,
        expected_slots=_minimax_slots(),
    )

    assert loaded.manifest.schema_version == TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION
    assert loaded.manifest.run_contract_binding.canonical_projection is not None
    assert (
        loaded.manifest.run_contract_binding.canonical_projection["training_run_spec"]["on_nan"]
        == "raise"
    )


def test_checkpoint_fork_cli_batch_smoke_partial_failure(tmp_path: Path) -> None:
    run_spec = _run_spec(minimax=True)
    program = run_spec.worker_execution.method_contract.phase_program
    source_root = tmp_path / "source"
    write_checkpoint_transaction(
        source_root,
        run_spec=run_spec,
        phase_program=program,
        barrier_name="after_warmup",
        coordinate=_coordinate(),
        slots=_minimax_slots(),
    )
    good_spec = tmp_path / "good-spec.json"
    bad_spec = tmp_path / "bad-spec.json"
    _write_run_spec(good_spec, run_spec)
    _write_run_spec(bad_spec, _incompatible_slot_run_spec(run_spec))
    good_a = tmp_path / "good-a"
    bad = tmp_path / "bad"
    good_b = tmp_path / "good-b"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "feedbax",
            "checkpoint",
            "fork",
            "--source",
            str(source_root),
            "--target",
            f"{good_spec}:{good_a}",
            "--target",
            f"{bad_spec}:{bad}",
            "--target",
            f"{good_spec}:{good_b}",
        ],
        check=False,
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 1
    payload = json.loads(completed.stdout)
    assert [target["status"] for target in payload["targets"]] == [
        "ok",
        "error",
        "ok",
    ]
    assert (good_a / "latest.json").is_file()
    assert not (bad / "latest.json").exists()
    assert (good_b / "latest.json").is_file()


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
    incompatible["adversary_population"] = [
        jnp.array([0.1, 0.2, 0.3]),
        jnp.array([0.3, 0.4]),
    ]

    with pytest.raises(
        CheckpointCompatibilityError,
        match="structural ABI mismatch",
    ) as exc_info:
        load_latest_checkpoint(
            tmp_path,
            expected_run_spec=run_spec,
            expected_phase_program=program,
            expected_slots=incompatible,
        )

    message = str(exc_info.value)
    assert "treedef_equal=True" in message
    assert "leaf_count_delta=0" in message
    assert "checkpoint slot 'adversary_population'" in message
    assert "path=/0 field=shape recorded=[3] actual=[2]" in message
    assert "jax_enable_x64 differs" not in message


def test_manifest_structural_abi_tamper_fails_closed(tmp_path: Path) -> None:
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
    payload = json.loads(result.manifest_path.read_text())
    controller = next(slot for slot in payload["slots"] if slot["slot"] == "controller")
    controller["structural_abi_fingerprint"]["leaves"][0]["dtype"] = "float64"
    controller["structural_abi_fingerprint"]["fingerprint_sha256"] = "0" * 64
    _rewrite_manifest_and_latest(result, payload)

    with pytest.raises(CheckpointIntegrityError, match="structural ABI fingerprint is stale"):
        load_latest_checkpoint(
            tmp_path,
            expected_run_spec=run_spec,
            expected_phase_program=program,
            expected_slots=_minimax_slots(),
        )


def test_manifest_structural_abi_x64_mismatch_reports_leaf_diff_and_hint(
    tmp_path: Path,
) -> None:
    run_spec = _run_spec(minimax=True)
    program = run_spec.worker_execution.method_contract.phase_program
    previous_x64 = bool(jax.config.jax_enable_x64)
    try:
        jax.config.update("jax_enable_x64", True)
        write_checkpoint_transaction(
            tmp_path,
            run_spec=run_spec,
            phase_program=program,
            barrier_name="after_warmup",
            coordinate=_coordinate(),
            slots=_minimax_slots(),
        )
        jax.config.update("jax_enable_x64", False)

        with pytest.raises(
            CheckpointIntegrityError,
            match="structural ABI fingerprint is stale",
        ) as exc_info:
            load_latest_checkpoint(
                tmp_path,
                expected_run_spec=run_spec,
                expected_phase_program=program,
                expected_slots=_minimax_slots(),
            )
    finally:
        jax.config.update("jax_enable_x64", previous_x64)

    message = str(exc_info.value)
    assert "checkpoint slot 'controller'" in message
    assert 'path=/ field=dtype recorded="float64" actual="float32"' in message
    assert "recorded_x64_enabled=True" in message
    assert "actual_x64_enabled=False" in message
    assert "x64_side=recorded" in message
    assert "jax_enable_x64 differs between checkpoint writer and reader" in message


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



def test_batch_history_validates_per_batch_and_interval_without_declarations(
    tmp_path: Path,
) -> None:
    run_spec = _run_spec(minimax=True)
    program = run_spec.worker_execution.method_contract.phase_program
    slots = _minimax_slots()
    slots["controller"] = {
        "loss": BatchHistory(jnp.arange(4, dtype=jnp.float32), batch_axis=0),
        "chunks": BatchHistory(
            jnp.arange(6, dtype=jnp.float32).reshape(3, 2),
            batch_axis=1,
            granularity=Granularity.per_interval(2),
        ),
    }

    write_checkpoint_transaction(
        tmp_path,
        run_spec=run_spec,
        phase_program=program,
        barrier_name="after_warmup",
        coordinate=_coordinate(step=4),
        slots=slots,
        completed_training_batches=4,
    )
    loaded = load_latest_checkpoint(
        tmp_path,
        expected_run_spec=run_spec,
        expected_phase_program=program,
        expected_slots=slots,
    )

    assert isinstance(loaded.slots["controller"]["loss"], BatchHistory)
    assert loaded.slots["controller"]["chunks"].granularity.interval == 2


def test_batch_history_rejects_cumulative_length_at_segment_write(tmp_path: Path) -> None:
    run_spec = _run_spec(minimax=True)
    program = run_spec.worker_execution.method_contract.phase_program
    slots = _minimax_slots()
    slots["controller"] = BatchHistory(jnp.zeros((6,), dtype=jnp.float32), batch_axis=0)

    with pytest.raises(CheckpointConsistencyError, match="owning segment"):
        write_checkpoint_transaction(
            tmp_path,
            run_spec=run_spec,
            phase_program=program,
            barrier_name="after_warmup",
            coordinate=_coordinate(step=4),
            slots=slots,
            completed_training_batches=4,
        )


def test_segment_lineage_concatenates_histories_and_materializes_derived(
    tmp_path: Path,
) -> None:
    run_spec = _run_spec(minimax=True)
    program = run_spec.worker_execution.method_contract.phase_program
    parent_root = tmp_path / "parent"
    child_root = tmp_path / "child"
    parent_slots = _minimax_slots()
    parent_slots["controller"] = BatchHistory(jnp.arange(12_000), batch_axis=0)
    parent = write_checkpoint_transaction(
        parent_root,
        run_spec=run_spec,
        phase_program=program,
        barrier_name="after_warmup",
        coordinate=_coordinate(step=12_000),
        slots=parent_slots,
        completed_training_batches=12_000,
    )
    child_slots = _minimax_slots()
    child_slots["controller"] = BatchHistory(jnp.arange(12_000, 16_500), batch_axis=0)
    child = write_checkpoint_transaction(
        child_root,
        run_spec=run_spec,
        phase_program=program,
        barrier_name="after_warmup",
        coordinate=_coordinate(step=4_500),
        slots=child_slots,
        completed_training_batches=4_500,
    )
    payload = json.loads(child.manifest_path.read_text())
    payload["segment_lineage"].update(
        parent_transaction_id=parent.manifest.transaction_id,
        start_batch=12_000,
    )
    _rewrite_manifest_and_latest(child, payload)

    stitched = concatenate_checkpoint_histories(
        child_root,
        parent_roots={parent.manifest.transaction_id: parent_root},
    )
    assert stitched.completed_training_batches == 16_500
    assert jnp.array_equal(stitched.histories["controller/"].value, jnp.arange(16_500))

    output = materialize_concatenated_checkpoint_histories(
        child_root,
        tmp_path / "derived.pkl",
        parent_roots={parent.manifest.transaction_id: parent_root},
    )
    with output.open("rb") as stream:
        derived = pickle.load(stream)
    assert derived["derived"] is True
    assert derived["resume_source"] is False


@pytest.mark.parametrize("failure", ["missing", "offset", "granularity"])
def test_segment_lineage_reader_fails_closed(tmp_path: Path, failure: str) -> None:
    run_spec = _run_spec(minimax=True)
    program = run_spec.worker_execution.method_contract.phase_program
    parent_root = tmp_path / "parent"
    child_root = tmp_path / "child"
    parent_slots = _minimax_slots()
    parent_slots["controller"] = BatchHistory(jnp.arange(4), batch_axis=0)
    parent = write_checkpoint_transaction(
        parent_root, run_spec=run_spec, phase_program=program,
        barrier_name="after_warmup", coordinate=_coordinate(step=4),
        slots=parent_slots, completed_training_batches=4,
    )
    child_slots = _minimax_slots()
    child_slots["controller"] = BatchHistory(
        jnp.arange(1 if failure == "granularity" else 2), batch_axis=0,
        granularity=Granularity.per_interval(2) if failure == "granularity" else None,
    )
    child = write_checkpoint_transaction(
        child_root, run_spec=run_spec, phase_program=program,
        barrier_name="after_warmup", coordinate=_coordinate(step=2),
        slots=child_slots, completed_training_batches=2,
    )
    payload = json.loads(child.manifest_path.read_text())
    payload["segment_lineage"].update(
        parent_transaction_id=parent.manifest.transaction_id,
        start_batch=5 if failure == "offset" else 4,
    )
    _rewrite_manifest_and_latest(child, payload)
    roots = {} if failure == "missing" else {parent.manifest.transaction_id: parent_root}
    with pytest.raises(CheckpointIntegrityError):
        concatenate_checkpoint_histories(child_root, parent_roots=roots)



def test_checkpoint_continuation_rejects_unknown_schema_version() -> None:
    with pytest.raises(ValueError, match="migration_intentionally_absent=yes"):
        CheckpointContinuationRequest.model_validate(
            {
                "schema_version": "feedbax.spec.training_checkpoint_continuation.v0",
                "source_completed_batches": 12000,
                "additional_batches": 200,
            }
        )


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
            slot["coordinate"]["program_step"] = 1
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
            expected_population_member_ids={"adversary_population": ["adv-a", "adv-b", "adv-c"]},
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
    assert detect_known_legacy_checkpoint_layout(tmp_path) is None

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


@pytest.mark.parametrize(
    ("layout_name", "populate"),
    [
        (
            "Feedbax supervised trainer legacy checkpoint",
            lambda root: (root / "last_batch.txt").write_text("10\n"),
        ),
        (
            "RLRMP Equinox stream legacy checkpoint",
            lambda root: (
                (root / "checkpoint_000001").mkdir(),
                (root / "checkpoint_000001" / "model.eqx").write_bytes(b"model"),
                (root / "checkpoint_000001" / "optimizer_state.eqx").write_bytes(b"optimizer"),
                (root / "checkpoint_000001" / "metadata.json").write_text("{}"),
            ),
        ),
    ],
)
def test_known_legacy_layout_missing_pointer_names_adoption_remedy(
    tmp_path: Path,
    layout_name: str,
    populate,
) -> None:
    run_spec = _run_spec(minimax=True)
    program = run_spec.worker_execution.method_contract.phase_program
    populate(tmp_path)

    layout = detect_known_legacy_checkpoint_layout(tmp_path)
    assert layout is not None
    assert layout.name == layout_name

    with pytest.raises(CheckpointCompatibilityError) as excinfo:
        load_latest_checkpoint(
            tmp_path,
            expected_run_spec=run_spec,
            expected_phase_program=program,
            expected_slots=_minimax_slots(),
        )

    message = str(excinfo.value)
    assert layout_name in message
    assert LEGACY_CHECKPOINT_ADOPTION_ENTRYPOINT in message
    assert "producing commit" in message
    assert "path-mapping rules" in message
    assert LEGACY_CHECKPOINT_ADOPTION_DOCS in message


def test_changed_learning_rate_fails_closed_with_field_diff_unless_override(
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
    changed.training_config.learning_rate = 0.5

    with pytest.raises(CheckpointContractBindingError, match="learning_rate"):
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


def test_legacy_absent_binding_projection_loads_and_reports_hash_field_diff(
    tmp_path: Path,
) -> None:
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
    payload = json.loads(result.manifest_path.read_text())
    payload["schema_version"] = TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION_V2
    binding = payload["run_contract_binding"]
    binding.pop("canonical_projection")
    binding.pop("canonical_projection_sha256")
    _rewrite_manifest_and_latest(result, payload)

    loaded = load_latest_checkpoint(
        tmp_path,
        expected_run_spec=run_spec,
        expected_phase_program=program,
        expected_slots=_minimax_slots(),
    )

    assert loaded.manifest.run_contract_binding.canonical_projection is None
    assert loaded.manifest.run_contract_binding.metadata["projection_status"] == ("legacy_absent")

    changed = run_spec.model_copy(deep=True)
    changed.training_config.learning_rate = 0.5

    with pytest.raises(CheckpointContractBindingError) as exc_info:
        load_latest_checkpoint(
            tmp_path,
            expected_run_spec=changed,
            expected_phase_program=program,
            expected_slots=_minimax_slots(),
        )

    message = str(exc_info.value)
    assert "stored canonical projection is unavailable for this legacy binding" in message
    assert "hash_field_mismatches=['training_run_spec_sha256']" in message
    assert "method_payload_sha256" in message
    assert "phase_program_sha256" in message
    assert "graph_sha256" in message


def test_interrupted_toy_resume_matches_uninterrupted(tmp_path: Path) -> None:
    run_spec = _run_spec()
    program = run_spec.worker_execution.method_contract.phase_program

    def step(state: dict[str, object]) -> dict[str, object]:
        return {
            "model": state["model"] + state["optimizer"]["count"],
            "optimizer": {"count": state["optimizer"]["count"] + 1},
            "prng": state["prng"] + 17,
            "batch_counter": state["batch_counter"] + 1,
        }

    continuous = {
        "model": jnp.array([0.0]),
        "optimizer": {"count": jnp.array(1.0)},
        "prng": jnp.array([0, 1], dtype=jnp.uint32),
        "batch_counter": jnp.array(0, dtype=jnp.int32),
    }
    interrupted = dict(continuous)
    for _ in range(4):
        continuous = step(continuous)
    for _ in range(2):
        interrupted = step(interrupted)

    coordinate = ProgressCoordinate(
        run_id="run-2",
        phase="train_batch",
        program_step=2,
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


def test_distinct_barrier_mapping_requires_explicit_coordinate_provenance() -> None:
    with pytest.raises(
        ValueError, match="require explicit target_coordinate and coordinate_mapping"
    ):
        CheckpointForkBarrierMapping(
            source_barrier="after_train_chunk",
            target_barrier="after_adaptive_epsilon_train_chunk",
        )


def test_checkpoint_fork_rejects_mapping_for_wrong_actual_source_barrier(
    tmp_path: Path,
) -> None:
    run_spec = _run_spec(minimax=True)
    program = run_spec.worker_execution.method_contract.phase_program
    write_checkpoint_transaction(
        tmp_path / "source",
        run_spec=run_spec,
        phase_program=program,
        barrier_name="after_warmup",
        coordinate=_coordinate(),
        slots=_minimax_slots(),
    )
    mapping = CheckpointForkBarrierMapping(
        source_barrier="after_adversarial",
        target_barrier="after_warmup",
        target_coordinate=_coordinate(),
        coordinate_mapping={"identity": "tests.invalid_source_barrier.v1", "parameters": {}},
    )

    with pytest.raises(
        CheckpointCompatibilityError,
        match="source barrier mapping does not match source manifest",
    ):
        fork_checkpoint_transaction(
            tmp_path / "source",
            tmp_path / "target",
            target_run_spec=run_spec,
            target_phase_program=program,
            expected_slots=_minimax_slots(),
            barrier_mapping=mapping,
        )
