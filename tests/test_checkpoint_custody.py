from __future__ import annotations

import hashlib
import json
import math
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
    CheckpointForkPlan,
    CheckpointForkSourcePreparation,
    CheckpointForkTarget,
    CheckpointForkTransformRecord,
    CheckpointForkTransformStep,
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
    TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION_V7,
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
    CheckpointSlotSpec,
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
    CheckpointReferenceResolutionError,
    CheckpointForkPlanBindings,
    CheckpointForkTransformRegistration,
    CheckpointForkTransformRegistry,
    checkpoint_slot_names,
    derive_checkpoint_fork_compatibility_projection,
    detect_known_legacy_checkpoint_layout,
    fork_checkpoint_transaction,
    fork_checkpoint_plan,
    load_latest_checkpoint,
    load_checkpoint_custody_documents,
    load_checkpoint_latest_pointer_json,
    load_checkpoint_transaction_manifest_file,
    resolve_checkpoint_custody_ref,
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


def _write_resolver_checkpoint(
    tmp_path: Path,
    *,
    slots: dict[str, object] | None = None,
):
    run_spec = _run_spec(minimax=True)
    return write_checkpoint_transaction(
        tmp_path,
        run_spec=run_spec,
        phase_program=run_spec.worker_execution.method_contract.phase_program,
        barrier_name="after_warmup",
        coordinate=_coordinate(),
        slots=slots or _minimax_slots(),
    )


def _resolver_parent_ref(result, **updates: object) -> ParentRef:
    values: dict[str, object] = {
        "kind": "TrainingCheckpointTransactionManifest",
        "id": result.manifest.transaction_id,
        "role": "training_checkpoint_custody",
        "uri": result.manifest_path.relative_to(result.root).as_posix(),
        "metadata": {"manifest_sha256": _sha256_file(result.manifest_path)},
    }
    values.update(updates)
    return ParentRef.model_validate(values)


def _replace_resolver_slot_blob(result, slot_name: str, value: object) -> dict[str, object]:
    payload = json.loads(result.manifest_path.read_text())
    slot = next(item for item in payload["slots"] if item["slot"] == slot_name)
    digest = next(
        item
        for item in payload["content_integrity_digest"]["slots"]
        if item["slot"] == slot_name
    )
    blob_path = _manifest_blob_path(result.manifest_path, slot["relative_path"])
    blob_bytes = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    blob_sha256 = hashlib.sha256(blob_bytes).hexdigest()
    blob_path.write_bytes(blob_bytes)
    slot["sha256"] = blob_sha256
    slot["size_bytes"] = len(blob_bytes)
    for record in (slot["content_digest"], digest):
        record["blob_sha256"] = blob_sha256
        record["blob_size_bytes"] = len(blob_bytes)
        record["slot_root_sha256"] = custody_module._slot_root_sha256(
            slot_name,
            blob_sha256,
            [
                custody_module.SlotLeafContentDigest.model_validate(item)
                for item in record["leaf_hashes"]
            ],
        )
    payload["content_integrity_digest"]["transaction_root_sha256"] = (
        custody_module._transaction_root_sha256(
            [
                custody_module.SlotContentDigest.model_validate(item)
                for item in payload["content_integrity_digest"]["slots"]
            ]
        )
    )
    _write_json(result.manifest_path, payload)
    return payload


class _RuntimeErrorAfterRoundTrip:
    def __init__(self, *, decoded: bool = False) -> None:
        self.decoded = decoded

    def __reduce__(self):
        if self.decoded:
            raise RuntimeError("post-unpickle leaf verification failed")
        return (_decoded_runtime_error_leaf, ())

    def __repr__(self) -> str:
        return "_RuntimeErrorAfterRoundTrip()"


def _decoded_runtime_error_leaf() -> _RuntimeErrorAfterRoundTrip:
    return _RuntimeErrorAfterRoundTrip(decoded=True)


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
            TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION_V7,
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
            "training-checkpoint-transaction-v7-to-v8-mapped-axes",
    ]
    assert migrated.payload["metadata"]["batch_history_tree_migration"] == (
        "declared_paths_v5_to_v6"
    )
    assert migrated.payload["segment_lineage"] == {
        "start_batch": 0,
        "segment_batch_count": 0,
        "history_granularities": {},
    }


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


def test_custody_records_program_batch_and_visit_coordinates_independently(
    tmp_path: Path,
) -> None:
    spec, program = _batch_counter_program_and_spec()
    result = write_checkpoint_transaction(
        tmp_path,
        run_spec=spec,
        phase_program=program,
        barrier_name="after_train_batch",
        coordinate=ProgressCoordinate(
            run_id="independent-coordinates",
            phase="train_batch",
            program_step=500,
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

    assert result.manifest.completed_coordinate.program_step == 500
    assert result.manifest.completed_training_batches == 12000
    assert result.manifest.metadata["barrier_visit_ordinal"] == 24


def test_checkpoint_fork_preserves_completed_batches_independently_of_program_step(
    tmp_path: Path,
) -> None:
    run_spec = _run_spec()
    program = run_spec.worker_execution.method_contract.phase_program
    slots = {
        "model": 0,
        "optimizer": {"count": 0},
        "prng": [0, 1],
        "batch_counter": 3,
    }
    source = write_checkpoint_transaction(
        tmp_path / "source",
        run_spec=run_spec,
        phase_program=program,
        barrier_name="after_train_batch",
        coordinate=ProgressCoordinate(
            run_id="fork-source",
            phase="train_batch",
            program_step=500,
            completed_barrier="after_train_batch",
        ),
        slots=slots,
        metadata={"barrier_visit_ordinal": 0},
    )

    forked = fork_checkpoint_transaction(
        source.root,
        tmp_path / "target",
        target_run_spec=run_spec,
        target_phase_program=program,
        expected_slots=slots,
    )

    assert source.manifest.completed_training_batches == 3
    assert forked.manifest.completed_training_batches == 3
    assert forked.manifest.completed_coordinate.program_step == 500
    assert forked.manifest.metadata["barrier_visit_ordinal"] == 0


@pytest.mark.parametrize("visit_ordinal", [-1, True, "0"])
def test_custody_retains_strict_barrier_visit_ordinal_validation(
    tmp_path: Path,
    visit_ordinal: object,
) -> None:
    spec, program = _batch_counter_program_and_spec()
    with pytest.raises(CheckpointConsistencyError, match="must be a non-negative integer"):
        write_checkpoint_transaction(
            tmp_path,
            run_spec=spec,
            phase_program=program,
            barrier_name="after_train_batch",
            coordinate=ProgressCoordinate(
                run_id="invalid-visit",
                phase="train_batch",
                program_step=500,
            ),
            slots={
                "model": 0,
                "optimizer": {"count": 0},
                "prng": [0, 1],
                "batch_counter": 12000,
            },
            metadata={"barrier_visit_ordinal": visit_ordinal},
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


def test_run_contract_binding_normalizes_nested_signed_zero() -> None:
    positive = _run_spec()
    negative = positive.model_copy(deep=True)
    _set_graph_gain(positive, 0.0)
    _set_graph_gain(negative, -0.0)
    program = positive.worker_execution.method_contract.phase_program

    positive_binding = custody_module.run_contract_binding(positive, program)
    negative_binding = custody_module.run_contract_binding(negative, program)

    assert positive_binding.training_run_spec_sha256 == (
        negative_binding.training_run_spec_sha256
    )
    assert positive_binding.graph_sha256 == negative_binding.graph_sha256
    assert positive_binding.canonical_projection_sha256 == (
        negative_binding.canonical_projection_sha256
    )


def test_signed_zero_normalization_preserves_other_floating_values() -> None:
    payload = {
        "positive": 1.5,
        "negative": -2.5,
        "nan": float("nan"),
        "positive_infinity": float("inf"),
        "negative_infinity": float("-inf"),
        "nested": [0.0, {"negative_zero": -0.0}],
    }

    normalized = custody_module._normalize_signed_zero(payload)

    assert normalized["positive"] == 1.5
    assert normalized["negative"] == -2.5
    assert math.isnan(normalized["nan"])
    assert normalized["positive_infinity"] == float("inf")
    assert normalized["negative_infinity"] == float("-inf")
    assert math.copysign(1.0, normalized["nested"][0]) == 1.0
    assert math.copysign(1.0, normalized["nested"][1]["negative_zero"]) == 1.0


def _set_graph_gain(run_spec: TrainingRunSpec, gain: float) -> None:
    assert run_spec.graph.inline is not None
    run_spec.graph.inline["nodes"]["gain"]["params"]["gain"] = gain


def _rewrite_binding_as_legacy_lexical_hashes(binding: dict[str, object]) -> None:
    projection = binding["canonical_projection"]
    training_run_spec = projection["training_run_spec"]
    phase_program = projection["phase_program"]
    optimizer_bindings = phase_program["optimizer_bindings"]
    binding["algorithm_version"] = "feedbax.training_checkpoint.run_contract_binding.v2"
    binding["training_run_spec_sha256"] = custody_module._canonical_hash(training_run_spec)
    binding["method_payload_sha256"] = custody_module._canonical_hash(
        training_run_spec["method_payload"]
    )
    binding["phase_program_sha256"] = custody_module._canonical_hash(phase_program)
    binding["objective_sha256"] = custody_module._canonical_hash(training_run_spec["objective"])
    binding["graph_sha256"] = custody_module._canonical_hash(training_run_spec["graph"])
    binding["optimizer_bindings_sha256"] = custody_module._canonical_hash(optimizer_bindings)
    binding["canonical_projection_sha256"] = custody_module._canonical_hash(projection)


def test_strict_load_accepts_legacy_signed_zero_projection_with_proof(
    tmp_path: Path,
) -> None:
    recorded_run_spec = _run_spec(minimax=True)
    _set_graph_gain(recorded_run_spec, -0.0)
    program = recorded_run_spec.worker_execution.method_contract.phase_program
    result = write_checkpoint_transaction(
        tmp_path,
        run_spec=recorded_run_spec,
        phase_program=program,
        barrier_name="after_warmup",
        coordinate=_coordinate(),
        slots=_minimax_slots(),
    )
    payload = json.loads(result.manifest_path.read_text())
    _rewrite_binding_as_legacy_lexical_hashes(payload["run_contract_binding"])
    _rewrite_manifest_and_latest(result, payload)
    expected_run_spec = recorded_run_spec.model_copy(deep=True)
    _set_graph_gain(expected_run_spec, 0.0)

    loaded = load_latest_checkpoint(
        tmp_path,
        expected_run_spec=expected_run_spec,
        expected_phase_program=program,
        expected_slots=_minimax_slots(),
    )

    assert loaded.manifest.transaction_id == result.manifest.transaction_id


def test_strict_load_accepts_zero_free_v2_projection_with_proof(tmp_path: Path) -> None:
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
    payload["run_contract_binding"]["algorithm_version"] = (
        "feedbax.training_checkpoint.run_contract_binding.v2"
    )
    _rewrite_manifest_and_latest(result, payload)

    loaded = load_latest_checkpoint(
        tmp_path,
        expected_run_spec=run_spec,
        expected_phase_program=program,
        expected_slots=_minimax_slots(),
    )

    assert loaded.manifest.transaction_id == result.manifest.transaction_id


def test_strict_load_rejects_unsupported_algorithm_with_equal_projection_digest(
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
    payload["run_contract_binding"]["algorithm_version"] = (
        "feedbax.training_checkpoint.run_contract_binding.v99"
    )
    _rewrite_manifest_and_latest(result, payload)

    with pytest.raises(CheckpointContractBindingError):
        load_latest_checkpoint(
            tmp_path,
            expected_run_spec=run_spec,
            expected_phase_program=program,
            expected_slots=_minimax_slots(),
        )


@pytest.mark.parametrize(
    ("mutate_binding", "expected_gain"),
    [
        (lambda binding: binding.__setitem__("canonical_projection_sha256", "0" * 64), 0.0),
        (lambda binding: None, 1.0),
        (
            lambda binding: binding.__setitem__(
                "algorithm_version",
                "feedbax.training_checkpoint.run_contract_binding.v99",
            ),
            0.0,
        ),
    ],
    ids=["stale-projection-digest", "unequal-projection", "unsupported-algorithm"],
)
def test_legacy_signed_zero_projection_compatibility_fails_closed(
    tmp_path: Path,
    mutate_binding,
    expected_gain: float,
) -> None:
    recorded_run_spec = _run_spec(minimax=True)
    _set_graph_gain(recorded_run_spec, -0.0)
    program = recorded_run_spec.worker_execution.method_contract.phase_program
    result = write_checkpoint_transaction(
        tmp_path,
        run_spec=recorded_run_spec,
        phase_program=program,
        barrier_name="after_warmup",
        coordinate=_coordinate(),
        slots=_minimax_slots(),
    )
    payload = json.loads(result.manifest_path.read_text())
    binding = payload["run_contract_binding"]
    _rewrite_binding_as_legacy_lexical_hashes(binding)
    mutate_binding(binding)
    _rewrite_manifest_and_latest(result, payload)
    expected_run_spec = recorded_run_spec.model_copy(deep=True)
    _set_graph_gain(expected_run_spec, expected_gain)

    with pytest.raises(CheckpointContractBindingError):
        load_latest_checkpoint(
            tmp_path,
            expected_run_spec=expected_run_spec,
            expected_phase_program=program,
            expected_slots=_minimax_slots(),
        )


def test_projectionless_legacy_signed_zero_binding_keeps_lexical_match(
    tmp_path: Path,
) -> None:
    run_spec = _run_spec(minimax=True)
    _set_graph_gain(run_spec, -0.0)
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
    _rewrite_binding_as_legacy_lexical_hashes(binding)
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


@pytest.mark.parametrize("field", ["schema_id", "schema_version"])
def test_v8_transaction_rejects_unknown_nested_fork_provenance_identity(
    tmp_path: Path,
    field: str,
) -> None:
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
    forked = fork_checkpoint_transaction(
        source_root,
        tmp_path / "target",
        target_run_spec=run_spec,
        expected_slots=_minimax_slots(),
    )
    payload = json.loads(forked.manifest_path.read_text())
    payload["fork_provenance"][field] += ".unknown"
    _rewrite_manifest_and_latest(forked, payload)

    with pytest.raises(CheckpointIntegrityError, match=field):
        load_latest_checkpoint(
            forked.root,
            expected_run_spec=run_spec,
            expected_phase_program=program,
            expected_slots=_minimax_slots(),
        )


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
        metadata={"checkpoint_continuation_applied": True},
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
    assert "checkpoint_continuation_applied" not in forked.manifest.metadata


def _typed_fork_plan(
    run_spec: TrainingRunSpec,
    expected_slots: dict[str, object],
    *,
    transformed: bool = True,
) -> CheckpointForkPlan:
    transforms = []
    if transformed:
        transforms = [
            CheckpointForkTransformStep(
                step_id="append-controller",
                stage="source_pre",
                records=[
                    CheckpointForkTransformRecord(
                        slot="controller",
                        identity="tests.append-controller.v1",
                        parameters={"value": 3.0},
                    )
                ],
            ),
            CheckpointForkTransformStep(
                step_id="scale-controller",
                stage="source_pre",
                records=[
                    CheckpointForkTransformRecord(
                        slot="controller",
                        identity="tests.scale-controller.v1",
                        parameters={"factor": 2.0},
                    )
                ],
            ),
        ]
    return CheckpointForkPlan(
        source=CheckpointForkSourcePreparation(
            checkpoint_root_ref="source",
            transforms=transforms,
        ),
        targets=[
            CheckpointForkTarget(
                target_id="target-a",
                row_id="row-a",
                checkpoint_root_ref="target-a",
                run_spec_ref="run",
                slot_template_ref="slots",
                compatibility=derive_checkpoint_fork_compatibility_projection(
                    run_spec,
                    run_spec.worker_execution.method_contract.phase_program,
                    expected_slots,
                ),
            ),
            CheckpointForkTarget(
                target_id="target-b",
                row_id="row-b",
                checkpoint_root_ref="target-b",
                run_spec_ref="run",
                slot_template_ref="slots",
                compatibility=derive_checkpoint_fork_compatibility_projection(
                    run_spec,
                    run_spec.worker_execution.method_contract.phase_program,
                    expected_slots,
                ),
            ),
        ],
    )


def test_typed_checkpoint_fork_plan_preflights_once_and_records_projection(
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
    expected = _minimax_slots()
    expected["controller"] = jnp.zeros((3,), dtype=jnp.float32)
    calls: list[str] = []
    registry = CheckpointForkTransformRegistry()

    def append_controller(slots, parameters):
        calls.append("append")
        return {
            **slots,
            "controller": jnp.concatenate(
                [slots["controller"], jnp.asarray([parameters["value"]])]
            ),
        }

    def scale_controller(slots, parameters):
        calls.append("scale")
        return {**slots, "controller": slots["controller"] * parameters["factor"]}

    registry.register(
        CheckpointForkTransformRegistration("tests.append-controller.v1", append_controller)
    )
    registry.register(
        CheckpointForkTransformRegistration("tests.scale-controller.v1", scale_controller)
    )
    plan = _typed_fork_plan(run_spec, expected)
    results = fork_checkpoint_plan(
        plan,
        CheckpointForkPlanBindings(
            checkpoint_roots={
                "source": tmp_path / "source",
                "target-a": tmp_path / "target-a",
                "target-b": tmp_path / "target-b",
            },
            run_specs={"run": run_spec},
            slot_templates={"slots": expected},
        ),
        transform_registry=registry,
    )

    assert calls == ["append", "scale"]
    hashes = set()
    for target_id, result in results.items():
        loaded = load_latest_checkpoint(
            result.root,
            expected_run_spec=run_spec,
            expected_phase_program=program,
            expected_slots=expected,
        )
        assert loaded.slots["controller"].tolist() == [2.0, 4.0, 6.0]
        assert result.manifest.fork_provenance is not None
        metadata = result.manifest.fork_provenance.metadata
        hashes.add(metadata["checkpoint_fork_plan_sha256"])
        assert metadata["checkpoint_fork_plan_target_id"] == target_id
        assert metadata["checkpoint_fork_plan_compatibility_projection"]["targets"]
    assert len(hashes) == 1


def test_typed_checkpoint_fork_plan_unknown_transform_fails_before_writes(
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
    with pytest.raises(CheckpointCompatibilityError, match="unregistered"):
        fork_checkpoint_plan(
            _typed_fork_plan(run_spec, _minimax_slots()),
            CheckpointForkPlanBindings(
                checkpoint_roots={
                    "source": tmp_path / "source",
                    "target-a": tmp_path / "target-a",
                    "target-b": tmp_path / "target-b",
                },
                run_specs={"run": run_spec},
                slot_templates={"slots": _minimax_slots()},
            ),
            transform_registry=CheckpointForkTransformRegistry(),
        )
    assert not (tmp_path / "target-a").exists()
    assert not (tmp_path / "target-b").exists()


def test_typed_checkpoint_fork_plan_preserves_untransformed_hardlinks(tmp_path: Path) -> None:
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
    results = fork_checkpoint_plan(
        _typed_fork_plan(run_spec, _minimax_slots(), transformed=False),
        CheckpointForkPlanBindings(
            checkpoint_roots={
                "source": tmp_path / "source",
                "target-a": tmp_path / "target-a",
                "target-b": tmp_path / "target-b",
            },
            run_specs={"run": run_spec},
            slot_templates={"slots": _minimax_slots()},
        ),
    )
    assert set(results) == {"target-a", "target-b"}
    assert all(
        set(result.slot_transfer_modes.values()) == {"hardlink"}
        for result in results.values()
    )


def test_typed_checkpoint_fork_plan_requires_declared_target_only_slot(
    tmp_path: Path,
) -> None:
    source_spec = _run_spec(minimax=True)
    source_program = source_spec.worker_execution.method_contract.phase_program
    write_checkpoint_transaction(
        tmp_path / "source",
        run_spec=source_spec,
        phase_program=source_program,
        barrier_name="after_warmup",
        coordinate=_coordinate(),
        slots=_minimax_slots(),
    )
    target_spec = source_spec.model_copy(deep=True)
    target_program = target_spec.worker_execution.method_contract.phase_program
    target_program.checkpoint_barriers[0].slots.append(
        CheckpointSlotSpec(slot="adaptive_state")
    )
    expected = {**_minimax_slots(), "adaptive_state": jnp.zeros((2,), dtype=jnp.float32)}
    declaration = {"identity": "tests.adaptive-state-slot.v1"}
    plan = CheckpointForkPlan(
        source=CheckpointForkSourcePreparation(checkpoint_root_ref="source"),
        targets=[
            CheckpointForkTarget(
                target_id="target",
                checkpoint_root_ref="target",
                run_spec_ref="run",
                slot_template_ref="slots",
                compatibility=derive_checkpoint_fork_compatibility_projection(
                    target_spec,
                    target_program,
                    expected,
                ),
                transforms=[
                    CheckpointForkTransformStep(
                        step_id="add-adaptive-state",
                        stage="target_post",
                        records=[
                            CheckpointForkTransformRecord(
                                slot="adaptive_state",
                                identity="tests.add-adaptive-state.v1",
                            )
                        ],
                        target_only_slots={"adaptive_state": declaration},
                    )
                ],
            )
        ],
    )
    registry = CheckpointForkTransformRegistry()
    registry.register(
        CheckpointForkTransformRegistration(
            "tests.add-adaptive-state.v1",
            lambda slots, parameters: {
                **slots,
                "adaptive_state": jnp.zeros((2,), dtype=jnp.float32),
            },
        )
    )
    result = fork_checkpoint_plan(
        plan,
        CheckpointForkPlanBindings(
            checkpoint_roots={
                "source": tmp_path / "source",
                "target": tmp_path / "target",
            },
            run_specs={"run": target_spec},
            slot_templates={"slots": expected},
        ),
        transform_registry=registry,
    )["target"]
    provenance = {slot.slot: slot for slot in result.manifest.fork_provenance.slots}
    assert provenance["adaptive_state"].source_sha256 is None
    assert provenance["adaptive_state"].transform.metadata["target_only_declaration"] == declaration


def test_typed_checkpoint_fork_plan_history_policy_fails_closed_before_write(
    tmp_path: Path,
) -> None:
    source_spec = _run_spec(minimax=True)
    source_program = source_spec.worker_execution.method_contract.phase_program
    write_checkpoint_transaction(
        tmp_path / "source",
        run_spec=source_spec,
        phase_program=source_program,
        barrier_name="after_warmup",
        coordinate=_coordinate(),
        slots=_minimax_slots(),
    )
    continuation = CheckpointContinuationRequest(
        source_completed_batches=0,
        additional_batches=5,
    )
    target_spec = source_spec.model_copy(
        update={
            "checkpoint_progress": source_spec.checkpoint_progress.model_copy(
                update={"continuation": continuation}
            )
        }
    )
    plan = CheckpointForkPlan(
        source=CheckpointForkSourcePreparation(checkpoint_root_ref="source"),
        targets=[
            CheckpointForkTarget(
                target_id="target",
                checkpoint_root_ref="target",
                run_spec_ref="run",
                slot_template_ref="slots",
                compatibility=derive_checkpoint_fork_compatibility_projection(
                    target_spec,
                    target_spec.worker_execution.method_contract.phase_program,
                    _minimax_slots(),
                ),
            )
        ],
    )
    with pytest.raises(CheckpointCompatibilityError, match="history policy is preserve"):
        fork_checkpoint_plan(
            plan,
            CheckpointForkPlanBindings(
                checkpoint_roots={
                    "source": tmp_path / "source",
                    "target": tmp_path / "target",
                },
                run_specs={"run": target_spec},
                slot_templates={"slots": _minimax_slots()},
            ),
        )
    assert not (tmp_path / "target").exists()


def test_typed_checkpoint_fork_plan_preflights_all_target_structures(
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
    bad_slots = _minimax_slots()
    bad_slots["controller"] = jnp.zeros((9,), dtype=jnp.float32)
    plan = _typed_fork_plan(run_spec, _minimax_slots(), transformed=False)
    plan = plan.model_copy(
        update={
            "targets": [
                plan.targets[0],
                plan.targets[1].model_copy(update={"slot_template_ref": "bad-slots"}),
            ]
        }
    )
    with pytest.raises(CheckpointCompatibilityError, match="target-b.*compatibility ABI mismatch"):
        fork_checkpoint_plan(
            plan,
            CheckpointForkPlanBindings(
                checkpoint_roots={
                    "source": tmp_path / "source",
                    "target-a": tmp_path / "target-a",
                    "target-b": tmp_path / "target-b",
                },
                run_specs={"run": run_spec},
                slot_templates={"slots": _minimax_slots(), "bad-slots": bad_slots},
            ),
        )
    assert not (tmp_path / "target-a").exists()
    assert not (tmp_path / "target-b").exists()


def test_typed_checkpoint_fork_plan_rejects_runtime_run_contract_drift(
    tmp_path: Path,
) -> None:
    declared_spec = _run_spec(minimax=True)
    program = declared_spec.worker_execution.method_contract.phase_program
    write_checkpoint_transaction(
        tmp_path / "source",
        run_spec=declared_spec,
        phase_program=program,
        barrier_name="after_warmup",
        coordinate=_coordinate(),
        slots=_minimax_slots(),
    )
    plan = _typed_fork_plan(declared_spec, _minimax_slots(), transformed=False)
    runtime_spec = _incompatible_slot_run_spec(declared_spec)
    with pytest.raises(
        CheckpointCompatibilityError,
        match="target-a.*run-contract projection sha256 mismatch",
    ):
        fork_checkpoint_plan(
            plan,
            CheckpointForkPlanBindings(
                checkpoint_roots={
                    "source": tmp_path / "source",
                    "target-a": tmp_path / "target-a",
                    "target-b": tmp_path / "target-b",
                },
                run_specs={"run": runtime_spec},
                slot_templates={"slots": _minimax_slots()},
            ),
        )
    assert not (tmp_path / "target-a").exists()
    assert not (tmp_path / "target-b").exists()


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

    with pytest.raises(CheckpointConsistencyError, match="fixed-size streaming state"):
        write_checkpoint_transaction(
            tmp_path,
            run_spec=run_spec,
            phase_program=program,
            barrier_name="after_warmup",
            coordinate=_coordinate(step=4),
            slots=slots,
            completed_training_batches=4,
        )


def test_v5_declared_path_migrates_untyped_array_to_batch_history(tmp_path: Path) -> None:
    run_spec = _run_spec(minimax=True)
    program = run_spec.worker_execution.method_contract.phase_program
    slots = _minimax_slots()
    slots["controller"] = {"loss": jnp.arange(4, dtype=jnp.float32)}
    result = write_checkpoint_transaction(
        tmp_path,
        run_spec=run_spec,
        phase_program=program,
        barrier_name="after_warmup",
        coordinate=_coordinate(step=4),
        slots=slots,
        completed_training_batches=4,
    )
    payload = json.loads(result.manifest_path.read_text())
    payload["schema_version"] = TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION_V5
    payload["metadata"]["checkpoint_continuation"] = {
        "schema_version": "feedbax.spec.training_checkpoint_continuation.v1",
        "source_completed_batches": 4,
        "target_total_batches": 4,
        "batch_indexed_leaves": [{"slot": "controller", "tree_path": "/loss"}],
    }
    _rewrite_manifest_and_latest(result, payload)
    expected = _minimax_slots()
    expected["controller"] = {"loss": BatchHistory(jnp.arange(4, dtype=jnp.float32))}

    loaded = load_latest_checkpoint(
        tmp_path,
        expected_run_spec=run_spec,
        expected_phase_program=program,
        expected_slots=expected,
    )

    assert isinstance(loaded.slots["controller"]["loss"], BatchHistory)


def test_v5_declared_path_migration_rejects_missing_path(tmp_path: Path) -> None:
    run_spec = _run_spec(minimax=True)
    program = run_spec.worker_execution.method_contract.phase_program
    result = write_checkpoint_transaction(
        tmp_path,
        run_spec=run_spec,
        phase_program=program,
        barrier_name="after_warmup",
        coordinate=_coordinate(step=4),
        slots=_minimax_slots(),
        completed_training_batches=4,
    )
    payload = json.loads(result.manifest_path.read_text())
    payload["schema_version"] = TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION_V5
    payload["metadata"]["checkpoint_continuation"] = {
        "schema_version": "feedbax.spec.training_checkpoint_continuation.v1",
        "source_completed_batches": 4,
        "target_total_batches": 4,
        "batch_indexed_leaves": [{"slot": "controller", "tree_path": "/missing"}],
    }
    _rewrite_manifest_and_latest(result, payload)

    with pytest.raises(CheckpointCompatibilityError, match="migration path is missing"):
        load_latest_checkpoint(
            tmp_path,
            expected_run_spec=run_spec,
            expected_phase_program=program,
            expected_slots=_minimax_slots(),
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


def test_segment_lineage_reader_fails_closed_on_duplicate_cyclic_parent_chain(
    tmp_path: Path,
) -> None:
    run_spec = _run_spec(minimax=True)
    program = run_spec.worker_execution.method_contract.phase_program
    slots = _minimax_slots()
    slots["controller"] = BatchHistory(jnp.arange(2), batch_axis=0)
    result = write_checkpoint_transaction(
        tmp_path,
        run_spec=run_spec,
        phase_program=program,
        barrier_name="after_warmup",
        coordinate=_coordinate(step=2),
        slots=slots,
        completed_training_batches=2,
    )
    payload = json.loads(result.manifest_path.read_text())
    payload["segment_lineage"].update(
        parent_transaction_id=result.manifest.transaction_id,
        start_batch=2,
    )
    _rewrite_manifest_and_latest(result, payload)

    with pytest.raises(CheckpointIntegrityError, match="duplicate/cycle"):
        concatenate_checkpoint_histories(tmp_path, parent_roots={})



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


def test_continuation_applied_marker_absent_or_false_requires_allocation() -> None:
    request = CheckpointContinuationRequest(
        source_completed_batches=12_000,
        additional_batches=4_500,
    )

    for marker in (None, False):
        metadata = {} if marker is None else {"checkpoint_continuation_applied": marker}
        manifest = type("Manifest", (), {"metadata": metadata})()
        assert custody_module._continuation_was_applied(manifest, request) is False


@pytest.mark.parametrize("marker", ["true", 1, 0])
def test_continuation_applied_marker_rejects_non_boolean_true(marker: object) -> None:
    request = CheckpointContinuationRequest(
        source_completed_batches=12_000,
        additional_batches=4_500,
    )
    manifest = type(
        "Manifest",
        (),
        {
            "metadata": {
                "checkpoint_continuation_applied": marker,
                "checkpoint_continuation": request.model_dump(
                    mode="json",
                    exclude_none=True,
                ),
            }
        },
    )()

    with pytest.raises(CheckpointCompatibilityError, match="must be boolean true"):
        custody_module._continuation_was_applied(manifest, request)


@pytest.mark.parametrize(
    ("recorded", "error"),
    [
        (None, "recorded request is missing"),
        ({"schema_version": "invalid"}, "recorded request is invalid"),
        (
            CheckpointContinuationRequest(
                source_completed_batches=12_000,
                additional_batches=4_499,
            ).model_dump(mode="json", exclude_none=True),
            "does not match the already-applied fork contract",
        ),
    ],
)
def test_continuation_applied_marker_rejects_missing_invalid_or_mismatched_request(
    recorded: object,
    error: str,
) -> None:
    request = CheckpointContinuationRequest(
        source_completed_batches=12_000,
        additional_batches=4_500,
    )
    metadata = {"checkpoint_continuation_applied": True}
    if recorded is not None:
        metadata["checkpoint_continuation"] = recorded
    manifest = type("Manifest", (), {"metadata": metadata})()

    with pytest.raises(CheckpointCompatibilityError, match=error):
        custody_module._continuation_was_applied(manifest, request)


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


def test_public_checkpoint_custody_ref_resolver_round_trip_all_and_selected(
    tmp_path: Path,
) -> None:
    from feedbax.training import (
        CheckpointReferenceResolutionError as PublicResolutionError,
        ResolvedCheckpointTransaction,
        resolve_checkpoint_custody_ref as public_resolver,
    )

    result = _write_resolver_checkpoint(tmp_path)
    ref = _resolver_parent_ref(result)

    resolved = public_resolver(ref, allowed_root=tmp_path)
    selected = public_resolver(
        ref,
        allowed_root=tmp_path,
        slot_names=["rng", "controller"],
    )

    assert PublicResolutionError is CheckpointReferenceResolutionError
    assert isinstance(resolved, ResolvedCheckpointTransaction)
    assert resolved.parent_ref == ref
    assert resolved.manifest_sha256 == result.latest_pointer.manifest_sha256
    assert resolved.manifest == result.manifest
    assert tuple(resolved.slots) == tuple(slot.slot for slot in result.manifest.slots)
    assert resolved.slots["controller"].tolist() == [1.0, 2.0]
    assert tuple(selected.slots) == ("controller", "rng")
    assert selected.slots["rng"].dtype == jnp.uint32
    assert resolved.migration_records == ()


def test_checkpoint_custody_ref_resolver_returns_immutable_lineage_snapshots(
    tmp_path: Path,
) -> None:
    result = _write_resolver_checkpoint(tmp_path)
    ref = _resolver_parent_ref(result)
    expected_sha256 = ref.metadata["manifest_sha256"]

    resolved = resolve_checkpoint_custody_ref(ref, allowed_root=tmp_path)
    ref.metadata["manifest_sha256"] = "0" * 64

    assert resolved.parent_ref.metadata["manifest_sha256"] == expected_sha256
    assert resolved.manifest.transaction_id == result.manifest.transaction_id
    assert isinstance(resolved.manifest, custody_module.CheckpointTransactionManifest)
    assert resolved.manifest.model_dump(mode="json")["transaction_id"] == (
        result.manifest.transaction_id
    )
    with pytest.raises((TypeError, ValueError), match="frozen|immutable"):
        resolved.manifest.transaction_id = "mutated"
    with pytest.raises((TypeError, ValueError), match="frozen|immutable"):
        resolved.manifest.slots[0].content_digest.slot_root_sha256 = "0" * 64
    with pytest.raises(TypeError, match="immutable"):
        resolved.manifest.slots.append(resolved.manifest.slots[0])
    assert resolved.manifest.transaction_id == result.manifest.transaction_id


@pytest.mark.parametrize("absolute_uri", ["path", "file"])
def test_checkpoint_custody_ref_resolver_rejects_absolute_uri(
    tmp_path: Path,
    absolute_uri: str,
) -> None:
    result = _write_resolver_checkpoint(tmp_path)
    uri = (
        str(result.manifest_path)
        if absolute_uri == "path"
        else result.manifest_path.as_uri()
    )
    ref = _resolver_parent_ref(result, uri=uri)

    with pytest.raises(CheckpointReferenceResolutionError, match="root-relative"):
        resolve_checkpoint_custody_ref(
            ref,
            allowed_root=tmp_path,
            slot_names=["controller"],
        )


@pytest.mark.parametrize(
    "uri",
    [
        "transactions//manifest.json",
        "transactions/./manifest.json",
        "transactions/%2e/manifest.json",
        "transactions/%2e%2e/manifest.json",
        "transactions/%2fmanifest.json",
        "transactions/%00/manifest.json",
        "transactions/%5cmanifest.json",
    ],
)
def test_checkpoint_custody_ref_resolver_rejects_malformed_relative_uri(
    tmp_path: Path,
    uri: str,
) -> None:
    result = _write_resolver_checkpoint(tmp_path)

    with pytest.raises(CheckpointReferenceResolutionError, match="ParentRef uri|escapes"):
        resolve_checkpoint_custody_ref(
            _resolver_parent_ref(result, uri=uri),
            allowed_root=tmp_path,
        )


def test_checkpoint_custody_ref_resolver_rejects_absolute_slot_path(
    tmp_path: Path,
) -> None:
    result = _write_resolver_checkpoint(tmp_path)
    payload = json.loads(result.manifest_path.read_text())
    slot = payload["slots"][0]
    slot["relative_path"] = str(
        _manifest_blob_path(result.manifest_path, slot["relative_path"])
    )
    _write_json(result.manifest_path, payload)

    with pytest.raises(CheckpointReferenceResolutionError, match="must be relative"):
        resolve_checkpoint_custody_ref(
            _resolver_parent_ref(result),
            allowed_root=tmp_path,
            slot_names=[slot["slot"]],
        )


@pytest.mark.parametrize(
    ("target", "field", "message"),
    [
        ("content", "schema_id", "content integrity schema_id"),
        ("content", "schema_version", "content integrity schema_version"),
        ("abi", "schema_id", "structural ABI schema_id"),
        ("abi", "schema_version", "structural ABI schema_version"),
        ("abi", "fingerprint_algorithm_version", "structural ABI algorithm"),
    ],
)
def test_checkpoint_custody_ref_resolver_rejects_nested_schema_or_algorithm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    field: str,
    message: str,
) -> None:
    result = _write_resolver_checkpoint(tmp_path)
    payload = json.loads(result.manifest_path.read_text())
    record = (
        payload["content_integrity_digest"]
        if target == "content"
        else payload["slots"][0]["structural_abi_fingerprint"]
    )
    record[field] = f"feedbax.invalid.{field}.v999"
    _write_json(result.manifest_path, payload)

    def unexpected_decode(_data: bytes) -> object:
        raise AssertionError("nested schema gate must run before pickle decode")

    monkeypatch.setattr(custody_module.pickle, "loads", unexpected_decode)

    with pytest.raises(CheckpointReferenceResolutionError, match=message):
        resolve_checkpoint_custody_ref(
            _resolver_parent_ref(result),
            allowed_root=tmp_path,
            slot_names=[payload["slots"][0]["slot"]],
        )


def test_checkpoint_custody_ref_resolver_wraps_post_unpickle_runtime_error(
    tmp_path: Path,
) -> None:
    slots = _minimax_slots()
    slots["controller"] = _RuntimeErrorAfterRoundTrip()
    result = _write_resolver_checkpoint(tmp_path, slots=slots)

    with pytest.raises(
        CheckpointReferenceResolutionError,
        match="post-unpickle leaf verification failed",
    ) as exc_info:
        resolve_checkpoint_custody_ref(
            _resolver_parent_ref(result),
            allowed_root=tmp_path,
            slot_names=["controller"],
        )

    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_checkpoint_custody_ref_resolver_rejects_decoded_structural_abi_mismatch(
    tmp_path: Path,
) -> None:
    result = _write_resolver_checkpoint(tmp_path)
    _replace_resolver_slot_blob(
        result,
        "controller",
        jnp.array([[1.0, 2.0]]),
    )

    with pytest.raises(CheckpointReferenceResolutionError, match="structural ABI fingerprint"):
        resolve_checkpoint_custody_ref(
            _resolver_parent_ref(result),
            allowed_root=tmp_path,
            slot_names=["controller"],
        )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"kind": "CheckpointManifest"}, "kind"),
        ({"role": "resume_parent"}, "role"),
        ({"id": "wrong-transaction"}, "transaction_id"),
        ({"metadata": {"manifest_sha256": "not-a-sha"}}, "manifest_sha256"),
    ],
)
def test_checkpoint_custody_ref_resolver_rejects_wrong_reference_identity(
    tmp_path: Path,
    updates: dict[str, object],
    message: str,
) -> None:
    result = _write_resolver_checkpoint(tmp_path)
    ref = _resolver_parent_ref(result, **updates)

    with pytest.raises(CheckpointReferenceResolutionError, match=message):
        resolve_checkpoint_custody_ref(ref, allowed_root=tmp_path)


def test_checkpoint_custody_ref_resolver_authenticates_raw_manifest_before_parse(
    tmp_path: Path,
) -> None:
    result = _write_resolver_checkpoint(tmp_path)
    ref = _resolver_parent_ref(result)
    result.manifest_path.write_bytes(result.manifest_path.read_bytes() + b"\n")

    with pytest.raises(CheckpointReferenceResolutionError, match="raw manifest bytes"):
        resolve_checkpoint_custody_ref(ref, allowed_root=tmp_path)


@pytest.mark.parametrize(
    "slot_names",
    ["controller", [], ["controller", "controller"], [""], [1]],
)
def test_checkpoint_custody_ref_resolver_rejects_invalid_slot_selection(
    tmp_path: Path,
    slot_names: object,
) -> None:
    result = _write_resolver_checkpoint(tmp_path)

    with pytest.raises(CheckpointReferenceResolutionError, match="slot_names"):
        resolve_checkpoint_custody_ref(
            _resolver_parent_ref(result),
            allowed_root=tmp_path,
            slot_names=slot_names,  # type: ignore[arg-type]
        )


def test_checkpoint_custody_ref_resolver_rejects_missing_requested_slot(
    tmp_path: Path,
) -> None:
    result = _write_resolver_checkpoint(tmp_path)

    with pytest.raises(CheckpointReferenceResolutionError, match="slots are missing"):
        resolve_checkpoint_custody_ref(
            _resolver_parent_ref(result),
            allowed_root=tmp_path,
            slot_names=["missing"],
        )


def test_checkpoint_custody_ref_resolver_rejects_manifest_and_slot_path_escape(
    tmp_path: Path,
) -> None:
    result = _write_resolver_checkpoint(tmp_path / "custody")
    outside = tmp_path / "outside.json"
    outside.write_bytes(result.manifest_path.read_bytes())
    escaped_ref = _resolver_parent_ref(
        result,
        uri="../outside.json",
        metadata={"manifest_sha256": _sha256_file(outside)},
    )
    with pytest.raises(CheckpointReferenceResolutionError, match="escapes"):
        resolve_checkpoint_custody_ref(escaped_ref, allowed_root=tmp_path / "custody")

    payload = json.loads(result.manifest_path.read_text())
    slot = payload["slots"][0]
    original_blob = _manifest_blob_path(result.manifest_path, slot["relative_path"])
    escaped_blob = result.root / "escaped.pkl"
    escaped_blob.write_bytes(original_blob.read_bytes())
    slot["relative_path"] = "../../escaped.pkl"
    _write_json(result.manifest_path, payload)
    with pytest.raises(CheckpointReferenceResolutionError, match="escapes"):
        resolve_checkpoint_custody_ref(
            _resolver_parent_ref(result),
            allowed_root=result.root,
            slot_names=[slot["slot"]],
        )

    symlink_result = _write_resolver_checkpoint(tmp_path / "symlink-custody")
    symlink_payload = json.loads(symlink_result.manifest_path.read_text())
    symlink_slot = symlink_payload["slots"][0]
    source_blob = _manifest_blob_path(
        symlink_result.manifest_path,
        symlink_slot["relative_path"],
    )
    outside_blob = tmp_path / "outside.pkl"
    outside_blob.write_bytes(source_blob.read_bytes())
    blob_link = symlink_result.manifest_path.parent / "blob-link.pkl"
    blob_link.symlink_to(outside_blob)
    symlink_slot["relative_path"] = blob_link.name
    _write_json(symlink_result.manifest_path, symlink_payload)
    with pytest.raises(CheckpointReferenceResolutionError, match="escapes"):
        resolve_checkpoint_custody_ref(
            _resolver_parent_ref(symlink_result),
            allowed_root=symlink_result.root,
            slot_names=[symlink_slot["slot"]],
        )


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("transaction_root", "transaction content root"),
        ("slot_root", "content root"),
        ("content_correspondence", "content digests differ"),
        ("structural_abi", "structural ABI fingerprint"),
    ],
)
def test_checkpoint_custody_ref_resolver_rejects_authenticated_manifest_tamper(
    tmp_path: Path,
    tamper: str,
    message: str,
) -> None:
    result = _write_resolver_checkpoint(tmp_path)
    payload = json.loads(result.manifest_path.read_text())
    if tamper == "transaction_root":
        payload["content_integrity_digest"]["transaction_root_sha256"] = "0" * 64
    elif tamper == "slot_root":
        payload["slots"][0]["content_digest"]["slot_root_sha256"] = "0" * 64
        payload["content_integrity_digest"]["slots"][0]["slot_root_sha256"] = "0" * 64
    elif tamper == "content_correspondence":
        payload["slots"][0]["content_digest"]["blob_size_bytes"] += 1
    else:
        payload["slots"][0]["structural_abi_fingerprint"]["fingerprint_sha256"] = "0" * 64
    _write_json(result.manifest_path, payload)

    with pytest.raises(CheckpointReferenceResolutionError, match=message):
        resolve_checkpoint_custody_ref(
            _resolver_parent_ref(result),
            allowed_root=tmp_path,
        )


def test_checkpoint_custody_ref_resolver_rejects_missing_and_modified_blob(
    tmp_path: Path,
) -> None:
    missing_result = _write_resolver_checkpoint(tmp_path / "missing")
    missing_blob = _slot_blob_path(missing_result.manifest_path, "controller")
    missing_blob.unlink()
    with pytest.raises(CheckpointReferenceResolutionError, match="blob is missing"):
        resolve_checkpoint_custody_ref(
            _resolver_parent_ref(missing_result),
            allowed_root=missing_result.root,
            slot_names=["controller"],
        )

    modified_result = _write_resolver_checkpoint(tmp_path / "modified")
    modified_blob = _slot_blob_path(modified_result.manifest_path, "controller")
    modified_bytes = bytearray(modified_blob.read_bytes())
    modified_bytes[-1] ^= 1
    modified_blob.write_bytes(modified_bytes)
    with pytest.raises(CheckpointReferenceResolutionError, match="hash mismatch"):
        resolve_checkpoint_custody_ref(
            _resolver_parent_ref(modified_result),
            allowed_root=modified_result.root,
            slot_names=["controller"],
        )


def test_checkpoint_custody_ref_resolver_rejects_decoded_content_tamper(
    tmp_path: Path,
) -> None:
    result = _write_resolver_checkpoint(tmp_path)
    payload = json.loads(result.manifest_path.read_text())
    slot = payload["slots"][0]
    digest = payload["content_integrity_digest"]["slots"][0]
    blob_path = _manifest_blob_path(result.manifest_path, slot["relative_path"])
    altered_blob = pickle.dumps(jnp.array([9.0, 8.0]), protocol=pickle.HIGHEST_PROTOCOL)
    altered_sha256 = hashlib.sha256(altered_blob).hexdigest()
    blob_path.write_bytes(altered_blob)
    for record in (slot, slot["content_digest"], digest):
        if "sha256" in record:
            record["sha256"] = altered_sha256
        if "blob_sha256" in record:
            record["blob_sha256"] = altered_sha256
        if "size_bytes" in record:
            record["size_bytes"] = len(altered_blob)
        if "blob_size_bytes" in record:
            record["blob_size_bytes"] = len(altered_blob)
    slot_root = custody_module._slot_root_sha256(
        slot["slot"],
        altered_sha256,
        [
            custody_module.SlotLeafContentDigest.model_validate(item)
            for item in digest["leaf_hashes"]
        ],
    )
    slot["content_digest"]["slot_root_sha256"] = slot_root
    digest["slot_root_sha256"] = slot_root
    payload["content_integrity_digest"]["transaction_root_sha256"] = (
        custody_module._transaction_root_sha256(
            [
                custody_module.SlotContentDigest.model_validate(item)
                for item in payload["content_integrity_digest"]["slots"]
            ]
        )
    )
    _write_json(result.manifest_path, payload)

    with pytest.raises(CheckpointReferenceResolutionError, match="leaf content digest"):
        resolve_checkpoint_custody_ref(
            _resolver_parent_ref(result),
            allowed_root=tmp_path,
            slot_names=[slot["slot"]],
        )


def test_checkpoint_custody_ref_resolver_uses_registered_manifest_migration(
    tmp_path: Path,
) -> None:
    result = _write_resolver_checkpoint(tmp_path)
    payload = json.loads(result.manifest_path.read_text())
    payload["schema_version"] = TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION_V6
    payload.pop("segment_lineage")
    _write_json(result.manifest_path, payload)

    resolved = resolve_checkpoint_custody_ref(
        _resolver_parent_ref(result),
        allowed_root=tmp_path,
        slot_names=["controller"],
    )

    assert resolved.manifest.schema_version == TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION
    assert resolved.manifest.segment_lineage.start_batch == 0
    assert [record.migration_id for record in resolved.migration_records] == [
        "training-checkpoint-transaction-v6-to-v7-segment-lineage",
        "training-checkpoint-transaction-v7-to-v8-mapped-axes",
    ]


def test_checkpoint_custody_ref_resolver_v5_migration_keeps_raw_slot_tree(
    tmp_path: Path,
) -> None:
    slots = _minimax_slots()
    slots["controller"] = {"loss": jnp.arange(4, dtype=jnp.float32)}
    result = _write_resolver_checkpoint(tmp_path, slots=slots)
    payload = json.loads(result.manifest_path.read_text())
    payload["schema_version"] = TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION_V5
    payload.pop("segment_lineage")
    payload["metadata"]["checkpoint_continuation"] = {
        "schema_version": "feedbax.spec.training_checkpoint_continuation.v1",
        "source_completed_batches": 4,
        "target_total_batches": 4,
        "batch_indexed_leaves": [{"slot": "controller", "tree_path": "/loss"}],
    }
    _write_json(result.manifest_path, payload)

    resolved = resolve_checkpoint_custody_ref(
        _resolver_parent_ref(result),
        allowed_root=tmp_path,
        slot_names=["controller"],
    )

    assert not isinstance(resolved.slots["controller"]["loss"], BatchHistory)
    assert resolved.slots["controller"]["loss"].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert [record.migration_id for record in resolved.migration_records] == [
        "training-checkpoint-transaction-v5-to-v6-batch-history",
        "training-checkpoint-transaction-v6-to-v7-segment-lineage",
        "training-checkpoint-transaction-v7-to-v8-mapped-axes",
    ]


def test_checkpoint_custody_ref_resolver_rejects_unsupported_manifest_migration(
    tmp_path: Path,
) -> None:
    result = _write_resolver_checkpoint(tmp_path)
    payload = json.loads(result.manifest_path.read_text())
    payload["schema_version"] = "feedbax.manifest.training_checkpoint_transaction.v99"
    _write_json(result.manifest_path, payload)

    with pytest.raises(CheckpointReferenceResolutionError, match="source_version"):
        resolve_checkpoint_custody_ref(_resolver_parent_ref(result), allowed_root=tmp_path)


def test_checkpoint_custody_ref_resolver_rejects_invalid_supported_migration(
    tmp_path: Path,
) -> None:
    result = _write_resolver_checkpoint(tmp_path)
    payload = json.loads(result.manifest_path.read_text())
    payload["schema_version"] = TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION_V6
    payload.pop("segment_lineage")
    payload["completed_training_batches"] = "invalid"
    _write_json(result.manifest_path, payload)

    with pytest.raises(CheckpointReferenceResolutionError, match="non-negative"):
        resolve_checkpoint_custody_ref(_resolver_parent_ref(result), allowed_root=tmp_path)
