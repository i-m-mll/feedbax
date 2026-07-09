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
    TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION,
    TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION_V1,
    TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION_V2,
    TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION_V3,
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
    assert (
        integrity.structural_abi_fingerprint
        == custody_module.structural_abi_fingerprint(value)
    )


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
    ]


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
    binding["canonical_projection"]["training_run_spec"][
        "schema_version"
    ] = "feedbax.spec.training_run.v1"
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
    assert migrated_binding["canonical_projection"]["training_run_spec"][
        "schema_version"
    ] == run_spec.schema_version
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
    assert "not the primary training-batch" in migrated.payload[
        "completed_coordinate_semantics"
    ]


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

    assert result.manifest.completed_coordinate.global_step == 12009
    assert result.manifest.completed_training_batches == 16500
    assert result.latest_pointer.completed_training_batches == 16500
    assert latest["completed_training_batches"] == 16500
    assert latest["completed_coordinate"]["global_step"] == 12009
    assert "not the primary training-batch" in latest["completed_coordinate_semantics"]


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
    assert {
        mode for slot, mode in forked.slot_transfer_modes.items() if slot != "controller"
    } == {"hardlink"}
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
    controller_provenance = {
        slot.slot: slot for slot in forked.manifest.fork_provenance.slots
    }["controller"]
    source_controller_slot = {
        slot.slot: slot for slot in source.manifest.slots
    }["controller"]
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
    binding["canonical_projection"]["training_run_spec"][
        "schema_version"
    ] = "feedbax.spec.training_run.v1"
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
    assert loaded.manifest.run_contract_binding.canonical_projection["training_run_spec"][
        "on_nan"
    ] == "raise"


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
    assert (
        "path=/ field=dtype recorded=\"float64\" actual=\"float32\""
        in message
    )
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
                (root / "checkpoint_000001" / "optimizer_state.eqx").write_bytes(
                    b"optimizer"
                ),
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
    assert loaded.manifest.run_contract_binding.metadata["projection_status"] == (
        "legacy_absent"
    )

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
