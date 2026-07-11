"""Atomic training checkpoint custody helpers."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import platform
import shutil
import tempfile
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree as jt
import numpy as np
from pydantic import BaseModel, ValidationError

from feedbax.contracts.checkpoints import (
    BatchIndexedCheckpointLeafSpec,
    CheckpointContinuationRequest,
    CheckpointLatestPointer,
    CheckpointLineageRef,
    CheckpointForkProvenance,
    CheckpointForkSlotProvenance,
    CheckpointForkSourceRecord,
    CheckpointForkTransformRecord,
    CheckpointResumeResult,
    CheckpointProvenanceNotice,
    CheckpointSlotBlobRef,
    CheckpointTransactionManifest,
    ContentIntegrityDigest,
    PopulationIdentityRecord,
    RunContractBinding,
    SerializerVersionRecord,
    SlotContentDigest,
    SlotLeafContentDigest,
    SlotLeafFingerprint,
    StructuralAbiFingerprint,
)
from feedbax.contracts.manifest import canonical_json_bytes, feedbax_version, sha256_bytes
from feedbax.contracts.training import TRAINING_RUN_SPEC_SCHEMA_VERSION_V1, TrainingRunSpec
from feedbax.contracts.worker import (
    CheckpointBarrierSpec,
    CheckpointSlotSpec,
    PhaseProgramSpec,
    ProgressCoordinate,
    StateSlotSpec,
    derive_consistency_predicate,
)


LATEST_POINTER_NAME = "latest.json"
TRANSACTIONS_DIR_NAME = "transactions"
MANIFEST_NAME = "manifest.json"
_LOGGER = logging.getLogger(__name__)
_STRUCTURAL_ABI_DIFF_FIELDS = (
    "dtype",
    "shape",
    "weak_type",
    "leaf_type",
    "static_repr_sha256",
)
LEGACY_CHECKPOINT_ADOPTION_ENTRYPOINT = (
    "feedbax.training.legacy_checkpoint_adoption.adopt_legacy_checkpoint"
)
LEGACY_CHECKPOINT_ADOPTION_DOCS = "docs/structure.md#legacy-checkpoint-adoption"


class CheckpointCustodyError(ValueError):
    """Base class for checkpoint custody failures."""


class CheckpointIntegrityError(CheckpointCustodyError):
    """Raised when checkpoint bytes or manifests fail integrity validation."""


class CheckpointCompatibilityError(CheckpointCustodyError):
    """Raised when a checkpoint is structurally incompatible with resume templates."""


class CheckpointContractBindingError(CheckpointCustodyError):
    """Raised when a checkpoint belongs to a different run contract."""


class CheckpointConsistencyError(CheckpointCustodyError):
    """Raised when slot coordinates violate the method-declared consistency predicate."""


@dataclass(frozen=True)
class CheckpointWriteResult:
    """Paths and manifest returned by a successful checkpoint write."""

    root: Path
    transaction_dir: Path
    manifest_path: Path
    latest_pointer_path: Path
    manifest: CheckpointTransactionManifest
    latest_pointer: CheckpointLatestPointer


ResumeSlotTransform = Callable[[Mapping[str, Any]], Mapping[str, Any]]
CheckpointBlobLinkStrategy = Callable[[Path, Path], str]


@dataclass(frozen=True)
class CheckpointForkResult:
    """Paths and manifest returned by a successful checkpoint fork."""

    root: Path
    transaction_dir: Path
    manifest_path: Path
    latest_pointer_path: Path
    manifest: CheckpointTransactionManifest
    latest_pointer: CheckpointLatestPointer
    slot_transfer_modes: Mapping[str, str]
    source_provenance_notices: tuple[CheckpointProvenanceNotice, ...] = ()


@dataclass(frozen=True)
class DetectedLegacyCheckpointLayout:
    """Recognized pre-custody checkpoint layout evidence."""

    layout_id: str
    name: str
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class _LegacyCheckpointLayout:
    layout_id: str
    name: str
    detect: Callable[[Path], tuple[str, ...]]


@dataclass(frozen=True)
class _LoadedCheckpointTransaction:
    root: Path
    latest_pointer: CheckpointLatestPointer
    manifest_path: Path
    manifest: CheckpointTransactionManifest
    slots: Mapping[str, Any]
    provenance_notices: tuple[CheckpointProvenanceNotice, ...]


@dataclass(frozen=True)
class _StructuralAbiLeafDiff:
    path: str
    field: str
    recorded: Any
    actual: Any


@dataclass(frozen=True)
class _SlotIntegrityRecords:
    leaf_digests: list[SlotLeafContentDigest]
    structural_abi_fingerprint: StructuralAbiFingerprint


def checkpoint_barrier(program: PhaseProgramSpec, barrier_name: str) -> CheckpointBarrierSpec:
    """Return a checkpoint barrier from a phase program."""
    for barrier in program.checkpoint_barriers:
        if barrier.name == barrier_name:
            return barrier
    known = [barrier.name for barrier in program.checkpoint_barriers]
    raise CheckpointCustodyError(
        f"Unknown checkpoint barrier {barrier_name!r}; known barriers={known!r}"
    )


def checkpoint_slot_specs(
    program: PhaseProgramSpec,
    barrier_name: str,
) -> tuple[CheckpointSlotSpec, ...]:
    """Return the slot specs derived from a phase-program barrier."""
    return tuple(checkpoint_barrier(program, barrier_name).slots)


def checkpoint_slot_names(program: PhaseProgramSpec, barrier_name: str) -> tuple[str, ...]:
    """Return checkpoint slot names derived from a phase-program barrier."""
    return tuple(spec.slot for spec in checkpoint_slot_specs(program, barrier_name))


def _completed_training_batches(
    explicit: int | None,
    metadata: Mapping[str, Any],
    *,
    default: int | None = None,
) -> int | None:
    if explicit is not None:
        return int(explicit)
    for key in (
        "completed_training_batches",
        "completed_batches",
        "completed_batch",
        "n_batches",
        "batch",
    ):
        value = metadata.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return default


def write_checkpoint_transaction(
    root: str | Path,
    *,
    run_spec: TrainingRunSpec,
    phase_program: PhaseProgramSpec,
    barrier_name: str,
    coordinate: ProgressCoordinate,
    slots: Mapping[str, Any],
    status: str = "partial",
    slot_coordinates: Mapping[str, ProgressCoordinate] | None = None,
    population_member_ids: Mapping[str, Sequence[str]] | None = None,
    history_availability: Mapping[str, bool] | None = None,
    parent_lineage: Sequence[CheckpointLineageRef] | None = None,
    completed_training_batches: int | None = None,
    metadata: Mapping[str, Any] | None = None,
    publish_latest: bool = True,
) -> CheckpointWriteResult:
    """Write one atomic multi-slot checkpoint transaction and publish latest.

    Slot names are derived from ``phase_program`` and ``barrier_name``. The
    caller supplies the engine-state mapping, but not filenames.
    """
    if status not in {"partial", "final"}:
        raise CheckpointCustodyError("checkpoint status must be 'partial' or 'final'")
    barrier = checkpoint_barrier(phase_program, barrier_name)
    slot_specs = tuple(barrier.slots)
    _validate_required_slots(slot_specs, slots)
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    transactions_root = root_path / TRANSACTIONS_DIR_NAME
    transactions_root.mkdir(exist_ok=True)
    transaction_id = f"tx-{uuid.uuid4().hex}"
    tmp_dir = Path(tempfile.mkdtemp(prefix=f".{transaction_id}.", dir=transactions_root))
    final_dir = transactions_root / transaction_id

    try:
        blob_dir = tmp_dir / "blobs"
        blob_dir.mkdir()
        slot_roles = _slot_roles(run_spec)
        population_slots = _population_slot_names(run_spec)
        slot_records: list[CheckpointSlotBlobRef] = []
        slot_digests: list[SlotContentDigest] = []
        for spec in slot_specs:
            if spec.slot not in slots and not spec.required:
                continue
            value = slots[spec.slot]
            blob_bytes = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            blob_sha256 = sha256_bytes(blob_bytes)
            blob_path = blob_dir / f"{spec.slot}-{blob_sha256}.pkl"
            _write_bytes_atomic(blob_path, blob_bytes)
            integrity = _slot_integrity_records(value)
            content_digest = SlotContentDigest(
                slot=spec.slot,
                blob_sha256=blob_sha256,
                blob_size_bytes=len(blob_bytes),
                leaf_hashes=integrity.leaf_digests,
                slot_root_sha256=_slot_root_sha256(
                    spec.slot,
                    blob_sha256,
                    integrity.leaf_digests,
                ),
            )
            population = _population_record(
                spec.slot,
                value,
                population_member_ids=population_member_ids or {},
                population_slots=population_slots,
            )
            slot_record = CheckpointSlotBlobRef(
                slot=spec.slot,
                role=slot_roles.get(spec.slot, "auxiliary"),
                required=spec.required,
                relative_path=str(blob_path.relative_to(tmp_dir)),
                sha256=blob_sha256,
                size_bytes=len(blob_bytes),
                coordinate=(slot_coordinates or {}).get(spec.slot, coordinate),
                structural_abi_fingerprint=integrity.structural_abi_fingerprint,
                content_digest=content_digest,
                population=population,
                metadata=dict(spec.metadata),
            )
            slot_records.append(slot_record)
            slot_digests.append(content_digest)

        _validate_slot_coordinate_consistency(
            barrier=barrier,
            completed_coordinate=coordinate,
            slot_records=slot_records,
        )
        transaction_root = _transaction_root_sha256(slot_digests)
        manifest_metadata = {"phase": barrier.phase}
        manifest_metadata.update(dict(metadata or {}))
        completed_batches = _completed_training_batches(
            completed_training_batches,
            manifest_metadata,
        )
        manifest = CheckpointTransactionManifest(
            transaction_id=transaction_id,
            run_id=coordinate.run_id,
            status=status,  # type: ignore[arg-type]
            barrier=barrier.name,
            completed_coordinate=coordinate,
            completed_training_batches=completed_batches,
            consistency_predicate=derive_consistency_predicate(phase_program),
            run_contract_binding=run_contract_binding(run_spec, phase_program),
            slots=slot_records,
            content_integrity_digest=ContentIntegrityDigest(
                slots=slot_digests,
                transaction_root_sha256=transaction_root,
            ),
            history_availability=dict(history_availability or {}),
            parent_lineage=list(parent_lineage or ()),
            metadata=manifest_metadata,
        )
        manifest_path = tmp_dir / MANIFEST_NAME
        _write_json_atomic(manifest_path, manifest.model_dump(mode="json", exclude_none=True))
        manifest_sha256 = _sha256_file(manifest_path)

        os.replace(tmp_dir, final_dir)
        manifest_path = final_dir / MANIFEST_NAME
        latest_pointer = CheckpointLatestPointer(
            run_id=coordinate.run_id,
            transaction_id=transaction_id,
            manifest_relative_path=str(manifest_path.relative_to(root_path)),
            manifest_sha256=manifest_sha256,
            transaction_root_sha256=transaction_root,
            completed_coordinate=coordinate,
            completed_training_batches=completed_batches,
        )
        latest_path = root_path / LATEST_POINTER_NAME
        if publish_latest:
            _write_json_atomic(
                latest_path,
                latest_pointer.model_dump(mode="json", exclude_none=True),
            )
        return CheckpointWriteResult(
            root=root_path,
            transaction_dir=final_dir,
            manifest_path=manifest_path,
            latest_pointer_path=latest_path,
            manifest=manifest,
            latest_pointer=latest_pointer,
        )
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def load_latest_checkpoint(
    root: str | Path,
    *,
    expected_run_spec: TrainingRunSpec,
    expected_phase_program: PhaseProgramSpec,
    expected_slots: Mapping[str, Any],
    expected_population_member_ids: Mapping[str, Sequence[str]] | None = None,
    resume_slot_transform: ResumeSlotTransform | None = None,
    continuation_request: CheckpointContinuationRequest | Mapping[str, Any] | None = None,
    allow_new_lineage_override: bool = False,
) -> CheckpointResumeResult:
    """Load and validate the latest published transaction before resume."""
    root_path = Path(root)
    latest = _load_latest_pointer(root_path)
    return _load_checkpoint_from_pointer(
        root_path,
        latest,
        expected_run_spec=expected_run_spec,
        expected_phase_program=expected_phase_program,
        expected_slots=expected_slots,
        expected_population_member_ids=expected_population_member_ids,
        resume_slot_transform=resume_slot_transform,
        continuation_request=continuation_request,
        allow_new_lineage_override=allow_new_lineage_override,
    )


def detect_known_legacy_checkpoint_layout(
    root: str | Path,
) -> DetectedLegacyCheckpointLayout | None:
    """Return a recognized pre-custody checkpoint layout for *root*, if any."""
    root_path = Path(root)
    for layout in _KNOWN_LEGACY_CHECKPOINT_LAYOUTS:
        evidence = layout.detect(root_path)
        if evidence:
            return DetectedLegacyCheckpointLayout(
                layout_id=layout.layout_id,
                name=layout.name,
                evidence=evidence,
            )
    return None


def _load_checkpoint_from_pointer(
    root_path: Path,
    latest: CheckpointLatestPointer,
    *,
    expected_run_spec: TrainingRunSpec,
    expected_phase_program: PhaseProgramSpec,
    expected_slots: Mapping[str, Any],
    expected_population_member_ids: Mapping[str, Sequence[str]] | None = None,
    resume_slot_transform: ResumeSlotTransform | None = None,
    continuation_request: CheckpointContinuationRequest | Mapping[str, Any] | None = None,
    allow_new_lineage_override: bool = False,
) -> CheckpointResumeResult:
    """Load and validate a transaction through the same gates as latest resume."""
    manifest_path, manifest = _manifest_from_latest_pointer(root_path, latest)

    _validate_contract_binding(
        manifest,
        expected_run_spec,
        expected_phase_program,
        allow_new_lineage_override=allow_new_lineage_override,
    )
    expected_predicate = derive_consistency_predicate(expected_phase_program)
    if manifest.consistency_predicate != expected_predicate:
        raise CheckpointConsistencyError(
            "checkpoint consistency predicate does not match expected phase program"
        )
    barrier = checkpoint_barrier(expected_phase_program, manifest.barrier)
    _validate_slot_coordinate_consistency(
        barrier=barrier,
        completed_coordinate=manifest.completed_coordinate,
        slot_records=manifest.slots,
    )

    slots_by_name = {slot.slot: slot for slot in manifest.slots}
    _validate_required_slots(tuple(barrier.slots), slots_by_name)
    _validate_expected_slot_set(barrier, expected_slots)
    _validate_population_identities(
        manifest,
        expected_population_member_ids or {},
    )

    loaded_slots: dict[str, Any] = {}
    transaction_dir = manifest_path.parent
    for slot in manifest.slots:
        blob_path = transaction_dir / slot.relative_path
        blob_bytes = _read_blob(slot, blob_path)
        try:
            loaded_slots[slot.slot] = pickle.loads(blob_bytes)
        except Exception as exc:
            raise CheckpointIntegrityError(
                f"checkpoint slot {slot.slot!r} could not be deserialized"
            ) from exc
    provenance_notices, loaded_fingerprints = _validate_manifest_structural_abi(
        manifest,
        loaded_slots,
    )
    loaded_fingerprint_slots = dict(loaded_slots)
    if resume_slot_transform is not None:
        try:
            loaded_slots = dict(resume_slot_transform(loaded_slots))
        except CheckpointCustodyError:
            raise
        except Exception as exc:
            raise CheckpointCompatibilityError("resume_slot_transform failed") from exc
        _validate_required_slots(
            tuple(barrier.slots),
            loaded_slots,
            error_cls=CheckpointCompatibilityError,
        )
        loaded_fingerprints = {
            slot: fingerprint
            for slot, fingerprint in loaded_fingerprints.items()
            if loaded_slots.get(slot) is loaded_fingerprint_slots.get(slot)
        }
    request = _coerce_continuation_request(continuation_request)
    if request is not None:
        loaded_slots = _apply_declared_continuation_request(
            loaded_slots,
            expected_slots=expected_slots,
            manifest=manifest,
            request=request,
        )
        loaded_fingerprints = {
            slot: fingerprint
            for slot, fingerprint in loaded_fingerprints.items()
            if loaded_slots.get(slot) is loaded_fingerprint_slots.get(slot)
        }
    _validate_structural_abi(
        manifest,
        expected_slots,
        loaded_slots,
        loaded_fingerprints=loaded_fingerprints,
    )

    return CheckpointResumeResult(
        manifest=manifest,
        slots=loaded_slots,
        provenance_notices=provenance_notices,
        new_lineage_required=allow_new_lineage_override
        and not _contract_binding_matches(
            manifest.run_contract_binding,
            run_contract_binding(
                expected_run_spec,
                expected_phase_program,
            ),
            expected_run_spec,
            expected_phase_program,
        ),
        previous_transaction_id=(
            manifest.transaction_id if allow_new_lineage_override else None
        ),
    )


def _manifest_from_latest_pointer(
    root_path: Path,
    latest: CheckpointLatestPointer,
) -> tuple[Path, CheckpointTransactionManifest]:
    manifest_path = root_path / latest.manifest_relative_path
    if not manifest_path.is_file():
        raise CheckpointIntegrityError(
            f"latest pointer references missing manifest: {latest.manifest_relative_path}"
        )
    if _sha256_file(manifest_path) != latest.manifest_sha256:
        raise CheckpointIntegrityError("latest pointer manifest hash does not match bytes")
    manifest = _load_transaction_manifest(manifest_path)
    if manifest.transaction_id != latest.transaction_id:
        raise CheckpointIntegrityError("latest pointer transaction_id does not match manifest")
    if manifest.content_integrity_digest.transaction_root_sha256 != (
        latest.transaction_root_sha256
    ):
        raise CheckpointIntegrityError("latest pointer transaction root is stale")
    if manifest.status not in {"partial", "final"}:
        raise CheckpointIntegrityError(f"unsupported checkpoint status {manifest.status!r}")
    return manifest_path, manifest


def fork_checkpoint_transaction(
    source_root: str | Path,
    target_root: str | Path,
    *,
    target_run_spec: TrainingRunSpec,
    target_phase_program: PhaseProgramSpec | None = None,
    expected_slots: Mapping[str, Any] | None = None,
    target_coordinate: ProgressCoordinate | None = None,
    expected_population_member_ids: Mapping[str, Sequence[str]] | None = None,
    slot_transforms: Mapping[str, ResumeSlotTransform] | None = None,
    transform_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    continuation_request: CheckpointContinuationRequest | Mapping[str, Any] | None = None,
    link_strategy: CheckpointBlobLinkStrategy | None = None,
    tool_version: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> CheckpointForkResult:
    """Fork one valid custody checkpoint into a target run contract/root.

    Untransformed slots are hardlinked into the target transaction when possible
    and copied when hardlinking is unavailable. Transformed slots are serialized
    fresh. ``latest.json`` is written only after strict target-bound resume
    validation succeeds.
    """
    source = _load_latest_checkpoint_transaction(source_root)
    phase_program = target_phase_program or (
        target_run_spec.worker_execution.method_contract.phase_program
    )
    barrier = checkpoint_barrier(phase_program, source.manifest.barrier)
    coordinate = target_coordinate or source.manifest.completed_coordinate
    transforms = dict(slot_transforms or {})
    transform_meta = dict(transform_metadata or {})
    loaded_slots = dict(source.slots)
    prepared_slots = _apply_slot_transforms(
        loaded_slots,
        transforms=transforms,
    )
    validation_slots = dict(expected_slots or prepared_slots)
    request = _coerce_continuation_request(continuation_request)
    continuation_transformed_slots: set[str] = set()
    if request is not None:
        before_continuation = dict(prepared_slots)
        prepared_slots = _apply_declared_continuation_request(
            prepared_slots,
            expected_slots=validation_slots,
            manifest=source.manifest,
            request=request,
        )
        continuation_transformed_slots = {
            slot
            for slot in prepared_slots
            if prepared_slots[slot] is not before_continuation.get(slot)
        }
    _validate_required_slots(
        tuple(barrier.slots),
        prepared_slots,
        error_cls=CheckpointCompatibilityError,
    )
    _validate_expected_slot_set(barrier, validation_slots)

    target_root_path = Path(target_root)
    target_root_path.mkdir(parents=True, exist_ok=True)
    transactions_root = target_root_path / TRANSACTIONS_DIR_NAME
    transactions_root.mkdir(exist_ok=True)
    transaction_id = f"tx-{uuid.uuid4().hex}"
    tmp_dir = Path(tempfile.mkdtemp(prefix=f".{transaction_id}.", dir=transactions_root))
    final_dir = transactions_root / transaction_id
    moved_to_final = False

    try:
        blob_dir = tmp_dir / "blobs"
        blob_dir.mkdir()
        slot_roles = _slot_roles(target_run_spec)
        population_slots = _population_slot_names(target_run_spec)
        source_slots_by_name = {slot.slot: slot for slot in source.manifest.slots}
        source_transaction_dir = source.manifest_path.parent
        transfer = link_strategy or _hardlink_first_copy_fallback
        slot_records: list[CheckpointSlotBlobRef] = []
        slot_digests: list[SlotContentDigest] = []
        slot_provenance: list[CheckpointForkSlotProvenance] = []
        transfer_modes: dict[str, str] = {}
        transform_records = {
            slot: _transform_record(slot, transforms[slot], transform_meta.get(slot, {}))
            for slot in transforms
        }
        for slot in continuation_transformed_slots:
            transform_records[slot] = CheckpointForkTransformRecord(
                slot=slot,
                identity="feedbax.training_checkpoint.declared_continuation.v1",
                parameters=request.model_dump(mode="json", exclude_none=True)
                if request is not None
                else {},
            )

        for spec in barrier.slots:
            if spec.slot not in prepared_slots and not spec.required:
                continue
            if spec.slot not in prepared_slots:
                raise CheckpointCompatibilityError(
                    f"target checkpoint slot {spec.slot!r} is missing after transforms"
                )
            if spec.slot in transforms or spec.slot in continuation_transformed_slots:
                source_slot = source_slots_by_name.get(spec.slot)
                slot_record, content_digest = _write_fresh_slot_blob(
                    spec,
                    prepared_slots[spec.slot],
                    blob_dir=blob_dir,
                    transaction_dir=tmp_dir,
                    coordinate=coordinate,
                    slot_roles=slot_roles,
                    population_slots=population_slots,
                    population_member_ids=expected_population_member_ids or {},
                )
                provenance = CheckpointForkSlotProvenance(
                    slot=spec.slot,
                    source_sha256=source_slot.sha256 if source_slot is not None else None,
                    target_sha256=slot_record.sha256,
                    source_relative_path=(
                        source_slot.relative_path if source_slot is not None else None
                    ),
                    target_relative_path=slot_record.relative_path,
                    transfer_mode="serialized",
                    transform=transform_records[spec.slot],
                )
            else:
                source_slot = source_slots_by_name.get(spec.slot)
                if source_slot is None:
                    raise CheckpointCompatibilityError(
                        f"source checkpoint does not contain target slot {spec.slot!r}"
                    )
                source_blob_path = source_transaction_dir / source_slot.relative_path
                _verify_source_blob_before_transfer(source_slot, source_blob_path)
                target_blob_path = blob_dir / Path(source_slot.relative_path).name
                mode = transfer(source_blob_path, target_blob_path)
                if target_blob_path.is_symlink():
                    raise CheckpointIntegrityError(
                        f"checkpoint fork produced symlink for slot {spec.slot!r}"
                    )
                if mode not in {"hardlink", "copy"}:
                    raise CheckpointIntegrityError(
                        "checkpoint fork link strategy must return 'hardlink' or 'copy'; "
                        f"got {mode!r} for slot {spec.slot!r}"
                    )
                slot_record = CheckpointSlotBlobRef(
                    slot=spec.slot,
                    role=slot_roles.get(spec.slot, "auxiliary"),
                    required=spec.required,
                    relative_path=str(target_blob_path.relative_to(tmp_dir)),
                    sha256=source_slot.sha256,
                    size_bytes=source_slot.size_bytes,
                    coordinate=coordinate,
                    structural_abi_fingerprint=source_slot.structural_abi_fingerprint,
                    content_digest=source_slot.content_digest,
                    population=_population_record(
                        spec.slot,
                        prepared_slots[spec.slot],
                        population_member_ids=expected_population_member_ids or {},
                        population_slots=population_slots,
                    ),
                    metadata=dict(spec.metadata),
                )
                content_digest = source_slot.content_digest
                provenance = CheckpointForkSlotProvenance(
                    slot=spec.slot,
                    source_sha256=source_slot.sha256,
                    target_sha256=source_slot.sha256,
                    source_relative_path=source_slot.relative_path,
                    target_relative_path=slot_record.relative_path,
                    transfer_mode=mode,  # type: ignore[arg-type]
                )
            slot_records.append(slot_record)
            slot_digests.append(content_digest)
            slot_provenance.append(provenance)
            transfer_modes[spec.slot] = provenance.transfer_mode

        _validate_slot_coordinate_consistency(
            barrier=barrier,
            completed_coordinate=coordinate,
            slot_records=slot_records,
        )
        transaction_root = _transaction_root_sha256(slot_digests)
        manifest_metadata = dict(source.manifest.metadata)
        manifest_metadata["phase"] = barrier.phase
        manifest_metadata["forked_from_transaction_id"] = source.manifest.transaction_id
        if request is not None:
            manifest_metadata["checkpoint_continuation"] = request.model_dump(
                mode="json",
                exclude_none=True,
            )
        manifest_metadata.update(dict(metadata or {}))
        completed_batches = _completed_training_batches(
            None,
            manifest_metadata,
            default=source.manifest.completed_training_batches,
        )
        manifest = CheckpointTransactionManifest(
            transaction_id=transaction_id,
            run_id=coordinate.run_id,
            status=source.manifest.status,
            barrier=barrier.name,
            completed_coordinate=coordinate,
            completed_training_batches=completed_batches,
            consistency_predicate=derive_consistency_predicate(phase_program),
            run_contract_binding=run_contract_binding(target_run_spec, phase_program),
            slots=slot_records,
            content_integrity_digest=ContentIntegrityDigest(
                slots=slot_digests,
                transaction_root_sha256=transaction_root,
            ),
            history_availability=dict(source.manifest.history_availability),
            parent_lineage=[
                CheckpointLineageRef(
                    transaction_id=source.manifest.transaction_id,
                    relationship="new_lineage_override",
                    metadata={"fork_source": True},
                ),
                *source.manifest.parent_lineage,
            ],
            source_training_run=source.manifest.source_training_run,
            fork_provenance=CheckpointForkProvenance(
                source=CheckpointForkSourceRecord(
                    transaction_id=source.manifest.transaction_id,
                    run_id=source.manifest.run_id,
                    manifest_sha256=source.latest_pointer.manifest_sha256,
                    transaction_root_sha256=(
                        source.manifest.content_integrity_digest.transaction_root_sha256
                    ),
                    manifest_relative_path=source.latest_pointer.manifest_relative_path,
                    slot_content_digests=list(
                        source.manifest.content_integrity_digest.slots
                    ),
                ),
                slots=slot_provenance,
                tool_version=tool_version or feedbax_version(),
            ),
            metadata=manifest_metadata,
        )
        manifest_path = tmp_dir / MANIFEST_NAME
        _write_json_atomic(manifest_path, manifest.model_dump(mode="json", exclude_none=True))
        manifest_sha256 = _sha256_file(manifest_path)
        os.replace(tmp_dir, final_dir)
        moved_to_final = True
        manifest_path = final_dir / MANIFEST_NAME
        latest_pointer = CheckpointLatestPointer(
            run_id=coordinate.run_id,
            transaction_id=transaction_id,
            manifest_relative_path=str(manifest_path.relative_to(target_root_path)),
            manifest_sha256=manifest_sha256,
            transaction_root_sha256=transaction_root,
            completed_coordinate=coordinate,
            completed_training_batches=completed_batches,
        )
        _load_checkpoint_from_pointer(
            target_root_path,
            latest_pointer,
            expected_run_spec=target_run_spec,
            expected_phase_program=phase_program,
            expected_slots=validation_slots,
            expected_population_member_ids=expected_population_member_ids,
            continuation_request=request,
            allow_new_lineage_override=False,
        )
        latest_path = target_root_path / LATEST_POINTER_NAME
        _write_json_atomic(
            latest_path,
            latest_pointer.model_dump(mode="json", exclude_none=True),
        )
        return CheckpointForkResult(
            root=target_root_path,
            transaction_dir=final_dir,
            manifest_path=manifest_path,
            latest_pointer_path=latest_path,
            manifest=manifest,
            latest_pointer=latest_pointer,
            slot_transfer_modes=transfer_modes,
            source_provenance_notices=source.provenance_notices,
        )
    except Exception:
        shutil.rmtree(final_dir if moved_to_final else tmp_dir, ignore_errors=True)
        raise


def _load_latest_checkpoint_transaction(root: str | Path) -> _LoadedCheckpointTransaction:
    root_path = Path(root)
    latest = _load_latest_pointer(root_path)
    manifest_path, manifest = _manifest_from_latest_pointer(root_path, latest)
    loaded_slots: dict[str, Any] = {}
    transaction_dir = manifest_path.parent
    for slot in manifest.slots:
        blob_path = transaction_dir / slot.relative_path
        blob_bytes = _read_blob(slot, blob_path)
        try:
            loaded_slots[slot.slot] = pickle.loads(blob_bytes)
        except Exception as exc:
            raise CheckpointIntegrityError(
                f"checkpoint slot {slot.slot!r} could not be deserialized"
            ) from exc
    provenance_notices, _ = _validate_manifest_structural_abi(manifest, loaded_slots)
    return _LoadedCheckpointTransaction(
        root=root_path,
        latest_pointer=latest,
        manifest_path=manifest_path,
        manifest=manifest,
        slots=loaded_slots,
        provenance_notices=tuple(provenance_notices),
    )


def _apply_slot_transforms(
    loaded_slots: Mapping[str, Any],
    *,
    transforms: Mapping[str, ResumeSlotTransform],
) -> dict[str, Any]:
    transformed_slots = dict(loaded_slots)
    for slot, transform in transforms.items():
        if slot not in transformed_slots:
            raise CheckpointCompatibilityError(
                f"cannot transform missing checkpoint slot {slot!r}"
            )
        try:
            transformed_mapping = dict(transform(transformed_slots))
        except CheckpointCustodyError:
            raise
        except Exception as exc:
            raise CheckpointCompatibilityError(
                f"checkpoint fork transform failed for slot {slot!r}"
            ) from exc
        if slot not in transformed_mapping:
            raise CheckpointCompatibilityError(
                f"checkpoint fork transform dropped slot {slot!r}"
            )
        transformed_slots[slot] = transformed_mapping[slot]
    return transformed_slots


def _coerce_continuation_request(
    value: CheckpointContinuationRequest | Mapping[str, Any] | None,
) -> CheckpointContinuationRequest | None:
    if value is None:
        return None
    if isinstance(value, CheckpointContinuationRequest):
        return value
    try:
        return CheckpointContinuationRequest.model_validate(value)
    except ValidationError as exc:
        raise CheckpointCompatibilityError(
            f"checkpoint continuation request is invalid: {exc}"
        ) from exc


def _apply_declared_continuation_request(
    loaded_slots: Mapping[str, Any],
    *,
    expected_slots: Mapping[str, Any],
    manifest: CheckpointTransactionManifest,
    request: CheckpointContinuationRequest,
) -> dict[str, Any]:
    """Extend declared final-axis leaves while preserving checkpoint prefixes.

    The target template supplies only the new tail.  This keeps the loaded
    checkpoint's completed-history prefix byte-for-byte in value while avoiding
    any inference about arbitrary PyTree leaves.
    """
    completed = manifest.completed_training_batches
    if completed is None:
        raise CheckpointCompatibilityError(
            "checkpoint continuation requires manifest.completed_training_batches; "
            f"contract_source_completed={request.source_completed_batches}"
        )
    if int(completed) != request.source_completed_batches:
        raise CheckpointCompatibilityError(
            "checkpoint continuation source completed batches mismatch; "
            f"manifest={completed} contract={request.source_completed_batches}"
        )
    target_total = request.target_total
    if target_total < int(completed):
        raise CheckpointCompatibilityError(
            "checkpoint continuation target total precedes source; "
            f"source_completed={completed} target_total={target_total}"
        )
    if target_total == int(completed):
        return dict(loaded_slots)

    declarations = {(leaf.slot, leaf.tree_path): leaf for leaf in request.batch_indexed_leaves}
    extended: dict[str, Any] = dict(loaded_slots)
    for slot, source_value in loaded_slots.items():
        target_value = expected_slots.get(slot)
        if target_value is None:
            continue
        source_pairs, source_treedef = jt.flatten_with_path(source_value)
        target_pairs, _target_treedef = jt.flatten_with_path(target_value)
        target_by_path = {
            _key_path_to_text(path): leaf for path, leaf in target_pairs
        }
        source_paths = {_key_path_to_text(path) for path, _leaf in source_pairs}
        leaves = [leaf for _path, leaf in source_pairs]
        changed = False
        for index, (path, source_leaf) in enumerate(source_pairs):
            tree_path = _key_path_to_text(path)
            target_leaf = target_by_path.get(tree_path)
            declaration = declarations.get((slot, tree_path))
            if declaration is not None:
                leaves[index] = _extend_declared_batch_leaf(
                    source_leaf,
                    target_leaf,
                    slot=slot,
                    tree_path=tree_path,
                    source_completed=int(completed),
                    target_total=target_total,
                    declaration=declaration,
                )
                changed = True
                continue
            if _is_undeclared_batch_horizon_mismatch(
                source_leaf,
                target_leaf,
                source_completed=int(completed),
                target_total=target_total,
            ):
                raise CheckpointCompatibilityError(
                    "unsupported batch-indexed continuation leaf; "
                    f"slot={slot!r} path={tree_path!r} "
                    f"source_shape={tuple(jnp.asarray(source_leaf).shape)!r} "
                    f"target_shape={tuple(jnp.asarray(target_leaf).shape)!r} "
                    "contract=batch_indexed_leaves"
                )
        missing = [
            leaf
            for leaf in request.batch_indexed_leaves
            if leaf.slot == slot and leaf.tree_path not in source_paths
        ]
        if missing:
            raise CheckpointCompatibilityError(
                "declared batch-indexed continuation leaf is missing from source slot; "
                f"slot={slot!r} paths={[leaf.tree_path for leaf in missing]!r} "
                "contract=batch_indexed_leaves"
            )
        if changed:
            extended[slot] = jt.unflatten(source_treedef, leaves)

    undeclared_slots = sorted({leaf.slot for leaf in request.batch_indexed_leaves} - set(loaded_slots))
    if undeclared_slots:
        raise CheckpointCompatibilityError(
            "declared batch-indexed continuation slots are missing; "
            f"slots={undeclared_slots!r} contract=batch_indexed_leaves"
        )
    return extended


def _extend_declared_batch_leaf(
    source_leaf: Any,
    target_leaf: Any,
    *,
    slot: str,
    tree_path: str,
    source_completed: int,
    target_total: int,
    declaration: BatchIndexedCheckpointLeafSpec,
) -> Any:
    if target_leaf is None or not eqx.is_array(source_leaf) or not eqx.is_array(target_leaf):
        raise CheckpointCompatibilityError(
            "declared batch-indexed continuation leaf must be arrays in source and target; "
            f"slot={slot!r} path={tree_path!r} contract={declaration.model_dump()!r}"
        )
    source = jnp.asarray(source_leaf)
    target = jnp.asarray(target_leaf)
    if source.ndim < 1 or target.ndim < 1:
        raise CheckpointCompatibilityError(
            "declared batch-indexed continuation leaf must have a final axis; "
            f"slot={slot!r} path={tree_path!r} source_shape={source.shape!r} "
            f"target_shape={target.shape!r}"
        )
    if source.shape[:-1] != target.shape[:-1] or source.dtype != target.dtype:
        raise CheckpointCompatibilityError(
            "declared batch-indexed continuation leaf prefix shape or dtype mismatch; "
            f"slot={slot!r} path={tree_path!r} source_shape={source.shape!r} "
            f"target_shape={target.shape!r} source_dtype={source.dtype} "
            f"target_dtype={target.dtype} contract={declaration.model_dump()!r}"
        )
    if source.shape[-1] == target_total and target.shape[-1] == target_total:
        return source_leaf
    if source.shape[-1] != source_completed or target.shape[-1] != target_total:
        raise CheckpointCompatibilityError(
            "declared batch-indexed continuation leaf does not match declared horizons; "
            f"slot={slot!r} path={tree_path!r} source_shape={source.shape!r} "
            f"target_shape={target.shape!r} source_completed={source_completed} "
            f"target_total={target_total} contract={declaration.model_dump()!r}"
        )
    return jnp.concatenate((source, target[..., source_completed:]), axis=-1)


def _is_undeclared_batch_horizon_mismatch(
    source_leaf: Any,
    target_leaf: Any,
    *,
    source_completed: int,
    target_total: int,
) -> bool:
    if target_leaf is None or not eqx.is_array(source_leaf) or not eqx.is_array(target_leaf):
        return False
    source = jnp.asarray(source_leaf)
    target = jnp.asarray(target_leaf)
    return (
        source.ndim >= 1
        and target.ndim >= 1
        and source.shape[:-1] == target.shape[:-1]
        and source.shape[-1] == source_completed
        and target.shape[-1] == target_total
        and source.shape != target.shape
    )


def _write_fresh_slot_blob(
    spec: CheckpointSlotSpec,
    value: Any,
    *,
    blob_dir: Path,
    transaction_dir: Path,
    coordinate: ProgressCoordinate,
    slot_roles: Mapping[str, str],
    population_slots: set[str],
    population_member_ids: Mapping[str, Sequence[str]],
) -> tuple[CheckpointSlotBlobRef, SlotContentDigest]:
    blob_bytes = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    blob_sha256 = sha256_bytes(blob_bytes)
    blob_path = blob_dir / f"{spec.slot}-{blob_sha256}.pkl"
    _write_bytes_atomic(blob_path, blob_bytes)
    integrity = _slot_integrity_records(value)
    content_digest = SlotContentDigest(
        slot=spec.slot,
        blob_sha256=blob_sha256,
        blob_size_bytes=len(blob_bytes),
        leaf_hashes=integrity.leaf_digests,
        slot_root_sha256=_slot_root_sha256(
            spec.slot,
            blob_sha256,
            integrity.leaf_digests,
        ),
    )
    slot_record = CheckpointSlotBlobRef(
        slot=spec.slot,
        role=slot_roles.get(spec.slot, "auxiliary"),
        required=spec.required,
        relative_path=str(blob_path.relative_to(transaction_dir)),
        sha256=blob_sha256,
        size_bytes=len(blob_bytes),
        coordinate=coordinate,
        structural_abi_fingerprint=integrity.structural_abi_fingerprint,
        content_digest=content_digest,
        population=_population_record(
            spec.slot,
            value,
            population_member_ids=population_member_ids,
            population_slots=population_slots,
        ),
        metadata=dict(spec.metadata),
    )
    return slot_record, content_digest


def _verify_source_blob_before_transfer(
    slot: CheckpointSlotBlobRef,
    blob_path: Path,
) -> None:
    _read_blob(slot, blob_path)


def _hardlink_first_copy_fallback(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
        return "hardlink"
    except OSError as exc:
        _LOGGER.info(
            "checkpoint fork hardlink failed; copying blob instead",
            extra={"source": str(source), "target": str(target), "error": str(exc)},
        )
        shutil.copy2(source, target)
        return "copy"


def _transform_record(
    slot: str,
    transform: ResumeSlotTransform,
    metadata: Mapping[str, Any],
) -> CheckpointForkTransformRecord:
    parameters = metadata.get("parameters", {})
    if not isinstance(parameters, Mapping):
        raise CheckpointCompatibilityError(
            f"checkpoint fork transform parameters for slot {slot!r} must be a mapping"
        )
    identity = metadata.get("identity")
    if not isinstance(identity, str) or not identity:
        identity = _qualified_callable_name(transform)
    record_metadata = {
        key: value for key, value in metadata.items() if key not in {"identity", "parameters"}
    }
    return CheckpointForkTransformRecord(
        slot=slot,
        identity=identity,
        parameters=dict(parameters),
        metadata=record_metadata,
    )


def _qualified_callable_name(value: Callable[..., Any]) -> str:
    module = getattr(value, "__module__", "")
    qualname = getattr(value, "__qualname__", getattr(value, "__name__", repr(value)))
    return f"{module}:{qualname}" if module else str(qualname)


def run_contract_binding(
    run_spec: TrainingRunSpec,
    phase_program: PhaseProgramSpec,
) -> RunContractBinding:
    """Return the canonical content binding for a migrated run spec projection."""
    projection = run_contract_canonical_projection(run_spec, phase_program)
    training_run_spec = projection["training_run_spec"]
    method_payload = training_run_spec["method_payload"]
    objective = training_run_spec["objective"]
    graph = training_run_spec["graph"]
    optimizer_bindings = [
        binding.model_dump(mode="json", exclude_none=True)
        for binding in phase_program.optimizer_bindings
    ]
    return RunContractBinding(
        training_run_spec_schema_id=training_run_spec["schema_id"],
        training_run_spec_schema_version=training_run_spec["schema_version"],
        training_run_spec_sha256=_canonical_hash(training_run_spec),
        method_payload_schema_id=method_payload["schema_id"],
        method_payload_schema_version=method_payload["schema_version"],
        method_payload_sha256=_canonical_hash(method_payload),
        phase_program_sha256=_canonical_hash(phase_program),
        objective_sha256=_canonical_hash(objective),
        graph_sha256=_canonical_hash(graph),
        optimizer_bindings_sha256=_canonical_hash(optimizer_bindings),
        canonical_projection=projection,
        canonical_projection_sha256=_canonical_hash(projection),
    )


def run_contract_canonical_projection(
    run_spec: TrainingRunSpec,
    phase_program: PhaseProgramSpec,
) -> dict[str, Any]:
    """Return the stored semantic projection for run-contract binding v2.

    The projection intentionally includes the full migrated ``TrainingRunSpec``
    payload and the phase program used by the checkpoint barrier. No
    non-semantic fields are excluded today; adding exclusions would be a
    binding-algorithm change and must bump ``RunContractBinding.algorithm_version``.
    """
    migrated_run_spec = _canonical_training_run_spec_payload(run_spec)
    return {
        "schema_id": "feedbax.manifest.training_checkpoint.run_contract_projection",
        "schema_version": "feedbax.manifest.training_checkpoint.run_contract_projection.v1",
        "algorithm_version": "feedbax.training_checkpoint.run_contract_binding.v2",
        "training_run_spec": migrated_run_spec,
        "phase_program": phase_program.model_dump(mode="json", exclude_none=True),
    }


def structural_abi_fingerprint(value: Any) -> StructuralAbiFingerprint:
    """Return a structural PyTree ABI fingerprint for a slot value."""
    pairs, treedef = jt.flatten_with_path(value)
    leaves = [_leaf_fingerprint(path, leaf) for path, leaf in pairs]
    return _structural_abi_fingerprint_from_leaves(str(treedef), leaves)


def _structural_abi_fingerprint_from_leaves(
    treedef: str,
    leaves: Sequence[SlotLeafFingerprint],
) -> StructuralAbiFingerprint:
    environment_provenance = _serializer_versions()
    payload = _structural_abi_content_payload(treedef, leaves)
    return StructuralAbiFingerprint(
        treedef=treedef,
        leaf_count=len(leaves),
        leaves=list(leaves),
        environment_provenance=environment_provenance,
        fingerprint_sha256=_canonical_hash(payload),
    )


def _format_structural_abi_diff(
    recorded: StructuralAbiFingerprint,
    actual: StructuralAbiFingerprint,
) -> str:
    diffs = _structural_abi_leaf_diffs(recorded, actual)
    displayed_diffs = diffs[:10]
    leaf_diff_text = "; ".join(
        _format_structural_abi_leaf_diff(diff) for diff in displayed_diffs
    )
    if not leaf_diff_text:
        leaf_diff_text = "<none in comparable leaves>"
    elif len(diffs) > len(displayed_diffs):
        leaf_diff_text += f"; ... ({len(diffs)} total differing leaves)"
    suffix = (
        f"; treedef_equal={recorded.treedef == actual.treedef}"
        f"; leaf_count_delta={actual.leaf_count - recorded.leaf_count}"
        f" (recorded={recorded.leaf_count}, actual={actual.leaf_count})"
        f"; differing_leaves={leaf_diff_text}"
    )
    x64_hint = _structural_abi_x64_hint(recorded, actual, diffs)
    if x64_hint is not None:
        suffix += f"; {x64_hint}"
    return suffix


def _structural_abi_leaf_diffs(
    recorded: StructuralAbiFingerprint,
    actual: StructuralAbiFingerprint,
) -> list[_StructuralAbiLeafDiff]:
    diffs: list[_StructuralAbiLeafDiff] = []
    max_leaves = max(len(recorded.leaves), len(actual.leaves))
    for index in range(max_leaves):
        if index >= len(recorded.leaves):
            actual_leaf = actual.leaves[index]
            diffs.append(
                _StructuralAbiLeafDiff(
                    path=actual_leaf.path,
                    field="leaf",
                    recorded="<missing>",
                    actual=_leaf_structural_content_payload(actual_leaf),
                )
            )
            continue
        if index >= len(actual.leaves):
            recorded_leaf = recorded.leaves[index]
            diffs.append(
                _StructuralAbiLeafDiff(
                    path=recorded_leaf.path,
                    field="leaf",
                    recorded=_leaf_structural_content_payload(recorded_leaf),
                    actual="<missing>",
                )
            )
            continue

        recorded_leaf = recorded.leaves[index]
        actual_leaf = actual.leaves[index]
        if recorded_leaf.path != actual_leaf.path:
            diffs.append(
                _StructuralAbiLeafDiff(
                    path=f"leaf[{index}]",
                    field="path",
                    recorded=recorded_leaf.path,
                    actual=actual_leaf.path,
                )
            )
        for field in _STRUCTURAL_ABI_DIFF_FIELDS:
            recorded_value = getattr(recorded_leaf, field)
            actual_value = getattr(actual_leaf, field)
            if recorded_value == actual_value:
                continue
            diffs.append(
                _StructuralAbiLeafDiff(
                    path=recorded_leaf.path,
                    field=field,
                    recorded=recorded_value,
                    actual=actual_value,
                )
            )
    return diffs


def _format_structural_abi_leaf_diff(diff: _StructuralAbiLeafDiff) -> str:
    return (
        f"path={diff.path} field={diff.field} "
        f"recorded={_short_json(diff.recorded)} actual={_short_json(diff.actual)}"
    )


def _structural_abi_x64_hint(
    recorded: StructuralAbiFingerprint,
    actual: StructuralAbiFingerprint,
    diffs: Sequence[_StructuralAbiLeafDiff],
) -> str | None:
    if not diffs:
        return None
    x64_sides: set[str] = set()
    for diff in diffs:
        if diff.field == "dtype":
            x64_side = _x64_dtype_diff_side(diff.recorded, diff.actual)
            if x64_side is None:
                return None
            x64_sides.add(x64_side)
        elif diff.field == "weak_type":
            if not isinstance(diff.recorded, bool) or not isinstance(diff.actual, bool):
                return None
        else:
            return None
    recorded_x64 = _fingerprint_x64_enabled(recorded)
    actual_x64 = _fingerprint_x64_enabled(actual)
    if not x64_sides:
        if recorded_x64 is True and actual_x64 is False:
            x64_sides.add("recorded")
        elif recorded_x64 is False and actual_x64 is True:
            x64_sides.add("actual")
        else:
            return None
    if len(x64_sides) != 1:
        return None
    x64_side = next(iter(x64_sides))
    return (
        f"x64_side={x64_side}"
        f"; recorded_x64_enabled={recorded_x64}"
        f"; actual_x64_enabled={actual_x64}"
        "; hint=jax_enable_x64 differs between checkpoint writer and reader"
    )


def _x64_dtype_diff_side(recorded: Any, actual: Any) -> str | None:
    recorded_kind_bits = _dtype_kind_bits(recorded)
    actual_kind_bits = _dtype_kind_bits(actual)
    if recorded_kind_bits is None or actual_kind_bits is None:
        return None
    recorded_kind, recorded_bits = recorded_kind_bits
    actual_kind, actual_bits = actual_kind_bits
    if recorded_kind != actual_kind or {recorded_bits, actual_bits} != {32, 64}:
        return None
    if recorded_bits == 64:
        return "recorded"
    return "actual"


def _dtype_kind_bits(dtype: Any) -> tuple[str, int] | None:
    dtype_text = str(dtype)
    for kind in ("float", "int"):
        for bits in (32, 64):
            if dtype_text == f"{kind}{bits}":
                return kind, bits
    return None


def _fingerprint_x64_enabled(fingerprint: StructuralAbiFingerprint) -> bool | None:
    if fingerprint.environment_provenance is None:
        return None
    return fingerprint.environment_provenance.x64_enabled


def _validate_required_slots(
    slot_specs: tuple[CheckpointSlotSpec, ...],
    slots: Mapping[str, Any],
    *,
    error_cls: type[CheckpointCustodyError] = CheckpointCustodyError,
) -> None:
    missing = [spec.slot for spec in slot_specs if spec.required and spec.slot not in slots]
    if missing:
        raise error_cls(f"missing required checkpoint slots: {missing!r}")


def _validate_expected_slot_set(
    barrier: CheckpointBarrierSpec,
    expected_slots: Mapping[str, Any],
) -> None:
    missing = [
        spec.slot for spec in barrier.slots if spec.required and spec.slot not in expected_slots
    ]
    if missing:
        raise CheckpointCompatibilityError(
            f"resume templates missing required checkpoint slots: {missing!r}"
        )


def _validate_structural_abi(
    manifest: CheckpointTransactionManifest,
    expected_slots: Mapping[str, Any],
    loaded_slots: Mapping[str, Any],
    *,
    loaded_fingerprints: Mapping[str, StructuralAbiFingerprint] | None = None,
) -> None:
    loaded_fingerprints = loaded_fingerprints or {}
    for slot in manifest.slots:
        if slot.slot not in expected_slots:
            continue
        if slot.slot not in loaded_slots:
            continue
        loaded = loaded_slots[slot.slot]
        loaded_fingerprint = loaded_fingerprints.get(slot.slot)
        if loaded_fingerprint is None:
            loaded_fingerprint = structural_abi_fingerprint(loaded)
        expected = structural_abi_fingerprint(expected_slots[slot.slot])
        if loaded_fingerprint.fingerprint_sha256 != expected.fingerprint_sha256:
            diff_suffix = _format_structural_abi_diff(expected, loaded_fingerprint)
            raise CheckpointCompatibilityError(
                f"checkpoint slot {slot.slot!r} structural ABI mismatch"
                f"{diff_suffix}"
            )


def _validate_manifest_structural_abi(
    manifest: CheckpointTransactionManifest,
    loaded_slots: Mapping[str, Any],
) -> tuple[list[CheckpointProvenanceNotice], dict[str, StructuralAbiFingerprint]]:
    notices: list[CheckpointProvenanceNotice] = []
    loaded_fingerprints: dict[str, StructuralAbiFingerprint] = {}
    for slot in manifest.slots:
        if slot.slot not in loaded_slots:
            continue
        loaded_fingerprint = structural_abi_fingerprint(loaded_slots[slot.slot])
        loaded_fingerprints[slot.slot] = loaded_fingerprint
        if (
            loaded_fingerprint.fingerprint_sha256
            != slot.structural_abi_fingerprint.fingerprint_sha256
        ):
            diff_suffix = _format_structural_abi_diff(
                slot.structural_abi_fingerprint,
                loaded_fingerprint,
            )
            raise CheckpointIntegrityError(
                f"checkpoint slot {slot.slot!r} structural ABI fingerprint is stale"
                f"{diff_suffix}"
            )
        notice = _environment_provenance_notice(slot.slot, slot.structural_abi_fingerprint)
        if notice is not None:
            _LOGGER.warning(notice.message)
            notices.append(notice)
    return notices, loaded_fingerprints


def _validate_contract_binding(
    manifest: CheckpointTransactionManifest,
    expected_run_spec: TrainingRunSpec,
    expected_phase_program: PhaseProgramSpec,
    *,
    allow_new_lineage_override: bool,
) -> None:
    expected = run_contract_binding(expected_run_spec, expected_phase_program)
    if _contract_binding_matches(
        manifest.run_contract_binding,
        expected,
        expected_run_spec,
        expected_phase_program,
    ):
        return
    if allow_new_lineage_override:
        return
    diffs = _run_contract_binding_diffs(manifest.run_contract_binding, expected)
    if diffs:
        diff_text = "; ".join(_format_binding_diff(diff) for diff in diffs[:8])
        diff_suffix = f"; differing_fields={diff_text}"
    else:
        diff_suffix = (
            "; stored canonical projection is unavailable for this legacy binding"
            + _format_binding_hash_field_summary(
                manifest.run_contract_binding,
                expected,
                expected_run_spec,
                expected_phase_program,
            )
        )
    raise CheckpointContractBindingError(
        "checkpoint run-contract content binding does not match expected run spec; "
        "pass allow_new_lineage_override=True to resume as new lineage"
        f"{diff_suffix}"
    )


def _contract_binding_matches(
    recorded: RunContractBinding,
    expected: RunContractBinding,
    expected_run_spec: TrainingRunSpec,
    expected_phase_program: PhaseProgramSpec,
) -> bool:
    if (
        recorded.canonical_projection_sha256 is not None
        and expected.canonical_projection_sha256 is not None
    ):
        return recorded.canonical_projection_sha256 == expected.canonical_projection_sha256

    if _binding_hash_fields(recorded) == _binding_hash_fields(expected):
        return True

    legacy_hashes = _legacy_binding_hash_fields(
        expected_run_spec,
        expected_phase_program,
        recorded.training_run_spec_schema_version,
    )
    return _binding_hash_fields(recorded) == legacy_hashes


def _binding_hash_fields(binding: RunContractBinding) -> dict[str, str | None]:
    return {
        "training_run_spec_sha256": binding.training_run_spec_sha256,
        "method_payload_sha256": binding.method_payload_sha256,
        "phase_program_sha256": binding.phase_program_sha256,
        "objective_sha256": binding.objective_sha256,
        "graph_sha256": binding.graph_sha256,
        "optimizer_bindings_sha256": binding.optimizer_bindings_sha256,
    }


def _format_binding_hash_field_summary(
    recorded: RunContractBinding,
    expected: RunContractBinding,
    expected_run_spec: TrainingRunSpec,
    expected_phase_program: PhaseProgramSpec,
) -> str:
    recorded_hashes = _binding_hash_fields(recorded)
    current_hashes = _binding_hash_fields(expected)
    legacy_hashes = _legacy_binding_hash_fields(
        expected_run_spec,
        expected_phase_program,
        recorded.training_run_spec_schema_version,
    )

    comparison_label = "current"
    comparison_hashes = current_hashes
    if legacy_hashes != current_hashes:
        current_mismatches = _binding_hash_field_mismatches(
            recorded_hashes,
            current_hashes,
        )
        legacy_mismatches = _binding_hash_field_mismatches(
            recorded_hashes,
            legacy_hashes,
        )
        if len(legacy_mismatches) < len(current_mismatches):
            comparison_label = "legacy"
            comparison_hashes = legacy_hashes

    mismatches = _binding_hash_field_mismatches(recorded_hashes, comparison_hashes)
    matches = [
        field
        for field in recorded_hashes
        if recorded_hashes[field] == comparison_hashes.get(field)
    ]
    return (
        f"; hash_comparison={comparison_label}"
        f"; hash_field_mismatches={mismatches!r}"
        f"; hash_field_matches={matches!r}"
    )


def _binding_hash_field_mismatches(
    recorded_hashes: Mapping[str, str | None],
    expected_hashes: Mapping[str, str | None],
) -> list[str]:
    return [
        field
        for field, recorded_hash in recorded_hashes.items()
        if recorded_hash != expected_hashes.get(field)
    ]


def _legacy_binding_hash_fields(
    expected_run_spec: TrainingRunSpec,
    expected_phase_program: PhaseProgramSpec,
    source_schema_version: str,
) -> dict[str, str | None]:
    training_run_spec = _legacy_training_run_spec_payload(
        expected_run_spec,
        source_schema_version,
    )
    optimizer_bindings = [
        binding.model_dump(mode="json", exclude_none=True)
        for binding in expected_phase_program.optimizer_bindings
    ]
    return {
        "training_run_spec_sha256": _canonical_hash(training_run_spec),
        "method_payload_sha256": _canonical_hash(training_run_spec["method_payload"]),
        "phase_program_sha256": _canonical_hash(expected_phase_program),
        "objective_sha256": _canonical_hash(training_run_spec["objective"]),
        "graph_sha256": _canonical_hash(training_run_spec["graph"]),
        "optimizer_bindings_sha256": _canonical_hash(optimizer_bindings),
    }


def _run_contract_binding_diffs(
    recorded: RunContractBinding,
    expected: RunContractBinding,
) -> list[tuple[str, Any, Any]]:
    if recorded.canonical_projection is None or expected.canonical_projection is None:
        return []
    return _value_diffs(recorded.canonical_projection, expected.canonical_projection)


def _value_diffs(
    recorded: Any,
    expected: Any,
    *,
    path: str = "",
    limit: int = 32,
) -> list[tuple[str, Any, Any]]:
    if len(path) > 512:
        return [(path, recorded, expected)]
    if isinstance(recorded, Mapping) and isinstance(expected, Mapping):
        diffs: list[tuple[str, Any, Any]] = []
        for key in sorted(set(recorded) | set(expected), key=str):
            if len(diffs) >= limit:
                break
            child_path = f"{path}/{key}" if path else f"/{key}"
            if key not in recorded:
                diffs.append((child_path, "<missing>", expected[key]))
            elif key not in expected:
                diffs.append((child_path, recorded[key], "<missing>"))
            else:
                diffs.extend(
                    _value_diffs(
                        recorded[key],
                        expected[key],
                        path=child_path,
                        limit=limit - len(diffs),
                    )
                )
        return diffs[:limit]
    if isinstance(recorded, list) and isinstance(expected, list):
        diffs = []
        for index, (left, right) in enumerate(zip(recorded, expected, strict=False)):
            if len(diffs) >= limit:
                break
            diffs.extend(
                _value_diffs(
                    left,
                    right,
                    path=f"{path}/{index}",
                    limit=limit - len(diffs),
                )
            )
        if len(recorded) != len(expected) and len(diffs) < limit:
            diffs.append((f"{path}/length", len(recorded), len(expected)))
        return diffs[:limit]
    if recorded != expected:
        return [(path or "/", recorded, expected)]
    return []


def _format_binding_diff(diff: tuple[str, Any, Any]) -> str:
    path, recorded, expected = diff
    return (
        f"{path}: recorded={_short_json(recorded)}, "
        f"expected={_short_json(expected)}"
    )


def _short_json(value: Any, *, limit: int = 160) -> str:
    text = json.dumps(value, sort_keys=True, default=str)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _canonical_training_run_spec_payload(run_spec: TrainingRunSpec) -> dict[str, Any]:
    from feedbax.contracts.migrations import migrate_structured_spec_payload

    payload = run_spec.model_dump(mode="json", exclude_none=True)
    migrated = migrate_structured_spec_payload(
        "TrainingRunSpec",
        payload,
        path="checkpoint_binding/training_run_spec",
    ).payload
    return TrainingRunSpec.model_validate(migrated).model_dump(
        mode="json",
        exclude_none=True,
    )


def _legacy_training_run_spec_payload(
    run_spec: TrainingRunSpec,
    source_schema_version: str,
) -> dict[str, Any]:
    payload = _canonical_training_run_spec_payload(run_spec)
    if source_schema_version == TRAINING_RUN_SPEC_SCHEMA_VERSION_V1:
        payload["schema_version"] = TRAINING_RUN_SPEC_SCHEMA_VERSION_V1
        payload.pop("on_nan", None)
    return payload


def _environment_provenance_notice(
    slot: str,
    fingerprint: StructuralAbiFingerprint,
) -> CheckpointProvenanceNotice | None:
    current = _serializer_versions().model_dump(mode="json", exclude_none=True)
    if fingerprint.environment_provenance is None:
        return CheckpointProvenanceNotice(
            code="environment_provenance_unverifiable",
            slot=slot,
            message=(
                f"checkpoint slot {slot!r} has no recorded environment provenance; "
                "content integrity was verified"
            ),
            recorded=None,
            current=current,
            metadata={"provenance_status": fingerprint.provenance_status},
        )

    recorded = fingerprint.environment_provenance.model_dump(
        mode="json",
        exclude_none=True,
    )
    if recorded == current:
        return None
    return CheckpointProvenanceNotice(
        code="environment_provenance_mismatch",
        slot=slot,
        message=(
            f"checkpoint slot {slot!r} was written under different serializer/runtime "
            "provenance; content integrity was verified"
        ),
        recorded=recorded,
        current=current,
    )


def _validate_slot_coordinate_consistency(
    *,
    barrier: CheckpointBarrierSpec,
    completed_coordinate: ProgressCoordinate,
    slot_records: Sequence[CheckpointSlotBlobRef],
) -> None:
    mode = str(barrier.metadata.get("consistency_mode", "barrier-coordinate"))
    if mode not in {"barrier-coordinate", "population-barrier"}:
        raise CheckpointConsistencyError(f"unsupported checkpoint consistency mode {mode!r}")
    mismatches = [
        slot.slot
        for slot in slot_records
        if not _same_barrier_coordinate(slot.coordinate, completed_coordinate)
    ]
    if mismatches:
        raise CheckpointConsistencyError(
            "checkpoint slot coordinates violate method-declared consistency "
            f"predicate mode={mode!r}; mismatched_slots={mismatches!r}"
        )


def _same_barrier_coordinate(left: ProgressCoordinate, right: ProgressCoordinate) -> bool:
    return (
        left.run_id == right.run_id
        and left.phase == right.phase
        and left.global_step == right.global_step
        and left.outer_step == right.outer_step
        and left.inner_step == right.inner_step
        and left.completed_barrier == right.completed_barrier
        and left.schedule_origin_step == right.schedule_origin_step
    )


def _validate_population_identities(
    manifest: CheckpointTransactionManifest,
    expected_population_member_ids: Mapping[str, Sequence[str]],
) -> None:
    for slot in manifest.slots:
        if slot.population is None or slot.slot not in expected_population_member_ids:
            continue
        expected = [str(member_id) for member_id in expected_population_member_ids[slot.slot]]
        if slot.population.member_ids != expected:
            raise CheckpointCompatibilityError(
                f"population identity mismatch for slot {slot.slot!r}: "
                f"checkpoint={slot.population.member_ids!r}, expected={expected!r}"
            )


def _read_blob(slot: CheckpointSlotBlobRef, path: Path) -> bytes:
    if not path.is_file():
        raise CheckpointIntegrityError(f"checkpoint slot {slot.slot!r} blob is missing")
    data = path.read_bytes()
    if len(data) != slot.size_bytes:
        raise CheckpointIntegrityError(f"checkpoint slot {slot.slot!r} size mismatch")
    if sha256_bytes(data) != slot.sha256:
        raise CheckpointIntegrityError(f"checkpoint slot {slot.slot!r} hash mismatch")
    if slot.content_digest.blob_sha256 != slot.sha256:
        raise CheckpointIntegrityError(f"checkpoint slot {slot.slot!r} digest is stale")
    return data


def _load_latest_pointer(root: Path) -> CheckpointLatestPointer:
    path = root / LATEST_POINTER_NAME
    if not path.is_file():
        _reject_legacy_checkpoint(root)
        raise CheckpointIntegrityError("checkpoint latest pointer is missing")
    try:
        payload = json.loads(path.read_text())
        return CheckpointLatestPointer.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise CheckpointIntegrityError("checkpoint latest pointer is corrupt") from exc


def _reject_legacy_checkpoint(root: Path) -> None:
    layout = detect_known_legacy_checkpoint_layout(root)
    if layout is not None:
        raise CheckpointCompatibilityError(
            _legacy_checkpoint_adoption_message(layout)
        )


def _legacy_checkpoint_adoption_message(layout: DetectedLegacyCheckpointLayout) -> str:
    evidence = ", ".join(layout.evidence[:3])
    if len(layout.evidence) > 3:
        evidence += f", ... ({len(layout.evidence)} evidence paths)"
    return (
        f"recognized legacy checkpoint layout {layout.name!r} ({layout.layout_id}); "
        f"evidence: {evidence}. These checkpoints predate checkpoint custody and "
        "cannot be loaded directly because they do not contain a schema identity, "
        "slot manifest, or run-contract binding. Adopt them with "
        f"`{LEGACY_CHECKPOINT_ADOPTION_ENTRYPOINT}`; required inputs include the "
        "producing commit for the LeafManifest dump and path-mapping rules. See "
        f"{LEGACY_CHECKPOINT_ADOPTION_DOCS}."
    )


def _detect_feedbax_supervised_legacy_layout(root: Path) -> tuple[str, ...]:
    evidence: list[str] = []
    if (root / "last_batch.txt").is_file():
        evidence.append("last_batch.txt")
    evidence.extend(sorted(path.name for path in root.glob("ckpt_*.eqx"))[:8])
    return tuple(evidence)


def _detect_rlrmp_eqx_stream_legacy_layout(root: Path) -> tuple[str, ...]:
    try:
        candidates = sorted(root.iterdir(), key=lambda path: path.name)
    except OSError:
        return ()

    evidence: list[str] = []
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        suffix = candidate.name.removeprefix("checkpoint_")
        if suffix == candidate.name or not suffix.isdigit():
            continue
        required = ("model.eqx", "optimizer_state.eqx", "metadata.json")
        if all((candidate / name).is_file() for name in required):
            evidence.append(f"{candidate.name}/{{model.eqx,optimizer_state.eqx,metadata.json}}")
    return tuple(evidence)


_KNOWN_LEGACY_CHECKPOINT_LAYOUTS = (
    _LegacyCheckpointLayout(
        layout_id="feedbax_supervised_trainer_v0",
        name="Feedbax supervised trainer legacy checkpoint",
        detect=_detect_feedbax_supervised_legacy_layout,
    ),
    _LegacyCheckpointLayout(
        layout_id="rlrmp_eqx_stream_v0",
        name="RLRMP Equinox stream legacy checkpoint",
        detect=_detect_rlrmp_eqx_stream_legacy_layout,
    ),
)


def _load_transaction_manifest(path: Path) -> CheckpointTransactionManifest:
    try:
        payload = json.loads(path.read_text())
        from feedbax.contracts.migrations import migrate_structured_spec_payload

        payload = migrate_structured_spec_payload(
            "TrainingCheckpointTransactionManifest",
            payload,
            path="checkpoint_manifest",
        ).payload
        return CheckpointTransactionManifest.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise CheckpointIntegrityError("checkpoint transaction manifest is corrupt") from exc


def _slot_roles(run_spec: TrainingRunSpec) -> dict[str, str]:
    return {
        slot.name: slot.role
        for slot in run_spec.worker_execution.method_contract.state_slots
    }


def _population_slot_names(run_spec: TrainingRunSpec) -> set[str]:
    return {
        slot.name
        for slot in run_spec.worker_execution.method_contract.state_slots
        if isinstance(slot, StateSlotSpec) and slot.role == "population"
    }


def _population_record(
    slot: str,
    value: Any,
    *,
    population_member_ids: Mapping[str, Sequence[str]],
    population_slots: set[str],
) -> PopulationIdentityRecord | None:
    if slot not in population_slots and slot not in population_member_ids:
        return None
    if isinstance(value, Mapping):
        length = len(value)
        default_ids = [str(key) for key in value.keys()]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        length = len(value)
        default_ids = [str(index) for index in range(length)]
    else:
        raise CheckpointCompatibilityError(
            f"population slot {slot!r} must be a mapping or sequence"
        )
    member_ids = [str(member_id) for member_id in population_member_ids.get(slot, default_ids)]
    return PopulationIdentityRecord(slot=slot, length=length, member_ids=member_ids)


def _leaf_fingerprint(path: Any, leaf: Any) -> SlotLeafFingerprint:
    path_text = _key_path_to_text(path)
    if eqx.is_array(leaf):
        array = jnp.asarray(leaf)
        return SlotLeafFingerprint(
            path=path_text,
            leaf_type=_qualified_type_name(leaf),
            shape=tuple(int(dim) for dim in array.shape),
            dtype=str(array.dtype),
            weak_type=bool(getattr(array, "weak_type", False)),
            sharding=str(getattr(array, "sharding", None)),
            layout=str(getattr(array, "layout", None)),
        )
    static_repr = repr(leaf).encode("utf-8", errors="replace")
    return SlotLeafFingerprint(
        path=path_text,
        leaf_type=_qualified_type_name(leaf),
        static_repr_sha256=sha256_bytes(static_repr),
    )


def _structural_abi_content_payload(
    treedef: str,
    leaves: Sequence[SlotLeafFingerprint],
) -> dict[str, Any]:
    return {
        "fingerprint_algorithm_version": (
            "feedbax.training_checkpoint.structural_abi.content.v2"
        ),
        "treedef": treedef,
        "leaf_count": len(leaves),
        "leaves": [_leaf_structural_content_payload(leaf) for leaf in leaves],
    }


def _leaf_structural_content_payload(leaf: SlotLeafFingerprint) -> dict[str, Any]:
    payload = leaf.model_dump(mode="json", exclude_none=True)
    return {
        key: payload[key]
        for key in (
            "path",
            "leaf_type",
            "shape",
            "dtype",
            "weak_type",
            "static_repr_sha256",
        )
        if key in payload
    }


def _slot_integrity_records(value: Any) -> _SlotIntegrityRecords:
    pairs, treedef = jt.flatten_with_path(value)
    host_leaves = jt.leaves(jax.device_get(value))
    digests: list[SlotLeafContentDigest] = []
    fingerprints: list[SlotLeafFingerprint] = []
    for (path, leaf), host_leaf in zip(pairs, host_leaves, strict=True):
        fingerprints.append(_leaf_fingerprint(path, leaf))
        if eqx.is_array(leaf):
            data = np.asarray(host_leaf).tobytes()
        else:
            data = pickle.dumps(leaf, protocol=pickle.HIGHEST_PROTOCOL)
        digests.append(
            SlotLeafContentDigest(
                path=_key_path_to_text(path),
                sha256=sha256_bytes(data),
                size_bytes=len(data),
            )
        )
    return _SlotIntegrityRecords(
        leaf_digests=digests,
        structural_abi_fingerprint=_structural_abi_fingerprint_from_leaves(
            str(treedef),
            fingerprints,
        ),
    )


def _leaf_content_digests(value: Any) -> list[SlotLeafContentDigest]:
    return _slot_integrity_records(value).leaf_digests


def _slot_root_sha256(
    slot: str,
    blob_sha256: str,
    leaf_digests: Sequence[SlotLeafContentDigest],
) -> str:
    payload = {
        "slot": slot,
        "blob_sha256": blob_sha256,
        "leaf_hashes": [
            digest.model_dump(mode="json", exclude_none=True) for digest in leaf_digests
        ],
    }
    return _canonical_hash(payload)


def _transaction_root_sha256(slot_digests: Sequence[SlotContentDigest]) -> str:
    payload = [
        digest.model_dump(mode="json", exclude_none=True)
        for digest in sorted(slot_digests, key=lambda digest: digest.slot)
    ]
    return _canonical_hash(payload)


def _serializer_versions() -> SerializerVersionRecord:
    return SerializerVersionRecord(
        feedbax_version=feedbax_version(),
        jax_version=getattr(jax, "__version__", None),
        equinox_version=_package_version("equinox"),
        optax_version=_package_version("optax"),
        python_version=platform.python_version(),
        x64_enabled=bool(jax.config.jax_enable_x64),
    )


def _package_version(package_name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(package_name)
    except Exception:
        return None


def _canonical_hash(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", exclude_none=True)
    return sha256_bytes(canonical_json_bytes(value))


def _qualified_type_name(value: Any) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _key_path_to_text(path: Any) -> str:
    if not path:
        return "/"
    parts: list[str] = []
    for key in path:
        name = getattr(key, "name", None)
        idx = getattr(key, "idx", None)
        key_value = getattr(key, "key", None)
        if name is not None:
            parts.append(str(name))
        elif idx is not None:
            parts.append(str(idx))
        elif key_value is not None:
            parts.append(str(key_value))
        else:
            parts.append(str(key))
    return "/" + "/".join(parts)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, sort_keys=True, indent=2).encode("utf-8")
    _write_bytes_atomic(path, data)


def _write_bytes_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
