"""Atomic training checkpoint custody helpers."""

from __future__ import annotations

import hashlib
import json
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
from pydantic import BaseModel, ValidationError

from feedbax.contracts.checkpoints import (
    CheckpointLatestPointer,
    CheckpointLineageRef,
    CheckpointResumeResult,
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
from feedbax.contracts.training import TrainingRunSpec
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
            leaf_digests = _leaf_content_digests(value)
            content_digest = SlotContentDigest(
                slot=spec.slot,
                blob_sha256=blob_sha256,
                blob_size_bytes=len(blob_bytes),
                leaf_hashes=leaf_digests,
                slot_root_sha256=_slot_root_sha256(spec.slot, blob_sha256, leaf_digests),
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
                structural_abi_fingerprint=structural_abi_fingerprint(value),
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
        manifest = CheckpointTransactionManifest(
            transaction_id=transaction_id,
            run_id=coordinate.run_id,
            status=status,  # type: ignore[arg-type]
            barrier=barrier.name,
            completed_coordinate=coordinate,
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
    allow_new_lineage_override: bool = False,
) -> CheckpointResumeResult:
    """Load and validate the latest published transaction before resume."""
    root_path = Path(root)
    latest = _load_latest_pointer(root_path)
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
    _validate_manifest_structural_abi(manifest, loaded_slots)
    if resume_slot_transform is not None:
        loaded_slots = dict(resume_slot_transform(loaded_slots))
        _validate_required_slots(tuple(barrier.slots), loaded_slots)
    _validate_structural_abi(manifest, expected_slots, loaded_slots)

    return CheckpointResumeResult(
        manifest=manifest,
        slots=loaded_slots,
        new_lineage_required=allow_new_lineage_override
        and manifest.run_contract_binding != run_contract_binding(
            expected_run_spec,
            expected_phase_program,
        ),
        previous_transaction_id=(
            manifest.transaction_id if allow_new_lineage_override else None
        ),
    )


def run_contract_binding(
    run_spec: TrainingRunSpec,
    phase_program: PhaseProgramSpec,
) -> RunContractBinding:
    """Return the canonical content binding for a run spec and phase program."""
    method_payload = run_spec.method_payload.model_dump(mode="json", exclude_none=True)
    objective = run_spec.objective.model_dump(mode="json", exclude_none=True)
    graph = run_spec.graph.model_dump(mode="json", exclude_none=True)
    optimizer_bindings = [
        binding.model_dump(mode="json", exclude_none=True)
        for binding in phase_program.optimizer_bindings
    ]
    return RunContractBinding(
        training_run_spec_schema_id=run_spec.schema_id,
        training_run_spec_schema_version=run_spec.schema_version,
        training_run_spec_sha256=_canonical_hash(run_spec),
        method_payload_schema_id=run_spec.method_payload.schema_id,
        method_payload_schema_version=run_spec.method_payload.schema_version,
        method_payload_sha256=_canonical_hash(method_payload),
        phase_program_sha256=_canonical_hash(phase_program),
        objective_sha256=_canonical_hash(objective),
        graph_sha256=_canonical_hash(graph),
        optimizer_bindings_sha256=_canonical_hash(optimizer_bindings),
    )


def structural_abi_fingerprint(value: Any) -> StructuralAbiFingerprint:
    """Return a structural PyTree ABI fingerprint for a slot value."""
    pairs, treedef = jt.flatten_with_path(value)
    leaves = [_leaf_fingerprint(path, leaf) for path, leaf in pairs]
    payload = {
        "treedef": str(treedef),
        "leaf_count": len(leaves),
        "leaves": [leaf.model_dump(mode="json", exclude_none=True) for leaf in leaves],
        "serializer_versions": _serializer_versions().model_dump(
            mode="json",
            exclude_none=True,
        ),
    }
    return StructuralAbiFingerprint(
        treedef=str(treedef),
        leaf_count=len(leaves),
        leaves=leaves,
        serializer_versions=_serializer_versions(),
        fingerprint_sha256=_canonical_hash(payload),
    )


def _validate_required_slots(
    slot_specs: tuple[CheckpointSlotSpec, ...],
    slots: Mapping[str, Any],
) -> None:
    missing = [spec.slot for spec in slot_specs if spec.required and spec.slot not in slots]
    if missing:
        raise CheckpointCustodyError(f"missing required checkpoint slots: {missing!r}")


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
) -> None:
    for slot in manifest.slots:
        if slot.slot not in expected_slots:
            continue
        if slot.slot not in loaded_slots:
            continue
        loaded = loaded_slots[slot.slot]
        loaded_fingerprint = structural_abi_fingerprint(loaded)
        expected = structural_abi_fingerprint(expected_slots[slot.slot])
        if loaded_fingerprint.fingerprint_sha256 != expected.fingerprint_sha256:
            raise CheckpointCompatibilityError(
                f"checkpoint slot {slot.slot!r} structural ABI mismatch"
            )


def _validate_manifest_structural_abi(
    manifest: CheckpointTransactionManifest,
    loaded_slots: Mapping[str, Any],
) -> None:
    for slot in manifest.slots:
        if slot.slot not in loaded_slots:
            continue
        loaded_fingerprint = structural_abi_fingerprint(loaded_slots[slot.slot])
        if (
            loaded_fingerprint.fingerprint_sha256
            != slot.structural_abi_fingerprint.fingerprint_sha256
        ):
            raise CheckpointIntegrityError(
                f"checkpoint slot {slot.slot!r} structural ABI fingerprint is stale"
            )


def _validate_contract_binding(
    manifest: CheckpointTransactionManifest,
    expected_run_spec: TrainingRunSpec,
    expected_phase_program: PhaseProgramSpec,
    *,
    allow_new_lineage_override: bool,
) -> None:
    expected = run_contract_binding(expected_run_spec, expected_phase_program)
    if manifest.run_contract_binding == expected:
        return
    if allow_new_lineage_override:
        return
    raise CheckpointContractBindingError(
        "checkpoint run-contract content binding does not match expected run spec; "
        "pass allow_new_lineage_override=True to resume as new lineage"
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
        raise CheckpointIntegrityError("checkpoint latest pointer is missing")
    try:
        payload = json.loads(path.read_text())
        return CheckpointLatestPointer.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise CheckpointIntegrityError("checkpoint latest pointer is corrupt") from exc


def _load_transaction_manifest(path: Path) -> CheckpointTransactionManifest:
    try:
        payload = json.loads(path.read_text())
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


def _leaf_content_digests(value: Any) -> list[SlotLeafContentDigest]:
    digests: list[SlotLeafContentDigest] = []
    for path, leaf in jt.leaves_with_path(value):
        if eqx.is_array(leaf):
            data = bytes(jnp.asarray(leaf).tobytes())
        else:
            data = pickle.dumps(leaf, protocol=pickle.HIGHEST_PROTOCOL)
        digests.append(
            SlotLeafContentDigest(
                path=_key_path_to_text(path),
                sha256=sha256_bytes(data),
                size_bytes=len(data),
            )
        )
    return digests


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
