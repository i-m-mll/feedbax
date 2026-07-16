"""Atomic training checkpoint custody helpers."""

from __future__ import annotations

import ctypes
import hashlib
import gzip
import io
import json
import logging
import os
import pickle
import platform
import shutil
import stat
import tarfile
import tempfile
import uuid
import zlib
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any
from urllib.parse import unquote, urlsplit

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree as jt
from jax_cookbook import is_type
import numpy as np
from pydantic import BaseModel, ConfigDict, ValidationError

from feedbax.contracts.checkpoints import (
    BatchHistory,
    CheckpointContinuationRequest,
    CheckpointDocumentLoadResult,
    CheckpointForkBarrierMapping,
    CheckpointForkCompatibilityProjection,
    CheckpointForkPlan,
    CheckpointLatestPointer,
    CheckpointLineageRef,
    CheckpointForkProvenance,
    CheckpointForkSlotProvenance,
    CheckpointForkSourceRecord,
    CheckpointForkTarget,
    CheckpointForkTransformRecord,
    CheckpointForkTransformStep,
    CheckpointResumeResult,
    CheckpointSegmentLineage,
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
from feedbax.contracts.manifest import (
    ArtifactRef,
    ArtifactMigrationRecord,
    ParentRef,
    canonical_json_bytes,
    feedbax_version,
    sha256_bytes,
)
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.contracts.training import TRAINING_RUN_SPEC_SCHEMA_VERSION_V1, TrainingRunSpec
from feedbax.contracts.worker import (
    CheckpointBarrierSpec,
    CheckpointSlotSpec,
    MaterializedSlotAxisBinding,
    PhaseProgramSpec,
    ProgressCoordinate,
    StateSlotSpec,
    derive_consistency_predicate,
)
from feedbax.orchestration.events import normalize_serialized_metrics
from feedbax.training.worker_validation import resolve_execution_mapping


LATEST_POINTER_NAME = "latest.json"
TRANSACTIONS_DIR_NAME = "transactions"
MANIFEST_NAME = "manifest.json"
CHECKPOINT_CUSTODY_ARCHIVE_SCHEMA_ID = "feedbax.archive.training_checkpoint_custody"
CHECKPOINT_CUSTODY_ARCHIVE_SCHEMA_VERSION = "feedbax.archive.training_checkpoint_custody.v1"
CHECKPOINT_CUSTODY_ARCHIVE_MEDIA_TYPE = (
    "application/vnd.feedbax.training-checkpoint-custody.v1+tar+gzip"
)
_LOGGER = logging.getLogger(__name__)
_STRUCTURAL_ABI_DIFF_FIELDS = (
    "dtype",
    "shape",
    "weak_type",
    "leaf_type",
    "static_repr_sha256",
)
_JAX_ARRAY_LEAF_TYPE = "jax.Array"
LEGACY_CHECKPOINT_ADOPTION_ENTRYPOINT = (
    "feedbax.training.legacy_checkpoint_adoption.adopt_legacy_checkpoint"
)
LEGACY_CHECKPOINT_ADOPTION_DOCS = "docs/structure.md#legacy-checkpoint-adoption"
_RUN_CONTRACT_BINDING_ALGORITHM_V2 = "feedbax.training_checkpoint.run_contract_binding.v2"
_RUN_CONTRACT_BINDING_ALGORITHM_V3 = "feedbax.training_checkpoint.run_contract_binding.v3"
_RUN_CONTRACT_HASH_DOMAIN = "migrated-canonical-json"
_CONTENT_INTEGRITY_SCHEMA_ID = ContentIntegrityDigest.model_fields["schema_id"].default
_CONTENT_INTEGRITY_SCHEMA_VERSION = ContentIntegrityDigest.model_fields["schema_version"].default
_STRUCTURAL_ABI_SCHEMA_ID = StructuralAbiFingerprint.model_fields["schema_id"].default
_STRUCTURAL_ABI_SCHEMA_VERSION = StructuralAbiFingerprint.model_fields["schema_version"].default
_STRUCTURAL_ABI_ALGORITHM_VERSION = StructuralAbiFingerprint.model_fields[
    "fingerprint_algorithm_version"
].default
_READ_ONLY_MODEL_TYPES: dict[type[BaseModel], type[BaseModel]] = {}


def _reject_snapshot_mutation(*_args: Any, **_kwargs: Any) -> None:
    raise TypeError("verified checkpoint lineage snapshots are immutable")


class _FrozenDict(dict[Any, Any]):
    """A serialization-compatible immutable dictionary snapshot."""

    __setitem__ = _reject_snapshot_mutation
    __delitem__ = _reject_snapshot_mutation
    clear = _reject_snapshot_mutation
    pop = _reject_snapshot_mutation
    popitem = _reject_snapshot_mutation
    setdefault = _reject_snapshot_mutation
    update = _reject_snapshot_mutation
    __ior__ = _reject_snapshot_mutation


class _FrozenList(list[Any]):
    """A serialization-compatible immutable list snapshot."""

    __setitem__ = _reject_snapshot_mutation
    __delitem__ = _reject_snapshot_mutation
    append = _reject_snapshot_mutation
    clear = _reject_snapshot_mutation
    extend = _reject_snapshot_mutation
    insert = _reject_snapshot_mutation
    pop = _reject_snapshot_mutation
    remove = _reject_snapshot_mutation
    reverse = _reject_snapshot_mutation
    sort = _reject_snapshot_mutation
    __iadd__ = _reject_snapshot_mutation
    __imul__ = _reject_snapshot_mutation


class CheckpointCustodyError(ValueError):
    """Base class for checkpoint custody failures."""


def _validate_program_step_units(
    coordinate: ProgressCoordinate,
    metadata: Mapping[str, Any],
    *,
    context: str,
) -> None:
    """Validate the barrier visit ordinal without conflating coordinate units."""
    raw_ordinal = metadata.get("barrier_visit_ordinal")
    if raw_ordinal is None:
        return
    if isinstance(raw_ordinal, bool) or not isinstance(raw_ordinal, int) or raw_ordinal < 0:
        raise CheckpointConsistencyError(
            f"{context} /metadata/barrier_visit_ordinal must be a non-negative integer"
        )
    if coordinate.program_step < 0:
        raise CheckpointConsistencyError(
            f"{context} /completed_coordinate/program_step must be non-negative"
        )


class CheckpointIntegrityError(CheckpointCustodyError):
    """Raised when checkpoint bytes or manifests fail integrity validation."""


class CheckpointReferenceResolutionError(CheckpointIntegrityError):
    """Raised when a checkpoint-custody ``ParentRef`` cannot be verified."""


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


def _resolved_slot_axes(
    run_spec: TrainingRunSpec,
) -> dict[str, tuple[MaterializedSlotAxisBinding, ...]]:
    """Return target-derived axis evidence, omitting scalar declarations."""
    levels, bindings = resolve_execution_mapping(run_spec.worker_execution)
    return dict(bindings) if levels else {}


def _checkpoint_coordinate(
    coordinate: ProgressCoordinate,
    axes: Mapping[str, tuple[MaterializedSlotAxisBinding, ...]],
) -> ProgressCoordinate:
    payload, _ = normalize_serialized_metrics(coordinate, coordinate.metrics, axes)
    return ProgressCoordinate.model_validate(payload)


def _validate_slot_axes(
    slot: str,
    value: Any,
    axes: tuple[MaterializedSlotAxisBinding, ...] | None,
    *,
    error_cls: type[CheckpointCustodyError],
) -> None:
    """Validate actual dynamic leaves against one resolved slot-axis record."""
    if axes is None:
        return
    mapped = [axis for axis in axes if axis.mode == "mapped"]
    if not mapped:
        return
    arrays = [leaf for leaf in jt.leaves(value) if eqx.is_array(leaf)]
    if not arrays:
        raise error_cls(f"checkpoint slot {slot!r} has mapped axes but no dynamic array leaves")
    for axis in mapped:
        assert axis.array_axis is not None
        for index, leaf in enumerate(arrays):
            shape = np.shape(leaf)
            if len(shape) <= axis.array_axis or shape[axis.array_axis] != axis.size:
                raise error_cls(
                    f"checkpoint slot {slot!r} leaf {index} does not materialize axis "
                    f"{axis.axis!r} at position {axis.array_axis} with size {axis.size}; "
                    f"shape={shape!r}"
                )


def _validate_recorded_slot_axes(
    manifest: CheckpointTransactionManifest,
    expected_axes: Mapping[str, tuple[MaterializedSlotAxisBinding, ...]],
    values: Mapping[str, Any] | None = None,
) -> None:
    provenance = manifest.fork_provenance
    if provenance is not None:
        target_records = {record.slot: record.materialized_axes for record in manifest.slots}
        provenance_names = [record.slot for record in provenance.slots]
        if len(provenance_names) != len(set(provenance_names)):
            raise CheckpointCompatibilityError("fork provenance contains duplicate slot records")
        if set(provenance_names) != set(target_records):
            raise CheckpointCompatibilityError(
                "fork provenance slot records are missing or extra; "
                f"recorded={sorted(provenance_names)!r} target={sorted(target_records)!r}"
            )
        for record in provenance.slots:
            target = target_records[record.slot]
            if record.target_axes != target:
                raise CheckpointCompatibilityError(
                    f"fork provenance target axes mismatch for slot {record.slot!r}"
                )
            has_sha = record.source_sha256 is not None
            has_path = record.source_relative_path is not None
            if has_sha != has_path:
                raise CheckpointCompatibilityError(
                    f"fork provenance source record is partial for slot {record.slot!r}"
                )
            if has_sha and record.source_axes != record.target_axes:
                raise CheckpointCompatibilityError(
                    f"fork provenance cannot remap axes for slot {record.slot!r}"
                )
            if not has_sha and record.source_axes is not None:
                raise CheckpointCompatibilityError(
                    f"target-only fork slot {record.slot!r} must not record source axes"
                )
    for record in manifest.slots:
        target = expected_axes.get(record.slot)
        if record.materialized_axes != target:
            raise CheckpointCompatibilityError(
                f"checkpoint slot {record.slot!r} mapped-axis evidence mismatch; "
                f"recorded={record.materialized_axes!r} target={target!r}"
            )
        if values is not None and record.slot in values:
            _validate_slot_axes(
                record.slot,
                values[record.slot],
                target,
                error_cls=CheckpointCompatibilityError,
            )


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


CheckpointForkPlanTransform = Callable[
    [Mapping[str, Any], Mapping[str, Any]],
    Mapping[str, Any],
]


@dataclass(frozen=True)
class CheckpointForkTransformRegistration:
    """Runtime implementation for one durable fork-transform identity."""

    identity: str
    transform: CheckpointForkPlanTransform
    owner: str = "feedbax"


class CheckpointForkTransformRegistry:
    """Exact runtime registry for durable checkpoint-fork transforms."""

    def __init__(self) -> None:
        self._registrations: dict[str, CheckpointForkTransformRegistration] = {}

    def register(self, registration: CheckpointForkTransformRegistration) -> None:
        if not registration.identity:
            raise ValueError("checkpoint fork transform identity must not be empty")
        if not callable(registration.transform):
            raise TypeError(f"checkpoint fork transform {registration.identity!r} is not callable")
        if registration.identity in self._registrations:
            existing = self._registrations[registration.identity]
            raise ValueError(
                f"checkpoint fork transform {registration.identity!r} already registered "
                f"by {existing.owner!r}"
            )
        self._registrations[registration.identity] = registration

    def resolve(self, identity: str) -> CheckpointForkTransformRegistration:
        try:
            return self._registrations[identity]
        except KeyError as exc:
            raise CheckpointCompatibilityError(
                f"unregistered checkpoint fork transform {identity!r}; "
                f"available identities={list(self.available_keys())!r}"
            ) from exc

    def available_keys(self) -> tuple[str, ...]:
        """Return registered durable identities in deterministic order."""
        return tuple(sorted(self._registrations))


DEFAULT_CHECKPOINT_FORK_TRANSFORM_REGISTRY = CheckpointForkTransformRegistry()


@dataclass(frozen=True)
class CheckpointForkPlanBindings:
    """Runtime-only paths, run specs, and PyTree templates for a fork plan."""

    checkpoint_roots: Mapping[str, str | Path]
    run_specs: Mapping[str, TrainingRunSpec]
    slot_templates: Mapping[str, Mapping[str, Any]]
    segment_history_templates: Mapping[str, Mapping[str, Any]] = dataclass_field(
        default_factory=dict
    )
    population_member_ids: Mapping[str, Mapping[str, Sequence[str]]] = dataclass_field(
        default_factory=dict
    )


@dataclass(frozen=True)
class _PreparedForkPlanTarget:
    target_id: str
    target_root: Path
    run_spec: TrainingRunSpec
    phase_program: PhaseProgramSpec
    expected_slots: Mapping[str, Any]
    prepared_slots: Mapping[str, Any]
    segment_history_templates: Mapping[str, Any] | None
    continuation_request: CheckpointContinuationRequest | None
    continuation_applied: bool
    barrier_mapping: CheckpointForkBarrierMapping | None
    target_coordinate: ProgressCoordinate | None
    transformed_slots: frozenset[str]
    target_only_slots: Mapping[str, Mapping[str, Any]]
    transform_records: Mapping[str, tuple[CheckpointForkTransformRecord, ...]]
    population_member_ids: Mapping[str, Sequence[str]]


@dataclass(frozen=True)
class ConcatenatedCheckpointHistories:
    """Pure read result for a validated terminal segment lineage."""

    histories: Mapping[str, BatchHistory[Any]]
    transaction_ids: tuple[str, ...]
    completed_training_batches: int


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
class CheckpointCustodyDocuments:
    """Latest-pointer and transaction-manifest documents resolved under one root.

    Each document result retains whether its bytes were already current or
    required registered migration before strict model validation.
    """

    root: Path
    latest_pointer: CheckpointDocumentLoadResult[CheckpointLatestPointer]
    manifest_path: Path
    manifest: CheckpointDocumentLoadResult[CheckpointTransactionManifest]


@dataclass(frozen=True)
class ResolvedCheckpointTransaction:
    """Authenticated checkpoint transaction and its decoded requested slots.

    The manifest remains the lineage authority for transaction, checkpoint,
    content-root, slot-digest, and structural-ABI identities. Decoding uses
    pickle only after the caller-provided custody authority, manifest bytes,
    contained blob paths, and blob hashes have been authenticated. Therefore
    ``allowed_root`` must designate a trusted custody authority: hashes and
    containment do not make attacker-authored pickle safe to deserialize.
    """

    parent_ref: ParentRef
    manifest_sha256: str
    manifest: CheckpointTransactionManifest
    slots: Mapping[str, Any]
    migration_records: tuple[ArtifactMigrationRecord, ...]
    provenance_notices: tuple[CheckpointProvenanceNotice, ...]


@dataclass(frozen=True)
class CheckpointCustodyArchiveEvidence:
    """Authenticated identities and exact sizes bound into a custody archive."""

    schema_id: str
    schema_version: str
    media_type: str
    parent_ref: ParentRef
    transaction_root_sha256: str
    payload_member_count: int
    expanded_payload_size_bytes: int
    archive_sha256: str
    archive_size_bytes: int


@dataclass(frozen=True)
class CheckpointCustodyArchiveResult:
    """Immutable provider reference and evidence for canonical archive bytes."""

    artifact_ref: ArtifactRef
    evidence: CheckpointCustodyArchiveEvidence


@dataclass(frozen=True)
class MaterializedCheckpointCustodyArchive:
    """Authenticated archive materialization and resolved transaction evidence."""

    artifact_ref: ArtifactRef
    archive_evidence: CheckpointCustodyArchiveEvidence
    destination: Path
    manifest_sha256: str
    resolved_transaction: ResolvedCheckpointTransaction


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


def load_checkpoint_latest_pointer_json(
    data: bytes | str,
    *,
    path: str = "checkpoint_latest_pointer",
) -> CheckpointDocumentLoadResult[CheckpointLatestPointer]:
    """Load a latest-pointer JSON document through its registered migrations."""
    return _load_checkpoint_document_json(
        data,
        kind="TrainingCheckpointLatestPointer",
        model=CheckpointLatestPointer,
        path=path,
        document_name="checkpoint latest pointer",
    )


def load_checkpoint_latest_pointer_file(
    path: str | Path,
) -> CheckpointDocumentLoadResult[CheckpointLatestPointer]:
    """Load a latest-pointer file through its registered migrations."""
    path_obj = Path(path)
    try:
        return load_checkpoint_latest_pointer_json(
            path_obj.read_bytes(),
            path=str(path_obj),
        )
    except OSError as exc:
        raise CheckpointIntegrityError(
            f"checkpoint latest pointer could not be read: {path_obj}"
        ) from exc


def load_checkpoint_transaction_manifest_json(
    data: bytes | str,
    *,
    path: str = "checkpoint_transaction_manifest",
) -> CheckpointDocumentLoadResult[CheckpointTransactionManifest]:
    """Load a transaction-manifest JSON document through registered migrations."""
    return _load_checkpoint_document_json(
        data,
        kind="TrainingCheckpointTransactionManifest",
        model=CheckpointTransactionManifest,
        path=path,
        document_name="checkpoint transaction manifest",
    )


def load_checkpoint_transaction_manifest_file(
    path: str | Path,
) -> CheckpointDocumentLoadResult[CheckpointTransactionManifest]:
    """Load a transaction-manifest file through its registered migrations."""
    path_obj = Path(path)
    try:
        return load_checkpoint_transaction_manifest_json(
            path_obj.read_bytes(),
            path=str(path_obj),
        )
    except OSError as exc:
        raise CheckpointIntegrityError(
            f"checkpoint transaction manifest could not be read: {path_obj}"
        ) from exc


def load_checkpoint_custody_documents(root: str | Path) -> CheckpointCustodyDocuments:
    """Load the published latest pointer and its contained manifest safely.

    The pointer's manifest path is required to remain under ``root``; a
    traversal or absolute path is rejected before the manifest is opened.
    """
    root_path = Path(root).resolve()
    latest = load_checkpoint_latest_pointer_file(root_path / LATEST_POINTER_NAME)
    manifest_path = _resolve_latest_manifest_path(root_path, latest.document)
    manifest = load_checkpoint_transaction_manifest_file(manifest_path)
    return CheckpointCustodyDocuments(
        root=root_path,
        latest_pointer=latest,
        manifest_path=manifest_path,
        manifest=manifest,
    )


def resolve_checkpoint_custody_ref(
    ref: ParentRef,
    *,
    allowed_root: str | Path,
    slot_names: Collection[str] | None = None,
) -> ResolvedCheckpointTransaction:
    """Resolve an emitted checkpoint-custody reference without resume semantics.

    ``slot_names=None`` decodes every manifest slot. Explicit names are exact
    and case-sensitive. The function never selects a different checkpoint and
    never requires or applies caller templates, a training run spec, a phase
    program, continuation behavior, or resume-only slot migrations.

    Warning:
        Slot blobs use pickle. ``allowed_root`` must be a trusted checkpoint
        custody authority. Manifest and blob authentication prevents unnoticed
        modification but cannot make attacker-authored pickle safe.
    """
    try:
        return _resolve_checkpoint_custody_ref(
            ref,
            allowed_root=allowed_root,
            slot_names=slot_names,
        )
    except CheckpointReferenceResolutionError:
        raise
    except Exception as exc:
        raise CheckpointReferenceResolutionError(
            f"checkpoint custody reference resolution failed: {exc}"
        ) from exc


def produce_checkpoint_custody_archive(
    ref: ParentRef,
    *,
    allowed_root: str | Path,
    artifact_provider: ImmutableArtifactBlobProvider,
) -> CheckpointCustodyArchiveResult:
    """Validate and store the canonical v1 archive for one published transaction."""
    resolved = resolve_checkpoint_custody_ref(ref, allowed_root=allowed_root)
    authenticated_ref = resolved.parent_ref
    root = Path(allowed_root).expanduser().resolve()
    latest_path = root / LATEST_POINTER_NAME
    latest_bytes = _read_archive_source(latest_path, context="checkpoint latest pointer")
    try:
        loaded_latest = load_checkpoint_latest_pointer_json(latest_bytes, path=str(latest_path))
    except CheckpointIntegrityError as exc:
        raise CheckpointReferenceResolutionError(str(exc)) from exc
    manifest_path = _resolve_latest_manifest_path(root, loaded_latest.document)
    manifest_bytes = _read_archive_source(manifest_path, context="checkpoint manifest")
    try:
        loaded_manifest = load_checkpoint_transaction_manifest_json(
            manifest_bytes,
            path=str(manifest_path),
        )
    except CheckpointIntegrityError as exc:
        raise CheckpointReferenceResolutionError(str(exc)) from exc
    if resolved.migration_records or loaded_latest.migrated or loaded_manifest.migrated:
        raise CheckpointReferenceResolutionError(
            "checkpoint archive source documents must already use current schemas"
        )
    latest = loaded_latest.document
    transaction_root = resolved.manifest.content_integrity_digest.transaction_root_sha256
    if (
        latest.transaction_id != authenticated_ref.id
        or latest.manifest_relative_path != authenticated_ref.uri
        or latest.manifest_sha256 != resolved.manifest_sha256
        or latest.transaction_root_sha256 != transaction_root
    ):
        raise CheckpointReferenceResolutionError(
            "checkpoint latest pointer does not select the authenticated ParentRef transaction"
        )
    manifest_name = _canonical_archive_relative_path(
        authenticated_ref.uri,
        context="ParentRef uri",
    )
    if manifest_path != (root / Path(*PurePosixPath(manifest_name).parts)).resolve():
        raise CheckpointReferenceResolutionError(
            "checkpoint latest pointer manifest path differs from authenticated ParentRef"
        )

    payload: list[tuple[str, Path, bytes]] = []
    payload.append((f"checkpoint/{LATEST_POINTER_NAME}", latest_path, latest_bytes))
    if sha256_bytes(manifest_bytes) != resolved.manifest_sha256:
        raise CheckpointReferenceResolutionError(
            "checkpoint manifest changed after authenticated resolution"
        )
    payload.append((f"checkpoint/{manifest_name}", manifest_path, manifest_bytes))
    transaction_dir = manifest_path.parent.resolve()
    for slot in resolved.manifest.slots:
        slot_name = _canonical_archive_relative_path(
            slot.relative_path,
            context=f"checkpoint slot {slot.slot!r} relative_path",
        )
        blob_path = _resolve_checkpoint_slot_path(
            slot,
            transaction_dir=transaction_dir,
            allowed_root=root,
        )
        try:
            blob_bytes = _read_blob(slot, blob_path)
        except (CheckpointIntegrityError, OSError) as exc:
            raise CheckpointReferenceResolutionError(str(exc)) from exc
        payload.append(
            (f"checkpoint/{PurePosixPath(manifest_name).parent / slot_name}", blob_path, blob_bytes)
        )
    names = [name for name, _, _ in payload]
    if len(names) != len(set(names)):
        raise CheckpointReferenceResolutionError("checkpoint archive member names are not unique")

    payload_size = sum(len(data) for _, _, data in payload)
    archive_document = canonical_json_bytes(
        {
            "schema_id": CHECKPOINT_CUSTODY_ARCHIVE_SCHEMA_ID,
            "schema_version": CHECKPOINT_CUSTODY_ARCHIVE_SCHEMA_VERSION,
            "media_type": CHECKPOINT_CUSTODY_ARCHIVE_MEDIA_TYPE,
            "parent_ref": authenticated_ref.model_dump(mode="json", exclude_none=True),
            "transaction_root_sha256": transaction_root,
            "payload_member_count": len(payload),
            "expanded_payload_size_bytes": payload_size,
        }
    )
    archive_bytes = _checkpoint_custody_archive_bytes(archive_document, payload)
    for _, path, expected in payload:
        try:
            current = path.read_bytes()
        except OSError as exc:
            raise CheckpointReferenceResolutionError(
                f"checkpoint archive source disappeared before storage: {path}"
            ) from exc
        if current != expected:
            raise CheckpointReferenceResolutionError(
                f"checkpoint archive source changed before storage: {path}"
            )
    artifact = artifact_provider.store_bytes(
        archive_bytes,
        role="training_checkpoint_custody_archive",
        logical_name=f"{resolved.manifest.transaction_id}.checkpoint-custody.tar.gz",
        media_type=CHECKPOINT_CUSTODY_ARCHIVE_MEDIA_TYPE,
        metadata={
            "schema_id": CHECKPOINT_CUSTODY_ARCHIVE_SCHEMA_ID,
            "schema_version": CHECKPOINT_CUSTODY_ARCHIVE_SCHEMA_VERSION,
            "transaction_root_sha256": transaction_root,
        },
    )
    evidence = CheckpointCustodyArchiveEvidence(
        schema_id=CHECKPOINT_CUSTODY_ARCHIVE_SCHEMA_ID,
        schema_version=CHECKPOINT_CUSTODY_ARCHIVE_SCHEMA_VERSION,
        media_type=CHECKPOINT_CUSTODY_ARCHIVE_MEDIA_TYPE,
        parent_ref=authenticated_ref,
        transaction_root_sha256=transaction_root,
        payload_member_count=len(payload),
        expanded_payload_size_bytes=payload_size,
        archive_sha256=sha256_bytes(archive_bytes),
        archive_size_bytes=len(archive_bytes),
    )
    return CheckpointCustodyArchiveResult(
        artifact_ref=_immutable_model_snapshot(artifact),
        evidence=evidence,
    )


def materialize_checkpoint_custody_archive(
    artifact_provider: ImmutableArtifactBlobProvider,
    artifact_ref: ArtifactRef,
    destination: str | Path,
    *,
    expected_parent_ref: ParentRef,
    expected_transaction_root_sha256: str,
) -> MaterializedCheckpointCustodyArchive:
    """Authenticate, validate, and atomically publish one canonical v1 archive.

    ``expected_parent_ref`` and ``expected_transaction_root_sha256`` are trusted
    out-of-band authorities supplied by an authenticated caller or control plane.
    In particular, ``expected_parent_ref.metadata["manifest_sha256"]`` must never
    be derived from the hostile archive bytes being validated.
    """
    archive_bytes = artifact_provider.get_bytes(artifact_ref)
    if artifact_ref.media_type != CHECKPOINT_CUSTODY_ARCHIVE_MEDIA_TYPE:
        raise CheckpointReferenceResolutionError("checkpoint archive media type mismatch")
    raw_destination = os.fspath(destination)
    target = Path(raw_destination)
    if (
        "\x00" in raw_destination
        or not target.is_absolute()
        or ".." in target.parts
        or not target.name
    ):
        raise CheckpointReferenceResolutionError(
            "checkpoint archive destination must be an absolute canonical path"
        )
    parent = target.parent
    try:
        canonical_parent = parent.resolve(strict=True)
    except OSError as exc:
        raise CheckpointReferenceResolutionError(
            f"checkpoint archive destination parent is unavailable: {parent}"
        ) from exc
    if canonical_parent != parent or not parent.is_dir():
        raise CheckpointReferenceResolutionError(
            "checkpoint archive destination parent must be a real canonical directory"
        )
    parent_descriptor: int | None = None
    staging_descriptor: int | None = None
    staging_identity: os.stat_result | None = None
    directory_identities: dict[tuple[str, ...], tuple[int, int]] = {}
    try:
        directory_flags = _archive_directory_flags()
        parent_descriptor = os.open(parent, directory_flags)
        parent_identity = os.fstat(parent_descriptor)
        _require_external_archive_mapping(parent, parent_identity)
        try:
            os.stat(target.name, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(f"checkpoint archive destination already exists: {target}")

        staging_name = f".{target.name}.checkpoint-archive-{uuid.uuid4().hex}"
        os.mkdir(staging_name, mode=0o700, dir_fd=parent_descriptor)
        staging_identity = os.stat(
            staging_name, dir_fd=parent_descriptor, follow_symlinks=False
        )
        staging_descriptor = os.open(staging_name, directory_flags, dir_fd=parent_descriptor)
        opened_identity = os.fstat(staging_descriptor)
        current_identity = os.stat(
            staging_name, dir_fd=parent_descriptor, follow_symlinks=False
        )
        if not stat.S_ISDIR(current_identity.st_mode) or not (
            _identity(staging_identity)
            == _identity(opened_identity)
            == _identity(current_identity)
        ):
            raise CheckpointReferenceResolutionError(
                "checkpoint archive private staging identity changed during creation"
            )
        staging = parent / staging_name
        document, payload_names, payload_size = _extract_checkpoint_custody_archive(
            archive_bytes, staging_descriptor, directory_identities
        )
        expected_document = {
            "schema_id": CHECKPOINT_CUSTODY_ARCHIVE_SCHEMA_ID,
            "schema_version": CHECKPOINT_CUSTODY_ARCHIVE_SCHEMA_VERSION,
            "media_type": CHECKPOINT_CUSTODY_ARCHIVE_MEDIA_TYPE,
            "parent_ref": expected_parent_ref.model_dump(mode="json", exclude_none=True),
            "transaction_root_sha256": expected_transaction_root_sha256,
            "payload_member_count": len(payload_names),
            "expanded_payload_size_bytes": payload_size,
        }
        if document != expected_document:
            raise CheckpointReferenceResolutionError(
                "checkpoint archive document identity, schema, count, or size mismatch"
            )
        _require_external_archive_mapping(
            parent, parent_identity, child=staging, child_identity=staging_identity
        )
        loaded_latest = load_checkpoint_latest_pointer_file(staging / LATEST_POINTER_NAME)
        if loaded_latest.migrated:
            raise CheckpointReferenceResolutionError(
                "checkpoint archive documents must already use current schemas"
            )
        latest = loaded_latest.document
        if (
            latest.transaction_id != expected_parent_ref.id
            or latest.manifest_relative_path != expected_parent_ref.uri
            or latest.manifest_sha256
            != expected_parent_ref.metadata.get("manifest_sha256")
            or latest.transaction_root_sha256 != expected_transaction_root_sha256
        ):
            raise CheckpointReferenceResolutionError(
                "checkpoint archive latest pointer is stale or selects another transaction"
            )
        resolved = resolve_checkpoint_custody_ref(expected_parent_ref, allowed_root=staging)
        if resolved.migration_records:
            raise CheckpointReferenceResolutionError(
                "checkpoint archive documents must already use current schemas"
            )
        manifest = resolved.manifest
        expected_names = {
            f"checkpoint/{LATEST_POINTER_NAME}",
            f"checkpoint/{_canonical_archive_relative_path(expected_parent_ref.uri, context='ParentRef uri')}",
            *(
                f"checkpoint/{PurePosixPath(expected_parent_ref.uri).parent / _canonical_archive_relative_path(slot.relative_path, context='slot relative_path')}"
                for slot in manifest.slots
            ),
        }
        if payload_names != expected_names:
            raise CheckpointReferenceResolutionError(
                "checkpoint archive contains missing or unexpected governed members"
            )
        if (
            resolved.parent_ref != expected_parent_ref
            or manifest.content_integrity_digest.transaction_root_sha256
            != expected_transaction_root_sha256
        ):
            raise CheckpointReferenceResolutionError(
                "checkpoint archive resolved transaction identity mismatch"
            )
        _require_external_archive_mapping(
            parent, parent_identity, child=staging, child_identity=staging_identity
        )
        evidence = CheckpointCustodyArchiveEvidence(
            schema_id=document["schema_id"],
            schema_version=document["schema_version"],
            media_type=document["media_type"],
            parent_ref=_immutable_model_snapshot(expected_parent_ref),
            transaction_root_sha256=expected_transaction_root_sha256,
            payload_member_count=len(payload_names),
            expanded_payload_size_bytes=payload_size,
            archive_sha256=artifact_ref.sha256,
            archive_size_bytes=artifact_ref.size_bytes,
        )
        result = MaterializedCheckpointCustodyArchive(
            artifact_ref=_immutable_model_snapshot(artifact_ref),
            archive_evidence=evidence,
            destination=target,
            manifest_sha256=resolved.manifest_sha256,
            resolved_transaction=resolved,
        )
        _publish_directory_no_replace(
            parent_descriptor,
            staging_name,
            target.name,
            expected_identity=staging_identity,
        )
        return result
    finally:
        if staging_descriptor is not None:
            try:
                os.close(staging_descriptor)
            except OSError:
                pass
        if parent_descriptor is not None:
            try:
                os.close(parent_descriptor)
            except OSError:
                pass


def _extract_checkpoint_custody_archive(
    archive_bytes: bytes,
    staging_descriptor: int,
    directory_identities: dict[tuple[str, ...], tuple[int, int]],
) -> tuple[dict[str, Any], set[str], int]:
    """Stream canonical regular members into private staging."""
    document: dict[str, Any] | None = None
    names: set[str] = set()
    folded_names: set[str] = set()
    payload_size = 0
    expected_offset = 0
    framing = _CanonicalUstarFraming()
    compressed_stream = _SingleGzipMemberStream(archive_bytes, framing)
    try:
        archive = tarfile.open(fileobj=compressed_stream, mode="r|")
        with archive:
            for index, member in enumerate(archive):
                name = _canonical_archive_relative_path(member.name, context="archive member")
                if (
                    member.offset != expected_offset
                    or member.offset_data - member.offset != tarfile.BLOCKSIZE
                    or member.type != tarfile.REGTYPE
                    or member.pax_headers
                    or getattr(member, "sparse", None)
                    or (member.mode, member.uid, member.gid, member.mtime, member.uname, member.gname)
                    != (0o644, 0, 0, 0, "", "")
                    or name in names
                    or name.casefold() in folded_names
                ):
                    raise CheckpointReferenceResolutionError(
                        f"checkpoint archive member is unsafe or duplicated: {name!r}"
                    )
                expected_offset = member.offset_data + (
                    (member.size + tarfile.BLOCKSIZE - 1) // tarfile.BLOCKSIZE
                ) * tarfile.BLOCKSIZE
                framing.require_member(index, member)
                names.add(name)
                folded_names.add(name.casefold())
                source = archive.extractfile(member)
                if source is None:
                    raise CheckpointReferenceResolutionError(
                        f"checkpoint archive member cannot be read: {name!r}"
                    )
                if index == 0:
                    if name != "archive.json":
                        raise CheckpointReferenceResolutionError(
                            "checkpoint archive must begin with archive.json"
                        )
                    raw_document = source.read(member.size + 1)
                    try:
                        parsed = json.loads(raw_document)
                    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                        raise CheckpointReferenceResolutionError(
                            "checkpoint archive document is invalid JSON"
                        ) from exc
                    if not isinstance(parsed, dict) or raw_document != canonical_json_bytes(parsed):
                        raise CheckpointReferenceResolutionError(
                            "checkpoint archive document is not exact canonical JSON"
                        )
                    document = parsed
                    continue
                if not name.startswith("checkpoint/"):
                    raise CheckpointReferenceResolutionError(
                        f"checkpoint archive contains unexpected member: {name!r}"
                    )
                relative = PurePosixPath(name).relative_to("checkpoint")
                descriptor = _open_archive_member(
                    staging_descriptor, relative.parts, directory_identities
                )
                try:
                    sink = os.fdopen(descriptor, "wb")
                except BaseException:
                    os.close(descriptor)
                    raise
                with sink:
                    while chunk := source.read(1024 * 1024):
                        sink.write(chunk)
                    sink.flush()
                    written_size = os.fstat(sink.fileno()).st_size
                if written_size != member.size:
                    raise CheckpointReferenceResolutionError(
                        f"checkpoint archive member size mismatch: {name!r}"
                    )
                payload_size += member.size
        compressed_stream.finish()
    except (tarfile.TarError, OSError, EOFError) as exc:
        raise CheckpointReferenceResolutionError(
            f"checkpoint archive tar stream is invalid: {exc}"
        ) from exc
    if document is None:
        raise CheckpointReferenceResolutionError("checkpoint archive document is missing")
    names.remove("archive.json")
    return document, names, payload_size


class _CanonicalUstarFraming:
    """Validate canonical USTAR framing without retaining payload bytes."""

    def __init__(self) -> None:
        self._header = bytearray()
        self._remaining_data = 0
        self._remaining_padding = 0
        self._eof_offset: int | None = None
        self._expected_size: int | None = None
        self._offset = 0
        self._members: list[tuple[int, str, int]] = []

    def feed(self, data: bytes) -> None:
        view = memoryview(data)
        position = 0
        while position < len(view):
            if self._eof_offset is not None:
                chunk = view[position:]
                if any(chunk):
                    raise CheckpointReferenceResolutionError(
                        "checkpoint archive has noncanonical records after logical tar EOF"
                    )
                self._offset += len(chunk)
                return
            if self._remaining_data:
                consumed = min(self._remaining_data, len(view) - position)
                self._remaining_data -= consumed
                self._offset += consumed
                position += consumed
                continue
            if self._remaining_padding:
                consumed = min(self._remaining_padding, len(view) - position)
                padding = view[position : position + consumed]
                if any(padding):
                    raise CheckpointReferenceResolutionError(
                        "checkpoint archive member padding is not canonical"
                    )
                self._remaining_padding -= consumed
                self._offset += consumed
                position += consumed
                continue
            consumed = min(tarfile.BLOCKSIZE - len(self._header), len(view) - position)
            self._header.extend(view[position : position + consumed])
            self._offset += consumed
            position += consumed
            if len(self._header) != tarfile.BLOCKSIZE:
                continue
            header_offset = self._offset - tarfile.BLOCKSIZE
            header = bytes(self._header)
            self._header.clear()
            if not any(header):
                self._eof_offset = header_offset
                minimum = header_offset + 2 * tarfile.BLOCKSIZE
                self._expected_size = (
                    (minimum + tarfile.RECORDSIZE - 1) // tarfile.RECORDSIZE
                ) * tarfile.RECORDSIZE
                continue
            name, size = _validate_canonical_ustar_header(header)
            self._members.append((header_offset, name, size))
            self._remaining_data = size
            self._remaining_padding = (-size) % tarfile.BLOCKSIZE

    def require_member(self, index: int, member: tarfile.TarInfo) -> None:
        try:
            offset, name, size = self._members[index]
        except IndexError as exc:
            raise CheckpointReferenceResolutionError(
                "checkpoint archive USTAR framing does not match parsed members"
            ) from exc
        if (offset, name, size) != (member.offset, member.name, member.size):
            raise CheckpointReferenceResolutionError(
                "checkpoint archive USTAR framing does not match parsed members"
            )

    def finish(self) -> None:
        if (
            self._header
            or self._remaining_data
            or self._remaining_padding
            or self._eof_offset is None
            or self._expected_size != self._offset
        ):
            raise CheckpointReferenceResolutionError(
                "checkpoint archive tar stream has noncanonical termination"
            )


def _validate_canonical_ustar_header(header: bytes) -> tuple[str, int]:
    """Return the exact USTAR member name and size for one canonical header."""

    def text(field: bytes) -> str:
        value, separator, padding = field.partition(b"\0")
        if separator and any(padding):
            raise CheckpointReferenceResolutionError(
                "checkpoint archive USTAR text field is not canonical"
            )
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise CheckpointReferenceResolutionError(
                "checkpoint archive USTAR member name is not UTF-8"
            ) from exc

    try:
        size = int(header[124:135], 8)
    except ValueError as exc:
        raise CheckpointReferenceResolutionError(
            "checkpoint archive USTAR size field is invalid"
        ) from exc
    checksum_header = header[:148] + b" " * 8 + header[156:]
    checksum = sum(checksum_header)
    expected_fields = (
        (header[100:108], b"0000644\0"),
        (header[108:116], b"0000000\0"),
        (header[116:124], b"0000000\0"),
        (header[124:136], f"{size:011o}\0".encode("ascii")),
        (header[136:148], b"00000000000\0"),
        (header[148:156], f"{checksum:06o}\0 ".encode("ascii")),
        (header[156:157], tarfile.REGTYPE),
        (header[157:257], b"\0" * 100),
        (header[257:263], b"ustar\0"),
        (header[263:265], b"00"),
        (header[265:329], b"\0" * 64),
        (header[329:345], b"\0" * 16),
        (header[500:512], b"\0" * 12),
    )
    if any(actual != expected for actual, expected in expected_fields):
        raise CheckpointReferenceResolutionError(
            "checkpoint archive USTAR header is unsafe or noncanonical"
        )
    name = text(header[:100])
    prefix = text(header[345:500])
    return (f"{prefix}/{name}" if prefix else name), size


class _SingleGzipMemberStream:
    """Incrementally decode exactly one gzip member into a framing validator."""

    def __init__(self, compressed: bytes, framing: _CanonicalUstarFraming) -> None:
        self._compressed = compressed
        self._framing = framing
        self._decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
        self._input_offset = 0
        self._pending = b""
        self._output = bytearray()
        self._finished = False

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            raise CheckpointReferenceResolutionError(
                "checkpoint archive decoder requires bounded streaming reads"
            )
        while len(self._output) < size and not self._finished:
            self._pump()
        result = bytes(self._output[:size])
        del self._output[:size]
        return result

    def _pump(self) -> None:
        if self._decoder.eof:
            if (
                self._decoder.unused_data
                or self._pending
                or self._input_offset != len(self._compressed)
            ):
                raise CheckpointReferenceResolutionError(
                    "checkpoint archive has concatenated or trailing compressed bytes"
                )
            self._finished = True
            return
        if not self._pending and self._input_offset < len(self._compressed):
            end = min(self._input_offset + 64 * 1024, len(self._compressed))
            self._pending = self._compressed[self._input_offset : end]
            self._input_offset = end
        if not self._pending:
            raise CheckpointReferenceResolutionError(
                "checkpoint archive gzip member is incomplete"
            )
        try:
            output = self._decoder.decompress(self._pending, 1024 * 1024)
        except zlib.error as exc:
            raise CheckpointReferenceResolutionError(
                f"checkpoint archive gzip member is invalid: {exc}"
            ) from exc
        self._pending = self._decoder.unconsumed_tail
        if output:
            self._framing.feed(output)
            self._output.extend(output)

    def finish(self) -> None:
        self._output.clear()
        while not self._finished:
            self._pump()
            self._output.clear()
        self._framing.finish()


def _archive_directory_flags() -> int:
    required = ("O_DIRECTORY", "O_NOFOLLOW")
    if any(not getattr(os, name, 0) for name in required):
        raise CheckpointReferenceResolutionError(
            "descriptor-relative no-follow archive operations are unsupported"
        )
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)


def _open_archive_member(
    staging_descriptor: int,
    parts: tuple[str, ...],
    directory_identities: dict[tuple[str, ...], tuple[int, int]],
) -> int:
    """Open one archive member exclusively beneath the pinned staging directory."""
    directory_descriptor = os.dup(staging_descriptor)
    try:
        traversed: tuple[str, ...] = ()
        for part in parts[:-1]:
            traversed += (part,)
            expected_identity = directory_identities.get(traversed)
            if expected_identity is None:
                os.mkdir(part, mode=0o700, dir_fd=directory_descriptor)
                created = os.stat(part, dir_fd=directory_descriptor, follow_symlinks=False)
                expected_identity = _identity(created)
                directory_identities[traversed] = expected_identity
            next_descriptor: int | None = os.open(
                part, _archive_directory_flags(), dir_fd=directory_descriptor
            )
            try:
                opened = os.fstat(next_descriptor)
                current = os.stat(part, dir_fd=directory_descriptor, follow_symlinks=False)
                if (
                    not stat.S_ISDIR(current.st_mode)
                    or _identity(opened) != expected_identity
                    or _identity(current) != expected_identity
                ):
                    raise CheckpointReferenceResolutionError(
                        "checkpoint archive extraction directory identity changed"
                    )
                os.close(directory_descriptor)
                directory_descriptor = next_descriptor
                next_descriptor = None
            finally:
                if next_descriptor is not None:
                    os.close(next_descriptor)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        descriptor: int | None = os.open(
            parts[-1], flags, 0o600, dir_fd=directory_descriptor
        )
        try:
            identity = _identity(os.fstat(descriptor))
            current = os.stat(parts[-1], dir_fd=directory_descriptor, follow_symlinks=False)
            if identity != _identity(current) or not stat.S_ISREG(current.st_mode):
                raise CheckpointReferenceResolutionError(
                    "checkpoint archive extraction file identity changed"
                )
            result = descriptor
            descriptor = None
            return result
        finally:
            if descriptor is not None:
                os.close(descriptor)
    finally:
        os.close(directory_descriptor)


def _identity(value: os.stat_result) -> tuple[int, int]:
    return value.st_dev, value.st_ino


def _require_external_archive_mapping(
    parent: Path,
    parent_identity: os.stat_result,
    *,
    child: Path | None = None,
    child_identity: os.stat_result | None = None,
) -> None:
    """Require external paths to still identify the pinned archive objects."""
    try:
        actual_parent = parent.stat(follow_symlinks=False)
        actual_child = child.stat(follow_symlinks=False) if child is not None else None
    except OSError as exc:
        raise CheckpointReferenceResolutionError(
            "checkpoint archive destination mapping changed"
        ) from exc
    if _identity(actual_parent) != _identity(parent_identity) or (
        child_identity is not None
        and (actual_child is None or _identity(actual_child) != _identity(child_identity))
    ):
        raise CheckpointReferenceResolutionError(
            "checkpoint archive destination mapping changed"
        )


def _publish_directory_no_replace(
    parent_descriptor: int,
    source_name: str,
    destination_name: str,
    *,
    expected_identity: os.stat_result,
) -> None:
    """Publish a directory atomically, rejecting platforms without no-replace rename."""
    _rename_archive_directory_no_replace(
        parent_descriptor,
        source_name,
        destination_name,
        expected_identity=expected_identity,
    )


def _rename_archive_directory_no_replace(
    parent_descriptor: int,
    source_name: str,
    destination_name: str,
    *,
    expected_identity: os.stat_result,
) -> None:
    """Rename one entry under a pinned parent without replacing another name."""
    current = os.stat(source_name, dir_fd=parent_descriptor, follow_symlinks=False)
    if not stat.S_ISDIR(current.st_mode) or _identity(current) != _identity(expected_identity):
        raise CheckpointReferenceResolutionError(
            "checkpoint archive private staging identity changed before publication"
        )
    _perform_archive_rename_no_replace(parent_descriptor, source_name, destination_name)


def _perform_archive_rename_no_replace(
    parent_descriptor: int, source_name: str, destination_name: str
) -> None:
    """Invoke the supported descriptor-relative no-replace rename primitive."""
    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source_name)
    destination_bytes = os.fsencode(destination_name)
    if hasattr(libc, "renameatx_np"):
        result = libc.renameatx_np(
            parent_descriptor,
            source_bytes,
            parent_descriptor,
            destination_bytes,
            0x00000004,
        )
    elif hasattr(libc, "renameat2"):
        result = libc.renameat2(
            parent_descriptor, source_bytes, parent_descriptor, destination_bytes, 1
        )
    else:
        raise CheckpointReferenceResolutionError(
            "atomic no-replace directory publication is unsupported on this platform"
        )
    if result != 0:
        error = ctypes.get_errno()
        if error == getattr(os, "EEXIST", 17):
            raise FileExistsError(
                f"checkpoint archive destination won publication race: {destination_name}"
            )
        raise CheckpointReferenceResolutionError(
            f"atomic no-replace directory publication failed: {os.strerror(error)}"
        )


def _canonical_archive_relative_path(value: str | None, *, context: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise CheckpointReferenceResolutionError(f"{context} is not a safe POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in value.split("/")):
        raise CheckpointReferenceResolutionError(f"{context} is not a canonical relative path")
    return path.as_posix()


def _read_archive_source(path: Path, *, context: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise CheckpointReferenceResolutionError(f"{context} could not be read: {path}") from exc


def _checkpoint_custody_archive_bytes(
    archive_document: bytes,
    payload: Sequence[tuple[str, Path, bytes]],
) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", filename="", mtime=0, compresslevel=9) as gz:
        with tarfile.open(fileobj=gz, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            for name, data in [
                ("archive.json", archive_document),
                *((member_name, member_data) for member_name, _, member_data in payload),
            ]:
                info = tarfile.TarInfo(name)
                info.size = len(data)
                info.mode = 0o644
                info.uid = info.gid = info.mtime = 0
                info.uname = info.gname = ""
                archive.addfile(info, io.BytesIO(data))
    return output.getvalue()


def _resolve_checkpoint_custody_ref(
    ref: ParentRef,
    *,
    allowed_root: str | Path,
    slot_names: Collection[str] | None,
) -> ResolvedCheckpointTransaction:
    if not isinstance(ref, ParentRef):
        raise CheckpointReferenceResolutionError("ref must be a ParentRef")
    if ref.kind != "TrainingCheckpointTransactionManifest":
        raise CheckpointReferenceResolutionError(
            "checkpoint custody ParentRef kind must be 'TrainingCheckpointTransactionManifest'"
        )
    if ref.role != "training_checkpoint_custody":
        raise CheckpointReferenceResolutionError(
            "checkpoint custody ParentRef role must be 'training_checkpoint_custody'"
        )
    if not ref.id:
        raise CheckpointReferenceResolutionError("checkpoint custody ParentRef id is empty")

    manifest_sha256 = ref.metadata.get("manifest_sha256")
    if not _is_sha256(manifest_sha256):
        raise CheckpointReferenceResolutionError(
            "checkpoint custody ParentRef metadata.manifest_sha256 must be "
            "64 lowercase hexadecimal characters"
        )

    root_path = Path(allowed_root).expanduser().resolve()
    if not root_path.is_dir():
        raise CheckpointReferenceResolutionError(
            f"allowed checkpoint custody root is not a directory: {root_path}"
        )
    manifest_path = _resolve_checkpoint_parent_ref_uri(ref.uri, root_path)
    if not manifest_path.is_file():
        raise CheckpointReferenceResolutionError(
            f"checkpoint custody ParentRef manifest is missing: {manifest_path}"
        )

    try:
        raw_manifest = manifest_path.read_bytes()
    except OSError as exc:
        raise CheckpointReferenceResolutionError(
            f"checkpoint custody ParentRef manifest could not be read: {manifest_path}"
        ) from exc
    if sha256_bytes(raw_manifest) != manifest_sha256:
        raise CheckpointReferenceResolutionError(
            "checkpoint custody ParentRef manifest_sha256 does not match raw manifest bytes"
        )

    try:
        loaded_manifest = load_checkpoint_transaction_manifest_json(
            raw_manifest,
            path=str(manifest_path),
        )
    except CheckpointIntegrityError as exc:
        raise CheckpointReferenceResolutionError(
            f"checkpoint custody ParentRef manifest is invalid: {exc}"
        ) from exc
    manifest = loaded_manifest.document
    if manifest.transaction_id != ref.id:
        raise CheckpointReferenceResolutionError(
            "checkpoint custody ParentRef id does not match manifest transaction_id"
        )

    slots_by_name = _validate_checkpoint_transaction_integrity_records(manifest)
    selected_names = _checkpoint_slot_selection(slot_names, slots_by_name)
    transaction_dir = manifest_path.parent.resolve()
    loaded_slots: dict[str, Any] = {}
    for name in selected_names:
        slot = slots_by_name[name]
        blob_path = _resolve_checkpoint_slot_path(
            slot,
            transaction_dir=transaction_dir,
            allowed_root=root_path,
        )
        try:
            value = _deserialize_checkpoint_slot(slot, blob_path)
        except (CheckpointIntegrityError, OSError) as exc:
            raise CheckpointReferenceResolutionError(str(exc)) from exc
        _validate_decoded_slot_integrity(slot, value)
        loaded_slots[name] = value

    try:
        provenance_notices, _ = _validate_manifest_structural_abi(manifest, loaded_slots)
    except CheckpointIntegrityError as exc:
        raise CheckpointReferenceResolutionError(str(exc)) from exc
    return ResolvedCheckpointTransaction(
        parent_ref=_immutable_model_snapshot(ref),
        manifest_sha256=manifest_sha256,
        manifest=_immutable_model_snapshot(manifest),
        slots=MappingProxyType(loaded_slots),
        migration_records=tuple(
            _immutable_model_snapshot(record) for record in loaded_manifest.migration_records
        ),
        provenance_notices=tuple(
            _immutable_model_snapshot(notice) for notice in provenance_notices
        ),
    )


def _resolve_checkpoint_parent_ref_uri(uri: str | None, root_path: Path) -> Path:
    if not isinstance(uri, str) or not uri:
        raise CheckpointReferenceResolutionError(
            "checkpoint custody ParentRef uri must be a non-empty root-relative path"
        )
    parsed = urlsplit(uri)
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
        raise CheckpointReferenceResolutionError(
            "checkpoint custody ParentRef uri must be scheme-free, query-free, "
            "fragment-free, and root-relative"
        )
    decoded = unquote(parsed.path)
    if "\x00" in decoded or "\\" in decoded:
        raise CheckpointReferenceResolutionError(
            "checkpoint custody ParentRef uri contains an unsupported path separator"
        )
    raw_parts = decoded.split("/")
    relative = PurePosixPath(decoded)
    if relative.is_absolute():
        raise CheckpointReferenceResolutionError(
            "checkpoint custody ParentRef uri must be root-relative"
        )
    if ".." in raw_parts:
        raise CheckpointReferenceResolutionError(
            "checkpoint custody ParentRef manifest uri escapes its allowed root"
        )
    if not relative.parts or any(part in {"", "."} for part in raw_parts):
        raise CheckpointReferenceResolutionError(
            "checkpoint custody ParentRef uri must not contain empty or dot path segments"
        )
    candidate = (root_path / Path(*relative.parts)).resolve()
    _require_contained_path(
        candidate,
        root_path,
        context="checkpoint custody ParentRef manifest uri",
    )
    return candidate


def _checkpoint_slot_selection(
    slot_names: Collection[str] | None,
    slots_by_name: Mapping[str, CheckpointSlotBlobRef],
) -> tuple[str, ...]:
    if slot_names is None:
        return tuple(slots_by_name)
    if isinstance(slot_names, (str, bytes, bytearray)):
        raise CheckpointReferenceResolutionError(
            "slot_names must be a collection of names, not a bare string"
        )
    if not isinstance(slot_names, Collection):
        raise CheckpointReferenceResolutionError("slot_names must be a collection of strings")
    requested = list(slot_names)
    if not requested:
        raise CheckpointReferenceResolutionError("slot_names must not be empty")
    if any(not isinstance(name, str) or not name for name in requested):
        raise CheckpointReferenceResolutionError("slot_names entries must be non-empty strings")
    if len(requested) != len(set(requested)):
        raise CheckpointReferenceResolutionError("slot_names contains duplicate names")
    missing = [name for name in requested if name not in slots_by_name]
    if missing:
        raise CheckpointReferenceResolutionError(
            f"requested checkpoint slots are missing: {missing!r}"
        )
    requested_set = set(requested)
    return tuple(name for name in slots_by_name if name in requested_set)


def _validate_checkpoint_transaction_integrity_records(
    manifest: CheckpointTransactionManifest,
) -> dict[str, CheckpointSlotBlobRef]:
    integrity = manifest.content_integrity_digest
    if integrity.schema_id != _CONTENT_INTEGRITY_SCHEMA_ID:
        raise CheckpointReferenceResolutionError(
            "checkpoint content integrity schema_id is unsupported: "
            f"{integrity.schema_id!r}"
        )
    if integrity.schema_version != _CONTENT_INTEGRITY_SCHEMA_VERSION:
        raise CheckpointReferenceResolutionError(
            "checkpoint content integrity schema_version is unsupported: "
            f"{integrity.schema_version!r}"
        )
    slots_by_name = {slot.slot: slot for slot in manifest.slots}
    if len(slots_by_name) != len(manifest.slots):
        raise CheckpointReferenceResolutionError(
            "checkpoint transaction manifest slot names are not unique"
        )
    digest_records = manifest.content_integrity_digest.slots
    digests_by_name = {digest.slot: digest for digest in digest_records}
    if len(digests_by_name) != len(digest_records):
        raise CheckpointReferenceResolutionError(
            "checkpoint transaction content digest slot names are not unique"
        )
    if set(slots_by_name) != set(digests_by_name):
        raise CheckpointReferenceResolutionError(
            "checkpoint transaction slots and content digest records do not correspond"
        )
    for name, slot in slots_by_name.items():
        digest = digests_by_name[name]
        fingerprint = slot.structural_abi_fingerprint
        if fingerprint.schema_id != _STRUCTURAL_ABI_SCHEMA_ID:
            raise CheckpointReferenceResolutionError(
                f"checkpoint slot {name!r} structural ABI schema_id is unsupported: "
                f"{fingerprint.schema_id!r}"
            )
        if fingerprint.schema_version != _STRUCTURAL_ABI_SCHEMA_VERSION:
            raise CheckpointReferenceResolutionError(
                f"checkpoint slot {name!r} structural ABI schema_version is unsupported: "
                f"{fingerprint.schema_version!r}"
            )
        if fingerprint.fingerprint_algorithm_version != _STRUCTURAL_ABI_ALGORITHM_VERSION:
            raise CheckpointReferenceResolutionError(
                f"checkpoint slot {name!r} structural ABI algorithm is unsupported: "
                f"{fingerprint.fingerprint_algorithm_version!r}"
            )
        if slot.content_digest != digest:
            raise CheckpointReferenceResolutionError(
                f"checkpoint slot {name!r} embedded and transaction content digests differ"
            )
        if digest.blob_sha256 != slot.sha256 or digest.blob_size_bytes != slot.size_bytes:
            raise CheckpointReferenceResolutionError(
                f"checkpoint slot {name!r} blob metadata and content digest differ"
            )
        expected_slot_root = _slot_root_sha256(
            name,
            digest.blob_sha256,
            digest.leaf_hashes,
        )
        if digest.slot_root_sha256 != expected_slot_root:
            raise CheckpointReferenceResolutionError(
                f"checkpoint slot {name!r} content root is stale"
            )
        expected_fingerprint = _canonical_hash(
            _structural_abi_content_payload(fingerprint.treedef, fingerprint.leaves)
        )
        if fingerprint.fingerprint_sha256 != expected_fingerprint:
            raise CheckpointReferenceResolutionError(
                f"checkpoint slot {name!r} structural ABI fingerprint is stale"
            )
    expected_transaction_root = _transaction_root_sha256(digest_records)
    if manifest.content_integrity_digest.transaction_root_sha256 != expected_transaction_root:
        raise CheckpointReferenceResolutionError("checkpoint transaction content root is stale")
    return slots_by_name


def _resolve_checkpoint_slot_path(
    slot: CheckpointSlotBlobRef,
    *,
    transaction_dir: Path,
    allowed_root: Path,
) -> Path:
    if slot.media_type != "application/x-python-pickle":
        raise CheckpointReferenceResolutionError(
            f"checkpoint slot {slot.slot!r} has unsupported media_type {slot.media_type!r}"
        )
    relative = Path(slot.relative_path)
    if relative.is_absolute():
        raise CheckpointReferenceResolutionError(
            f"checkpoint slot {slot.slot!r} relative_path must be relative"
        )
    candidate = (transaction_dir / relative).resolve()
    _require_contained_path(
        candidate,
        allowed_root,
        context=f"checkpoint slot {slot.slot!r} path",
    )
    _require_contained_path(
        candidate,
        transaction_dir,
        context=f"checkpoint slot {slot.slot!r} path",
    )
    return candidate


def _require_contained_path(candidate: Path, root: Path, *, context: str) -> None:
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise CheckpointReferenceResolutionError(f"{context} escapes allowed custody root") from exc


def _validate_decoded_slot_integrity(slot: CheckpointSlotBlobRef, value: Any) -> None:
    actual = _slot_integrity_records(value)
    recorded = slot.content_digest
    if actual.leaf_digests != recorded.leaf_hashes:
        raise CheckpointReferenceResolutionError(
            f"checkpoint slot {slot.slot!r} decoded leaf content digest mismatch"
        )
    actual_root = _slot_root_sha256(slot.slot, slot.sha256, actual.leaf_digests)
    if actual_root != recorded.slot_root_sha256:
        raise CheckpointReferenceResolutionError(
            f"checkpoint slot {slot.slot!r} decoded content root mismatch"
        )


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _immutable_model_snapshot(value: BaseModel) -> Any:
    """Return a recursively immutable, typed snapshot of one verified record."""
    model_type = type(value)
    read_only_type = _READ_ONLY_MODEL_TYPES.get(model_type)
    if read_only_type is None:
        config_values = dict(model_type.model_config)
        config_values["frozen"] = True
        config = ConfigDict(**config_values)

        def snapshot_eq(self: BaseModel, other: Any) -> bool:
            return isinstance(other, model_type) and _snapshot_comparison_value(
                self
            ) == _snapshot_comparison_value(other)

        read_only_type = type(
            f"_ReadOnly{model_type.__name__}",
            (model_type,),
            {
                "model_config": config,
                "__module__": __name__,
                "__eq__": snapshot_eq,
            },
        )
        _READ_ONLY_MODEL_TYPES[model_type] = read_only_type
    snapshot = read_only_type.model_validate(value.model_dump(mode="python", round_trip=True))
    for field_name in type(snapshot).model_fields:
        object.__setattr__(
            snapshot,
            field_name,
            _immutable_snapshot_value(getattr(snapshot, field_name)),
        )
    return snapshot


def _immutable_snapshot_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _immutable_model_snapshot(value)
    if isinstance(value, Mapping):
        return _FrozenDict(
            (
                _immutable_snapshot_value(key),
                _immutable_snapshot_value(item),
            )
            for key, item in value.items()
        )
    if isinstance(value, list):
        return _FrozenList(_immutable_snapshot_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_immutable_snapshot_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_immutable_snapshot_value(item) for item in value)
    return value


def _snapshot_comparison_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return {
            field_name: _snapshot_comparison_value(getattr(value, field_name))
            for field_name in type(value).model_fields
        }
    if isinstance(value, Mapping):
        return {
            _snapshot_comparison_value(key): _snapshot_comparison_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return tuple(_snapshot_comparison_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_snapshot_comparison_value(item) for item in value)
    return value


def _load_checkpoint_document_json(
    data: bytes | str,
    *,
    kind: str,
    model: type[CheckpointLatestPointer] | type[CheckpointTransactionManifest],
    path: str,
    document_name: str,
) -> CheckpointDocumentLoadResult[Any]:
    """Migrate one JSON mapping through the shared durable-schema registry."""
    try:
        payload = json.loads(data)
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointIntegrityError(f"{document_name} is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise CheckpointIntegrityError(f"{document_name} must be a JSON object")
    try:
        from feedbax.contracts.migrations import migrate_structured_spec_payload

        migrated = migrate_structured_spec_payload(kind, payload, path=path)
        document = model.model_validate(migrated.payload)
    except (ValueError, ValidationError) as exc:
        raise CheckpointIntegrityError(f"{document_name} is invalid: {exc}") from exc
    return CheckpointDocumentLoadResult(
        document=document,
        schema_id=migrated.schema_id,
        source_version=migrated.source_version,
        target_version=migrated.target_version,
        migration_records=tuple(migrated.migration_records),
    )


def _resolve_latest_manifest_path(
    root_path: Path,
    latest: CheckpointLatestPointer,
) -> Path:
    relative = Path(latest.manifest_relative_path)
    if relative.is_absolute():
        raise CheckpointIntegrityError(
            "latest pointer manifest_relative_path must be relative to custody root"
        )
    candidate = (root_path / relative).resolve()
    try:
        candidate.relative_to(root_path)
    except ValueError as exc:
        raise CheckpointIntegrityError(
            "latest pointer manifest_relative_path escapes custody root: "
            f"{latest.manifest_relative_path!r}"
        ) from exc
    return candidate


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
    ):
        value = metadata.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return default


def _declared_batch_progress(
    phase_program: PhaseProgramSpec,
    slots: Mapping[str, Any],
    slot_axis_bindings: Mapping[str, tuple[MaterializedSlotAxisBinding, ...]] | None = None,
) -> int | None:
    """Read completed batches from the method-declared bookkeeping authority."""
    authority = phase_program.batch_progress
    if authority is None:
        return None
    path = "/phase_program/batch_progress"
    if authority.slot not in slots:
        raise CheckpointConsistencyError(
            f"{path}/slot={authority.slot!r} is missing from checkpoint slots"
        )
    bindings = dict(slot_axis_bindings or {}).get(authority.slot, ())
    if bindings and bindings[0].mode == "mapped":
        values = [
            _declared_batch_progress(
                phase_program,
                {
                    **slots,
                    authority.slot: jt.map(
                        lambda leaf: leaf[index] if eqx.is_array(leaf) else leaf,
                        slots[authority.slot],
                    ),
                },
            )
            for index in range(bindings[0].size)
        ]
        if any(value != values[0] for value in values[1:]):
            raise CheckpointConsistencyError(
                f"{path} mapped authorities diverged: {values!r}"
            )
        return values[0]
    value = slots[authority.slot]
    for index, segment in enumerate(authority.field_path):
        segment_path = f"{path}/field_path/{index}"
        if isinstance(value, Mapping):
            if segment not in value:
                raise CheckpointConsistencyError(
                    f"{segment_path}={segment!r} is missing in slot {authority.slot!r}"
                )
            value = value[segment]
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if not isinstance(segment, int) or segment >= len(value):
                raise CheckpointConsistencyError(
                    f"{segment_path}={segment!r} is not a valid index in slot {authority.slot!r}"
                )
            value = value[segment]
        else:
            raise CheckpointConsistencyError(
                f"{segment_path} cannot traverse non-container in slot {authority.slot!r}"
            )
    try:
        scalar = jax.device_get(value)
        scalar_array = np.asarray(scalar)
        if scalar_array.size != 1:
            raise ValueError(f"expected one scalar, got shape={scalar_array.shape!r}")
        batches = int(scalar_array.item())
    except (TypeError, ValueError, OverflowError) as exc:
        raise CheckpointConsistencyError(
            f"{path} resolved slot {authority.slot!r} to non-integer {value!r}"
        ) from exc
    if batches < 0:
        raise CheckpointConsistencyError(
            f"{path} resolved slot {authority.slot!r} to negative batches={batches}"
        )
    return batches


def _resolve_completed_training_batches(
    *,
    phase_program: PhaseProgramSpec,
    slots: Mapping[str, Any],
    explicit: int | None,
    metadata: Mapping[str, Any],
    default: int | None = None,
    slot_axis_bindings: Mapping[str, tuple[MaterializedSlotAxisBinding, ...]] | None = None,
) -> int | None:
    """Resolve custody batch total and fail closed on authority disagreement."""
    declared = _completed_training_batches(explicit, metadata, default=default)
    authoritative = _declared_batch_progress(
        phase_program,
        slots,
        slot_axis_bindings,
    )
    if authoritative is None:
        return declared
    if declared is not None and declared != authoritative:
        raise CheckpointConsistencyError(
            "completed-training-batches disagreement: "
            f"/completed_training_batches={declared} differs from "
            f"/phase_program/batch_progress={authoritative}"
        )
    return authoritative


def _validate_batch_histories(
    slots: Mapping[str, Any],
    *,
    segment_batch_count: int | None,
) -> None:
    """Validate every marked history against this transaction's segment length."""
    marked: list[tuple[str, str, BatchHistory[Any]]] = []
    for slot, value in slots.items():
        pairs, _ = jt.flatten_with_path(value, is_leaf=is_type(BatchHistory))
        marked.extend(
            (slot, _key_path_to_text(path), leaf)
            for path, leaf in pairs
            if isinstance(leaf, BatchHistory)
        )
    if not marked:
        return
    if segment_batch_count is None:
        raise CheckpointConsistencyError(
            "BatchHistory custody requires completed_training_batches for segment validation"
        )
    for slot, path, history in marked:
        if not eqx.is_array(history.value):
            raise CheckpointConsistencyError(
                f"BatchHistory must wrap an array; slot={slot!r} path={path!r}"
            )
        shape = tuple(jnp.asarray(history.value).shape)
        axis = history.batch_axis
        if axis < 0:
            axis += len(shape)
        if axis < 0 or axis >= len(shape):
            raise CheckpointConsistencyError(
                "BatchHistory batch_axis is out of bounds; "
                f"slot={slot!r} path={path!r} shape={shape!r} axis={history.batch_axis}"
            )
        expected = history.granularity.expected_entries(segment_batch_count)
        if shape[axis] != expected:
            raise CheckpointConsistencyError(
                "BatchHistory length must match the owning segment; "
                f"slot={slot!r} path={path!r} axis={history.batch_axis} "
                f"actual={shape[axis]} expected={expected} "
                f"segment_batches={segment_batch_count} "
                f"interval={history.granularity.interval}. Histories must be segment-local; "
                "methods needing statistics over the whole training past must express them "
                "as fixed-size streaming state in dynamical state (for example an EMA, "
                "ring buffer, reservoir sample, or scalar accumulator)."
            )


def _history_granularities(slots: Mapping[str, Any]) -> dict[str, int]:
    granularities: dict[str, int] = {}
    for slot, value in slots.items():
        pairs, _ = jt.flatten_with_path(value, is_leaf=is_type(BatchHistory))
        for path, leaf in pairs:
            if isinstance(leaf, BatchHistory):
                granularities[f"{slot}{_key_path_to_text(path)}"] = leaf.granularity.interval
    return granularities


def _wrap_migrated_v5_batch_histories(
    slots: Mapping[str, Any],
    manifest: CheckpointTransactionManifest,
) -> dict[str, Any]:
    """Wrap v5 declared paths after envelope migration, without extension semantics."""
    if manifest.metadata.get("batch_history_tree_migration") != "declared_paths_v5_to_v6":
        return dict(slots)
    raw_request = manifest.metadata.get("checkpoint_continuation")
    if not isinstance(raw_request, Mapping):
        return dict(slots)
    if raw_request.get("schema_version") != "feedbax.spec.training_checkpoint_continuation.v1":
        return dict(slots)
    raw_declarations = raw_request.get("batch_indexed_leaves")
    if not isinstance(raw_declarations, list):
        raise CheckpointCompatibilityError(
            "legacy v1 continuation batch_indexed_leaves must be a list"
        )
    by_slot: dict[str, set[str]] = {}
    for index, declaration in enumerate(raw_declarations):
        if not isinstance(declaration, Mapping):
            raise CheckpointCompatibilityError(
                f"legacy v1 BatchHistory declaration must be a mapping; index={index}"
            )
        slot = declaration.get("slot")
        path = declaration.get("tree_path")
        if not isinstance(slot, str) or not slot or not isinstance(path, str) or not path.startswith(
            "/"
        ):
            raise CheckpointCompatibilityError(
                f"legacy v1 BatchHistory declaration is invalid; index={index}"
            )
        by_slot.setdefault(slot, set()).add(path)
    migrated = dict(slots)
    for slot, paths in by_slot.items():
        if slot not in slots:
            raise CheckpointCompatibilityError(
                f"legacy BatchHistory migration slot is missing; slot={slot!r}"
            )
        pairs, treedef = jt.flatten_with_path(slots[slot])
        found: set[str] = set()
        leaves: list[Any] = []
        for path, leaf in pairs:
            path_text = _key_path_to_text(path)
            if path_text in paths:
                if not eqx.is_array(leaf):
                    raise CheckpointCompatibilityError(
                        "legacy BatchHistory migration target must be an array; "
                        f"slot={slot!r} path={path_text!r}"
                    )
                leaf = BatchHistory(leaf)
                found.add(path_text)
            leaves.append(leaf)
        missing = sorted(paths - found)
        if missing:
            raise CheckpointCompatibilityError(
                "legacy BatchHistory migration path is missing; "
                f"slot={slot!r} paths={missing!r}"
            )
        migrated[slot] = jt.unflatten(treedef, leaves)
    return migrated


def _allocate_segment_histories(
    source_slots: Mapping[str, Any],
    templates: Mapping[str, Any],
) -> tuple[dict[str, Any], set[str]]:
    """Preserve dynamical leaves while replacing histories with segment templates."""
    result = dict(source_slots)
    changed: set[str] = set()
    for slot, source in source_slots.items():
        if slot not in templates:
            continue
        source_leaves, source_def = jt.flatten(source, is_leaf=is_type(BatchHistory))
        target_leaves, target_def = jt.flatten(templates[slot], is_leaf=is_type(BatchHistory))
        if source_def != target_def or len(source_leaves) != len(target_leaves):
            raise CheckpointCompatibilityError(
                f"continuation slot structure differs before history allocation; slot={slot!r}"
            )
        output: list[Any] = []
        slot_changed = False
        for source_leaf, target_leaf in zip(source_leaves, target_leaves, strict=True):
            if isinstance(source_leaf, BatchHistory) or isinstance(target_leaf, BatchHistory):
                if not isinstance(source_leaf, BatchHistory) or not isinstance(
                    target_leaf, BatchHistory
                ):
                    raise CheckpointCompatibilityError(
                        f"continuation BatchHistory marking differs; slot={slot!r}"
                    )
                if (
                    source_leaf.batch_axis != target_leaf.batch_axis
                    or source_leaf.granularity.interval != target_leaf.granularity.interval
                ):
                    raise CheckpointCompatibilityError(
                        f"continuation BatchHistory granularity differs; slot={slot!r}"
                    )
                output.append(target_leaf)
                slot_changed = True
            else:
                output.append(source_leaf)
        if slot_changed:
            result[slot] = jt.unflatten(source_def, output)
            changed.add(slot)
    return result, changed


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
    segment_start_batch: int = 0,
    segment_batch_count: int | None = None,
    segment_parent_transaction_id: str | None = None,
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
    resolved_axes = _resolved_slot_axes(run_spec)
    coordinate = _checkpoint_coordinate(coordinate, resolved_axes)
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
            materialized_axes = resolved_axes.get(spec.slot)
            _validate_slot_axes(
                spec.slot,
                value,
                materialized_axes,
                error_cls=CheckpointCompatibilityError,
            )
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
                coordinate=_checkpoint_coordinate(
                    (slot_coordinates or {}).get(spec.slot, coordinate),
                    resolved_axes,
                ),
                structural_abi_fingerprint=integrity.structural_abi_fingerprint,
                content_digest=content_digest,
                materialized_axes=materialized_axes,
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
        _validate_program_step_units(
            coordinate,
            manifest_metadata,
            context="checkpoint write",
        )
        completed_batches = _resolve_completed_training_batches(
            phase_program=phase_program,
            slots=slots,
            explicit=completed_training_batches,
            metadata=manifest_metadata,
            slot_axis_bindings=resolved_axes,
        )
        resolved_segment_batches = (
            segment_batch_count if segment_batch_count is not None else completed_batches
        )
        _validate_batch_histories(slots, segment_batch_count=resolved_segment_batches)
        manifest = CheckpointTransactionManifest(
            transaction_id=transaction_id,
            run_id=coordinate.run_id,
            status=status,  # type: ignore[arg-type]
            barrier=barrier.name,
            completed_coordinate=coordinate,
            completed_training_batches=completed_batches,
            segment_lineage=CheckpointSegmentLineage(
                parent_transaction_id=segment_parent_transaction_id,
                start_batch=segment_start_batch,
                segment_batch_count=resolved_segment_batches or 0,
                history_granularities=_history_granularities(slots),
            ),
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
    expected_axes = _resolved_slot_axes(expected_run_spec)
    _validate_recorded_slot_axes(manifest, expected_axes)
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
        loaded_slots[slot.slot] = _deserialize_checkpoint_slot(slot, blob_path)
    provenance_notices, loaded_fingerprints = _validate_manifest_structural_abi(
        manifest,
        loaded_slots,
    )
    migrated_slots = _wrap_migrated_v5_batch_histories(loaded_slots, manifest)
    if any(migrated_slots.get(slot) is not value for slot, value in loaded_slots.items()):
        loaded_slots = migrated_slots
        loaded_fingerprints = {}
    request = _coerce_continuation_request(continuation_request)
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
    if request is not None and not _continuation_was_applied(manifest, request):
        lineage_total = (
            manifest.segment_lineage.start_batch
            + manifest.segment_lineage.segment_batch_count
        )
        if request.source_completed_batches != lineage_total:
            raise CheckpointCompatibilityError(
                "checkpoint continuation source offset mismatch; "
                f"lineage_total={lineage_total} "
                f"requested={request.source_completed_batches}"
            )
        loaded_slots, _ = _allocate_segment_histories(loaded_slots, expected_slots)
        loaded_fingerprints = {}
    _validate_structural_abi(
        manifest,
        expected_slots,
        loaded_slots,
        loaded_fingerprints=loaded_fingerprints,
    )
    _validate_recorded_slot_axes(manifest, expected_axes, loaded_slots)

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
        previous_transaction_id=(manifest.transaction_id if allow_new_lineage_override else None),
    )


def _manifest_from_latest_pointer(
    root_path: Path,
    latest: CheckpointLatestPointer,
) -> tuple[Path, CheckpointTransactionManifest]:
    manifest_path = _resolve_latest_manifest_path(root_path.resolve(), latest)
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


def checkpoint_fork_plan_canonical_projection(plan: CheckpointForkPlan) -> dict[str, Any]:
    """Return the portable semantic projection hashed into fork provenance."""
    return plan.model_dump(mode="json", exclude_none=True, exclude={"metadata"})


def checkpoint_fork_plan_sha256(plan: CheckpointForkPlan) -> str:
    """Return the deterministic compatibility hash for ``plan``."""
    return sha256_bytes(canonical_json_bytes(checkpoint_fork_plan_canonical_projection(plan)))


def derive_checkpoint_fork_compatibility_projection(
    run_spec: TrainingRunSpec,
    phase_program: PhaseProgramSpec,
    slot_templates: Mapping[str, Any],
    *,
    population_member_ids: Mapping[str, Sequence[str]] | None = None,
) -> CheckpointForkCompatibilityProjection:
    """Bind a fork target to its canonical run contract and template ABIs."""
    binding = run_contract_binding(run_spec, phase_program)
    projection_sha256 = binding.canonical_projection_sha256
    if projection_sha256 is None:
        raise CheckpointCompatibilityError(
            "checkpoint fork run-contract binding has no canonical projection hash"
        )
    normalized_population = {
        slot: [str(member_id) for member_id in member_ids]
        for slot, member_ids in sorted((population_member_ids or {}).items())
    }
    return CheckpointForkCompatibilityProjection(
        run_contract_algorithm_version=binding.algorithm_version,
        run_contract_hash_domain=binding.hash_domain,
        run_contract_projection_sha256=projection_sha256,
        slot_structural_abi_sha256={
            slot: structural_abi_fingerprint(value).fingerprint_sha256
            for slot, value in sorted(slot_templates.items())
        },
        population_member_ids_sha256=(
            sha256_bytes(canonical_json_bytes(normalized_population))
            if population_member_ids is not None
            else None
        ),
    )


def _validate_checkpoint_fork_compatibility(
    target: CheckpointForkTarget,
    run_spec: TrainingRunSpec,
    phase_program: PhaseProgramSpec,
    slot_templates: Mapping[str, Any],
    population_member_ids: Mapping[str, Sequence[str]],
) -> None:
    actual = derive_checkpoint_fork_compatibility_projection(
        run_spec,
        phase_program,
        slot_templates,
        population_member_ids=(
            population_member_ids if target.population_member_ids_ref is not None else None
        ),
    )
    declared = target.compatibility
    for label, declared_value, actual_value in (
        (
            "run-contract algorithm",
            declared.run_contract_algorithm_version,
            actual.run_contract_algorithm_version,
        ),
        (
            "run-contract hash domain",
            declared.run_contract_hash_domain,
            actual.run_contract_hash_domain,
        ),
        (
            "run-contract projection sha256",
            declared.run_contract_projection_sha256,
            actual.run_contract_projection_sha256,
        ),
        (
            "population member ids sha256",
            declared.population_member_ids_sha256,
            actual.population_member_ids_sha256,
        ),
    ):
        if declared_value != actual_value:
            raise CheckpointCompatibilityError(
                f"checkpoint fork target {target.target_id!r} {label} mismatch; "
                f"declared={declared_value!r} actual={actual_value!r}"
            )
    declared_slots = set(declared.slot_structural_abi_sha256)
    actual_slots = set(actual.slot_structural_abi_sha256)
    if declared_slots != actual_slots:
        raise CheckpointCompatibilityError(
            f"checkpoint fork target {target.target_id!r} compatibility slot coverage "
            f"mismatch; declared={sorted(declared_slots)!r} actual={sorted(actual_slots)!r}"
        )
    for slot in sorted(actual_slots):
        declared_hash = declared.slot_structural_abi_sha256[slot]
        actual_hash = actual.slot_structural_abi_sha256[slot]
        if declared_hash != actual_hash:
            raise CheckpointCompatibilityError(
                f"checkpoint fork target {target.target_id!r} slot {slot!r} compatibility "
                f"ABI mismatch; declared={declared_hash!r} actual={actual_hash!r}"
            )


def _coerce_checkpoint_fork_plan(
    value: CheckpointForkPlan | Mapping[str, Any],
) -> CheckpointForkPlan:
    if isinstance(value, CheckpointForkPlan):
        return value
    try:
        from feedbax.contracts.migrations import migrate_structured_spec_payload

        migrated = migrate_structured_spec_payload("CheckpointForkPlan", value)
        return CheckpointForkPlan.model_validate(migrated.payload)
    except (ValueError, ValidationError) as exc:
        raise CheckpointCompatibilityError(f"checkpoint fork plan is invalid: {exc}") from exc


def _require_plan_binding(values: Mapping[str, Any], ref: str, *, kind: str) -> Any:
    try:
        return values[ref]
    except KeyError as exc:
        raise CheckpointCompatibilityError(
            f"checkpoint fork plan references unknown {kind} {ref!r}"
        ) from exc


def _apply_registered_fork_step(
    slots: Mapping[str, Any],
    step: CheckpointForkTransformStep,
    registry: CheckpointForkTransformRegistry,
) -> tuple[dict[str, Any], set[str]]:
    record = step.records[0]
    registration = registry.resolve(record.identity)
    declared = {item.slot for item in step.records}
    target_only = set(step.target_only_slots)

    def transform(values: Mapping[str, Any]) -> Mapping[str, Any]:
        return registration.transform(values, record.parameters)

    transformed, changed = _apply_target_slot_transform(
        slots,
        transform=transform,
        declared_transformed_slots=declared - target_only,
        declared_target_only_slots=target_only,
    )
    if changed != declared:
        raise CheckpointCompatibilityError(
            f"checkpoint fork transform step {step.step_id!r} did not change exactly its "
            f"declared slots; declared={sorted(declared)!r} actual={sorted(changed)!r}"
        )
    return transformed, changed


def _plan_step_records(
    records: dict[str, list[CheckpointForkTransformRecord]],
    step: CheckpointForkTransformStep,
) -> None:
    for record in step.records:
        metadata = dict(record.metadata)
        metadata.update({"stage": step.stage, "step_id": step.step_id})
        records.setdefault(record.slot, []).append(record.model_copy(update={"metadata": metadata}))


def _resolve_plan_barrier(
    source: _LoadedCheckpointTransaction,
    target: CheckpointForkTarget,
    phase_program: PhaseProgramSpec,
) -> tuple[CheckpointBarrierSpec, ProgressCoordinate]:
    mapping = target.barrier_mapping
    if mapping is None:
        barrier = checkpoint_barrier(phase_program, source.manifest.barrier)
        return barrier, target.target_coordinate or source.manifest.completed_coordinate
    if source.manifest.barrier != mapping.source_barrier:
        raise CheckpointCompatibilityError(
            "checkpoint fork source barrier mapping does not match source manifest; "
            f"declared={mapping.source_barrier!r} actual={source.manifest.barrier!r}"
        )
    if target.target_coordinate is not None and mapping.target_coordinate is not None:
        raise CheckpointCompatibilityError(
            "checkpoint fork target declares both target_coordinate and "
            "barrier_mapping.target_coordinate"
        )
    barrier = checkpoint_barrier(phase_program, mapping.target_barrier)
    coordinate = (
        mapping.target_coordinate
        or target.target_coordinate
        or source.manifest.completed_coordinate
    )
    if coordinate.completed_barrier != barrier.name:
        raise CheckpointCompatibilityError(
            "checkpoint fork target coordinate does not name mapped target barrier; "
            f"coordinate.completed_barrier={coordinate.completed_barrier!r} "
            f"target_barrier={barrier.name!r}"
        )
    return barrier, coordinate


def _validate_prepared_fork_templates(
    target_id: str,
    barrier: CheckpointBarrierSpec,
    prepared_slots: Mapping[str, Any],
    expected_slots: Mapping[str, Any],
) -> None:
    _validate_required_slots(
        tuple(barrier.slots), prepared_slots, error_cls=CheckpointCompatibilityError
    )
    _validate_expected_slot_set(barrier, expected_slots)
    for spec in barrier.slots:
        if spec.slot not in prepared_slots or spec.slot not in expected_slots:
            continue
        actual = structural_abi_fingerprint(prepared_slots[spec.slot])
        expected = structural_abi_fingerprint(expected_slots[spec.slot])
        if actual.fingerprint_sha256 != expected.fingerprint_sha256:
            raise CheckpointCompatibilityError(
                f"checkpoint fork target {target_id!r} slot {spec.slot!r} structural ABI "
                f"mismatch{_format_structural_abi_diff(expected, actual)}"
            )


def _prepare_checkpoint_fork_plan(
    plan: CheckpointForkPlan,
    bindings: CheckpointForkPlanBindings,
    registry: CheckpointForkTransformRegistry,
) -> list[_PreparedForkPlanTarget]:
    source_root = Path(
        _require_plan_binding(
            bindings.checkpoint_roots,
            plan.source.checkpoint_root_ref,
            kind="checkpoint root",
        )
    )
    target_roots = [
        Path(
            _require_plan_binding(
                bindings.checkpoint_roots,
                target.checkpoint_root_ref,
                kind="checkpoint root",
            )
        )
        for target in plan.targets
    ]
    resolved_roots = [source_root.resolve(), *(root.resolve() for root in target_roots)]
    if len(resolved_roots) != len(set(resolved_roots)):
        raise CheckpointCompatibilityError(
            "checkpoint fork source and target checkpoint roots must be distinct"
        )

    all_steps = [
        *plan.source.transforms,
        *(step for target in plan.targets for step in target.transforms),
    ]
    for step in all_steps:
        registry.resolve(step.records[0].identity)
    for target in plan.targets:
        _require_plan_binding(bindings.run_specs, target.run_spec_ref, kind="run spec")
        _require_plan_binding(
            bindings.slot_templates,
            target.slot_template_ref,
            kind="slot template",
        )
        policy = target.history_policy
        if policy.segment_history_template_ref is not None:
            _require_plan_binding(
                bindings.segment_history_templates,
                policy.segment_history_template_ref,
                kind="segment history template",
            )
        if target.population_member_ids_ref is not None:
            _require_plan_binding(
                bindings.population_member_ids,
                target.population_member_ids_ref,
                kind="population member ids",
            )

    source = _load_latest_checkpoint_transaction(source_root)
    source_axes = {
        record.slot: record.materialized_axes
        for record in source.manifest.slots
        if record.materialized_axes is not None
    }
    _validate_recorded_slot_axes(source.manifest, source_axes, source.slots)
    if (
        plan.source.expected_transaction_id is not None
        and source.manifest.transaction_id != plan.source.expected_transaction_id
    ):
        raise CheckpointCompatibilityError(
            "checkpoint fork source transaction id does not match plan; "
            f"declared={plan.source.expected_transaction_id!r} "
            f"actual={source.manifest.transaction_id!r}"
        )
    actual_root = source.manifest.content_integrity_digest.transaction_root_sha256
    if (
        plan.source.expected_transaction_root_sha256 is not None
        and actual_root != plan.source.expected_transaction_root_sha256
    ):
        raise CheckpointCompatibilityError(
            "checkpoint fork source transaction root does not match plan; "
            f"declared={plan.source.expected_transaction_root_sha256!r} "
            f"actual={actual_root!r}"
        )

    common_slots = dict(source.slots)
    common_records: dict[str, list[CheckpointForkTransformRecord]] = {}
    common_changed: set[str] = set()
    for step in plan.source.transforms:
        common_slots, changed = _apply_registered_fork_step(common_slots, step, registry)
        common_changed.update(changed)
        _plan_step_records(common_records, step)

    prepared_targets: list[_PreparedForkPlanTarget] = []
    for target, target_root in zip(plan.targets, target_roots, strict=True):
        run_spec = _require_plan_binding(bindings.run_specs, target.run_spec_ref, kind="run spec")
        expected_slots = _require_plan_binding(
            bindings.slot_templates,
            target.slot_template_ref,
            kind="slot template",
        )
        phase_program = run_spec.worker_execution.method_contract.phase_program
        population_ids = (
            {}
            if target.population_member_ids_ref is None
            else _require_plan_binding(
                bindings.population_member_ids,
                target.population_member_ids_ref,
                kind="population member ids",
            )
        )
        _validate_checkpoint_fork_compatibility(
            target,
            run_spec,
            phase_program,
            expected_slots,
            population_ids,
        )
        barrier, _ = _resolve_plan_barrier(source, target, phase_program)
        slots = dict(common_slots)
        changed_slots = set(common_changed)
        records = {slot: list(values) for slot, values in common_records.items()}
        target_only: dict[str, Mapping[str, Any]] = {}
        for step in (item for item in target.transforms if item.stage == "source_pre"):
            slots, changed = _apply_registered_fork_step(slots, step, registry)
            changed_slots.update(changed)
            _plan_step_records(records, step)

        policy = target.history_policy
        declared_continuation = run_spec.checkpoint_progress.continuation
        if policy.mode == "preserve":
            if declared_continuation is not None:
                raise CheckpointCompatibilityError(
                    f"checkpoint fork target {target.target_id!r} declares continuation "
                    "but history policy is preserve"
                )
            request = None
            history_templates = None
        else:
            if policy.mode == "prepare_continuation" and declared_continuation is None:
                raise CheckpointCompatibilityError(
                    f"checkpoint fork target {target.target_id!r} prepare_continuation "
                    "requires the target run spec to declare continuation"
                )
            request = _resolve_fork_continuation_request(
                target_run_spec=run_spec,
                continuation_request=policy.continuation_request,
            )
            assert request is not None
            history_templates = (
                _require_plan_binding(
                    bindings.segment_history_templates,
                    policy.segment_history_template_ref or "",
                    kind="segment history template",
                )
                if policy.mode == "continue_segment"
                else None
            )
            source_total = (
                source.manifest.segment_lineage.start_batch
                + source.manifest.segment_lineage.segment_batch_count
            )
            if request.source_completed_batches != source_total:
                raise CheckpointCompatibilityError(
                    "checkpoint continuation source offset mismatch; "
                    f"lineage_total={source_total} requested={request.source_completed_batches}"
                )
            if history_templates is not None:
                slots, _ = _allocate_segment_histories(slots, dict(history_templates))

        for step in (item for item in target.transforms if item.stage == "target_post"):
            slots, changed = _apply_registered_fork_step(slots, step, registry)
            changed_slots.update(changed)
            target_only.update(step.target_only_slots)
            _plan_step_records(records, step)
        if policy.mode == "prepare_continuation":
            _validate_required_slots(
                tuple(barrier.slots), slots, error_cls=CheckpointCompatibilityError
            )
            _validate_expected_slot_set(barrier, expected_slots)
        else:
            _validate_prepared_fork_templates(target.target_id, barrier, slots, expected_slots)
        population_slots = _population_slot_names(run_spec)
        for slot in sorted(population_slots | set(population_ids)):
            if slot not in slots:
                raise CheckpointCompatibilityError(
                    f"checkpoint fork target {target.target_id!r} population slot "
                    f"{slot!r} is missing"
                )
            _population_record(
                slot,
                slots[slot],
                population_member_ids=population_ids,
                population_slots=population_slots,
            )
        prepared_targets.append(
            _PreparedForkPlanTarget(
                target_id=target.target_id,
                target_root=target_root,
                run_spec=run_spec,
                phase_program=phase_program,
                expected_slots=expected_slots,
                prepared_slots=slots,
                segment_history_templates=history_templates,
                continuation_request=request,
                continuation_applied=policy.mode != "prepare_continuation",
                barrier_mapping=target.barrier_mapping,
                target_coordinate=target.target_coordinate,
                transformed_slots=frozenset(changed_slots),
                target_only_slots=target_only,
                transform_records={slot: tuple(values) for slot, values in records.items()},
                population_member_ids=population_ids,
            )
        )
    return prepared_targets


def fork_checkpoint_plan(
    plan: CheckpointForkPlan | Mapping[str, Any],
    bindings: CheckpointForkPlanBindings,
    *,
    transform_registry: CheckpointForkTransformRegistry | None = None,
    tool_version: str | None = None,
) -> dict[str, CheckpointForkResult]:
    """Preflight and execute a portable multi-target checkpoint fork plan."""
    resolved_plan = _coerce_checkpoint_fork_plan(plan)
    registry = transform_registry or DEFAULT_CHECKPOINT_FORK_TRANSFORM_REGISTRY
    projection = checkpoint_fork_plan_canonical_projection(resolved_plan)
    plan_sha256 = checkpoint_fork_plan_sha256(resolved_plan)
    prepared_targets = _prepare_checkpoint_fork_plan(resolved_plan, bindings, registry)
    results: dict[str, CheckpointForkResult] = {}
    source_root = _require_plan_binding(
        bindings.checkpoint_roots,
        resolved_plan.source.checkpoint_root_ref,
        kind="checkpoint root",
    )
    for prepared in prepared_targets:
        transformed_source_slots = set(prepared.transformed_slots) - set(
            prepared.target_only_slots
        )
        materializer: ResumeSlotTransform | None = None
        transform_metadata: dict[str, Any] | None = None
        if prepared.transformed_slots:

            def materializer(
                current: Mapping[str, Any],
                *,
                values: Mapping[str, Any] = prepared.prepared_slots,
                changed: frozenset[str] = prepared.transformed_slots,
            ) -> Mapping[str, Any]:
                materialized = dict(current)
                for slot in changed:
                    materialized[slot] = values[slot]
                return materialized

            transform_metadata = {
                "identity": "feedbax.training_checkpoint.plan_materialization.v1",
                "parameters": {"plan_sha256": plan_sha256},
                "declared_plan_stages": {
                    slot: [record.model_dump(mode="json", exclude_none=True) for record in records]
                    for slot, records in prepared.transform_records.items()
                },
            }
        result = fork_checkpoint_transaction(
            source_root,
            prepared.target_root,
            target_run_spec=prepared.run_spec,
            target_phase_program=prepared.phase_program,
            expected_slots=prepared.expected_slots,
            target_coordinate=prepared.target_coordinate,
            barrier_mapping=prepared.barrier_mapping,
            expected_population_member_ids=prepared.population_member_ids,
            segment_history_templates=prepared.segment_history_templates,
            target_slot_transform=materializer,
            target_transform_metadata=transform_metadata,
            target_transformed_slots=(
                sorted(transformed_source_slots) if materializer is not None else None
            ),
            target_only_slots=prepared.target_only_slots or None,
            continuation_request=prepared.continuation_request,
            continuation_applied=prepared.continuation_applied,
            tool_version=tool_version,
            metadata={"checkpoint_fork_plan_target_id": prepared.target_id},
            fork_provenance_metadata={
                "checkpoint_fork_plan_schema_id": resolved_plan.schema_id,
                "checkpoint_fork_plan_schema_version": resolved_plan.schema_version,
                "checkpoint_fork_plan_sha256": plan_sha256,
                "checkpoint_fork_plan_target_id": prepared.target_id,
                "checkpoint_fork_plan_compatibility_projection": projection,
            },
        )
        results[prepared.target_id] = result
    return results


def fork_checkpoint_transaction(
    source_root: str | Path,
    target_root: str | Path,
    *,
    target_run_spec: TrainingRunSpec,
    target_phase_program: PhaseProgramSpec | None = None,
    expected_slots: Mapping[str, Any] | None = None,
    target_coordinate: ProgressCoordinate | None = None,
    barrier_mapping: CheckpointForkBarrierMapping | Mapping[str, Any] | None = None,
    expected_population_member_ids: Mapping[str, Sequence[str]] | None = None,
    slot_transforms: Mapping[str, ResumeSlotTransform] | None = None,
    transform_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    source_slot_transforms: Mapping[str, ResumeSlotTransform] | None = None,
    source_transform_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    segment_history_templates: Mapping[str, Any] | None = None,
    target_slot_transform: ResumeSlotTransform | None = None,
    target_transform_metadata: Mapping[str, Any] | None = None,
    target_transformed_slots: Sequence[str] | None = None,
    target_only_slots: Mapping[str, Mapping[str, Any]] | None = None,
    continuation_request: CheckpointContinuationRequest | Mapping[str, Any] | None = None,
    continuation_applied: bool = True,
    link_strategy: CheckpointBlobLinkStrategy | None = None,
    tool_version: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    fork_provenance_metadata: Mapping[str, Any] | None = None,
) -> CheckpointForkResult:
    """Fork one valid custody checkpoint into a target run contract/root.

    The pipeline is deliberately topology-aware: source/pre transforms run on
    the source topology, declared continuation extends that raw topology from a
    dedicated continuation template, then an optional target/post transform
    produces the target topology. ``target_only_slots`` declares root slots
    introduced only by that post transform. Untransformed slots are hardlinked
    when possible; changed slots are serialized fresh. ``latest.json`` is
    written only after strict target-bound resume validation succeeds.

    A fork retains its source barrier and progress coordinate unless the caller
    supplies ``barrier_mapping``. Crossing barriers is therefore explicit: the
    mapping must name the actual source barrier, an existing target barrier,
    and a declared target coordinate/mapping rationale.

    ``slot_transforms`` and ``transform_metadata`` are retained as legacy
    aliases for ``source_slot_transforms`` and ``source_transform_metadata``.
    Callers must not supply both forms.
    """
    source = _load_latest_checkpoint_transaction(source_root)
    source_axes = {
        record.slot: record.materialized_axes
        for record in source.manifest.slots
        if record.materialized_axes is not None
    }
    _validate_recorded_slot_axes(source.manifest, source_axes, source.slots)
    _validate_program_step_units(
        source.manifest.completed_coordinate,
        source.manifest.metadata,
        context="checkpoint fork source",
    )
    phase_program = target_phase_program or (
        target_run_spec.worker_execution.method_contract.phase_program
    )
    target_axes = _resolved_slot_axes(target_run_spec)
    resolved_barrier_mapping = _coerce_barrier_mapping(barrier_mapping)
    if resolved_barrier_mapping is None:
        barrier = checkpoint_barrier(phase_program, source.manifest.barrier)
        coordinate = target_coordinate or source.manifest.completed_coordinate
    else:
        if source.manifest.barrier != resolved_barrier_mapping.source_barrier:
            raise CheckpointCompatibilityError(
                "checkpoint fork source barrier mapping does not match source manifest; "
                f"declared={resolved_barrier_mapping.source_barrier!r} "
                f"actual={source.manifest.barrier!r}"
            )
        if target_coordinate is not None and resolved_barrier_mapping.target_coordinate is not None:
            raise CheckpointCompatibilityError(
                "checkpoint fork received both target_coordinate and "
                "barrier_mapping.target_coordinate"
            )
        barrier = checkpoint_barrier(
            phase_program,
            resolved_barrier_mapping.target_barrier,
        )
        coordinate = (
            resolved_barrier_mapping.target_coordinate
            or target_coordinate
            or source.manifest.completed_coordinate
        )
        if (
            resolved_barrier_mapping.target_coordinate is not None
            and coordinate.completed_barrier != barrier.name
        ):
            raise CheckpointCompatibilityError(
                "checkpoint fork target coordinate does not name mapped target barrier; "
                f"coordinate.completed_barrier={coordinate.completed_barrier!r} "
                f"target_barrier={barrier.name!r}"
            )
    coordinate = _checkpoint_coordinate(coordinate, target_axes)
    if slot_transforms is not None and source_slot_transforms is not None:
        raise CheckpointCompatibilityError(
            "checkpoint fork received both slot_transforms and source_slot_transforms; "
            "use source_slot_transforms"
        )
    if transform_metadata is not None and source_transform_metadata is not None:
        raise CheckpointCompatibilityError(
            "checkpoint fork received both transform_metadata and "
            "source_transform_metadata; use source_transform_metadata"
        )
    transforms = dict(source_slot_transforms or slot_transforms or {})
    transform_meta = dict(source_transform_metadata or transform_metadata or {})
    target_only_metadata = dict(target_only_slots or {})
    declared_target_slots = set(target_transformed_slots or ())
    if target_slot_transform is None and (
        target_transform_metadata is not None
        or target_transformed_slots is not None
        or target_only_slots is not None
    ):
        raise CheckpointCompatibilityError(
            "target transform declarations require target_slot_transform"
        )
    if target_slot_transform is not None and target_transform_metadata is None:
        raise CheckpointCompatibilityError(
            "target_slot_transform requires target_transform_metadata with identity"
        )
    if target_slot_transform is not None:
        _validate_target_transform_metadata(target_transform_metadata)
    if len(declared_target_slots) != len(tuple(target_transformed_slots or ())):
        raise CheckpointCompatibilityError("target_transformed_slots must not contain duplicates")
    overlap = declared_target_slots & set(target_only_metadata)
    if overlap:
        raise CheckpointCompatibilityError(
            "target slots cannot be both transformed source slots and target-only; "
            f"slots={sorted(overlap)!r}"
        )
    loaded_slots = dict(source.slots)
    prepared_slots = _apply_slot_transforms(
        loaded_slots,
        transforms=transforms,
    )
    validation_slots = dict(expected_slots or prepared_slots)
    request = _resolve_fork_continuation_request(
        target_run_spec=target_run_spec,
        continuation_request=continuation_request,
    )
    continuation_transformed_slots: set[str] = set()
    if request is not None:
        source_total = (
            source.manifest.segment_lineage.start_batch
            + source.manifest.segment_lineage.segment_batch_count
        )
        if request.source_completed_batches != source_total:
            raise CheckpointCompatibilityError(
                "checkpoint continuation source offset mismatch; "
                f"lineage_total={source_total} requested={request.source_completed_batches}"
            )
        if not continuation_applied:
            if segment_history_templates is not None:
                raise CheckpointCompatibilityError(
                    "pending checkpoint continuation cannot allocate segment history templates"
                )
            if source.manifest.completed_training_batches != request.source_completed_batches:
                raise CheckpointCompatibilityError(
                    "checkpoint pending continuation progress mismatch; "
                    f"source_completed={source.manifest.completed_training_batches} "
                    f"requested={request.source_completed_batches}"
                )
        else:
            prepared_slots, continuation_transformed_slots = _allocate_segment_histories(
                prepared_slots,
                dict(segment_history_templates or validation_slots),
            )
    elif not continuation_applied:
        raise CheckpointCompatibilityError(
            "checkpoint continuation cannot be pending without a continuation request"
        )
    target_post_transformed_slots: set[str] = set()
    if target_slot_transform is not None:
        prepared_slots, target_post_transformed_slots = _apply_target_slot_transform(
            prepared_slots,
            transform=target_slot_transform,
            declared_transformed_slots=declared_target_slots,
            declared_target_only_slots=set(target_only_metadata),
        )
    _validate_required_slots(
        tuple(barrier.slots),
        prepared_slots,
        error_cls=CheckpointCompatibilityError,
    )
    _validate_expected_slot_set(barrier, validation_slots)
    mapped_source_drops = sorted(
        slot
        for slot, axes in source_axes.items()
        if axes and slot not in {spec.slot for spec in barrier.slots}
    )
    if mapped_source_drops:
        raise CheckpointCompatibilityError(
            f"checkpoint fork cannot drop mapped source slots {mapped_source_drops!r}"
        )

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
        transform_stages: dict[str, list[CheckpointForkTransformRecord]] = {
            slot: [
                _transform_record(
                    slot,
                    transforms[slot],
                    transform_meta.get(slot, {}),
                    stage="source_pre",
                )
            ]
            for slot in transforms
        }
        for slot in continuation_transformed_slots:
            transform_stages.setdefault(slot, []).append(
                CheckpointForkTransformRecord(
                    slot=slot,
                    identity="feedbax.training_checkpoint.segment_history_allocation.v1",
                    parameters=request.model_dump(mode="json", exclude_none=True)
                    if request is not None
                    else {},
                    metadata={"stage": "segment_history_allocation"},
                )
            )
        if target_slot_transform is not None:
            for slot in target_post_transformed_slots:
                target_metadata = dict(target_transform_metadata or {})
                if slot in target_only_metadata:
                    target_metadata["target_only_declaration"] = dict(target_only_metadata[slot])
                transform_stages.setdefault(slot, []).append(
                    _transform_record(
                        slot,
                        target_slot_transform,
                        target_metadata,
                        stage="target_post",
                    )
                )
        transformed_slots = (
            set(transforms) | continuation_transformed_slots | target_post_transformed_slots
        )

        for spec in barrier.slots:
            if spec.slot not in prepared_slots and not spec.required:
                continue
            if spec.slot not in prepared_slots:
                raise CheckpointCompatibilityError(
                    f"target checkpoint slot {spec.slot!r} is missing after transforms"
                )
            if spec.slot in transformed_slots:
                source_slot = source_slots_by_name.get(spec.slot)
                expected_axes = target_axes.get(spec.slot)
                if source_slot is not None and source_slot.materialized_axes != expected_axes:
                    raise CheckpointCompatibilityError(
                        f"checkpoint fork slot {spec.slot!r} cannot change mapped axes; "
                        f"source={source_slot.materialized_axes!r} target={expected_axes!r}"
                    )
                slot_record, content_digest = _write_fresh_slot_blob(
                    spec,
                    prepared_slots[spec.slot],
                    blob_dir=blob_dir,
                    transaction_dir=tmp_dir,
                    coordinate=coordinate,
                    slot_roles=slot_roles,
                    population_slots=population_slots,
                    population_member_ids=expected_population_member_ids or {},
                    materialized_axes=expected_axes,
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
                    transform=_combine_transform_stages(
                        spec.slot,
                        transform_stages[spec.slot],
                    ),
                    source_axes=(
                        source_slot.materialized_axes if source_slot is not None else None
                    ),
                    target_axes=expected_axes,
                )
            else:
                source_slot = source_slots_by_name.get(spec.slot)
                if source_slot is None:
                    raise CheckpointCompatibilityError(
                        f"source checkpoint does not contain target slot {spec.slot!r}"
                    )
                expected_axes = target_axes.get(spec.slot)
                if source_slot.materialized_axes != expected_axes:
                    raise CheckpointCompatibilityError(
                        f"checkpoint fork slot {spec.slot!r} mapped axes differ for exact "
                        f"transfer; source={source_slot.materialized_axes!r} "
                        f"target={expected_axes!r}"
                    )
                _validate_slot_axes(
                    spec.slot,
                    prepared_slots[spec.slot],
                    expected_axes,
                    error_cls=CheckpointCompatibilityError,
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
                    materialized_axes=expected_axes,
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
                    source_axes=source_slot.materialized_axes,
                    target_axes=expected_axes,
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
        manifest_metadata.update(dict(metadata or {}))
        if request is not None:
            manifest_metadata["checkpoint_continuation"] = request.model_dump(
                mode="json",
                exclude_none=True,
            )
            manifest_metadata["checkpoint_continuation_applied"] = continuation_applied
        else:
            manifest_metadata.pop("checkpoint_continuation_applied", None)
        _validate_program_step_units(
            coordinate,
            manifest_metadata,
            context="checkpoint fork target",
        )
        completed_batches = _resolve_completed_training_batches(
            phase_program=phase_program,
            slots=prepared_slots,
            # A declared continuation creates a new target horizon.  The source
            # manifest records the completed source prefix and must never be
            # reused as the target custody total.  The target's declared
            # bookkeeping slot remains an equality assertion below.
            explicit=(
                request.target_total
                if request is not None and continuation_applied
                else request.source_completed_batches if request is not None else None
            ),
            metadata=manifest_metadata,
            default=None if request is not None else source.manifest.completed_training_batches,
            slot_axis_bindings=target_axes,
        )
        _validate_batch_histories(
            prepared_slots,
            segment_batch_count=(
                request.additional_batches
                if request is not None and continuation_applied
                else completed_batches
            ),
        )
        manifest = CheckpointTransactionManifest(
            transaction_id=transaction_id,
            run_id=coordinate.run_id,
            status=source.manifest.status,
            barrier=barrier.name,
            completed_coordinate=coordinate,
            completed_training_batches=completed_batches,
            segment_lineage=CheckpointSegmentLineage(
                parent_transaction_id=(
                    source.manifest.transaction_id
                    if request is not None and continuation_applied
                    else None
                ),
                start_batch=(
                    request.source_completed_batches
                    if request is not None and continuation_applied
                    else 0
                ),
                segment_batch_count=(
                    request.additional_batches
                    if request is not None and continuation_applied
                    else (completed_batches or 0)
                ),
                history_granularities=_history_granularities(prepared_slots),
            ),
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
                    slot_content_digests=list(source.manifest.content_integrity_digest.slots),
                ),
                slots=slot_provenance,
                tool_version=tool_version or feedbax_version(),
                barrier_mapping=resolved_barrier_mapping,
                metadata=dict(fork_provenance_metadata or {}),
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
            continuation_request=(request if request is not None and not continuation_applied else None),
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
        loaded_slots[slot.slot] = _deserialize_checkpoint_slot(slot, blob_path)
    provenance_notices, _ = _validate_manifest_structural_abi(manifest, loaded_slots)
    return _LoadedCheckpointTransaction(
        root=root_path,
        latest_pointer=latest,
        manifest_path=manifest_path,
        manifest=manifest,
        slots=loaded_slots,
        provenance_notices=tuple(provenance_notices),
    )


def _load_checkpoint_transaction_by_id(
    root: str | Path,
    transaction_id: str,
) -> _LoadedCheckpointTransaction:
    root_path = Path(root)
    manifest_path = root_path / TRANSACTIONS_DIR_NAME / transaction_id / MANIFEST_NAME
    if not manifest_path.is_file():
        raise CheckpointIntegrityError(
            f"checkpoint segment lineage parent is unavailable; transaction={transaction_id!r}"
        )
    manifest = _load_transaction_manifest(manifest_path)
    if manifest.transaction_id != transaction_id:
        raise CheckpointIntegrityError(
            f"checkpoint transaction directory identity mismatch; expected={transaction_id!r}"
        )
    slots: dict[str, Any] = {}
    for slot in manifest.slots:
        slots[slot.slot] = pickle.loads(_read_blob(slot, manifest_path.parent / slot.relative_path))
    return _LoadedCheckpointTransaction(
        root=root_path,
        latest_pointer=CheckpointLatestPointer(
            run_id=manifest.run_id,
            transaction_id=transaction_id,
            manifest_relative_path=str(manifest_path.relative_to(root_path)),
            manifest_sha256=_sha256_file(manifest_path),
            transaction_root_sha256=manifest.content_integrity_digest.transaction_root_sha256,
            completed_coordinate=manifest.completed_coordinate,
            completed_training_batches=manifest.completed_training_batches,
        ),
        manifest_path=manifest_path,
        manifest=manifest,
        slots=slots,
        provenance_notices=(),
    )


def concatenate_checkpoint_histories(
    terminal_root: str | Path,
    *,
    parent_roots: Mapping[str, str | Path],
) -> ConcatenatedCheckpointHistories:
    """Walk and concatenate a checkpoint segment lineage without mutating custody.

    ``parent_roots`` deliberately resolves transaction identities outside the
    manifests: local paths are deployment details, not durable artifact identity.
    """
    newest = _load_latest_checkpoint_transaction(terminal_root)
    reversed_segments = [newest]
    seen = {newest.manifest.transaction_id}
    while newest.manifest.segment_lineage.parent_transaction_id is not None:
        parent_id = newest.manifest.segment_lineage.parent_transaction_id
        if parent_id in seen:
            raise CheckpointIntegrityError(
                f"checkpoint segment lineage contains duplicate/cycle transaction={parent_id!r}"
            )
        parent_root = parent_roots.get(parent_id, newest.root)
        if parent_root is None:
            raise CheckpointIntegrityError(
                f"checkpoint segment lineage parent is unavailable; transaction={parent_id!r}"
            )
        parent = _load_checkpoint_transaction_by_id(parent_root, parent_id)
        if parent.manifest.transaction_id != parent_id:
            raise CheckpointIntegrityError(
                "checkpoint segment lineage parent root resolved the wrong transaction; "
                f"expected={parent_id!r} actual={parent.manifest.transaction_id!r}"
            )
        parent_total = (
            parent.manifest.segment_lineage.start_batch
            + parent.manifest.segment_lineage.segment_batch_count
        )
        if newest.manifest.segment_lineage.start_batch != parent_total:
            raise CheckpointIntegrityError(
                "checkpoint segment lineage offset discontinuity; "
                f"parent_total={parent_total} child_start="
                f"{newest.manifest.segment_lineage.start_batch}"
            )
        reversed_segments.append(parent)
        seen.add(parent_id)
        newest = parent
    segments = list(reversed(reversed_segments))
    if segments[0].manifest.segment_lineage.start_batch != 0:
        raise CheckpointIntegrityError("checkpoint segment lineage root must start at batch zero")

    per_segment: list[dict[str, BatchHistory[Any]]] = []
    for segment in segments:
        found: dict[str, BatchHistory[Any]] = {}
        for slot, value in segment.slots.items():
            pairs, _ = jt.flatten_with_path(value, is_leaf=is_type(BatchHistory))
            for path, leaf in pairs:
                if isinstance(leaf, BatchHistory):
                    found[f"{slot}{_key_path_to_text(path)}"] = leaf
        declared = segment.manifest.segment_lineage.history_granularities
        actual = {path: history.granularity.interval for path, history in found.items()}
        if actual != declared:
            raise CheckpointIntegrityError(
                "checkpoint segment history granularities do not match manifest; "
                f"transaction={segment.manifest.transaction_id!r}"
            )
        per_segment.append(found)
    keys = set(per_segment[0])
    if any(set(histories) != keys for histories in per_segment[1:]):
        raise CheckpointIntegrityError("checkpoint segment history paths differ across lineage")
    stitched: dict[str, BatchHistory[Any]] = {}
    for path in sorted(keys):
        histories = [segment[path] for segment in per_segment]
        first = histories[0]
        if any(
            history.batch_axis != first.batch_axis
            or history.granularity.interval != first.granularity.interval
            for history in histories[1:]
        ):
            raise CheckpointIntegrityError(
                f"checkpoint segment history granularity mismatch; path={path!r}"
            )
        stitched[path] = BatchHistory(
            jnp.concatenate([history.value for history in histories], axis=first.batch_axis),
            batch_axis=first.batch_axis,
            granularity=first.granularity,
        )
    total = segments[-1].manifest.segment_lineage.start_batch + (
        segments[-1].manifest.segment_lineage.segment_batch_count
    )
    return ConcatenatedCheckpointHistories(
        histories=stitched,
        transaction_ids=tuple(segment.manifest.transaction_id for segment in segments),
        completed_training_batches=total,
    )


def materialize_concatenated_checkpoint_histories(
    terminal_root: str | Path,
    output_path: str | Path,
    *,
    parent_roots: Mapping[str, str | Path],
) -> Path:
    """Write an explicitly derived, provenance-stamped stitched history product."""
    result = concatenate_checkpoint_histories(terminal_root, parent_roots=parent_roots)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "feedbax.derived.checkpoint_histories.v1",
        "derived": True,
        "resume_source": False,
        "source_transaction_ids": result.transaction_ids,
        "completed_training_batches": result.completed_training_batches,
        "histories": result.histories,
    }
    _write_bytes_atomic(output, pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    return output


def _apply_slot_transforms(
    loaded_slots: Mapping[str, Any],
    *,
    transforms: Mapping[str, ResumeSlotTransform],
) -> dict[str, Any]:
    transformed_slots = dict(loaded_slots)
    for slot, transform in transforms.items():
        if slot not in transformed_slots:
            raise CheckpointCompatibilityError(f"cannot transform missing checkpoint slot {slot!r}")
        try:
            transformed_mapping = dict(transform(transformed_slots))
        except CheckpointCustodyError:
            raise
        except Exception as exc:
            raise CheckpointCompatibilityError(
                f"checkpoint fork transform failed for slot {slot!r}"
            ) from exc
        if slot not in transformed_mapping:
            raise CheckpointCompatibilityError(f"checkpoint fork transform dropped slot {slot!r}")
        transformed_slots[slot] = transformed_mapping[slot]
    return transformed_slots


def _validate_target_transform_metadata(metadata: Mapping[str, Any] | None) -> None:
    """Require durable provenance for a topology-changing target transform."""
    assert metadata is not None
    identity = metadata.get("identity")
    if not isinstance(identity, str) or not identity:
        raise CheckpointCompatibilityError(
            "target_transform_metadata.identity must be a non-empty string"
        )
    parameters = metadata.get("parameters", {})
    if not isinstance(parameters, Mapping):
        raise CheckpointCompatibilityError("target_transform_metadata.parameters must be a mapping")


def _apply_target_slot_transform(
    slots: Mapping[str, Any],
    *,
    transform: ResumeSlotTransform,
    declared_transformed_slots: set[str],
    declared_target_only_slots: set[str],
) -> tuple[dict[str, Any], set[str]]:
    """Apply the target/post stage and fail closed on topology drift.

    Source slots may only be changed when listed in
    ``declared_transformed_slots``. New root slots may only be introduced when
    declared target-only; every declared target-only slot must be initialized
    by this caller-owned transform. Source slots cannot be dropped.
    """
    source_slots = dict(slots)
    unknown_declared = declared_transformed_slots - set(source_slots)
    if unknown_declared:
        raise CheckpointCompatibilityError(
            "target_transformed_slots names missing source slots; "
            f"slots={sorted(unknown_declared)!r}"
        )
    overlap = declared_target_only_slots & set(source_slots)
    if overlap:
        raise CheckpointCompatibilityError(
            f"target-only slots already exist in source topology; slots={sorted(overlap)!r}"
        )
    try:
        transformed = dict(transform(source_slots))
    except CheckpointCustodyError:
        raise
    except Exception as exc:
        raise CheckpointCompatibilityError("checkpoint fork target transform failed") from exc

    dropped = set(source_slots) - set(transformed)
    if dropped:
        raise CheckpointCompatibilityError(
            f"checkpoint fork target transform dropped source slots; slots={sorted(dropped)!r}"
        )
    added = set(transformed) - set(source_slots)
    undeclared_added = added - declared_target_only_slots
    if undeclared_added:
        raise CheckpointCompatibilityError(
            "checkpoint fork target transform introduced undeclared target-only slots; "
            f"slots={sorted(undeclared_added)!r}"
        )
    missing_target_only = declared_target_only_slots - added
    if missing_target_only:
        raise CheckpointCompatibilityError(
            "checkpoint fork target transform did not initialize declared target-only slots; "
            f"slots={sorted(missing_target_only)!r}"
        )
    changed_source_slots = {
        slot for slot, source_value in source_slots.items() if transformed[slot] is not source_value
    }
    undeclared_changes = changed_source_slots - declared_transformed_slots
    if undeclared_changes:
        raise CheckpointCompatibilityError(
            "checkpoint fork target transform changed undeclared source slots; "
            f"slots={sorted(undeclared_changes)!r} contract=target_transformed_slots"
        )
    return transformed, changed_source_slots | added


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


def _continuation_was_applied(
    manifest: CheckpointTransactionManifest,
    request: CheckpointContinuationRequest,
) -> bool:
    """Return whether this fork already applied exactly ``request``."""
    marker = manifest.metadata.get("checkpoint_continuation_applied")
    if marker is None:
        return False
    if not isinstance(marker, bool):
        raise CheckpointCompatibilityError(
            "checkpoint continuation applied marker must be boolean true or false; "
            f"value={marker!r}"
        )
    recorded_payload = manifest.metadata.get("checkpoint_continuation")
    state = "applied" if marker else "pending"
    if not isinstance(recorded_payload, Mapping):
        raise CheckpointCompatibilityError(
            f"checkpoint continuation is marked {state} but recorded request is missing"
        )
    try:
        recorded = CheckpointContinuationRequest.model_validate(recorded_payload)
    except ValidationError as exc:
        raise CheckpointCompatibilityError(
            f"checkpoint continuation is marked {state} but recorded request is invalid"
        ) from exc
    if recorded != request:
        raise CheckpointCompatibilityError(
            "checkpoint continuation request does not match the "
            f"{'already-applied' if marker else 'pending'} "
            "fork contract; "
            f"recorded={recorded.model_dump(mode='json', exclude_none=True)!r} "
            f"requested={request.model_dump(mode='json', exclude_none=True)!r}"
        )
    return marker


def _coerce_barrier_mapping(
    value: CheckpointForkBarrierMapping | Mapping[str, Any] | None,
) -> CheckpointForkBarrierMapping | None:
    """Validate an optional durable source-to-target barrier mapping."""
    if value is None:
        return None
    if isinstance(value, CheckpointForkBarrierMapping):
        return value
    try:
        return CheckpointForkBarrierMapping.model_validate(value)
    except ValidationError as exc:
        raise CheckpointCompatibilityError(
            f"checkpoint fork barrier mapping is invalid: {exc}"
        ) from exc


def _resolve_fork_continuation_request(
    *,
    target_run_spec: TrainingRunSpec,
    continuation_request: CheckpointContinuationRequest | Mapping[str, Any] | None,
) -> CheckpointContinuationRequest | None:
    """Resolve the target-bound continuation contract for a checkpoint fork.

    A fork is published under ``target_run_spec`` and therefore must not omit or
    replace that spec's continuation declaration.
    """
    declared = target_run_spec.checkpoint_progress.continuation
    supplied = _coerce_continuation_request(continuation_request)
    if declared is None:
        return supplied
    if supplied is not None and supplied != declared:
        raise CheckpointCompatibilityError(
            "checkpoint fork continuation request differs from target run spec; "
            "contract=checkpoint_progress.continuation"
        )
    return declared


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
    materialized_axes: tuple[MaterializedSlotAxisBinding, ...] | None = None,
) -> tuple[CheckpointSlotBlobRef, SlotContentDigest]:
    _validate_slot_axes(
        spec.slot,
        value,
        materialized_axes,
        error_cls=CheckpointCompatibilityError,
    )
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
        materialized_axes=materialized_axes,
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
    *,
    stage: str | None = None,
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
    if stage is not None:
        record_metadata["stage"] = stage
    return CheckpointForkTransformRecord(
        slot=slot,
        identity=identity,
        parameters=dict(parameters),
        metadata=record_metadata,
    )


def _combine_transform_stages(
    slot: str,
    stages: Sequence[CheckpointForkTransformRecord],
) -> CheckpointForkTransformRecord:
    """Keep ordered stage provenance without widening the manifest schema.

    ``CheckpointForkSlotProvenance`` historically has one transform field. The
    final record remains its compatibility projection while the existing
    metadata surface carries every stage in order.
    """
    if not stages:
        raise CheckpointCompatibilityError(
            f"checkpoint fork transformed slot {slot!r} has no provenance stages"
        )
    final = stages[-1]
    metadata = dict(final.metadata)
    metadata["stages"] = [
        {
            "stage": record.metadata.get("stage"),
            "identity": record.identity,
            "parameters": record.parameters,
            "metadata": {key: value for key, value in record.metadata.items() if key != "stage"},
        }
        for record in stages
    ]
    return CheckpointForkTransformRecord(
        slot=slot,
        identity=final.identity,
        parameters=final.parameters,
        metadata=metadata,
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
        algorithm_version=_RUN_CONTRACT_BINDING_ALGORITHM_V3,
        training_run_spec_schema_id=training_run_spec["schema_id"],
        training_run_spec_schema_version=training_run_spec["schema_version"],
        training_run_spec_sha256=_run_contract_hash(training_run_spec),
        method_payload_schema_id=method_payload["schema_id"],
        method_payload_schema_version=method_payload["schema_version"],
        method_payload_sha256=_run_contract_hash(method_payload),
        phase_program_sha256=_run_contract_hash(phase_program),
        objective_sha256=_run_contract_hash(objective),
        graph_sha256=_run_contract_hash(graph),
        optimizer_bindings_sha256=_run_contract_hash(optimizer_bindings),
        canonical_projection=projection,
        canonical_projection_sha256=_run_contract_hash(projection),
    )


def run_contract_canonical_projection(
    run_spec: TrainingRunSpec,
    phase_program: PhaseProgramSpec,
) -> dict[str, Any]:
    """Return the stored semantic projection for run-contract binding v3.

    The projection intentionally includes the full migrated ``TrainingRunSpec``
    payload and the phase program used by the checkpoint barrier. No
    non-semantic fields are excluded today; adding exclusions would be a
    binding-algorithm change and must bump ``RunContractBinding.algorithm_version``.
    The projection-envelope algorithm remains v2 because its selection and
    shape did not change in v3; v3 normalizes signed zero only while hashing.
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
    # This local import avoids the preparation/checkpoint-custody import cycle.
    from feedbax.training.preparation import _thaw_runtime_value

    value = _thaw_runtime_value(value)
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
    leaf_diff_text = "; ".join(_format_structural_abi_leaf_diff(diff) for diff in displayed_diffs)
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
            if field == "leaf_type":
                recorded_value = _canonical_leaf_type(recorded_value)
                actual_value = _canonical_leaf_type(actual_value)
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
                f"checkpoint slot {slot.slot!r} structural ABI mismatch{diff_suffix}"
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
        if _semantic_structural_abi_sha256(
            loaded_fingerprint
        ) != _semantic_structural_abi_sha256(slot.structural_abi_fingerprint):
            diff_suffix = _format_structural_abi_diff(
                slot.structural_abi_fingerprint,
                loaded_fingerprint,
            )
            raise CheckpointIntegrityError(
                f"checkpoint slot {slot.slot!r} structural ABI fingerprint is stale{diff_suffix}"
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
        expected.algorithm_version != _RUN_CONTRACT_BINDING_ALGORITHM_V3
        or expected.hash_domain != _RUN_CONTRACT_HASH_DOMAIN
        or recorded.algorithm_version
        not in {_RUN_CONTRACT_BINDING_ALGORITHM_V2, _RUN_CONTRACT_BINDING_ALGORITHM_V3}
        or recorded.hash_domain != _RUN_CONTRACT_HASH_DOMAIN
    ):
        return False
    if (
        recorded.canonical_projection_sha256 is not None
        and expected.canonical_projection_sha256 is not None
    ):
        if recorded.algorithm_version == _RUN_CONTRACT_BINDING_ALGORITHM_V3:
            return (
                recorded.canonical_projection_sha256
                == expected.canonical_projection_sha256
            )
        return _compatible_stored_canonical_projection(recorded, expected)

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


def _compatible_stored_canonical_projection(
    recorded: RunContractBinding,
    expected: RunContractBinding,
) -> bool:
    """Prove equality for a binding stored before signed-zero normalization."""
    if (
        recorded.algorithm_version != _RUN_CONTRACT_BINDING_ALGORITHM_V2
        or expected.algorithm_version != _RUN_CONTRACT_BINDING_ALGORITHM_V3
        or recorded.hash_domain != _RUN_CONTRACT_HASH_DOMAIN
        or expected.hash_domain != _RUN_CONTRACT_HASH_DOMAIN
    ):
        return False
    projection = recorded.canonical_projection
    recorded_sha256 = recorded.canonical_projection_sha256
    expected_projection = expected.canonical_projection
    if projection is None or recorded_sha256 is None or expected_projection is None:
        return False

    # The stored projection is evidence only when its content agrees with its
    # binding. Accept either the legacy lexical hash or the normalized current
    # hash; never use projection equality to excuse a stale or forged digest.
    valid_projection_hashes = {
        _canonical_hash(projection),
        _run_contract_hash(projection),
    }
    if recorded_sha256 not in valid_projection_hashes:
        return False
    return canonical_json_bytes(_normalize_signed_zero(projection)) == canonical_json_bytes(
        _normalize_signed_zero(expected_projection)
    )


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
        field for field in recorded_hashes if recorded_hashes[field] == comparison_hashes.get(field)
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
        # Legacy bindings used lexical canonical JSON, including its signed-zero
        # spelling. Keep this reconstruction exact for projection-less records.
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
    return f"{path}: recorded={_short_json(recorded)}, expected={_short_json(expected)}"


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
        and left.program_step == right.program_step
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


def _deserialize_checkpoint_slot(slot: CheckpointSlotBlobRef, path: Path) -> Any:
    """Deserialize one hash-verified slot blob without resume transformations."""
    blob_bytes = _read_blob(slot, path)
    try:
        return pickle.loads(blob_bytes)
    except Exception as exc:
        raise CheckpointIntegrityError(
            f"checkpoint slot {slot.slot!r} could not be deserialized"
        ) from exc


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
        return load_checkpoint_latest_pointer_file(path).document
    except CheckpointIntegrityError as exc:
        # Keep the custody reader's established boundary contract while the
        # public loader retains its more specific typed diagnostic.
        raise CheckpointIntegrityError("checkpoint latest pointer is corrupt") from exc


def _reject_legacy_checkpoint(root: Path) -> None:
    layout = detect_known_legacy_checkpoint_layout(root)
    if layout is not None:
        raise CheckpointCompatibilityError(_legacy_checkpoint_adoption_message(layout))


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
    return load_checkpoint_transaction_manifest_file(path).document


def _slot_roles(run_spec: TrainingRunSpec) -> dict[str, str]:
    return {slot.name: slot.role for slot in run_spec.worker_execution.method_contract.state_slots}


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
            leaf_type=_JAX_ARRAY_LEAF_TYPE,
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
        "fingerprint_algorithm_version": ("feedbax.training_checkpoint.structural_abi.content.v2"),
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


def _semantic_structural_abi_sha256(fingerprint: StructuralAbiFingerprint) -> str:
    leaves = [
        leaf.model_copy(update={"leaf_type": _canonical_leaf_type(leaf.leaf_type)})
        for leaf in fingerprint.leaves
    ]
    return _canonical_hash(_structural_abi_content_payload(fingerprint.treedef, leaves))


def _canonical_leaf_type(leaf_type: str) -> str:
    if leaf_type in ("jaxlib.xla_extension.ArrayImpl", "jaxlib._jax.ArrayImpl"):
        return _JAX_ARRAY_LEAF_TYPE
    return leaf_type


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


def _run_contract_hash(value: Any) -> str:
    """Hash run-contract content after normalizing IEEE signed zero."""
    return _canonical_hash(_normalize_signed_zero(value))


def _normalize_signed_zero(value: Any) -> Any:
    """Return JSON-shaped content with every floating signed zero normalized."""
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, float):
        return 0.0 if value == 0.0 else value
    if isinstance(value, Mapping):
        return {key: _normalize_signed_zero(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_signed_zero(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_normalize_signed_zero(item) for item in value)
    return value


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
