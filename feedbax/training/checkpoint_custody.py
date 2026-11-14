"""Atomic training checkpoint custody helpers."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import logging
import os
import pickle
import platform
import re
import shutil
import sys
import tempfile
import uuid
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field as dataclass_field, fields as dataclass_fields
from dataclasses import is_dataclass
from datetime import datetime, timedelta, timezone
from functools import partial
from pathlib import Path, PurePosixPath
from types import FunctionType, MappingProxyType
from typing import Any, Literal
from urllib.parse import unquote, urlsplit

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree as jt
from jax.tree_util import PyTreeDef
from feedbax._vendor.jax_cookbook import is_type
import numpy as np
from pydantic import BaseModel, ConfigDict, ValidationError

from feedbax._atomic import sync_parent_directories
from feedbax._atomic import write_bytes_atomic as _write_bytes_atomic
from feedbax._secure_fs import (
    SecurePathRigor,
    directory_open_flags,
    open_directory_chain,
    open_existing_directory,
    open_existing_file,
    take_directory_chain_leaf,
)
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
    CheckpointSemanticAbi,
    CheckpointSegmentLineage,
    CheckpointProvenanceNotice,
    CheckpointSlotBlobRef,
    CheckpointTransactionManifest,
    CHECKPOINT_CONTENT_INTEGRITY_SCHEMA_ID,
    CHECKPOINT_CONTENT_INTEGRITY_SCHEMA_VERSION,
    CHECKPOINT_CONTENT_INTEGRITY_SCHEMA_VERSION_V1,
    ContentIntegrityDigest,
    LegacySlotContentDigest,
    PHASE_BOUNDARY_DECISION_METADATA_KEY,
    PopulationIdentityRecord,
    RunContractBinding,
    SerializerVersionRecord,
    SlotContentDigest,
    SlotLeafContentDigest,
    SlotLeafFingerprint,
    SEMANTIC_STATE_ROOT_ALGORITHM_VERSION,
    StructuralAbiFingerprint,
    structural_abi_content_projection,
    structural_abi_content_sha256,
    structural_abi_leaf_content_projection,
)
from feedbax.contracts.checkpoint_initialization import (
    INITIALIZE_FROM_SOURCE_ROLES,
    checkpoint_leaf_path,
    derive_checkpoint_semantic_abis,
    validate_checkpoint_semantic_abi,
)
from feedbax.contracts.artifact_custody import (
    ArtifactBlobProvider,
    ContentBinding,
    ContentObjectRef,
    ContentRootRecord,
)
from feedbax.contracts.base import (
    ArtifactMigrationRecord,
    ParentRef,
    authenticated_manifest_ref_metadata,
    canonical_json_bytes,
    feedbax_version,
    sha256_bytes,
    sha256_file,
)
from feedbax.contracts.publication import (
    BlobRef,
    CheckpointSet,
    CheckpointSlot,
    ExactRef,
    checkpoint_set_id,
)
from feedbax.contracts.run_matrix import (
    ContinuationReconciliation,
    ExecutionDependency,
    ExecutionDependencyV6,
    ExecutionHashForkSourceAuthority,
    ForkFromSelectedCheckpoint,
    ForkFromSelectedCheckpointV6,
    LineageGraftDependency,
    StoppedRowStatus,
    TaskIdentityGate,
)
from feedbax.contracts.strict_json import (
    DuplicateJsonKeyError,
    strict_model_validate_json,
)
from feedbax.contracts.training import (
    TrainingRunSpec,
)
from feedbax.contracts.studio_training import STUDIO_GRAPH_METHOD
from feedbax.contracts.worker import (
    CheckpointBarrierSpec,
    CheckpointSlotSpec,
    MaterializedSlotAxisBinding,
    PhaseProgramSpec,
    ProgressCoordinate,
    StateSlotSpec,
    derive_consistency_predicate,
)
from feedbax.training.metrics import normalize_serialized_metrics
from feedbax.training.checkpoint_errors import CheckpointCustodyError
from feedbax.training.model_checkpoint import (
    MODEL_TREE_MEDIA_TYPE,
    MODEL_TREE_SERIALIZER,
    SemanticModelSlot,
    SemanticModelWriter,
    semantic_model_member_files,
)
from feedbax.training.standard_train_checkpoint import (
    STANDARD_TRAIN_SLOTS,
    STANDARD_TRAIN_STATE_CODEC_ID,
    STANDARD_TRAIN_STATE_CODEC_VERSION,
    STANDARD_TRAIN_STATE_MEDIA_TYPE,
    decode_standard_train_slot,
    encode_standard_train_slot,
    standard_train_optimizer_template,
    standard_train_scalar_template,
)
from feedbax.training.worker_validation import resolve_execution_mapping


LATEST_POINTER_NAME = "latest.json"
CheckpointReadScope = Literal["full", "model_only"] | frozenset[str]


def _checkpoint_content_kind(scope: CheckpointReadScope) -> str:
    if isinstance(scope, frozenset):
        return "checkpoint-custody:" + ",".join(sorted(scope))
    return "checkpoint-model-custody" if scope == "model_only" else "checkpoint-custody"
TRANSACTIONS_DIR_NAME = "transactions"
MANIFEST_NAME = "manifest.json"
CHECKPOINT_SET_NAME = "checkpoint-set.json"
STRUCTURAL_ABI_PROVENANCE = "feedbax.training.checkpoint.structural_abi.v1"
_LOGGER = logging.getLogger(__name__)
_STRUCTURAL_ABI_DIFF_FIELDS = (
    "dtype",
    "shape",
    "weak_type",
    "leaf_type",
    "static_repr_sha256",
)
_JAX_ARRAY_LEAF_TYPE = "jax.Array"
_RUN_CONTRACT_BINDING_ALGORITHM_V2 = "feedbax.training_checkpoint.run_contract_binding.v2"
_RUN_CONTRACT_BINDING_ALGORITHM_V3 = "feedbax.training_checkpoint.run_contract_binding.v3"
_RUN_CONTRACT_BINDING_ALGORITHM_V4 = "feedbax.training_checkpoint.run_contract_binding.v4"
_RUN_CONTRACT_HASH_DOMAIN = "migrated-canonical-json"
_STRUCTURAL_ABI_SCHEMA_ID = StructuralAbiFingerprint.model_fields["schema_id"].default
_STRUCTURAL_ABI_SCHEMA_VERSION = StructuralAbiFingerprint.model_fields["schema_version"].default
_STRUCTURAL_ABI_ALGORITHM_VERSION = StructuralAbiFingerprint.model_fields[
    "fingerprint_algorithm_version"
].default
_READ_ONLY_MODEL_TYPES: dict[type[BaseModel], type[BaseModel]] = {}


def _reject_snapshot_mutation(*_args: Any, **_kwargs: Any) -> None:
    raise TypeError("verified authenticated snapshots are immutable")


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

    def __copy__(self) -> "_FrozenDict":
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> dict[Any, Any]:
        copied: dict[Any, Any] = {}
        memo[id(self)] = copied
        copied.update(
            (copy.deepcopy(key, memo), copy.deepcopy(value, memo)) for key, value in self.items()
        )
        return copied


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

    def __copy__(self) -> "_FrozenList":
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> list[Any]:
        copied: list[Any] = []
        memo[id(self)] = copied
        copied.extend(copy.deepcopy(item, memo) for item in self)
        return copied


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


class CheckpointSemanticStateRootUnavailable(CheckpointCustodyError):
    """Raised when a legacy checkpoint has no governed semantic-state identity."""


class CheckpointSemanticStateUnsupported(CheckpointCustodyError):
    """Raised when a checkpoint leaf cannot enter semantic-state identity."""


class CheckpointSemanticStateMismatch(CheckpointCustodyError):
    """Raised when two governed checkpoint semantic-state roots differ."""


class CheckpointContractBindingError(CheckpointCustodyError):
    """Raised when a checkpoint belongs to a different run contract."""


class CheckpointConsistencyError(CheckpointCustodyError):
    """Raised when slot coordinates violate the method-declared consistency predicate."""


class CheckpointCustodyMutationError(CheckpointCustodyError):
    """Raised when a custody directory that must stay immutable was mutated.

    This is the mutation tripwire for side-effect-free rehearsal probes: it
    fires when a directory fingerprinted as an immutable probe input differs
    afterwards, so an isolation regression is reported at probe time rather than
    discovered later when custody archiving refuses a stale ``latest.json``.
    """


@dataclass(frozen=True)
class CheckpointWriteResult:
    """Paths and manifest returned by a successful checkpoint write."""

    root: Path
    transaction_dir: Path
    manifest_path: Path
    latest_pointer_path: Path
    manifest: CheckpointTransactionManifest
    latest_pointer: CheckpointLatestPointer
    checkpoint_set_path: Path
    checkpoint_set: CheckpointSet
    content_binding: ContentBinding | None = None
    content_root: ContentRootRecord | None = None


ResumeSlotTransform = Callable[[Mapping[str, Any]], Mapping[str, Any]]


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


def _published_checkpoint_set(
    *,
    run_spec: TrainingRunSpec,
    manifest: CheckpointTransactionManifest,
    parent: ExactRef | None = None,
) -> CheckpointSet:
    """Project the internal transaction protocol into its public checkpoint identity."""
    binding = manifest.run_contract_binding
    projection = binding.canonical_projection
    if projection is None or binding.graph_sha256 is None:
        raise CheckpointCustodyError(
            "checkpoint publication requires the current canonical run-contract projection"
        )
    training_run = projection["training_run_spec"]
    graph = training_run["graph"]
    training_run_bytes = canonical_json_bytes(_normalize_signed_zero(training_run))
    graph_bytes = canonical_json_bytes(_normalize_signed_zero(graph))
    graph_ref = ExactRef(
        domain="semantic_ir",
        identity=f"sha256:{binding.graph_sha256}",
        bytes=BlobRef(digest=binding.graph_sha256, size_bytes=len(graph_bytes)),
    )
    experiment_ref = ExactRef(
        domain="document_revision",
        identity=f"sha256:{binding.training_run_spec_sha256}",
        bytes=BlobRef(
            digest=binding.training_run_spec_sha256,
            size_bytes=len(training_run_bytes),
        ),
    )
    slots = tuple(
        CheckpointSlot(
            name=slot.slot,
            state_type=slot.role,
            array_structure_id=slot.structural_abi_fingerprint.fingerprint_sha256,
            codec_schema_id={
                MODEL_TREE_MEDIA_TYPE: MODEL_TREE_SERIALIZER.removesuffix(".v1"),
                STANDARD_TRAIN_STATE_MEDIA_TYPE: STANDARD_TRAIN_STATE_CODEC_ID,
            }[slot.media_type],
            codec_schema_version=_supported_slot_serializer(slot),
            blob=BlobRef(digest=slot.sha256, size_bytes=slot.size_bytes),
        )
        for slot in manifest.slots
    )
    prng_slot = next(
        (slot for slot in manifest.slots if slot.role == "prng" or slot.slot == "prng"),
        None,
    )
    parent_transaction_id = manifest.segment_lineage.parent_transaction_id
    if parent_transaction_id is None and parent is not None:
        raise CheckpointCustodyError(
            "checkpoint publication supplied a parent without transaction lineage"
        )
    if parent_transaction_id is not None and parent is None:
        raise CheckpointCustodyError(
            "continuation checkpoint requires its authenticated parent CheckpointSet"
        )
    progress = {
        key: value
        for key, value in manifest.completed_coordinate.model_dump(mode="json").items()
        if value is not None and isinstance(value, (int, float, str))
    }
    if manifest.completed_training_batches is not None:
        progress["completed_training_batches"] = manifest.completed_training_batches
    manifest_bytes = _durable_json_bytes(manifest.model_dump(mode="json", exclude_none=True))
    values = {
        "transaction": ExactRef(
            domain="checkpoint_transaction",
            identity=manifest.transaction_id,
            bytes=BlobRef(digest=sha256_bytes(manifest_bytes), size_bytes=len(manifest_bytes)),
        ),
        "training_program_id": run_spec.method_ref.key,
        "graph": graph_ref,
        "experiment": experiment_ref,
        "progress": progress,
        "prng_state": (
            None
            if prng_slot is None
            else BlobRef(digest=prng_slot.sha256, size_bytes=prng_slot.size_bytes)
        ),
        "slots": slots,
        "continuation": "resume" if parent is not None else "fork",
        "parent": parent,
    }
    return CheckpointSet(checkpoint_id=checkpoint_set_id(**values), **values)


def load_checkpoint_set(
    root: str | Path,
    transaction_id: str,
    *,
    read_bytes: Callable[[Path], bytes] = Path.read_bytes,
) -> CheckpointSet:
    """Load one native checkpoint publication bound to its exact manifest bytes."""
    transaction_root = Path(root).expanduser().resolve() / TRANSACTIONS_DIR_NAME
    if (
        not transaction_id.startswith("tx-")
        or len(transaction_id) != 35
        or any(character not in "0123456789abcdef" for character in transaction_id[3:])
    ):
        raise CheckpointCustodyError(
            f"invalid native checkpoint transaction identity: {transaction_id!r}"
        )
    transaction_dir = transaction_root / transaction_id
    manifest_path = transaction_dir / MANIFEST_NAME
    checkpoint_set_path = transaction_dir / CHECKPOINT_SET_NAME
    try:
        manifest_bytes = read_bytes(manifest_path)
        checkpoint_set_bytes = read_bytes(checkpoint_set_path)
    except OSError as exc:
        raise CheckpointCustodyError(
            "native checkpoint transaction is missing its manifest or CheckpointSet; "
            f"transaction={transaction_id!r}"
        ) from exc
    manifest = load_checkpoint_transaction_manifest_json(
        manifest_bytes, path=str(manifest_path),
    ).document
    if manifest.transaction_id != transaction_id:
        raise CheckpointCustodyError(
            "native checkpoint manifest does not match its transaction directory"
        )
    try:
        checkpoint_set = strict_model_validate_json(CheckpointSet, checkpoint_set_bytes)
    except ValidationError as exc:
        raise CheckpointCustodyError(
            f"native checkpoint set is invalid: {checkpoint_set_path}: {exc}"
        ) from exc
    if (
        checkpoint_set.transaction.identity != transaction_id
        or checkpoint_set.transaction.bytes.digest != sha256_bytes(manifest_bytes)
        or checkpoint_set.transaction.bytes.size_bytes != len(manifest_bytes)
    ):
        raise CheckpointCustodyError(
            "native CheckpointSet does not identify its exact transaction manifest"
        )
    recorded_parent = manifest.metadata.get("checkpoint_segment_parent_ref")
    if recorded_parent is not None and (
        checkpoint_set.parent is None
        or recorded_parent != checkpoint_set.parent.model_dump(mode="json")
    ):
        raise CheckpointCustodyError(
            "native CheckpointSet parent disagrees with the authenticated segment parent"
        )
    return checkpoint_set


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
    checkpoint_set_path: Path
    checkpoint_set: CheckpointSet
    slot_transfer_modes: Mapping[str, str]
    source_provenance_notices: tuple[CheckpointProvenanceNotice, ...] = ()
    content_binding: ContentBinding | None = None
    content_root: ContentRootRecord | None = None


def _publish_checkpoint_content(
    provider: ArtifactBlobProvider,
    *,
    latest_pointer_path: Path | ContentObjectRef,
    manifest_path: Path,
    checkpoint_set_path: Path,
    manifest: CheckpointTransactionManifest,
    checkpoint_set: CheckpointSet,
    reusable_objects: Mapping[str, ContentObjectRef] | None = None,
    scope: CheckpointReadScope = "full",
) -> tuple[ContentBinding, ContentRootRecord]:
    """Publish one complete checkpoint view through the canonical content provider."""
    reusable = dict(reusable_objects or {})
    transaction_prefix = PurePosixPath(TRANSACTIONS_DIR_NAME, manifest.transaction_id)
    members: dict[str, Path | ContentObjectRef] = {
        LATEST_POINTER_NAME: latest_pointer_path,
        (transaction_prefix / MANIFEST_NAME).as_posix(): manifest_path,
        (transaction_prefix / CHECKPOINT_SET_NAME).as_posix(): checkpoint_set_path,
    }
    for slot in _checkpoint_scope_slots(manifest, scope):
        source = manifest_path.parent / slot.relative_path
        reused = reusable.get(slot.sha256)
        if reused is not None:
            if reused.size_bytes != slot.size_bytes:
                raise CheckpointIntegrityError(
                    f"canonical object size differs for checkpoint slot {slot.slot!r}"
                )
            member: Path | ContentObjectRef = reused
        else:
            member = source
        members[(transaction_prefix / slot.relative_path).as_posix()] = member
        if slot.media_type == MODEL_TREE_MEDIA_TYPE:
            for relative in semantic_model_member_files(slot, _read_blob(slot, source)):
                members[(transaction_prefix / relative.as_posix()).as_posix()] = (
                    manifest_path.parent / relative
                )

    kind = _checkpoint_content_kind(scope)
    root_id = f"checkpoint:{checkpoint_set.checkpoint_id}"
    if scope == "model_only":
        root_id += ":model-only"
    elif isinstance(scope, frozenset):
        root_id += f":{kind}"
    tree_ref, content_root = provider.publish_rooted_tree(
        members,
        root_id=root_id,
        root_kind="logical-ref",
    )
    return (
        ContentBinding(
            binding_name=manifest.transaction_id,
            root_kind=kind,
            tree=tree_ref,
        ),
        content_root,
    )


CheckpointForkPlanTransform = Callable[
    [Mapping[str, Any], Mapping[str, Any]],
    Mapping[str, Any],
]


@dataclass(frozen=True)
class CheckpointForkTransformRegistration:
    """Runtime implementation for one durable fork-transform identity."""

    identity: str
    version: str
    implementation_sha256: str
    transform: CheckpointForkPlanTransform
    owner: str = "feedbax"


class CheckpointForkTransformRegistry:
    """Exact runtime registry for durable checkpoint-fork transforms."""

    def __init__(self) -> None:
        self._registrations: dict[tuple[str, str], CheckpointForkTransformRegistration] = {}

    def register(self, registration: CheckpointForkTransformRegistration) -> None:
        if not registration.identity:
            raise ValueError("checkpoint fork transform identity must not be empty")
        if not registration.version:
            raise ValueError("checkpoint fork transform version must not be empty")
        if len(registration.implementation_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in registration.implementation_sha256
        ):
            raise ValueError("checkpoint fork transform implementation_sha256 is invalid")
        if not callable(registration.transform):
            raise TypeError(f"checkpoint fork transform {registration.identity!r} is not callable")
        key = (registration.identity, registration.version)
        if key in self._registrations:
            existing = self._registrations[key]
            raise ValueError(
                f"checkpoint fork transform {key!r} already registered by {existing.owner!r}"
            )
        self._registrations[key] = registration

    def resolve(
        self,
        identity: str,
        version: str,
        implementation_sha256: str,
    ) -> CheckpointForkTransformRegistration:
        try:
            registration = self._registrations[(identity, version)]
        except KeyError as exc:
            raise CheckpointCompatibilityError(
                f"unregistered checkpoint fork transform {(identity, version)!r}; "
                f"available identities={list(self.available_keys())!r}"
            ) from exc
        if registration.implementation_sha256 != implementation_sha256:
            raise CheckpointCompatibilityError(
                f"checkpoint fork transform {(identity, version)!r} implementation drift; "
                f"declared={implementation_sha256!r} "
                f"registered={registration.implementation_sha256!r}"
            )
        return registration

    def available_keys(self) -> tuple[tuple[str, str], ...]:
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
    semantic_model_writers: Mapping[str, SemanticModelWriter] = dataclass_field(default_factory=dict)
    source_scope: CheckpointReadScope = "full"


@dataclass(frozen=True)
class CheckpointForkDerivedDigestDifference:
    """One exact stored-versus-candidate derived-digest difference."""

    target_id: str
    path: str
    stored: str | None
    candidate: str | None


@dataclass(frozen=True)
class CheckpointForkDerivedDigestComparison:
    """Read-only candidate rebuild and deterministic stored-lock comparison."""

    candidates: Mapping[str, CheckpointForkCompatibilityProjection]
    differences: tuple[CheckpointForkDerivedDigestDifference, ...]

    @property
    def matches_stored_lock(self) -> bool:
        """Whether every stored compatibility digest equals its candidate."""
        return not self.differences


@dataclass(frozen=True)
class CheckpointForkPlanRelockResult:
    """Explicit compatibility re-lock result with required re-qualification."""

    plan: CheckpointForkPlan
    comparison: CheckpointForkDerivedDigestComparison
    requalification_requirements: tuple[str, ...]


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

    Each document result records the current schema validated from its bytes.
    """

    root: Path
    latest_pointer: CheckpointDocumentLoadResult[CheckpointLatestPointer]
    manifest_path: Path
    manifest: CheckpointDocumentLoadResult[CheckpointTransactionManifest]


@dataclass(frozen=True)
class ResolvedCheckpointTransaction:
    """Authenticated checkpoint transaction and its decoded requested slots.

    The manifest remains the lineage authority for transaction, checkpoint,
    content-root, slot-digest, and structural-ABI identities. Slot decoding
    follows authentication of the caller-provided custody authority, manifest,
    contained blob paths, and blob hashes.
    """

    parent_ref: ParentRef
    manifest_sha256: str
    manifest: CheckpointTransactionManifest
    slots: Mapping[str, Any]
    migration_records: tuple[ArtifactMigrationRecord, ...]
    provenance_notices: tuple[CheckpointProvenanceNotice, ...]


@dataclass(frozen=True)
class AuthenticatedCheckpointTransaction:
    """Checkpoint transaction authenticated without decoding slot payloads.

    This evidence is safe to derive from untrusted collected custody bytes:
    every declared blob is contained under the custody root and its exact size
    and SHA-256 digest are verified.
    """

    parent_ref: ParentRef
    manifest_sha256: str
    manifest: CheckpointTransactionManifest
    slot_names: tuple[str, ...]
    migration_records: tuple[ArtifactMigrationRecord, ...]


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
    """Load the current latest-pointer JSON schema."""
    return _load_checkpoint_document_json(
        data,
        model=CheckpointLatestPointer,
        path=path,
        document_name="checkpoint latest pointer",
    )


def load_checkpoint_latest_pointer_file(
    path: str | Path,
) -> CheckpointDocumentLoadResult[CheckpointLatestPointer]:
    """Load the current latest-pointer file."""
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
    """Load the current transaction-manifest JSON schema."""
    return _load_checkpoint_document_json(
        data,
        model=CheckpointTransactionManifest,
        path=path,
        document_name="checkpoint transaction manifest",
    )


def load_checkpoint_transaction_manifest_file(
    path: str | Path,
) -> CheckpointDocumentLoadResult[CheckpointTransactionManifest]:
    """Load the current transaction-manifest file."""
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


def authenticate_checkpoint_custody_ref(
    ref: ParentRef,
    *,
    allowed_root: str | Path,
    scope: CheckpointReadScope = "full",
) -> AuthenticatedCheckpointTransaction:
    """Authenticate one checkpoint transaction without deserializing its slots.

    This is the verification boundary for checkpoint custody collected from an
    untrusted executor or provider. It validates the exact parent reference,
    manifest identity and integrity records, path containment, and the selected
    slot bytes against their declared size and SHA-256 digest.
    """
    try:
        root_descriptor = _open_checkpoint_custody_root(allowed_root)
        try:
            manifest_parts = _checkpoint_custody_relative_parts(
                ref.uri,
                context="checkpoint custody ParentRef manifest uri",
            )
            raw_manifest = _read_checkpoint_custody_member(
                root_descriptor,
                manifest_parts,
                context="checkpoint custody ParentRef manifest",
            )
            return _authenticate_checkpoint_custody_bytes(
                ref,
                raw_manifest=raw_manifest,
                manifest_parts=manifest_parts,
                root_descriptor=root_descriptor,
                scope=scope,
            )
        finally:
            os.close(root_descriptor)
    except CheckpointReferenceResolutionError:
        raise
    except Exception as exc:
        raise CheckpointReferenceResolutionError(
            f"checkpoint custody authentication failed: {exc}"
        ) from exc


def _checkpoint_latest_parent_ref(latest: CheckpointLatestPointer) -> ParentRef:
    return ParentRef(
        kind="TrainingCheckpointTransactionManifest",
        id=latest.transaction_id,
        role="training_checkpoint_custody",
        uri=latest.manifest_relative_path,
        metadata={"manifest_sha256": latest.manifest_sha256},
    )


def authenticate_published_checkpoint_custody(
    root: str | Path,
    *,
    require_single_transaction: bool = False,
    scope: CheckpointReadScope = "full",
) -> AuthenticatedCheckpointTransaction:
    """Authenticate the published latest checkpoint without executing payloads."""
    try:
        root_descriptor = _open_checkpoint_custody_root(root)
        try:
            raw_latest = _read_checkpoint_custody_member(
                root_descriptor,
                (LATEST_POINTER_NAME,),
                context="checkpoint latest pointer",
            )
            latest_result = load_checkpoint_latest_pointer_json(
                raw_latest,
                path=LATEST_POINTER_NAME,
            )
            latest = latest_result.document
            manifest_parts = _checkpoint_custody_relative_parts(
                latest.manifest_relative_path,
                context="checkpoint latest pointer manifest_relative_path",
            )
            raw_manifest = _read_checkpoint_custody_member(
                root_descriptor,
                manifest_parts,
                context="checkpoint transaction manifest",
            )
            manifest_sha256 = sha256_bytes(raw_manifest)
            if manifest_sha256 != latest.manifest_sha256:
                raise CheckpointReferenceResolutionError(
                    "checkpoint latest pointer manifest_sha256 does not match raw manifest bytes"
                )
            authenticated = _authenticate_checkpoint_custody_bytes(
                _checkpoint_latest_parent_ref(latest),
                raw_manifest=raw_manifest,
                manifest_parts=manifest_parts,
                root_descriptor=root_descriptor,
                scope=scope,
            )
            if (
                authenticated.manifest.content_integrity_digest.transaction_root_sha256
                != latest.transaction_root_sha256
            ):
                raise CheckpointReferenceResolutionError(
                    "checkpoint latest pointer transaction root does not match manifest"
                )
            if require_single_transaction:
                transaction_root = os.open(
                    TRANSACTIONS_DIR_NAME,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=root_descriptor,
                )
                try:
                    if os.listdir(transaction_root) != [latest.transaction_id]:
                        raise CheckpointReferenceResolutionError(
                            "inherited checkpoint custody must contain only the source transaction"
                        )
                finally:
                    os.close(transaction_root)
            return authenticated
        finally:
            os.close(root_descriptor)
    except CheckpointReferenceResolutionError:
        raise
    except Exception as exc:
        raise CheckpointReferenceResolutionError(
            f"published checkpoint custody authentication failed: {exc}"
        ) from exc


@dataclass(frozen=True)
class CheckpointCustodyFingerprint:
    """Immutable identity of a checkpoint custody directory's inputs.

    ``paths`` is the sorted tuple of every regular file's POSIX-relative path
    under the custody root, so structural mutation — a new transaction
    directory, an advanced ``latest.json`` write, or a removed blob — changes the
    fingerprint. ``metadata_digests`` carries ``(relative_path, sha256)`` pairs
    over the mutable custody metadata (``latest.json`` and every transaction
    ``manifest.json``), and over every regular file when ``digested_blobs`` is
    set, so in-place content edits are detected too. Large immutable blob trees
    are enumerated but not re-hashed by default.
    """

    root_present: bool
    digested_blobs: bool
    paths: tuple[str, ...]
    metadata_digests: tuple[tuple[str, str], ...]


def _is_custody_metadata_file(relative: PurePosixPath) -> bool:
    return relative.name in (LATEST_POINTER_NAME, MANIFEST_NAME, CHECKPOINT_SET_NAME)


def fingerprint_checkpoint_custody_inputs(
    root: str | Path,
    *,
    digest_blobs: bool = False,
) -> CheckpointCustodyFingerprint:
    """Fingerprint a checkpoint custody directory before a side-effecting probe.

    The fingerprint enumerates every regular file for structural identity and
    digests the mutable custody metadata (``latest.json`` and transaction
    manifests) by default. ``digest_blobs`` extends the content digests to every
    regular file for full byte identity at the cost of hashing blob trees.
    Symlinks are recorded by path but never followed, so a probe can never write
    through a shared link into the source unnoticed.
    """
    root_path = Path(root).expanduser()
    if not root_path.exists():
        return CheckpointCustodyFingerprint(
            root_present=False,
            digested_blobs=digest_blobs,
            paths=(),
            metadata_digests=(),
        )
    paths: list[str] = []
    digests: list[tuple[str, str]] = []
    for path in sorted(root_path.rglob("*")):
        relative = path.relative_to(root_path).as_posix()
        if path.is_symlink():
            paths.append(relative)
            continue
        if not path.is_file():
            continue
        paths.append(relative)
        if digest_blobs or _is_custody_metadata_file(PurePosixPath(relative)):
            digests.append((relative, sha256_file(path)))
    return CheckpointCustodyFingerprint(
        root_present=True,
        digested_blobs=digest_blobs,
        paths=tuple(paths),
        metadata_digests=tuple(digests),
    )


def _describe_custody_fingerprint_difference(
    before: CheckpointCustodyFingerprint,
    after: CheckpointCustodyFingerprint,
) -> str:
    parts: list[str] = []
    if before.root_present != after.root_present:
        parts.append(f"root_present {before.root_present}->{after.root_present}")
    added = sorted(set(after.paths) - set(before.paths))
    removed = sorted(set(before.paths) - set(after.paths))
    if added:
        parts.append(f"added={added[:8]!r}")
    if removed:
        parts.append(f"removed={removed[:8]!r}")
    before_digests = dict(before.metadata_digests)
    after_digests = dict(after.metadata_digests)
    changed = sorted(
        name
        for name in before_digests.keys() & after_digests.keys()
        if before_digests[name] != after_digests[name]
    )
    if changed:
        parts.append(f"changed={changed[:8]!r}")
    return "; ".join(parts) or "custody content digests differ"


def assert_checkpoint_custody_unchanged(
    root: str | Path,
    before: CheckpointCustodyFingerprint,
) -> CheckpointCustodyFingerprint:
    """Assert a custody directory still matches a prior fingerprint.

    Returns the freshly computed fingerprint on success. Raises
    ``CheckpointCustodyMutationError`` with a compact difference summary when the
    custody inputs changed, using the same ``digest_blobs`` depth captured in
    ``before``.
    """
    after = fingerprint_checkpoint_custody_inputs(root, digest_blobs=before.digested_blobs)
    if after == before:
        return after
    raise CheckpointCustodyMutationError(
        f"checkpoint custody inputs at {Path(root)} changed during a probe: "
        f"{_describe_custody_fingerprint_difference(before, after)}"
    )


@dataclass(frozen=True)
class IsolatedCheckpointProbe:
    """Read-only source root and disposable output namespace for one probe."""

    source_root: Path
    output_root: Path


def _remove_probe_namespace(output_root: Path) -> OSError | None:
    try:
        shutil.rmtree(output_root)
    except FileNotFoundError:
        return None
    except OSError as exc:
        return exc
    return None


@contextmanager
def isolated_checkpoint_probe(
    source_root: str | Path,
    *,
    output_parent: str | Path | None = None,
    digest_blobs: bool = False,
) -> Iterator[IsolatedCheckpointProbe]:
    """Run a bounded checkpoint probe without mutating its source directory.

    The source custody directory is treated as immutable input: its inputs are
    fingerprinted on entry and re-asserted on a clean exit (the mutation
    tripwire). A fresh disposable output namespace is created for the probe to
    write every new transaction and its ``latest.json`` pointer into, and it is
    removed on both success and failure; a cleanup failure is surfaced but never
    endangers the source. Pass the yielded ``source_root`` as the executor's
    ``checkpoint_source_root`` and ``output_root`` as its ``checkpoint_root`` so
    the executor resumes from the read-only source and diverges every write into
    the disposable namespace.
    """
    source = Path(source_root).expanduser().resolve()
    if not (source / LATEST_POINTER_NAME).is_file():
        raise CheckpointCustodyError(
            f"isolated checkpoint probe source has no published latest pointer: {source}"
        )
    before = fingerprint_checkpoint_custody_inputs(source, digest_blobs=digest_blobs)
    parent = (
        Path(output_parent).expanduser().resolve() if output_parent is not None else source.parent
    )
    parent.mkdir(parents=True, exist_ok=True)
    output_root = Path(tempfile.mkdtemp(prefix=".rehearsal-probe-", dir=parent))
    probe = IsolatedCheckpointProbe(source_root=source, output_root=output_root)
    body_error: BaseException | None = None
    try:
        latest = _load_latest_pointer(source)
        load_checkpoint_set(source, latest.transaction_id)
        source_checkpoint_set = (
            source / TRANSACTIONS_DIR_NAME / latest.transaction_id / CHECKPOINT_SET_NAME
        )
        output_checkpoint_set = (
            output_root / TRANSACTIONS_DIR_NAME / latest.transaction_id / CHECKPOINT_SET_NAME
        )
        _write_bytes_atomic(output_checkpoint_set, source_checkpoint_set.read_bytes())
        yield probe
    except BaseException as exc:  # re-raised below after tripwire and cleanup
        body_error = exc
        raise
    finally:
        cleanup_error = _remove_probe_namespace(output_root)
        try:
            assert_checkpoint_custody_unchanged(source, before)
        except CheckpointCustodyMutationError:
            if body_error is None:
                raise
            _LOGGER.error("checkpoint probe source %s was mutated during a failed probe", source)
        if cleanup_error is not None:
            if body_error is None:
                raise cleanup_error
            _LOGGER.error("checkpoint probe namespace cleanup failed: %s", cleanup_error)


_archive_directory_flags = partial(
    directory_open_flags,
    error_factory=CheckpointReferenceResolutionError,
)


def _identity(value: os.stat_result) -> tuple[int, int]:
    return value.st_dev, value.st_ino


def _resolve_checkpoint_custody_ref(
    ref: ParentRef,
    *,
    allowed_root: str | Path,
    slot_names: Collection[str] | None,
) -> ResolvedCheckpointTransaction:
    (
        root_path,
        manifest_path,
        manifest_sha256,
        loaded_manifest,
        slots_by_name,
    ) = _load_trusted_checkpoint_custody_ref(ref, allowed_root=allowed_root)
    manifest = loaded_manifest.document
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
            value = _deserialize_checkpoint_slot(
                slot, transaction_dir, manifest, blob_path=blob_path,
                loaded_model=loaded_slots.get("model"),
            )
        except (CheckpointIntegrityError, CheckpointCompatibilityError, OSError) as exc:
            raise CheckpointReferenceResolutionError(str(exc)) from exc
        try:
            _validate_decoded_slot_integrity(slot, value)
        except CheckpointReferenceResolutionError:
            raise
        except Exception as exc:
            raise CheckpointReferenceResolutionError(str(exc)) from exc
        loaded_slots[name] = value

    try:
        provenance_notices, _ = _validate_manifest_structural_abi(
            manifest, loaded_slots, layout="authenticated_source",
        )
    except CheckpointIntegrityError as exc:
        raise CheckpointReferenceResolutionError(str(exc)) from exc
    _validate_decoded_semantic_state_roots(manifest, loaded_slots)
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


def _load_trusted_checkpoint_custody_ref(
    ref: ParentRef,
    *,
    allowed_root: str | Path,
) -> tuple[
    Path,
    Path,
    str,
    CheckpointDocumentLoadResult[CheckpointTransactionManifest],
    dict[str, CheckpointSlotBlobRef],
]:
    """Load one exact reference from a caller-designated trusted custody root."""
    _validate_checkpoint_custody_parent_ref(ref)
    manifest_sha256 = ref.metadata["manifest_sha256"]

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
    return root_path, manifest_path, manifest_sha256, loaded_manifest, slots_by_name


def _authenticate_checkpoint_custody_bytes(
    ref: ParentRef,
    *,
    raw_manifest: bytes,
    manifest_parts: tuple[str, ...],
    root_descriptor: int,
    scope: CheckpointReadScope,
) -> AuthenticatedCheckpointTransaction:
    """Authenticate exact manifest bytes and all declared blobs under one root fd."""
    _validate_checkpoint_custody_parent_ref(ref)
    manifest_sha256 = ref.metadata["manifest_sha256"]
    if sha256_bytes(raw_manifest) != manifest_sha256:
        raise CheckpointReferenceResolutionError(
            "checkpoint custody ParentRef manifest_sha256 does not match raw manifest bytes"
        )
    try:
        loaded_manifest = load_checkpoint_transaction_manifest_json(
            raw_manifest,
            path="/".join(manifest_parts),
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

    _validate_checkpoint_transaction_integrity_records(manifest)
    selected = _checkpoint_scope_slots(manifest, scope)
    transaction_parts = manifest_parts[:-1]
    for slot in selected:
        if slot.media_type not in {
            MODEL_TREE_MEDIA_TYPE, STANDARD_TRAIN_STATE_MEDIA_TYPE,
        }:
            raise CheckpointReferenceResolutionError(
                f"checkpoint slot {slot.slot!r} has unsupported media_type {slot.media_type!r}"
            )
        slot_parts = _checkpoint_custody_relative_parts(
            slot.relative_path,
            context=f"checkpoint slot {slot.slot!r} relative_path",
        )
        blob_bytes = _read_checkpoint_custody_member(
            root_descriptor,
            (*transaction_parts, *slot_parts),
            context=f"checkpoint slot {slot.slot!r} blob",
        )
        if len(blob_bytes) != slot.size_bytes:
            raise CheckpointReferenceResolutionError(f"checkpoint slot {slot.slot!r} size mismatch")
        if sha256_bytes(blob_bytes) != slot.sha256:
            raise CheckpointReferenceResolutionError(f"checkpoint slot {slot.slot!r} hash mismatch")
        if slot.media_type == MODEL_TREE_MEDIA_TYPE:
            for relative, member in semantic_model_member_files(slot, blob_bytes).items():
                raw = _read_checkpoint_custody_member(
                    root_descriptor, (*transaction_parts, *relative.parts),
                    context=f"semantic model member {relative}",
                )
                if len(raw) != member.size_bytes or sha256_bytes(raw) != member.sha256:
                    raise CheckpointReferenceResolutionError(f"semantic model member {relative} changed")

    return AuthenticatedCheckpointTransaction(
        parent_ref=_immutable_model_snapshot(ref),
        manifest_sha256=manifest_sha256,
        manifest=_immutable_model_snapshot(manifest),
        slot_names=tuple(slot.slot for slot in selected),
        migration_records=tuple(
            _immutable_model_snapshot(record) for record in loaded_manifest.migration_records
        ),
    )


def _validate_checkpoint_custody_parent_ref(ref: ParentRef) -> None:
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
    if not _is_sha256(ref.metadata.get("manifest_sha256")):
        raise CheckpointReferenceResolutionError(
            "checkpoint custody ParentRef metadata.manifest_sha256 must be "
            "64 lowercase hexadecimal characters"
        )


def _checkpoint_custody_relative_parts(
    value: str | None,
    *,
    context: str,
) -> tuple[str, ...]:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise CheckpointReferenceResolutionError(f"{context} is not a safe POSIX relative path")
    path = PurePosixPath(value)
    parts = tuple(value.split("/"))
    if path.is_absolute() or any(part in {"", ".", ".."} for part in parts):
        raise CheckpointReferenceResolutionError(f"{context} is not a canonical relative path")
    return parts


def _open_checkpoint_custody_root(root: str | Path) -> int:
    """Pin one real custody root directory without following symlinks."""
    try:
        records = open_directory_chain(
            Path(root),
            create=False,
            error_factory=CheckpointReferenceResolutionError,
            context="checkpoint custody root",
        )
        return take_directory_chain_leaf(records)
    except OSError as exc:
        raise CheckpointReferenceResolutionError(
            "checkpoint custody root is not an available real directory"
        ) from exc


def _read_checkpoint_custody_member(
    root_descriptor: int,
    parts: tuple[str, ...],
    *,
    context: str,
) -> bytes:
    """Read one regular file via no-follow traversal from a pinned root fd."""
    directory_descriptor = os.dup(root_descriptor)
    try:
        for part in parts[:-1]:
            try:
                next_descriptor = open_existing_directory(
                    part,
                    error_factory=CheckpointReferenceResolutionError,
                    context=f"{context} directory",
                    dir_fd=directory_descriptor,
                )
            except OSError as exc:
                raise CheckpointReferenceResolutionError(
                    f"{context} directory traversal is unsafe or unavailable"
                ) from exc
            try:
                os.close(directory_descriptor)
                directory_descriptor = next_descriptor
                next_descriptor = -1
            finally:
                if next_descriptor >= 0:
                    os.close(next_descriptor)

        name = parts[-1]
        try:
            descriptor, opened = open_existing_file(
                name,
                rigor=SecurePathRigor.REGULAR_FILE_IDENTITY,
                error_factory=CheckpointReferenceResolutionError,
                context=context,
                dir_fd=directory_descriptor,
            )
        except CheckpointReferenceResolutionError:
            raise
        except OSError as exc:
            raise CheckpointReferenceResolutionError(f"{context} is unsafe or unavailable") from exc
        try:
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            completed = os.fstat(descriptor)
            after_read = os.stat(
                name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            if (
                _identity(completed) != _identity(opened)
                or _identity(after_read) != _identity(opened)
                or completed.st_size != opened.st_size
                or completed.st_mtime_ns != opened.st_mtime_ns
            ):
                raise CheckpointReferenceResolutionError(
                    f"{context} identity changed while reading"
                )
            return b"".join(chunks)
        finally:
            os.close(descriptor)
    finally:
        os.close(directory_descriptor)


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


def _checkpoint_scope_slots(
    manifest: CheckpointTransactionManifest, scope: CheckpointReadScope,
) -> tuple[CheckpointSlotBlobRef, ...]:
    if scope == "full":
        return tuple(manifest.slots)
    if isinstance(scope, frozenset):
        selected = tuple(slot for slot in manifest.slots if slot.slot in scope)
        if {slot.slot for slot in selected} != scope:
            raise CheckpointReferenceResolutionError("The bound checkpoint lacks projected slots")
        return selected
    if scope != "model_only":
        raise CheckpointReferenceResolutionError(f"Unknown checkpoint read scope {scope!r}")
    selected = tuple(slot for slot in manifest.slots if slot.role in INITIALIZE_FROM_SOURCE_ROLES)
    if not selected:
        raise CheckpointReferenceResolutionError("The bound checkpoint has no model slot")
    return selected


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
    if integrity.schema_id != CHECKPOINT_CONTENT_INTEGRITY_SCHEMA_ID:
        raise CheckpointReferenceResolutionError(
            f"checkpoint content integrity schema_id is unsupported: {integrity.schema_id!r}"
        )
    expected_integrity_version = (
        CHECKPOINT_CONTENT_INTEGRITY_SCHEMA_VERSION
        if manifest.semantic_state_roots_status == "available"
        else CHECKPOINT_CONTENT_INTEGRITY_SCHEMA_VERSION_V1
    )
    if integrity.schema_version != expected_integrity_version:
        raise CheckpointReferenceResolutionError(
            "checkpoint content integrity schema_version is unsupported: "
            f"{integrity.schema_version!r}"
        )
    if isinstance(integrity, ContentIntegrityDigest) and (
        integrity.semantic_state_root_algorithm_version != SEMANTIC_STATE_ROOT_ALGORITHM_VERSION
    ):
        raise CheckpointReferenceResolutionError(
            "checkpoint semantic state root algorithm is unsupported: "
            f"{integrity.semantic_state_root_algorithm_version!r}"
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
        if slot.semantic_abi is None:
            raise CheckpointReferenceResolutionError(
                f"checkpoint slot {name!r} semantic ABI ownership is missing"
            )
        try:
            validate_checkpoint_semantic_abi(slot.semantic_abi)
        except ValueError as exc:
            raise CheckpointReferenceResolutionError(
                f"checkpoint slot {name!r} semantic ABI fingerprint is stale"
            ) from exc
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
        expected_fingerprint = structural_abi_content_sha256(fingerprint)
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
    if slot.media_type not in {
        MODEL_TREE_MEDIA_TYPE, STANDARD_TRAIN_STATE_MEDIA_TYPE,
    }:
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


def _validate_decoded_semantic_state_roots(
    manifest: CheckpointTransactionManifest,
    loaded_slots: Mapping[str, Any],
) -> None:
    for slot in manifest.slots:
        if slot.slot not in loaded_slots or not isinstance(slot.content_digest, SlotContentDigest):
            continue
        value = loaded_slots[slot.slot]
        actual_semantic_root = semantic_state_root_sha256(
            slot=slot.slot,
            role=slot.role,
            value=value,
            structural_abi_fingerprint=_source_layout_abi(slot, structural_abi_fingerprint(value)),
            materialized_axes=slot.materialized_axes,
            population=slot.population,
        )
        if actual_semantic_root != slot.content_digest.semantic_state_root_sha256:
            raise CheckpointReferenceResolutionError(
                f"checkpoint slot {slot.slot!r} semantic state root mismatch"
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

        def snapshot_deepcopy(
            self: BaseModel,
            memo: dict[int, Any] | None = None,
        ) -> BaseModel:
            memo = {} if memo is None else memo
            existing = memo.get(id(self))
            if existing is not None:
                return existing
            payload = copy.deepcopy(
                self.model_dump(mode="python", round_trip=True),
                memo,
            )
            copied = model_type.model_validate(payload)
            memo[id(self)] = copied
            return copied

        read_only_type = type(
            f"_ReadOnly{model_type.__name__}",
            (model_type,),
            {
                "model_config": config,
                "__module__": __name__,
                "__eq__": snapshot_eq,
                "__deepcopy__": snapshot_deepcopy,
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
        items = tuple(_immutable_snapshot_value(item) for item in value)
        if hasattr(value, "_fields"):
            return type(value)(*items)
        return items
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
    model: type[CheckpointLatestPointer] | type[CheckpointTransactionManifest],
    path: str,
    document_name: str,
) -> CheckpointDocumentLoadResult[Any]:
    """Read the current custody document and reject ambiguous JSON."""
    try:
        document = strict_model_validate_json(model, data, ref=path)
    except DuplicateJsonKeyError as exc:
        raise CheckpointIntegrityError(f"{document_name} states a member twice: {exc}") from exc
    except (ValueError, TypeError) as exc:
        raise CheckpointIntegrityError(f"{document_name} is invalid: {exc}") from exc
    return CheckpointDocumentLoadResult(
        document=document,
        schema_id=document.schema_id,
        source_version=document.schema_version,
        target_version=document.schema_version,
        migration_records=(),
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
            raise CheckpointConsistencyError(f"{path} mapped authorities diverged: {values!r}")
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
            (slot, checkpoint_leaf_path(path), leaf)
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
                granularities[f"{slot}{checkpoint_leaf_path(path)}"] = leaf.granularity.interval
    return granularities


def _allocate_segment_histories(
    source_slots: Mapping[str, Any],
    templates: Mapping[str, Any],
    *,
    preserve_history_abi: bool = False,
) -> tuple[dict[str, Any], set[str]]:
    """Preserve dynamical leaves while replacing histories with segment templates."""
    result = dict(source_slots)
    changed: set[str] = set()
    for slot, source in source_slots.items():
        if slot not in templates:
            continue
        source_leaves, source_def = jt.flatten(source, is_leaf=is_type(BatchHistory))
        target_leaves, target_def = jt.flatten(templates[slot], is_leaf=is_type(BatchHistory))
        if preserve_history_abi and not any(
            isinstance(leaf, BatchHistory) for leaf in (*source_leaves, *target_leaves)
        ):
            continue
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
                if preserve_history_abi:
                    _validate_batch_history_template(source_leaf, target_leaf, slot=slot)
                output.append(target_leaf)
                slot_changed = True
            else:
                output.append(source_leaf)
        if slot_changed:
            result[slot] = jt.unflatten(source_def, output)
            changed.add(slot)
    return result, changed


def _validate_batch_history_template(
    template: BatchHistory[Any],
    restored: BatchHistory[Any],
    *,
    slot: str,
) -> None:
    template_value, restored_value = template.value, restored.value
    if eqx.is_array(template_value) and eqx.is_array(restored_value):
        rank = template_value.ndim
        axis = template.batch_axis % rank if rank else 0
        if (
            -rank <= template.batch_axis < rank == restored_value.ndim
            and template_value.shape[:axis] == restored_value.shape[:axis]
            and template_value.shape[axis + 1:] == restored_value.shape[axis + 1:]
            and template_value.dtype == restored_value.dtype
            and bool(getattr(template_value, "weak_type", False))
            == bool(getattr(restored_value, "weak_type", False))
        ):
            return
    raise CheckpointCompatibilityError(
        f"continuation BatchHistory array ABI differs outside the batch extent; slot={slot!r}"
    )


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
    segment_parent_checkpoint_set: CheckpointSet | None = None,
    segment_parent_ref: ExactRef | None = None,
    metadata: Mapping[str, Any] | None = None,
    publish_latest: bool = True,
    content_provider: ArtifactBlobProvider | None = None,
    semantic_model: SemanticModelWriter | None = None,
) -> CheckpointWriteResult:
    """Write one atomic multi-slot checkpoint transaction and publish latest.

    Slot names are derived from ``phase_program`` and ``barrier_name``. The
    caller supplies the engine-state mapping, but not filenames.
    """
    if status not in {"partial", "final"}:
        raise CheckpointCustodyError("checkpoint status must be 'partial' or 'final'")
    if segment_parent_checkpoint_set is not None:
        if (
            segment_parent_ref is not None
            or segment_parent_checkpoint_set.transaction.identity != segment_parent_transaction_id
        ):
            raise CheckpointCustodyError(
                "checkpoint parent publication does not match transaction lineage"
            )
        segment_parent_ref = segment_parent_checkpoint_set.exact_ref
    barrier = checkpoint_barrier(phase_program, barrier_name)
    resolved_axes = _resolved_slot_axes(run_spec)
    coordinate = _checkpoint_coordinate(coordinate, resolved_axes)
    slot_specs = tuple(barrier.slots)
    _validate_required_slots(slot_specs, slots)
    slot_roles = _slot_roles(run_spec)
    _validate_standard_train_writer(run_spec, barrier, slot_roles, semantic_model)
    manifest_metadata = {"phase": barrier.phase, **dict(metadata or {})}
    if segment_parent_ref is not None:
        manifest_metadata["checkpoint_segment_parent_ref"] = segment_parent_ref.model_dump(
            mode="json"
        )
    else:
        manifest_metadata.pop("checkpoint_segment_parent_ref", None)
    _validate_program_step_units(coordinate, manifest_metadata, context="checkpoint write")
    completed_batches = _resolve_completed_training_batches(
        phase_program=phase_program,
        slots=slots,
        explicit=completed_training_batches,
        metadata=manifest_metadata,
        slot_axis_bindings=resolved_axes,
    )
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
        semantic_abis = derive_checkpoint_semantic_abis(
            {
                spec.slot: structural_abi_fingerprint(slots[spec.slot])
                for spec in slot_specs
                if spec.slot in slots
            },
            slot_roles=slot_roles,
            optimizer_bindings=phase_program.optimizer_bindings,
        )
        population_slots = _population_slot_names(run_spec)
        slot_records: list[CheckpointSlotBlobRef] = []
        slot_digests: list[SlotContentDigest] = []
        for spec in slot_specs:
            if spec.slot not in slots and not spec.required:
                continue
            slot_record, content_digest = _write_fresh_slot_blob(
                spec, slots[spec.slot],
                transaction_dir=tmp_dir,
                coordinate=_checkpoint_coordinate(
                    (slot_coordinates or {}).get(spec.slot, coordinate), resolved_axes,
                ),
                slot_roles=slot_roles,
                population_slots=population_slots,
                population_member_ids=population_member_ids or {},
                semantic_abi=semantic_abis[spec.slot],
                materialized_axes=resolved_axes.get(spec.slot),
                semantic_model=semantic_model,
                completed_batches=completed_batches,
            )
            slot_records.append(slot_record)
            slot_digests.append(content_digest)

        _validate_slot_coordinate_consistency(
            barrier=barrier,
            completed_coordinate=coordinate,
            slot_records=slot_records,
        )
        transaction_root = _transaction_root_sha256(slot_digests)
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
                semantic_state_root_algorithm_version=SEMANTIC_STATE_ROOT_ALGORITHM_VERSION,
                slots=slot_digests,
                transaction_root_sha256=transaction_root,
            ),
            history_availability=dict(history_availability or {}),
            parent_lineage=list(parent_lineage or ()),
            metadata=manifest_metadata,
        )
        manifest_path = tmp_dir / MANIFEST_NAME
        _write_json_atomic(manifest_path, manifest.model_dump(mode="json", exclude_none=True))
        manifest_sha256 = sha256_file(manifest_path)
        checkpoint_set = _published_checkpoint_set(
            run_spec=run_spec,
            manifest=manifest,
            parent=segment_parent_ref,
        )
        checkpoint_set_path = tmp_dir / CHECKPOINT_SET_NAME
        _write_json_atomic(
            checkpoint_set_path,
            checkpoint_set.model_dump(mode="json", exclude_none=True),
        )

        os.replace(tmp_dir, final_dir)
        manifest_path = final_dir / MANIFEST_NAME
        checkpoint_set_path = final_dir / CHECKPOINT_SET_NAME
        sync_parent_directories(manifest_path)
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
        elif content_provider is not None:
            raise CheckpointCustodyError(
                "canonical checkpoint publication requires its immutable latest.json snapshot"
            )
        content_binding: ContentBinding | None = None
        content_root: ContentRootRecord | None = None
        if content_provider is not None:
            content_binding, content_root = _publish_checkpoint_content(
                content_provider,
                latest_pointer_path=latest_path,
                manifest_path=manifest_path,
                checkpoint_set_path=checkpoint_set_path,
                manifest=manifest,
                checkpoint_set=checkpoint_set,
            )
        return CheckpointWriteResult(
            root=root_path,
            transaction_dir=final_dir,
            manifest_path=manifest_path,
            latest_pointer_path=latest_path,
            manifest=manifest,
            latest_pointer=latest_pointer,
            checkpoint_set_path=checkpoint_set_path,
            checkpoint_set=checkpoint_set,
            content_binding=content_binding,
            content_root=content_root,
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
    published_pointer: CheckpointLatestPointer | None = None,
) -> CheckpointResumeResult:
    """Resume from a bound publication or resolve the current latest pointer."""
    root_path = Path(root)
    latest = published_pointer if published_pointer is not None else _load_latest_pointer(root_path)
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


def bind_published_checkpoint_content(
    root: Path,
    pointer: CheckpointLatestPointer,
    provider: ArtifactBlobProvider,
    *,
    scope: CheckpointReadScope = "full",
) -> tuple[ParentRef, ContentBinding]:
    """Retain the exact bound publication for native checkpoint initialization."""
    authenticated = authenticate_checkpoint_custody_ref(
        _checkpoint_latest_parent_ref(pointer), allowed_root=root, scope=scope,
    )
    manifest_path, manifest = _bound_checkpoint_manifest(root, pointer)
    checkpoint_set = load_checkpoint_set(root, pointer.transaction_id)
    latest = provider.store_object(_durable_json_bytes(
        pointer.model_dump(mode="json", exclude_none=True)
    ))
    binding, _ = _publish_checkpoint_content(
        provider,
        latest_pointer_path=latest,
        manifest_path=manifest_path,
        checkpoint_set_path=manifest_path.parent / CHECKPOINT_SET_NAME,
        manifest=manifest,
        checkpoint_set=checkpoint_set,
        scope=scope,
    )
    source = authenticated.parent_ref.model_copy(update={
        "uri": None,
        "metadata": authenticated_manifest_ref_metadata(
            pointer.manifest_sha256, checkpoint_set.transaction.bytes.size_bytes,
        ),
    })
    return source, binding


def open_published_model_slot(
    root: Path, pointer: CheckpointLatestPointer,
) -> SemanticModelSlot | None:
    """Select a model from one exact publication without reading sibling slots."""
    transaction, slot = _published_model_slot(root, pointer)
    if slot is None:
        return None
    return SemanticModelSlot.open(transaction, slot)


def published_model_tree_sha256(root: Path, pointer: CheckpointLatestPointer) -> str | None:
    """Read the semantic model tree identity from the authenticated transaction."""
    _, slot = _published_model_slot(root, pointer)
    return slot.sha256 if slot is not None else None


def _published_model_slot(
    root: Path, pointer: CheckpointLatestPointer,
) -> tuple[Path, CheckpointSlotBlobRef | None]:
    manifest_path, manifest = _bound_checkpoint_manifest(root, pointer)
    slot = _validate_checkpoint_transaction_integrity_records(manifest).get("model")
    if slot is None or slot.media_type != MODEL_TREE_MEDIA_TYPE:
        return manifest_path.parent, None
    _supported_slot_serializer(slot)
    return manifest_path.parent, slot


def _bound_checkpoint_manifest(
    root: Path, pointer: CheckpointLatestPointer,
    *, read_bytes: Callable[[Path], bytes] = Path.read_bytes,
) -> tuple[Path, CheckpointTransactionManifest]:
    manifest_path, manifest = _manifest_from_latest_pointer(root, pointer, read_bytes=read_bytes)
    if (
        manifest.run_id != pointer.run_id
        or manifest.completed_coordinate != pointer.completed_coordinate
        or manifest.completed_training_batches != pointer.completed_training_batches
    ):
        raise CheckpointIntegrityError("Bound checkpoint disagrees with the published coordinate.")
    return manifest_path, manifest


def checkpoint_slot_files(
    transaction_dir: Path, slot: CheckpointSlotBlobRef,
    *, read_bytes: Callable[[Path], bytes] = Path.read_bytes,
) -> dict[Path, tuple[str, int]]:
    """Return each file in a slot's exact byte closure."""
    path = transaction_dir / slot.relative_path
    files = {path: (slot.sha256, slot.size_bytes)}
    if slot.media_type == MODEL_TREE_MEDIA_TYPE:
        files.update({
            transaction_dir / relative: (ref.sha256, ref.size_bytes)
            for relative, ref in semantic_model_member_files(
                slot, _read_blob(slot, path, read_bytes=read_bytes),
            ).items()
        })
    return files


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

    request = _coerce_continuation_request(continuation_request)
    continuation_applied = request is not None and _continuation_was_applied(manifest, request)
    validate_checkpoint_run_contract(
        manifest,
        expected_run_spec,
        expected_phase_program,
        allow_new_lineage_override=allow_new_lineage_override and not continuation_applied,
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
        loaded_slots[slot.slot] = _deserialize_checkpoint_slot(
            slot, transaction_dir, manifest, template=expected_slots.get(slot.slot),
        )
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
    if request is not None:
        if continuation_applied:
            _validate_batch_histories(
                loaded_slots, segment_batch_count=manifest.segment_lineage.segment_batch_count
            )
            expected_slots, _ = _allocate_segment_histories(
                expected_slots, loaded_slots, preserve_history_abi=True
            )
        else:
            validate_checkpoint_continuation_source_count(manifest, request)
            loaded_slots, _ = _allocate_segment_histories(loaded_slots, expected_slots)
            loaded_fingerprints = {}
    _validate_structural_abi(
        manifest,
        expected_slots,
        loaded_slots,
        expected_phase_program=expected_phase_program,
        loaded_fingerprints=loaded_fingerprints,
    )
    _validate_recorded_slot_axes(manifest, expected_axes, loaded_slots)

    return CheckpointResumeResult(
        manifest=manifest,
        parent_ref=_checkpoint_latest_parent_ref(latest),
        slots=loaded_slots,
        provenance_notices=provenance_notices,
        new_lineage_required=allow_new_lineage_override
        and not _contract_binding_matches(
            manifest.run_contract_binding,
            run_contract_binding(
                expected_run_spec,
                expected_phase_program,
            ),
        ),
        previous_transaction_id=(manifest.transaction_id if allow_new_lineage_override else None),
    )


def validate_checkpoint_continuation_source_count(
    manifest: CheckpointTransactionManifest,
    request: CheckpointContinuationRequest,
) -> None:
    """Validate the continuation boundary against authenticated segment lineage."""
    lineage_total = (
        manifest.segment_lineage.start_batch + manifest.segment_lineage.segment_batch_count
    )
    if request.source_completed_batches != lineage_total:
        raise CheckpointCompatibilityError(
            "checkpoint continuation source offset mismatch; "
            f"lineage_total={lineage_total} requested={request.source_completed_batches}"
        )


def _manifest_from_latest_pointer(
    root_path: Path,
    latest: CheckpointLatestPointer,
    *,
    read_bytes: Callable[[Path], bytes] = Path.read_bytes,
) -> tuple[Path, CheckpointTransactionManifest]:
    manifest_path = _resolve_latest_manifest_path(root_path.resolve(), latest)
    if not manifest_path.is_file():
        raise CheckpointIntegrityError(
            f"latest pointer references missing manifest: {latest.manifest_relative_path}"
        )
    try:
        raw_manifest = read_bytes(manifest_path)
    except OSError as exc:
        raise CheckpointIntegrityError(
            f"latest pointer references unreadable manifest: {latest.manifest_relative_path}"
        ) from exc
    if sha256_bytes(raw_manifest) != latest.manifest_sha256:
        raise CheckpointIntegrityError("latest pointer manifest hash does not match bytes")
    manifest = load_checkpoint_transaction_manifest_json(raw_manifest, path=str(manifest_path)).document
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


def validate_checkpoint_fork_execution_dependencies(
    plan: CheckpointForkPlan,
    dependencies: Sequence[ExecutionDependency | ExecutionDependencyV6],
    *,
    allow_task_identity: bool = False,
) -> None:
    unsupported = [
        item.kind
        for item in dependencies
        if isinstance(item, (LineageGraftDependency, StoppedRowStatus))
        or isinstance(item, TaskIdentityGate)
        and not allow_task_identity
    ]
    if unsupported:
        raise CheckpointCompatibilityError(
            f"unsupported checkpoint fork dependencies {unsupported!r}"
        )
    forks = [
        item
        for item in dependencies
        if isinstance(item, (ForkFromSelectedCheckpoint, ForkFromSelectedCheckpointV6))
    ]
    if len(forks) != 1:
        raise CheckpointCompatibilityError("checkpoint fork requires one typed fork dependency")
    fork = forks[0]
    source = plan.source
    if isinstance(fork, ForkFromSelectedCheckpointV6):
        if not isinstance(fork.source_authority, ExecutionHashForkSourceAuthority):
            raise CheckpointCompatibilityError(
                "resolved-output-root fork authority uses direct checkpoint custody; "
                "CheckpointForkPlan v1 cannot relabel it as an execution hash"
            )
        source_execution_hash = fork.source_authority.execution_hash
    else:
        source_execution_hash = fork.source_execution_hash
    if (
        source.expected_transaction_id != fork.checkpoint_transaction_id
        or source.expected_transaction_root_sha256 != fork.checkpoint_root_hash
        or source.source_execution_hash != source_execution_hash
        or source.source_row_id != fork.source_row_id
    ):
        raise CheckpointCompatibilityError("checkpoint fork source authority drifts")

    def placed(target: str | None, step: CheckpointForkTransformStep) -> list[dict[str, Any]]:
        return [
            {
                "transform_id": record.identity,
                "version": record.version,
                "implementation_sha256": record.implementation_sha256,
                "stage": step.stage,
                "target_row_id": target,
                "slot": record.slot,
                "parameters": record.parameters,
                **(
                    {"target_only": step.target_only_slots[record.slot]}
                    if isinstance(fork, ForkFromSelectedCheckpointV6)
                    and record.slot in step.target_only_slots
                    else {}
                ),
            }
            for record in step.records
        ]

    projected = sum((placed(None, step) for step in source.transforms), [])
    projected += sum(
        (
            placed(target.row_id or target.target_id, step)
            for target in plan.targets
            for step in target.transforms
        ),
        [],
    )
    declared = [item.model_dump(mode="json", exclude={"kind"}) for item in fork.slot_transforms]
    if declared != projected:
        raise CheckpointCompatibilityError("checkpoint fork transform placement/order drifts")
    declared_continuations = [
        (item.source_completed_batches, item.additional_batches, item.expected_target_total)
        for item in dependencies
        if isinstance(item, ContinuationReconciliation)
    ]
    planned_continuations = [
        (request.source_completed_batches, request.additional_batches, request.target_total)
        for target in plan.targets
        if (request := target.history_policy.continuation_request) is not None
    ]
    if declared_continuations != planned_continuations:
        raise CheckpointCompatibilityError("checkpoint fork continuation authority drifts")


def checkpoint_fork_binding_content_sha256(values: Mapping[str, Any]) -> str:
    return _canonical_hash(
        {
            slot: [item.model_dump(mode="json") for item in _leaf_content_digests(value)]
            for slot, value in sorted(values.items())
        }
    )


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


def _checkpoint_fork_compatibility_differences(
    target_id: str,
    stored: CheckpointForkCompatibilityProjection,
    candidate: CheckpointForkCompatibilityProjection,
) -> tuple[CheckpointForkDerivedDigestDifference, ...]:
    differences: list[CheckpointForkDerivedDigestDifference] = []
    for path, stored_value, candidate_value in (
        (
            "/compatibility/run_contract_algorithm_version",
            stored.run_contract_algorithm_version,
            candidate.run_contract_algorithm_version,
        ),
        (
            "/compatibility/run_contract_hash_domain",
            stored.run_contract_hash_domain,
            candidate.run_contract_hash_domain,
        ),
        (
            "/compatibility/run_contract_projection_sha256",
            stored.run_contract_projection_sha256,
            candidate.run_contract_projection_sha256,
        ),
        (
            "/compatibility/population_member_ids_sha256",
            stored.population_member_ids_sha256,
            candidate.population_member_ids_sha256,
        ),
    ):
        if stored_value != candidate_value:
            differences.append(
                CheckpointForkDerivedDigestDifference(
                    target_id=target_id,
                    path=path,
                    stored=stored_value,
                    candidate=candidate_value,
                )
            )
    stored_slots = stored.slot_structural_abi_sha256
    candidate_slots = candidate.slot_structural_abi_sha256
    for slot in sorted(set(stored_slots) | set(candidate_slots)):
        stored_value = stored_slots.get(slot)
        candidate_value = candidate_slots.get(slot)
        if stored_value != candidate_value:
            differences.append(
                CheckpointForkDerivedDigestDifference(
                    target_id=target_id,
                    path=f"/compatibility/slot_structural_abi_sha256/{slot}",
                    stored=stored_value,
                    candidate=candidate_value,
                )
            )
    return tuple(differences)


def rebuild_checkpoint_fork_derived_digests(
    plan: CheckpointForkPlan,
    bindings: CheckpointForkPlanBindings,
) -> CheckpointForkDerivedDigestComparison:
    """Rebuild all target compatibility candidates without modifying ``plan``.

    Candidate values come only from each target's existing run-contract and
    structural-ABI authorities. Missing bindings are errors; this routine never
    substitutes stored values for absent runtime inputs.
    """
    candidates: dict[str, CheckpointForkCompatibilityProjection] = {}
    differences: list[CheckpointForkDerivedDigestDifference] = []
    for target in plan.targets:
        run_spec = _require_plan_binding(bindings.run_specs, target.run_spec_ref, kind="run spec")
        slot_templates = _require_plan_binding(
            bindings.slot_templates,
            target.slot_template_ref,
            kind="slot template",
        )
        population_member_ids = (
            _require_plan_binding(
                bindings.population_member_ids,
                target.population_member_ids_ref,
                kind="population member ids",
            )
            if target.population_member_ids_ref is not None
            else None
        )
        candidate = derive_checkpoint_fork_compatibility_projection(
            run_spec,
            run_spec.worker_execution.method_contract.phase_program,
            slot_templates,
            population_member_ids=population_member_ids,
        )
        candidates[target.target_id] = candidate
        differences.extend(
            _checkpoint_fork_compatibility_differences(
                target.target_id, target.compatibility, candidate
            )
        )
    return CheckpointForkDerivedDigestComparison(
        candidates=MappingProxyType(candidates),
        differences=tuple(differences),
    )


def _checkpoint_fork_authored_projection(plan: CheckpointForkPlan) -> dict[str, Any]:
    """Return every plan field except the per-target derived compatibility lock."""
    projection = plan.model_dump(mode="json", exclude_none=True)
    for target in projection["targets"]:
        del target["compatibility"]
    return projection


def _replace_checkpoint_fork_compatibility_locks(
    plan: CheckpointForkPlan,
    targets: Sequence[CheckpointForkTarget],
) -> CheckpointForkPlan:
    """Accept replacement target rows only when authored plan fields are identical."""
    relocked = plan.model_copy(update={"targets": list(targets)})
    if _checkpoint_fork_authored_projection(relocked) != _checkpoint_fork_authored_projection(plan):
        raise CheckpointCompatibilityError(
            "checkpoint fork re-lock attempted an authored-field change"
        )
    return relocked


def relock_checkpoint_fork_derived_digests(
    plan: CheckpointForkPlan,
    bindings: CheckpointForkPlanBindings,
    *,
    requalification_requirements: Sequence[str],
) -> CheckpointForkPlanRelockResult:
    """Explicitly replace stale derived locks and return re-qualification duties.

    This is the sole lock-mutating API. It refuses a no-op refresh and preserves
    all authored plan fields, including source, target routing, transforms,
    policies, and metadata.
    """
    requirements = tuple(requirement.strip() for requirement in requalification_requirements)
    if not requirements or any(not requirement for requirement in requirements):
        raise CheckpointCompatibilityError(
            "checkpoint fork re-lock requires non-empty re-qualification requirements"
        )
    comparison = rebuild_checkpoint_fork_derived_digests(plan, bindings)
    if comparison.matches_stored_lock:
        raise CheckpointCompatibilityError(
            "checkpoint fork re-lock requires a derived-digest difference"
        )
    relocked = _replace_checkpoint_fork_compatibility_locks(
        plan,
        [
            target.model_copy(update={"compatibility": comparison.candidates[target.target_id]})
            for target in plan.targets
        ],
    )
    return CheckpointForkPlanRelockResult(
        plan=relocked,
        comparison=comparison,
        requalification_requirements=requirements,
    )


def _coerce_checkpoint_fork_plan(
    value: CheckpointForkPlan | Mapping[str, Any],
) -> CheckpointForkPlan:
    if isinstance(value, CheckpointForkPlan):
        return value
    try:
        return CheckpointForkPlan.model_validate(value)
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
    assert record.version is not None
    assert record.implementation_sha256 is not None
    registration = registry.resolve(
        record.identity,
        record.version,
        record.implementation_sha256,
    )
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
        if _semantic_structural_abi_sha256(
            actual,
            value=prepared_slots[spec.slot],
        ) != _semantic_structural_abi_sha256(
            expected,
            value=expected_slots[spec.slot],
        ):
            diff_suffix = _format_structural_abi_diff(
                expected,
                actual,
                recorded_value=expected_slots[spec.slot],
                actual_value=prepared_slots[spec.slot],
            )
            raise CheckpointCompatibilityError(
                f"checkpoint fork target {target_id!r} slot {spec.slot!r} structural ABI "
                f"mismatch{diff_suffix}"
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
        record = step.records[0]
        assert record.version is not None
        assert record.implementation_sha256 is not None
        registry.resolve(record.identity, record.version, record.implementation_sha256)
    for target in plan.targets:
        policy = target.history_policy
        if policy.segment_history_template_ref is not None:
            _require_plan_binding(
                bindings.segment_history_templates,
                policy.segment_history_template_ref,
                kind="segment history template",
            )

    comparison = rebuild_checkpoint_fork_derived_digests(plan, bindings)
    if comparison.differences:
        changes = ", ".join(
            f"{difference.target_id}{difference.path}" for difference in comparison.differences
        )
        raise CheckpointCompatibilityError(f"checkpoint fork compatibility lock differs: {changes}")

    missing_writers = {
        target.target_id for target in plan.targets
    } - bindings.semantic_model_writers.keys()
    if missing_writers:
        raise CheckpointCompatibilityError(f"checkpoint fork requires typed writers for {sorted(missing_writers)!r}")
    source = _load_latest_checkpoint_transaction(source_root, scope=bindings.source_scope)
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
        population_ids = (
            {}
            if target.population_member_ids_ref is None
            else _require_plan_binding(
                bindings.population_member_ids,
                target.population_member_ids_ref,
                kind="population member ids",
            )
        )
        phase_program = run_spec.worker_execution.method_contract.phase_program
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
            if history_templates is not None:
                actual = checkpoint_fork_binding_content_sha256(history_templates)
                if actual != policy.segment_history_template_sha256:
                    raise CheckpointCompatibilityError(
                        "checkpoint continuation template content drift; "
                        f"declared={policy.segment_history_template_sha256!r} "
                        f"actual={actual!r}"
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
    content_provider: ArtifactBlobProvider,
    source_content_binding: ContentBinding,
) -> dict[str, CheckpointForkResult]:
    """Preflight and write typed fork targets from an already leased source projection."""
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
        transformed_source_slots = set(prepared.transformed_slots) - set(prepared.target_only_slots)
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
        result = _fork_checkpoint_transaction_from_path(
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
            content_provider=content_provider,
            source_content_binding=source_content_binding,
            semantic_model=bindings.semantic_model_writers[prepared.target_id],
            source_scope=bindings.source_scope,
        )
        results[prepared.target_id] = result
    return results


def _fork_checkpoint_transaction_from_path(
    source_root: str | Path,
    target_root: str | Path,
    *,
    target_run_spec: TrainingRunSpec,
    target_phase_program: PhaseProgramSpec | None = None,
    expected_slots: Mapping[str, Any] | None = None,
    target_coordinate: ProgressCoordinate | None = None,
    barrier_mapping: CheckpointForkBarrierMapping | Mapping[str, Any] | None = None,
    expected_population_member_ids: Mapping[str, Sequence[str]] | None = None,
    source_slot_transforms: Mapping[str, ResumeSlotTransform] | None = None,
    source_transform_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    segment_history_templates: Mapping[str, Any] | None = None,
    target_slot_transform: ResumeSlotTransform | None = None,
    target_transform_metadata: Mapping[str, Any] | None = None,
    target_transformed_slots: Sequence[str] | None = None,
    target_only_slots: Mapping[str, Mapping[str, Any]] | None = None,
    continuation_request: CheckpointContinuationRequest | Mapping[str, Any] | None = None,
    continuation_applied: bool = True,
    tool_version: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    fork_provenance_metadata: Mapping[str, Any] | None = None,
    content_provider: ArtifactBlobProvider | None = None,
    source_content_binding: ContentBinding | None = None,
    semantic_model: SemanticModelWriter | None = None,
    source_scope: CheckpointReadScope = "full",
) -> CheckpointForkResult:
    """Fork one valid custody checkpoint into a target run contract/root.

    The pipeline is deliberately topology-aware: source/pre transforms run on
    the source topology, declared continuation extends that raw topology from a
    dedicated continuation template, then an optional target/post transform
    produces the target topology. ``target_only_slots`` declares root slots
    introduced only by that post transform. Target slots use the typed writer.
    ``latest.json`` is written after strict target-bound resume validation succeeds.

    A fork retains its source barrier and progress coordinate unless the caller
    supplies ``barrier_mapping``. Crossing barriers is therefore explicit: the
    mapping must name the actual source barrier, an existing target barrier,
    and a declared target coordinate/mapping rationale.

    """
    source = _load_latest_checkpoint_transaction(source_root, scope=source_scope)
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
    transforms = dict(source_slot_transforms or {})
    transform_meta = dict(source_transform_metadata or {})
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
    slot_roles = _slot_roles(target_run_spec)
    _validate_standard_train_writer(target_run_spec, barrier, slot_roles, semantic_model)
    if semantic_model is not None:
        _validate_standard_train_writer(target_run_spec, barrier, slot_roles, semantic_model)
    mapped_source_drops = sorted(
        slot
        for slot, axes in source_axes.items()
        if axes and slot not in {spec.slot for spec in barrier.slots}
    )
    if mapped_source_drops:
        raise CheckpointCompatibilityError(
            f"checkpoint fork cannot drop mapped source slots {mapped_source_drops!r}"
        )

    manifest_metadata = dict(source.manifest.metadata)
    manifest_metadata["phase"] = barrier.phase
    manifest_metadata["forked_from_transaction_id"] = source.manifest.transaction_id
    manifest_metadata.update(dict(metadata or {}))
    if semantic_model is not None:
        from feedbax.contracts.studio_training import StudioGraphMethodPayload
        from feedbax.training.studio_executor import standard_train_optimizer_step

        target_method = StudioGraphMethodPayload.model_validate(target_run_spec.method_payload.payload)
        optimizer_step = standard_train_optimizer_step(target_method, prepared_slots)
        manifest_metadata["optimizer_step"] = optimizer_step
        schedule = target_method.optimizer.lr_schedule
        if (
            request is not None and schedule is not None and schedule.kind != "constant"
            and optimizer_step != (
                request.target_total if continuation_applied else request.source_completed_batches
            )
        ):
            raise CheckpointCompatibilityError(
                "/slots/optimizer: nonconstant schedule requires a carried count at the batch cursor"
            )
    target_binding = run_contract_binding(target_run_spec, phase_program)
    source_coordinate = source.manifest.completed_coordinate
    boundary_coordinate_fields = {
        "phase", "inner_step", "outer_step", "program_step", "schedule_origin_step"
    }
    manifest_metadata.pop(PHASE_BOUNDARY_DECISION_METADATA_KEY, None)
    source_decision = source.manifest.metadata.get(PHASE_BOUNDARY_DECISION_METADATA_KEY)
    same_boundary = (
        source.manifest.run_contract_binding.phase_program_sha256
        == target_binding.phase_program_sha256
        and source_coordinate.model_dump(include=boundary_coordinate_fields)
        == coordinate.model_dump(include=boundary_coordinate_fields)
    )
    # Pending standard Train continues the same terminal phase. Its target
    # guard and schedule origin may differ, so preserve only the terminal fact.
    pending_studio_terminal = (
        semantic_model is not None and request is not None and not continuation_applied
        and source_decision == {"schema_version": 1, "next_phase": None}
        and source_coordinate.model_dump(include=boundary_coordinate_fields - {"schedule_origin_step"})
        == coordinate.model_dump(include=boundary_coordinate_fields - {"schedule_origin_step"})
    )
    if source_decision is not None and resolved_barrier_mapping is None and (
        same_boundary or pending_studio_terminal
    ):
        manifest_metadata[PHASE_BOUNDARY_DECISION_METADATA_KEY] = source_decision
    if request is not None:
        manifest_metadata["checkpoint_continuation"] = request.model_dump(
            mode="json",
            exclude_none=True,
        )
        manifest_metadata["checkpoint_continuation_applied"] = continuation_applied
    else:
        manifest_metadata.pop("checkpoint_continuation_applied", None)
    segment_parent = None
    if request is not None and continuation_applied and request.source_completed_batches > 0:
        segment_parent = load_checkpoint_set(source.root, source.manifest.transaction_id).exact_ref
        manifest_metadata["checkpoint_segment_parent_ref"] = segment_parent.model_dump(
            mode="json"
        )
    else:
        manifest_metadata.pop("checkpoint_segment_parent_ref", None)
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
            else request.source_completed_batches
            if request is not None
            else None
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
        semantic_abis = derive_checkpoint_semantic_abis(
            {
                spec.slot: structural_abi_fingerprint(prepared_slots[spec.slot])
                for spec in barrier.slots
                if spec.slot in prepared_slots
            },
            slot_roles=slot_roles,
            optimizer_bindings=phase_program.optimizer_bindings,
        )
        population_slots = _population_slot_names(target_run_spec)
        source_slots_by_name = {slot.slot: slot for slot in _checkpoint_scope_slots(source.manifest, source_scope)}
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
                transaction_dir=tmp_dir,
                coordinate=coordinate,
                slot_roles=slot_roles,
                population_slots=population_slots,
                population_member_ids=expected_population_member_ids or {},
                materialized_axes=expected_axes,
                semantic_abi=semantic_abis[spec.slot],
                semantic_model=semantic_model,
                completed_batches=completed_batches,
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
                transform=(
                    _combine_transform_stages(spec.slot, transform_stages[spec.slot])
                    if spec.slot in transformed_slots else None
                ),
                source_axes=(
                    source_slot.materialized_axes if source_slot is not None else None
                ),
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
                    if segment_parent is not None
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
            run_contract_binding=target_binding,
            slots=slot_records,
            content_integrity_digest=ContentIntegrityDigest(
                semantic_state_root_algorithm_version=SEMANTIC_STATE_ROOT_ALGORITHM_VERSION,
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
        manifest_sha256 = sha256_file(manifest_path)
        checkpoint_set = _published_checkpoint_set(
            run_spec=target_run_spec,
            manifest=manifest,
            parent=segment_parent,
        )
        checkpoint_set_path = tmp_dir / CHECKPOINT_SET_NAME
        _write_json_atomic(
            checkpoint_set_path,
            checkpoint_set.model_dump(mode="json", exclude_none=True),
        )
        os.replace(tmp_dir, final_dir)
        moved_to_final = True
        manifest_path = final_dir / MANIFEST_NAME
        checkpoint_set_path = final_dir / CHECKPOINT_SET_NAME
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
            continuation_request=(
                request if request is not None and not continuation_applied else None
            ),
            allow_new_lineage_override=False,
        )
        latest_path = target_root_path / LATEST_POINTER_NAME
        _write_json_atomic(
            latest_path,
            latest_pointer.model_dump(mode="json", exclude_none=True),
        )
        content_binding: ContentBinding | None = None
        content_root: ContentRootRecord | None = None
        if content_provider is not None:
            reusable_objects: dict[str, ContentObjectRef] = {}
            if source_content_binding is not None:
                source_tree = content_provider.get_tree(source_content_binding.tree)
                reusable_objects = {
                    member.object.sha256: member.object for member in source_tree.files
                }
            content_binding, content_root = _publish_checkpoint_content(
                content_provider,
                latest_pointer_path=latest_path,
                manifest_path=manifest_path,
                checkpoint_set_path=checkpoint_set_path,
                manifest=manifest,
                checkpoint_set=checkpoint_set,
                reusable_objects=reusable_objects,
            )
        return CheckpointForkResult(
            root=target_root_path,
            transaction_dir=final_dir,
            manifest_path=manifest_path,
            latest_pointer_path=latest_path,
            manifest=manifest,
            latest_pointer=latest_pointer,
            checkpoint_set_path=checkpoint_set_path,
            checkpoint_set=checkpoint_set,
            slot_transfer_modes=transfer_modes,
            source_provenance_notices=source.provenance_notices,
            content_binding=content_binding,
            content_root=content_root,
        )
    except Exception:
        shutil.rmtree(final_dir if moved_to_final else tmp_dir, ignore_errors=True)
        raise


@contextmanager
def materialize_checkpoint_content(
    binding: ContentBinding,
    provider: ArtifactBlobProvider,
    *,
    owner: str,
    scope: CheckpointReadScope = "full",
) -> Iterator[Path]:
    """Lease and materialize one canonical checkpoint tree for a bounded operation."""
    expected_kind = _checkpoint_content_kind(scope)
    allowed_kinds = {expected_kind, "checkpoint-custody"} if scope == "model_only" else {expected_kind}
    if binding.root_kind not in allowed_kinds:
        raise CheckpointCustodyError(
            f"canonical checkpoint materialization requires a binding for {scope} custody"
        )
    from feedbax.persistence.content_materialization import (
        acquire_content_projection_lease,
        compile_content_input_projection,
        materialize_content_input_projection,
    )

    # The private scope is checked above. Transport uses the existing
    # checkpoint tree role and authenticates every member of the chosen view.
    transport = binding.model_copy(update={"root_kind": "checkpoint-custody"})
    projection = compile_content_input_projection((transport,), provider)
    lease_id = f"{owner}:{uuid.uuid4().hex}"
    acquire_content_projection_lease(
        projection,
        provider,
        lease_id=lease_id,
        owner=owner,
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
    )
    try:
        with tempfile.TemporaryDirectory(prefix="feedbax-checkpoint-content-") as temporary:
            staged = materialize_content_input_projection(
                projection,
                provider,
                destination_root=Path(temporary) / "trees",
            )
            if len(staged) != 1:
                raise CheckpointCustodyError("checkpoint source projection must contain one tree")
            yield staged[0].destination
    finally:
        provider.release_lease(lease_id)


def _load_latest_checkpoint_transaction(
    root: str | Path, *,
    scope: CheckpointReadScope = "full",
) -> _LoadedCheckpointTransaction:
    root_path = Path(root)
    latest = _load_latest_pointer(root_path)
    manifest_path, manifest = _manifest_from_latest_pointer(root_path, latest)
    loaded_slots: dict[str, Any] = {}
    transaction_dir = manifest_path.parent
    for slot in _checkpoint_scope_slots(manifest, scope):
        loaded_slots[slot.slot] = _deserialize_checkpoint_slot(
            slot, transaction_dir, manifest, loaded_model=loaded_slots.get("model"),
        )
    provenance_notices, _ = _validate_manifest_structural_abi(
        manifest, loaded_slots, layout="authenticated_source",
    )
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


def _validate_standard_train_writer(
    run_spec: TrainingRunSpec,
    barrier: CheckpointBarrierSpec,
    slot_roles: Mapping[str, str],
    writer: SemanticModelWriter | None,
) -> None:
    from feedbax.contracts.graph import GraphSpec

    if (
        run_spec.method_ref != STUDIO_GRAPH_METHOD or writer is None
        or not STANDARD_TRAIN_SLOTS.items() <= slot_roles.items()
        or {spec.slot for spec in barrier.slots} != {"model", *STANDARD_TRAIN_SLOTS}
    ):
        raise CheckpointCompatibilityError(
            "/slots: typed Train state requires the standard Studio checkpoint layout"
        )
    if run_spec.graph.inline is None or GraphSpec.model_validate(run_spec.graph.inline) != writer.graph:
        raise CheckpointCompatibilityError("/graph: typed Train writer differs from the bound graph")


def _write_fresh_slot_blob(
    spec: CheckpointSlotSpec,
    value: Any,
    *,
    transaction_dir: Path,
    coordinate: ProgressCoordinate,
    slot_roles: Mapping[str, str],
    population_slots: set[str],
    population_member_ids: Mapping[str, Sequence[str]],
    semantic_abi: CheckpointSemanticAbi,
    materialized_axes: tuple[MaterializedSlotAxisBinding, ...] | None = None,
    semantic_model: SemanticModelWriter | None = None,
    completed_batches: int | None = None,
) -> tuple[CheckpointSlotBlobRef, SlotContentDigest]:
    _validate_slot_axes(
        spec.slot,
        value,
        materialized_axes,
        error_cls=CheckpointCompatibilityError,
    )
    metadata_for_slot = dict(spec.metadata)
    if spec.slot == "model" and semantic_model is not None:
        relative, tree, encoding = semantic_model.publish(transaction_dir, value)
        blob_sha256, blob_size_bytes = tree.sha256, tree.size_bytes
        media_type = MODEL_TREE_MEDIA_TYPE
        serializer = MODEL_TREE_SERIALIZER
        metadata_for_slot["semantic_model"] = encoding
    elif spec.slot in STANDARD_TRAIN_SLOTS:
        encoded = encode_standard_train_slot(
            spec.slot, value, semantic_abi, completed_updates=completed_batches,
        )
        blob_sha256, blob_size_bytes = sha256_bytes(encoded.data), len(encoded.data)
        relative = Path("blobs", f"{spec.slot}-{blob_sha256}.safetensors")
        _write_bytes_atomic(transaction_dir / relative, encoded.data)
        media_type = STANDARD_TRAIN_STATE_MEDIA_TYPE
        serializer = STANDARD_TRAIN_STATE_CODEC_VERSION
        metadata_for_slot["array_descriptor_sha256"] = encoded.descriptor_sha256
    else:
        raise CheckpointCompatibilityError(f"no typed checkpoint writer for slot {spec.slot!r}")
    integrity = _slot_integrity_records(value)
    structural_abi = integrity.structural_abi_fingerprint.model_copy(update={
        "environment_provenance": _serializer_versions(serializer),
    })
    population = _population_record(
        spec.slot,
        value,
        population_member_ids=population_member_ids,
        population_slots=population_slots,
    )
    role = slot_roles.get(spec.slot, "auxiliary")
    content_digest = _slot_content_digest(
        slot=spec.slot,
        role=role,
        value=value,
        blob_sha256=blob_sha256,
        blob_size_bytes=blob_size_bytes,
        integrity=integrity,
        materialized_axes=materialized_axes,
        population=population,
    )
    slot_record = CheckpointSlotBlobRef(
        slot=spec.slot,
        role=role,
        required=spec.required,
        media_type=media_type,
        relative_path=relative.as_posix(),
        sha256=blob_sha256,
        size_bytes=blob_size_bytes,
        coordinate=coordinate,
        structural_abi_fingerprint=structural_abi,
        semantic_abi=semantic_abi,
        content_digest=content_digest,
        materialized_axes=materialized_axes,
        population=population,
        metadata=metadata_for_slot,
    )
    return slot_record, content_digest


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
    objective = training_run_spec["trial_documents"]["objective"]
    graph = training_run_spec["graph"]
    optimizer_bindings = [
        binding.model_dump(mode="json", exclude_none=True)
        for binding in phase_program.optimizer_bindings
    ]
    return RunContractBinding(
        algorithm_version=_RUN_CONTRACT_BINDING_ALGORITHM_V4,
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
    """Return the stored scientific projection for run-contract binding v4.

    The projection includes the current ``TrainingRunSpec`` payload and the
    phase program used by the checkpoint barrier, with signed-zero normalization
    applied when hashing.
    """
    migrated_run_spec = _canonical_training_run_spec_payload(run_spec)
    return {
        "schema_id": "feedbax.manifest.training_checkpoint.run_contract_projection",
        "schema_version": "feedbax.manifest.training_checkpoint.run_contract_projection.v1",
        "algorithm_version": _RUN_CONTRACT_BINDING_ALGORITHM_V4,
        "training_run_spec": migrated_run_spec,
        "phase_program": phase_program.model_dump(mode="json", exclude_none=True),
    }


def authenticated_run_contract_source_projection(
    manifest: CheckpointTransactionManifest,
) -> tuple[TrainingRunSpec, PhaseProgramSpec]:
    """Load the coherent canonical source spec stored by binding algorithm v4.

    This accepts no reconstruction path: a pre-v4 binding, absent projection,
    mismatched projection digest, or invalid embedded contract fails closed.
    """
    binding = manifest.run_contract_binding
    if binding.algorithm_version != _RUN_CONTRACT_BINDING_ALGORITHM_V4:
        raise CheckpointCompatibilityError(
            "continuation schedule preflight requires run-contract binding algorithm v4; "
            f"observed={binding.algorithm_version!r}"
        )
    projection = binding.canonical_projection
    if projection is None or binding.canonical_projection_sha256 is None:
        raise CheckpointCompatibilityError(
            "run-contract binding v4 lacks its authenticated canonical projection"
        )
    if _run_contract_hash(projection) != binding.canonical_projection_sha256:
        raise CheckpointCompatibilityError("run-contract canonical projection digest mismatch")
    expected_identity = {
        "schema_id": "feedbax.manifest.training_checkpoint.run_contract_projection",
        "schema_version": "feedbax.manifest.training_checkpoint.run_contract_projection.v1",
        "algorithm_version": _RUN_CONTRACT_BINDING_ALGORITHM_V4,
    }
    observed_identity = {key: projection.get(key) for key in expected_identity}
    if observed_identity != expected_identity:
        raise CheckpointCompatibilityError(
            "run-contract canonical projection identity is incoherent; "
            f"observed={observed_identity!r}"
        )
    try:
        run_spec = TrainingRunSpec.model_validate(projection["training_run_spec"])
        phase_program = PhaseProgramSpec.model_validate(projection["phase_program"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointCompatibilityError(
            f"run-contract canonical projection is invalid: {exc}"
        ) from exc
    return run_spec, phase_program


def structural_abi_fingerprint(value: Any) -> StructuralAbiFingerprint:
    """Return a structural PyTree ABI fingerprint for a slot value."""
    # This local import avoids the preparation/checkpoint-custody import cycle.
    from feedbax.training.preparation import _thaw_runtime_value

    value = _thaw_runtime_value(value)
    pairs, treedef = jt.flatten_with_path(value)
    leaves = [_leaf_fingerprint(path, leaf) for path, leaf in pairs]
    return _structural_abi_fingerprint_from_leaves(_structural_treedef_text(treedef), leaves)


_MJX_ARRAY_WRAPPER = "mujoco.mjx._src.dataclasses._NumPyArrayHashWrapper"
_MJX_ARRAY_REPR = re.compile(rf"<{re.escape(_MJX_ARRAY_WRAPPER)} object at 0x[0-9a-fA-F]+>")


def _structural_treedef_text(treedef: PyTreeDef) -> str:
    """Identify pinned MJX static metadata by array content instead of address."""
    text = str(treedef)
    if _MJX_ARRAY_WRAPPER not in text:
        return text

    replacements: dict[str, str] = {}
    nodes = [treedef]
    while nodes:
        node = nodes.pop()
        nodes.extend(node.children())
        pending: list[Any] = [node.node_data()]
        while pending:
            value = pending.pop()
            if isinstance(value, tuple):
                pending.extend(value)
            elif _qualified_type_name(value) == _MJX_ARRAY_WRAPPER:
                array = value.array
                digest = sha256_bytes(array.tobytes(order="C"))
                replacements[repr(value)] = (
                    f"<MJX static array dtype={array.dtype.str!r} "
                    f"shape={array.shape!r} sha256={digest}>"
                )
    return _MJX_ARRAY_REPR.sub(lambda match: replacements[match[0]], text)


def _structural_abi_fingerprint_from_leaves(
    treedef: str,
    leaves: Sequence[SlotLeafFingerprint],
) -> StructuralAbiFingerprint:
    environment_provenance = _serializer_versions()
    fingerprint = StructuralAbiFingerprint(
        treedef=treedef,
        leaf_count=len(leaves),
        leaves=list(leaves),
        environment_provenance=environment_provenance,
        fingerprint_sha256="",
    )
    return fingerprint.model_copy(
        update={"fingerprint_sha256": structural_abi_content_sha256(fingerprint)}
    )


def _format_structural_abi_diff(
    recorded: StructuralAbiFingerprint,
    actual: StructuralAbiFingerprint,
    *,
    recorded_value: Any | None = None,
    actual_value: Any | None = None,
) -> str:
    diffs = _structural_abi_leaf_diffs(recorded, actual)
    displayed_diffs = diffs[:10]
    leaf_diff_text = "; ".join(_format_structural_abi_leaf_diff(diff) for diff in displayed_diffs)
    if not leaf_diff_text:
        leaf_diff_text = "<none in comparable leaves>"
    elif len(diffs) > len(displayed_diffs):
        leaf_diff_text += f"; ... ({len(diffs)} total differing leaves)"
    treedef_equal = _canonical_structural_abi_treedef(
        recorded.treedef,
        value=recorded_value,
    ) == _canonical_structural_abi_treedef(actual.treedef, value=actual_value)
    suffix = (
        f"; treedef_equal={treedef_equal}"
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
                    actual=structural_abi_leaf_content_projection(actual_leaf),
                )
            )
            continue
        if index >= len(actual.leaves):
            recorded_leaf = recorded.leaves[index]
            diffs.append(
                _StructuralAbiLeafDiff(
                    path=recorded_leaf.path,
                    field="leaf",
                    recorded=structural_abi_leaf_content_projection(recorded_leaf),
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
    expected_phase_program: PhaseProgramSpec,
    loaded_fingerprints: Mapping[str, StructuralAbiFingerprint] | None = None,
) -> None:
    loaded_fingerprints = loaded_fingerprints or {}
    manifest_slots = {slot.slot for slot in manifest.slots}
    expected_fingerprints = {
        name: structural_abi_fingerprint(value)
        for name, value in expected_slots.items()
        if name in manifest_slots
    }
    actual_fingerprints = {
        name: loaded_fingerprints.get(name) or structural_abi_fingerprint(value)
        for name, value in loaded_slots.items()
        if name in manifest_slots
    }
    expected_semantic = derive_checkpoint_semantic_abis(
        expected_fingerprints,
        slot_roles={slot.slot: slot.role for slot in manifest.slots},
        optimizer_bindings=expected_phase_program.optimizer_bindings,
    )
    transformed_semantic = derive_checkpoint_semantic_abis(
        actual_fingerprints,
        slot_roles={slot.slot: slot.role for slot in manifest.slots},
        optimizer_bindings=expected_phase_program.optimizer_bindings,
    )
    for slot in manifest.slots:
        if slot.slot not in expected_slots:
            continue
        if slot.slot not in loaded_slots:
            continue
        loaded = loaded_slots[slot.slot]
        loaded_fingerprint = actual_fingerprints[slot.slot]
        expected = expected_fingerprints[slot.slot]
        target_abi = expected_semantic[slot.slot]
        source_abi = (
            slot.semantic_abi
            if slot.slot in loaded_fingerprints and slot.semantic_abi is not None
            else transformed_semantic[slot.slot]
        )
        source_identities = {leaf.identity: leaf for leaf in source_abi.leaves}
        target_identities = {leaf.identity: leaf for leaf in target_abi.leaves}
        missing = sorted(
            identity
            for identity, leaf in target_identities.items()
            if leaf.required and identity not in source_identities
        )
        unknown = sorted(
            identity
            for identity, leaf in source_identities.items()
            if leaf.required and identity not in target_identities
        )
        if missing or unknown:
            raise CheckpointCompatibilityError(
                f"checkpoint slot {slot.slot!r} structural ABI mismatch: semantic identity; "
                f"missing_required={missing!r} unknown_required={unknown!r}"
            )
        loaded_by_path = {leaf.path: leaf for leaf in loaded_fingerprint.leaves}
        expected_by_path = {leaf.path: leaf for leaf in expected.leaves}
        semantic_diffs: list[_StructuralAbiLeafDiff] = []
        for identity in sorted(set(source_identities) & set(target_identities)):
            source_identity = source_identities[identity]
            target_identity = target_identities[identity]
            source_leaf = loaded_by_path[source_identity.raw_path]
            target_leaf = expected_by_path[target_identity.raw_path]
            for field in _STRUCTURAL_ABI_DIFF_FIELDS:
                source_value = getattr(source_leaf, field)
                target_value = getattr(target_leaf, field)
                if field == "leaf_type":
                    source_value = _canonical_leaf_type(source_value)
                    target_value = _canonical_leaf_type(target_value)
                if source_value != target_value:
                    semantic_diffs.append(
                        _StructuralAbiLeafDiff(
                            path=target_identity.raw_path,
                            field=field,
                            recorded=target_value,
                            actual=source_value,
                        )
                    )
        if semantic_diffs:
            detail = "; ".join(
                _format_structural_abi_leaf_diff(diff) for diff in semantic_diffs[:10]
            )
            treedef_equal = _canonical_structural_abi_treedef(
                loaded_fingerprint.treedef,
                value=loaded,
            ) == _canonical_structural_abi_treedef(
                expected.treedef,
                value=expected_slots[slot.slot],
            )
            raise CheckpointCompatibilityError(
                f"checkpoint slot {slot.slot!r} structural ABI mismatch: semantic leaf; "
                f"treedef_equal={treedef_equal}; "
                f"leaf_count_delta={expected.leaf_count - loaded_fingerprint.leaf_count}; "
                f"{detail}"
            )
        raw_paths_equal = {
            identity: source_identities[identity].raw_path
            for identity in set(source_identities) & set(target_identities)
        } == {
            identity: target_identities[identity].raw_path
            for identity in set(source_identities) & set(target_identities)
        }
        if raw_paths_equal and _canonical_structural_abi_treedef(
            loaded_fingerprint.treedef,
            value=loaded,
        ) != _canonical_structural_abi_treedef(
            expected.treedef,
            value=expected_slots[slot.slot],
        ):
            raise CheckpointCompatibilityError(
                f"checkpoint slot {slot.slot!r} structural ABI mismatch: stable static "
                "component metadata differs at unchanged semantic paths; treedef_equal=False"
            )


def _source_layout_abi(
    slot: CheckpointSlotBlobRef, actual: StructuralAbiFingerprint,
) -> StructuralAbiFingerprint:
    """Retain the producer's legacy PyTree spelling for a codec-built source layout.

    These codecs reconstruct static layout from the authenticated graph and
    registered builder. Their named leaves are checked against the manifest.
    The original runtime representation can order equivalent static maps
    differently from the canonical JSON source.
    """
    if slot.media_type == MODEL_TREE_MEDIA_TYPE or (
        slot.media_type == STANDARD_TRAIN_STATE_MEDIA_TYPE and slot.slot == "optimizer"
    ):
        return actual.model_copy(update={"treedef": slot.structural_abi_fingerprint.treedef})
    return actual


def _validate_manifest_structural_abi(
    manifest: CheckpointTransactionManifest,
    loaded_slots: Mapping[str, Any],
    *,
    layout: Literal["prepared_template", "authenticated_source"] = "prepared_template",
) -> tuple[list[CheckpointProvenanceNotice], dict[str, StructuralAbiFingerprint]]:
    notices: list[CheckpointProvenanceNotice] = []
    loaded_fingerprints: dict[str, StructuralAbiFingerprint] = {}
    for slot in manifest.slots:
        if slot.slot not in loaded_slots:
            continue
        loaded_fingerprint = structural_abi_fingerprint(loaded_slots[slot.slot])
        loaded_fingerprints[slot.slot] = loaded_fingerprint
        comparable = (
            _source_layout_abi(slot, loaded_fingerprint)
            if layout == "authenticated_source" else loaded_fingerprint
        )
        if _semantic_structural_abi_sha256(
            comparable,
            value=loaded_slots[slot.slot],
        ) != _semantic_structural_abi_sha256(
            slot.structural_abi_fingerprint,
            value=loaded_slots[slot.slot],
        ):
            diff_suffix = _format_structural_abi_diff(
                slot.structural_abi_fingerprint,
                loaded_fingerprint,
                recorded_value=loaded_slots[slot.slot],
                actual_value=loaded_slots[slot.slot],
            )
            raise CheckpointIntegrityError(
                f"checkpoint slot {slot.slot!r} structural ABI fingerprint is stale{diff_suffix}"
            )
        notice = _environment_provenance_notice(
            slot.slot, slot.structural_abi_fingerprint,
            serializer=_supported_slot_serializer(slot),
        )
        if notice is not None:
            _LOGGER.warning(notice.message)
            notices.append(notice)
    current_slots = {
        slot.slot
        for slot in manifest.slots
        if slot.semantic_abi is not None
        and slot.semantic_abi.algorithm_version
        == "feedbax.training_checkpoint.semantic_abi.v1"
    }
    semantic_fingerprints = {
        name: fingerprint for name, fingerprint in loaded_fingerprints.items() if name in current_slots
    }
    if any(
        slot.role == "optimizer" and slot.slot in semantic_fingerprints
        for slot in manifest.slots
    ):
        for slot in manifest.slots:
            if slot.role == "model" and slot.slot in current_slots:
                semantic_fingerprints.setdefault(slot.slot, slot.structural_abi_fingerprint)
    current_semantic = derive_checkpoint_semantic_abis(
        semantic_fingerprints,
        slot_roles={slot.slot: slot.role for slot in manifest.slots if slot.slot in current_slots},
        optimizer_bindings=_manifest_optimizer_bindings(manifest),
    )
    for slot in manifest.slots:
        if (
            slot.slot not in loaded_fingerprints or slot.slot not in current_semantic
            or slot.semantic_abi is None
        ):
            continue
        if slot.semantic_abi != current_semantic[slot.slot]:
            raise CheckpointIntegrityError(
                f"checkpoint slot {slot.slot!r} semantic ABI fingerprint is stale: "
                "recorded ownership differs from decoded slot ownership"
            )
    return notices, loaded_fingerprints


def _manifest_optimizer_bindings(
    manifest: CheckpointTransactionManifest,
) -> Sequence[Mapping[str, Any]]:
    projection = manifest.run_contract_binding.canonical_projection
    if not isinstance(projection, Mapping):
        return ()
    phase_program = projection.get("phase_program")
    if not isinstance(phase_program, Mapping):
        return ()
    bindings = phase_program.get("optimizer_bindings")
    return tuple(binding for binding in bindings or () if isinstance(binding, Mapping))


def validate_checkpoint_run_contract(
    manifest: CheckpointTransactionManifest,
    expected_run_spec: TrainingRunSpec,
    expected_phase_program: PhaseProgramSpec,
    *,
    allow_new_lineage_override: bool = False,
) -> None:
    """Check the resume contract without loading checkpoint slots."""
    expected = run_contract_binding(expected_run_spec, expected_phase_program)
    if _contract_binding_matches(manifest.run_contract_binding, expected):
        return
    if allow_new_lineage_override:
        return
    diffs = _run_contract_binding_diffs(manifest.run_contract_binding, expected)
    diff_suffix = ""
    if diffs:
        diff_text = "; ".join(_format_binding_diff(diff) for diff in diffs[:8])
        diff_suffix = f"; differing_fields={diff_text}"
    raise CheckpointContractBindingError(
        "checkpoint run-contract content binding does not match expected run spec"
        f"{diff_suffix}"
    )


def _contract_binding_matches(recorded: RunContractBinding, expected: RunContractBinding) -> bool:
    projection = recorded.canonical_projection
    return (
        recorded.algorithm_version == expected.algorithm_version == _RUN_CONTRACT_BINDING_ALGORITHM_V4
        and recorded.hash_domain == expected.hash_domain == _RUN_CONTRACT_HASH_DOMAIN
        and projection is not None
        and _run_contract_hash(projection)
        == recorded.canonical_projection_sha256
        == expected.canonical_projection_sha256
    )


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
    from feedbax.contracts.spec_storage import canonical_training_run_spec_projection

    return canonical_training_run_spec_projection(run_spec)


def _environment_provenance_notice(
    slot: str,
    fingerprint: StructuralAbiFingerprint,
    *,
    serializer: str = STRUCTURAL_ABI_PROVENANCE,
) -> CheckpointProvenanceNotice | None:
    current = _serializer_versions(serializer).model_dump(mode="json", exclude_none=True)
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


def _supported_slot_serializer(slot: CheckpointSlotBlobRef) -> str:
    provenance = slot.structural_abi_fingerprint.environment_provenance
    serializer = None if provenance is None else provenance.serializer
    if slot.media_type == MODEL_TREE_MEDIA_TYPE:
        if slot.slot != "model" or slot.role != "model" or serializer != MODEL_TREE_SERIALIZER:
            raise CheckpointCompatibilityError("Invalid semantic model slot codec declaration")
        return MODEL_TREE_SERIALIZER
    if slot.media_type == STANDARD_TRAIN_STATE_MEDIA_TYPE:
        if (
            STANDARD_TRAIN_SLOTS.get(slot.slot) != slot.role
            or serializer != STANDARD_TRAIN_STATE_CODEC_VERSION
        ):
            raise CheckpointCompatibilityError("Invalid typed standard Train slot codec declaration")
        return STANDARD_TRAIN_STATE_CODEC_VERSION
    raise CheckpointCompatibilityError(
        f"checkpoint slot {slot.slot!r} has unsupported media_type {slot.media_type!r}"
    )


def _deserialize_checkpoint_slot(
    slot: CheckpointSlotBlobRef, transaction_dir: Path, manifest: CheckpointTransactionManifest, *,
    blob_path: Path | None = None, template: Any = None, loaded_model: Any = None,
) -> Any:
    """Deserialize one hash-verified slot blob without resume transformations."""
    codec = _supported_slot_serializer(slot)
    if codec == MODEL_TREE_SERIALIZER:
        value = SemanticModelSlot.open(transaction_dir, slot).restore(template)
        _validate_decoded_slot_integrity(slot, value)
        return value
    blob_bytes = _read_blob(slot, blob_path or transaction_dir / slot.relative_path)
    if template is None:
        template = (
            standard_train_optimizer_template(manifest, transaction_dir, loaded_model)
            if slot.slot == "optimizer" else standard_train_scalar_template(slot.slot)
        )
    value = decode_standard_train_slot(
        slot, blob_bytes, template=template,
        completed_updates=manifest.completed_training_batches,
    )
    _validate_decoded_slot_integrity(slot, value)
    return value


def _read_blob(
    slot: CheckpointSlotBlobRef, path: Path,
    *, read_bytes: Callable[[Path], bytes] = Path.read_bytes,
) -> bytes:
    if not path.is_file():
        raise CheckpointIntegrityError(f"checkpoint slot {slot.slot!r} blob is missing")
    data = read_bytes(path)
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
        return load_checkpoint_latest_pointer_file(path).document
    except CheckpointIntegrityError as exc:
        # Keep the custody reader's established boundary contract while the
        # public loader retains its more specific typed diagnostic.
        raise CheckpointIntegrityError("checkpoint latest pointer is corrupt") from exc


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
    path_text = checkpoint_leaf_path(path)
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
    identity = _stable_function_identity(leaf) if isinstance(leaf, FunctionType) else None
    static_repr = (
        identity.semantic_name.encode("utf-8")
        if identity is not None
        else repr(leaf).encode("utf-8", errors="replace")
    )
    return SlotLeafFingerprint(
        path=path_text,
        leaf_type=_qualified_type_name(leaf),
        static_repr_sha256=sha256_bytes(static_repr),
    )


def _semantic_structural_abi_sha256(
    fingerprint: StructuralAbiFingerprint,
    *,
    value: Any | None = None,
) -> str:
    return _canonical_hash(_semantic_structural_abi_projection(fingerprint, value=value))


def _semantic_structural_abi_projection(
    fingerprint: StructuralAbiFingerprint,
    *,
    value: Any | None = None,
) -> dict[str, Any]:
    """Project structural ABI fields after canonical semantic normalization."""
    leaves = [
        leaf.model_copy(update={"leaf_type": _canonical_leaf_type(leaf.leaf_type)})
        for leaf in fingerprint.leaves
    ]
    return structural_abi_content_projection(
        fingerprint.model_copy(
            update={
                "treedef": _canonical_structural_abi_treedef(
                    fingerprint.treedef,
                    value=value,
                ),
                "leaves": leaves,
            }
        )
    )


_ANONYMOUS_STATE_INDEX_MARKER = re.compile(
    r"(StateIndex\[\('init',\), \(\('marker', )"
    r"<object object at 0x[0-9a-fA-F]+>"
    r"(\),\)\])"
)


_FUNCTION_REPR = re.compile(r"<function (?P<qualname>[^>]+?) at 0x[0-9a-fA-F]+>")


@dataclass(frozen=True)
class _StableFunctionIdentity:
    qualname: str
    semantic_name: str


def _canonical_structural_abi_treedef(
    treedef: str,
    *,
    value: Any | None = None,
) -> str:
    treedef = _ANONYMOUS_STATE_INDEX_MARKER.sub(r"\1<anonymous StateIndex marker>\2", treedef)
    if value is None:
        return treedef
    identities = _stable_treedef_function_identities(value)
    if identities is None:
        return treedef
    matches = list(_FUNCTION_REPR.finditer(treedef))
    if len(matches) != len(identities) or any(
        match.group("qualname") != identity.qualname
        for match, identity in zip(matches, identities, strict=True)
    ):
        return treedef
    parts: list[str] = []
    start = 0
    for match, identity in zip(matches, identities, strict=True):
        parts.extend((treedef[start : match.start()], f"<function {identity.semantic_name}>"))
        start = match.end()
    parts.append(treedef[start:])
    return "".join(parts)


def _stable_treedef_function_identities(
    value: Any,
) -> list[_StableFunctionIdentity] | None:
    from feedbax.training.preparation import _thaw_runtime_value

    _, treedef = jt.flatten(_thaw_runtime_value(value))
    identities: list[_StableFunctionIdentity] = []

    def collect(node: Any) -> bool:
        node_data = node.node_data()
        if node_data is not None and not _collect_stable_function_identities(
            node_data,
            identities,
        ):
            return False
        return all(collect(child) for child in node.children())

    if not collect(treedef):
        return None
    return identities


def _collect_stable_function_identities(
    value: Any,
    identities: list[_StableFunctionIdentity],
) -> bool:
    static_fields = getattr(value, "static_fields", None)
    if isinstance(static_fields, tuple):
        return all(
            _collect_stable_function_identities(field_value, identities)
            for _, field_value in static_fields
        )
    if inspect.isclass(value):
        return True
    if isinstance(value, FunctionType):
        identity = _stable_function_identity(value)
        if identity is None:
            return False
        identities.append(identity)
        return True
    if callable(value):
        return False
    if isinstance(value, Mapping):
        return all(
            _collect_stable_function_identities(item, identities)
            for pair in value.items()
            for item in pair
        )
    if isinstance(value, (tuple, list)):
        return all(_collect_stable_function_identities(item, identities) for item in value)
    return True


def _stable_function_identity(function: FunctionType) -> _StableFunctionIdentity | None:
    module_name = function.__module__
    qualname = function.__qualname__
    if (
        function.__closure__ is not None
        or not module_name
        or not qualname
        or function.__name__ == "<lambda>"
        or "<locals>" in qualname
    ):
        return None
    resolved: Any = sys.modules.get(module_name)
    if resolved is None:
        return None
    for segment in qualname.split("."):
        resolved = getattr(resolved, segment, None)
        if resolved is None:
            return None
    if resolved is not function:
        return None
    return _StableFunctionIdentity(
        qualname=qualname,
        semantic_name=f"{module_name}.{qualname}",
    )


def _canonical_leaf_type(leaf_type: str) -> str:
    if leaf_type in ("jaxlib.xla_extension.ArrayImpl", "jaxlib._jax.ArrayImpl"):
        return _JAX_ARRAY_LEAF_TYPE
    return leaf_type


def _array_leaf_content_digest(leaf: Any) -> tuple[str, int]:
    """Digest one array leaf's canonical C-order bytes without copying them.

    Returns ``(sha256_hex, size_bytes)`` for exactly the bytes
    ``np.asarray(leaf).tobytes()`` would produce, but hashes the array buffer in
    place so no payload-sized ``bytes`` object is materialized per leaf.
    """
    host = np.asarray(leaf)
    if not host.flags["C_CONTIGUOUS"]:
        host = np.ascontiguousarray(host)
    try:
        # A flat uint8 view exposes the same buffer for every buffer-protocol
        # dtype, including extension dtypes such as bfloat16 that have no
        # native struct format.
        buffer = memoryview(host.reshape(-1).view(np.uint8))
    except (BufferError, TypeError, ValueError):
        data = host.tobytes()
        return sha256_bytes(data), len(data)
    with buffer:
        return hashlib.sha256(buffer).hexdigest(), buffer.nbytes


def _semantic_value_projection(
    leaf: Any,
    *,
    path_text: str,
    active_dataclasses: set[int],
) -> dict[str, Any]:
    if eqx.is_array(leaf):
        array = np.asarray(leaf)
        if array.dtype.hasobject:
            raise CheckpointSemanticStateUnsupported(
                f"checkpoint semantic state does not support object array leaf {path_text!r}"
            )
        value_sha256, size_bytes = _array_leaf_content_digest(array)
        return {
            "kind": "array",
            "dtype": str(array.dtype),
            "dtype_str": array.dtype.str,
            "shape": [int(dim) for dim in array.shape],
            "weak_type": bool(getattr(leaf, "weak_type", False)),
            "value_sha256": value_sha256,
            "size_bytes": size_bytes,
        }
    if leaf is None:
        return {"kind": "none", "value": None}
    if isinstance(leaf, FunctionType):
        identity = _stable_function_identity(leaf)
        if identity is not None:
            return {"kind": "global_function", "identity": identity.semantic_name}
    if type(leaf) is bool:
        return {"kind": "bool", "value": leaf}
    if type(leaf) is int:
        return {"kind": "int", "value": str(leaf)}
    if type(leaf) is float:
        return {
            "kind": "float64",
            "value_bits": np.asarray(leaf, dtype=">f8").tobytes().hex(),
        }
    if type(leaf) is complex:
        return {
            "kind": "complex128",
            "value_bits": np.asarray(leaf, dtype=">c16").tobytes().hex(),
        }
    if type(leaf) is str:
        return {"kind": "str", "value": leaf}
    if type(leaf) is bytes:
        return {"kind": "bytes", "value_hex": leaf.hex()}
    if isinstance(leaf, np.generic):
        scalar = np.asarray(leaf)
        if scalar.dtype.hasobject:
            raise CheckpointSemanticStateUnsupported(
                f"checkpoint semantic state does not support object scalar leaf {path_text!r}"
            )
        return {
            "kind": "numpy_scalar",
            "dtype": str(scalar.dtype),
            "dtype_str": scalar.dtype.str,
            "value_hex": scalar.tobytes().hex(),
        }
    if is_dataclass(leaf) and not isinstance(leaf, type):
        identity = id(leaf)
        if identity in active_dataclasses:
            raise CheckpointSemanticStateUnsupported(
                f"checkpoint semantic state contains a typed dataclass cycle at {path_text!r}"
            )
        active_dataclasses.add(identity)
        try:
            projected_fields = []
            for field in dataclass_fields(leaf):
                field_path = f"{path_text.rstrip('/')}/{field.name}"
                try:
                    field_value = getattr(leaf, field.name)
                except AttributeError as exc:
                    raise CheckpointSemanticStateUnsupported(
                        "checkpoint semantic state cannot read typed dataclass field "
                        f"{field_path!r}"
                    ) from exc
                projected_fields.append(
                    {
                        "name": field.name,
                        "value": _semantic_value_projection(
                            field_value,
                            path_text=field_path,
                            active_dataclasses=active_dataclasses,
                        ),
                    }
                )
        finally:
            active_dataclasses.remove(identity)
        return {
            "kind": "typed_dataclass",
            "type": _qualified_type_name(leaf),
            "fields": projected_fields,
        }
    raise CheckpointSemanticStateUnsupported(
        "checkpoint semantic state has unsupported or ambiguous leaf "
        f"{path_text!r} of type {_qualified_type_name(leaf)!r}"
    )


def _semantic_leaf_value_projection(path: Any, leaf: Any) -> dict[str, Any]:
    path_text = checkpoint_leaf_path(path)
    return {
        "path": path_text,
        **_semantic_value_projection(
            leaf,
            path_text=path_text,
            active_dataclasses=set(),
        ),
    }


def _semantic_state_content_projection(
    *,
    slot: str,
    role: str,
    value: Any,
    structural_abi_fingerprint: StructuralAbiFingerprint,
    materialized_axes: Sequence[MaterializedSlotAxisBinding] | None,
    population: PopulationIdentityRecord | None,
) -> dict[str, Any]:
    """Return the one canonical semantic-state projection for a checkpoint slot."""
    path_leaves, _ = jt.flatten_with_path(value)
    return {
        "algorithm_version": SEMANTIC_STATE_ROOT_ALGORITHM_VERSION,
        "slot": slot,
        "role": role,
        "structural_abi": _semantic_structural_abi_projection(
            structural_abi_fingerprint,
            value=value,
        ),
        "leaves": [_semantic_leaf_value_projection(path, leaf) for path, leaf in path_leaves],
        "materialized_axes": [
            binding.model_dump(mode="json", exclude_none=True)
            for binding in materialized_axes or ()
        ],
        "population": (
            population.model_dump(mode="json", exclude_none=True)
            if population is not None
            else None
        ),
    }


def semantic_state_root_sha256(
    *,
    slot: str,
    role: str,
    value: Any,
    structural_abi_fingerprint: StructuralAbiFingerprint,
    materialized_axes: Sequence[MaterializedSlotAxisBinding] | None,
    population: PopulationIdentityRecord | None,
) -> str:
    """Hash the canonical semantic-state projection for one checkpoint slot."""
    return _canonical_hash(
        _semantic_state_content_projection(
            slot=slot,
            role=role,
            value=value,
            structural_abi_fingerprint=structural_abi_fingerprint,
            materialized_axes=materialized_axes,
            population=population,
        )
    )


def _slot_integrity_records(value: Any) -> _SlotIntegrityRecords:
    pairs, treedef = jt.flatten_with_path(value)
    digests: list[SlotLeafContentDigest] = []
    fingerprints: list[SlotLeafFingerprint] = []
    for path, leaf in pairs:
        fingerprints.append(_leaf_fingerprint(path, leaf))
        if eqx.is_array(leaf):
            # Transfer this leaf alone; the whole-tree host copy that used to be
            # taken here duplicated the entire slot payload in RSS.
            leaf_sha256, leaf_size = _array_leaf_content_digest(leaf)
        else:
            data = pickle.dumps(leaf, protocol=pickle.HIGHEST_PROTOCOL)
            leaf_sha256, leaf_size = sha256_bytes(data), len(data)
        digests.append(
            SlotLeafContentDigest(
                path=checkpoint_leaf_path(path),
                sha256=leaf_sha256,
                size_bytes=leaf_size,
            )
        )
    return _SlotIntegrityRecords(
        leaf_digests=digests,
        structural_abi_fingerprint=_structural_abi_fingerprint_from_leaves(
            _structural_treedef_text(treedef),
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


def _slot_content_digest(
    *,
    slot: str,
    role: str,
    value: Any,
    blob_sha256: str,
    blob_size_bytes: int,
    integrity: _SlotIntegrityRecords,
    materialized_axes: Sequence[MaterializedSlotAxisBinding] | None,
    population: PopulationIdentityRecord | None,
) -> SlotContentDigest:
    return SlotContentDigest(
        slot=slot,
        blob_sha256=blob_sha256,
        blob_size_bytes=blob_size_bytes,
        leaf_hashes=integrity.leaf_digests,
        slot_root_sha256=_slot_root_sha256(
            slot,
            blob_sha256,
            integrity.leaf_digests,
        ),
        semantic_state_root_sha256=semantic_state_root_sha256(
            slot=slot,
            role=role,
            value=value,
            structural_abi_fingerprint=integrity.structural_abi_fingerprint,
            materialized_axes=materialized_axes,
            population=population,
        ),
    )


def _legacy_slot_content_projection(
    digest: SlotContentDigest | LegacySlotContentDigest,
) -> dict[str, Any]:
    """Project the unchanged content-integrity v1 custody identity."""
    return digest.model_dump(
        mode="json",
        exclude_none=True,
        exclude={"semantic_state_root_sha256"},
    )


def _transaction_root_sha256(
    slot_digests: Sequence[SlotContentDigest | LegacySlotContentDigest],
) -> str:
    payload = [
        _legacy_slot_content_projection(digest)
        for digest in sorted(slot_digests, key=lambda digest: digest.slot)
    ]
    return _canonical_hash(payload)


def checkpoint_semantic_state_root(
    manifest: CheckpointTransactionManifest,
    slot: str,
) -> str:
    """Return one stored semantic-state root or report typed legacy unavailability."""
    if manifest.semantic_state_roots_status != "available" or not isinstance(
        manifest.content_integrity_digest,
        ContentIntegrityDigest,
    ):
        raise CheckpointSemanticStateRootUnavailable(
            "checkpoint semantic state root is unavailable for migrated legacy "
            f"content-integrity v1; slot={slot!r}"
        )
    if (
        manifest.content_integrity_digest.schema_id != CHECKPOINT_CONTENT_INTEGRITY_SCHEMA_ID
        or manifest.content_integrity_digest.schema_version
        != CHECKPOINT_CONTENT_INTEGRITY_SCHEMA_VERSION
    ):
        raise CheckpointSemanticStateRootUnavailable(
            "checkpoint semantic state root content-integrity schema is unsupported: "
            f"schema_id={manifest.content_integrity_digest.schema_id!r} "
            f"schema_version={manifest.content_integrity_digest.schema_version!r}"
        )
    if (
        manifest.content_integrity_digest.semantic_state_root_algorithm_version
        != SEMANTIC_STATE_ROOT_ALGORITHM_VERSION
    ):
        raise CheckpointSemanticStateRootUnavailable(
            "checkpoint semantic state root algorithm is unsupported: "
            f"{manifest.content_integrity_digest.semantic_state_root_algorithm_version!r}"
        )
    for digest in manifest.content_integrity_digest.slots:
        if digest.slot == slot:
            if not _is_sha256(digest.semantic_state_root_sha256):
                raise CheckpointSemanticStateRootUnavailable(
                    f"checkpoint semantic state root is invalid for slot {slot!r}"
                )
            return digest.semantic_state_root_sha256
    raise CheckpointSemanticStateRootUnavailable(
        f"checkpoint semantic state root slot is missing: {slot!r}"
    )


def require_checkpoint_semantic_state_equal(
    left: CheckpointTransactionManifest,
    right: CheckpointTransactionManifest,
    slot: str,
) -> None:
    """Require one slot to have the same governed semantic state in two manifests."""
    left_root = checkpoint_semantic_state_root(left, slot)
    right_root = checkpoint_semantic_state_root(right, slot)
    if left_root != right_root:
        raise CheckpointSemanticStateMismatch(
            f"checkpoint semantic state root mismatch for slot {slot!r}: "
            f"left={left_root!r} right={right_root!r}"
        )


def _serializer_versions(serializer: str = STRUCTURAL_ABI_PROVENANCE) -> SerializerVersionRecord:
    return SerializerVersionRecord(
        serializer=serializer,
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


def _durable_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, indent=2).encode("utf-8")


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_bytes_atomic(path, _durable_json_bytes(payload))
