"""Training checkpoint custody contracts.

These records describe Feedbax-owned training checkpoint transactions. They are
separate from evaluation checkpoint-selection records: this module owns resume
state, slot integrity, and run-contract binding for training writers.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from feedbax.contracts.manifest import ArtifactRef, ParentRef, StrictModel
from feedbax.contracts.worker import ConsistencyPredicateSpec, ProgressCoordinate

TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_ID = (
    "feedbax.manifest.training_checkpoint_transaction"
)
TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION = (
    "feedbax.manifest.training_checkpoint_transaction.v1"
)
TRAINING_CHECKPOINT_LATEST_POINTER_SCHEMA_ID = (
    "feedbax.manifest.training_checkpoint_latest_pointer"
)
TRAINING_CHECKPOINT_LATEST_POINTER_SCHEMA_VERSION = (
    "feedbax.manifest.training_checkpoint_latest_pointer.v1"
)
LEGACY_CHECKPOINT_LEAF_MANIFEST_SCHEMA_ID = (
    "feedbax.manifest.legacy_checkpoint_leaf_manifest"
)
LEGACY_CHECKPOINT_LEAF_MANIFEST_SCHEMA_VERSION = (
    "feedbax.manifest.legacy_checkpoint_leaf_manifest.v1"
)
LEGACY_CHECKPOINT_LEAF_MANIFEST_SCHEMA_VERSION_V0 = (
    "feedbax.manifest.legacy_checkpoint_leaf_manifest.v0"
)

CheckpointSlotRole = Literal[
    "model",
    "optimizer",
    "prng",
    "auxiliary",
    "population",
    "environment",
    "objective",
    "checkpoint",
    "metric",
]


class SerializerVersionRecord(StrictModel):
    """Serializer and runtime versions that affect checkpoint compatibility."""

    serializer: str = "feedbax.training.checkpoint_custody.pickle.v1"
    feedbax_version: str
    jax_version: str | None = None
    equinox_version: str | None = None
    optax_version: str | None = None
    python_version: str
    x64_enabled: bool
    metadata: dict[str, Any] = Field(default_factory=dict)


class SlotLeafFingerprint(StrictModel):
    """Structural fingerprint for one PyTree leaf."""

    path: str
    leaf_type: str
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    weak_type: bool | None = None
    sharding: str | None = None
    layout: str | None = None
    static_repr_sha256: str | None = None


class LeafManifestEntry(StrictModel):
    """One ordered leaf slot in a legacy Equinox checkpoint stream ABI."""

    tree_path: str
    kind: Literal["array", "static"]
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    static_repr_sha256: str | None = None

    @model_validator(mode="after")
    def _validate_leaf_metadata(self) -> "LeafManifestEntry":
        if self.kind == "array" and (self.shape is None or self.dtype is None):
            raise ValueError("array manifest entries must include shape and dtype")
        if self.kind == "static" and (self.shape is not None or self.dtype is not None):
            raise ValueError("static manifest entries must not include shape or dtype")
        return self


class LeafManifestProvenance(StrictModel):
    """Where and how a legacy leaf manifest was produced."""

    producing_commit: str
    spec_ref: str | None = None
    spec_hash: str | None = None
    dumped_at: str
    dumper_version: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class LeafManifest(StrictModel):
    """Versioned ABI manifest for legacy ``tree_serialise_leaves`` streams."""

    kind: Literal["LegacyCheckpointLeafManifest"] = "LegacyCheckpointLeafManifest"
    schema_id: str = LEGACY_CHECKPOINT_LEAF_MANIFEST_SCHEMA_ID
    schema_version: str = LEGACY_CHECKPOINT_LEAF_MANIFEST_SCHEMA_VERSION
    model: list[LeafManifestEntry]
    optimizer: list[LeafManifestEntry] = Field(default_factory=list)
    provenance: LeafManifestProvenance
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_schema_identity(self) -> "LeafManifest":
        if self.schema_id != LEGACY_CHECKPOINT_LEAF_MANIFEST_SCHEMA_ID:
            raise ValueError(
                f"unsupported legacy checkpoint leaf manifest schema_id {self.schema_id!r}"
            )
        if self.schema_version != LEGACY_CHECKPOINT_LEAF_MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                "unsupported legacy checkpoint leaf manifest schema_version "
                f"{self.schema_version!r}"
            )
        return self


class StructuralAbiFingerprint(StrictModel):
    """Compatibility fingerprint for a slot PyTree."""

    schema_id: str = "feedbax.manifest.training_checkpoint.structural_abi"
    schema_version: str = "feedbax.manifest.training_checkpoint.structural_abi.v1"
    treedef: str
    leaf_count: int
    leaves: list[SlotLeafFingerprint]
    serializer_versions: SerializerVersionRecord
    fingerprint_sha256: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SlotLeafContentDigest(StrictModel):
    """Content hash for one PyTree leaf when available."""

    path: str
    sha256: str
    size_bytes: int | None = None


class SlotContentDigest(StrictModel):
    """Per-slot content digest record."""

    slot: str
    blob_sha256: str
    blob_size_bytes: int
    leaf_hashes: list[SlotLeafContentDigest] = Field(default_factory=list)
    slot_root_sha256: str


class ContentIntegrityDigest(StrictModel):
    """Content hashes for all slots and the complete transaction."""

    schema_id: str = "feedbax.manifest.training_checkpoint.content_integrity"
    schema_version: str = "feedbax.manifest.training_checkpoint.content_integrity.v1"
    slots: list[SlotContentDigest]
    transaction_root_sha256: str


class RunContractBinding(StrictModel):
    """Content binding between a checkpoint and the run contract that produced it."""

    schema_id: str = "feedbax.manifest.training_checkpoint.run_contract_binding"
    schema_version: str = "feedbax.manifest.training_checkpoint.run_contract_binding.v1"
    hash_domain: Literal["migrated-canonical-json"] = "migrated-canonical-json"
    training_run_spec_schema_id: str
    training_run_spec_schema_version: str
    training_run_spec_sha256: str
    method_payload_schema_id: str
    method_payload_schema_version: str
    method_payload_sha256: str
    phase_program_sha256: str
    objective_sha256: str | None = None
    graph_sha256: str | None = None
    optimizer_bindings_sha256: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PopulationIdentityRecord(StrictModel):
    """Population length and member identity record for a slot."""

    slot: str
    length: int
    member_ids: list[str]

    @model_validator(mode="after")
    def _validate_length(self) -> "PopulationIdentityRecord":
        if self.length != len(self.member_ids):
            raise ValueError(
                f"population slot {self.slot!r} length must match member_ids length"
            )
        return self


class CheckpointSlotBlobRef(StrictModel):
    """One content-addressed slot blob in a transaction."""

    slot: str
    role: CheckpointSlotRole = "auxiliary"
    required: bool = True
    media_type: str = "application/x-python-pickle"
    relative_path: str
    sha256: str
    size_bytes: int
    coordinate: ProgressCoordinate
    structural_abi_fingerprint: StructuralAbiFingerprint
    content_digest: SlotContentDigest
    population: PopulationIdentityRecord | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CheckpointLineageRef(StrictModel):
    """Parent checkpoint lineage for resumed or overridden runs."""

    transaction_id: str
    manifest: ParentRef | ArtifactRef | None = None
    relationship: Literal["parent", "new_lineage_override"] = "parent"
    metadata: dict[str, Any] = Field(default_factory=dict)


class CheckpointTransactionManifest(StrictModel):
    """Durable manifest for one atomic multi-slot checkpoint transaction."""

    kind: Literal["TrainingCheckpointTransactionManifest"] = (
        "TrainingCheckpointTransactionManifest"
    )
    schema_id: str = TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_ID
    schema_version: str = TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION
    transaction_id: str
    run_id: str
    status: Literal["partial", "final"] = "partial"
    barrier: str
    completed_coordinate: ProgressCoordinate
    consistency_predicate: ConsistencyPredicateSpec
    run_contract_binding: RunContractBinding
    slots: list[CheckpointSlotBlobRef]
    content_integrity_digest: ContentIntegrityDigest
    history_availability: dict[str, bool] = Field(default_factory=dict)
    parent_lineage: list[CheckpointLineageRef] = Field(default_factory=list)
    source_training_run: ParentRef | ArtifactRef | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_schema_identity(self) -> "CheckpointTransactionManifest":
        if self.schema_id != TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_ID:
            raise ValueError(
                f"unsupported checkpoint transaction schema_id {self.schema_id!r}"
            )
        if self.schema_version != TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION:
            raise ValueError(
                "unsupported checkpoint transaction schema_version "
                f"{self.schema_version!r}"
            )
        slot_names = [slot.slot for slot in self.slots]
        if len(slot_names) != len(set(slot_names)):
            raise ValueError("checkpoint transaction slot names must be unique")
        return self


class CheckpointLatestPointer(StrictModel):
    """Governed latest pointer published only after a transaction is durable."""

    schema_id: str = TRAINING_CHECKPOINT_LATEST_POINTER_SCHEMA_ID
    schema_version: str = TRAINING_CHECKPOINT_LATEST_POINTER_SCHEMA_VERSION
    run_id: str
    transaction_id: str
    manifest_relative_path: str
    manifest_sha256: str
    transaction_root_sha256: str
    completed_coordinate: ProgressCoordinate


class CheckpointResumeResult(StrictModel):
    """Result of a resume load and validation pass."""

    manifest: CheckpointTransactionManifest
    slots: dict[str, Any]
    new_lineage_required: bool = False
    previous_transaction_id: str | None = None
