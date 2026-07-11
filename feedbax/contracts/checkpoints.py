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

TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_ID = "feedbax.manifest.training_checkpoint_transaction"
TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION_V1 = (
    "feedbax.manifest.training_checkpoint_transaction.v1"
)
TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION_V2 = (
    "feedbax.manifest.training_checkpoint_transaction.v2"
)
TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION_V3 = (
    "feedbax.manifest.training_checkpoint_transaction.v3"
)
TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION = (
    "feedbax.manifest.training_checkpoint_transaction.v4"
)
TRAINING_CHECKPOINT_LATEST_POINTER_SCHEMA_ID = "feedbax.manifest.training_checkpoint_latest_pointer"
TRAINING_CHECKPOINT_LATEST_POINTER_SCHEMA_VERSION = (
    "feedbax.manifest.training_checkpoint_latest_pointer.v2"
)
LEGACY_CHECKPOINT_LEAF_MANIFEST_SCHEMA_ID = "feedbax.manifest.legacy_checkpoint_leaf_manifest"
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


CHECKPOINT_CONTINUATION_REQUEST_SCHEMA_ID = "feedbax.spec.training_checkpoint_continuation"
CHECKPOINT_CONTINUATION_REQUEST_SCHEMA_VERSION = "feedbax.spec.training_checkpoint_continuation.v1"


class BatchIndexedCheckpointLeafSpec(StrictModel):
    """One explicitly declared final-axis checkpoint leaf extended on resume."""

    slot: str = Field(min_length=1)
    tree_path: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_tree_path(self) -> "BatchIndexedCheckpointLeafSpec":
        if not self.tree_path.startswith("/"):
            raise ValueError("/batch_indexed_leaves/tree_path must start with '/'")
        return self


class CheckpointContinuationRequest(StrictModel):
    """Durable declaration for extending a checkpointed row's total horizon.

    Feedbax deliberately does not infer batch-indexed leaves from arbitrary
    PyTrees. The producer must declare the exact slot/tree-path leaves whose
    final axis represents the global training-batch horizon.
    """

    schema_id: str = CHECKPOINT_CONTINUATION_REQUEST_SCHEMA_ID
    schema_version: str = CHECKPOINT_CONTINUATION_REQUEST_SCHEMA_VERSION
    source_completed_batches: int = Field(ge=0)
    additional_batches: int | None = Field(default=None, gt=0)
    target_total_batches: int | None = Field(default=None, ge=0)
    batch_indexed_leaves: list[BatchIndexedCheckpointLeafSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_request(self) -> "CheckpointContinuationRequest":
        if self.schema_id != CHECKPOINT_CONTINUATION_REQUEST_SCHEMA_ID:
            raise ValueError(
                "/schema_id unsupported checkpoint continuation schema_id "
                f"{self.schema_id!r}; expected "
                f"{CHECKPOINT_CONTINUATION_REQUEST_SCHEMA_ID!r}"
            )
        if self.schema_version != CHECKPOINT_CONTINUATION_REQUEST_SCHEMA_VERSION:
            raise ValueError(
                "/schema_version unsupported checkpoint continuation schema_version "
                f"{self.schema_version!r}; expected "
                f"{CHECKPOINT_CONTINUATION_REQUEST_SCHEMA_VERSION!r}; "
                "migration_intentionally_absent=yes"
            )
        if (self.additional_batches is None) == (self.target_total_batches is None):
            raise ValueError(
                "exactly one of /additional_batches or /target_total_batches is required"
            )
        if self.target_total_batches is not None and (
            self.target_total_batches < self.source_completed_batches
        ):
            raise ValueError("/target_total_batches must not precede /source_completed_batches")
        identities = [(leaf.slot, leaf.tree_path) for leaf in self.batch_indexed_leaves]
        if len(identities) != len(set(identities)):
            raise ValueError("/batch_indexed_leaves slot/tree_path pairs must be unique")
        return self

    @property
    def target_total(self) -> int:
        """Return the target total implied by this unambiguous declaration."""
        if self.additional_batches is not None:
            return self.source_completed_batches + self.additional_batches
        assert self.target_total_batches is not None
        return self.target_total_batches


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
        if self.kind == "static" and ((self.shape is None) != (self.dtype is None)):
            raise ValueError(
                "static manifest entries must include both shape and dtype, or neither"
            )
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
    schema_version: str = "feedbax.manifest.training_checkpoint.structural_abi.v2"
    fingerprint_algorithm_version: str = "feedbax.training_checkpoint.structural_abi.content.v2"
    treedef: str
    leaf_count: int
    leaves: list[SlotLeafFingerprint]
    environment_provenance: SerializerVersionRecord | None = None
    provenance_status: Literal["recorded", "unverifiable_legacy"] = "recorded"
    fingerprint_sha256: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class CheckpointProvenanceNotice(StrictModel):
    """Non-fatal provenance drift observed while validating checkpoint structure."""

    code: Literal["environment_provenance_mismatch", "environment_provenance_unverifiable"]
    slot: str
    message: str
    recorded: dict[str, Any] | None = None
    current: dict[str, Any] | None = None
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
    schema_version: str = "feedbax.manifest.training_checkpoint.run_contract_binding.v2"
    algorithm_version: str = "feedbax.training_checkpoint.run_contract_binding.v2"
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
    canonical_projection: dict[str, Any] | None = None
    canonical_projection_sha256: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PopulationIdentityRecord(StrictModel):
    """Population length and member identity record for a slot."""

    slot: str
    length: int
    member_ids: list[str]

    @model_validator(mode="after")
    def _validate_length(self) -> "PopulationIdentityRecord":
        if self.length != len(self.member_ids):
            raise ValueError(f"population slot {self.slot!r} length must match member_ids length")
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


class CheckpointForkSourceRecord(StrictModel):
    """Identity of the source transaction used to create a forked checkpoint."""

    transaction_id: str
    run_id: str
    manifest_sha256: str
    transaction_root_sha256: str
    manifest_relative_path: str | None = None
    slot_content_digests: list[SlotContentDigest] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CheckpointForkTransformRecord(StrictModel):
    """One slot transform applied while forking a checkpoint."""

    slot: str
    identity: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CheckpointForkBarrierMapping(StrictModel):
    """Caller-declared source-to-target checkpoint barrier mapping.

    A normal fork keeps the source barrier and its coordinate exactly. A fork
    that crosses barriers must instead declare both the target coordinate and
    its durable mapping rationale; Feedbax never infers a barrier or progress
    coordinate transition from a target program.
    """

    source_barrier: str = Field(min_length=1)
    target_barrier: str = Field(min_length=1)
    target_coordinate: ProgressCoordinate | None = None
    coordinate_mapping: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _validate_distinct_barrier_coordinate_mapping(
        self,
    ) -> "CheckpointForkBarrierMapping":
        if self.source_barrier == self.target_barrier:
            return self
        if self.target_coordinate is None or self.coordinate_mapping is None:
            raise ValueError(
                "distinct source/target checkpoint barriers require explicit "
                "target_coordinate and coordinate_mapping"
            )
        if self.target_coordinate.completed_barrier != self.target_barrier:
            raise ValueError("/target_coordinate/completed_barrier must equal /target_barrier")
        identity = self.coordinate_mapping.get("identity")
        if not isinstance(identity, str) or not identity:
            raise ValueError("/coordinate_mapping/identity must be a non-empty string")
        parameters = self.coordinate_mapping.get("parameters", {})
        if not isinstance(parameters, dict):
            raise ValueError("/coordinate_mapping/parameters must be a mapping")
        return self


class CheckpointForkSlotProvenance(StrictModel):
    """Per-slot payload provenance for a forked checkpoint."""

    slot: str
    source_sha256: str | None = None
    target_sha256: str
    source_relative_path: str | None = None
    target_relative_path: str
    transfer_mode: Literal["hardlink", "copy", "serialized"]
    transform: CheckpointForkTransformRecord | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CheckpointForkProvenance(StrictModel):
    """Provenance for a checkpoint transaction forked from an existing one."""

    schema_id: str = "feedbax.manifest.training_checkpoint.fork_provenance"
    schema_version: str = "feedbax.manifest.training_checkpoint.fork_provenance.v1"
    source: CheckpointForkSourceRecord
    slots: list[CheckpointForkSlotProvenance]
    tool_version: str
    barrier_mapping: CheckpointForkBarrierMapping | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CheckpointTransactionManifest(StrictModel):
    """Durable manifest for one atomic multi-slot checkpoint transaction."""

    kind: Literal["TrainingCheckpointTransactionManifest"] = "TrainingCheckpointTransactionManifest"
    schema_id: str = TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_ID
    schema_version: str = TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION
    transaction_id: str
    run_id: str
    status: Literal["partial", "final"] = "partial"
    barrier: str
    completed_coordinate: ProgressCoordinate
    completed_training_batches: int | None = None
    completed_coordinate_semantics: str = (
        "Checkpoint/barrier coordinate for custody ordering; not the primary "
        "training-batch progress field."
    )
    consistency_predicate: ConsistencyPredicateSpec
    run_contract_binding: RunContractBinding
    slots: list[CheckpointSlotBlobRef]
    content_integrity_digest: ContentIntegrityDigest
    history_availability: dict[str, bool] = Field(default_factory=dict)
    parent_lineage: list[CheckpointLineageRef] = Field(default_factory=list)
    source_training_run: ParentRef | ArtifactRef | None = None
    fork_provenance: CheckpointForkProvenance | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_schema_identity(self) -> "CheckpointTransactionManifest":
        if self.schema_id != TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_ID:
            raise ValueError(f"unsupported checkpoint transaction schema_id {self.schema_id!r}")
        if self.schema_version != TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported checkpoint transaction schema_version {self.schema_version!r}"
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
    completed_training_batches: int | None = None
    completed_coordinate_semantics: str = (
        "Checkpoint/barrier coordinate for custody ordering; not the primary "
        "training-batch progress field."
    )


class CheckpointResumeResult(StrictModel):
    """Result of a resume load and validation pass."""

    manifest: CheckpointTransactionManifest
    slots: dict[str, Any]
    provenance_notices: list[CheckpointProvenanceNotice] = Field(default_factory=list)
    new_lineage_required: bool = False
    previous_transaction_id: str | None = None
