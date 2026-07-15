"""Training checkpoint custody contracts.

These records describe Feedbax-owned training checkpoint transactions. They are
separate from evaluation checkpoint-selection records: this module owns resume
state, slot integrity, and run-contract binding for training writers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar

from pydantic import Field, model_validator

from feedbax.contracts.manifest import ArtifactMigrationRecord, ArtifactRef, ParentRef, StrictModel
from feedbax.contracts.worker import (
    ConsistencyPredicateSpec,
    MaterializedSlotAxisBinding,
    ProgressCoordinate,
)

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
TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION_V4 = (
    "feedbax.manifest.training_checkpoint_transaction.v4"
)
TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION_V5 = (
    "feedbax.manifest.training_checkpoint_transaction.v5"
)
TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION_V6 = (
    "feedbax.manifest.training_checkpoint_transaction.v6"
)
TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION_V7 = (
    "feedbax.manifest.training_checkpoint_transaction.v7"
)
TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_VERSION = (
    "feedbax.manifest.training_checkpoint_transaction.v8"
)
TRAINING_CHECKPOINT_LATEST_POINTER_SCHEMA_ID = "feedbax.manifest.training_checkpoint_latest_pointer"
TRAINING_CHECKPOINT_LATEST_POINTER_SCHEMA_VERSION_V2 = (
    "feedbax.manifest.training_checkpoint_latest_pointer.v2"
)
TRAINING_CHECKPOINT_LATEST_POINTER_SCHEMA_VERSION = (
    "feedbax.manifest.training_checkpoint_latest_pointer.v3"
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
CHECKPOINT_CONTINUATION_REQUEST_SCHEMA_VERSION = "feedbax.spec.training_checkpoint_continuation.v2"
CHECKPOINT_FORK_PLAN_SCHEMA_ID = "feedbax.spec.training_checkpoint_fork_plan"
CHECKPOINT_FORK_PLAN_SCHEMA_VERSION = "feedbax.spec.training_checkpoint_fork_plan.v1"


CheckpointDocumentT = TypeVar("CheckpointDocumentT")


def __getattr__(name: str) -> Any:
    if name not in {"BatchHistory", "Granularity"}:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from feedbax.contracts import checkpoint_history

    value = getattr(checkpoint_history, name)
    globals()[name] = value
    return value


@dataclass(frozen=True)
class CheckpointDocumentLoadResult(Generic[CheckpointDocumentT]):
    """One public custody document after registered migration and strict validation.

    ``source_version`` and ``migration_records`` retain the durable provenance
    needed to distinguish an already-current document from a migrated legacy
    document. The loader never alters the input file or byte payload.
    """

    document: CheckpointDocumentT
    schema_id: str
    source_version: str
    target_version: str
    migration_records: tuple[ArtifactMigrationRecord, ...]

    @property
    def migrated(self) -> bool:
        """Whether one or more registered migrations were applied."""
        return bool(self.migration_records)


class CheckpointContinuationRequest(StrictModel):
    """Durable declaration for a segment-local continuation."""

    schema_id: str = CHECKPOINT_CONTINUATION_REQUEST_SCHEMA_ID
    schema_version: str = CHECKPOINT_CONTINUATION_REQUEST_SCHEMA_VERSION
    source_completed_batches: int = Field(ge=0)
    additional_batches: int | None = Field(default=None, gt=0)
    self_contained: bool = False

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
        if self.additional_batches is None:
            raise ValueError("/additional_batches is required for segment-local continuation")
        return self

    @property
    def target_total(self) -> int:
        """Return the target total implied by this unambiguous declaration."""
        assert self.additional_batches is not None
        return self.source_completed_batches + self.additional_batches


class CheckpointSegmentLineage(StrictModel):
    """Versioned identity and offsets for one canonical checkpoint segment."""

    schema_id: str = "feedbax.manifest.training_checkpoint.segment_lineage"
    schema_version: str = "feedbax.manifest.training_checkpoint.segment_lineage.v1"
    parent_transaction_id: str | None = None
    start_batch: int = Field(ge=0)
    segment_batch_count: int = Field(ge=0)
    history_granularities: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_lineage(self) -> "CheckpointSegmentLineage":
        if self.start_batch == 0 and self.parent_transaction_id is not None:
            raise ValueError("root segment cannot declare parent_transaction_id")
        if self.start_batch > 0 and self.parent_transaction_id is None:
            raise ValueError("continuation segment requires parent_transaction_id")
        if any(interval < 1 for interval in self.history_granularities.values()):
            raise ValueError("history granularity intervals must be positive")
        return self


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
    materialized_axes: tuple[MaterializedSlotAxisBinding, ...] | None = None
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


class CheckpointForkTransformStep(StrictModel):
    """One ordered, registry-resolved transform in a checkpoint fork plan.

    ``records`` explicitly names every changed or introduced root slot. The
    runtime never infers transform scope from values, shapes, or slot names.
    """

    step_id: str = Field(min_length=1)
    stage: Literal["source_pre", "target_post"]
    records: list[CheckpointForkTransformRecord] = Field(min_length=1)
    target_only_slots: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_step(self) -> "CheckpointForkTransformStep":
        slots = [record.slot for record in self.records]
        if len(slots) != len(set(slots)):
            raise ValueError("/records checkpoint fork transform slots must be unique")
        identities = {
            (record.identity, _canonical_json(record.parameters)) for record in self.records
        }
        if len(identities) != 1:
            raise ValueError(
                "/records in one checkpoint fork transform step must share identity and parameters"
            )
        if self.stage == "source_pre":
            if len(self.records) != 1:
                raise ValueError("source_pre transform steps must name exactly one slot")
            if self.target_only_slots:
                raise ValueError("source_pre transform steps cannot declare target-only slots")
        unknown_target_only = set(self.target_only_slots) - set(slots)
        if unknown_target_only:
            raise ValueError(
                "/target_only_slots must be named by transform records; "
                f"slots={sorted(unknown_target_only)!r}"
            )
        return self


class CheckpointForkHistoryPolicy(StrictModel):
    """Explicit target history behavior for a checkpoint fork."""

    mode: Literal["preserve", "continue_segment"] = "preserve"
    continuation_request: CheckpointContinuationRequest | None = None
    segment_history_template_ref: str | None = None

    @model_validator(mode="after")
    def _validate_policy(self) -> "CheckpointForkHistoryPolicy":
        if self.mode == "preserve":
            if (
                self.continuation_request is not None
                or self.segment_history_template_ref is not None
            ):
                raise ValueError("preserve history policy cannot declare continuation inputs")
        elif self.continuation_request is None or not self.segment_history_template_ref:
            raise ValueError(
                "continue_segment history policy requires continuation_request and "
                "segment_history_template_ref"
            )
        return self


class CheckpointForkSourcePreparation(StrictModel):
    """Portable source declaration for a fork plan."""

    checkpoint_root_ref: str = Field(min_length=1)
    expected_transaction_id: str | None = None
    expected_transaction_root_sha256: str | None = None
    transforms: list[CheckpointForkTransformStep] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_source(self) -> "CheckpointForkSourcePreparation":
        if any(step.stage != "source_pre" for step in self.transforms):
            raise ValueError("source preparation only accepts source_pre transforms")
        if self.expected_transaction_root_sha256 is not None:
            value = self.expected_transaction_root_sha256
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError("expected_transaction_root_sha256 must be a lowercase sha256")
        return self


class CheckpointForkCompatibilityProjection(StrictModel):
    """Durable run-contract and slot-ABI identity for one fork target."""

    schema_id: str = "feedbax.spec.training_checkpoint_fork_compatibility"
    schema_version: str = "feedbax.spec.training_checkpoint_fork_compatibility.v1"
    run_contract_algorithm_version: str = Field(min_length=1)
    run_contract_hash_domain: str = Field(min_length=1)
    run_contract_projection_sha256: str
    slot_structural_abi_sha256: dict[str, str] = Field(min_length=1)
    population_member_ids_sha256: str | None = None

    @model_validator(mode="after")
    def _validate_projection(self) -> "CheckpointForkCompatibilityProjection":
        _validate_sha256(
            self.run_contract_projection_sha256,
            path="/run_contract_projection_sha256",
        )
        if any(not slot for slot in self.slot_structural_abi_sha256):
            raise ValueError("/slot_structural_abi_sha256 keys must be non-empty")
        for slot, value in self.slot_structural_abi_sha256.items():
            _validate_sha256(value, path=f"/slot_structural_abi_sha256/{slot}")
        if self.population_member_ids_sha256 is not None:
            _validate_sha256(
                self.population_member_ids_sha256,
                path="/population_member_ids_sha256",
            )
        return self


class CheckpointForkTarget(StrictModel):
    """One target row in a portable checkpoint fork plan."""

    target_id: str = Field(min_length=1)
    row_id: str | None = None
    checkpoint_root_ref: str = Field(min_length=1)
    run_spec_ref: str = Field(min_length=1)
    slot_template_ref: str = Field(min_length=1)
    compatibility: CheckpointForkCompatibilityProjection
    history_policy: CheckpointForkHistoryPolicy = Field(
        default_factory=CheckpointForkHistoryPolicy
    )
    transforms: list[CheckpointForkTransformStep] = Field(default_factory=list)
    barrier_mapping: CheckpointForkBarrierMapping | None = None
    target_coordinate: ProgressCoordinate | None = None
    population_member_ids_ref: str | None = None


class CheckpointForkPlan(StrictModel):
    """Versioned portable declaration for one source and several fork targets."""

    schema_id: str = CHECKPOINT_FORK_PLAN_SCHEMA_ID
    schema_version: str = CHECKPOINT_FORK_PLAN_SCHEMA_VERSION
    source: CheckpointForkSourcePreparation
    targets: list[CheckpointForkTarget] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_plan(self) -> "CheckpointForkPlan":
        if self.schema_id != CHECKPOINT_FORK_PLAN_SCHEMA_ID:
            raise ValueError(f"unsupported checkpoint fork plan schema_id {self.schema_id!r}")
        if self.schema_version != CHECKPOINT_FORK_PLAN_SCHEMA_VERSION:
            raise ValueError(
                "unsupported checkpoint fork plan schema_version "
                f"{self.schema_version!r}; migration_intentionally_absent=yes"
            )
        for label, values in (
            ("target_id", [target.target_id for target in self.targets]),
            ("checkpoint_root_ref", [target.checkpoint_root_ref for target in self.targets]),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"checkpoint fork target {label} values must be unique")
        row_ids = [target.row_id for target in self.targets if target.row_id is not None]
        if len(row_ids) != len(set(row_ids)):
            raise ValueError("checkpoint fork target row_id values must be unique")
        step_ids = [
            step.step_id
            for step in (*self.source.transforms, *(s for t in self.targets for s in t.transforms))
        ]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("checkpoint fork transform step_id values must be unique")
        return self


def _canonical_json(value: Any) -> str:
    """Return a stable JSON representation used by local model validators."""
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ValueError("checkpoint fork transform parameters must be JSON-serializable") from exc


def _validate_sha256(value: str, *, path: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{path} must be a lowercase sha256")


class CheckpointForkSlotProvenance(StrictModel):
    """Per-slot payload provenance for a forked checkpoint."""

    slot: str
    source_sha256: str | None = None
    target_sha256: str
    source_relative_path: str | None = None
    target_relative_path: str
    transfer_mode: Literal["hardlink", "copy", "serialized"]
    transform: CheckpointForkTransformRecord | None = None
    source_axes: tuple[MaterializedSlotAxisBinding, ...] | None = None
    target_axes: tuple[MaterializedSlotAxisBinding, ...] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CheckpointForkProvenance(StrictModel):
    """Provenance for a checkpoint transaction forked from an existing one."""

    schema_id: Literal["feedbax.manifest.training_checkpoint.fork_provenance"] = (
        "feedbax.manifest.training_checkpoint.fork_provenance"
    )
    schema_version: Literal["feedbax.manifest.training_checkpoint.fork_provenance.v2"] = (
        "feedbax.manifest.training_checkpoint.fork_provenance.v2"
    )
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
    segment_lineage: CheckpointSegmentLineage
    completed_coordinate_semantics: str = (
        "Cumulative phase-program coordinate for custody ordering; not the primary "
        "training-batch progress field or checkpoint count."
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
        "Cumulative phase-program coordinate for custody ordering; not the primary "
        "training-batch progress field or checkpoint count."
    )

    @model_validator(mode="after")
    def _validate_schema_identity(self) -> "CheckpointLatestPointer":
        if self.schema_id != TRAINING_CHECKPOINT_LATEST_POINTER_SCHEMA_ID:
            raise ValueError(f"unsupported checkpoint latest schema_id {self.schema_id!r}")
        if self.schema_version != TRAINING_CHECKPOINT_LATEST_POINTER_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported checkpoint latest schema_version {self.schema_version!r}"
            )
        return self


class CheckpointResumeResult(StrictModel):
    """Result of a resume load and validation pass."""

    manifest: CheckpointTransactionManifest
    slots: dict[str, Any]
    provenance_notices: list[CheckpointProvenanceNotice] = Field(default_factory=list)
    new_lineage_required: bool = False
    previous_transaction_id: str | None = None
