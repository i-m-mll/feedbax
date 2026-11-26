"""Durable manifest and artifact-reference models for Feedbax runs.

The database remains useful as an index, but these models are the portable
records that describe specs, executions, lineage, and large output artifacts.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from feedbax.contracts.artifact_store import (
    _validate_retention_artifact_ref_metadata,
)
from feedbax.contracts.loaded_model import ImportedModelProvenance
from feedbax.contracts.model_address import authored_model_field_target
from feedbax.contracts.model_origin import ModelOrigin
from feedbax.contracts.base import (
    AUTHENTICATED_MANIFEST_REF_SCHEMA_ID as AUTHENTICATED_MANIFEST_REF_SCHEMA_ID,
)
from feedbax.contracts.base import (
    AUTHENTICATED_MANIFEST_REF_SCHEMA_VERSION as AUTHENTICATED_MANIFEST_REF_SCHEMA_VERSION,
)
from feedbax.contracts.base import (
    ArrayStoreRef,
    ArtifactMigrationRecord,
    ArtifactRef,
    ArtifactValidationRecord,
    ParentRef,
    Provenance,
    StrictModel,
    canonical_json_bytes,
    default_manifest_root,
    feedbax_version,
    sha256_bytes,
    utc_now,
)
from feedbax.contracts.base import (
    authenticated_manifest_ref_metadata as authenticated_manifest_ref_metadata,
)
from feedbax.contracts.strict_json import strict_json_loads

SCHEMA_VERSION = "feedbax.manifest.v1"
PROVIDER_VERSION = "feedbax-provider.v1"
TRAINING_RUN_MATRIX_DELTA_SPEC_SCHEMA_ID = "feedbax.spec.training_run_matrix_delta"
TRAINING_RUN_MATRIX_DELTA_SPEC_SCHEMA_VERSION = "feedbax.spec.training_run_matrix_delta.v1"
EVALUATION_RUN_MANIFEST_SCHEMA_VERSION = "feedbax.manifest.evaluation_run.v2"
TRAINING_RUN_CERTIFICATION_SCHEMA_ID = "feedbax.manifest.training_run_certification"
TRAINING_RUN_CERTIFICATION_SCHEMA_VERSION = "feedbax.manifest.training_run_certification.v1"
TRAINING_RUN_CERTIFICATION_REJECTED_VERSIONS = ("feedbax.manifest.training_run_certification.v0",)
ManifestStatus = Literal["pending", "running", "completed", "failed", "cancelled", "stale"]


class SpecPayload(StrictModel):
    """Inline spec payload plus optional stable reference metadata."""

    kind: str
    inline: dict[str, Any]
    schema_id: Optional[str] = None
    schema_version: Optional[str] = None
    ref: Optional[str] = None
    sha256: Optional[str] = None
    source_sha256: Optional[str] = None
    migration_records: list[ArtifactMigrationRecord] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class OverridePatch(StrictModel):
    """Machine-readable override applied relative to a base spec."""

    path: str
    value: Any = None
    op: Literal["add", "replace", "remove"] = "replace"

    @model_validator(mode="after")
    def _validate_patch(self) -> "OverridePatch":
        if not self.path.strip():
            raise ValueError("OverridePatch path must not be empty")
        if any(not part for part in self.path.split(".")) and authored_model_field_target(self.path) is None:
            raise ValueError(f"OverridePatch path is not dotted-path-like: {self.path!r}")
        has_value = "value" in self.model_fields_set
        if self.op == "remove":
            if has_value:
                raise ValueError("OverridePatch remove operation must not carry value")
            return self
        if not has_value:
            raise ValueError(f"OverridePatch {self.op} operation requires value")
        return self


class TrainingSweepAxisVariation(StrictModel):
    """Authored enumerable variation for one training run-set axis."""

    kind: Literal["explicit", "linspace", "logspace", "sampler"] = "explicit"
    values: list[Any] = Field(default_factory=list)
    min: Optional[float] = None
    max: Optional[float] = None
    n: Optional[int] = None
    sampler: Optional[str] = None
    seed: Optional[int] = None
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_variation(self) -> "TrainingSweepAxisVariation":
        if self.kind == "explicit":
            if not self.values:
                raise ValueError("explicit sweep variation requires at least one value")
            return self
        if self.kind in {"linspace", "logspace"}:
            if self.min is None or self.max is None or self.n is None:
                raise ValueError(f"{self.kind} sweep variation requires min, max, and n")
            if self.n <= 0:
                raise ValueError(f"{self.kind} sweep variation n must be positive")
            if self.kind == "logspace" and (self.min <= 0 or self.max <= 0):
                raise ValueError("logspace sweep variation min and max must be positive")
            return self
        if self.sampler is None or self.n is None:
            raise ValueError("sampler sweep variation requires sampler and n")
        if self.n <= 0:
            raise ValueError("sampler sweep variation n must be positive")
        return self


class TrainingSweepAxis(StrictModel):
    """One authored run-set axis and its durable expanded values."""

    id: str
    path: str
    variation: TrainingSweepAxisVariation
    role: Literal["authored_sweep"] = "authored_sweep"
    label: Optional[str] = None
    authored_parameter: dict[str, Any] | None = None
    values: list[Any] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TrainingSweepAxisGroup(StrictModel):
    """Axes that combine internally before the run-set groups are crossed."""

    id: str
    axes: list[str]
    mode: Literal["cross", "zip"] = "zip"

    @model_validator(mode="after")
    def _validate_group(self) -> "TrainingSweepAxisGroup":
        if not self.axes:
            raise ValueError("sweep axis group requires at least one axis")
        return self


class TrainingSweepCombinationSpec(StrictModel):
    """How run-set axes are combined into concrete training runs."""

    mode: Literal["cross", "zip", "manual"] = "cross"
    groups: list[TrainingSweepAxisGroup] = Field(default_factory=list)
    manual_coordinates: list[dict[str, int]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TrainingRunAxisCoordinate(StrictModel):
    """Concrete coordinate for one pending run inside a run set."""

    run_id: str
    index: int
    value_indices: dict[str, int] = Field(default_factory=dict)
    values: dict[str, Any] = Field(default_factory=dict)
    label: Optional[str] = None


class TrainingRunSetAxes(StrictModel):
    """Durable axis block stored on a training run-set manifest."""

    axes: list[TrainingSweepAxis] = Field(default_factory=list)
    combination: TrainingSweepCombinationSpec = Field(default_factory=TrainingSweepCombinationSpec)
    runs: list[TrainingRunAxisCoordinate] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BaseManifest(StrictModel):
    """Common manifest fields."""

    kind: str
    schema_version: str = SCHEMA_VERSION
    id: str
    created_at: datetime = Field(default_factory=utc_now)
    feedbax_version: str = Field(default_factory=feedbax_version)
    provider_version: str = PROVIDER_VERSION
    status: Optional[ManifestStatus] = None
    provenance: Provenance = Field(default_factory=Provenance)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphSpecManifest(BaseManifest):
    kind: Literal["GraphSpecManifest"] = "GraphSpecManifest"
    graph_spec: SpecPayload
    migration_records: list[ArtifactMigrationRecord] = Field(default_factory=list)


class ModelArtifactManifest(BaseManifest):
    """Manifest binding a graph spec to role-addressed params/state stores."""

    kind: Literal["ModelArtifactManifest"] = "ModelArtifactManifest"
    graph_spec: ParentRef | SpecPayload
    parameter_store: Optional[ArrayStoreRef] = None
    state_store: Optional[ArrayStoreRef] = None
    optimizer_store: Optional[ArrayStoreRef] = None
    validation_records: list[ArtifactValidationRecord] = Field(default_factory=list)
    migration_records: list[ArtifactMigrationRecord] = Field(default_factory=list)


class TrainingRunCertification(StrictModel):
    """Why execution ended and the exact artifact prefix it certified.

    This optional versioned subrecord extends the existing training-run
    envelope without reinterpreting legacy ``status='failed'``. Consumers that
    need the split must reject an old failed manifest without this record.
    """

    schema_id: Literal["feedbax.manifest.training_run_certification"]
    schema_version: Literal["feedbax.manifest.training_run_certification.v1"]
    termination_reason: Literal["completed", "diverged", "crashed", "cancelled"]
    certified_artifacts: list[ParentRef | ArtifactRef] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_certified_artifacts(self) -> "TrainingRunCertification":
        serialized = [
            artifact.model_dump_json(exclude_none=False) for artifact in self.certified_artifacts
        ]
        if len(set(serialized)) != len(serialized):
            raise ValueError("certified_artifacts must not contain duplicate refs")
        for artifact in self.certified_artifacts:
            digest = (
                artifact.sha256
                if isinstance(artifact, ArtifactRef)
                else artifact.metadata.get("manifest_sha256")
            )
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                label = artifact.logical_name if isinstance(artifact, ArtifactRef) else artifact.id
                raise ValueError(
                    f"certified artifact {label!r} requires an exact lowercase SHA-256"
                )
        return self


class InheritedTerminalCheckpoint(StrictModel):
    """A terminal checkpoint reused by a resume that performed no work."""

    checkpoint: ParentRef


class TrainingRunManifest(BaseManifest):
    kind: Literal["TrainingRunManifest"] = "TrainingRunManifest"
    run_set_id: Optional[str] = None
    job_id: Optional[str] = None
    graph_spec: Optional[SpecPayload | ParentRef] = None
    training_spec: Optional[SpecPayload] = None
    task_spec: Optional[SpecPayload] = None
    task_binding_spec: Optional[SpecPayload] = None
    checkpoint_custody: list[ParentRef | ArtifactRef] = Field(default_factory=list)
    overrides: list[OverridePatch] = Field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    completed_batches: Optional[int] = Field(default=None, ge=0)
    stopped: bool = False
    stop_reason: Optional[str] = None
    failure_kind: Literal["nan_guard"] | None = None
    intent_hash: Optional[str] = None
    execution_hash: Optional[str] = None
    resolved_semantics_root_hash: Optional[str] = None
    input_data_identities: list[dict[str, Any]] = Field(default_factory=list)
    summary_metrics: dict[str, Any] = Field(default_factory=dict)
    terminal_certification: TrainingRunCertification | None = None
    inherited_terminal_checkpoint: InheritedTerminalCheckpoint | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @property
    def terminal_checkpoint(self) -> ParentRef | ArtifactRef | None:
        if self.inherited_terminal_checkpoint is not None:
            return self.inherited_terminal_checkpoint.checkpoint
        return self.checkpoint_custody[-1] if self.checkpoint_custody else None

    @model_validator(mode="after")
    def _validate_execution_identity(self) -> "TrainingRunManifest":
        from feedbax.contracts.spec_storage import validate_sha256

        if self.stopped and self.completed_at is None:
            raise ValueError("stopped training runs require completed_at")
        if self.failure_kind is not None and self.status != "failed":
            raise ValueError("training-run failure_kind requires status='failed'")
        if self.status == "failed" and self.failure_kind == "nan_guard" and not self.stopped:
            raise ValueError("nan_guard failures must be marked stopped")
        if self.terminal_certification is not None:
            expected_status = {
                "completed": "completed",
                "diverged": "failed",
                "crashed": "failed",
                "cancelled": "cancelled",
            }[self.terminal_certification.termination_reason]
            if self.status != expected_status:
                raise ValueError(
                    "terminal_certification termination_reason disagrees with legacy "
                    f"status: reason={self.terminal_certification.termination_reason!r}, "
                    f"status={self.status!r}, expected={expected_status!r}"
                )
        inherited = self.inherited_terminal_checkpoint
        if inherited is not None and (
            self.status != "completed"
            or self.stopped
            or self.completed_batches is None
            or self.checkpoint_custody
            or self.terminal_certification is None
            or self.terminal_certification.certified_artifacts != [inherited.checkpoint]
        ):
            raise ValueError(
                "inherited terminal custody requires a completed run, no new checkpoints, "
                "and certification of the exact inherited checkpoint"
            )
        if self.intent_hash is not None:
            validate_sha256(self.intent_hash, field_name="intent_hash")
        if self.resolved_semantics_root_hash is not None:
            validate_sha256(
                self.resolved_semantics_root_hash,
                field_name="resolved_semantics_root_hash",
            )
        if self.execution_hash is not None:
            validate_sha256(self.execution_hash, field_name="execution_hash")
        if self.execution_hash is not None and self.resolved_semantics_root_hash is None:
            raise ValueError("execution_hash requires resolved_semantics_root_hash")
        if self.resolved_semantics_root_hash is not None:
            from feedbax.contracts.spec_storage import training_run_execution_hash

            expected = training_run_execution_hash(
                self.resolved_semantics_root_hash,
                self.input_data_identities,
            )
            if self.execution_hash is not None and self.execution_hash != expected:
                raise ValueError(
                    "materializer drift: archived TrainingRunManifest execution_hash must "
                    f"never be overwritten; archived={self.execution_hash!r}, "
                    f"computed={expected!r}"
                )
            object.__setattr__(self, "execution_hash", expected)
        return self


def has_training_checkpoint_custody(payload: Mapping[str, Any]) -> bool:
    """Read checkpoint availability from an indexed training manifest payload."""
    custody = payload.get("checkpoint_custody")
    inherited = payload.get("inherited_terminal_checkpoint")
    return bool(isinstance(custody, list) and custody) or (
        isinstance(inherited, Mapping) and isinstance(inherited.get("checkpoint"), Mapping)
    )


EVALUATION_RUN_SPEC_SCHEMA_ID = "feedbax.spec.evaluation_run"
EVALUATION_RUN_SPEC_SCHEMA_VERSION = "feedbax.spec.evaluation_run.v1"


class EvaluationRunSpec(StrictModel):
    """Managed rollout specification projected into an evaluation manifest."""

    schema_id: Literal["feedbax.spec.evaluation_run"] = EVALUATION_RUN_SPEC_SCHEMA_ID
    schema_version: Literal["feedbax.spec.evaluation_run.v1"] = EVALUATION_RUN_SPEC_SCHEMA_VERSION
    evaluation_type: str
    training_run_ids: list[str] = Field(default_factory=list)
    inputs: list[ParentRef] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)


class EvaluationRunManifest(BaseManifest):
    kind: Literal["EvaluationRunManifest"] = "EvaluationRunManifest"
    schema_version: Literal["feedbax.manifest.evaluation_run.v2"] = EVALUATION_RUN_MANIFEST_SCHEMA_VERSION
    evaluation_spec: SpecPayload
    input_training_runs: list[ParentRef] = Field(default_factory=list)
    loaded_model: ImportedModelProvenance | None = Field(default=None, exclude_if=lambda value: value is None)
    model_origin: ModelOrigin | None = Field(default=None, exclude_if=lambda value: value is None)
    summary_metrics: dict[str, Any] = Field(default_factory=dict)
    migration_records: list[ArtifactMigrationRecord] = Field(default_factory=list)


class CheckpointScorerIdentity(StrictModel):
    """Stable identity for a downstream-provided checkpoint scorer."""

    scorer_id: str
    name: Optional[str] = None
    version: Optional[str] = None
    plugin: Optional[str] = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CheckpointSelectionBank(StrictModel):
    """Validation or evaluation bank used to score candidate checkpoints."""

    role: Literal["validation", "evaluation", "fixed"] = "validation"
    status: Literal["available", "missing", "unavailable"] = "available"
    bank_id: Optional[str] = None
    logical_name: Optional[str] = None
    ref: Optional[ParentRef | ArtifactRef] = None
    fallback_ref: Optional[ParentRef | ArtifactRef] = None
    fallback_reason: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_bank_status(self) -> "CheckpointSelectionBank":
        if self.status == "available" and self.ref is None:
            raise ValueError("available checkpoint-selection banks must include ref")
        if (
            self.status != "available"
            and self.fallback_ref is not None
            and not self.fallback_reason
        ):
            raise ValueError("checkpoint-selection bank fallback_ref requires fallback_reason")
        return self


class CheckpointCandidateRef(StrictModel):
    """Reference to one candidate checkpoint and its available lineage."""

    id: str
    checkpoint: ParentRef | ArtifactRef
    run_id: Optional[str] = None
    replicate_id: Optional[str] = None
    step: Optional[int] = None
    training_run: Optional[ParentRef] = None
    model_artifact: Optional[ParentRef] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CheckpointScoreSummary(StrictModel):
    """Scorer output summary for one candidate checkpoint."""

    candidate_id: str
    primary_metric: str
    primary_value: float
    objective: Literal["minimize", "maximize"]
    rank: Optional[int] = None
    metrics: dict[str, float] = Field(default_factory=dict)
    status: Literal["scored", "failed", "missing"] = "scored"
    metadata: dict[str, Any] = Field(default_factory=dict)


class CheckpointSelectionGroup(StrictModel):
    """Candidate and selected checkpoint records for a run or replicate."""

    scope: Literal["run", "replicate"] = "run"
    run_id: str
    replicate_id: Optional[str] = None
    candidate_checkpoints: list[CheckpointCandidateRef] = Field(default_factory=list)
    selected_checkpoint: Optional[CheckpointCandidateRef] = None
    score_summaries: list[CheckpointScoreSummary] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_group(self) -> "CheckpointSelectionGroup":
        if self.scope == "replicate" and not self.replicate_id:
            raise ValueError("replicate checkpoint-selection groups require replicate_id")
        if self.selected_checkpoint is not None:
            candidate_ids = {candidate.id for candidate in self.candidate_checkpoints}
            if self.selected_checkpoint.id not in candidate_ids:
                raise ValueError("selected checkpoint must also appear in candidate_checkpoints")
        return self


class CheckpointSelectionSpec(StrictModel):
    """Declarative request for generic checkpoint selection."""

    selection_type: str
    scorer: CheckpointScorerIdentity
    bank: CheckpointSelectionBank
    group_by: Literal["run", "replicate"] = "run"
    candidate_checkpoints: list[CheckpointCandidateRef] = Field(default_factory=list)
    inputs: list[ParentRef] = Field(default_factory=list)
    fallback_allowed: bool = False
    params: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CheckpointSelectionManifest(BaseManifest):
    """Durable custody record for selected checkpoints."""

    kind: Literal["CheckpointSelectionManifest"] = "CheckpointSelectionManifest"
    selection_spec: SpecPayload
    scorer: CheckpointScorerIdentity
    bank: CheckpointSelectionBank
    selection_status: Literal["selected", "fallback_selected", "failed"]
    fallback_allowed: bool = False
    failure_reason: Optional[str] = None
    inputs: list[ParentRef] = Field(default_factory=list)
    selections: list[CheckpointSelectionGroup] = Field(default_factory=list)
    summary_metrics: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_selection_status(self) -> "CheckpointSelectionManifest":
        selected_groups = [
            group for group in self.selections if group.selected_checkpoint is not None
        ]
        if self.selection_status in {"selected", "fallback_selected"} and not selected_groups:
            raise ValueError(
                "checkpoint-selection manifests with selected status require a selected checkpoint"
            )

        bank_missing = self.bank.status != "available"
        if bank_missing and self.selection_status == "selected":
            raise ValueError(
                "missing or unavailable checkpoint-selection banks cannot produce "
                "selection_status='selected'"
            )
        if self.selection_status == "fallback_selected":
            if not self.fallback_allowed:
                raise ValueError("fallback_selected requires fallback_allowed=True")
            if self.bank.status == "available":
                raise ValueError("fallback_selected requires a missing or unavailable bank")
            if not (self.failure_reason or self.bank.fallback_reason or self.bank.fallback_ref):
                raise ValueError(
                    "fallback_selected requires failure_reason, bank.fallback_reason, "
                    "or bank.fallback_ref"
                )
        if self.selection_status == "failed" and not (
            self.failure_reason or self.bank.fallback_reason
        ):
            raise ValueError("failed checkpoint-selection manifests require failure_reason")
        return self


AnyManifest = (
    GraphSpecManifest
    | ModelArtifactManifest
    | TrainingRunManifest
    | EvaluationRunManifest
    | CheckpointSelectionManifest
)


class GraphSpecLoadResult(StrictModel):
    """Migrated GraphSpec payload plus the manifest that owns its migration records."""

    payload: dict[str, Any]
    manifest: GraphSpecManifest | ModelArtifactManifest
    custody_manifest_kind: Literal["GraphSpecManifest", "ModelArtifactManifest"]
    custody_manifest_id: str
    applied_migration_records: list[ArtifactMigrationRecord] = Field(default_factory=list)
    migration_records: list[ArtifactMigrationRecord] = Field(default_factory=list)
    downstream_migration_records: list[ArtifactMigrationRecord] = Field(default_factory=list)


MANIFEST_MODELS: dict[str, type[BaseManifest]] = {
    "GraphSpecManifest": GraphSpecManifest,
    "ModelArtifactManifest": ModelArtifactManifest,
    "TrainingRunManifest": TrainingRunManifest,
    "EvaluationRunManifest": EvaluationRunManifest,
    "CheckpointSelectionManifest": CheckpointSelectionManifest,
}


def _manifest_model_for_kind(kind: str) -> type[BaseModel] | None:
    return MANIFEST_MODELS.get(kind)


SPEC_PAYLOAD_FIELDS_BY_MANIFEST_KIND: dict[str, tuple[str, ...]] = {
    "GraphSpecManifest": ("graph_spec",),
    "ModelArtifactManifest": ("graph_spec",),
    "TrainingRunManifest": (
        "graph_spec",
        "training_spec",
        "task_spec",
        "task_binding_spec",
    ),
    "EvaluationRunManifest": ("evaluation_spec",),
    "CheckpointSelectionManifest": ("selection_spec",),
}


def _spec_payload_record_metadata(
    record: ArtifactMigrationRecord,
    *,
    kind: str,
    path: str,
) -> ArtifactMigrationRecord:
    metadata = {
        **record.metadata,
        "spec_payload_kind": kind,
        "spec_payload_path": path,
    }
    return record.model_copy(update={"metadata": metadata})


def _ensure_spec_payload_hash(payload: SpecPayload, *, path: str) -> str:
    inline_sha256 = sha256_bytes(canonical_json_bytes(payload.inline))
    if payload.sha256 is not None and payload.sha256 != inline_sha256:
        raise ValueError(
            "Embedded SpecPayload sha256 does not match canonical inline payload: "
            f"path={path!r}, kind={payload.kind!r}, sha256={payload.sha256!r}, "
            f"computed_sha256={inline_sha256!r}"
        )
    return inline_sha256


def migrate_spec_payload(
    payload: SpecPayload | dict[str, Any],
    *,
    path: str = "spec",
    registry: Any | None = None,
    component_registry: Any | None = None,
    assume_current: bool = False,
) -> SpecPayload:
    """Accept or migrate one manifest-embedded structured spec payload.

    ``sha256`` is the content hash of the canonical, post-migration inline
    payload. If migration changes the inline payload, ``source_sha256`` retains
    the original inline hash for provenance.
    """
    from feedbax.contracts.migrations import (
        UnknownSpecFamily,
        UnsupportedSpecVersion,
        default_spec_registry,
        migrate_graph_spec,
        migrate_acausal_graph_spec,
    )

    spec_payload_obj = (
        payload if isinstance(payload, SpecPayload) else SpecPayload.model_validate(payload)
    )
    source_version = spec_payload_obj.schema_version
    inline_schema_version = spec_payload_obj.inline.get("schema_version")
    if (
        source_version is not None
        and isinstance(inline_schema_version, str)
        and inline_schema_version
        and inline_schema_version != source_version
    ):
        raise UnsupportedSpecVersion(
            "Embedded SpecPayload schema version disagrees with inline payload: "
            f"path={path!r}, kind={spec_payload_obj.kind!r}, "
            f"schema_version={source_version!r}, "
            f"inline_schema_version={inline_schema_version!r}"
        )
    active_registry = registry or default_spec_registry
    try:
        family = active_registry.resolve(spec_payload_obj.kind)
    except UnknownSpecFamily as exc:
        if spec_payload_obj.metadata.get("external") is True:
            if spec_payload_obj.schema_id is None or spec_payload_obj.schema_version is None:
                raise UnknownSpecFamily(
                    "External embedded SpecPayload requires schema_id and schema_version: "
                    f"path={path!r}, kind={spec_payload_obj.kind!r}"
                ) from exc
            if spec_payload_obj.schema_id.startswith("feedbax."):
                raise UnknownSpecFamily(
                    "Unknown Feedbax embedded SpecPayload family cannot be marked external: "
                    f"path={path!r}, kind={spec_payload_obj.kind!r}, "
                    f"schema_id={spec_payload_obj.schema_id!r}"
                ) from exc
            if spec_payload_obj.sha256 is None:
                source_sha256 = _ensure_spec_payload_hash(spec_payload_obj, path=path)
                return spec_payload_obj.model_copy(update={"sha256": source_sha256})
            _ensure_spec_payload_hash(spec_payload_obj, path=path)
            return spec_payload_obj
        raise UnknownSpecFamily(
            "Unknown embedded SpecPayload family: "
            f"path={path!r}, kind={spec_payload_obj.kind!r}; {exc}"
        ) from exc
    expected_schema_id = family.identity
    if spec_payload_obj.schema_id is not None and spec_payload_obj.schema_id != expected_schema_id:
        raise UnsupportedSpecVersion(
            "Unsupported embedded SpecPayload schema identity: "
            f"path={path!r}, kind={spec_payload_obj.kind!r}, "
            f"schema_id={spec_payload_obj.schema_id!r}, expected={expected_schema_id!r}"
        )

    source_sha256 = _ensure_spec_payload_hash(spec_payload_obj, path=path)
    try:
        if spec_payload_obj.kind in {"GraphSpec", "AcausalGraphSpec"}:
            migrate_graph = migrate_graph_spec if spec_payload_obj.kind == "GraphSpec" else migrate_acausal_graph_spec
            result = migrate_graph(
                spec_payload_obj.inline,
                source_version=source_version,
                path=path,
                registry=active_registry,
                component_registry=component_registry,
                resolve_components=not assume_current or spec_payload_obj.kind == "AcausalGraphSpec",
            )
        else:
            result = active_registry.migrate(
                spec_payload_obj.kind,
                spec_payload_obj.inline,
                source_version=source_version,
                assume_current=assume_current,
            )
    except UnsupportedSpecVersion as exc:
        raise UnsupportedSpecVersion(
            "Unsupported embedded SpecPayload version: "
            f"path={path!r}, kind={spec_payload_obj.kind!r}; {exc}"
        ) from exc

    migrated_sha256 = sha256_bytes(canonical_json_bytes(result.payload))
    migration_records = [
        *spec_payload_obj.migration_records,
        *(
            _spec_payload_record_metadata(record, kind=spec_payload_obj.kind, path=path)
            for record in result.migration_records
        ),
    ]
    update: dict[str, Any] = {
        "inline": result.payload,
        "schema_id": result.schema_id,
        "schema_version": result.target_version,
        "sha256": migrated_sha256,
        "migration_records": migration_records,
    }
    if source_sha256 != migrated_sha256:
        update["source_sha256"] = spec_payload_obj.source_sha256 or source_sha256
    elif spec_payload_obj.source_sha256 is not None:
        update["source_sha256"] = spec_payload_obj.source_sha256
    return spec_payload_obj.model_copy(update=update)


def spec_payload(kind: str, inline: dict[str, Any], ref: Optional[str] = None) -> SpecPayload:
    """Build a registry-stamped spec payload hashed after inline migration."""
    payload = SpecPayload(kind=kind, inline=inline, ref=ref)
    return migrate_spec_payload(payload, path=kind, assume_current=True)


#: Durable `(manifest kind) -> manifest-root subdirectory` storage layout.
#:
#: This mapping is the authoritative on-disk layout for every manifest kind
#: :func:`write_manifest` can persist. It is ordered so that deterministic
#: candidate-path enumeration is stable across processes and platforms.
MANIFEST_KIND_DIRECTORIES: dict[str, str] = {
    "GraphSpecManifest": "graph_specs",
    "ModelArtifactManifest": "model_artifacts",
    "TrainingRunManifest": "training_runs",
    "EvaluationRunManifest": "evaluation_runs",
    "CheckpointSelectionManifest": "checkpoint_selections",
}


def manifest_kind_directory(kind: str) -> str:
    """Return the manifest-root subdirectory that durably stores one manifest kind.

    Args:
        kind: A ``BaseManifest.kind`` discriminator.

    Returns:
        The single directory name beneath ``<root>/manifests`` for that kind.

    Raises:
        ValueError: If the kind has no declared durable location. Unknown kinds
            fail closed rather than inventing a storage location.
    """
    try:
        return MANIFEST_KIND_DIRECTORIES[kind]
    except KeyError as exc:
        known = ", ".join(sorted(MANIFEST_KIND_DIRECTORIES))
        raise ValueError(
            f"Manifest kind {kind!r} has no canonical storage location; known kinds: {known}"
        ) from exc


def canonical_manifest_relative_path(kind: str, manifest_id: str) -> str:
    """Return the root-relative POSIX location of one manifest's canonical bytes."""
    return f"manifests/{manifest_kind_directory(kind)}/{safe_manifest_key(manifest_id)}.json"


def canonical_manifest_path(
    kind: str,
    manifest_id: str,
    *,
    root: Path | str | None = None,
) -> Path:
    """Return the canonical `(kind, id, root)` path that :func:`write_manifest` uses.

    This derivation is authoritative. The SQLite manifest index is derived
    acceleration over it: an absent or stale index entry never changes where a
    manifest's bytes canonically live.

    Args:
        kind: A ``BaseManifest.kind`` discriminator.
        manifest_id: The deterministic manifest identifier.
        root: Manifest root; defaults to :func:`default_manifest_root`.
    """
    root_path = Path(root) if root is not None else default_manifest_root()
    directory = root_path / "manifests" / manifest_kind_directory(kind)
    return directory / f"{safe_manifest_key(manifest_id)}.json"


def canonical_manifest_candidate_paths(
    manifest_id: str,
    *,
    root: Path | str | None = None,
) -> list[Path]:
    """Return every canonical path one manifest id could occupy, in declared kind order.

    A manifest identifier does not encode its kind, so canonical resolution
    probes each declared kind directory instead of inferring a kind from the
    identifier's prefix. The order follows :data:`MANIFEST_KIND_DIRECTORIES`.
    """
    root_path = Path(root) if root is not None else default_manifest_root()
    return [
        canonical_manifest_path(kind, manifest_id, root=root_path)
        for kind in MANIFEST_KIND_DIRECTORIES
    ]


def _manifest_dir(root: Path, kind: str) -> Path:
    return root / "manifests" / manifest_kind_directory(kind)


def _safe_manifest_filename(manifest_id: str) -> str:
    return f"{safe_manifest_key(manifest_id)}.json"


def safe_manifest_key(manifest_id: str) -> str:
    """Return a filesystem-safe key derived from a manifest identifier."""
    return manifest_id.replace(":", "_").replace("/", "_")


def _is_spec_payload_data(value: Any) -> bool:
    return isinstance(value, dict) and "kind" in value and "inline" in value


def _normalize_spec_payload_field(value: Any, *, path: str, component_registry: Any | None = None) -> Any:
    if value is None:
        return None
    if isinstance(value, list):
        return [
            _normalize_spec_payload_field(item, path=f"{path}[{index}]", component_registry=component_registry)
            for index, item in enumerate(value)
        ]
    if isinstance(value, SpecPayload) or _is_spec_payload_data(value):
        return migrate_spec_payload(value, path=path, component_registry=component_registry)
    return value


def _normalize_manifest_data_spec_payloads(data: dict[str, Any], *, component_registry: Any | None = None) -> dict[str, Any]:
    raw_kind = data.get("kind")
    if not isinstance(raw_kind, str):
        return data
    kind = raw_kind
    fields = SPEC_PAYLOAD_FIELDS_BY_MANIFEST_KIND.get(kind)
    if not fields:
        return data
    normalized = dict(data)
    for field_name in fields:
        if field_name not in normalized:
            continue
        normalized[field_name] = _normalize_spec_payload_field(
            normalized[field_name],
            path=field_name,
            component_registry=component_registry,
        )
    return normalized


def normalize_manifest_spec_payloads(manifest: AnyManifest) -> AnyManifest:
    """Return a copy with embedded spec payloads registry-stamped and migrated."""
    fields = SPEC_PAYLOAD_FIELDS_BY_MANIFEST_KIND.get(manifest.kind)
    if not fields:
        if isinstance(manifest, CheckpointSelectionManifest):
            return normalize_checkpoint_selection_lineage(manifest)
        return manifest
    updates: dict[str, Any] = {}
    for field_name in fields:
        value = getattr(manifest, field_name, None)
        normalized = _normalize_spec_payload_field(value, path=field_name)
        if normalized is not value:
            updates[field_name] = normalized
    normalized_manifest = manifest.model_copy(update=updates) if updates else manifest
    if isinstance(normalized_manifest, CheckpointSelectionManifest):
        return normalize_checkpoint_selection_lineage(normalized_manifest)
    return normalized_manifest  # type: ignore[return-value]


def _parent_ref_key(parent: ParentRef) -> tuple[str, str, Optional[str]]:
    return (parent.kind, parent.id, parent.role)


def _append_unique_parent_refs(
    existing: list[ParentRef],
    added: list[ParentRef],
) -> list[ParentRef]:
    refs = list(existing)
    seen = {_parent_ref_key(parent) for parent in refs}
    for parent in added:
        key = _parent_ref_key(parent)
        if key not in seen:
            refs.append(parent)
            seen.add(key)
    return refs


def _parent_ref_from_ref(ref: ParentRef | ArtifactRef | None) -> ParentRef | None:
    return ref if isinstance(ref, ParentRef) else None


def _checkpoint_candidate_lineage_refs(candidate: CheckpointCandidateRef) -> list[ParentRef]:
    refs = [
        _parent_ref_from_ref(candidate.checkpoint),
        candidate.training_run,
        candidate.model_artifact,
    ]
    return [ref for ref in refs if ref is not None]


def checkpoint_selection_parent_refs(
    manifest: CheckpointSelectionManifest,
) -> list[ParentRef]:
    """Return manifest lineage refs discoverable from checkpoint-selection custody."""
    refs: list[ParentRef] = list(manifest.inputs)
    for bank_ref in (manifest.bank.ref, manifest.bank.fallback_ref):
        parent = _parent_ref_from_ref(bank_ref)
        if parent is not None:
            refs.append(parent)
    for group in manifest.selections:
        if group.selected_checkpoint is not None:
            refs.extend(_checkpoint_candidate_lineage_refs(group.selected_checkpoint))
    return _append_unique_parent_refs([], refs)


def normalize_checkpoint_selection_lineage(
    manifest: CheckpointSelectionManifest,
) -> CheckpointSelectionManifest:
    """Attach selected checkpoint and bank refs to provenance.parents for indexing."""
    parents = _append_unique_parent_refs(
        manifest.provenance.parents,
        checkpoint_selection_parent_refs(manifest),
    )
    if parents == manifest.provenance.parents:
        return manifest
    provenance = manifest.provenance.model_copy(update={"parents": parents})
    return manifest.model_copy(update={"provenance": provenance})


#: Fields that declare a document's schema identity rather than its meaning.
SPEC_SCHEMA_IDENTITY_FIELDS: tuple[str, ...] = ("schema_id", "schema_version")


def planned_training_run_manifest_id(
    *,
    graph_spec: dict[str, Any],
    training_spec: dict[str, Any],
    task_spec: dict[str, Any],
    task_binding_spec: dict[str, Any] | None = None,
    seed: Any | None = None,
    axis_coordinates: dict[str, Any] | None = None,
    row_provenance_identity: dict[str, Any] | None = None,
) -> str:
    """Return deterministic identity for a planned Studio training run."""
    identity = {
        "graph_spec": graph_spec,
        "training_spec": training_spec,
        "task_spec": task_spec,
        "task_binding_spec": task_binding_spec,
        "seed": seed,
        "axis_coordinates": axis_coordinates or {},
    }
    if row_provenance_identity is not None:
        identity["row_provenance_identity"] = row_provenance_identity
    digest = sha256_bytes(canonical_json_bytes(identity))
    return f"feedbax-training-run:{digest[:32]}"


def checkpoint_selection_manifest_id(spec: CheckpointSelectionSpec) -> str:
    """Return deterministic identity for a checkpoint-selection spec."""
    digest = sha256_bytes(canonical_json_bytes(spec))
    return f"feedbax-checkpoint-selection:{digest[:32]}"


def load_manifest_bytes(raw: bytes, *, component_registry: Any | None = None) -> AnyManifest:
    """Parse one known Feedbax manifest from already-authenticated raw bytes."""
    data = strict_json_loads(raw, ref="manifest document")
    data = _normalize_manifest_data_spec_payloads(data, component_registry=component_registry)
    data = _validate_retention_artifact_ref_metadata(data)
    raw_kind = data.get("kind")
    if not isinstance(raw_kind, str):
        raise ValueError(f"Unknown Feedbax manifest kind: {raw_kind!r}")
    kind = raw_kind
    model = _manifest_model_for_kind(kind)
    if model is None:
        raise ValueError(f"Unknown Feedbax manifest kind: {kind!r}")
    return model.model_validate(data)  # type: ignore[return-value]


def load_manifest(path: Path | str, *, component_registry: Any | None = None) -> AnyManifest:
    """Load a known Feedbax manifest from disk."""
    return load_manifest_bytes(Path(path).read_bytes(), component_registry=component_registry)


def load_graph_spec_from_manifest(
    manifest: GraphSpecManifest | ModelArtifactManifest | Path | str,
    *,
    root: Path | str | None = None,
    component_registry: Any | None = None,
) -> GraphSpecLoadResult:
    """Load a manifest-owned GraphSpec while preserving migration-record custody.

    Feedbax GraphSpec migration records stay on the manifest that directly owns
    the inline GraphSpec payload. Downstream legacy-conversion records already
    present on a model artifact remain distinct and are surfaced separately.
    Supplying a component registry interprets persisted component revisions;
    current-format nodes remain opaque without one.
    """
    manifest_obj, manifest_path = _load_graph_spec_manifest_source(manifest, component_registry=component_registry)
    base = (
        Path(root)
        if root is not None
        else (manifest_path.parent if manifest_path is not None else Path("."))
    )

    if isinstance(manifest_obj, GraphSpecManifest):
        return _migrate_manifest_graph_spec_payload(
            manifest_obj,
            manifest_obj.graph_spec,
            existing_records=manifest_obj.migration_records,
            custody_manifest_kind="GraphSpecManifest",
            component_registry=component_registry,
        )

    graph_spec = manifest_obj.graph_spec
    if isinstance(graph_spec, ParentRef):
        referenced = _load_parent_graph_spec_manifest(graph_spec, base=base, component_registry=component_registry)
        referenced_result = load_graph_spec_from_manifest(referenced, root=base, component_registry=component_registry)
        downstream_records = _append_unique_migration_records(
            referenced_result.downstream_migration_records,
            _downstream_migration_records(manifest_obj.migration_records),
        )
        return referenced_result.model_copy(
            update={"downstream_migration_records": downstream_records}
        )

    return _migrate_manifest_graph_spec_payload(
        manifest_obj,
        graph_spec,
        existing_records=manifest_obj.migration_records,
        custody_manifest_kind="ModelArtifactManifest",
        component_registry=component_registry,
    )


def _load_graph_spec_manifest_source(
    manifest: GraphSpecManifest | ModelArtifactManifest | Path | str,
    *, component_registry: Any | None = None,
) -> tuple[GraphSpecManifest | ModelArtifactManifest, Path | None]:
    if isinstance(manifest, GraphSpecManifest | ModelArtifactManifest):
        return manifest, None
    path = Path(manifest)
    loaded = load_manifest(path, component_registry=component_registry)
    if not isinstance(loaded, GraphSpecManifest | ModelArtifactManifest):
        raise TypeError(
            f"Expected GraphSpecManifest or ModelArtifactManifest, got {type(loaded).__name__}."
        )
    return loaded, path


def _load_parent_graph_spec_manifest(parent: ParentRef, *, base: Path, component_registry: Any | None = None) -> GraphSpecManifest:
    if parent.uri is None:
        raise ValueError(
            f"Model artifact graph_spec parent {parent.id!r} has no URI; "
            "GraphSpec migration custody is not discoverable."
        )
    path = Path(parent.uri)
    if not path.is_absolute():
        path = base / path
    loaded = load_manifest(path, component_registry=component_registry)
    if not isinstance(loaded, GraphSpecManifest):
        raise TypeError(
            f"Model artifact graph_spec parent {parent.id!r} resolved to "
            f"{type(loaded).__name__}, expected GraphSpecManifest."
        )
    return loaded


def _migrate_manifest_graph_spec_payload(
    manifest: GraphSpecManifest | ModelArtifactManifest,
    payload: SpecPayload,
    *,
    existing_records: list[ArtifactMigrationRecord],
    custody_manifest_kind: Literal["GraphSpecManifest", "ModelArtifactManifest"],
    component_registry: Any | None = None,
) -> GraphSpecLoadResult:
    if payload.kind != "GraphSpec":
        raise TypeError(f"Expected GraphSpec payload, got {payload.kind!r}.")

    migrated_payload = migrate_spec_payload(payload, path="graph_spec", component_registry=component_registry)
    applied_records = [
        record for record in migrated_payload.migration_records if record.tool == "feedbax"
    ]
    migration_records = _append_unique_migration_records(
        existing_records,
        applied_records,
    )
    updated_manifest = manifest.model_copy(
        update={
            "graph_spec": migrated_payload,
            "migration_records": migration_records,
        }
    )
    return GraphSpecLoadResult(
        payload=migrated_payload.inline,
        manifest=updated_manifest,
        custody_manifest_kind=custody_manifest_kind,
        custody_manifest_id=manifest.id,
        applied_migration_records=applied_records,
        migration_records=migration_records,
        downstream_migration_records=_downstream_migration_records(migration_records),
    )


def _append_unique_migration_records(
    existing: list[ArtifactMigrationRecord],
    added: list[ArtifactMigrationRecord],
) -> list[ArtifactMigrationRecord]:
    records = list(existing)
    seen = {_migration_record_key(record) for record in records}
    for record in added:
        key = _migration_record_key(record)
        if key not in seen:
            records.append(record)
            seen.add(key)
    return records


def _migration_record_key(record: ArtifactMigrationRecord) -> tuple[str, str, str, str, str]:
    metadata = json.dumps(record.metadata, sort_keys=True, separators=(",", ":"))
    return (
        record.tool,
        record.migration_id,
        record.source_schema_version,
        record.target_schema_version,
        metadata,
    )


def _downstream_migration_records(
    records: list[ArtifactMigrationRecord],
) -> list[ArtifactMigrationRecord]:
    return [record for record in records if record.tool != "feedbax"]


def training_run_manifest_id(job_id: Optional[str] = None) -> str:
    key = job_id or str(uuid.uuid4())
    return f"feedbax-training-run:{key}"
