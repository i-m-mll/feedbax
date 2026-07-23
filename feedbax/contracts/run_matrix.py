"""Contracts for governed multi-row training run matrices."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import Field, field_validator, model_validator

from feedbax.contracts.expressions import Coalesce, ValueExpr, ValueQuery
from feedbax.contracts.extraction import SourceBinding
from feedbax.contracts.matrix_core import RowDerivation
from feedbax.contracts.manifest import (
    OverridePatch,
    StrictModel,
    TrainingSweepAxis,
    TrainingSweepCombinationSpec,
)
from feedbax.contracts.spec_storage import training_spec_sha256


TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID = "feedbax.spec.training_run_matrix"
TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION_V1 = "feedbax.spec.training_run_matrix.v1"
TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION_V2 = "feedbax.spec.training_run_matrix.v2"
TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION_V3 = "feedbax.spec.training_run_matrix.v3"
TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION_V4 = "feedbax.spec.training_run_matrix.v4"
TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION = "feedbax.spec.training_run_matrix.v5"
AUTHORED_TRAINING_ROW_SCHEMA_ID = "feedbax.spec.authored_training_row"
AUTHORED_TRAINING_ROW_SCHEMA_VERSION = f"{AUTHORED_TRAINING_ROW_SCHEMA_ID}.v1"
TRAINING_ROW_LOWERING_RESULT_SCHEMA_ID = "feedbax.spec.training_row_lowering_result"
TRAINING_ROW_LOWERING_RESULT_SCHEMA_VERSION = (
    f"{TRAINING_ROW_LOWERING_RESULT_SCHEMA_ID}.v1"
)
TRAINING_ROW_LOWERER_REF_SCHEMA_ID = "feedbax.spec.training_row_lowerer_ref"
TRAINING_ROW_LOWERER_REF_SCHEMA_VERSION = f"{TRAINING_ROW_LOWERER_REF_SCHEMA_ID}.v1"
TRAINING_ROW_LOWERER_REF_FIELD = "feedbax_row_lowerer"
TRAINING_ROW_PROVENANCE_SCHEMA_ID = "feedbax.spec.training_row_provenance"
TRAINING_ROW_PROVENANCE_SCHEMA_VERSION_V1 = f"{TRAINING_ROW_PROVENANCE_SCHEMA_ID}.v1"
TRAINING_ROW_PROVENANCE_SCHEMA_VERSION = f"{TRAINING_ROW_PROVENANCE_SCHEMA_ID}.v2"
TRAINING_ROW_PLANNING_PROVENANCE_SCHEMA_ID = (
    "feedbax.spec.training_row_planning_provenance"
)
TRAINING_ROW_PLANNING_PROVENANCE_SCHEMA_VERSION_V1 = (
    f"{TRAINING_ROW_PLANNING_PROVENANCE_SCHEMA_ID}.v1"
)
TRAINING_ROW_PLANNING_PROVENANCE_SCHEMA_VERSION = (
    f"{TRAINING_ROW_PLANNING_PROVENANCE_SCHEMA_ID}.v2"
)
RUN_MATRIX_MATERIALIZATION_SCHEMA_ID = "feedbax.manifest.run_matrix_materialization"
RUN_MATRIX_MATERIALIZATION_SCHEMA_VERSION_V1 = "feedbax.run_matrix_materialization.v1"
RUN_MATRIX_MATERIALIZATION_SCHEMA_VERSION_V2 = f"{RUN_MATRIX_MATERIALIZATION_SCHEMA_ID}.v2"
RUN_MATRIX_MATERIALIZATION_SCHEMA_VERSION = f"{RUN_MATRIX_MATERIALIZATION_SCHEMA_ID}.v3"
TRAINING_RUN_MATRIX_PREFLIGHT_BINDING_SCHEMA_ID = (
    "feedbax.orchestration.training_run_matrix_preflight_binding"
)
TRAINING_RUN_MATRIX_PREFLIGHT_BINDING_SCHEMA_VERSION = (
    f"{TRAINING_RUN_MATRIX_PREFLIGHT_BINDING_SCHEMA_ID}.v1"
)
TRAINING_RUN_MATRIX_AUTHORITY_SCHEMA_ID = "feedbax.orchestration.training_run_matrix_authority"
TRAINING_RUN_MATRIX_AUTHORITY_SCHEMA_VERSION = f"{TRAINING_RUN_MATRIX_AUTHORITY_SCHEMA_ID}.v1"
RUNPOD_PREFLIGHT_EVIDENCE_SCHEMA_ID_V1 = "feedbax.runpod_preflight_evidence"
RUNPOD_PREFLIGHT_BASE_EVIDENCE_SCHEMA_ID = (
    "feedbax.orchestration.runpod_preflight_base_evidence"
)
RUNPOD_PREFLIGHT_EVIDENCE_SCHEMA_ID = "feedbax.orchestration.runpod_preflight_evidence"
RUNPOD_PREFLIGHT_EVIDENCE_SCHEMA_VERSION_V1 = (
    f"{RUNPOD_PREFLIGHT_EVIDENCE_SCHEMA_ID_V1}.v1"
)
RUNPOD_PREFLIGHT_EVIDENCE_SCHEMA_VERSION = (
    f"{RUNPOD_PREFLIGHT_BASE_EVIDENCE_SCHEMA_ID}.v2"
)
RUNPOD_PREFLIGHT_EVIDENCE_SCHEMA_VERSION_V2 = f"{RUNPOD_PREFLIGHT_EVIDENCE_SCHEMA_ID}.v2"
RUNPOD_PREFLIGHT_EVIDENCE_SCHEMA_VERSION_V3 = f"{RUNPOD_PREFLIGHT_EVIDENCE_SCHEMA_ID}.v3"
RUNPOD_PREFLIGHT_EVIDENCE_SCHEMA_VERSION_V4 = f"{RUNPOD_PREFLIGHT_EVIDENCE_SCHEMA_ID}.v4"
_PATH_SAFE_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class TrainingRunMatrixArtifactBinding(StrictModel):
    """Portable identity of the one governed matrix shared by every row."""

    schema_id: Literal["feedbax.spec.training_run_matrix"]
    schema_version: Literal["feedbax.spec.training_run_matrix.v5"]
    artifact_id: str = Field(min_length=1)
    artifact_sha256: str
    canonical_sha256: str
    intent_hash: str

    @field_validator("artifact_sha256", "canonical_sha256", "intent_hash")
    @classmethod
    def _validate_hashes(cls, value: str) -> str:
        _validate_digest(value, "/matrix")
        return value


class TrainingRunMatrixRowPreflightBinding(StrictModel):
    """Governed and custodied identities for one ordered matrix row."""

    row_id: str = Field(min_length=1)
    planned_run_id: str = Field(min_length=1)
    authored_payload_hash: str
    lowered_execution_payload_hash: str
    runtime_payload_sha256: str
    resolved_root_hash: str
    execution_hash: str
    locked_training_depth: int = Field(gt=0)

    @field_validator(
        "authored_payload_hash",
        "lowered_execution_payload_hash",
        "runtime_payload_sha256",
        "resolved_root_hash",
        "execution_hash",
    )
    @classmethod
    def _validate_hashes(cls, value: str) -> str:
        _validate_digest(value, "/rows")
        return value

    @model_validator(mode="after")
    def _bind_lowered_payload_to_runtime(self) -> "TrainingRunMatrixRowPreflightBinding":
        if self.lowered_execution_payload_hash != self.runtime_payload_sha256:
            raise ValueError("lowered execution payload hash does not match runtime custody")
        return self


class TrainingRunMatrixInputCustodyBinding(StrictModel):
    """Root-free digest of one complete authenticated input-custody contract."""

    role: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    identifier: str = Field(min_length=1)
    identity_sha256: str
    provider_binding: str = Field(min_length=1)
    artifact_sha256: str
    custody_sha256: str

    @field_validator(
        "identity_sha256",
        "artifact_sha256",
        "custody_sha256",
    )
    @classmethod
    def _validate_hashes(cls, value: str) -> str:
        _validate_digest(value, "/resolved_inputs")
        return value

    @model_validator(mode="after")
    def _validate_binding(self) -> "TrainingRunMatrixInputCustodyBinding":
        if self.identity_sha256 != self.artifact_sha256:
            raise ValueError("input scientific identity does not match custodied artifact")
        return self


class TrainingRunMatrixCodeAuthority(StrictModel):
    """Authenticated, root-free observation of one protected checkout."""

    repo: str = Field(min_length=1)
    declared_revision: str
    protected_ref: str = Field(min_length=1)
    protected_revision: str
    observed_revision: str
    clean: Literal[True] = True

    @field_validator("declared_revision", "protected_revision", "observed_revision")
    @classmethod
    def _validate_revisions(cls, value: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{40}", value):
            raise ValueError("code authority revisions must be full lowercase Git commit ids")
        return value

    @field_validator("protected_ref")
    @classmethod
    def _validate_protected_ref(cls, value: str) -> str:
        if not value.startswith("refs/heads/"):
            raise ValueError("protected_ref must name an explicit local branch ref")
        return value

    @model_validator(mode="after")
    def _validate_authority(self) -> "TrainingRunMatrixCodeAuthority":
        if len({self.declared_revision, self.protected_revision, self.observed_revision}) != 1:
            raise ValueError("declared, protected-ref, and observed revisions must match")
        return self


class TrainingRunMatrixMonitorBinding(StrictModel):
    """Exact per-row event and readiness protocol monitored after launch."""

    run_event_schema_id: Literal["feedbax.run_event"] = "feedbax.run_event"
    run_event_schema_version: Literal["feedbax.run_event.v2"] = "feedbax.run_event.v2"
    ready_event_type: Literal["ready"] = "ready"
    progress_event_type: Literal["progress"] = "progress"
    heartbeat_event_type: Literal["heartbeat"] = "heartbeat"
    terminal_event_types: tuple[Literal["complete"], Literal["failed"]] = (
        "complete",
        "failed",
    )
    event_paths: list[str] = Field(min_length=1)


def _validate_matrix_authority_sets(
    rows: list[TrainingRunMatrixRowPreflightBinding],
    resolved_inputs: list[TrainingRunMatrixInputCustodyBinding],
    code_authorities: list[TrainingRunMatrixCodeAuthority],
    monitor: TrainingRunMatrixMonitorBinding,
) -> None:
    row_ids = [row.row_id for row in rows]
    if len(row_ids) != len(set(row_ids)):
        raise ValueError("matrix preflight rows must be unique")
    if monitor.event_paths != [f"events/{row_id}.events.jsonl" for row_id in row_ids]:
        raise ValueError("matrix monitor event paths must exactly follow ordered rows")
    input_keys = [(item.role, item.kind, item.identifier) for item in resolved_inputs]
    if input_keys != sorted(input_keys) or len(input_keys) != len(set(input_keys)):
        raise ValueError("matrix input custody must be complete, unique, and canonical")
    repos = [item.repo for item in code_authorities]
    if repos != sorted(repos) or len(repos) != len(set(repos)):
        raise ValueError("matrix code authorities must be unique and canonically ordered")


class TrainingRunMatrixPreflightBinding(StrictModel):
    """Provider evidence bound to authenticated matrix authority."""

    schema_id: Literal["feedbax.orchestration.training_run_matrix_preflight_binding"] = (
        TRAINING_RUN_MATRIX_PREFLIGHT_BINDING_SCHEMA_ID
    )
    schema_version: Literal["feedbax.orchestration.training_run_matrix_preflight_binding.v1"] = (
        TRAINING_RUN_MATRIX_PREFLIGHT_BINDING_SCHEMA_VERSION
    )
    matrix: TrainingRunMatrixArtifactBinding
    rows: list[TrainingRunMatrixRowPreflightBinding] = Field(min_length=1)
    resolved_inputs: list[TrainingRunMatrixInputCustodyBinding]
    code_authorities: list[TrainingRunMatrixCodeAuthority] = Field(min_length=1)
    monitor: TrainingRunMatrixMonitorBinding
    bundle_sha256: str
    nested_preflight_evidence_sha256: str

    @field_validator("bundle_sha256", "nested_preflight_evidence_sha256")
    @classmethod
    def _validate_hashes(cls, value: str) -> str:
        _validate_digest(value, "/preflight_binding")
        return value

    @model_validator(mode="after")
    def _validate_complete_ordered_sets(self) -> "TrainingRunMatrixPreflightBinding":
        _validate_matrix_authority_sets(
            self.rows, self.resolved_inputs, self.code_authorities, self.monitor
        )
        return self


class TrainingRunMatrixAuthority(StrictModel):
    """Durable provider-unverified authority for one governed matrix."""

    schema_id: Literal["feedbax.orchestration.training_run_matrix_authority"] = (
        TRAINING_RUN_MATRIX_AUTHORITY_SCHEMA_ID
    )
    schema_version: Literal["feedbax.orchestration.training_run_matrix_authority.v1"] = (
        TRAINING_RUN_MATRIX_AUTHORITY_SCHEMA_VERSION
    )
    authority_state: Literal["provider_unverified"] = "provider_unverified"
    matrix: TrainingRunMatrixArtifactBinding
    rows: list[TrainingRunMatrixRowPreflightBinding] = Field(min_length=1)
    resolved_inputs: list[TrainingRunMatrixInputCustodyBinding]
    code_authorities: list[TrainingRunMatrixCodeAuthority] = Field(min_length=1)
    monitor: TrainingRunMatrixMonitorBinding
    bundle_sha256: str

    @field_validator("bundle_sha256")
    @classmethod
    def _validate_hash(cls, value: str) -> str:
        _validate_digest(value, "/matrix_authority")
        return value

    @model_validator(mode="after")
    def _validate_complete_ordered_sets(self) -> "TrainingRunMatrixAuthority":
        _validate_matrix_authority_sets(
            self.rows, self.resolved_inputs, self.code_authorities, self.monitor
        )
        return self


def training_run_matrix_authority_sha256(authority: TrainingRunMatrixAuthority) -> str:
    """Return the canonical digest of provider-unverified matrix authority."""
    return training_spec_sha256(authority.model_dump(mode="json", exclude_none=True))


def training_run_matrix_preflight_binding_sha256(
    binding: TrainingRunMatrixPreflightBinding,
) -> str:
    """Return the canonical digest of one matrix preflight binding."""
    return training_spec_sha256(binding.model_dump(mode="json", exclude_none=True))


class RowLowererIdentity(StrictModel):
    """Stable identity of the implementation that lowered one authored row."""

    lowerer_id: str = Field(min_length=1)
    lowerer_version: str = Field(min_length=1)


class TrainingRowLowererRef(StrictModel):
    """Content-pinned authority selecting a public authored-row lowerer."""

    schema_id: Literal["feedbax.spec.training_row_lowerer_ref"] = (
        TRAINING_ROW_LOWERER_REF_SCHEMA_ID
    )
    schema_version: Literal["feedbax.spec.training_row_lowerer_ref.v1"] = (
        TRAINING_ROW_LOWERER_REF_SCHEMA_VERSION
    )
    lowerer_id: str = Field(min_length=1)
    lowerer_version: str = Field(min_length=1)
    implementation_sha256: str

    @field_validator("implementation_sha256")
    @classmethod
    def _validate_implementation_sha256(cls, value: str) -> str:
        _validate_digest(value, f"/{TRAINING_ROW_LOWERER_REF_FIELD}")
        return value


class AuthoredTrainingRow(StrictModel):
    """Axis-patched authored row supplied to a registered row lowerer."""

    schema_id: Literal["feedbax.spec.authored_training_row"] = (
        AUTHORED_TRAINING_ROW_SCHEMA_ID
    )
    schema_version: Literal["feedbax.spec.authored_training_row.v1"] = (
        AUTHORED_TRAINING_ROW_SCHEMA_VERSION
    )
    row_id: str = Field(min_length=1)
    row_index: int = Field(ge=0)
    payload: dict[str, Any]
    payload_hash: str
    seed: int | None = None
    axis_coordinates: dict[str, Any]
    overrides: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("payload_hash")
    @classmethod
    def _validate_payload_hash(cls, value: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("payload_hash must be a lowercase sha256 digest")
        return value


class TrainingRowLoweringResult(StrictModel):
    """Authoritative execution payload returned by one declared row lowerer."""

    schema_id: Literal["feedbax.spec.training_row_lowering_result"] = (
        TRAINING_ROW_LOWERING_RESULT_SCHEMA_ID
    )
    schema_version: Literal["feedbax.spec.training_row_lowering_result.v1"] = (
        TRAINING_ROW_LOWERING_RESULT_SCHEMA_VERSION
    )
    execution_payload: dict[str, Any]
    lowerer_identities: list[RowLowererIdentity] = Field(min_length=1)


class TrainingRowPlanningProvenance(StrictModel):
    """Authored and lowerer identity bound into deterministic planned-run IDs."""

    schema_id: Literal["feedbax.spec.training_row_planning_provenance"] = (
        TRAINING_ROW_PLANNING_PROVENANCE_SCHEMA_ID
    )
    schema_version: Literal["feedbax.spec.training_row_planning_provenance.v2"] = (
        TRAINING_ROW_PLANNING_PROVENANCE_SCHEMA_VERSION
    )
    authored_payload_hash: str
    lowered_execution_payload_hash: str
    lowerer_identities: list[RowLowererIdentity] = Field(default_factory=list)

    @field_validator("authored_payload_hash", "lowered_execution_payload_hash")
    @classmethod
    def _validate_payload_hash(cls, value: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("payload hashes must be lowercase sha256 digests")
        return value


class TrainingRowProvenance(StrictModel):
    """Canonical authored-to-execution provenance for one materialized row."""

    schema_id: Literal["feedbax.spec.training_row_provenance"] = (
        TRAINING_ROW_PROVENANCE_SCHEMA_ID
    )
    schema_version: Literal["feedbax.spec.training_row_provenance.v2"] = (
        TRAINING_ROW_PROVENANCE_SCHEMA_VERSION
    )
    row_id: str = Field(min_length=1)
    row_index: int = Field(ge=0)
    planned_run_id: str = Field(min_length=1)
    authored_payload_hash: str
    lowered_execution_payload_hash: str
    seed: int | None = None
    axis_coordinates: dict[str, Any]
    overrides: list[dict[str, Any]] = Field(default_factory=list)
    lowerer_identities: list[RowLowererIdentity] = Field(default_factory=list)

    @field_validator("authored_payload_hash", "lowered_execution_payload_hash")
    @classmethod
    def _validate_payload_hash(cls, value: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("payload hashes must be lowercase sha256 digests")
        return value


class InlineMatrixBaseSpec(StrictModel):
    """Inline base reserved for tests, fixtures, and explicit emission opt-in."""

    kind: Literal["inline"] = "inline"
    inline: dict[str, Any]


class AuthoredIntentMatrixBaseSpec(StrictModel):
    """Canonical-content-pinned reference to an authored matrix envelope."""

    kind: Literal["authored_intent"] = "authored_intent"
    ref: str
    content_hash: str
    pin_algorithm: Literal["canonical_json_v1", "legacy_raw_sha256"] = "canonical_json_v1"
    payload_path: str | None = None
    symbolic_name: str | None = None

    @model_validator(mode="after")
    def _validate_ref(self) -> "AuthoredIntentMatrixBaseSpec":
        _validate_base_reference(self.ref, self.content_hash, self.payload_path)
        return self


class ResolvedOutputMatrixBaseSpec(StrictModel):
    """Resolved-root-pinned reference to immutable layer-2 semantics."""

    kind: Literal["resolved_output"] = "resolved_output"
    ref: str
    resolved_root_hash: str
    payload_path: str | None = None
    symbolic_name: str | None = None

    @model_validator(mode="after")
    def _validate_ref(self) -> "ResolvedOutputMatrixBaseSpec":
        _validate_base_reference(self.ref, self.resolved_root_hash, self.payload_path)
        return self


MatrixBaseSpec: TypeAlias = Annotated[
    InlineMatrixBaseSpec | AuthoredIntentMatrixBaseSpec | ResolvedOutputMatrixBaseSpec,
    Field(discriminator="kind"),
]


def _validate_base_reference(ref: str, digest: str, payload_path: str | None) -> None:
    if Path(ref).is_absolute():
        raise ValueError("/base/ref must be repo-relative")
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("/base content identity must be a lowercase sha256 digest")
    if payload_path is not None:
        _validate_dotted_path(payload_path, "/base/payload_path")


MatrixDerivation = RowDerivation


class MatrixRow(StrictModel):
    """One explicit named row in a run matrix."""

    row_id: str
    label: str | None = None
    overrides: list[OverridePatch] = Field(default_factory=list)
    seed: int | None = None
    notes: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_row(self) -> "MatrixRow":
        if not _PATH_SAFE_RE.match(self.row_id):
            raise ValueError(f"/rows/row_id is not path-safe: {self.row_id!r}")
        return self


class MatrixCompositionDelta(StrictModel):
    """One ordered authored layer with explicit ancestor-override acknowledgement."""

    layer_id: str
    patches: list[OverridePatch] = Field(default_factory=list)
    acknowledges_ancestor_paths: list[str] = Field(default_factory=list)
    schema_id: str | None = None
    schema_version: str | None = None

    @model_validator(mode="after")
    def _validate_delta(self) -> "MatrixCompositionDelta":
        if not _PATH_SAFE_RE.match(self.layer_id):
            raise ValueError(f"/deltas/layer_id is not path-safe: {self.layer_id!r}")
        if (self.schema_id is None) != (self.schema_version is None):
            raise ValueError(
                "/deltas schema_id and schema_version must be declared together at a boundary"
            )
        for path in self.acknowledges_ancestor_paths:
            _validate_dotted_path(path, "/deltas/acknowledges_ancestor_paths")
        return self


class DurableSlotTransform(StrictModel):
    transform_id: str
    version: str
    implementation_sha256: str
    stage: Literal["source_pre", "target_post"]
    target_row_id: str | None = None
    slot: str
    parameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("implementation_sha256")
    @classmethod
    def _implementation_hash(cls, value: str) -> str:
        _validate_digest(value, "/dependencies/implementation_sha256")
        return value

    @model_validator(mode="after")
    def _placement(self) -> "DurableSlotTransform":
        if self.stage == "target_post" and self.target_row_id is None:
            raise ValueError("target_post durable transforms require target_row_id")
        return self


class ForkFromSelectedCheckpoint(StrictModel):
    kind: Literal["fork_from_selected_checkpoint"] = "fork_from_selected_checkpoint"
    source_execution_hash: str
    source_row_id: str
    checkpoint_transaction_id: str
    checkpoint_root_hash: str
    slot_transforms: list[DurableSlotTransform] = Field(default_factory=list)

    @model_validator(mode="after")
    def _hashes(self) -> "ForkFromSelectedCheckpoint":
        _validate_digest(self.source_execution_hash, "/dependencies/source_execution_hash")
        _validate_digest(self.checkpoint_root_hash, "/dependencies/checkpoint_root_hash")
        return self


class ContinuationReconciliation(StrictModel):
    kind: Literal["continuation_reconciliation"] = "continuation_reconciliation"
    source_completed_batches: int = Field(ge=0)
    additional_batches: int = Field(gt=0)
    expected_target_total: int = Field(gt=0)

    @model_validator(mode="after")
    def _total(self) -> "ContinuationReconciliation":
        total = self.source_completed_batches + self.additional_batches
        if total != self.expected_target_total:
            raise ValueError(f"/dependencies target total drift: computed={total}")
        return self


class LineageGraftDependency(StrictModel):
    kind: Literal["lineage_graft"] = "lineage_graft"
    lineage_event_hash: str
    interpretation: Literal["supersedes_for_interpretation", "new_execution"]

    @model_validator(mode="after")
    def _hash(self) -> "LineageGraftDependency":
        _validate_digest(self.lineage_event_hash, "/dependencies/lineage_event_hash")
        return self


class StoppedRowStatus(StrictModel):
    kind: Literal["stopped_row"] = "stopped_row"
    row_id: str
    completed_batches: int = Field(ge=0)
    reason: str
    checkpoint_root_hash: str | None = None

    @model_validator(mode="after")
    def _hash(self) -> "StoppedRowStatus":
        if self.checkpoint_root_hash is not None:
            _validate_digest(self.checkpoint_root_hash, "/dependencies/checkpoint_root_hash")
        return self


class TaskIdentityGate(StrictModel):
    kind: Literal["task_identity_gate"] = "task_identity_gate"
    identity_kind: Literal["training_run_spec", "method_payload", "task", "dataset"]
    expected_identity_hash: str

    @model_validator(mode="after")
    def _hash(self) -> "TaskIdentityGate":
        _validate_digest(self.expected_identity_hash, "/dependencies/expected_identity_hash")
        return self


ExecutionDependency: TypeAlias = Annotated[
    ForkFromSelectedCheckpoint
    | ContinuationReconciliation
    | LineageGraftDependency
    | StoppedRowStatus
    | TaskIdentityGate,
    Field(discriminator="kind"),
]


class MatrixForkSpec(StrictModel):
    """Fork-from-source-checkpoint launch semantics for a matrix."""

    lr_continuation: Literal["continue", "restart"]
    parity: Literal["require", "skip"] = "require"
    expected_slots: list[str] = Field(default_factory=list)


class TrainingRunMatrixSpec(StrictModel):
    """Durable governed authoring contract for a matrix of training runs."""

    schema_id: str = TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID
    schema_version: str = TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION
    name: str
    issue: str | None = None
    base: MatrixBaseSpec
    deltas: list[MatrixCompositionDelta] = Field(default_factory=list)
    execution_dependencies: list[ExecutionDependency] = Field(default_factory=list)
    sources: list[SourceBinding] = Field(default_factory=list)
    derivations: list[MatrixDerivation] = Field(default_factory=list)
    rows: list[MatrixRow] = Field(default_factory=list)
    axes: list[TrainingSweepAxis] = Field(default_factory=list)
    combination: TrainingSweepCombinationSpec = Field(default_factory=TrainingSweepCombinationSpec)
    fork: MatrixForkSpec | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_spec(self) -> "TrainingRunMatrixSpec":
        if self.schema_id != TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID:
            raise ValueError(
                f"/schema_id unsupported TrainingRunMatrixSpec schema_id {self.schema_id!r}; "
                f"expected {TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID!r}"
            )
        if self.schema_version != TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION:
            raise ValueError(
                "/schema_version unsupported TrainingRunMatrixSpec schema_version "
                f"{self.schema_version!r}; expected {TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION!r}"
            )
        if not self.name.strip():
            raise ValueError("/name must not be empty")
        if bool(self.rows) == bool(self.axes):
            raise ValueError("/rows and /axes are mutually exclusive and one is required")
        row_ids = [row.row_id for row in self.rows]
        if len(set(row_ids)) != len(row_ids):
            raise ValueError("/rows row_id values must be unique")
        axis_ids = [axis.id for axis in self.axes]
        if len(set(axis_ids)) != len(axis_ids):
            raise ValueError("/axes axis ids must be unique")
        for axis in self.axes:
            if axis.path not in {"seed", "master_prng_key", "prng_key"}:
                _validate_dotted_path(axis.path, f"/axes/{axis.id}/path")
        output_paths = [derivation.output_path for derivation in self.derivations]
        if len(set(output_paths)) != len(output_paths):
            raise ValueError("/derivations output_path values must not collide")
        aliases = [source.alias for source in self.sources]
        if len(set(aliases)) != len(aliases):
            raise ValueError("/sources aliases must be unique")
        if self.derivations and not self.sources:
            for derivation in self.derivations:
                if not (
                    _query_can_evaluate_without_sources(derivation.query)
                    or _query_uses_only_row_context(derivation.query)
                ):
                    raise ValueError("/derivations require /sources unless all queries have defaults")
        return self


def apply_override_patches(
    payload: dict[str, Any],
    patches: list[OverridePatch | dict[str, Any]],
) -> dict[str, Any]:
    """Apply ``OverridePatch`` records to a deep copy of ``payload``."""
    result = deepcopy(payload)
    for raw_patch in patches:
        patch = (
            raw_patch
            if isinstance(raw_patch, OverridePatch)
            else OverridePatch.model_validate(raw_patch)
        )
        _apply_patch(result, patch)
    return result


def apply_composition_deltas(
    payload: dict[str, Any],
    deltas: list[MatrixCompositionDelta],
    *,
    ancestor_written_paths: set[str] | None = None,
) -> tuple[dict[str, Any], dict[str, str], set[str]]:
    """Apply ordered layers and fail closed on unacknowledged ancestor overrides."""
    result = deepcopy(payload)
    written = set(ancestor_written_paths or ())
    attribution: dict[str, str] = {}
    current_schema = _payload_schema_identity(result)
    for delta in deltas:
        acknowledged = set(delta.acknowledges_ancestor_paths)
        declared_boundary = (delta.schema_id, delta.schema_version)
        prior_identities = _schema_identities(result)
        for patch in delta.patches:
            if patch.path in written and patch.path not in acknowledged:
                raise ValueError(
                    f"/deltas/{delta.layer_id}/{patch.path} overrides an ancestor-written "
                    "path without explicit acknowledgement"
                )
            try:
                _apply_patch(result, patch)
            except ValueError as error:
                raise ValueError(f"/deltas/{delta.layer_id}: {error}") from error
            written.add(patch.path)
            attribution[patch.path] = delta.layer_id
        resulting_schema = _payload_schema_identity(result)
        resulting_identities = _schema_identities(result)
        if declared_boundary != (None, None):
            if declared_boundary == current_schema:
                raise ValueError(
                    f"/deltas/{delta.layer_id} declares schema boundary "
                    f"{declared_boundary!r} but does not change the active identity"
                )
            if resulting_schema != declared_boundary:
                raise ValueError(
                    f"/deltas/{delta.layer_id} declares schema boundary {declared_boundary!r} "
                    f"but flattened payload has identity {resulting_schema!r}"
                )
        elif resulting_identities != prior_identities:
            raise ValueError(
                f"/deltas/{delta.layer_id} changes schema identity from "
                f"{prior_identities!r} to {resulting_identities!r} without a declared "
                "schema_id/schema_version boundary"
            )
        current_schema = resulting_schema
    return result, attribution, written


def _payload_schema_identity(payload: dict[str, Any]) -> tuple[Any, Any]:
    return payload.get("schema_id"), payload.get("schema_version")


def _schema_identities(value: Any, path: str = "") -> dict[str, tuple[Any, Any]]:
    identities: dict[str, tuple[Any, Any]] = {}
    if isinstance(value, dict):
        if "schema_id" in value or "schema_version" in value:
            identities[path or "/"] = (value.get("schema_id"), value.get("schema_version"))
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            identities.update(_schema_identities(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_path = f"{path}.{index}" if path else str(index)
            identities.update(_schema_identities(child, child_path))
    return identities


def _validate_digest(value: str, path: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{path} must be a lowercase sha256 digest")


def _apply_patch(root: dict[str, Any], patch: OverridePatch) -> None:
    parts = patch.path.split(".")
    parent = _resolve_parent(root, parts, path=patch.path)
    leaf = parts[-1]
    if patch.op == "add" and isinstance(parent, list) and _is_list_append_token(parent, leaf):
        parent.append(deepcopy(patch.value))
        return
    exists = _contains_key(parent, leaf)
    if patch.op == "add":
        if exists:
            raise ValueError(f"override add patch path already exists: {patch.path!r}")
        _set_child(parent, leaf, deepcopy(patch.value), path=patch.path)
        return
    if patch.op == "replace":
        if not exists:
            raise ValueError(
                f"override replace patch targets a missing key/index: {patch.path!r}"
            )
        _set_child(parent, leaf, deepcopy(patch.value), path=patch.path)
        return
    if not exists:
        raise ValueError(
            f"override remove patch targets a missing key/index: {patch.path!r}"
        )
    _remove_child(parent, leaf, path=patch.path)


def _is_list_append_token(parent: list[Any], leaf: str) -> bool:
    """Return whether ``leaf`` names the JSON-Patch list-append position.

    Follows JSON-Patch ``add`` semantics (RFC 6902 section 4.1): the literal
    ``-`` token, or a numeric index exactly equal to the list's current
    length, both mean "insert after the last element". Any other index —
    in range or beyond it — targets a specific position and is handled by
    the existing add/replace/remove-at-index path, which still rejects
    beyond-range indices.
    """
    if leaf == "-":
        return True
    return leaf.isdigit() and int(leaf) == len(parent)


def _resolve_parent(root: Any, parts: list[str], *, path: str) -> Any:
    if not parts:
        raise ValueError(f"patch path must be non-empty: {path!r}")
    current = root
    for part in parts[:-1]:
        if isinstance(current, dict) and part in current:
            current = current[part]
            continue
        if isinstance(current, list) and part.isdigit():
            index = int(part)
            if 0 <= index < len(current):
                current = current[index]
                continue
        raise ValueError(f"patch path cannot traverse missing segment {part!r}: {path!r}")
    return current


def _contains_key(parent: Any, key: str) -> bool:
    if isinstance(parent, dict):
        return key in parent
    if isinstance(parent, list) and key.isdigit():
        index = int(key)
        return 0 <= index < len(parent)
    return False


def _set_child(parent: Any, key: str, value: Any, *, path: str) -> None:
    if isinstance(parent, dict):
        parent[key] = value
        return
    if isinstance(parent, list) and key.isdigit():
        index = int(key)
        if 0 <= index < len(parent):
            parent[index] = value
            return
    raise ValueError(f"patch path cannot set segment {key!r}: {path!r}")


def _remove_child(parent: Any, key: str, *, path: str) -> None:
    if isinstance(parent, dict):
        del parent[key]
        return
    if isinstance(parent, list) and key.isdigit():
        index = int(key)
        if 0 <= index < len(parent):
            del parent[index]
            return
    raise ValueError(f"patch path cannot remove segment {key!r}: {path!r}")


def _validate_dotted_path(path: str, field_path: str) -> None:
    if not path.strip() or any(not part for part in path.split(".")):
        raise ValueError(f"{field_path} is not dotted-path-like: {path!r}")


def _query_can_evaluate_without_sources(query: ValueExpr) -> bool:
    if isinstance(query, ValueQuery):
        return "default" in query.model_fields_set
    if isinstance(query, Coalesce):
        return query.default is not None or all(
            "default" in child.model_fields_set for child in query.queries
        )
    return False


def _query_uses_only_row_context(query: ValueExpr) -> bool:
    """Return whether a derivation can evaluate from its evolving row payload."""
    if isinstance(query, ValueQuery):
        return query.item == "row"
    if isinstance(query, Coalesce):
        return all(child.item == "row" for child in query.queries)
    return False
