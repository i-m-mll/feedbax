"""Contracts for governed multi-row training run matrices."""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import re
from typing import Annotated, Any, Literal, Mapping, TypeAlias

from pydantic import Field, field_validator, model_validator

from feedbax.contracts.expressions import Coalesce, MapObjectList, ValueExpr, ValueQuery
from feedbax.contracts.extraction import SourceBinding
from feedbax.contracts.matrix_core import (
    RowDerivation,
    _apply_patch,
    apply_override_patches as apply_override_patches,
)
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
TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION_V5 = "feedbax.spec.training_run_matrix.v5"
TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION = "feedbax.spec.training_run_matrix.v6"
AUTHORED_TRAINING_ROW_SCHEMA_ID = "feedbax.spec.authored_training_row"
AUTHORED_TRAINING_ROW_SCHEMA_VERSION = f"{AUTHORED_TRAINING_ROW_SCHEMA_ID}.v1"
TRAINING_ROW_LOWERING_RESULT_SCHEMA_ID = "feedbax.spec.training_row_lowering_result"
TRAINING_ROW_LOWERING_RESULT_SCHEMA_VERSION = f"{TRAINING_ROW_LOWERING_RESULT_SCHEMA_ID}.v1"
TRAINING_ROW_LOWERER_REF_SCHEMA_ID = "feedbax.spec.training_row_lowerer_ref"
TRAINING_ROW_LOWERER_REF_SCHEMA_VERSION_V1 = f"{TRAINING_ROW_LOWERER_REF_SCHEMA_ID}.v1"
TRAINING_ROW_LOWERER_REF_SCHEMA_VERSION = f"{TRAINING_ROW_LOWERER_REF_SCHEMA_ID}.v2"
TRAINING_ROW_LOWERING_CONTEXT_API_VERSION = "feedbax.training_row_lowering_context.v1"
TRAINING_ROW_LOWERER_REF_FIELD = "feedbax_row_lowerer"
TRAINING_ROW_PROVENANCE_SCHEMA_ID = "feedbax.spec.training_row_provenance"
TRAINING_ROW_PROVENANCE_SCHEMA_VERSION_V1 = f"{TRAINING_ROW_PROVENANCE_SCHEMA_ID}.v1"
TRAINING_ROW_PROVENANCE_SCHEMA_VERSION_V2 = f"{TRAINING_ROW_PROVENANCE_SCHEMA_ID}.v2"
TRAINING_ROW_PROVENANCE_SCHEMA_VERSION = f"{TRAINING_ROW_PROVENANCE_SCHEMA_ID}.v3"
TRAINING_ROW_PLANNING_PROVENANCE_SCHEMA_ID = "feedbax.spec.training_row_planning_provenance"
TRAINING_ROW_PLANNING_PROVENANCE_SCHEMA_VERSION_V1 = (
    f"{TRAINING_ROW_PLANNING_PROVENANCE_SCHEMA_ID}.v1"
)
TRAINING_ROW_PLANNING_PROVENANCE_SCHEMA_VERSION = f"{TRAINING_ROW_PLANNING_PROVENANCE_SCHEMA_ID}.v2"
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
RUNPOD_PREFLIGHT_BASE_EVIDENCE_SCHEMA_ID = "feedbax.orchestration.runpod_preflight_base_evidence"
RUNPOD_PREFLIGHT_EVIDENCE_SCHEMA_ID = "feedbax.orchestration.runpod_preflight_evidence"
RUNPOD_PREFLIGHT_EVIDENCE_SCHEMA_VERSION_V1 = f"{RUNPOD_PREFLIGHT_EVIDENCE_SCHEMA_ID_V1}.v1"
RUNPOD_PREFLIGHT_EVIDENCE_SCHEMA_VERSION = f"{RUNPOD_PREFLIGHT_BASE_EVIDENCE_SCHEMA_ID}.v2"
RUNPOD_PREFLIGHT_EVIDENCE_SCHEMA_VERSION_V2 = f"{RUNPOD_PREFLIGHT_EVIDENCE_SCHEMA_ID}.v2"
RUNPOD_PREFLIGHT_EVIDENCE_SCHEMA_VERSION_V3 = f"{RUNPOD_PREFLIGHT_EVIDENCE_SCHEMA_ID}.v3"
RUNPOD_PREFLIGHT_EVIDENCE_SCHEMA_VERSION_V4 = f"{RUNPOD_PREFLIGHT_EVIDENCE_SCHEMA_ID}.v4"
_PATH_SAFE_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class TrainingRunMatrixArtifactBinding(StrictModel):
    """Portable identity of the one governed matrix shared by every row."""

    schema_id: Literal["feedbax.spec.training_run_matrix"]
    schema_version: Literal[
        "feedbax.spec.training_run_matrix.v5",
        "feedbax.spec.training_run_matrix.v6",
    ]
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

    schema_id: Literal["feedbax.spec.training_row_lowerer_ref"] = TRAINING_ROW_LOWERER_REF_SCHEMA_ID
    schema_version: Literal["feedbax.spec.training_row_lowerer_ref.v2"] = (
        TRAINING_ROW_LOWERER_REF_SCHEMA_VERSION
    )
    context_api_version: Literal["feedbax.training_row_lowering_context.v1"] = (
        TRAINING_ROW_LOWERING_CONTEXT_API_VERSION
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

    schema_id: Literal["feedbax.spec.authored_training_row"] = AUTHORED_TRAINING_ROW_SCHEMA_ID
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


class TrainingRowParentProvenance(StrictModel):
    """Content and semantic identity of one declared compile-time parent."""

    role: str = Field(min_length=1)
    parent_kind: Literal["authored_intent", "resolved_output"]
    ref: str = Field(min_length=1)
    semantic_hash: str
    artifact_id: str = Field(min_length=1)
    artifact_sha256: str
    schema_id: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)

    @field_validator("semantic_hash", "artifact_sha256")
    @classmethod
    def _validate_hashes(cls, value: str) -> str:
        _validate_digest(value, "/parent_inputs")
        return value


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

    schema_id: Literal["feedbax.spec.training_row_provenance"] = TRAINING_ROW_PROVENANCE_SCHEMA_ID
    schema_version: Literal["feedbax.spec.training_row_provenance.v3"] = (
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
    parent_inputs: list[TrainingRowParentProvenance] = Field(default_factory=list)

    @field_validator("authored_payload_hash", "lowered_execution_payload_hash")
    @classmethod
    def _validate_payload_hash(cls, value: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("payload hashes must be lowercase sha256 digests")
        return value

    @model_validator(mode="after")
    def _canonical_parent_inputs(self) -> "TrainingRowProvenance":
        keys = [
            (item.parent_kind, item.ref, item.role, item.artifact_id) for item in self.parent_inputs
        ]
        if keys != sorted(keys) or len(keys) != len(set(keys)):
            raise ValueError("parent_inputs must be unique and canonically ordered")
        return self


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


class ResolvedOutputMatrixBaseSpecV6(ResolvedOutputMatrixBaseSpec):
    """Resolved output selected to one exact row checkpoint in matrix v6."""

    row_id: str = Field(min_length=1)
    checkpoint_transaction_id: str = Field(min_length=1)

    @field_validator("row_id")
    @classmethod
    def _validate_row_id(cls, value: str) -> str:
        if not _PATH_SAFE_RE.match(value):
            raise ValueError("/base/row_id must be path-safe")
        return value


MatrixBaseSpecV6: TypeAlias = Annotated[
    InlineMatrixBaseSpec | AuthoredIntentMatrixBaseSpec | ResolvedOutputMatrixBaseSpecV6,
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


class TargetOnlySlotAuthority(StrictModel):
    """Exact target method and slot declaration for one newly initialized slot."""

    method_ref: str = Field(min_length=1)
    slot_identity: str

    @field_validator("slot_identity")
    @classmethod
    def _slot_identity(cls, value: str) -> str:
        _validate_digest(value, "/dependencies/slot_identity")
        return value


class DurableSlotTransformV6(DurableSlotTransform):
    """Matrix-v6 durable transform with closed target-only slot authority."""

    target_only: TargetOnlySlotAuthority | None = None

    @model_validator(mode="after")
    def _v6_placement(self) -> "DurableSlotTransformV6":
        if self.stage == "source_pre" and self.target_row_id is not None:
            raise ValueError("source_pre durable transforms cannot name target_row_id")
        if self.target_only is not None and self.stage != "target_post":
            raise ValueError("target-only slot authority is legal only on target_post transforms")
        return self


class ExecutionHashForkSourceAuthority(StrictModel):
    """Exact produced execution identity for an ordinary checkpoint fork."""

    kind: Literal["execution_hash"] = "execution_hash"
    execution_hash: str

    @field_validator("execution_hash")
    @classmethod
    def _execution_hash(cls, value: str) -> str:
        _validate_digest(value, "/dependencies/source_authority/execution_hash")
        return value


class ResolvedOutputRootForkSourceAuthority(StrictModel):
    """Exact resolved-output semantic root when no execution hash exists."""

    kind: Literal["resolved_output_root"] = "resolved_output_root"
    source_run_id: str = Field(min_length=1)
    resolved_root_hash: str

    @field_validator("resolved_root_hash")
    @classmethod
    def _resolved_root_hash(cls, value: str) -> str:
        _validate_digest(value, "/dependencies/source_authority/resolved_root_hash")
        return value


ForkSourceAuthority: TypeAlias = Annotated[
    ExecutionHashForkSourceAuthority | ResolvedOutputRootForkSourceAuthority,
    Field(discriminator="kind"),
]


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


class ForkFromSelectedCheckpointV6(StrictModel):
    """Matrix-v6 selected checkpoint with one closed source authority domain."""

    kind: Literal["fork_from_selected_checkpoint"] = "fork_from_selected_checkpoint"
    source_authority: ForkSourceAuthority
    source_row_id: str
    checkpoint_transaction_id: str
    checkpoint_root_hash: str
    source_barrier: str | None = None
    slot_transforms: list[DurableSlotTransformV6] = Field(default_factory=list)

    @model_validator(mode="after")
    def _authority(self) -> "ForkFromSelectedCheckpointV6":
        _validate_digest(self.checkpoint_root_hash, "/dependencies/checkpoint_root_hash")
        if not _PATH_SAFE_RE.match(self.source_row_id):
            raise ValueError("/dependencies/source_row_id must be path-safe")
        if not self.checkpoint_transaction_id:
            raise ValueError("/dependencies/checkpoint_transaction_id must be nonempty")
        if isinstance(self.source_authority, ResolvedOutputRootForkSourceAuthority):
            if self.source_barrier is None or not self.source_barrier.strip():
                raise ValueError("resolved-output-root fork authority requires source_barrier")
        elif self.source_barrier is not None:
            raise ValueError("execution-hash fork authority does not acquire a source_barrier")
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

ExecutionDependencyV6: TypeAlias = Annotated[
    ForkFromSelectedCheckpointV6
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


class MatrixForkSpecV6(MatrixForkSpec):
    """Matrix-v6 fork policy with signed absolute LR comparison authority."""

    absolute_lr_tolerance: float | None = None

    @model_validator(mode="after")
    def _validate_tolerance(self) -> "MatrixForkSpecV6":
        tolerance = self.absolute_lr_tolerance
        if tolerance is None:
            return self
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("/fork/absolute_lr_tolerance must be finite and nonnegative")
        if self.parity != "require" or self.lr_continuation != "continue":
            raise ValueError(
                "/fork/absolute_lr_tolerance requires continue LR semantics and required parity"
            )
        return self


class _TrainingRunMatrixSpecCore(StrictModel):
    """Fields and invariants shared without reinterpretation by matrix v5 and v6."""

    schema_id: str = TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID
    schema_version: str
    name: str
    issue: str | None = None
    base: Any
    deltas: list[MatrixCompositionDelta] = Field(default_factory=list)
    execution_dependencies: list[Any] = Field(default_factory=list)
    sources: list[SourceBinding] = Field(default_factory=list)
    derivations: list[MatrixDerivation] = Field(default_factory=list)
    rows: list[MatrixRow] = Field(default_factory=list)
    axes: list[TrainingSweepAxis] = Field(default_factory=list)
    combination: TrainingSweepCombinationSpec = Field(default_factory=TrainingSweepCombinationSpec)
    fork: Any | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def _validate_common(self) -> None:
        if self.schema_id != TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID:
            raise ValueError(
                f"/schema_id unsupported TrainingRunMatrixSpec schema_id {self.schema_id!r}; "
                f"expected {TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID!r}"
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
                    raise ValueError(
                        "/derivations require /sources unless all queries have defaults"
                    )


class TrainingRunMatrixSpecV5(_TrainingRunMatrixSpecCore):
    """Exact durable matrix-v5 contract retained for parsing and runtime use."""

    schema_version: Literal["feedbax.spec.training_run_matrix.v5"] = (
        TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION_V5
    )
    base: MatrixBaseSpec
    execution_dependencies: list[ExecutionDependency] = Field(default_factory=list)
    fork: MatrixForkSpec | None = None

    @model_validator(mode="after")
    def _validate_spec(self) -> "TrainingRunMatrixSpecV5":
        self._validate_common()
        return self


class TrainingRunMatrixSpec(_TrainingRunMatrixSpecCore):
    """Current durable matrix-v6 contract with closed fork authority."""

    schema_version: Literal["feedbax.spec.training_run_matrix.v6"] = (
        TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION
    )
    base: MatrixBaseSpecV6
    execution_dependencies: list[ExecutionDependencyV6] = Field(default_factory=list)
    fork: MatrixForkSpecV6 | None = None

    @model_validator(mode="after")
    def _validate_spec(self) -> "TrainingRunMatrixSpec":
        self._validate_common()
        forks = [
            item
            for item in self.execution_dependencies
            if isinstance(item, ForkFromSelectedCheckpointV6)
        ]
        if len(forks) > 1:
            raise ValueError("/execution_dependencies admits at most one selected checkpoint")
        row_ids = {row.row_id for row in self.rows}
        placed: set[tuple[str, str]] = set()
        for dependency in forks:
            for transform in dependency.slot_transforms:
                if transform.stage != "target_post":
                    continue
                assert transform.target_row_id is not None
                if transform.target_row_id not in row_ids:
                    raise ValueError(
                        "/execution_dependencies target transform names an unknown matrix row"
                    )
                key = (transform.target_row_id, transform.slot)
                if key in placed:
                    raise ValueError(
                        "/execution_dependencies target row-slot transforms must be unique"
                    )
                placed.add(key)
        if forks:
            dependency = forks[0]
            if isinstance(self.base, ResolvedOutputMatrixBaseSpecV6):
                if not isinstance(
                    dependency.source_authority, ResolvedOutputRootForkSourceAuthority
                ):
                    raise ValueError(
                        "/base resolved output and /execution_dependencies source authority disagree"
                    )
                if (
                    dependency.source_authority.resolved_root_hash != self.base.resolved_root_hash
                    or dependency.source_row_id != self.base.row_id
                    or dependency.checkpoint_transaction_id != self.base.checkpoint_transaction_id
                ):
                    raise ValueError(
                        "/base resolved output and /execution_dependencies selected checkpoint drift"
                    )
            elif isinstance(dependency.source_authority, ResolvedOutputRootForkSourceAuthority):
                raise ValueError(
                    "/execution_dependencies resolved-output-root authority requires a resolved-output base"
                )
        return self


TrainingRunMatrixSpecVersioned: TypeAlias = TrainingRunMatrixSpecV5 | TrainingRunMatrixSpec


def apply_composition_deltas(
    payload: dict[str, Any],
    deltas: list[MatrixCompositionDelta],
    *,
    ancestor_written_paths: set[str] | None = None,
    allowed_same_schema_structural_additions_by_layer: (
        Mapping[str, Mapping[str, tuple[str, str]]] | None
    ) = None,
) -> tuple[dict[str, Any], dict[str, str], set[str]]:
    """Apply ordered layers and fail closed on unacknowledged ancestor overrides."""
    result = deepcopy(payload)
    written = set(ancestor_written_paths or ())
    attribution: dict[str, str] = {}
    current_schema = _payload_schema_identity(result)
    for delta in deltas:
        acknowledged = set(delta.acknowledges_ancestor_paths)
        same_layer_appends: set[str] = set()
        declared_boundary = (delta.schema_id, delta.schema_version)
        prior_identities = _schema_identities(result)
        for patch in delta.patches:
            conflicts = sorted(
                path
                for path in written
                if _paths_overlap(path, patch.path)
                and not (
                    patch.op == "add"
                    and patch.path.endswith(".-")
                    and path == patch.path
                    and path in same_layer_appends
                )
            )
            unacknowledged = [
                path
                for path in conflicts
                if not any(
                    _paths_overlap(path, acknowledgement)
                    and _paths_overlap(patch.path, acknowledgement)
                    for acknowledgement in acknowledged
                )
            ]
            if unacknowledged:
                raise ValueError(
                    f"/deltas/{delta.layer_id}/{patch.path} overlaps ancestor-written "
                    f"paths {unacknowledged!r} without explicit acknowledgement"
                )
            try:
                _apply_patch(result, patch, error_context="override")
            except ValueError as error:
                raise ValueError(f"/deltas/{delta.layer_id}: {error}") from error
            written.add(patch.path)
            if patch.op == "add" and patch.path.endswith(".-"):
                same_layer_appends.add(patch.path)
            attribution[patch.path] = delta.layer_id
        resulting_schema = _payload_schema_identity(result)
        resulting_identities = _schema_identities(result)
        declared_additions = dict(
            (allowed_same_schema_structural_additions_by_layer or {}).get(delta.layer_id, {})
        )
        added_identities = {
            path: identity
            for path, identity in resulting_identities.items()
            if path not in prior_identities
        }
        changed_or_removed_identities = {
            path: (identity, resulting_identities.get(path))
            for path, identity in prior_identities.items()
            if resulting_identities.get(path) != identity
        }
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
        elif declared_additions:
            if resulting_schema != current_schema:
                raise ValueError(
                    f"/deltas/{delta.layer_id} same-schema structural additions change "
                    f"the active root identity from {current_schema!r} to "
                    f"{resulting_schema!r}"
                )
            if changed_or_removed_identities:
                raise ValueError(
                    f"/deltas/{delta.layer_id} same-schema structural additions change or "
                    f"remove existing schema identities {changed_or_removed_identities!r}"
                )
            required_additions = added_identities
            unqualified = {
                path: identity
                for path, identity in required_additions.items()
                if declared_additions.get(path) != identity
            }
            stale_or_invalid = {
                path: identity
                for path, identity in declared_additions.items()
                if path in added_identities and added_identities.get(path) != identity
            }
            if unqualified or stale_or_invalid:
                raise ValueError(
                    f"/deltas/{delta.layer_id} same-schema structural additions must exactly "
                    "qualify every added schema identity; "
                    f"unqualified_or_mismatched={unqualified!r}, "
                    f"declared_but_absent_or_invalid={stale_or_invalid!r}"
                )
            add_patch_paths = [patch.path for patch in delta.patches if patch.op == "add"]
            not_added = sorted(
                path
                for path in declared_additions
                if not any(_path_contains(patch_path, path) for patch_path in add_patch_paths)
            )
            if not_added:
                raise ValueError(
                    f"/deltas/{delta.layer_id} same-schema structural additions are not "
                    f"introduced by an absent-only add patch: {not_added!r}"
                )
        elif resulting_identities != prior_identities:
            raise ValueError(
                f"/deltas/{delta.layer_id} changes schema identity from "
                f"{prior_identities!r} to {resulting_identities!r} without a declared "
                "schema_id/schema_version boundary"
            )
        current_schema = resulting_schema
    return result, attribution, written


def _paths_overlap(left: str, right: str) -> bool:
    """Return whether either dotted path denotes the other's subtree."""
    return left == right or left.startswith(f"{right}.") or right.startswith(f"{left}.")


def _path_contains(parent: str, child: str) -> bool:
    """Return whether ``parent`` denotes ``child`` or one of its ancestors."""
    return parent == child or child.startswith(f"{parent}.")


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
    if isinstance(query, MapObjectList):
        return "default" in query.items.model_fields_set
    return False


def _query_uses_only_row_context(query: ValueExpr) -> bool:
    """Return whether a derivation can evaluate from its evolving row payload."""
    if isinstance(query, ValueQuery):
        return query.item == "row"
    if isinstance(query, Coalesce):
        return all(child.item == "row" for child in query.queries)
    if isinstance(query, MapObjectList):
        return query.items.item == "row"
    return False
