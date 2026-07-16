"""Durable manifest and artifact-reference models for Feedbax runs.

The database remains useful as an index, but these models are the portable
records that describe specs, executions, lineage, and large output artifacts.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from feedbax.contracts.graph import AnalysisInputRequirement
from feedbax.contracts.retention_artifact_schema import (
    RETENTION_ARTIFACT_ROLE_SCHEMAS,
    retained_observables_to_json,
    retention_artifact_metadata,
    retention_artifact_schema,
)
from feedbax.contracts.schema_namespace import validate_schema_identity

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:  # pragma: no cover - Python 3.12 always has importlib.metadata.
    PackageNotFoundError = Exception  # type: ignore[assignment]
    version = None  # type: ignore[assignment]


SCHEMA_VERSION = "feedbax.manifest.v1"
TRAINING_RUN_SET_SCHEMA_VERSION_V1 = SCHEMA_VERSION
TRAINING_RUN_SET_SCHEMA_VERSION = "feedbax.manifest.training_run_set.v2"
PROVIDER_VERSION = "feedbax-provider.v1"
DEFAULT_MANIFEST_ROOT_ENV = "FEEDBAX_RUNS_DIR"
REGENERATION_SPEC_SCHEMA_ID = "feedbax.spec.regeneration"
REGENERATION_SPEC_SCHEMA_VERSION = "feedbax.spec.regeneration.v1"
ANALYSIS_DATA_PRODUCT_SCHEMA_ID = "feedbax.manifest.analysis_data_product"
ANALYSIS_DATA_PRODUCT_SCHEMA_VERSION = "feedbax.manifest.analysis_data_product.v1"
EVALUATION_STATES_CONTAINER_SCHEMA_ID = "feedbax.manifest.evaluation_states_container"
EVALUATION_STATES_CONTAINER_SCHEMA_VERSION_V1 = "feedbax.manifest.evaluation_states_container.v1"
EVALUATION_STATES_CONTAINER_SCHEMA_VERSION = "feedbax.manifest.evaluation_states_container.v2"
ANALYSIS_RUN_SPEC_SCHEMA_ID = "feedbax.spec.analysis_run"
ANALYSIS_RUN_SPEC_SCHEMA_VERSION_V1 = "feedbax.spec.analysis_run.v1"
ANALYSIS_RUN_SPEC_SCHEMA_VERSION = "feedbax.spec.analysis_run.v2"
ANALYSIS_EVALUATION_STATE_SOURCE_SCHEMA_ID = (
    "feedbax.manifest.analysis_evaluation_state_source"
)
ANALYSIS_EVALUATION_STATE_SOURCE_SCHEMA_VERSION_V1 = (
    "feedbax.manifest.analysis_evaluation_state_source.v1"
)
ANALYSIS_EVALUATION_STATE_SOURCE_SCHEMA_VERSION = (
    "feedbax.manifest.analysis_evaluation_state_source.v2"
)
ANALYSIS_EVALUATION_STATE_RESOLUTION_DIAGNOSTIC_SCHEMA_ID = (
    "feedbax.manifest.analysis_evaluation_state_resolution_diagnostic"
)
ANALYSIS_EVALUATION_STATE_RESOLUTION_DIAGNOSTIC_SCHEMA_VERSION = (
    "feedbax.manifest.analysis_evaluation_state_resolution_diagnostic.v1"
)
ANALYSIS_RUN_MANIFEST_SCHEMA_VERSION_V1 = SCHEMA_VERSION
ANALYSIS_RUN_MANIFEST_SCHEMA_VERSION = "feedbax.manifest.analysis_run.v2"
FIGURE_MANIFEST_SCHEMA_ID = "feedbax.manifest.figure"
FIGURE_MANIFEST_SCHEMA_VERSION = "feedbax.manifest.figure.v1"
TRAINING_MANIFEST_METADATA_PROJECTION_CUSTODY_SCHEMA_ID = (
    "feedbax.manifest.training_metadata_projection_custody"
)
TRAINING_MANIFEST_METADATA_PROJECTION_CUSTODY_SCHEMA_VERSION = (
    "feedbax.manifest.training_metadata_projection_custody.v1"
)
TRAINING_MANIFEST_METADATA_PROJECTION_PROVENANCE_KEY = "manifest_metadata_projection"
STAGED_EVALUATION_PREREQUISITE_SCHEMA_ID = "feedbax.spec.staged_evaluation_prerequisite"
STAGED_EVALUATION_PREREQUISITE_SCHEMA_VERSION = (
    "feedbax.spec.staged_evaluation_prerequisite.v1"
)
AUTHENTICATED_MANIFEST_REF_SCHEMA_ID = "feedbax.ref.authenticated_manifest"
AUTHENTICATED_MANIFEST_REF_SCHEMA_VERSION = "feedbax.ref.authenticated_manifest.v1"

_AUTHENTICATED_MANIFEST_REF_PROFILE_DISCRIMINATORS = frozenset(
    {"ref_schema_id", "ref_schema_version"}
)
_AUTHENTICATED_MANIFEST_REF_PROFILE_KEYS = (
    _AUTHENTICATED_MANIFEST_REF_PROFILE_DISCRIMINATORS
    | {"manifest_sha256", "size_bytes"}
)

ManifestStatus = Literal["pending", "running", "completed", "failed", "cancelled", "stale"]


def feedbax_version() -> str:
    """Return the installed Feedbax package version, or a useful local fallback."""
    if version is None:
        return "unknown"
    try:
        return version("feedbax")
    except PackageNotFoundError:
        return "unknown"


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp with stable second precision."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def default_manifest_root() -> Path:
    """Return the root directory for local manifests and artifacts."""
    configured = os.environ.get(DEFAULT_MANIFEST_ROOT_ENV)
    if configured:
        return Path(configured).expanduser()
    return Path.cwd() / "feedbax_runs"


class StrictModel(BaseModel):
    """Base model for provider-contract records."""

    model_config = ConfigDict(extra="forbid")


class ArtifactRef(StrictModel):
    """Reference to a large output artifact stored outside a manifest."""

    role: str
    logical_name: str
    artifact_id: Optional[str] = None
    sha256: Optional[str] = None
    media_type: str = "application/octet-stream"
    size_bytes: Optional[int] = None
    storage_backend: str = "feedbax-local"
    uri: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ArrayStoreRef(StrictModel):
    """Reference to a role-addressed parameter/state array store."""

    role: Literal["params", "state", "optimizer", "history"]
    schema_version: str
    storage_backend: str
    logical_name: str
    artifact_id: Optional[str] = None
    sha256: Optional[str] = None
    uri: Optional[str] = None
    array_count: int
    roles: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ArtifactValidationRecord(StrictModel):
    """Validation outcome for a durable artifact or migration step."""

    name: str
    status: Literal["passed", "failed", "warning"]
    checked_at: datetime = Field(default_factory=utc_now)
    schema_version: Optional[str] = None
    details: dict[str, Any] = Field(default_factory=dict)


class ArtifactMigrationRecord(StrictModel):
    """Provenance for a schema-to-schema artifact migration."""

    migration_id: str
    source_schema_version: str
    target_schema_version: str
    applied_at: datetime = Field(default_factory=utc_now)
    tool: str = "feedbax"
    deterministic: bool = True
    validation: list[ArtifactValidationRecord] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EntrypointRef(StrictModel):
    """How a manifest-producing operation was invoked."""

    kind: str
    command: Optional[str] = None
    name: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParentRef(StrictModel):
    """Reference to an input spec, parent manifest, or parent artifact."""

    kind: str
    id: str
    role: Optional[str] = None
    uri: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def authenticated_manifest_ref_profile(ref: ParentRef) -> tuple[str, int] | None:
    """Return one ref's authenticated byte profile, if it declares one.

    Partial or unsupported authenticated profiles raise rather than degrading to
    an unauthenticated manifest reference.
    """

    discriminators = _AUTHENTICATED_MANIFEST_REF_PROFILE_DISCRIMINATORS.intersection(
        ref.metadata
    )
    if not discriminators:
        return None
    present = _AUTHENTICATED_MANIFEST_REF_PROFILE_KEYS.intersection(ref.metadata)
    if present != _AUTHENTICATED_MANIFEST_REF_PROFILE_KEYS:
        missing = ", ".join(
            sorted(_AUTHENTICATED_MANIFEST_REF_PROFILE_KEYS - present)
        )
        raise ValueError(f"Authenticated manifest ref {ref.id!r} is incomplete: {missing}")
    schema_id = ref.metadata["ref_schema_id"]
    schema_version = ref.metadata["ref_schema_version"]
    digest = ref.metadata["manifest_sha256"]
    size = ref.metadata["size_bytes"]
    if schema_id != AUTHENTICATED_MANIFEST_REF_SCHEMA_ID:
        raise ValueError(f"Unsupported authenticated manifest ref schema_id: {schema_id!r}")
    if schema_version != AUTHENTICATED_MANIFEST_REF_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported authenticated manifest ref schema_version: {schema_version!r}"
        )
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"Authenticated manifest ref {ref.id!r} has invalid SHA-256")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError(f"Authenticated manifest ref {ref.id!r} has invalid byte size")
    if ref.uri is not None:
        raise ValueError("Authenticated manifest refs must keep machine-local locators out of uri")
    return digest, size


class StagedEvaluationPrerequisite(StrictModel):
    """Portable prerequisite injected into one staged evaluation's parameters."""

    schema_id: str = STAGED_EVALUATION_PREREQUISITE_SCHEMA_ID
    schema_version: str = STAGED_EVALUATION_PREREQUISITE_SCHEMA_VERSION
    parent: ParentRef
    artifact_provider: str | None = None

    @model_validator(mode="after")
    def _validate_schema_identity(self) -> "StagedEvaluationPrerequisite":
        if self.schema_id != STAGED_EVALUATION_PREREQUISITE_SCHEMA_ID:
            raise ValueError(f"unsupported staged prerequisite schema_id: {self.schema_id!r}")
        if self.schema_version != STAGED_EVALUATION_PREREQUISITE_SCHEMA_VERSION:
            raise ValueError(
                "unsupported staged prerequisite schema_version: "
                f"{self.schema_version!r}"
            )
        return self


class EvaluationParamsBase(StrictModel):
    """Strict public base for recipe params, including Feedbax-reserved fields."""

    staged_prerequisites: dict[str, StagedEvaluationPrerequisite] | None = None


class Provenance(StrictModel):
    """Shared provenance fields recorded on durable manifests."""

    source_repo: Optional[str] = None
    source_branch: Optional[str] = None
    source_commit: Optional[str] = None
    dirty: Optional[bool] = None
    entrypoint: Optional[EntrypointRef] = None
    issues: list[str] = Field(default_factory=list)
    parents: list[ParentRef] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class FileHashRef(StrictModel):
    """Deterministic content hash for one source or artifact file."""

    path: str
    sha256: str
    size_bytes: int
    role: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TreeHashEntry(StrictModel):
    """One file entry included in a deterministic tree hash."""

    path: str
    sha256: str
    size_bytes: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class TreeHashRef(StrictModel):
    """Deterministic hash for a directory tree and its member file hashes."""

    path: str
    sha256: str
    file_count: int
    total_size_bytes: int
    files: list[TreeHashEntry] = Field(default_factory=list)
    role: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RegenerationCommand(StrictModel):
    """Command form used to regenerate one or more analysis/report artifacts."""

    argv: list[str] = Field(default_factory=list)
    shell_command: Optional[str] = None
    cwd: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_invocation(self) -> "RegenerationCommand":
        if not self.argv and not self.shell_command:
            raise ValueError("regeneration command requires argv or shell_command")
        return self


class RegenerationSpec(StrictModel):
    """Generic replay record for regenerating analysis or report artifacts."""

    schema_id: str = REGENERATION_SPEC_SCHEMA_ID
    schema_version: str = REGENERATION_SPEC_SCHEMA_VERSION
    command: RegenerationCommand
    parameters: dict[str, Any] = Field(default_factory=dict)
    inputs: list[ParentRef | ArtifactRef] = Field(default_factory=list)
    outputs: list[ParentRef | ArtifactRef] = Field(default_factory=list)
    source_files: list[FileHashRef] = Field(default_factory=list)
    source_trees: list[TreeHashRef] = Field(default_factory=list)
    provenance: Provenance = Field(default_factory=Provenance)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_schema_identity(self) -> "RegenerationSpec":
        if self.schema_id != REGENERATION_SPEC_SCHEMA_ID:
            raise ValueError(
                "unsupported RegenerationSpec schema_id: "
                f"{self.schema_id!r}, expected {REGENERATION_SPEC_SCHEMA_ID!r}"
            )
        if self.schema_version != REGENERATION_SPEC_SCHEMA_VERSION:
            raise ValueError(
                "unsupported RegenerationSpec schema_version: "
                f"{self.schema_version!r}, expected {REGENERATION_SPEC_SCHEMA_VERSION!r}"
            )
        return self


class DataProductParentRef(StrictModel):
    """Parent manifest identity included in an analysis data-product envelope."""

    kind: str
    id: str
    role: Optional[str] = None
    manifest_hash: Optional[str] = None
    uri: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AnalysisDataProduct(StrictModel):
    """Typed semantic product emitted by an analysis or materialization step.

    ``descriptor_basis_hash`` is an optional contract slot for issue 844acc6.
    Producers set it when descriptor/component selector identity affects the
    product values; this model only preserves and compares the hash.
    """

    schema_id: str = ANALYSIS_DATA_PRODUCT_SCHEMA_ID
    schema_version: str = ANALYSIS_DATA_PRODUCT_SCHEMA_VERSION
    product_schema_id: str
    product_schema_version: str
    role: str
    logical_name: str
    label: Optional[str] = None
    producer_manifest_id: str
    producer_manifest_hash: Optional[str] = None
    parent_manifests: list[DataProductParentRef] = Field(default_factory=list)
    checkpoint_policy: dict[str, Any] = Field(default_factory=dict)
    rollout_policy: dict[str, Any] = Field(default_factory=dict)
    parameters: dict[str, Any] = Field(default_factory=dict)
    descriptor_basis_hash: Optional[str] = None
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    materialization: dict[str, Any] = Field(default_factory=dict)
    regeneration: list[RegenerationSpec | ParentRef | ArtifactRef] = Field(default_factory=list)
    product_identity_hash: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_identity(self) -> "AnalysisDataProduct":
        if self.schema_id != ANALYSIS_DATA_PRODUCT_SCHEMA_ID:
            raise ValueError(
                "unsupported AnalysisDataProduct schema_id: "
                f"{self.schema_id!r}, expected {ANALYSIS_DATA_PRODUCT_SCHEMA_ID!r}"
            )
        if self.schema_version != ANALYSIS_DATA_PRODUCT_SCHEMA_VERSION:
            raise ValueError(
                "unsupported AnalysisDataProduct schema_version: "
                f"{self.schema_version!r}, expected {ANALYSIS_DATA_PRODUCT_SCHEMA_VERSION!r}"
            )
        if not self.product_schema_id.strip():
            raise ValueError("AnalysisDataProduct product_schema_id must not be empty")
        if self.product_schema_id.startswith("feedbax."):
            validate_schema_identity(
                self.product_schema_id,
                family="AnalysisDataProduct.product_schema_id",
            )
        if not self.product_schema_version.strip():
            raise ValueError("AnalysisDataProduct product_schema_version must not be empty")
        if not self.role.strip():
            raise ValueError("AnalysisDataProduct role must not be empty")
        if not self.logical_name.strip():
            raise ValueError("AnalysisDataProduct logical_name must not be empty")
        if not self.producer_manifest_id.strip():
            raise ValueError("AnalysisDataProduct producer_manifest_id must not be empty")

        expected_hash = analysis_data_product_identity_hash(self)
        if self.product_identity_hash is not None and self.product_identity_hash != expected_hash:
            raise ValueError(
                "AnalysisDataProduct product_identity_hash does not match semantic "
                f"envelope: product_identity_hash={self.product_identity_hash!r}, "
                f"computed={expected_hash!r}"
            )
        self.product_identity_hash = expected_hash
        return self


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
        if any(not part for part in self.path.split(".")):
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


class TrainingRunSetManifest(BaseManifest):
    kind: Literal["TrainingRunSetManifest"] = "TrainingRunSetManifest"
    schema_version: str = TRAINING_RUN_SET_SCHEMA_VERSION
    name: str
    run_ids: list[str] = Field(default_factory=list)
    graph_spec: Optional[ParentRef | SpecPayload] = None
    axes: TrainingRunSetAxes = Field(default_factory=TrainingRunSetAxes)
    tags: list[str] = Field(default_factory=list)
    migration_records: list[ArtifactMigrationRecord] = Field(default_factory=list)


class TrainingManifestMetadataProjectionCustody(StrictModel):
    """Auditable consistency binding for governed projected metadata.

    The hashes detect partial drift among this record, the embedded source,
    root metadata, and provenance. They do not prove authorship or authenticity;
    that requires an external signed or content-addressed custody anchor.
    """

    schema_id: Literal["feedbax.manifest.training_metadata_projection_custody"] = (
        TRAINING_MANIFEST_METADATA_PROJECTION_CUSTODY_SCHEMA_ID
    )
    schema_version: str = TRAINING_MANIFEST_METADATA_PROJECTION_CUSTODY_SCHEMA_VERSION
    source_payload_kind: str = Field(min_length=1)
    source_payload_schema_id: str = Field(min_length=1)
    source_payload_schema_version: str = Field(min_length=1)
    source_payload_sha256: str
    projection_schema_id: str = Field(min_length=1)
    projection_schema_version: str = Field(min_length=1)
    values: dict[str, Any]
    values_sha256: str
    registration_owner: str = Field(min_length=1)
    registration_package: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_custody(self) -> "TrainingManifestMetadataProjectionCustody":
        from feedbax.contracts.spec_storage import (
            training_spec_canonical_bytes,
            training_spec_sha256,
            validate_sha256,
        )

        if self.schema_version != TRAINING_MANIFEST_METADATA_PROJECTION_CUSTODY_SCHEMA_VERSION:
            raise ValueError(
                "unsupported training metadata projection custody schema_version "
                f"{self.schema_version!r}; expected "
                f"{TRAINING_MANIFEST_METADATA_PROJECTION_CUSTODY_SCHEMA_VERSION!r}; "
                "migration_intentionally_absent=yes"
            )
        validate_sha256(self.source_payload_sha256, field_name="source_payload_sha256")
        validate_sha256(self.values_sha256, field_name="values_sha256")
        canonical_values = json.loads(training_spec_canonical_bytes(self.values))
        if self.values != canonical_values:
            raise ValueError("training metadata projection values are not JSON-canonical")
        expected_digest = training_spec_sha256(self.values)
        if self.values_sha256 != expected_digest:
            raise ValueError(
                "training metadata projection values_sha256 does not match canonical values; "
                f"expected={expected_digest}, observed={self.values_sha256}"
            )
        return self

    def provenance_summary(self) -> dict[str, Any]:
        """Return the exact compact provenance record bound to this custody envelope."""
        return {
            "source_payload_kind": self.source_payload_kind,
            "source_payload_schema_id": self.source_payload_schema_id,
            "source_payload_schema_version": self.source_payload_schema_version,
            "source_payload_sha256": self.source_payload_sha256,
            "projection_schema_id": self.projection_schema_id,
            "projection_schema_version": self.projection_schema_version,
            "projected_keys": sorted(self.values),
            "values_sha256": self.values_sha256,
            "registration_owner": self.registration_owner,
            "registration_package": self.registration_package,
        }


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
    intent_hash: Optional[str] = None
    execution_hash: Optional[str] = None
    resolved_semantics_root_hash: Optional[str] = None
    input_data_identities: list[dict[str, Any]] = Field(default_factory=list)
    summary_metrics: dict[str, Any] = Field(default_factory=dict)
    metadata_projection_custody: TrainingManifestMetadataProjectionCustody | None = None

    @model_validator(mode="after")
    def _validate_execution_identity(self) -> "TrainingRunManifest":
        from feedbax.contracts.spec_storage import validate_sha256

        if self.stopped and self.completed_at is None:
            raise ValueError("stopped training runs require completed_at")
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
            self.execution_hash = expected
        projection = self.metadata_projection_custody
        if projection is not None:
            if self.training_spec is None:
                raise ValueError(
                    "training metadata projection custody requires embedded training_spec"
                )
            training_spec_identity = (
                self.training_spec.kind,
                self.training_spec.schema_id,
                self.training_spec.schema_version,
            )
            projection_source_identity = (
                projection.source_payload_kind,
                projection.source_payload_schema_id,
                projection.source_payload_schema_version,
            )
            if projection_source_identity != training_spec_identity:
                raise ValueError(
                    "training metadata projection source identity disagrees with "
                    "embedded training_spec"
                )
            from feedbax.contracts.spec_storage import training_spec_sha256

            observed_source_sha256 = training_spec_sha256(self.training_spec.inline)
            if projection.source_payload_sha256 != observed_source_sha256:
                raise ValueError(
                    "training metadata projection source hash disagrees with embedded training_spec"
                )
            for key, value in projection.values.items():
                if key not in self.metadata or self.metadata[key] != value:
                    raise ValueError(
                        "training metadata projection custody disagrees with root metadata; "
                        f"key={key!r}"
                    )
            expected_provenance = projection.provenance_summary()
            observed_provenance = self.provenance.metadata.get(
                TRAINING_MANIFEST_METADATA_PROJECTION_PROVENANCE_KEY
            )
            if observed_provenance != expected_provenance:
                raise ValueError(
                    "training metadata projection custody disagrees with provenance summary"
                )
        return self


class EvaluationRunSpec(StrictModel):
    """Declarative request for an evaluation run."""

    evaluation_type: str
    training_run_ids: list[str] = Field(default_factory=list)
    inputs: list[ParentRef] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)


EVALUATION_RUN_MATRIX_SPEC_SCHEMA_ID = "feedbax.spec.evaluation_run_matrix"
EVALUATION_RUN_MATRIX_SPEC_SCHEMA_VERSION_V1 = "feedbax.spec.evaluation_run_matrix.v1"
EVALUATION_RUN_MATRIX_SPEC_SCHEMA_VERSION_V2 = "feedbax.spec.evaluation_run_matrix.v2"
EVALUATION_RUN_MATRIX_SPEC_SCHEMA_VERSION = "feedbax.spec.evaluation_run_matrix.v3"
EVALUATION_AXIS_EXPANSION_PROVENANCE_SCHEMA_ID = (
    "feedbax.manifest.evaluation_axis_expansion_provenance"
)
EVALUATION_AXIS_EXPANSION_PROVENANCE_SCHEMA_VERSION = (
    "feedbax.manifest.evaluation_axis_expansion_provenance.v1"
)


class EvaluationRunManifest(BaseManifest):
    kind: Literal["EvaluationRunManifest"] = "EvaluationRunManifest"
    evaluation_spec: SpecPayload
    input_training_runs: list[ParentRef] = Field(default_factory=list)
    summary_metrics: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AuthenticatedManifestDigest:
    """Authenticated byte identity for one manifest in a provenance envelope."""

    kind: str
    id: str
    role: str | None
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class EvaluationManifestProvenanceEnvelope:
    """Verified producer and source authority derived from an evaluation manifest."""

    producer_identity: str
    source_refs: tuple[ParentRef, ...]
    digest_envelope: tuple[AuthenticatedManifestDigest, ...]


def verify_evaluation_manifest_provenance(
    manifest_ref: ParentRef,
    raw_bytes: bytes,
    *,
    expected_producer_identity: str,
    expected_source_refs: tuple[ParentRef, ...] | None = None,
) -> EvaluationManifestProvenanceEnvelope:
    """Verify one authenticated evaluation manifest's producer and source envelope.

    The manifest is parsed from ``raw_bytes`` only after ``manifest_ref`` authenticates
    their exact size and digest. Remaining digest records align positionally with
    ``source_refs``. This contract does not introduce a new durable schema.
    """

    manifest_profile = authenticated_manifest_ref_profile(manifest_ref)
    if manifest_profile is None:
        raise ValueError("evaluation manifest authority is not authenticated")
    manifest_digest, manifest_size = manifest_profile
    if len(raw_bytes) != manifest_size:
        raise ValueError("evaluation manifest authority byte size mismatch")
    if hashlib.sha256(raw_bytes).hexdigest() != manifest_digest:
        raise ValueError("evaluation manifest authority SHA-256 mismatch")

    manifest = load_manifest_bytes(raw_bytes)
    if manifest.status != "completed":
        raise ValueError("evaluation provenance requires a completed manifest")
    if (
        not isinstance(manifest, EvaluationRunManifest)
        or manifest_ref.kind != manifest.kind
        or manifest_ref.id != manifest.id
        or manifest_ref.role != "evaluation_run"
    ):
        raise ValueError("evaluation manifest authority disagrees with the manifest")
    run_spec = EvaluationRunSpec.model_validate(manifest.evaluation_spec.inline)
    entrypoint = manifest.provenance.entrypoint
    if (
        run_spec.evaluation_type != expected_producer_identity
        or entrypoint is None
        or entrypoint.kind != "feedbax-evaluation-recipe"
        or entrypoint.name != run_spec.evaluation_type
    ):
        raise ValueError("evaluation manifest producer identity is not canonical")
    if manifest.input_training_runs != run_spec.inputs:
        raise ValueError("evaluation manifest training sources disagree with its spec")

    staged = run_spec.params.get("staged_prerequisites") or {}
    if not isinstance(staged, dict):
        raise ValueError("evaluation staged prerequisites must be a mapping")
    staged_refs = tuple(
        StagedEvaluationPrerequisite.model_validate(value).parent
        for value in staged.values()
    )
    source_refs = (*run_spec.inputs, *staged_refs)
    if tuple(manifest.provenance.parents) != source_refs:
        raise ValueError("evaluation provenance parents disagree with declared sources")
    if expected_source_refs is not None and expected_source_refs != source_refs:
        raise ValueError("evaluation manifest sources disagree with expected sources")

    profiles = (manifest_profile,)
    for ref in source_refs:
        profile = authenticated_manifest_ref_profile(ref)
        if profile is None:
            raise ValueError(f"evaluation source {ref.id!r} is not authenticated")
        profiles += (profile,)
    refs = (manifest_ref, *source_refs)
    digests = tuple(
        AuthenticatedManifestDigest(ref.kind, ref.id, ref.role, digest, size)
        for ref, (digest, size) in zip(refs, profiles)
    )
    return EvaluationManifestProvenanceEnvelope(
        producer_identity=run_spec.evaluation_type,
        source_refs=source_refs,
        digest_envelope=digests,
    )


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


EvaluationStatesConsumptionPolicy = Literal["recompute", "require_durable"]


class AnalysisRunSpec(StrictModel):
    """Declarative request for an analysis run."""

    schema_id: Literal["feedbax.spec.analysis_run"] = ANALYSIS_RUN_SPEC_SCHEMA_ID
    schema_version: Literal["feedbax.spec.analysis_run.v2"] = ANALYSIS_RUN_SPEC_SCHEMA_VERSION
    analysis_type: str
    inputs: list[ParentRef] = Field(default_factory=list)
    input_requirements: list[AnalysisInputRequirement] = Field(default_factory=list)
    evaluation_states_policy: EvaluationStatesConsumptionPolicy = "recompute"
    params: dict[str, Any] = Field(default_factory=dict)


class AnalysisEvaluationStateSource(StrictModel):
    """Queryable provenance for evaluation states consumed by one analysis."""

    schema_id: Literal[
        "feedbax.manifest.analysis_evaluation_state_source"
    ] = ANALYSIS_EVALUATION_STATE_SOURCE_SCHEMA_ID
    schema_version: Literal[
        "feedbax.manifest.analysis_evaluation_state_source.v2"
    ] = ANALYSIS_EVALUATION_STATE_SOURCE_SCHEMA_VERSION
    source_kind: Literal["evaluation_cache", "durable", "analysis_time_recompute"]
    requested_evaluation_manifest_id: str
    evaluation_manifest_authority: Optional[ParentRef] = None
    supplying_evaluation_manifest_id: Optional[str] = None
    resulting_evaluation_manifest_id: Optional[str] = None
    resulting_evaluation_manifest_authority: Optional[ParentRef] = None
    cache_schema_version: Optional[str] = None
    cache_key: Optional[str] = None
    artifact_id: Optional[str] = None
    artifact_sha256: Optional[str] = None
    artifact_size_bytes: Optional[int] = Field(default=None, ge=0)
    artifact_storage_backend: Optional[str] = None
    container_schema_id: Optional[str] = None
    container_schema_version: Optional[str] = None
    container_storage_backend: Optional[str] = None

    @model_validator(mode="after")
    def _validate_source_evidence(self) -> "AnalysisEvaluationStateSource":
        if self.source_kind == "durable":
            required = {
                "supplying_evaluation_manifest_id": self.supplying_evaluation_manifest_id,
                "evaluation_manifest_authority": self.evaluation_manifest_authority,
                "artifact_id": self.artifact_id,
                "artifact_sha256": self.artifact_sha256,
                "artifact_size_bytes": self.artifact_size_bytes,
                "artifact_storage_backend": self.artifact_storage_backend,
                "container_schema_id": self.container_schema_id,
                "container_schema_version": self.container_schema_version,
                "container_storage_backend": self.container_storage_backend,
            }
            missing = sorted(
                name
                for name, value in required.items()
                if value is None or (isinstance(value, str) and not value)
            )
            if missing:
                raise ValueError(
                    "durable analysis evaluation-state source is missing evidence: "
                    f"{missing}"
                )
            assert self.evaluation_manifest_authority is not None
            if authenticated_manifest_ref_profile(self.evaluation_manifest_authority) is None:
                raise ValueError(
                    "durable analysis evaluation-state source requires a complete "
                    "authenticated evaluation_manifest_authority"
                )
        elif self.source_kind == "analysis_time_recompute":
            if (
                not self.resulting_evaluation_manifest_id
                or self.resulting_evaluation_manifest_authority is None
            ):
                raise ValueError(
                    "analysis_time_recompute source requires authenticated resulting "
                    "evaluation authority"
                )
            if (
                authenticated_manifest_ref_profile(
                    self.resulting_evaluation_manifest_authority
                )
                is None
            ):
                raise ValueError(
                    "analysis_time_recompute source requires a complete authenticated "
                    "resulting_evaluation_manifest_authority"
                )
        elif self.source_kind == "evaluation_cache":
            if not self.cache_schema_version or not self.cache_key:
                raise ValueError(
                    "evaluation_cache source requires its stable schema version and key"
                )
        return self


AnalysisEvaluationStateResolutionCode = Literal[
    "missing_durable_states",
    "custody_unavailable",
    "schema_mismatch",
    "provenance_mismatch",
]


class AnalysisEvaluationStateResolutionDiagnostic(StrictModel):
    """Stable actionable diagnostic for failed evaluation-state resolution."""

    schema_id: Literal[
        "feedbax.manifest.analysis_evaluation_state_resolution_diagnostic"
    ] = ANALYSIS_EVALUATION_STATE_RESOLUTION_DIAGNOSTIC_SCHEMA_ID
    schema_version: Literal[
        "feedbax.manifest.analysis_evaluation_state_resolution_diagnostic.v1"
    ] = ANALYSIS_EVALUATION_STATE_RESOLUTION_DIAGNOSTIC_SCHEMA_VERSION
    code: AnalysisEvaluationStateResolutionCode
    evaluation_manifest_id: str
    message: str
    artifact_id: Optional[str] = None
    details: dict[str, Any] = Field(default_factory=dict)


class AnalysisRunManifest(BaseManifest):
    kind: Literal["AnalysisRunManifest"] = "AnalysisRunManifest"
    schema_version: str = ANALYSIS_RUN_MANIFEST_SCHEMA_VERSION
    analysis_spec: SpecPayload
    inputs: list[ParentRef] = Field(default_factory=list)
    evaluation_state_sources: list[AnalysisEvaluationStateSource] = Field(default_factory=list)
    evaluation_state_resolution_diagnostics: list[
        AnalysisEvaluationStateResolutionDiagnostic
    ] = Field(default_factory=list)
    regeneration_specs: list[SpecPayload | ParentRef | ArtifactRef] = Field(default_factory=list)
    produced_data: list[AnalysisDataProduct] = Field(default_factory=list)
    summary_metrics: dict[str, Any] = Field(default_factory=dict)


class ReportSpec(StrictModel):
    """Declarative request for a report product."""

    report_type: str
    inputs: list[ParentRef] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    narrative: Optional[str] = None


class ReportManifest(BaseManifest):
    kind: Literal["ReportManifest"] = "ReportManifest"
    report_spec: SpecPayload
    inputs: list[ParentRef] = Field(default_factory=list)
    regeneration_specs: list[SpecPayload | ParentRef | ArtifactRef] = Field(default_factory=list)


class FigureBindingRecord(StrictModel):
    """Recorded outcome for one declarative figure binding."""

    name: str
    status: Literal["included", "omitted", "failed"]
    reason: Optional[str] = None
    panel: Optional[str] = None
    constructor: Optional[str] = None
    expression_hashes: list[str] = Field(default_factory=list)


class FigurePieceResolution(StrictModel):
    """Resolved identity for one piece consumed by a figure."""

    name: str
    source_kind: Literal["artifact_ref", "manifest_predicate", "generator_spec"]
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    manifest_refs: list[ParentRef] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class FigureManifest(BaseManifest):
    """Durable provenance for a rendered declarative figure."""

    kind: Literal["FigureManifest"] = "FigureManifest"
    schema_version: str = FIGURE_MANIFEST_SCHEMA_VERSION
    figure_spec: SpecPayload
    inputs: list[ParentRef] = Field(default_factory=list)
    resolved_inputs: list[ParentRef] = Field(default_factory=list)
    resolved_pieces: list[FigurePieceResolution] = Field(default_factory=list)
    constructor_versions: dict[str, str] = Field(default_factory=dict)
    template_name: Optional[str] = None
    template_version: Optional[str] = None
    binding_records: list[FigureBindingRecord] = Field(default_factory=list)
    expression_results_digest: Optional[str] = None
    failure: Optional[dict[str, Any]] = None
    regeneration_specs: list[SpecPayload | ParentRef | ArtifactRef] = Field(default_factory=list)


AnyManifest = (
    GraphSpecManifest
    | ModelArtifactManifest
    | TrainingRunSetManifest
    | TrainingRunManifest
    | EvaluationRunManifest
    | CheckpointSelectionManifest
    | AnalysisRunManifest
    | ReportManifest
    | FigureManifest
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
    "TrainingRunSetManifest": TrainingRunSetManifest,
    "TrainingRunManifest": TrainingRunManifest,
    "EvaluationRunManifest": EvaluationRunManifest,
    "CheckpointSelectionManifest": CheckpointSelectionManifest,
    "AnalysisRunManifest": AnalysisRunManifest,
    "ReportManifest": ReportManifest,
    "FigureManifest": FigureManifest,
}


def _manifest_model_for_kind(kind: str) -> type[BaseModel] | None:
    if kind == "TrainingCheckpointTransactionManifest":
        from feedbax.contracts.checkpoints import CheckpointTransactionManifest

        return CheckpointTransactionManifest
    return MANIFEST_MODELS.get(kind)


SPEC_PAYLOAD_FIELDS_BY_MANIFEST_KIND: dict[str, tuple[str, ...]] = {
    "GraphSpecManifest": ("graph_spec",),
    "ModelArtifactManifest": ("graph_spec",),
    "TrainingRunSetManifest": ("graph_spec",),
    "TrainingRunManifest": (
        "graph_spec",
        "training_spec",
        "task_spec",
        "task_binding_spec",
    ),
    "EvaluationRunManifest": ("evaluation_spec",),
    "CheckpointSelectionManifest": ("selection_spec",),
    "AnalysisRunManifest": ("analysis_spec", "regeneration_specs"),
    "ReportManifest": ("report_spec", "regeneration_specs"),
    "FigureManifest": ("figure_spec", "regeneration_specs"),
}


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a value using stable JSON for hashing."""
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", exclude_none=True)
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def analysis_data_product_identity_envelope(product: AnalysisDataProduct) -> dict[str, Any]:
    """Return the semantic envelope hashed into ``product_identity_hash``.

    The envelope includes the typed product schema, role, logical name, producer
    and parent manifest identities, checkpoint/rollout policies, value-affecting
    parameters, optional descriptor basis, artifact byte identities, external
    materialization metadata, and regeneration metadata. It intentionally
    excludes human labels, arbitrary product metadata, and mutable local URIs.
    """

    def artifact_identity(artifact: ArtifactRef) -> dict[str, Any]:
        return {
            "role": artifact.role,
            "logical_name": artifact.logical_name,
            "artifact_id": artifact.artifact_id,
            "sha256": artifact.sha256,
            "media_type": artifact.media_type,
            "size_bytes": artifact.size_bytes,
            "storage_backend": artifact.storage_backend,
            "metadata": artifact.metadata,
        }

    return {
        "schema_id": product.schema_id,
        "schema_version": product.schema_version,
        "product_schema_id": product.product_schema_id,
        "product_schema_version": product.product_schema_version,
        "role": product.role,
        "logical_name": product.logical_name,
        "producer_manifest_id": product.producer_manifest_id,
        "producer_manifest_hash": product.producer_manifest_hash,
        "parent_manifests": [
            parent.model_dump(mode="json", exclude_none=True) for parent in product.parent_manifests
        ],
        "checkpoint_policy": product.checkpoint_policy,
        "rollout_policy": product.rollout_policy,
        "parameters": product.parameters,
        "descriptor_basis_hash": product.descriptor_basis_hash,
        "artifacts": [artifact_identity(artifact) for artifact in product.artifacts],
        "materialization": product.materialization,
        "regeneration": [
            item.model_dump(mode="json", exclude_none=True) for item in product.regeneration
        ],
    }


def analysis_data_product_identity_hash(product: AnalysisDataProduct) -> str:
    """Hash the deterministic semantic envelope for an analysis data product."""
    return sha256_bytes(canonical_json_bytes(analysis_data_product_identity_envelope(product)))


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_hash_ref(
    path: Path | str,
    *,
    root: Path | str | None = None,
    role: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> FileHashRef:
    """Return a deterministic hash reference for one file."""
    file_path = Path(path)
    display_path = str(file_path if root is None else file_path.relative_to(Path(root)))
    stat = file_path.stat()
    return FileHashRef(
        path=display_path,
        sha256=sha256_file(file_path),
        size_bytes=stat.st_size,
        role=role,
        metadata=dict(metadata or {}),
    )


def tree_hash_ref(
    path: Path | str,
    *,
    root: Path | str | None = None,
    role: Optional[str] = None,
    include_files: bool = True,
    metadata: Optional[dict[str, Any]] = None,
) -> TreeHashRef:
    """Return a deterministic hash reference for regular files under a directory."""
    tree_path = Path(path)
    if not tree_path.is_dir():
        raise NotADirectoryError(tree_path)

    entries: list[TreeHashEntry] = []
    total_size = 0
    for file_path in sorted(candidate for candidate in tree_path.rglob("*") if candidate.is_file()):
        relative_path = str(file_path.relative_to(tree_path))
        stat = file_path.stat()
        total_size += stat.st_size
        entries.append(
            TreeHashEntry(
                path=relative_path,
                sha256=sha256_file(file_path),
                size_bytes=stat.st_size,
            )
        )
    digest_payload = [entry.model_dump(mode="json", exclude_none=True) for entry in entries]
    display_path = str(tree_path if root is None else tree_path.relative_to(Path(root)))
    return TreeHashRef(
        path=display_path,
        sha256=sha256_bytes(canonical_json_bytes(digest_payload)),
        file_count=len(entries),
        total_size_bytes=total_size,
        files=entries if include_files else [],
        role=role,
        metadata=dict(metadata or {}),
    )


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
    assume_current: bool = False,
) -> SpecPayload:
    """Accept or migrate one manifest-embedded structured spec payload.

    ``sha256`` is the content hash of the canonical, post-migration inline
    payload. If migration changes the inline payload, ``source_sha256`` retains
    the original inline hash for provenance.
    """
    from feedbax.contracts.graph import GRAPH_SPEC_SCHEMA_ID
    from feedbax.contracts.migrations import (
        UnknownSpecFamily,
        UnsupportedSpecVersion,
        default_spec_registry,
        migrate_graph_spec,
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
        if spec_payload_obj.kind == "GraphSpec" and expected_schema_id == GRAPH_SPEC_SCHEMA_ID:
            result = migrate_graph_spec(
                spec_payload_obj.inline,
                source_version=source_version,
                path=path,
                registry=active_registry,
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


def collect_git_provenance(cwd: Path | str | None = None) -> Provenance:
    """Collect best-effort local Git provenance without mutating repository state."""
    repo_cwd = Path(cwd) if cwd is not None else Path.cwd()

    def _git(*args: str) -> Optional[str]:
        try:
            proc = subprocess.run(
                ["git", *args],
                cwd=repo_cwd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return proc.stdout.strip() or None

    status = _git("status", "--porcelain")
    return Provenance(
        source_repo=_git("config", "--get", "remote.origin.url"),
        source_branch=_git("rev-parse", "--abbrev-ref", "HEAD"),
        source_commit=_git("rev-parse", "HEAD"),
        dirty=(bool(status) if status is not None else None),
    )


def _artifact_path(root: Path, digest: str, suffix: str = "") -> Path:
    return root / "artifacts" / "sha256" / digest[:2] / f"{digest}{suffix}"


class ArtifactStoreSecurityError(RuntimeError):
    """Raised when the local artifact store cannot preserve secure CAS semantics."""


class ArtifactStoreIntegrityError(ArtifactStoreSecurityError):
    """Raised when existing content-addressed bytes do not match their identity."""


def _require_secure_artifact_store_capabilities() -> None:
    required_constants = ("O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK")
    missing = [name for name in required_constants if not getattr(os, name, 0)]
    dir_fd_functions = (os.open, os.mkdir, os.link, os.stat, os.unlink)
    supports_dir_fd = getattr(os, "supports_dir_fd", set())
    missing.extend(
        function.__name__ for function in dir_fd_functions if function not in supports_dir_fd
    )
    supports_follow_symlinks = getattr(os, "supports_follow_symlinks", set())
    if os.stat not in supports_follow_symlinks:
        missing.append("stat(follow_symlinks=False)")
    if os.link not in supports_follow_symlinks:
        missing.append("link(follow_symlinks=False)")
    if missing:
        raise ArtifactStoreSecurityError(
            "secure artifact storage requires descriptor-relative no-follow filesystem "
            "operations; unavailable: " + ", ".join(sorted(set(missing)))
        )


def _secure_directory_flags() -> int:
    _require_secure_artifact_store_capabilities()
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)


def _secure_file_flags(*, writable: bool = False) -> int:
    flags = os.O_RDWR if writable else os.O_RDONLY
    return flags | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)


def _canonicalize_trusted_system_aliases(path: Path) -> Path:
    """Canonicalize only Darwin's fixed first-level /tmp and /var aliases."""
    absolute_path = Path(os.path.abspath(path))
    if sys.platform != "darwin" or len(absolute_path.parts) < 2:
        return absolute_path
    alias_name = absolute_path.parts[1]
    expected = {
        "tmp": (Path("/private/tmp"), {"private/tmp", "/private/tmp"}),
        "var": (Path("/private/var"), {"private/var", "/private/var"}),
    }.get(alias_name)
    if expected is None:
        return absolute_path
    canonical_prefix, allowed_targets = expected
    alias_path = Path(absolute_path.anchor) / alias_name
    try:
        alias_stat = alias_path.lstat()
        alias_target = os.readlink(alias_path)
    except OSError:
        return absolute_path
    if not stat.S_ISLNK(alias_stat.st_mode) or alias_target not in allowed_targets:
        return absolute_path
    return canonical_prefix.joinpath(*absolute_path.parts[2:])


def _open_secure_directory_chain(
    directory: Path,
    *,
    create: bool,
) -> list[tuple[Path, int, os.stat_result]]:
    absolute_directory = _canonicalize_trusted_system_aliases(directory)
    anchor = Path(absolute_directory.anchor)
    if not anchor.anchor:
        raise ArtifactStoreSecurityError(
            f"artifact store directory must resolve to an absolute path: {directory}"
        )

    records: list[tuple[Path, int, os.stat_result]] = []
    flags = _secure_directory_flags()
    try:
        descriptor = os.open(anchor, flags)
        anchor_stat = os.fstat(descriptor)
        if not stat.S_ISDIR(anchor_stat.st_mode):
            raise ArtifactStoreSecurityError(f"artifact store anchor is not a directory: {anchor}")
        records.append((anchor, descriptor, anchor_stat))
        current_path = anchor
        for component in absolute_directory.parts[1:]:
            current_path = current_path / component
            try:
                next_descriptor = os.open(component, flags, dir_fd=records[-1][1])
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(component, mode=0o777, dir_fd=records[-1][1])
                except FileExistsError:
                    pass
                next_descriptor = os.open(component, flags, dir_fd=records[-1][1])
            next_stat = os.fstat(next_descriptor)
            if not stat.S_ISDIR(next_stat.st_mode):
                os.close(next_descriptor)
                raise ArtifactStoreSecurityError(
                    f"artifact store component is not a directory: {current_path}"
                )
            records.append((current_path, next_descriptor, next_stat))
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            error = ArtifactStoreSecurityError(
                f"artifact store directory traverses a symlink or non-directory: {directory}"
            )
            for _, descriptor, _ in reversed(records):
                os.close(descriptor)
            raise error from exc
        for _, descriptor, _ in reversed(records):
            os.close(descriptor)
        raise
    except Exception:
        for _, descriptor, _ in reversed(records):
            os.close(descriptor)
        raise
    return records


def _recheck_secure_directory_chain(
    records: list[tuple[Path, int, os.stat_result]],
) -> None:
    for path, descriptor, initial_stat in records:
        descriptor_stat = os.fstat(descriptor)
        try:
            path_stat = os.stat(path, follow_symlinks=False)
        except FileNotFoundError as exc:
            raise ArtifactStoreSecurityError(
                f"artifact store directory disappeared during write: {path}"
            ) from exc
        expected_identity = (initial_stat.st_dev, initial_stat.st_ino)
        if (
            not stat.S_ISDIR(path_stat.st_mode)
            or (descriptor_stat.st_dev, descriptor_stat.st_ino) != expected_identity
            or (path_stat.st_dev, path_stat.st_ino) != expected_identity
        ):
            raise ArtifactStoreSecurityError(
                f"artifact store directory identity changed during write: {path}"
            )


def _close_secure_directory_chain(
    records: list[tuple[Path, int, os.stat_result]],
) -> None:
    for _, descriptor, _ in reversed(records):
        os.close(descriptor)


def _write_all_bytes(file_descriptor: int, data: bytes) -> None:
    remaining = memoryview(data)
    while remaining:
        written = os.write(file_descriptor, remaining)
        if written <= 0:
            raise ArtifactStoreSecurityError("artifact store write made no progress")
        remaining = remaining[written:]


def _read_all_bytes(file_descriptor: int) -> bytes:
    os.lseek(file_descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while chunk := os.read(file_descriptor, 1024 * 1024):
        chunks.append(chunk)
    return b"".join(chunks)


def _link_artifact_file(
    temporary_name: str,
    final_name: str,
    *,
    temporary_parent_descriptor: int,
    parent_descriptor: int,
) -> None:
    os.link(
        temporary_name,
        final_name,
        src_dir_fd=temporary_parent_descriptor,
        dst_dir_fd=parent_descriptor,
        follow_symlinks=False,
    )


def _open_artifact_staging_container(
    *,
    parent_descriptor: int,
) -> int:
    """Open the fixed descriptor-pinned private staging container.

    POSIX has no conditional unlink-by-inode operation. Keeping the temporary
    name inside an owned mode-0700 container gives this operation exclusive
    name mutation. The public container name is never removed, so a replacement
    cannot be deleted during cleanup.
    """
    directory_name = ".feedbax-artifact-staging"
    try:
        os.mkdir(directory_name, mode=0o700, dir_fd=parent_descriptor)
    except FileExistsError:
        pass
    directory_descriptor: int | None = None
    try:
        directory_descriptor = os.open(
            directory_name,
            _secure_directory_flags(),
            dir_fd=parent_descriptor,
        )
        descriptor_stat = os.fstat(directory_descriptor)
        path_stat = os.stat(
            directory_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(descriptor_stat.st_mode)
            or descriptor_stat.st_uid != os.geteuid()
            or stat.S_IMODE(descriptor_stat.st_mode) & 0o077
            or (descriptor_stat.st_dev, descriptor_stat.st_ino)
            != (path_stat.st_dev, path_stat.st_ino)
        ):
            raise ArtifactStoreSecurityError(
                "artifact staging container must be owned, mode-0700, and stable"
            )
        return directory_descriptor
    except Exception:
        if directory_descriptor is not None:
            os.close(directory_descriptor)
        raise


def _remove_private_artifact_staging_name(
    *,
    directory_descriptor: int,
    temporary_name: str,
) -> None:
    """Remove one unguessable name from the pinned private container."""
    try:
        os.unlink(temporary_name, dir_fd=directory_descriptor)
    except FileNotFoundError:
        pass
    os.close(directory_descriptor)


def _secure_store_bytes_artifact(
    data: bytes,
    *,
    destination: Path,
) -> os.stat_result:
    records = _open_secure_directory_chain(destination.parent, create=True)
    parent_descriptor = records[-1][1]
    final_name = destination.name
    temporary_name = f"payload-{uuid.uuid4().hex}"
    staging_descriptor: int | None = None
    temporary_descriptor: int | None = None
    final_descriptor: int | None = None
    try:
        staging_descriptor = _open_artifact_staging_container(parent_descriptor=parent_descriptor)
        temporary_descriptor = os.open(
            temporary_name,
            _secure_file_flags(writable=True) | os.O_CREAT | os.O_EXCL,
            0o666,
            dir_fd=staging_descriptor,
        )
        _write_all_bytes(temporary_descriptor, data)
        os.fsync(temporary_descriptor)
        if _read_all_bytes(temporary_descriptor) != data:
            raise ArtifactStoreIntegrityError(
                f"artifact temporary bytes failed verification: {destination}"
            )

        try:
            _link_artifact_file(
                temporary_name,
                final_name,
                temporary_parent_descriptor=staging_descriptor,
                parent_descriptor=parent_descriptor,
            )
        except FileExistsError:
            pass
        _remove_private_artifact_staging_name(
            directory_descriptor=staging_descriptor,
            temporary_name=temporary_name,
        )
        staging_descriptor = None

        for attempt in range(101):
            final_descriptor = os.open(
                final_name,
                _secure_file_flags(),
                dir_fd=parent_descriptor,
            )
            final_stat_before = os.fstat(final_descriptor)
            if not stat.S_ISREG(final_stat_before.st_mode) or final_stat_before.st_nlink == 1:
                break
            os.close(final_descriptor)
            final_descriptor = None
            if attempt == 100:
                raise ArtifactStoreIntegrityError(
                    f"canonical artifact has mutable hard-link aliases: {destination}"
                )
            time.sleep(0.001)
        if final_descriptor is None:  # pragma: no cover - loop either opens or raises.
            raise ArtifactStoreIntegrityError(
                f"canonical artifact could not be opened securely: {destination}"
            )
        if not stat.S_ISREG(final_stat_before.st_mode):
            raise ArtifactStoreIntegrityError(
                f"canonical artifact is not a regular file: {destination}"
            )
        stored_data = _read_all_bytes(final_descriptor)
        final_stat_after = os.fstat(final_descriptor)
        if (
            (final_stat_before.st_dev, final_stat_before.st_ino)
            != (final_stat_after.st_dev, final_stat_after.st_ino)
            or final_stat_before.st_size != final_stat_after.st_size
            or stored_data != data
        ):
            raise ArtifactStoreIntegrityError(
                f"canonical artifact bytes do not match content identity: {destination}"
            )
        path_stat = os.stat(final_name, dir_fd=parent_descriptor, follow_symlinks=False)
        if (path_stat.st_dev, path_stat.st_ino) != (
            final_stat_after.st_dev,
            final_stat_after.st_ino,
        ):
            raise ArtifactStoreIntegrityError(
                f"canonical artifact identity changed during write: {destination}"
            )
        _recheck_secure_directory_chain(records)
        return final_stat_after
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise ArtifactStoreSecurityError(
                f"canonical artifact path traverses a symlink or non-directory: {destination}"
            ) from exc
        raise
    finally:
        if temporary_descriptor is not None:
            os.close(temporary_descriptor)
        if final_descriptor is not None:
            os.close(final_descriptor)
        if staging_descriptor is not None:
            try:
                _remove_private_artifact_staging_name(
                    directory_descriptor=staging_descriptor,
                    temporary_name=temporary_name,
                )
            except OSError:
                # Preserve an exceptional private orphan for diagnosis. Never
                # widen cleanup to a public canonical name.
                os.close(staging_descriptor)
        _close_secure_directory_chain(records)


def store_artifact(
    source_path: Path | str,
    *,
    root: Path | str | None = None,
    role: str,
    logical_name: Optional[str] = None,
    media_type: str = "application/octet-stream",
    metadata: Optional[dict[str, Any]] = None,
) -> ArtifactRef:
    """Copy an artifact into the local content-addressed store and return its ref."""
    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(source)
    root_path = Path(root) if root is not None else default_manifest_root()
    digest = sha256_file(source)
    dest = _artifact_path(root_path, digest, source.suffix)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        shutil.copy2(source, dest)
    stat = dest.stat()
    artifact_metadata = dict(metadata or {})
    artifact_metadata.setdefault("original_uri", str(source))
    artifact_metadata.setdefault("relative_path", str(dest.relative_to(root_path)))
    return ArtifactRef(
        role=role,
        logical_name=logical_name or source.name,
        artifact_id=f"artifact://sha256/{digest}",
        sha256=digest,
        media_type=media_type,
        size_bytes=stat.st_size,
        uri=str(dest),
        metadata=artifact_metadata,
    )


def store_json_artifact(
    value: Any,
    *,
    root: Path | str | None = None,
    role: str,
    logical_name: str,
    metadata: Optional[dict[str, Any]] = None,
) -> ArtifactRef:
    """Write stable JSON into the local content-addressed store."""
    root_path = Path(root) if root is not None else default_manifest_root()
    data = json.dumps(value, indent=2, sort_keys=True).encode() + b"\n"
    digest = sha256_bytes(data)
    dest = _artifact_path(root_path, digest, ".json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        dest.write_bytes(data)
    stat = dest.stat()
    artifact_metadata = dict(metadata or {})
    artifact_metadata.setdefault("relative_path", str(dest.relative_to(root_path)))
    return ArtifactRef(
        role=role,
        logical_name=logical_name,
        artifact_id=f"artifact://sha256/{digest}",
        sha256=digest,
        media_type="application/json",
        size_bytes=stat.st_size,
        uri=str(dest),
        metadata=artifact_metadata,
    )


def store_bytes_artifact(
    data: bytes,
    *,
    root: Path | str | None = None,
    role: str,
    logical_name: str,
    media_type: str = "application/octet-stream",
    suffix: str = "",
    metadata: Optional[dict[str, Any]] = None,
) -> ArtifactRef:
    """Atomically write opaque bytes into the local content-addressed store.

    The canonical name is published only after the exact temporary bytes are
    flushed and verified. Platforms without descriptor-relative, no-follow
    operations fail closed with :class:`ArtifactStoreSecurityError`.
    """
    if not isinstance(data, bytes):
        raise TypeError("artifact data must be bytes")
    if not isinstance(suffix, str) or "\0" in suffix or Path(f"x{suffix}").name != f"x{suffix}":
        raise ValueError("artifact suffix must not contain path components")
    root_path = Path(root) if root is not None else default_manifest_root()
    digest = sha256_bytes(data)
    dest = _artifact_path(root_path, digest, suffix)
    destination = Path(os.path.abspath(dest))
    artifact_stat = _secure_store_bytes_artifact(data, destination=destination)
    artifact_metadata = dict(metadata or {})
    artifact_metadata.setdefault("relative_path", str(dest.relative_to(root_path)))
    return ArtifactRef(
        role=role,
        logical_name=logical_name,
        artifact_id=f"artifact://sha256/{digest}",
        sha256=digest,
        media_type=media_type,
        size_bytes=artifact_stat.st_size,
        uri=str(dest),
        metadata=artifact_metadata,
    )


def _validate_retention_artifact_version(
    role: str,
    payload: dict[str, Any],
    *,
    path: str,
) -> dict[str, Any]:
    """Validate or stamp a governed retention artifact payload."""
    from feedbax.contracts.migrations import UnsupportedSpecVersion, default_spec_registry

    kind, expected_schema_id, current_version = retention_artifact_schema(role)
    schema_id = payload.get("schema_id")
    if schema_id is not None and schema_id != expected_schema_id:
        raise UnsupportedSpecVersion(
            "Unsupported retention artifact schema identity: "
            f"path={path!r}, role={role!r}, kind={kind!r}, "
            f"schema_id={schema_id!r}, expected={expected_schema_id!r}"
        )

    source_version = payload.get("schema_version")
    if source_version is not None and not isinstance(source_version, str):
        raise UnsupportedSpecVersion(
            "Retention artifact schema_version must be a string: "
            f"path={path!r}, role={role!r}, kind={kind!r}, "
            f"schema_version={source_version!r}"
        )
    if isinstance(source_version, str) and source_version and source_version != current_version:
        try:
            default_spec_registry.migrate(kind, payload, source_version=source_version)
        except UnsupportedSpecVersion as exc:
            raise UnsupportedSpecVersion(
                "Unsupported retention artifact schema version: "
                f"path={path!r}, role={role!r}, kind={kind!r}; {exc}"
            ) from exc

    stamped = dict(payload)
    stamped["schema_id"] = expected_schema_id
    stamped["schema_version"] = current_version
    return stamped


def _retention_artifact_payload(
    role: str,
    value: Any,
    *,
    path: str,
) -> dict[str, Any]:
    if role == "retained_observables":
        if (
            isinstance(value, dict)
            and ("schema_id" in value or "schema_version" in value)
            and "observables" in value
        ):
            payload = dict(value)
        else:
            payload = retained_observables_to_json(value)
    elif role == "retention_plan":
        if not isinstance(value, dict):
            raise TypeError(
                "retention_plan artifact payload must be a mapping: "
                f"path={path!r}, got={type(value).__name__}"
            )
        payload = dict(value)
    else:
        payload = value
    if not isinstance(payload, dict):
        raise TypeError(
            "retention artifact payload must be a mapping after schema wrapping: "
            f"path={path!r}, role={role!r}, got={type(payload).__name__}"
        )
    return _validate_retention_artifact_version(role, payload, path=path)


def _validate_retention_artifact_ref_metadata(data: dict[str, Any]) -> dict[str, Any]:
    artifacts = data.get("artifacts")
    if not isinstance(artifacts, list):
        return data
    normalized = dict(data)
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict):
            continue
        role = artifact.get("role")
        if role not in RETENTION_ARTIFACT_ROLE_SCHEMAS:
            continue
        metadata = artifact.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        missing = [
            key
            for key in ("schema_id", "schema_version")
            if not isinstance(metadata.get(key), str) or not metadata.get(key)
        ]
        if missing:
            from feedbax.contracts.migrations import UnsupportedSpecVersion

            raise UnsupportedSpecVersion(
                "Retention artifact ref is missing governed schema metadata: "
                f"path='artifacts/{index}/metadata', role={role!r}, missing={missing}"
            )
        _validate_retention_artifact_version(
            role,
            {
                "schema_id": metadata["schema_id"],
                "schema_version": metadata["schema_version"],
            },
            path=f"artifacts/{index}/metadata",
        )
    return normalized


def _manifest_dir(root: Path, kind: str) -> Path:
    names = {
        "GraphSpecManifest": "graph_specs",
        "ModelArtifactManifest": "model_artifacts",
        "TrainingRunSetManifest": "training_run_sets",
        "TrainingRunManifest": "training_runs",
        "EvaluationRunManifest": "evaluation_runs",
        "CheckpointSelectionManifest": "checkpoint_selections",
        "AnalysisRunManifest": "analysis_runs",
        "ReportManifest": "reports",
    }
    return root / "manifests" / names.get(kind, kind)


def _safe_manifest_filename(manifest_id: str) -> str:
    safe = manifest_id.replace(":", "_").replace("/", "_")
    return f"{safe}.json"


def safe_manifest_key(manifest_id: str) -> str:
    """Return a filesystem-safe key derived from a manifest identifier."""
    return manifest_id.replace(":", "_").replace("/", "_")


def _is_spec_payload_data(value: Any) -> bool:
    return isinstance(value, dict) and "kind" in value and "inline" in value


def _normalize_spec_payload_field(value: Any, *, path: str) -> Any:
    if value is None:
        return None
    if isinstance(value, list):
        return [
            _normalize_spec_payload_field(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, SpecPayload) or _is_spec_payload_data(value):
        return migrate_spec_payload(value, path=path)
    return value


def _normalize_manifest_data_spec_payloads(data: dict[str, Any]) -> dict[str, Any]:
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
        )
    return normalized


def _normalize_training_run_set_manifest_data(data: dict[str, Any]) -> dict[str, Any]:
    if data.get("kind") != "TrainingRunSetManifest":
        return data
    schema_version = data.get("schema_version", TRAINING_RUN_SET_SCHEMA_VERSION_V1)
    if schema_version == TRAINING_RUN_SET_SCHEMA_VERSION:
        return data
    if schema_version != TRAINING_RUN_SET_SCHEMA_VERSION_V1:
        raise ValueError(
            "Unsupported TrainingRunSetManifest schema_version: "
            f"{schema_version!r}; expected {TRAINING_RUN_SET_SCHEMA_VERSION!r}"
        )
    migrated = dict(data)
    migrated["schema_version"] = TRAINING_RUN_SET_SCHEMA_VERSION
    migrated.setdefault("axes", {})
    records = list(migrated.get("migration_records") or [])
    records.append(
        ArtifactMigrationRecord(
            migration_id="training-run-set-manifest-v1-to-v2-axes",
            source_schema_version=TRAINING_RUN_SET_SCHEMA_VERSION_V1,
            target_schema_version=TRAINING_RUN_SET_SCHEMA_VERSION,
            metadata={
                "description": (
                    "Add an empty run-set axes block for pre-sweep collection manifests."
                )
            },
        ).model_dump(mode="json", exclude_none=True)
    )
    migrated["migration_records"] = records
    return migrated


def _normalize_analysis_run_manifest_data(data: dict[str, Any]) -> dict[str, Any]:
    if data.get("kind") != "AnalysisRunManifest":
        return data
    from feedbax.contracts.migrations import default_spec_registry

    schema_version = data.get("schema_version", ANALYSIS_RUN_MANIFEST_SCHEMA_VERSION_V1)
    result = default_spec_registry.migrate(
        "AnalysisRunManifest",
        data,
        source_version=schema_version if isinstance(schema_version, str) else None,
    )
    return result.payload


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


def evaluation_run_manifest_id(spec: EvaluationRunSpec) -> str:
    """Return deterministic run identity for an evaluation spec."""
    digest = sha256_bytes(canonical_json_bytes(spec))
    return f"feedbax-evaluation-run:{digest[:32]}"


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


def planned_training_run_set_manifest_id(
    *,
    graph_spec: dict[str, Any],
    base_training_spec: dict[str, Any],
    task_spec: dict[str, Any],
    task_binding_spec: dict[str, Any] | None = None,
    axes: dict[str, Any] | None = None,
) -> str:
    """Return deterministic identity for a planned Studio training run set."""
    digest = sha256_bytes(
        canonical_json_bytes(
            {
                "graph_spec": graph_spec,
                "base_training_spec": base_training_spec,
                "task_spec": task_spec,
                "task_binding_spec": task_binding_spec,
                "axes": axes or {},
            }
        )
    )
    return f"feedbax-training-run-set:{digest[:32]}"


def analysis_run_manifest_id(spec: AnalysisRunSpec) -> str:
    """Return deterministic run identity for an analysis spec."""
    digest = sha256_bytes(canonical_json_bytes(spec))
    return f"feedbax-analysis-run:{digest[:32]}"


def report_manifest_id(spec: ReportSpec) -> str:
    """Return deterministic report identity for a report spec."""
    digest = sha256_bytes(canonical_json_bytes(spec))
    return f"feedbax-report:{digest[:32]}"


def figure_manifest_id(spec: Any) -> str:
    """Return deterministic figure identity for a figure spec-like object."""
    digest = sha256_bytes(canonical_json_bytes(spec))
    return f"feedbax-figure:{digest[:32]}"


def checkpoint_selection_manifest_id(spec: CheckpointSelectionSpec) -> str:
    """Return deterministic identity for a checkpoint-selection spec."""
    digest = sha256_bytes(canonical_json_bytes(spec))
    return f"feedbax-checkpoint-selection:{digest[:32]}"


def evaluation_states_cache_path(
    manifest_id: str,
    *,
    root: Path | str | None = None,
) -> Path:
    """Return the manifest-root cache path for evaluated state trajectories."""
    root_path = Path(root) if root is not None else default_manifest_root()
    return root_path / "cache" / "states" / f"{safe_manifest_key(manifest_id)}.pkl"


def analysis_results_cache_dir(
    manifest_id: str,
    *,
    root: Path | str | None = None,
) -> Path:
    """Return the manifest-root cache directory for computed analysis results."""
    root_path = Path(root) if root is not None else default_manifest_root()
    return root_path / "cache" / "analysis_results" / safe_manifest_key(manifest_id)


def write_manifest(
    manifest: AnyManifest,
    *,
    root: Path | str | None = None,
    index: bool = True,
) -> Path:
    """Write a manifest to the local manifest layout and optionally index it."""
    manifest = normalize_manifest_spec_payloads(manifest)
    root_path = Path(root) if root is not None else default_manifest_root()
    manifest_dir = _manifest_dir(root_path, manifest.kind)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_dir / _safe_manifest_filename(manifest.id)
    path.write_text(
        manifest.model_dump_json(indent=2, exclude_none=True) + "\n",
        encoding="utf-8",
    )
    if index:
        from feedbax.persistence.manifest_index import index_manifest_file

        index_manifest_file(path, root=root_path)
    return path


def load_manifest_bytes(raw: bytes) -> AnyManifest:
    """Parse one known Feedbax manifest from already-authenticated raw bytes."""
    data = json.loads(raw)
    data = _normalize_training_run_set_manifest_data(data)
    data = _normalize_analysis_run_manifest_data(data)
    data = _normalize_manifest_data_spec_payloads(data)
    data = _validate_retention_artifact_ref_metadata(data)
    raw_kind = data.get("kind")
    if not isinstance(raw_kind, str):
        raise ValueError(f"Unknown Feedbax manifest kind: {raw_kind!r}")
    kind = raw_kind
    model = _manifest_model_for_kind(kind)
    if model is None:
        raise ValueError(f"Unknown Feedbax manifest kind: {kind!r}")
    return model.model_validate(data)  # type: ignore[return-value]


def load_manifest(path: Path | str) -> AnyManifest:
    """Load a known Feedbax manifest from disk."""
    return load_manifest_bytes(Path(path).read_bytes())


def load_graph_spec_from_manifest(
    manifest: GraphSpecManifest | ModelArtifactManifest | Path | str,
    *,
    root: Path | str | None = None,
) -> GraphSpecLoadResult:
    """Load a manifest-owned GraphSpec while preserving migration-record custody.

    Feedbax GraphSpec migration records stay on the manifest that directly owns
    the inline GraphSpec payload. Downstream legacy-conversion records already
    present on a model artifact remain distinct and are surfaced separately.
    """
    manifest_obj, manifest_path = _load_graph_spec_manifest_source(manifest)
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
        )

    graph_spec = manifest_obj.graph_spec
    if isinstance(graph_spec, ParentRef):
        referenced = _load_parent_graph_spec_manifest(graph_spec, base=base)
        referenced_result = load_graph_spec_from_manifest(referenced, root=base)
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
    )


def _load_graph_spec_manifest_source(
    manifest: GraphSpecManifest | ModelArtifactManifest | Path | str,
) -> tuple[GraphSpecManifest | ModelArtifactManifest, Path | None]:
    if isinstance(manifest, GraphSpecManifest | ModelArtifactManifest):
        return manifest, None
    path = Path(manifest)
    loaded = load_manifest(path)
    if not isinstance(loaded, GraphSpecManifest | ModelArtifactManifest):
        raise TypeError(
            f"Expected GraphSpecManifest or ModelArtifactManifest, got {type(loaded).__name__}."
        )
    return loaded, path


def _load_parent_graph_spec_manifest(parent: ParentRef, *, base: Path) -> GraphSpecManifest:
    if parent.uri is None:
        raise ValueError(
            f"Model artifact graph_spec parent {parent.id!r} has no URI; "
            "GraphSpec migration custody is not discoverable."
        )
    path = Path(parent.uri)
    if not path.is_absolute():
        path = base / path
    loaded = load_manifest(path)
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
) -> GraphSpecLoadResult:
    if payload.kind != "GraphSpec":
        raise TypeError(f"Expected GraphSpec payload, got {payload.kind!r}.")

    migrated_payload = migrate_spec_payload(payload, path="graph_spec")
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


def write_training_run_manifest(
    *,
    job_id: Optional[str],
    run_set_id: Optional[str] = None,
    total_batches: int,
    training_spec: Optional[dict[str, Any]] = None,
    task_spec: Optional[dict[str, Any]] = None,
    task_binding_spec: Optional[dict[str, Any]] = None,
    graph_spec: Optional[dict[str, Any]] = None,
    checkpoint_path: Optional[Path | str] = None,
    history_events: Optional[list[dict[str, Any]]] = None,
    retention_plan: Optional[dict[str, Any]] = None,
    retained_observables: Optional[dict[str, Any] | list[dict[str, Any]]] = None,
    status: ManifestStatus = "completed",
    final_loss: Optional[float] = None,
    root: Path | str | None = None,
    provenance: Optional[Provenance] = None,
    issues: Optional[list[str]] = None,
) -> tuple[TrainingRunManifest, Path]:
    """Build, store, and index a local training-run manifest."""
    root_path = Path(root) if root is not None else default_manifest_root()
    artifacts: list[ArtifactRef] = []
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        artifacts.append(
            store_artifact(
                checkpoint_path,
                root=root_path,
                role="training_checkpoint",
                logical_name=f"feedbax_checkpoint_{job_id}.eqx" if job_id else None,
                media_type="application/x-equinox",
            )
        )
    if history_events is not None:
        artifacts.append(
            store_json_artifact(
                history_events,
                root=root_path,
                role="training_history",
                logical_name=f"feedbax_training_history_{job_id}.json"
                if job_id
                else "feedbax_training_history.json",
            )
        )
    if retention_plan is not None:
        artifacts.append(
            store_json_artifact(
                _retention_artifact_payload(
                    "retention_plan",
                    retention_plan,
                    path="retention_plan",
                ),
                root=root_path,
                role="retention_plan",
                logical_name=f"feedbax_retention_plan_{job_id}.json"
                if job_id
                else "feedbax_retention_plan.json",
                metadata=retention_artifact_metadata("retention_plan"),
            )
        )
    if retained_observables is not None:
        artifacts.append(
            store_json_artifact(
                _retention_artifact_payload(
                    "retained_observables",
                    retained_observables,
                    path="retained_observables",
                ),
                root=root_path,
                role="retained_observables",
                logical_name=f"feedbax_retained_observables_{job_id}.json"
                if job_id
                else "feedbax_retained_observables.json",
                metadata=retention_artifact_metadata("retained_observables"),
            )
        )

    prov = provenance or collect_git_provenance()
    if issues:
        prov.issues.extend(issue for issue in issues if issue not in prov.issues)
    if prov.entrypoint is None:
        prov.entrypoint = EntrypointRef(kind="feedbax-worker", name="training")

    manifest = TrainingRunManifest(
        id=training_run_manifest_id(job_id),
        job_id=job_id,
        run_set_id=run_set_id,
        status=status,
        completed_at=utc_now() if status in {"completed", "failed", "cancelled"} else None,
        graph_spec=spec_payload("GraphSpec", graph_spec) if graph_spec is not None else None,
        training_spec=spec_payload("TrainingSpec", training_spec)
        if training_spec is not None
        else None,
        task_spec=spec_payload("TaskSpec", task_spec) if task_spec is not None else None,
        task_binding_spec=spec_payload("StudioTaskBindingSpec", task_binding_spec)
        if task_binding_spec is not None
        else None,
        summary_metrics={
            key: value
            for key, value in {
                "final_loss": final_loss,
                "total_batches": total_batches,
            }.items()
            if value is not None
        },
        provenance=prov,
        artifacts=artifacts,
    )
    return manifest, write_manifest(manifest, root=root_path)
