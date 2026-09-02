"""Durable run-bundle contract for Feedbax orchestration."""

from __future__ import annotations

import hashlib
import os
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TypeAlias

from pydantic import Field, field_validator, model_validator

from feedbax.contracts.artifact_custody import ImmutableArtifactBlobProviderSpec
from feedbax.contracts.canonical_json import canonical_json_v1_bytes
from feedbax.contracts.evaluation_preflight import EvaluationOutputPreflightEvidence
from feedbax.contracts.manifest import ArtifactMigrationRecord, ParentRef, StrictModel
from feedbax.contracts.run_matrix import TrainingRowProvenance
from feedbax.contracts.staged_execution import validate_staged_binding_name
from feedbax.contracts.spec_storage import (
    canonicalize_immutable_input_identities,
    training_run_execution_hash,
    validate_sha256,
)
from feedbax.orchestration.staged_root_custody import StagedRootCustody


RUN_BUNDLE_SCHEMA_ID = "feedbax.orchestration.run_bundle"
RUN_BUNDLE_SCHEMA_VERSION_V1 = "feedbax.orchestration.run_bundle.v1"
RUN_BUNDLE_SCHEMA_VERSION_V2 = "feedbax.orchestration.run_bundle.v2"
RUN_BUNDLE_SCHEMA_VERSION_V3 = "feedbax.orchestration.run_bundle.v3"
RUN_BUNDLE_SCHEMA_VERSION_V4 = "feedbax.orchestration.run_bundle.v4"
RUN_BUNDLE_SCHEMA_VERSION_V5 = "feedbax.orchestration.run_bundle.v5"
RUN_BUNDLE_SCHEMA_VERSION_V6 = "feedbax.orchestration.run_bundle.v6"
RUN_BUNDLE_SCHEMA_VERSION_V7 = "feedbax.orchestration.run_bundle.v7"
RUN_BUNDLE_SCHEMA_VERSION_V8 = "feedbax.orchestration.run_bundle.v8"
RUN_BUNDLE_SCHEMA_VERSION_V9 = "feedbax.orchestration.run_bundle.v9"
RUN_BUNDLE_SCHEMA_VERSION_V10 = "feedbax.orchestration.run_bundle.v10"
RUN_BUNDLE_SCHEMA_VERSION_V11 = "feedbax.orchestration.run_bundle.v11"
RUN_BUNDLE_SCHEMA_VERSION = "feedbax.orchestration.run_bundle.v12"
DEPLOYMENT_POLICY_SCHEMA_ID = "feedbax.spec.deployment_policy"
DEPLOYMENT_POLICY_SCHEMA_VERSION_V1 = "feedbax.spec.deployment_policy.v1"
DEPLOYMENT_POLICY_SCHEMA_VERSION = "feedbax.spec.deployment_policy.v2"
EXECUTION_IDENTITY_ENVELOPE_SCHEMA_ID = "feedbax.spec.execution_identity_envelope"
EXECUTION_IDENTITY_ENVELOPE_SCHEMA_VERSION_V1 = "feedbax.spec.execution_identity_envelope.v1"
EXECUTION_IDENTITY_ENVELOPE_SCHEMA_VERSION = "feedbax.spec.execution_identity_envelope.v2"
ROW_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
CHECKPOINT_CUSTODY_ARCHIVE_FORMAT_ID = "feedbax.archive.training_checkpoint_custody"
CHECKPOINT_CUSTODY_ARCHIVE_FORMAT_VERSION = "feedbax.archive.training_checkpoint_custody.v1"
CHECKPOINT_CUSTODY_ARCHIVE_MEDIA_TYPE = (
    "application/vnd.feedbax.training-checkpoint-custody.v1+tar+gzip"
)
CHECKPOINT_CUSTODY_ARCHIVE_MATERIALIZER_ID = (
    "feedbax.materializer.training_checkpoint_custody_archive"
)
CHECKPOINT_CUSTODY_ARCHIVE_MATERIALIZER_VERSION = (
    "feedbax.materializer.training_checkpoint_custody_archive.v1"
)
ExecutionFamily: TypeAlias = Literal["native-training", "evaluation-matrix"]


def mint_run_set_id(now: datetime | None = None) -> str:
    """Mint ``<utc-date>-<8-hex>`` run-set identity."""
    timestamp = now or datetime.now(timezone.utc)
    return f"{timestamp.astimezone(timezone.utc).date().isoformat()}-{secrets.token_hex(4)}"


def default_orchestration_root(run_set_id: str) -> Path:
    """Return the default orchestration root for a run set."""
    configured = os.environ.get("FEEDBAX_ORCHESTRATION_ROOT")
    if configured:
        return Path(configured).expanduser() / run_set_id
    return Path.home() / ".cache" / "feedbax" / "orchestration" / run_set_id


def canonical_run_bundle_sha256(bundle: "RunBundle") -> str:
    """Return the SHA-256 identity of the persisted run-bundle representation."""
    payload = bundle.model_dump(mode="json", exclude_none=True)
    encoded = canonical_json_v1_bytes(payload)
    return hashlib.sha256(encoded).hexdigest()


class SchemaArtifactRef(StrictModel):
    """Immutable registry reference to a governed structured payload."""

    schema_id: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)
    artifact_id: str = Field(min_length=1)
    sha256: str
    uri: str | None = None

    @field_validator("sha256")
    @classmethod
    def _validate_sha256(cls, value: str) -> str:
        validate_sha256(value, field_name="sha256")
        return value

    @field_validator("artifact_id")
    @classmethod
    def _reject_mutable_locator(cls, value: str) -> str:
        if "latest.json" in value:
            raise ValueError("artifact_id must be an immutable registry-resolvable locator")
        return value


class IdentityArtifactRef(SchemaArtifactRef):
    """Base immutable artifact reference carrying a separate semantic hash."""


class AuthoredIntentRef(IdentityArtifactRef):
    """Authored document reference and its family-defined intent identity."""

    intent_hash: str

    @field_validator("intent_hash")
    @classmethod
    def _validate_intent_hash(cls, value: str) -> str:
        validate_sha256(value, field_name="intent_hash")
        return value


class ResolvedSnapshotRef(IdentityArtifactRef):
    """Resolved-semantics snapshot reference and decoded root identity."""

    root_hash: str

    @field_validator("root_hash")
    @classmethod
    def _validate_root_hash(cls, value: str) -> str:
        validate_sha256(value, field_name="root_hash")
        return value


class ExecutionCapsuleRef(IdentityArtifactRef):
    """Execution capsule reference and derived execution identity."""

    execution_hash: str

    @field_validator("execution_hash")
    @classmethod
    def _validate_execution_hash(cls, value: str) -> str:
        validate_sha256(value, field_name="execution_hash")
        return value


class ImmutableInputDigest(StrictModel):
    """Digest identifying immutable external input bytes."""

    algorithm: Literal["sha256"] = "sha256"
    value: str

    @field_validator("value")
    @classmethod
    def _validate_value(cls, value: str) -> str:
        validate_sha256(value, field_name="digest.value")
        return value


class ImmutableInputIdentity(StrictModel):
    """Canonical identity for one external execution input."""

    role: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    identifier: str = Field(min_length=1)
    digest: ImmutableInputDigest
    schema_id: str | None = None
    schema_version: str | None = None

    @model_validator(mode="after")
    def _validate_schema_pair(self) -> "ImmutableInputIdentity":
        if (self.schema_id is None) != (self.schema_version is None):
            raise ValueError("schema_id and schema_version must be supplied together")
        if "latest.json" in self.identifier:
            raise ValueError("immutable input identifier must not reference latest.json")
        return self


class ExecutionIdentityEnvelope(StrictModel):
    """Immutable scientific identity committed by ASSEMBLE for one row."""

    schema_id: Literal["feedbax.spec.execution_identity_envelope"] = (
        EXECUTION_IDENTITY_ENVELOPE_SCHEMA_ID
    )
    schema_version: Literal["feedbax.spec.execution_identity_envelope.v2"] = (
        EXECUTION_IDENTITY_ENVELOPE_SCHEMA_VERSION
    )
    payload: SchemaArtifactRef
    authored_intent: AuthoredIntentRef
    resolved_snapshot: ResolvedSnapshotRef
    execution_capsule: ExecutionCapsuleRef
    immutable_inputs: list[ImmutableInputIdentity]
    row_provenance: TrainingRowProvenance | None = None

    @model_validator(mode="after")
    def _validate_execution_identity(self) -> "ExecutionIdentityEnvelope":
        canonical = canonicalize_immutable_input_identities(self.immutable_inputs)
        supplied = [
            identity.model_dump(mode="json", exclude_none=True)
            for identity in self.immutable_inputs
        ]
        if supplied != canonical:
            raise ValueError(
                "immutable_inputs must be in canonical (role, kind, identifier, digest) order"
            )
        expected = training_run_execution_hash(self.resolved_snapshot.root_hash, canonical)
        if self.execution_capsule.execution_hash != expected:
            raise ValueError(
                "execution_hash does not match resolved root and canonical immutable inputs; "
                f"envelope={self.execution_capsule.execution_hash!r}, computed={expected!r}"
            )
        if (
            self.row_provenance is not None
            and self.row_provenance.lowered_execution_payload_hash != self.payload.sha256
        ):
            raise ValueError(
                "row provenance lowered_execution_payload_hash does not match the "
                "custodied execution payload"
            )
        return self


def execution_identity_projection(
    envelope: ExecutionIdentityEnvelope,
) -> dict[str, Any]:
    """Project one execution envelope onto durable manifest identity fields."""
    return {
        "intent_hash": envelope.authored_intent.intent_hash,
        "resolved_semantics_root_hash": envelope.resolved_snapshot.root_hash,
        "execution_hash": envelope.execution_capsule.execution_hash,
        "input_data_identities": canonicalize_immutable_input_identities(envelope.immutable_inputs),
    }


class RowLaunchSpec(StrictModel):
    """Operational launch instructions, excluded from scientific identity."""

    command: list[str] = Field(default_factory=list)
    entry: str | None = None
    collect: list[str] = Field(default_factory=list)
    payload_routing: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_launch_entry(self) -> "RowLaunchSpec":
        if not self.command and not self.entry:
            raise ValueError("row launch requires command or entry")
        return self


class RunRowSpec(StrictModel):
    """One compiled row in a run bundle."""

    row_id: str = Field(min_length=1)
    execution_family: ExecutionFamily = "native-training"
    execution: ExecutionIdentityEnvelope
    launch: RowLaunchSpec

    @field_validator("row_id")
    @classmethod
    def _validate_row_id(cls, value: str) -> str:
        if not ROW_ID_RE.fullmatch(value):
            raise ValueError("row_id must match ^[A-Za-z0-9_.-]+$")
        return value

    @model_validator(mode="after")
    def _validate_provenance_row(self) -> "RunRowSpec":
        provenance = self.execution.row_provenance
        if provenance is not None and provenance.row_id != self.row_id:
            raise ValueError("run-row provenance row_id must match row_id")
        return self


class RepoRevision(StrictModel):
    """Repository revision declaration used in environment fingerprints."""

    path: str = "."
    revision: str
    dirty_allowed: bool = False


class EnvironmentDeclaration(StrictModel):
    """Declared execution environment for a run bundle."""

    python_version: str | None = None
    repo_revisions: list[RepoRevision] = Field(default_factory=list)
    lockfile_hashes: dict[str, str] = Field(default_factory=dict)
    overlay_steps: list[str] = Field(default_factory=list)
    image_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def environment_declaration_identity_projection(
    environment: EnvironmentDeclaration,
) -> dict[str, Any]:
    """Project the declared fields that determine environment identity."""
    return {
        "python_version": environment.python_version,
        "repo_revisions": [
            revision.model_dump(mode="json", exclude_none=True)
            for revision in environment.repo_revisions
        ],
        "lockfile_hashes": dict(sorted(environment.lockfile_hashes.items())),
        "overlay_steps": list(environment.overlay_steps),
        "image_id": environment.image_id,
    }


class DeploymentResourceRequest(StrictModel):
    """Requested resources, distinct from provider-realized resource facts."""

    gpu_id: str | None = Field(default=None, min_length=1)
    regions: list[str] = Field(default_factory=list)

    @field_validator("regions")
    @classmethod
    def _validate_regions(cls, value: list[str]) -> list[str]:
        if any(not region.strip() for region in value):
            raise ValueError("requested regions must be non-empty strings")
        if len(value) != len(set(value)):
            raise ValueError("requested regions must be unique and remain in preference order")
        return value


class DeploymentPolicy(StrictModel):
    """Versioned requested launch venue, authorization, and resource policy."""

    schema_id: Literal["feedbax.spec.deployment_policy"] = DEPLOYMENT_POLICY_SCHEMA_ID
    schema_version: Literal["feedbax.spec.deployment_policy.v2"] = DEPLOYMENT_POLICY_SCHEMA_VERSION
    driver: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9._-]*(?::[a-z0-9][a-z0-9._-]*)*$")
    venue: Literal["local", "remote"]
    cloud_authorized: bool
    review_required: bool
    review_authorized: bool
    resources: DeploymentResourceRequest = Field(default_factory=DeploymentResourceRequest)


class LaunchPolicy(StrictModel):
    """Row launch concurrency policy."""

    max_parallel_rows: int = Field(default=1, ge=1)
    warm_first: bool = False
    stagger_seconds: float = Field(default=0.0, ge=0.0)


class BudgetPolicy(StrictModel):
    """Run-set budget guards enforced by MONITOR."""

    max_wall_clock_seconds: float = Field(gt=0.0)
    max_spend_usd: float | None = Field(default=None, ge=0.0)


class ImmutableInputArtifactRef(StrictModel):
    """Immutable provider artifact containing bytes staged for an input."""

    artifact_id: str = Field(min_length=1)
    sha256: str
    size_bytes: int = Field(ge=0)
    media_type: str = Field(min_length=1)
    storage_backend: str = Field(min_length=1)

    @field_validator("sha256")
    @classmethod
    def _validate_sha256(cls, value: str) -> str:
        validate_sha256(value, field_name="sha256")
        return value

    @field_validator("artifact_id")
    @classmethod
    def _reject_mutable_latest(cls, value: str) -> str:
        if "latest.json" in value:
            raise ValueError("input artifacts must name immutable bytes, not latest.json")
        return value

    @model_validator(mode="after")
    def _bind_artifact_id_to_digest(self) -> "ImmutableInputArtifactRef":
        if self.artifact_id != f"artifact://sha256/{self.sha256}":
            raise ValueError("input artifact_id does not match its SHA-256 digest")
        return self


class InputFormatIdentity(StrictModel):
    """Stable identity for the byte format consumed by an input materializer."""

    format_id: str = Field(min_length=1)
    format_version: str = Field(min_length=1)
    media_type: str = Field(min_length=1)


class CheckpointCustodyArchiveMaterializer(StrictModel):
    """Authenticated materialization authority for a checkpoint-custody archive."""

    kind: Literal["checkpoint-custody-archive"] = "checkpoint-custody-archive"
    materializer_id: Literal["feedbax.materializer.training_checkpoint_custody_archive"] = (
        CHECKPOINT_CUSTODY_ARCHIVE_MATERIALIZER_ID
    )
    materializer_version: Literal["feedbax.materializer.training_checkpoint_custody_archive.v1"] = (
        CHECKPOINT_CUSTODY_ARCHIVE_MATERIALIZER_VERSION
    )
    expected_parent_ref: ParentRef
    expected_transaction_root_sha256: str

    @field_validator("expected_transaction_root_sha256")
    @classmethod
    def _validate_transaction_root(cls, value: str) -> str:
        validate_sha256(value, field_name="expected_transaction_root_sha256")
        return value

    @model_validator(mode="after")
    def _validate_parent_ref(self) -> "CheckpointCustodyArchiveMaterializer":
        ref = self.expected_parent_ref
        if ref.kind != "TrainingCheckpointTransactionManifest":
            raise ValueError("checkpoint archive ParentRef has the wrong kind")
        if ref.role != "training_checkpoint_custody":
            raise ValueError("checkpoint archive ParentRef has the wrong role")
        if not ref.id:
            raise ValueError("checkpoint archive ParentRef requires a transaction id")
        if (
            not ref.uri
            or "latest.json" in ref.uri
            or ref.uri.startswith("/")
            or "://" in ref.uri
            or "\\" in ref.uri
            or any(part in {"", ".", ".."} for part in ref.uri.split("/"))
        ):
            raise ValueError("checkpoint archive ParentRef requires an immutable manifest uri")
        digest = ref.metadata.get("manifest_sha256")
        if not isinstance(digest, str):
            raise ValueError("checkpoint archive ParentRef requires metadata.manifest_sha256")
        validate_sha256(digest, field_name="expected_parent_ref.metadata.manifest_sha256")
        return self


class InputCustodySource(StrictModel):
    """Driver-neutral authenticated source for one materialized execution input."""

    target_role: str = Field(min_length=1)
    provider: ImmutableArtifactBlobProviderSpec
    provider_binding: str = Field(min_length=1)
    artifact: ImmutableInputArtifactRef
    format: InputFormatIdentity
    materializer: CheckpointCustodyArchiveMaterializer

    @model_validator(mode="after")
    def _validate_checkpoint_archive_contract(self) -> "InputCustodySource":
        validate_staged_binding_name(self.provider_binding)
        if self.artifact.storage_backend != self.provider.config.storage_backend:
            raise ValueError("input artifact storage_backend does not match its provider")
        expected = (
            CHECKPOINT_CUSTODY_ARCHIVE_FORMAT_ID,
            CHECKPOINT_CUSTODY_ARCHIVE_FORMAT_VERSION,
            CHECKPOINT_CUSTODY_ARCHIVE_MEDIA_TYPE,
        )
        observed = (
            self.format.format_id,
            self.format.format_version,
            self.format.media_type,
        )
        if observed != expected:
            raise ValueError("checkpoint custody source has an unsupported format identity")
        if self.artifact.media_type != self.format.media_type:
            raise ValueError("input artifact media_type does not match its format identity")
        return self


class ResolvedAssemblyInput(StrictModel):
    """Canonical scientific identity paired with authenticated staging custody."""

    identity: ImmutableInputIdentity
    custody: InputCustodySource

    @model_validator(mode="after")
    def _bind_identity_to_custody(self) -> "ResolvedAssemblyInput":
        if self.identity.role != self.custody.target_role:
            raise ValueError("input identity role does not match custody target_role")
        if self.identity.digest.value != self.custody.artifact.sha256:
            raise ValueError("input identity digest does not match custody artifact bytes")
        return self


class RunBundle(StrictModel):
    """Schema-versioned orchestration request for a run set.

    ``feedbax_revision`` is copied from the authored
    ``RunAssemblyRequest.feedbax_revision`` after ASSEMBLE has verified it
    against the imported package. It is never minted from whatever package
    happens to be imported, because a stale install would then simply mint its
    own revision and satisfy every later check against itself.
    """

    schema_id: Literal["feedbax.orchestration.run_bundle"] = RUN_BUNDLE_SCHEMA_ID
    schema_version: Literal["feedbax.orchestration.run_bundle.v12"] = RUN_BUNDLE_SCHEMA_VERSION
    run_set_id: str = Field(default_factory=mint_run_set_id)
    feedbax_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    execution_family: ExecutionFamily = "native-training"
    deployment_policy: DeploymentPolicy
    migration_evidence: list[ArtifactMigrationRecord] = Field(default_factory=list)
    rows: list[RunRowSpec] = Field(min_length=1)
    environment: EnvironmentDeclaration
    launch_policy: LaunchPolicy = Field(default_factory=LaunchPolicy)
    budget: BudgetPolicy
    resolved_inputs: list[ResolvedAssemblyInput] = Field(default_factory=list)
    staged_roots: list[StagedRootCustody] = Field(default_factory=list)
    evaluation_output_preflight: EvaluationOutputPreflightEvidence | None = None
    orchestration_root: str | None = None
    keep_alive: bool = False
    deadman_enabled: bool = False
    deadman_silence_seconds: int = Field(default=1800, ge=60)
    smoke_enabled: bool = True
    smoke_update_budget: int = Field(default=2, ge=1)
    smoke_deadline_seconds: int = Field(default=1800, ge=60)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("smoke_update_budget", mode="before")
    @classmethod
    def _validate_smoke_update_budget(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("smoke_update_budget must be a positive integer, not a boolean")
        return value

    @model_validator(mode="after")
    def _validate_bundle(self) -> "RunBundle":
        seen: set[str] = set()
        for row in self.rows:
            if row.row_id in seen:
                raise ValueError(f"duplicate row_id: {row.row_id!r}")
            if row.execution_family != self.execution_family:
                raise ValueError(f"row {row.row_id!r} execution_family does not match the bundle")
            seen.add(row.row_id)
        canonical_identities = canonicalize_immutable_input_identities(
            [item.identity for item in self.resolved_inputs]
        )
        supplied_identities = [
            item.identity.model_dump(mode="json", exclude_none=True)
            for item in self.resolved_inputs
        ]
        if supplied_identities != canonical_identities:
            raise ValueError("resolved_inputs must be in canonical immutable-input order")
        target_roles = [item.custody.target_role for item in self.resolved_inputs]
        if len(target_roles) != len(set(target_roles)):
            raise ValueError("resolved_inputs contain duplicate custody target roles")
        payload_filenames = {f"{row.row_id}.json" for row in self.rows}
        collisions = sorted(set(target_roles) & payload_filenames)
        if collisions:
            raise ValueError(
                "resolved input custody target roles collide with generated row payload "
                f"filenames: {collisions!r}"
            )
        staged_root_keys = [(item.root_kind, item.binding_name) for item in self.staged_roots]
        if staged_root_keys != sorted(staged_root_keys):
            raise ValueError("staged_roots must be in canonical (root_kind, binding_name) order")
        if len(staged_root_keys) != len(set(staged_root_keys)):
            raise ValueError("staged_roots contain duplicate typed binding names")
        if self.staged_roots and "staged-roots" in target_roles:
            raise ValueError(
                "resolved input target_role 'staged-roots' collides with staged-root custody"
            )
        for row in self.rows:
            row_identities = canonicalize_immutable_input_identities(row.execution.immutable_inputs)
            if row_identities != canonical_identities:
                raise ValueError(
                    f"row {row.row_id!r} immutable inputs do not match resolved_inputs"
                )
        if self.evaluation_output_preflight is not None:
            if self.execution_family != "evaluation-matrix":
                raise ValueError(
                    "evaluation_output_preflight is only valid for evaluation-matrix bundles"
                )
            root = (
                Path(self.orchestration_root).expanduser()
                if self.orchestration_root
                else default_orchestration_root(self.run_set_id)
            )
            expected_output_root = root if root.name == self.run_set_id else root / self.run_set_id
            if Path(self.evaluation_output_preflight.output_root) != expected_output_root:
                raise ValueError(
                    "evaluation_output_preflight output_root does not match the bundle run-set root"
                )
        return self

    @property
    def run_set_dir(self) -> Path:
        """Return the local directory where orchestration state is stored."""
        if self.orchestration_root:
            root = Path(self.orchestration_root).expanduser()
            return root if root.name == self.run_set_id else root / self.run_set_id
        return default_orchestration_root(self.run_set_id)

    def row(self, row_id: str) -> RunRowSpec:
        """Return one row spec by id."""
        for row in self.rows:
            if row.row_id == row_id:
                return row
        raise KeyError(row_id)
