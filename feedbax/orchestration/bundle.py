"""Durable run-bundle contract for Feedbax orchestration."""

from __future__ import annotations

import hashlib
import os
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TypeAlias

from pydantic import Field, FiniteFloat, field_validator, model_validator

from feedbax.contracts.artifact_custody import ContentBinding, ImmutableArtifactBlobProviderSpec
from feedbax.contracts.canonical_json import canonical_json_v1_bytes
from feedbax.contracts.evaluation_preflight import EvaluationOutputPreflightEvidence
from feedbax.contracts.base import (
    ArtifactMigrationRecord,
    ParentRef,
    StrictModel,
)
from feedbax.contracts.execution_identity import (
    AuthoredIntentRef as AuthoredIntentRef,
    ExecutionCapsuleRef as ExecutionCapsuleRef,
    ExecutionIdentityEnvelope,
    IdentityArtifactRef as IdentityArtifactRef,
    ImmutableInputDigest as ImmutableInputDigest,
    ImmutableInputIdentity,
    ResolvedSnapshotRef as ResolvedSnapshotRef,
    SchemaArtifactRef as SchemaArtifactRef,
    execution_identity_projection as execution_identity_projection,
)
from feedbax.contracts.execution_schema import (
    DEPLOYMENT_POLICY_SCHEMA_ID,
    DEPLOYMENT_POLICY_SCHEMA_VERSION,
    DEPLOYMENT_POLICY_SCHEMA_VERSION_V1 as DEPLOYMENT_POLICY_SCHEMA_VERSION_V1,
    EXECUTION_IDENTITY_ENVELOPE_SCHEMA_ID as EXECUTION_IDENTITY_ENVELOPE_SCHEMA_ID,
    EXECUTION_IDENTITY_ENVELOPE_SCHEMA_VERSION as EXECUTION_IDENTITY_ENVELOPE_SCHEMA_VERSION,
    EXECUTION_IDENTITY_ENVELOPE_SCHEMA_VERSION_V1 as EXECUTION_IDENTITY_ENVELOPE_SCHEMA_VERSION_V1,
    RUN_BUNDLE_SCHEMA_ID,
    RUN_BUNDLE_SCHEMA_VERSION,
    RUN_BUNDLE_SCHEMA_VERSION_V1 as RUN_BUNDLE_SCHEMA_VERSION_V1,
    RUN_BUNDLE_SCHEMA_VERSION_V10 as RUN_BUNDLE_SCHEMA_VERSION_V10,
    RUN_BUNDLE_SCHEMA_VERSION_V11 as RUN_BUNDLE_SCHEMA_VERSION_V11,
    RUN_BUNDLE_SCHEMA_VERSION_V12 as RUN_BUNDLE_SCHEMA_VERSION_V12,
    RUN_BUNDLE_SCHEMA_VERSION_V13 as RUN_BUNDLE_SCHEMA_VERSION_V13,
    RUN_BUNDLE_SCHEMA_VERSION_V2 as RUN_BUNDLE_SCHEMA_VERSION_V2,
    RUN_BUNDLE_SCHEMA_VERSION_V3 as RUN_BUNDLE_SCHEMA_VERSION_V3,
    RUN_BUNDLE_SCHEMA_VERSION_V4 as RUN_BUNDLE_SCHEMA_VERSION_V4,
    RUN_BUNDLE_SCHEMA_VERSION_V5 as RUN_BUNDLE_SCHEMA_VERSION_V5,
    RUN_BUNDLE_SCHEMA_VERSION_V6 as RUN_BUNDLE_SCHEMA_VERSION_V6,
    RUN_BUNDLE_SCHEMA_VERSION_V7 as RUN_BUNDLE_SCHEMA_VERSION_V7,
    RUN_BUNDLE_SCHEMA_VERSION_V8 as RUN_BUNDLE_SCHEMA_VERSION_V8,
    RUN_BUNDLE_SCHEMA_VERSION_V9 as RUN_BUNDLE_SCHEMA_VERSION_V9,
)
from feedbax.contracts.staged_execution import validate_staged_binding_name
from feedbax.contracts.spec_storage import (
    canonicalize_immutable_input_identities,
    validate_sha256,
)


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


def canonical_orchestration_root(root: Path | str) -> Path:
    """Resolve one explicitly declared outer orchestration root.

    The declaration is the trust boundary: policy-managed aliases may occur in
    it, but all paths created below the returned physical root remain subject
    to the immutable custody provider's no-follow containment checks.
    """
    root_path = Path(root).expanduser()
    try:
        return root_path.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise ValueError(
            f"orchestration_root cannot be resolved to one physical custody root: {root_path}"
        ) from exc


def orchestration_run_set_dir(root: Path | str, run_set_id: str) -> Path:
    """Return the physical run-set directory below a declared outer root."""
    declared_root = Path(root).expanduser()
    canonical_root = canonical_orchestration_root(root)
    return canonical_root if declared_root.name == run_set_id else canonical_root / run_set_id


def canonical_run_bundle_sha256(bundle: "RunBundle") -> str:
    """Return the SHA-256 identity of the persisted run-bundle representation."""
    payload = bundle.model_dump(mode="json", exclude_none=True)
    encoded = canonical_json_v1_bytes(payload)
    return hashlib.sha256(encoded).hexdigest()


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
    dependency_groups: tuple[str, ...] | None = Field(default=None, min_length=1)
    overlay_steps: list[str] = Field(default_factory=list)
    image_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("dependency_groups")
    @classmethod
    def _validate_dependency_groups(cls, value: tuple[str, ...] | None) -> tuple[str, ...] | None:
        if value is None:
            return None
        if any(not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", group) for group in value):
            raise ValueError("dependency groups must be normalized uv group names")
        if len(value) != len(set(value)):
            raise ValueError("dependency groups must be unique")
        return tuple(sorted(value))

    @model_validator(mode="after")
    def _validate_grouped_environment(self) -> "EnvironmentDeclaration":
        if self.dependency_groups is not None and not re.fullmatch(
            r"[0-9]+\.[0-9]+\.[0-9]+", self.python_version or ""
        ):
            raise ValueError("a grouped environment requires an exact Python version")
        return self


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
        **(
            {"dependency_groups": list(environment.dependency_groups)}
            if environment.dependency_groups is not None
            else {}
        ),
        "overlay_steps": list(environment.overlay_steps),
        "image_id": environment.image_id,
    }


class RunPodCudaPlacement(StrictModel):
    """Exact Secure Cloud placement for one GPU with ephemeral storage."""

    cuda_version: str = Field(pattern=r"^[0-9]+\.[0-9]+$")
    min_driver_major: int = Field(strict=True, gt=0)
    compute_capability: str = Field(pattern=r"^[0-9]+\.[0-9]+$")
    ephemeral_disk_gb: int = Field(strict=True, gt=0)
    max_hourly_rate_usd: FiniteFloat = Field(gt=0)
    startup_seconds: int = Field(strict=True, gt=0)


class DeploymentResourceRequest(StrictModel):
    """Requested resources, distinct from provider-realized resource facts."""

    gpu_id: str | None = Field(default=None, min_length=1)
    regions: list[str] = Field(default_factory=list)
    runpod_cuda: RunPodCudaPlacement | None = None

    @field_validator("regions")
    @classmethod
    def _validate_regions(cls, value: list[str]) -> list[str]:
        if any(not region.strip() for region in value):
            raise ValueError("requested regions must be non-empty strings")
        if len(value) != len(set(value)):
            raise ValueError("requested regions must be unique and remain in preference order")
        return value

    @model_validator(mode="after")
    def _validate_exact_placement(self) -> "DeploymentResourceRequest":
        if self.runpod_cuda is not None and (self.gpu_id is None or len(self.regions) != 1):
            raise ValueError("exact RunPod CUDA placement requires a GPU and one region")
        return self


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

    @model_validator(mode="after")
    def _validate_runpod_placement_venue(self) -> "DeploymentPolicy":
        if self.resources.runpod_cuda is not None and (self.driver, self.venue) != (
            "runpod", "remote"
        ):
            raise ValueError("exact RunPod CUDA placement requires the remote RunPod driver")
        return self


class LaunchRowGroup(StrictModel):
    """One ordered row group and its in-group concurrency ceiling."""

    row_ids: list[str] = Field(min_length=1)
    max_concurrent: int = Field(ge=1)

    @field_validator("row_ids")
    @classmethod
    def _validate_row_ids(cls, value: list[str]) -> list[str]:
        for row_id in value:
            if not ROW_ID_RE.fullmatch(row_id):
                raise ValueError(f"invalid launch-group row_id: {row_id!r}")
        if len(value) != len(set(value)):
            raise ValueError("launch group contains duplicate row IDs")
        return value


class LaunchPolicy(StrictModel):
    """Row launch concurrency policy with optional ordered group barriers."""

    max_parallel_rows: int = Field(default=1, ge=1)
    warm_first: bool = False
    stagger_seconds: float = Field(default=0.0, ge=0.0)
    groups: list[LaunchRowGroup] | None = None

    @model_validator(mode="after")
    def _validate_groups(self) -> "LaunchPolicy":
        if self.groups is None:
            return self
        if not self.groups:
            raise ValueError("launch policy groups must be non-empty when declared")
        row_ids = [row_id for group in self.groups for row_id in group.row_ids]
        if len(row_ids) != len(set(row_ids)):
            raise ValueError("launch policy groups contain duplicate row membership")
        return self


class BudgetPolicy(StrictModel):
    """Run-set budget guards enforced by the stage engine."""

    max_wall_clock_seconds: FiniteFloat = Field(gt=0.0)
    max_spend_usd: FiniteFloat | None = Field(default=None, ge=0.0)
    max_create_attempts: int | None = Field(default=None, strict=True, ge=1)
    teardown_reserve_seconds: FiniteFloat | None = Field(default=None, gt=0.0)
    discard_uncollected_on_abort: bool | None = None

    @model_validator(mode="after")
    def _validate_teardown_reserve(self) -> "BudgetPolicy":
        if (
            self.teardown_reserve_seconds is not None
            and self.teardown_reserve_seconds >= self.max_wall_clock_seconds
        ):
            raise ValueError("teardown reserve must be shorter than the run budget")
        if self.discard_uncollected_on_abort and self.teardown_reserve_seconds is None:
            raise ValueError("abort-time output discard requires a teardown reserve")
        return self


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


class StagedTrainingBinding(StrictModel):
    """One exact staged generation selected for a native local run set."""

    manifest_root: str
    manifest_id: str = Field(min_length=1)
    generation: str = Field(min_length=1)

    @field_validator("manifest_root")
    @classmethod
    def _require_absolute_root(cls, value: str) -> str:
        if not Path(value).is_absolute():
            raise ValueError("staged training manifest_root must be absolute")
        return value


class RunBundle(StrictModel):
    """Schema-versioned orchestration request for a run set.

    ``feedbax_revision`` is copied from the authored
    ``RunAssemblyRequest.feedbax_revision`` after ASSEMBLE has verified it
    against the imported package. It is never minted from whatever package
    happens to be imported, because a stale install would then simply mint its
    own revision and satisfy every later check against itself.
    """

    schema_id: Literal["feedbax.orchestration.run_bundle"] = RUN_BUNDLE_SCHEMA_ID
    schema_version: Literal["feedbax.orchestration.run_bundle.v14"] = RUN_BUNDLE_SCHEMA_VERSION
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
    content_bindings: list[ContentBinding] = Field(default_factory=list)
    staged_training: StagedTrainingBinding | None = None
    evaluation_output_preflight: EvaluationOutputPreflightEvidence | None = None
    orchestration_root: str | None = None
    keep_alive: bool = False
    deadman_enabled: bool = False
    deadman_silence_seconds: int = Field(default=1800, ge=60)
    smoke_enabled: bool = True
    smoke_update_budget: int = Field(default=2, ge=1)
    smoke_deadline_seconds: int = Field(default=1800, ge=60)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _reject_legacy_staged_roots(cls, value: Any) -> Any:
        if isinstance(value, dict) and "staged_roots" in value:
            raise ValueError(
                "RunBundle v14 does not admit staged_roots; run "
                "python -m feedbax migrate-content-store --run-bundle-v12 ..."
            )
        return value

    @field_validator("smoke_update_budget", mode="before")
    @classmethod
    def _validate_smoke_update_budget(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("smoke_update_budget must be a positive integer, not a boolean")
        return value

    @model_validator(mode="after")
    def _validate_bundle(self) -> "RunBundle":
        placement = self.deployment_policy.resources.runpod_cuda
        if placement is not None and (
            self.keep_alive
            or self.deadman_enabled
            or self.budget.max_create_attempts is None
            or self.budget.teardown_reserve_seconds is None
            or self.budget.max_spend_usd is None
            or self.budget.discard_uncollected_on_abort is not True
        ):
            raise ValueError(
                "exact RunPod CUDA requires attempt, spend and teardown budgets, "
                "with controller-owned credentials and abort-time teardown authorization"
            )
        seen: set[str] = set()
        for row in self.rows:
            if row.row_id in seen:
                raise ValueError(f"duplicate row_id: {row.row_id!r}")
            if row.execution_family != self.execution_family:
                raise ValueError(f"row {row.row_id!r} execution_family does not match the bundle")
            seen.add(row.row_id)
        if self.launch_policy.groups is not None:
            grouped = {
                row_id for group in self.launch_policy.groups for row_id in group.row_ids
            }
            unknown = sorted(grouped - seen)
            omitted = sorted(seen - grouped)
            if unknown:
                raise ValueError(f"launch policy groups contain unknown row IDs: {unknown!r}")
            if omitted:
                raise ValueError(f"launch policy groups omit bundle row IDs: {omitted!r}")
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
        content_keys = [(item.root_kind, item.binding_name) for item in self.content_bindings]
        if content_keys != sorted(content_keys):
            raise ValueError(
                "content_bindings must be in canonical (root_kind, binding_name) order"
            )
        if len(content_keys) != len(set(content_keys)):
            raise ValueError("content_bindings contain duplicate typed binding names")
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
            expected_output_root = self.run_set_dir
            if Path(self.evaluation_output_preflight.output_root) != expected_output_root:
                raise ValueError(
                    "evaluation_output_preflight output_root does not match the bundle run-set root"
                )
        return self

    @property
    def run_set_dir(self) -> Path:
        """Return the local directory where orchestration state is stored."""
        if self.orchestration_root:
            return orchestration_run_set_dir(self.orchestration_root, self.run_set_id)
        return default_orchestration_root(self.run_set_id)

    def row(self, row_id: str) -> RunRowSpec:
        """Return one row spec by id."""
        for row in self.rows:
            if row.row_id == row_id:
                return row
        raise KeyError(row_id)
