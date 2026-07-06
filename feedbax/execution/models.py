"""Pydantic models for Feedbax execution contracts."""

from __future__ import annotations

import json
import mimetypes
import os
import shlex
import uuid
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from feedbax.contracts.manifest import (
    ArtifactRef,
    ManifestStatus,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from feedbax.contracts.training import (
    TRAINING_RUN_SPEC_SCHEMA_ID,
    TRAINING_RUN_SPEC_SCHEMA_VERSION,
    TrainingRunSpec,
)


EXECUTION_SPEC_SCHEMA_VERSION = "feedbax.spec.execution.v2"
EXECUTION_PLAN_SCHEMA_VERSION = "feedbax.manifest.execution.v3"
LOCAL_EXECUTION_RESULT_SCHEMA_VERSION = "feedbax.manifest.execution.v3"
EXECUTION_CLOUD_PAYLOAD_SCHEMA_ID = "feedbax.manifest.execution_cloud_payload"
EXECUTION_CLOUD_PAYLOAD_SCHEMA_VERSION = "feedbax.manifest.execution_cloud_payload.v1"
EXECUTION_REPRODUCIBILITY_SCHEMA_ID = "feedbax.manifest.execution_reproducibility"
EXECUTION_REPRODUCIBILITY_SCHEMA_VERSION = "feedbax.manifest.execution_reproducibility.v1"

ExecutionBackend = Literal["local", "ssh", "runpod", "modal"]
ExecutionKind = Literal["training", "evaluation", "analysis", "report", "custom"]
InstallMode = Literal["pypi", "github-ref", "local-rsync", "local-embed"]
RepoRole = Literal["project", "dependency", "tooling"]

DEFAULT_LOCAL_EMBED_IGNORE_PARTS = [
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    "_artifacts",
    "worktrees",
]
DEFAULT_LOCAL_EMBED_IGNORE_SUFFIXES = [".assets"]
DEFAULT_LOCAL_EMBED_REWRITE_FILES = ["pyproject.toml", "uv.lock"]


class ExecutionModel(BaseModel):
    """Base model for execution contract records."""

    model_config = ConfigDict(extra="forbid")


class RepoSource(ExecutionModel):
    """A code source that must be available before a job can run."""

    name: str
    role: RepoRole = "dependency"
    install_mode: InstallMode = "github-ref"
    package: Optional[str] = None
    git_url: Optional[str] = None
    git_ref: Optional[str] = None
    local_path: Optional[str] = None
    target_path: Optional[str] = None
    editable: bool = True
    ignore_parts: list[str] = Field(
        default_factory=lambda: list(DEFAULT_LOCAL_EMBED_IGNORE_PARTS)
    )
    ignore_suffixes: list[str] = Field(
        default_factory=lambda: list(DEFAULT_LOCAL_EMBED_IGNORE_SUFFIXES)
    )
    extra_path_rewrites: dict[str, str] = Field(default_factory=dict)
    rewrite_files: list[str] = Field(
        default_factory=lambda: list(DEFAULT_LOCAL_EMBED_REWRITE_FILES)
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_local_embed(self) -> "RepoSource":
        if self.install_mode == "local-embed" and not self.local_path:
            raise ValueError("local-embed sources require local_path")
        return self

    def remote_path(self, workspace: str) -> str:
        """Return the path where this source should live on a worker."""
        return self.target_path or f"{workspace.rstrip('/')}/{self.name}"


class ExecutionCell(ExecutionModel):
    """One independent unit within a sweep/grid-style execution."""

    id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    command: Optional[str] = None
    env: dict[str, str] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)


class ArtifactPolicy(ExecutionModel):
    """How execution outputs should be separated and collected."""

    manifest_root: str = "feedbax_runs"
    tracked_paths: list[str] = Field(default_factory=lambda: ["results"])
    bulk_paths: list[str] = Field(default_factory=lambda: ["_artifacts"])
    log_dir: str = "logs"
    sync_back: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class TrainingRunSpecSource(ExecutionModel):
    """Inline or referenced source of truth for a training execution."""

    kind: Literal["inline", "ref"] = "inline"
    inline: Optional[TrainingRunSpec] = None
    ref: Optional[str] = None
    path: str = "training-run-spec.json"
    identity: Optional[str] = None
    schema_id: str = TRAINING_RUN_SPEC_SCHEMA_ID
    schema_version: str = TRAINING_RUN_SPEC_SCHEMA_VERSION
    content_sha256: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_source(self) -> "TrainingRunSpecSource":
        if self.schema_id != TRAINING_RUN_SPEC_SCHEMA_ID:
            raise ValueError(
                "training_run_spec.schema_id must be "
                f"{TRAINING_RUN_SPEC_SCHEMA_ID!r}; found {self.schema_id!r}"
            )
        if self.schema_version != TRAINING_RUN_SPEC_SCHEMA_VERSION:
            raise ValueError(
                "training_run_spec.schema_version must be "
                f"{TRAINING_RUN_SPEC_SCHEMA_VERSION!r}; found {self.schema_version!r}"
            )
        if self.kind == "inline":
            if self.inline is None:
                raise ValueError("training_run_spec.inline is required when kind='inline'")
            expected_hash = self.resolved_content_sha256()
            if self.content_sha256 is not None and self.content_sha256 != expected_hash:
                raise ValueError(
                    "training_run_spec.content_sha256 does not match inline TrainingRunSpec"
                )
            self.content_sha256 = expected_hash
            return self
        if self.ref is None:
            raise ValueError("training_run_spec.ref is required when kind='ref'")
        if self.inline is not None:
            raise ValueError("training_run_spec.inline must be omitted when kind='ref'")
        if self.content_sha256 is None:
            raise ValueError("training_run_spec.content_sha256 is required when kind='ref'")
        return self

    def inline_payload(self) -> dict[str, Any] | None:
        """Return the inline TrainingRunSpec payload, if this source embeds one."""
        if self.inline is None:
            return None
        return self.inline.model_dump(mode="json", exclude_none=True)

    def resolved_content_sha256(self) -> str:
        """Return the canonical content hash for the referenced TrainingRunSpec."""
        if self.inline is not None:
            return sha256_bytes(canonical_json_bytes(self.inline_payload()))
        if self.content_sha256 is None:
            raise ValueError("training_run_spec.content_sha256 is required for ref sources")
        return self.content_sha256

    def resolved_identity(self) -> str:
        """Return a stable human-readable identity for planning records."""
        if self.identity:
            return self.identity
        if self.ref:
            return self.ref
        return f"sha256:{self.resolved_content_sha256()}"

    def plan_record(self, *, path: str) -> dict[str, Any]:
        """Return deterministic provenance fields for an execution plan."""
        record: dict[str, Any] = {
            "source_kind": self.kind,
            "identity": self.resolved_identity(),
            "content_sha256": self.resolved_content_sha256(),
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "path": path,
        }
        if self.ref is not None:
            record["ref"] = self.ref
        if self.metadata:
            record["metadata"] = self.metadata
        return record


class LocalBackendConfig(ExecutionModel):
    cwd: Optional[str] = None
    shell: str = "bash"


class SshBackendConfig(ExecutionModel):
    host: str = ""
    user: str = "root"
    port: int = 22
    key_path: Optional[str] = None
    workspace: str = "/workspace"
    use_nohup: bool = True

    def ssh_prefix(self) -> str:
        parts = ["ssh"]
        if self.key_path:
            key_path = str(Path(os.path.expandvars(self.key_path)).expanduser())
            parts.extend(["-i", key_path])
        if self.port:
            parts.extend(["-p", str(self.port)])
        parts.append(f"{self.user}@{self.host}" if self.host else f"{self.user}@<host>")
        return shlex.join(parts)


class RunPodBackendConfig(ExecutionModel):
    """RunPod pod-allocation fields plus the SSH worker config."""

    name: Optional[str] = None
    image_name: str = "runpod/pytorch:1.0.3-cu1281-torch290-ubuntu2204"
    cloud_type: Literal["SECURE", "COMMUNITY"] = "SECURE"
    gpu_type_ids: list[str] = Field(
        default_factory=lambda: ["NVIDIA GeForce RTX 5090", "NVIDIA GeForce RTX 4090"]
    )
    gpu_count: int = 1
    data_center_ids: list[str] = Field(default_factory=list)
    container_disk_in_gb: int = 30
    volume_in_gb: int = 30
    volume_mount_path: str = "/workspace"
    ports: list[str] = Field(default_factory=lambda: ["22/tcp", "8080/http"])
    support_public_ip: bool = True
    interruptible: bool = False
    api_key_env: str = "RUNPOD_API_KEY"
    ssh_key_path: str = "~/.runpod/ssh/RunPod-Key-Go"

    def pod_request(self, job_id: str, env: dict[str, str]) -> dict[str, Any]:
        """Return the RunPod REST create-pod payload."""
        payload: dict[str, Any] = {
            "name": self.name or f"feedbax-{job_id}",
            "cloudType": self.cloud_type,
            "computeType": "GPU",
            "imageName": self.image_name,
            "gpuTypeIds": self.gpu_type_ids,
            "gpuCount": self.gpu_count,
            "containerDiskInGb": self.container_disk_in_gb,
            "volumeInGb": self.volume_in_gb,
            "volumeMountPath": self.volume_mount_path,
            "ports": self.ports,
            "interruptible": self.interruptible,
            "env": env,
        }
        if self.cloud_type == "COMMUNITY":
            payload["supportPublicIp"] = self.support_public_ip
        if self.data_center_ids:
            payload["dataCenterIds"] = self.data_center_ids
            payload["dataCenterPriority"] = "availability"
        return payload

    def pod_create_command(self, job_id: str, env: dict[str, str]) -> str:
        """Return an equivalent runpodctl create command for operator review."""
        parts = [
            "runpodctl",
            "pod",
            "create",
            "--image",
            self.image_name,
            "--gpu-id",
            self.gpu_type_ids[0] if self.gpu_type_ids else "NVIDIA GeForce RTX 4090",
            "--gpu-count",
            str(self.gpu_count),
            "--cloud-type",
            self.cloud_type,
            "--container-disk-in-gb",
            str(self.container_disk_in_gb),
            "--volume-in-gb",
            str(self.volume_in_gb),
            "--volume-mount-path",
            self.volume_mount_path,
            "--ports",
            ",".join(self.ports),
            "--name",
            self.name or f"feedbax-{job_id}",
        ]
        if env:
            parts.extend(["--env", json.dumps(env, sort_keys=True)])
        if self.data_center_ids:
            parts.extend(["--data-center-ids", ",".join(self.data_center_ids)])
        if self.cloud_type == "COMMUNITY" and self.support_public_ip:
            parts.append("--public-ip")
        return shlex.join(parts)


class ModalBackendConfig(ExecutionModel):
    """Modal function execution fields."""

    app_name: str = "feedbax-execution"
    image_packages: list[str] = Field(
        default_factory=lambda: ["feedbax", "jax[cuda12]", "uv"]
    )
    gpu: str | list[str] = Field(default_factory=lambda: ["L40S", "A100"])
    secrets: list[str] = Field(default_factory=list)
    volume_name: Optional[str] = "feedbax-runs"
    volume_mount_path: str = "/vol"
    timeout_seconds: int = 6 * 60 * 60
    max_containers: Optional[int] = None
    use_spawn_map: bool = True
    python_version: str = "3.12"
    apt_packages: list[str] = Field(default_factory=lambda: ["git"])
    extra_install_commands: list[str] = Field(default_factory=list)
    workspace: str = "/workspace"


class ExecutionSpec(ExecutionModel):
    """Versioned request to prepare or run a Feedbax execution."""

    schema_version: Literal[EXECUTION_SPEC_SCHEMA_VERSION] = EXECUTION_SPEC_SCHEMA_VERSION
    kind: ExecutionKind = "training"
    job_id: Optional[str] = None
    backend: ExecutionBackend = "local"
    command: Optional[str] = None
    training_run_spec: Optional[TrainingRunSpecSource] = None
    cells: list[ExecutionCell] = Field(default_factory=list)
    repos: list[RepoSource] = Field(default_factory=list)
    primary_repo: Optional[str] = None
    env: dict[str, str] = Field(
        default_factory=lambda: {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
    )
    artifact_policy: ArtifactPolicy = Field(default_factory=ArtifactPolicy)
    local: LocalBackendConfig = Field(default_factory=LocalBackendConfig)
    ssh: SshBackendConfig = Field(default_factory=SshBackendConfig)
    runpod: RunPodBackendConfig = Field(default_factory=RunPodBackendConfig)
    modal: ModalBackendConfig = Field(default_factory=ModalBackendConfig)
    issues: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_execution_source(self) -> "ExecutionSpec":
        if self.training_run_spec is not None and self.kind != "training":
            raise ValueError("training_run_spec is only valid for kind='training'")
        if self.kind == "training":
            if self.training_run_spec is None and not self.command:
                raise ValueError("training executions require command or training_run_spec")
            return self
        if not self.command:
            raise ValueError(f"{self.kind!r} executions require command")
        return self

    def resolved_job_id(self) -> str:
        return self.job_id or f"{self.kind}-{uuid.uuid4().hex[:12]}"


class PlanStep(ExecutionModel):
    id: str
    title: str
    command: Optional[str] = None
    description: str = ""
    critical: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class HealthCheck(ExecutionModel):
    id: str
    command: str
    expected: str
    critical: bool = True


class MappingModel(ExecutionModel):
    """Small mapping compatibility layer for existing plan consumers."""

    def __getitem__(self, key: str) -> Any:
        return self.model_dump(mode="json", exclude_none=True)[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.model_dump(mode="json", exclude_none=True).get(key, default)

    def __contains__(self, key: object) -> bool:
        return key in self.model_dump(mode="json", exclude_none=True)


class ExecutionCloudPayload(MappingModel):
    """Versioned cloud-provider payload embedded in an execution plan."""

    schema_id: Literal[EXECUTION_CLOUD_PAYLOAD_SCHEMA_ID] = EXECUTION_CLOUD_PAYLOAD_SCHEMA_ID
    schema_version: Literal[EXECUTION_CLOUD_PAYLOAD_SCHEMA_VERSION] = (
        EXECUTION_CLOUD_PAYLOAD_SCHEMA_VERSION
    )
    provider: Literal["none", "runpod", "modal"] = "none"
    api: Optional[str] = None
    api_key_env: Optional[str] = None
    pod_request: Optional[dict[str, Any]] = None
    runpodctl_create: Optional[str] = None
    worker_transport: Optional[dict[str, Any]] = None
    readiness: list[str] = Field(default_factory=list)
    app_name: Optional[str] = None
    image_packages: list[str] = Field(default_factory=list)
    computed_image_packages: list[str] = Field(default_factory=list)
    gpu: Optional[str | list[str]] = None
    secrets: list[str] = Field(default_factory=list)
    volume: Optional[dict[str, Any]] = None
    timeout_seconds: Optional[int] = None
    max_containers: Optional[int] = None
    parallel_submission: Optional[Literal["spawn_map", "map"]] = None
    cells: list[dict[str, Any]] = Field(default_factory=list)
    generated_app: Optional[dict[str, Any]] = None
    training_run_spec: Optional[dict[str, Any]] = None

    @model_validator(mode="before")
    @classmethod
    def _require_explicit_schema_identity(cls, data: Any) -> Any:
        if isinstance(data, dict) and (
            "schema_id" not in data or "schema_version" not in data
        ):
            raise ValueError(
                "ExecutionCloudPayload requires explicit schema_id and schema_version"
            )
        return data

    def __bool__(self) -> bool:
        return self.provider != "none"

    @model_validator(mode="after")
    def _validate_provider_shape(self) -> "ExecutionCloudPayload":
        if self.provider == "none":
            return self
        if self.provider == "runpod":
            required = {
                "api": self.api,
                "api_key_env": self.api_key_env,
                "pod_request": self.pod_request,
                "runpodctl_create": self.runpodctl_create,
                "worker_transport": self.worker_transport,
            }
            missing = [key for key, value in required.items() if value in (None, "", {})]
            if missing:
                raise ValueError(f"runpod cloud payload missing required fields: {missing}")
            return self
        if self.generated_app is None:
            raise ValueError("modal cloud payload requires generated_app")
        if self.parallel_submission is None:
            raise ValueError("modal cloud payload requires parallel_submission")
        return self


class ExecutionReproducibility(MappingModel):
    """Versioned reproducibility payload embedded in an execution plan."""

    schema_id: Literal[EXECUTION_REPRODUCIBILITY_SCHEMA_ID] = (
        EXECUTION_REPRODUCIBILITY_SCHEMA_ID
    )
    schema_version: Literal[EXECUTION_REPRODUCIBILITY_SCHEMA_VERSION] = (
        EXECUTION_REPRODUCIBILITY_SCHEMA_VERSION
    )
    install_modes: dict[str, InstallMode] = Field(default_factory=dict)
    repo_refs: dict[str, dict[str, Any]] = Field(default_factory=dict)
    env: dict[str, str] = Field(default_factory=dict)
    issues: list[str] = Field(default_factory=list)
    generated_at: str = ""
    training_run_spec: Optional[dict[str, Any]] = None
    local_embed_sources: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _require_explicit_schema_identity(cls, data: Any) -> Any:
        if isinstance(data, dict) and (
            "schema_id" not in data or "schema_version" not in data
        ):
            raise ValueError(
                "ExecutionReproducibility requires explicit schema_id and schema_version"
            )
        return data


def execution_artifact_ref(
    *,
    role: str,
    logical_name: str,
    uri: str,
    media_type: str | None = None,
    storage_backend: str = "feedbax-local",
    metadata: Optional[dict[str, Any]] = None,
) -> ArtifactRef:
    """Return a prospective execution artifact route with deferred byte custody."""
    artifact_metadata = {
        "hash_status": "deferred_until_materialization",
        "size_status": "deferred_until_materialization",
        **(metadata or {}),
    }
    return ArtifactRef(
        role=role,
        logical_name=logical_name,
        uri=uri,
        media_type=media_type or _media_type_for_path(uri),
        storage_backend=storage_backend,
        metadata=artifact_metadata,
    )


def materialized_execution_artifact_ref(
    path: Path | str,
    *,
    role: str,
    logical_name: str | None = None,
    media_type: str | None = None,
    storage_backend: str = "feedbax-local",
    metadata: Optional[dict[str, Any]] = None,
) -> ArtifactRef:
    """Return an ``ArtifactRef`` stamped with the current bytes for a local file."""
    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise FileNotFoundError(artifact_path)
    stat = artifact_path.stat()
    artifact_metadata = {"hash_status": "materialized", **(metadata or {})}
    return ArtifactRef(
        role=role,
        logical_name=logical_name or artifact_path.name,
        sha256=sha256_file(artifact_path),
        media_type=media_type or _media_type_for_path(str(artifact_path)),
        size_bytes=stat.st_size,
        storage_backend=storage_backend,
        uri=str(artifact_path),
        metadata=artifact_metadata,
    )


def validate_materialized_execution_artifact(
    ref: ArtifactRef,
    *,
    path: Path | str | None = None,
    expected_role: str | None = None,
) -> ArtifactRef:
    """Validate that a materialized artifact ref still matches bytes and role."""
    if expected_role is not None and ref.role != expected_role:
        raise ValueError(
            f"Artifact role mismatch: expected {expected_role!r}, found {ref.role!r}"
        )
    artifact_path = Path(path if path is not None else ref.uri or "")
    if not artifact_path.is_file():
        raise FileNotFoundError(artifact_path)
    if ref.sha256 is None:
        raise ValueError(
            f"Artifact {ref.logical_name!r} has no sha256; materialized validation requires it"
        )
    actual_sha256 = sha256_file(artifact_path)
    if actual_sha256 != ref.sha256:
        raise ValueError(
            f"Artifact sha256 mismatch for {ref.logical_name!r}: "
            f"expected {ref.sha256}, found {actual_sha256}"
        )
    actual_size = artifact_path.stat().st_size
    if ref.size_bytes is not None and actual_size != ref.size_bytes:
        raise ValueError(
            f"Artifact size mismatch for {ref.logical_name!r}: "
            f"expected {ref.size_bytes}, found {actual_size}"
        )
    return ref


def _media_type_for_path(path: str) -> str:
    if path.endswith("/"):
        return "inode/directory"
    guessed, _ = mimetypes.guess_type(path)
    if guessed is not None:
        return guessed
    if path.endswith(".log"):
        return "text/plain"
    return "application/octet-stream"


class ExecutionPlan(ExecutionModel):
    """Concrete, inspectable execution plan generated from an execution spec."""

    kind: Literal["ExecutionPlan"] = "ExecutionPlan"
    schema_version: Literal[EXECUTION_PLAN_SCHEMA_VERSION] = EXECUTION_PLAN_SCHEMA_VERSION
    job_id: str
    backend: ExecutionBackend
    command: str
    run_directory: str
    bootstrap: list[PlanStep]
    health_checks: list[HealthCheck]
    launch: PlanStep
    monitor: list[PlanStep] = Field(default_factory=list)
    artifact_routes: list[ArtifactRef] = Field(default_factory=list)
    cloud_payload: ExecutionCloudPayload = Field(default_factory=ExecutionCloudPayload)
    reproducibility: ExecutionReproducibility = Field(default_factory=ExecutionReproducibility)
    warnings: list[str] = Field(default_factory=list)


class LocalExecutionResult(ExecutionModel):
    """Result from an explicitly local execution."""

    kind: Literal["LocalExecutionResult"] = "LocalExecutionResult"
    schema_version: Literal[LOCAL_EXECUTION_RESULT_SCHEMA_VERSION] = (
        LOCAL_EXECUTION_RESULT_SCHEMA_VERSION
    )
    job_id: str
    status: ManifestStatus
    return_code: int
    stdout: ArtifactRef
    stderr: ArtifactRef
    manifest: ArtifactRef
    execution_plan: ArtifactRef
    produced_artifacts: list[ArtifactRef] = Field(default_factory=list)
    manifest_payload: dict[str, Any]
    plan: ExecutionPlan

    @property
    def stdout_path(self) -> str:
        return self.stdout.uri or ""

    @property
    def stderr_path(self) -> str:
        return self.stderr.uri or ""

    @property
    def manifest_path(self) -> str:
        return self.manifest.uri or ""
