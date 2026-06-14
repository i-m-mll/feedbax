"""Pydantic models for Feedbax execution contracts."""

from __future__ import annotations

import json
import os
import shlex
import uuid
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from feedbax.contracts.manifest import ManifestStatus


ExecutionBackend = Literal["local", "ssh", "runpod", "modal"]
ExecutionKind = Literal["training", "evaluation", "analysis", "report", "custom"]
InstallMode = Literal["pypi", "github-ref", "local-rsync"]
RepoRole = Literal["project", "dependency", "tooling"]


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
    metadata: dict[str, Any] = Field(default_factory=dict)

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


class ExecutionSpec(ExecutionModel):
    """Versioned request to prepare or run a Feedbax execution."""

    kind: ExecutionKind = "training"
    job_id: Optional[str] = None
    backend: ExecutionBackend = "local"
    command: str
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


class ArtifactRoute(ExecutionModel):
    role: str
    source: str
    destination: Optional[str] = None
    tracked: bool = False
    description: str = ""


class ExecutionPlan(ExecutionModel):
    """Concrete, inspectable execution plan generated from an execution spec."""

    kind: Literal["ExecutionPlan"] = "ExecutionPlan"
    schema_version: str = "feedbax.manifest.execution.v1"
    job_id: str
    backend: ExecutionBackend
    command: str
    run_directory: str
    bootstrap: list[PlanStep]
    health_checks: list[HealthCheck]
    launch: PlanStep
    monitor: list[PlanStep] = Field(default_factory=list)
    artifact_routes: list[ArtifactRoute] = Field(default_factory=list)
    cloud_payload: dict[str, Any] = Field(default_factory=dict)
    reproducibility: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class LocalExecutionResult(ExecutionModel):
    """Result from an explicitly local execution."""

    job_id: str
    status: ManifestStatus
    return_code: int
    stdout_path: str
    stderr_path: str
    manifest_path: str
    manifest_payload: dict[str, Any]
    plan: ExecutionPlan
