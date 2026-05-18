"""Provider-neutral execution contracts for Feedbax local and cloud runs."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import textwrap
import uuid
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from feedbax.manifest import (
    EntrypointRef,
    ManifestStatus,
    Provenance,
    default_manifest_root,
    utc_now,
    write_training_run_manifest,
)


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
    schema_version: str = "feedbax.execution.v1"
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


def default_feedbax_sources(feedbax_ref: str = "develop") -> list[RepoSource]:
    """Return default sources for provider-driven Feedbax development runs."""
    return [
        RepoSource(
            name="feedbax",
            role="project",
            install_mode="github-ref",
            package="feedbax",
            git_url="https://github.com/i-m-mll/feedbax.git",
            git_ref=feedbax_ref,
        ),
        RepoSource(
            name="jax-cookbook",
            role="dependency",
            install_mode="github-ref",
            package="jax-cookbook",
            git_url="https://github.com/i-m-mll/jax-cookbook.git",
            git_ref="main",
        ),
    ]


def prepare_execution_plan(spec: ExecutionSpec) -> ExecutionPlan:
    """Generate an inspectable backend plan without contacting cloud providers."""
    job_id = spec.resolved_job_id()
    workspace = _workspace_for_backend(spec)
    run_directory = f"{workspace.rstrip('/')}/{spec.artifact_policy.manifest_root}/{job_id}"
    command = _remote_command(spec, spec.command)
    bootstrap = _bootstrap_steps(spec, workspace)
    health_checks = _health_checks(spec)
    launch = _launch_step(spec, command, run_directory)
    cloud_payload = _cloud_payload(spec, job_id)
    warnings = _warnings(spec)
    return ExecutionPlan(
        job_id=job_id,
        backend=spec.backend,
        command=command,
        run_directory=run_directory,
        bootstrap=bootstrap,
        health_checks=health_checks,
        launch=launch,
        monitor=_monitor_steps(spec, run_directory),
        artifact_routes=_artifact_routes(spec, run_directory),
        cloud_payload=cloud_payload,
        reproducibility=_reproducibility(spec),
        warnings=warnings,
    )


def run_local_execution(
    spec: ExecutionSpec,
    *,
    root: Path | str | None = None,
    timeout: Optional[float] = None,
) -> LocalExecutionResult:
    """Run a local execution and emit a durable training manifest."""
    if spec.backend != "local":
        raise ValueError("run_local_execution only accepts backend='local'")
    job_id = spec.resolved_job_id()
    root_path = Path(root) if root is not None else default_manifest_root()
    run_dir = root_path / "executions" / job_id
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    plan = prepare_execution_plan(ExecutionSpec(**{**spec.model_dump(), "job_id": job_id}))
    (run_dir / "execution-plan.json").write_text(
        plan.model_dump_json(indent=2, exclude_none=True) + "\n",
        encoding="utf-8",
    )

    cwd = Path(spec.local.cwd).expanduser() if spec.local.cwd else None
    env = {**os.environ, **spec.env}
    proc = subprocess.run(
        [spec.local.shell, "-lc", spec.command],
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        check=False,
    )
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    status: ManifestStatus = "completed" if proc.returncode == 0 else "failed"
    studio_metadata = (
        spec.metadata.get("studio", {}) if isinstance(spec.metadata.get("studio"), dict) else {}
    )
    training_spec = (
        studio_metadata.get("training_spec")
        if isinstance(studio_metadata.get("training_spec"), dict)
        else None
    )
    task_spec = (
        studio_metadata.get("task_spec")
        if isinstance(studio_metadata.get("task_spec"), dict)
        else None
    )
    graph_spec = (
        studio_metadata.get("graph_spec")
        if isinstance(studio_metadata.get("graph_spec"), dict)
        else None
    )
    total_batches = 0
    if training_spec is not None:
        try:
            total_batches = int(training_spec.get("n_batches") or 0)
        except (TypeError, ValueError):
            total_batches = 0
    final_loss: Optional[float] = None
    training_summary: dict[str, Any] | None = None
    summary_path = cwd / "artifacts" / "training-summary.json" if cwd is not None else None
    if summary_path is not None and summary_path.exists():
        try:
            training_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary_loss = training_summary.get("final_loss")
            if isinstance(summary_loss, (int, float)):
                final_loss = float(summary_loss)
        except (OSError, json.JSONDecodeError):
            training_summary = None
    history_events: list[dict[str, Any]] = [
        {
            "type": "execution_result",
            "backend": "local",
            "return_code": proc.returncode,
            "completed_at": utc_now().isoformat(),
        }
    ]
    if training_summary is not None:
        for event in training_summary.get("history", []):
            if isinstance(event, dict):
                history_events.append({"type": "training_progress", **event})
    provenance = Provenance(
        entrypoint=EntrypointRef(
            kind="feedbax-execution",
            command=spec.command,
            metadata={"backend": "local", "return_code": proc.returncode},
        ),
        issues=list(spec.issues),
        metadata={
            "execution_plan": str(run_dir / "execution-plan.json"),
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "execution_metadata": spec.metadata,
        },
    )
    manifest, manifest_path = write_training_run_manifest(
        job_id=job_id,
        total_batches=total_batches,
        training_spec=training_spec,
        task_spec=task_spec,
        graph_spec=graph_spec,
        status=status,
        history_events=history_events,
        final_loss=final_loss,
        root=root_path,
        provenance=provenance,
    )
    return LocalExecutionResult(
        job_id=job_id,
        status=status,
        return_code=proc.returncode,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        manifest_path=str(manifest_path),
        manifest_payload=manifest.model_dump(mode="json", exclude_none=True),
        plan=plan,
    )


def _workspace_for_backend(spec: ExecutionSpec) -> str:
    if spec.backend == "runpod":
        return spec.runpod.volume_mount_path
    if spec.backend == "modal":
        return spec.modal.volume_mount_path
    if spec.backend == "ssh":
        return spec.ssh.workspace
    return str(Path(spec.local.cwd).expanduser()) if spec.local.cwd else str(Path.cwd())


def _remote_command(
    spec: ExecutionSpec,
    command: str,
    extra_env: Optional[dict[str, str]] = None,
) -> str:
    env = {**spec.env, **(extra_env or {})}
    env_prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in sorted(env.items()))
    return f"{env_prefix} {command}".strip()


def _bootstrap_steps(spec: ExecutionSpec, workspace: str) -> list[PlanStep]:
    if spec.backend == "modal":
        return _modal_bootstrap_steps(spec)

    steps = [
        PlanStep(
            id="create-workspace",
            title="Create worker workspace",
            command=f"mkdir -p {shlex.quote(workspace)}",
        )
    ]
    for source in spec.repos:
        steps.extend(_repo_steps(source, workspace, spec))
    primary = spec.primary_repo or _primary_repo(spec)
    if primary:
        primary_path = _repo_by_name(spec, primary).remote_path(workspace)
        steps.append(
            PlanStep(
                id="sync-primary-environment",
                title="Install primary project environment",
                command=f"cd {shlex.quote(primary_path)} && uv sync",
                description="Run once before CUDA/JAX wheel overrides; later runs should use uv run --no-sync.",
            )
        )
    return steps


def _modal_bootstrap_steps(spec: ExecutionSpec) -> list[PlanStep]:
    package_list = ", ".join(_modal_image_packages(spec)) or "<none>"
    steps = [
        PlanStep(
            id="render-modal-app",
            title="Render Modal app",
            command="feedbax-provider modal-app <execution-spec.json> --output feedbax_modal_execution.py",
            description="Generate the Modal App/Function wrapper from the same ExecutionSpec.",
        ),
        PlanStep(
            id="build-modal-image",
            title="Build Modal image",
            command="modal run feedbax_modal_execution.py::health",
            description=f"Modal image installs: {package_list}",
        ),
    ]
    if spec.modal.volume_name:
        steps.append(
            PlanStep(
                id="prepare-modal-volume",
                title="Prepare Modal volume",
                command=f"modal volume ls | grep -F {shlex.quote(spec.modal.volume_name)} || true",
                description="Generated app uses Volume.from_name(..., create_if_missing=True).",
                critical=False,
            )
        )
    for source in spec.repos:
        if source.install_mode == "github-ref":
            description = (
                f"{source.name} is installed into the Modal image from "
                f"{source.git_url or '<git-url>'}@{source.git_ref or '<git-ref>'}."
            )
        elif source.install_mode == "pypi":
            description = f"{source.name} is installed into the Modal image from PyPI."
        else:
            description = (
                f"{source.name} uses local-rsync, which is not embedded into the Modal image; "
                "switch to github-ref or pypi for reproducible Modal runs."
            )
        steps.append(
            PlanStep(
                id=f"modal-source-{source.name}",
                title=f"Resolve Modal source {source.name}",
                description=description,
                critical=source.install_mode != "local-rsync",
            )
        )
    return steps


def _repo_steps(
    source: RepoSource,
    workspace: str,
    spec: ExecutionSpec,
) -> list[PlanStep]:
    target = source.remote_path(workspace)
    if source.install_mode == "pypi":
        package = source.package or source.name
        return [
            PlanStep(
                id=f"install-{source.name}",
                title=f"Install {package} from PyPI",
                command=f"uv pip install {shlex.quote(package)}",
            )
        ]
    if source.install_mode == "local-rsync":
        if not source.local_path:
            return [
                PlanStep(
                    id=f"sync-{source.name}",
                    title=f"Sync {source.name} from local path",
                    description="local-rsync source is missing local_path",
                    critical=True,
                )
            ]
        local = str(Path(source.local_path).expanduser())
        excludes = " ".join(
            f"--exclude={shlex.quote(value)}"
            for value in [".git", ".venv", "worktrees", "__pycache__", ".pytest_cache"]
        )
        if spec.backend in {"ssh", "runpod"}:
            remote = _rsync_remote(spec, target)
            port = _rsync_transport(spec)
            command = (
                f"rsync -az --stats --no-owner --no-group {excludes} {port} "
                f"{shlex.quote(local.rstrip('/') + '/')} {remote}"
            )
        else:
            command = (
                f"mkdir -p {shlex.quote(target)} && "
                f"rsync -az --stats --no-owner --no-group {excludes} "
                f"{shlex.quote(local.rstrip('/') + '/')} {shlex.quote(target.rstrip('/') + '/')}"
            )
        return [
            PlanStep(
                id=f"sync-{source.name}",
                title=f"Sync local {source.name}",
                command=command,
                description="Development override; record as non-reproducible until committed.",
            )
        ]
    clone = (
        f"if [ -d {shlex.quote(target)}/.git ]; then "
        f"git -C {shlex.quote(target)} fetch --all --tags; "
        f"else git clone {shlex.quote(source.git_url or '<git-url>')} {shlex.quote(target)}; fi"
    )
    checkout = f"git -C {shlex.quote(target)} checkout {shlex.quote(source.git_ref or 'HEAD')}"
    steps = [
        PlanStep(
            id=f"fetch-{source.name}",
            title=f"Fetch {source.name} Git source",
            command=f"{clone} && {checkout}",
        )
    ]
    if source.role == "dependency" and source.editable:
        steps.append(
            PlanStep(
                id=f"install-{source.name}-editable",
                title=f"Install {source.name} editable",
                command=f"uv pip install -e {shlex.quote(target)}",
            )
        )
    return steps


def _health_checks(spec: ExecutionSpec) -> list[HealthCheck]:
    if spec.backend == "modal":
        return [
            HealthCheck(
                id="modal-cli",
                command="modal --version",
                expected="Modal CLI is installed and authenticated",
            ),
            HealthCheck(
                id="modal-function-health",
                command="modal run feedbax_modal_execution.py::health",
                expected="Generated Modal function imports Feedbax/JAX and sees the requested GPU.",
            ),
        ]

    checks = [
        HealthCheck(id="uv", command="uv --version", expected="uv is installed"),
        HealthCheck(
            id="feedbax-import",
            command="python -c 'import feedbax; print(feedbax.__version__)'",
            expected="Feedbax imports from the deployed environment",
        ),
        HealthCheck(
            id="jax-devices",
            command="python -c 'import jax; print(jax.devices())'",
            expected="JAX reports available devices",
        ),
    ]
    if spec.backend in {"ssh", "runpod", "modal"}:
        checks.append(
            HealthCheck(
                id="gpu",
                command="nvidia-smi --query-gpu=name --format=csv,noheader",
                expected="GPU is visible to the worker",
            )
        )
    if spec.backend in {"ssh", "runpod"}:
        checks.insert(
            0,
            HealthCheck(
                id="ssh",
                command=f"{_ssh_prefix_for_backend(spec)} 'true'",
                expected="SSH command succeeds",
            ),
        )
    return checks


def _ssh_prefix_for_backend(spec: ExecutionSpec) -> str:
    if spec.backend == "runpod":
        key_path = _expand_local_path(spec.runpod.ssh_key_path)
        return shlex.join(["ssh", "-i", key_path, "-p", "<ssh.port>", "root@<ssh.ip>"])
    return spec.ssh.ssh_prefix()


def _rsync_transport(spec: ExecutionSpec) -> str:
    if spec.backend == "runpod":
        key_path = _expand_local_path(spec.runpod.ssh_key_path)
        transport = shlex.join(["ssh", "-i", key_path, "-p", "<ssh.port>"])
    else:
        parts = ["ssh"]
        if spec.ssh.key_path:
            parts.extend(["-i", _expand_local_path(spec.ssh.key_path)])
        if spec.ssh.port:
            parts.extend(["-p", str(spec.ssh.port)])
        transport = shlex.join(parts)
    return f"-e {shlex.quote(transport)}"


def _rsync_remote(spec: ExecutionSpec, target: str) -> str:
    if spec.backend == "runpod":
        return f"root@<ssh.ip>:{target.rstrip('/')}/"
    return f"{spec.ssh.user}@{spec.ssh.host or '<host>'}:{target.rstrip('/')}/"


def _expand_local_path(path: str) -> str:
    return str(Path(os.path.expandvars(path)).expanduser())


def _launch_step(spec: ExecutionSpec, command: str, run_directory: str) -> PlanStep:
    if spec.backend == "modal":
        submit = "modal run --detach feedbax_modal_execution.py"
        if spec.cells:
            submit += "  # generated app uses spawn_map/map for independent cells"
        return PlanStep(
            id="modal-submit",
            title="Submit Modal execution",
            command=submit,
            description="Generated Modal app runs the command in a function with configured GPU/image/volume.",
        )
    log_dir = f"{run_directory.rstrip('/')}/{spec.artifact_policy.log_dir}"
    if spec.cells:
        script = _cell_launcher_script(spec, log_dir, run_directory)
        title = "Launch execution cells"
    else:
        script = _single_launcher_script(command, log_dir, run_directory)
        title = "Launch execution"
    launch = _background_launcher_command(script, log_dir, run_directory)
    description = (
        "Starts the worker script in the background and writes pid, exit_code, and done "
        "sentinels under the run directory."
    )
    return PlanStep(id="launch", title=title, command=launch, description=description)


def _background_launcher_command(script: str, log_dir: str, run_directory: str) -> str:
    return (
        f"mkdir -p {shlex.quote(log_dir)} {shlex.quote(run_directory)} && "
        f"nohup bash -lc {shlex.quote(script)} "
        f"> {shlex.quote(log_dir + '/launcher.stdout.log')} "
        f"2> {shlex.quote(log_dir + '/launcher.stderr.log')} & "
        f"echo $! > {shlex.quote(run_directory.rstrip('/') + '/pid')}"
    )


def _single_launcher_script(command: str, log_dir: str, run_directory: str) -> str:
    stdout_path = shlex.quote(log_dir + "/stdout.log")
    stderr_path = shlex.quote(log_dir + "/stderr.log")
    exit_path = shlex.quote(run_directory.rstrip("/") + "/exit_code")
    done_path = shlex.quote(run_directory.rstrip("/") + "/done")
    return (
        "set -o pipefail; "
        f"mkdir -p {shlex.quote(log_dir)} {shlex.quote(run_directory)}; "
        f"bash -lc {shlex.quote(command)} > {stdout_path} 2> {stderr_path}; "
        "rc=$?; "
        f"echo \"$rc\" > {exit_path}; "
        f"touch {done_path}; "
        "exit \"$rc\""
    )


def _cell_launcher_script(spec: ExecutionSpec, log_dir: str, run_directory: str) -> str:
    cell_dir = f"{log_dir.rstrip('/')}/cells"
    fragments = [
        "set -o pipefail",
        f"mkdir -p {shlex.quote(cell_dir)} {shlex.quote(run_directory)}",
        "pids=''",
    ]
    for cell in spec.cells:
        cell_command = _remote_command(spec, cell.command or spec.command, cell.env)
        safe_id = shlex.quote(cell.id)
        prefix = f"{cell_dir.rstrip('/')}/{cell.id}"
        fragments.append(
            "("
            f"bash -lc {shlex.quote(cell_command)} "
            f"> {shlex.quote(prefix + '.stdout.log')} "
            f"2> {shlex.quote(prefix + '.stderr.log')}; "
            "rc=$?; "
            f"echo \"$rc\" > {shlex.quote(prefix + '.exit_code')}; "
            f"touch {shlex.quote(prefix + '.done')}; "
            "exit \"$rc\""
            ") & "
            "pid=$!; "
            f"echo \"$pid\" > {shlex.quote(prefix + '.pid')}; "
            f"echo {safe_id} > {shlex.quote(prefix + '.cell_id')}; "
            'pids="$pids $pid"'
        )
    fragments.extend(
        [
            "overall=0",
            "for pid in $pids; do wait \"$pid\" || overall=$?; done",
            f"echo \"$overall\" > {shlex.quote(run_directory.rstrip('/') + '/exit_code')}",
            f"touch {shlex.quote(run_directory.rstrip('/') + '/done')}",
            'exit "$overall"',
        ]
    )
    return "; ".join(fragments)


def _monitor_steps(spec: ExecutionSpec, run_directory: str) -> list[PlanStep]:
    if spec.backend == "modal":
        return [
            PlanStep(
                id="modal-logs",
                title="Monitor Modal logs",
                command="modal app logs feedbax-execution",
            )
        ]
    return [
        PlanStep(
            id="tail-stdout",
            title="Tail execution stdout",
            command=f"tail -f {shlex.quote(run_directory.rstrip('/') + '/logs/stdout.log')}",
            critical=False,
        ),
        PlanStep(
            id="poll-sentinel",
            title="Poll completion sentinel",
            command=f"test -f {shlex.quote(run_directory.rstrip('/') + '/done')}",
            critical=False,
        ),
    ]


def _artifact_routes(spec: ExecutionSpec, run_directory: str) -> list[ArtifactRoute]:
    routes = [
        ArtifactRoute(
            role="execution_log",
            source=f"{run_directory.rstrip('/')}/{spec.artifact_policy.log_dir}/",
            tracked=False,
            description="Bulk logs are artifact-store data, not tracked source.",
        )
    ]
    for path in spec.artifact_policy.tracked_paths:
        routes.append(
            ArtifactRoute(
                role="tracked_spec",
                source=path,
                tracked=True,
                description="Small specs/narratives that may be committed to git.",
            )
        )
    for path in spec.artifact_policy.bulk_paths:
        routes.append(
            ArtifactRoute(
                role="bulk_output",
                source=path,
                tracked=False,
                description="Checkpoints, histories, trajectories, figures, and large outputs.",
            )
        )
    return routes


def _cloud_payload(spec: ExecutionSpec, job_id: str) -> dict[str, Any]:
    if spec.backend == "runpod":
        return {
            "provider": "runpod",
            "api": "REST /v1/pods",
            "api_key_env": spec.runpod.api_key_env,
            "pod_request": spec.runpod.pod_request(job_id, spec.env),
            "runpodctl_create": spec.runpod.pod_create_command(job_id, spec.env),
            "worker_transport": {
                "protocol": "ssh",
                "host": "<ssh.ip>",
                "port": "<ssh.port>",
                "user": "root",
                "key_path": _expand_local_path(spec.runpod.ssh_key_path),
                "workspace": spec.runpod.volume_mount_path,
            },
            "readiness": [
                "Poll pod detail until ssh.ip, ssh.port, and ssh_command are present.",
                "Probe SSH with nvidia-smi; do not rely on runtime.ports or uptimeSeconds.",
            ],
        }
    if spec.backend == "modal":
        cells = spec.cells or [ExecutionCell(id="main", command=spec.command)]
        return {
            "provider": "modal",
            "app_name": spec.modal.app_name,
            "image_packages": spec.modal.image_packages,
            "computed_image_packages": _modal_image_packages(spec),
            "gpu": spec.modal.gpu,
            "secrets": spec.modal.secrets,
            "volume": (
                {
                    "name": spec.modal.volume_name,
                    "mount_path": spec.modal.volume_mount_path,
                    "commit_required": True,
                }
                if spec.modal.volume_name
                else None
            ),
            "timeout_seconds": spec.modal.timeout_seconds,
            "max_containers": spec.modal.max_containers,
            "parallel_submission": "spawn_map" if spec.modal.use_spawn_map else "map",
            "cells": [cell.model_dump(mode="json", exclude_none=True) for cell in cells],
            "generated_app": {
                "entrypoint": "feedbax_modal_execution.py",
                "command": (
                    "feedbax-provider modal-app <execution-spec.json> "
                    "--output feedbax_modal_execution.py"
                ),
                "health_function": "health",
                "run_function": "run_cell",
            },
        }
    return {}


def _warnings(spec: ExecutionSpec) -> list[str]:
    warnings: list[str] = []
    for source in spec.repos:
        if source.install_mode == "local-rsync":
            warnings.append(
                f"{source.name}: local-rsync is a development override; prefer github-ref "
                "or pypi for reproducible runs."
            )
        if source.install_mode == "github-ref" and not source.git_ref:
            warnings.append(f"{source.name}: github-ref source has no explicit git_ref.")
        if spec.backend == "modal" and source.install_mode == "local-rsync":
            warnings.append(
                f"{source.name}: Modal cannot replay local-rsync reproducibly; push a GitHub "
                "ref or publish a package before cloud execution."
            )
    if spec.backend == "modal" and spec.cells and spec.modal.volume_name:
        warnings.append(
            "Modal cells must write to disjoint output paths or explicitly coordinate volume commits."
        )
    return warnings


def _reproducibility(spec: ExecutionSpec) -> dict[str, Any]:
    return {
        "install_modes": {source.name: source.install_mode for source in spec.repos},
        "repo_refs": {
            source.name: {
                "git_url": source.git_url,
                "git_ref": source.git_ref,
                "package": source.package,
                "local_path": source.local_path,
            }
            for source in spec.repos
        },
        "env": spec.env,
        "issues": spec.issues,
        "generated_at": utc_now().isoformat(),
    }


def _primary_repo(spec: ExecutionSpec) -> Optional[str]:
    for source in spec.repos:
        if source.role == "project":
            return source.name
    return spec.repos[0].name if spec.repos else None


def _repo_by_name(spec: ExecutionSpec, name: str) -> RepoSource:
    for source in spec.repos:
        if source.name == name:
            return source
    raise ValueError(f"Unknown repo source: {name}")


def load_execution_spec(path: Path | str) -> ExecutionSpec:
    """Load an execution spec from a JSON file."""
    return ExecutionSpec.model_validate_json(Path(path).read_text(encoding="utf-8"))


def write_execution_plan(plan: ExecutionPlan, path: Path | str) -> Path:
    """Write an execution plan JSON file."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(plan.model_dump_json(indent=2, exclude_none=True) + "\n", encoding="utf-8")
    return output


def render_modal_app(spec: ExecutionSpec) -> str:
    """Render a Modal app script for a Modal execution spec."""
    if spec.backend != "modal":
        raise ValueError("render_modal_app only accepts backend='modal'")
    job_id = spec.resolved_job_id()
    cells = spec.cells or [ExecutionCell(id="main", command=spec.command)]
    cell_payload = [
        {
            "id": cell.id,
            "command": cell.command or spec.command,
            "env": cell.env,
            "params": cell.params,
        }
        for cell in cells
    ]
    packages = json.dumps(_modal_image_packages(spec))
    secrets = json.dumps(spec.modal.secrets)
    gpu = json.dumps(spec.modal.gpu)
    volume_name = json.dumps(spec.modal.volume_name)
    volume_mount_path = json.dumps(spec.modal.volume_mount_path)
    default_env = json.dumps(spec.env, sort_keys=True)
    cells_json = json.dumps(cell_payload, sort_keys=True)
    app_name = json.dumps(spec.modal.app_name)
    manifest_root = json.dumps(spec.artifact_policy.manifest_root)
    log_dir = json.dumps(spec.artifact_policy.log_dir)
    use_spawn_map = "True" if spec.modal.use_spawn_map else "False"
    max_containers = "None" if spec.modal.max_containers is None else str(spec.modal.max_containers)
    return (
        textwrap.dedent(
            f"""
            # Generated by feedbax-provider modal-app. Edit the ExecutionSpec, not this file.
            from __future__ import annotations

            import json
            import os
            import pathlib
            import subprocess

            import modal


            APP_NAME = {app_name}
            JOB_ID = {json.dumps(job_id)}
            DEFAULT_COMMAND = {json.dumps(spec.command)}
            DEFAULT_ENV = {default_env}
            CELLS = {cells_json}
            GPU = {gpu}
            IMAGE_PACKAGES = {packages}
            LOG_DIR = {log_dir}
            MANIFEST_ROOT = {manifest_root}
            MAX_CONTAINERS = {max_containers}
            SECRETS = {secrets}
            TIMEOUT_SECONDS = {spec.modal.timeout_seconds}
            USE_SPAWN_MAP = {use_spawn_map}
            VOLUME_MOUNT_PATH = {volume_mount_path}
            VOLUME_NAME = {volume_name}

            image = modal.Image.debian_slim().pip_install(*IMAGE_PACKAGES)
            app = modal.App(APP_NAME)
            volume = (
                modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
                if VOLUME_NAME
                else None
            )
            volumes = {{VOLUME_MOUNT_PATH: volume}} if volume is not None else {{}}
            secrets = [modal.Secret.from_name(name) for name in SECRETS]
            function_kwargs = dict(
                image=image,
                gpu=GPU,
                timeout=TIMEOUT_SECONDS,
                secrets=secrets,
                volumes=volumes,
            )
            if MAX_CONTAINERS is not None:
                function_kwargs["max_containers"] = MAX_CONTAINERS


            @app.function(**function_kwargs)
            def health() -> dict:
                import feedbax
                import jax

                return {{
                    "feedbax_version": getattr(feedbax, "__version__", "unknown"),
                    "jax_devices": [str(device) for device in jax.devices()],
                }}


            @app.function(**function_kwargs)
            def run_cell(cell: dict) -> dict:
                run_root = (
                    pathlib.Path(VOLUME_MOUNT_PATH)
                    / MANIFEST_ROOT
                    / JOB_ID
                    / "cells"
                    / cell["id"]
                )
                run_root.mkdir(parents=True, exist_ok=True)
                env = os.environ.copy()
                env.update(DEFAULT_ENV)
                env.update(cell.get("env") or {{}})
                command = cell.get("command") or DEFAULT_COMMAND
                stdout_path = run_root / "stdout.log"
                stderr_path = run_root / "stderr.log"
                with stdout_path.open("w", encoding="utf-8") as stdout:
                    with stderr_path.open("w", encoding="utf-8") as stderr:
                        proc = subprocess.run(
                            ["bash", "-lc", command],
                            stdout=stdout,
                            stderr=stderr,
                            env=env,
                            text=True,
                            check=False,
                        )
                (run_root / "exit_code").write_text(f"{{proc.returncode}}\\n", encoding="utf-8")
                (run_root / "done").touch()
                (run_root / "params.json").write_text(
                    json.dumps(cell.get("params") or {{}}, indent=2, sort_keys=True) + "\\n",
                    encoding="utf-8",
                )
                if volume is not None:
                    volume.commit()
                return {{"id": cell["id"], "return_code": proc.returncode}}


            @app.local_entrypoint()
            def main() -> None:
                if USE_SPAWN_MAP:
                    run_cell.spawn_map(CELLS)
                else:
                    for result in run_cell.map(CELLS):
                        print(json.dumps(result, sort_keys=True))
            """
        ).lstrip()
        + "\n"
    )


def write_modal_app(spec: ExecutionSpec, path: Path | str) -> Path:
    """Write a generated Modal app script for a Modal execution spec."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_modal_app(spec), encoding="utf-8")
    return output


def _modal_image_packages(spec: ExecutionSpec) -> list[str]:
    packages = list(spec.modal.image_packages)
    for source in spec.repos:
        package = source.package or source.name
        if source.install_mode == "pypi":
            requirement = package
        elif source.install_mode == "github-ref" and source.git_url and source.git_ref:
            packages = [existing for existing in packages if existing != package]
            requirement = f"{package} @ git+{source.git_url}@{source.git_ref}"
        else:
            continue
        seen = set(packages)
        if requirement not in seen:
            packages.append(requirement)
    return packages
