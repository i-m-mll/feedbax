"""Execution plan generation for Feedbax local and cloud runs."""

from __future__ import annotations

import shlex
import json
from pathlib import Path
from typing import Any, Optional

from feedbax.execution.backends import (
    cloud_payload,
    modal_bootstrap_steps,
    rsync_remote,
    rsync_transport,
    ssh_prefix_for_backend,
)
from feedbax.execution.models import (
    ArtifactRoute,
    ExecutionPlan,
    ExecutionSpec,
    HealthCheck,
    PlanStep,
    RepoSource,
)
from feedbax.contracts.manifest import utc_now


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
    base_command = _execution_command(spec, job_id=job_id, run_directory=run_directory)
    command = _remote_command(spec, base_command)
    bootstrap = _bootstrap_steps(spec, workspace, run_directory)
    health_checks = _health_checks(spec)
    launch = _launch_step(spec, command, run_directory)
    payload = cloud_payload(spec, job_id)
    training_record = _training_run_spec_record(spec, run_directory)
    if training_record is not None and payload:
        payload = {**payload, "training_run_spec": training_record}
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
        cloud_payload=payload,
        reproducibility=_reproducibility(spec, run_directory),
        warnings=warnings,
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


def _execution_command(spec: ExecutionSpec, *, job_id: str, run_directory: str) -> str:
    if spec.training_run_spec is None:
        if spec.command is None:
            raise ValueError("ExecutionSpec.command is required without training_run_spec")
        return spec.command
    spec_path = _training_run_spec_path(spec, run_directory)
    return shlex.join(
        [
            "python",
            "-m",
            "feedbax",
            "execute-training-run-spec",
            spec_path,
            "--manifest-root",
            run_directory,
            "--checkpoint-root",
            f"{run_directory.rstrip('/')}/checkpoints",
            "--run-id",
            job_id,
        ]
    )


def _training_run_spec_path(spec: ExecutionSpec, run_directory: str) -> str:
    if spec.training_run_spec is None:
        raise ValueError("training_run_spec is required")
    path = spec.training_run_spec.path
    if Path(path).is_absolute():
        return path
    return f"{run_directory.rstrip('/')}/{path.lstrip('/')}"


def _training_manifest_path(job_id: str, run_directory: str) -> str:
    return f"{run_directory.rstrip('/')}/manifests/training_runs/feedbax-training-run_{job_id}.json"


def _training_run_spec_record(
    spec: ExecutionSpec,
    run_directory: str,
) -> dict[str, Any] | None:
    if spec.training_run_spec is None:
        return None
    return spec.training_run_spec.plan_record(
        path=_training_run_spec_path(spec, run_directory)
    )


def _bootstrap_steps(
    spec: ExecutionSpec,
    workspace: str,
    run_directory: str,
) -> list[PlanStep]:
    if spec.backend == "modal":
        return modal_bootstrap_steps(spec)

    steps = [
        PlanStep(
            id="create-workspace",
            title="Create worker workspace",
            command=f"mkdir -p {shlex.quote(workspace)}",
        )
    ]
    for source in spec.repos:
        steps.extend(_repo_steps(source, workspace, spec))
    if spec.training_run_spec is not None and spec.training_run_spec.inline is not None:
        steps.append(
            PlanStep(
                id="materialize-training-run-spec",
                title="Materialize TrainingRunSpec",
                command=_materialize_training_run_spec_command(spec, run_directory),
                description=(
                    "Writes the inline TrainingRunSpec consumed by "
                    "python -m feedbax execute-training-run-spec."
                ),
            )
        )
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


def _materialize_training_run_spec_command(spec: ExecutionSpec, run_directory: str) -> str:
    if spec.training_run_spec is None:
        raise ValueError("training_run_spec is required")
    payload = spec.training_run_spec.inline_payload()
    if payload is None:
        raise ValueError("inline TrainingRunSpec payload is required")
    spec_path = _training_run_spec_path(spec, run_directory)
    spec_dir = str(Path(spec_path).parent)
    body = json.dumps(payload, indent=2, sort_keys=True)
    return (
        f"mkdir -p {shlex.quote(spec_dir)} && "
        f"cat > {shlex.quote(spec_path)} <<'JSON'\n"
        f"{body}\n"
        "JSON"
    )


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
            remote = rsync_remote(spec, target)
            port = rsync_transport(spec)
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
                command=f"{ssh_prefix_for_backend(spec)} 'true'",
                expected="SSH command succeeds",
            ),
        )
    return checks


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
    base_command = _execution_command(
        spec,
        job_id=run_directory.rstrip("/").rsplit("/", 1)[-1],
        run_directory=run_directory,
    )
    for cell in spec.cells:
        cell_command = _remote_command(spec, cell.command or base_command, cell.env)
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
    if spec.training_run_spec is not None:
        routes.extend(
            [
                ArtifactRoute(
                    role="training_run_spec",
                    source=_training_run_spec_path(spec, run_directory),
                    tracked=True,
                    description="Source TrainingRunSpec consumed by the generic Feedbax executor.",
                ),
                ArtifactRoute(
                    role="training_run_manifest",
                    source=_training_manifest_path(
                        run_directory.rstrip("/").rsplit("/", 1)[-1],
                        run_directory,
                    ),
                    tracked=True,
                    description="Native TrainingRunManifest emitted by execute-training-run-spec.",
                ),
            ]
        )
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


def _warnings(spec: ExecutionSpec) -> list[str]:
    warnings: list[str] = []
    if spec.training_run_spec is not None and spec.command is not None:
        warnings.append(
            "TrainingRunSpec-driven executions ignore command and derive the runner invocation "
            "from python -m feedbax execute-training-run-spec."
        )
    if spec.training_run_spec is not None:
        for cell in spec.cells:
            if cell.command is not None:
                warnings.append(
                    f"{cell.id}: cell.command overrides the TrainingRunSpec-derived command; "
                    "prefer env/params for training cells."
                )
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


def _reproducibility(spec: ExecutionSpec, run_directory: str) -> dict[str, Any]:
    record: dict[str, Any] = {
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
    training_source = _training_run_spec_record(spec, run_directory)
    if training_source is not None:
        record["training_run_spec"] = training_source
    return record


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
