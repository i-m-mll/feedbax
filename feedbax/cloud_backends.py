"""Cloud backend rendering helpers for Feedbax execution plans."""

from __future__ import annotations

import json
import os
import shlex
import textwrap
from pathlib import Path
from typing import Any

from feedbax.execution_models import ExecutionCell, ExecutionSpec, PlanStep


def modal_bootstrap_steps(spec: ExecutionSpec) -> list[PlanStep]:
    package_list = ", ".join(modal_image_packages(spec)) or "<none>"
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


def ssh_prefix_for_backend(spec: ExecutionSpec) -> str:
    if spec.backend == "runpod":
        key_path = expand_local_path(spec.runpod.ssh_key_path)
        return shlex.join(["ssh", "-i", key_path, "-p", "<ssh.port>", "root@<ssh.ip>"])
    return spec.ssh.ssh_prefix()


def rsync_transport(spec: ExecutionSpec) -> str:
    if spec.backend == "runpod":
        key_path = expand_local_path(spec.runpod.ssh_key_path)
        transport = shlex.join(["ssh", "-i", key_path, "-p", "<ssh.port>"])
    else:
        parts = ["ssh"]
        if spec.ssh.key_path:
            parts.extend(["-i", expand_local_path(spec.ssh.key_path)])
        if spec.ssh.port:
            parts.extend(["-p", str(spec.ssh.port)])
        transport = shlex.join(parts)
    return f"-e {shlex.quote(transport)}"


def rsync_remote(spec: ExecutionSpec, target: str) -> str:
    if spec.backend == "runpod":
        return f"root@<ssh.ip>:{target.rstrip('/')}/"
    return f"{spec.ssh.user}@{spec.ssh.host or '<host>'}:{target.rstrip('/')}/"


def expand_local_path(path: str) -> str:
    return str(Path(os.path.expandvars(path)).expanduser())


def cloud_payload(spec: ExecutionSpec, job_id: str) -> dict[str, Any]:
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
                "key_path": expand_local_path(spec.runpod.ssh_key_path),
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
            "computed_image_packages": modal_image_packages(spec),
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
    packages = json.dumps(modal_image_packages(spec))
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


def modal_image_packages(spec: ExecutionSpec) -> list[str]:
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
