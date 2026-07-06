"""Cloud backend rendering helpers for Feedbax execution plans."""

from __future__ import annotations

import json
import shlex
import textwrap
from pathlib import Path
from typing import Any

from feedbax.execution.commands import (
    execution_command,
    expand_local_path,
    source_provenance_record,
    training_run_spec_path,
)
from feedbax.execution.models import (
    EXECUTION_CLOUD_PAYLOAD_SCHEMA_ID,
    EXECUTION_CLOUD_PAYLOAD_SCHEMA_VERSION,
    ExecutionCell,
    ExecutionSpec,
    PlanStep,
    RepoSource,
)


def modal_bootstrap_steps(spec: ExecutionSpec) -> list[PlanStep]:
    package_list = ", ".join(modal_image_packages(spec)) or "<none>"
    steps = [
        PlanStep(
            id="render-modal-app",
            title="Render Modal app",
            command=(
                "feedbax-provider modal-app <execution-spec.json> "
                "--output feedbax_modal_execution.py"
            ),
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
        elif source.install_mode == "local-embed":
            provenance = source_provenance_record(source)
            description = (
                f"{source.name} is embedded into the Modal image from "
                f"{source.local_path} at commit {provenance.get('commit') or 'unknown'}; "
                "development mode."
            )
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


def cloud_payload(spec: ExecutionSpec, job_id: str) -> dict[str, Any]:
    if spec.backend == "runpod":
        return {
            "schema_id": EXECUTION_CLOUD_PAYLOAD_SCHEMA_ID,
            "schema_version": EXECUTION_CLOUD_PAYLOAD_SCHEMA_VERSION,
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
        cell_records = modal_cell_payloads(spec, job_id=job_id)
        return {
            "schema_id": EXECUTION_CLOUD_PAYLOAD_SCHEMA_ID,
            "schema_version": EXECUTION_CLOUD_PAYLOAD_SCHEMA_VERSION,
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
            "cells": cell_records,
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
    return {
        "schema_id": EXECUTION_CLOUD_PAYLOAD_SCHEMA_ID,
        "schema_version": EXECUTION_CLOUD_PAYLOAD_SCHEMA_VERSION,
        "provider": "none",
    }

def render_modal_app(spec: ExecutionSpec) -> str:
    """Render a Modal app script for a Modal execution spec."""
    if spec.backend != "modal":
        raise ValueError("render_modal_app only accepts backend='modal'")
    if _local_embed_sources(spec):
        return _render_modal_app_local_embed(spec)
    return _render_modal_app_pip_install(spec)


def _json_loads_literal(payload: Any, *, sort_keys: bool = False) -> str:
    """Render a Python expression that loads a JSON payload safely at import time."""
    payload_json = json.dumps(payload, sort_keys=sort_keys)
    return f"json.loads({json.dumps(payload_json)})"


def _render_modal_app_pip_install(spec: ExecutionSpec) -> str:
    """Render the legacy pip-install Modal app path."""
    job_id = spec.resolved_job_id()
    default_command = modal_execution_command(spec, job_id=job_id)
    cell_payload = modal_cell_payloads(spec, job_id=job_id)
    packages = _json_loads_literal(modal_image_packages(spec))
    secrets = _json_loads_literal(spec.modal.secrets)
    gpu = _json_loads_literal(spec.modal.gpu)
    volume_name = _json_loads_literal(spec.modal.volume_name)
    volume_mount_path = json.dumps(spec.modal.volume_mount_path)
    default_env = _json_loads_literal(spec.env, sort_keys=True)
    cells_json = _json_loads_literal(cell_payload, sort_keys=True)
    app_name = json.dumps(spec.modal.app_name)
    manifest_root = json.dumps(spec.artifact_policy.manifest_root)
    log_dir = json.dumps(spec.artifact_policy.log_dir)
    training_run_spec_payload = _json_loads_literal(
        (
            spec.training_run_spec.inline_payload()
            if spec.training_run_spec is not None
            else None
        ),
        sort_keys=True,
    )
    training_run_spec_path = _json_loads_literal(
        modal_training_run_spec_path(spec, job_id=job_id)
        if spec.training_run_spec is not None
        else None
    )
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
            DEFAULT_COMMAND = {json.dumps(default_command)}
            DEFAULT_ENV = {default_env}
            CELLS = {cells_json}
            GPU = {gpu}
            IMAGE_PACKAGES = {packages}
            LOG_DIR = {log_dir}
            MANIFEST_ROOT = {manifest_root}
            MAX_CONTAINERS = {max_containers}
            SECRETS = {secrets}
            TIMEOUT_SECONDS = {spec.modal.timeout_seconds}
            TRAINING_RUN_SPEC_PATH = {training_run_spec_path}
            TRAINING_RUN_SPEC_PAYLOAD = {training_run_spec_payload}
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
                if TRAINING_RUN_SPEC_PAYLOAD is not None:
                    spec_path = pathlib.Path(TRAINING_RUN_SPEC_PATH)
                    spec_path.parent.mkdir(parents=True, exist_ok=True)
                    spec_path.write_text(
                        json.dumps(TRAINING_RUN_SPEC_PAYLOAD, indent=2, sort_keys=True) + "\\n",
                        encoding="utf-8",
                    )
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
    )


def _render_modal_app_local_embed(spec: ExecutionSpec) -> str:
    job_id = spec.resolved_job_id()
    default_command = modal_execution_command(spec, job_id=job_id)
    cell_payload = modal_cell_payloads(spec, job_id=job_id)
    embedded_sources = _local_embed_sources(spec)
    primary_source = _primary_repo_source(spec, embedded_sources)
    primary_remote_path = primary_source.remote_path(spec.modal.workspace)
    image_env = {"PYTHONPATH": primary_remote_path, **spec.env}
    rewrite_files = _local_embed_rewrite_files(embedded_sources, spec.modal.workspace)
    replacements = _local_embed_replacements(embedded_sources, spec.modal.workspace)

    source_provenance = _local_embed_source_provenance(embedded_sources)
    add_local_dir_lines = "\n".join(
        (
            "image = image.add_local_dir(\n"
            f"    {json.dumps(expand_local_path(source.local_path or ''))},\n"
            f"    remote_path={json.dumps(source.remote_path(spec.modal.workspace))},\n"
            "    copy=True,\n"
            "    ignore=_ignore_source_factory(\n"
            f"        {json.dumps(source.ignore_parts, sort_keys=True)},\n"
            f"        {json.dumps(source.ignore_suffixes, sort_keys=True)},\n"
            "    ),\n"
            ")"
        )
        for source in embedded_sources
    )
    secrets = _json_loads_literal(spec.modal.secrets)
    gpu = _json_loads_literal(spec.modal.gpu)
    volume_name = _json_loads_literal(spec.modal.volume_name)
    volume_mount_path = json.dumps(spec.modal.volume_mount_path)
    cells_json = _json_loads_literal(cell_payload, sort_keys=True)
    app_name = json.dumps(spec.modal.app_name)
    manifest_root = json.dumps(spec.artifact_policy.manifest_root)
    log_dir = json.dumps(spec.artifact_policy.log_dir)
    apt_packages = _json_loads_literal(spec.modal.apt_packages)
    default_env = _json_loads_literal(spec.env, sort_keys=True)
    extra_install_commands = _json_loads_literal(spec.modal.extra_install_commands)
    image_env_literal = _json_loads_literal(image_env, sort_keys=True)
    source_provenance_literal = _json_loads_literal(source_provenance, sort_keys=True)
    rewrite_files_literal = _json_loads_literal(rewrite_files, sort_keys=True)
    rewrite_replacements_literal = _json_loads_literal(replacements, sort_keys=True)
    training_run_spec_payload = _json_loads_literal(
        (
            spec.training_run_spec.inline_payload()
            if spec.training_run_spec is not None
            else None
        ),
        sort_keys=True,
    )
    training_run_spec_path = _json_loads_literal(
        modal_training_run_spec_path(spec, job_id=job_id)
        if spec.training_run_spec is not None
        else None
    )
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
            APT_PACKAGES = {apt_packages}
            DEFAULT_COMMAND = {json.dumps(_uv_run_command(default_command))}
            DEFAULT_ENV = {default_env}
            CELLS = {cells_json}
            EXTRA_INSTALL_COMMANDS = {extra_install_commands}
            GPU = {gpu}
            IMAGE_ENV = {image_env_literal}
            JOB_ID = {json.dumps(job_id)}
            LOG_DIR = {log_dir}
            MANIFEST_ROOT = {manifest_root}
            MAX_CONTAINERS = {max_containers}
            PRIMARY_REPO_REMOTE_PATH = {json.dumps(primary_remote_path)}
            PYTHON_VERSION = {json.dumps(spec.modal.python_version)}
            SECRETS = {secrets}
            SOURCE_PROVENANCE = {source_provenance_literal}
            TIMEOUT_SECONDS = {spec.modal.timeout_seconds}
            TRAINING_RUN_SPEC_PATH = {training_run_spec_path}
            TRAINING_RUN_SPEC_PAYLOAD = {training_run_spec_payload}
            USE_SPAWN_MAP = {use_spawn_map}
            VOLUME_MOUNT_PATH = {volume_mount_path}
            VOLUME_NAME = {volume_name}

            REWRITE_FILES = {rewrite_files_literal}
            REWRITE_REPLACEMENTS = {rewrite_replacements_literal}
            REWRITE_COMMAND = r'''python - <<'PY'
            from pathlib import Path

            REWRITE_FILES = {json.dumps(rewrite_files, sort_keys=True)}
            REWRITE_REPLACEMENTS = {json.dumps(replacements, sort_keys=True)}

            def rewrite_embedded_paths(files, replacements):
                for file_path in files:
                    if not file_path.exists():
                        continue
                    text = file_path.read_text(encoding="utf-8")
                    for old, new in replacements.items():
                        text = text.replace(old, new)
                    file_path.write_text(text, encoding="utf-8")

            rewrite_embedded_paths(
                [Path(file_path) for file_path in REWRITE_FILES],
                REWRITE_REPLACEMENTS,
            )
            PY'''


            def _ignore_source_factory(ignore_parts: list[str], ignore_suffixes: list[str]):
                ignore_part_set = set(ignore_parts)
                ignore_suffix_tuple = tuple(ignore_suffixes)

                def _ignore_source(path: str) -> bool:
                    parts = pathlib.PurePath(path).parts
                    return any(part in ignore_part_set for part in parts) or any(
                        part.endswith(ignore_suffix_tuple) for part in parts
                    )

                return _ignore_source


            image = (
                modal.Image.debian_slim(python_version=PYTHON_VERSION)
                .apt_install(*APT_PACKAGES)
                .pip_install("uv")
                .env(IMAGE_ENV)
            )
            {textwrap.indent(add_local_dir_lines, "            ").lstrip()}
            image = image.workdir(PRIMARY_REPO_REMOTE_PATH).run_commands(
                REWRITE_COMMAND,
                "uv sync",
                *EXTRA_INSTALL_COMMANDS,
            )
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
                (run_root / "source_provenance.json").write_text(
                    json.dumps(SOURCE_PROVENANCE, indent=2, sort_keys=True) + "\\n",
                    encoding="utf-8",
                )
                env = os.environ.copy()
                env.update(DEFAULT_ENV)
                env.update(cell.get("env") or {{}})
                command = cell.get("command") or DEFAULT_COMMAND
                if TRAINING_RUN_SPEC_PAYLOAD is not None:
                    spec_path = pathlib.Path(TRAINING_RUN_SPEC_PATH)
                    spec_path.parent.mkdir(parents=True, exist_ok=True)
                    spec_path.write_text(
                        json.dumps(TRAINING_RUN_SPEC_PAYLOAD, indent=2, sort_keys=True) + "\\n",
                        encoding="utf-8",
                    )
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
    )


def write_modal_app(spec: ExecutionSpec, path: Path | str) -> Path:
    """Write a generated Modal app script for a Modal execution spec."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_modal_app(spec), encoding="utf-8")
    return output


def modal_execution_command(spec: ExecutionSpec, *, job_id: str) -> str:
    """Return the command rendered into Modal apps and Modal cloud payloads."""
    return execution_command(
        spec,
        job_id=job_id,
        run_directory=_modal_run_directory(spec, job_id=job_id),
    )


def _modal_run_directory(spec: ExecutionSpec, *, job_id: str) -> str:
    return (
        f"{spec.modal.volume_mount_path.rstrip('/')}/"
        f"{spec.artifact_policy.manifest_root}/{job_id}"
    )


def modal_training_run_spec_path(spec: ExecutionSpec, *, job_id: str) -> str:
    """Return the Modal worker path for a TrainingRunSpec source."""
    return training_run_spec_path(
        spec,
        _modal_run_directory(spec, job_id=job_id),
    )


def modal_cell_payloads(spec: ExecutionSpec, *, job_id: str) -> list[dict[str, Any]]:
    """Return Modal cell payloads shared by plan JSON and generated app globals."""
    default_command = modal_execution_command(spec, job_id=job_id)
    cells = spec.cells or [ExecutionCell(id="main", command=default_command)]
    if _local_embed_sources(spec):
        return [
            {
                **cell.model_dump(mode="json", exclude_none=True),
                "command": _uv_run_command(cell.command or default_command),
            }
            for cell in cells
        ]
    return [
        {
            "id": cell.id,
            "command": cell.command or default_command,
            "env": cell.env,
            "params": cell.params,
        }
        for cell in cells
    ]


def _local_embed_sources(spec: ExecutionSpec) -> list[RepoSource]:
    return [source for source in spec.repos if source.install_mode == "local-embed"]


def _primary_repo_source(
    spec: ExecutionSpec,
    embedded_sources: list[RepoSource],
) -> RepoSource:
    if spec.primary_repo is not None:
        for source in embedded_sources:
            if source.name == spec.primary_repo:
                return source
        raise ValueError(f"primary_repo {spec.primary_repo!r} is not a local-embed source")
    for source in embedded_sources:
        if source.role == "project":
            return source
    return embedded_sources[0]


def _uv_run_command(command: str) -> str:
    return f"uv run --no-sync {command}"


def _local_embed_replacements(
    sources: list[RepoSource],
    workspace: str,
) -> dict[str, str]:
    replacements: dict[str, str] = {}
    for source in sources:
        if source.local_path is None:
            continue
        replacements[expand_local_path(source.local_path)] = source.remote_path(workspace)
        replacements.update(source.extra_path_rewrites)
    return replacements


def _local_embed_rewrite_files(
    sources: list[RepoSource],
    workspace: str,
) -> list[str]:
    files: list[str] = []
    for source in sources:
        remote_root = source.remote_path(workspace).rstrip("/")
        for rewrite_file in source.rewrite_files:
            files.append(f"{remote_root}/{rewrite_file.lstrip('/')}")
    return files


def _local_embed_source_provenance(sources: list[RepoSource]) -> dict[str, dict[str, Any]]:
    records = [source_provenance_record(source) for source in sources]
    return {record["name"]: record for record in records}


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
