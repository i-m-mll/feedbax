"""Shared execution command, path, and provenance helpers."""

from __future__ import annotations

import os
import shlex
from pathlib import Path
from typing import Any

from feedbax.execution.container import collect_source_provenance
from feedbax.execution.models import ExecutionSpec, RepoSource


def expand_local_path(path: str) -> str:
    """Expand environment variables and user markers in a local filesystem path."""
    return str(Path(os.path.expandvars(path)).expanduser())


def execution_command(spec: ExecutionSpec, *, job_id: str, run_directory: str) -> str:
    """Return the canonical command for one execution run directory."""
    if spec.training_run_spec is None:
        if spec.command is None:
            raise ValueError("ExecutionSpec.command is required without training_run_spec")
        return spec.command
    spec_path = training_run_spec_path(spec, run_directory)
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


def remote_command(
    spec: ExecutionSpec,
    command: str,
    extra_env: dict[str, str] | None = None,
) -> str:
    """Prefix a command with the deterministic execution environment."""
    env = {**spec.env, **(extra_env or {})}
    env_prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in sorted(env.items()))
    return f"{env_prefix} {command}".strip()


def training_run_spec_path(spec: ExecutionSpec, run_directory: str) -> str:
    """Return the worker path for a TrainingRunSpec source inside a run directory."""
    if spec.training_run_spec is None:
        raise ValueError("training_run_spec is required")
    path = spec.training_run_spec.path
    if Path(path).is_absolute():
        return path
    return f"{run_directory.rstrip('/')}/{path.lstrip('/')}"


def training_run_spec_record(
    spec: ExecutionSpec,
    run_directory: str,
) -> dict[str, Any] | None:
    """Return the canonical execution-plan record for the TrainingRunSpec source."""
    if spec.training_run_spec is None:
        return None
    return spec.training_run_spec.plan_record(
        path=training_run_spec_path(spec, run_directory)
    )


def source_provenance_record(source: RepoSource) -> dict[str, Any]:
    """Return reproducibility fields for a local source checkout."""
    local_path = source.local_path
    provenance = (
        collect_source_provenance(Path(expand_local_path(local_path)))
        if local_path is not None
        else {}
    )
    status_short = provenance.get("status_short")
    return {
        "name": source.name,
        "local_path": local_path,
        "commit": provenance.get("commit"),
        "branch": provenance.get("branch"),
        "dirty": bool(status_short) if status_short is not None else None,
    }

