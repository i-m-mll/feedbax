"""Runtime helpers for containerized Feedbax execution."""

from __future__ import annotations

import os
import platform
import shlex
import site
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


def activate_project_venv(venv_dir: Path) -> Path:
    """Expose a uv-created project venv to the current Python process."""

    lib_dir = venv_dir / "lib"
    site_packages = sorted(lib_dir.glob("python*/site-packages"))
    if not site_packages:
        raise FileNotFoundError(f"No site-packages directory found under {lib_dir}")

    before = list(sys.path)
    site_path = str(site_packages[-1])
    site.addsitedir(site_path)
    activated_paths = [
        path for path in sys.path if path == site_path or (path not in before and path)
    ]
    sys.path[:] = [
        *activated_paths,
        *(path for path in sys.path if path not in set(activated_paths)),
    ]
    evict_bundled_modules(("typing_extensions",))

    bin_path = str(venv_dir / "bin")
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    if bin_path not in path_parts:
        os.environ["PATH"] = os.pathsep.join([bin_path, *path_parts])
    os.environ["VIRTUAL_ENV"] = str(venv_dir)
    return site_packages[-1]


def evict_bundled_modules(
    module_names: Sequence[str],
    marker_parts: Sequence[str] = ("__modal", "deps"),
) -> None:
    """Evict already-imported modules loaded from a bundled provider path."""

    marker_set = set(marker_parts)
    for module_name in module_names:
        module = sys.modules.get(module_name)
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            continue
        parts = set(Path(module_file).parts)
        if marker_set.issubset(parts):
            sys.modules.pop(module_name, None)


def collect_environment_provenance(
    packages: Sequence[str],
    repo_root: Path | None = None,
    prefix: str = "MODAL_",
) -> dict[str, Any]:
    """Collect best-effort local or container environment provenance."""

    provenance: dict[str, Any] = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "cwd": str(Path.cwd()),
        "provider_env": {
            key: value for key, value in os.environ.items() if key.startswith(prefix)
        },
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    for package in packages:
        try:
            module = __import__(package)
            provenance[f"{package}_version"] = getattr(module, "__version__", "unknown")
            if package == "jax":
                provenance["jax_devices"] = [str(device) for device in module.devices()]
                provenance["jax_default_backend"] = module.default_backend()
        except Exception as exc:
            provenance[f"{package}_error"] = str(exc)

    if repo_root is not None:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            provenance["repo_commit"] = result.stdout.strip() if result.returncode == 0 else None
        except Exception as exc:
            provenance["repo_commit_error"] = str(exc)

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        provenance["nvidia_smi"] = result.stdout.strip()
        provenance["nvidia_smi_returncode"] = result.returncode
        provenance["nvidia_smi_stderr"] = result.stderr.strip()
    except Exception as exc:
        provenance["nvidia_smi_error"] = str(exc)
    return provenance


def collect_source_provenance(repo_root: Path) -> dict[str, Any]:
    """Collect local source provenance for a repository checkout."""

    commands = {
        "commit": ["git", "rev-parse", "HEAD"],
        "branch": ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        "status_short": ["git", "status", "--short"],
    }
    provenance: dict[str, Any] = {}
    for key, command in commands.items():
        try:
            result = subprocess.run(
                command,
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            provenance[key] = result.stdout.strip() if result.returncode == 0 else None
            if result.returncode != 0:
                provenance[f"{key}_stderr"] = result.stderr.strip()
        except Exception as exc:
            provenance[f"{key}_error"] = str(exc)
    return provenance


def run_subprocess(
    command: Sequence[str],
    *,
    timeout_seconds: int,
    cwd: Path | None = None,
) -> int:
    """Run a bounded subprocess and return its exit code."""

    print(shlex.join(command), flush=True)
    result = subprocess.run(list(command), timeout=timeout_seconds, cwd=cwd, check=False)
    return result.returncode


def run_shell_command(
    command: str,
    *,
    timeout_seconds: int,
    cwd: Path | None = None,
) -> int:
    """Run a rendered execution-plan command through the shell."""

    print(command, flush=True)
    result = subprocess.run(
        ["bash", "-lc", command],
        timeout=timeout_seconds,
        cwd=cwd,
        check=False,
    )
    return result.returncode


def rewrite_embedded_paths(
    files: Sequence[Path],
    replacements: Mapping[str, str],
) -> None:
    """Rewrite literal editable source references inside embedded project files."""

    for file_path in files:
        if not file_path.exists():
            continue
        text = file_path.read_text(encoding="utf-8")
        for old, new in replacements.items():
            text = text.replace(old, new)
        file_path.write_text(text, encoding="utf-8")


__all__ = [
    "activate_project_venv",
    "collect_environment_provenance",
    "collect_source_provenance",
    "evict_bundled_modules",
    "rewrite_embedded_paths",
    "run_shell_command",
    "run_subprocess",
]
