"""The Python environment a Studio project's code runs in: the uv project that contains its folder."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
from urllib.parse import urlsplit
from urllib.request import url2pathname

from feedbax.contracts.strict_json import strict_json_loads
from feedbax.plugins.bootstrap import PackageLoadFailure
from feedbax.plugins.declarations import DistributionSource, declaration_package

ENVIRONMENT_VARIABLE = "FEEDBAX_STUDIO_ENVIRONMENT"
"""Names the uv project whose environment a Studio backend was started in; unset in Studio's own runtime."""

_FEEDBAX_PROBE = (
    "import json, importlib.util as u; "
    "spec = u.find_spec('feedbax'); "
    "print(json.dumps(None if spec is None or any(u.find_spec(name) is None for name in ('fastapi', 'uvicorn')) "
    "else spec.submodule_search_locations[0]))"
)


@dataclass(frozen=True)
class StudioRuntime:
    """No ``pyproject.toml`` lies at or above the folder: the project uses Studio's own runtime and its built-ins."""


@dataclass(frozen=True)
class UvProject:
    root: Path
    """The directory holding the ``pyproject.toml`` and ``uv.lock`` that define the environment."""
    interpreter: Path
    """The environment's Python, after ``uv sync --locked``."""
    feedbax: Path
    """The ``feedbax`` package directory that interpreter imports."""


type ProjectEnvironment = StudioRuntime | UvProject


class EnvironmentFault(StrEnum):
    LOCK_STALE = "lock_stale"
    SYNC_FAILED = "sync_failed"
    NO_FEEDBAX = "no_feedbax"
    NO_UV = "no_uv"


@dataclass(frozen=True)
class EnvironmentFailure:
    fault: EnvironmentFault
    root: Path
    reason: str
    """uv's own message, or what the environment lacks."""

    def __str__(self) -> str:
        fix = {
            EnvironmentFault.LOCK_STALE: f"Run `uv lock --project {self.root}` to update the lock.",
            EnvironmentFault.SYNC_FAILED: f"Fix {self.root / 'pyproject.toml'} until `uv sync --locked` succeeds there.",
            EnvironmentFault.NO_FEEDBAX: f"Add `feedbax[web]` to the dependencies in {self.root / 'pyproject.toml'}.",
            EnvironmentFault.NO_UV: "Install uv: https://docs.astral.sh/uv/",
        }[self.fault]
        return f"Studio cannot make the environment of {self.root / 'pyproject.toml'}.\n{self.reason}\n{fix}"


def environment_root(folder: Path) -> Path | None:
    """The nearest directory at or above ``folder`` that holds a ``pyproject.toml``."""
    for directory in (folder, *folder.parents):
        if (directory / "pyproject.toml").is_file():
            return directory
    return None


def project_environment(folder: Path) -> ProjectEnvironment | EnvironmentFailure:
    """Sync the folder's uv project exactly as locked and find its interpreter; the lock is never rewritten."""
    folder = folder.resolve()
    root = environment_root(folder)
    if root is None:
        return StudioRuntime()
    uv = shutil.which("uv")
    if uv is None:
        return EnvironmentFailure(EnvironmentFault.NO_UV, root, "No `uv` command is on PATH.")
    environment = {key: value for key, value in os.environ.items() if key != "VIRTUAL_ENV"}

    def run(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [uv, *arguments, "--project", str(root)], env=environment, capture_output=True, text=True, check=False,
        )

    checked = run("lock", "--check")
    if checked.returncode:
        return EnvironmentFailure(EnvironmentFault.LOCK_STALE, root, checked.stderr.strip())
    synced = run("sync", "--locked", "--output-format", "json")
    if synced.returncode:
        return EnvironmentFailure(EnvironmentFault.SYNC_FAILED, root, synced.stderr.strip())
    interpreter = Path(json.loads(synced.stdout)["sync"]["environment"]["python"]["path"])
    probe = subprocess.run(
        [str(interpreter), "-I", "-c", _FEEDBAX_PROBE], capture_output=True, text=True, check=True,
    )
    feedbax = json.loads(probe.stdout)
    if feedbax is None:
        return EnvironmentFailure(
            EnvironmentFault.NO_FEEDBAX, root, f"{interpreter} cannot import feedbax with its web dependencies.",
        )
    return UvProject(root, interpreter, Path(feedbax))


def running_environment() -> Path | None:
    """The uv project whose environment this process was started in, as the Studio launcher named it."""
    named = os.environ.get(ENVIRONMENT_VARIABLE)
    return Path(named) if named else None


def editable_source(distribution: str) -> Path | None:
    """The local directory an editable install of ``distribution`` was made from, if it was made so.

    Raises ``ValueError`` when the install's ``direct_url.json`` is not strict JSON.
    """
    try:
        found = metadata.distribution(distribution)
    except metadata.PackageNotFoundError:
        return None
    installed = strict_json_loads(found.read_text("direct_url.json") or "{}", ref=f"{distribution} direct_url.json")
    url = urlsplit(installed.get("url", "")) if isinstance(installed, dict) else None
    editable = isinstance(installed, dict) and isinstance(installed.get("dir_info"), dict) and (
        installed["dir_info"].get("editable") is True
    )
    if url is None or not editable or url.scheme != "file" or url.netloc not in ("", "localhost") or url.query or url.fragment:
        return None
    return Path(url2pathname(url.path))


@dataclass(frozen=True)
class PackageAvailability:
    """Which packages' declarations this process registered, and why any other one is missing."""

    loaded: frozenset[str]
    """The packages, as declaration keys name them, whose declarations are registered."""
    failures: Mapping[str, PackageLoadFailure]
    environment: Path | None
    """The uv project this process runs in; None for Studio's own runtime."""

    @classmethod
    def of(
        cls, sources: tuple[DistributionSource, ...], failures: tuple[PackageLoadFailure, ...], environment: Path | None,
    ) -> PackageAvailability:
        return cls(
            frozenset(declaration_package(source.package) for source in sources),
            {declaration_package(failure.source.package): failure for failure in failures},
            environment,
        )

    def unavailable(self, package: str, folder: Path) -> str | None:
        """Why the project in ``folder`` cannot use ``package``, or None when it can."""
        if package in self.loaded:
            return None
        if (failure := self.failures.get(package)) is not None:
            return (
                f"Package {package!r} ({failure.source.distribution} {failure.source.version}) failed to import: "
                f"module {failure.module} raised {failure.error}"
            )
        root = environment_root(folder)
        if root is None:
            return (
                f"Package {package!r} is not available: no pyproject.toml lies at or above {folder}, "
                "so the project uses Studio's own runtime, which does not include it."
            )
        if root != self.environment:
            running = (
                f"the environment of {self.environment / 'pyproject.toml'}" if self.environment is not None
                else "Studio's own runtime"
            )
            return (
                f"Package {package!r} is not available: this Studio runs in {running}, not in the project's "
                f"environment {root / 'pyproject.toml'}. Open the project with `scripts/dev.sh --project {folder}`."
            )
        return f"Package {package!r} is not installed in the project's environment {root / 'pyproject.toml'}."
