"""Project environment selection through lock and installed distribution metadata."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from importlib import metadata
from pathlib import Path
import tomllib

from packaging.markers import Marker
from packaging.utils import canonicalize_name


@dataclass(frozen=True)
class EnvironmentSnapshot:
    versions: dict[str, str]
    lock_sha256: str | None


class EnvironmentMismatch(ValueError):
    """Installed distributions do not agree with the worker's governing lock."""


def installed_versions() -> dict[str, str]:
    versions = {}
    for distribution in metadata.distributions():
        fields = distribution.metadata
        name, version = fields.get("Name"), fields.get("Version")
        if name and version:
            versions.setdefault(canonicalize_name(name), version)
    return versions


@lru_cache(maxsize=16)
def _read_lock(path: Path, size: int, modified_ns: int) -> tuple[str, dict[str, frozenset[str]]]:
    data = path.read_bytes()
    current = path.stat()
    if (current.st_size, current.st_mtime_ns) != (size, modified_ns):
        raise EnvironmentMismatch(f"Dependency lock changed while reading {path}")
    document = tomllib.loads(data.decode())
    candidates: dict[str, set[str]] = defaultdict(set)
    for package in document["package"]:
        markers = package.get("resolution-markers", ())
        if not markers or any(Marker(marker).evaluate() for marker in markers):
            candidates[canonicalize_name(package["name"])].add(package["version"])
    return sha256(data).hexdigest(), {key: frozenset(values) for key, values in candidates.items()}


def read_environment(
    project_root: Path, *, installed: dict[str, str] | None = None, verify_installation: bool = False,
) -> EnvironmentSnapshot:
    """Read the present lock projections without importing package code."""
    installed = installed_versions() if installed is None else installed
    path = project_root / "uv.lock"
    try:
        info = path.stat()
    except FileNotFoundError:
        return EnvironmentSnapshot(installed, None)
    digest, candidates = _read_lock(path, info.st_size, info.st_mtime_ns)
    chosen = {}
    for name, versions in candidates.items():
        if len(versions) == 1:
            (chosen[name],) = versions
        elif installed.get(name) in versions:
            chosen[name] = installed[name]
        else:
            raise EnvironmentMismatch(f"No installed version selects {name} from {', '.join(sorted(versions))}")
    if verify_installation:
        mismatches = [
            f"{name} {version} (lock requires {chosen.get(name, 'no distribution')})"
            for name, version in sorted(installed.items()) if chosen.get(name) != version
        ]
        if mismatches:
            raise EnvironmentMismatch("Installed environment differs from the lock: " + ", ".join(mismatches))
    return EnvironmentSnapshot(chosen, digest)
