"""Project environment selection through lock and installed distribution metadata."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from importlib import metadata
from pathlib import Path
import sys
import tomllib

from packaging.markers import Marker
from packaging.utils import canonicalize_name

from feedbax.validity.occurrence import FileStamp, occurrence


@dataclass(frozen=True)
class EnvironmentSnapshot:
    versions: dict[str, str]
    lock_sha256: str | None


class EnvironmentMismatch(ValueError):
    """Installed distributions do not agree with the worker's governing lock."""


def _distribution_version(distribution: metadata.Distribution) -> tuple[str, str] | None:
    fields = distribution.metadata
    name, version = fields.get("Name"), fields.get("Version")
    return (canonicalize_name(name), version) if name and version else None


def installed_versions() -> dict[str, str]:
    return InstalledInventory().versions()


class InstalledInventory:
    """Reuse parsed installed versions while every metadata occurrence remains exact."""

    def __init__(self, paths: list[str] | None = None):
        self._paths = paths
        self._search_stamp: tuple[tuple[Path, FileStamp | None], ...] | None = None
        self._entries: tuple[metadata.Distribution, ...] = ()
        self._parsed: dict[Path, tuple[tuple[FileStamp | None, ...], tuple[str, str] | None]] = {}
        self._versions: dict[str, str] = {}

    def versions(self, *, fresh: bool = False) -> dict[str, str]:
        paths = sys.path if self._paths is None else self._paths
        named = tuple(Path(path).absolute() for path in paths)
        search_stamp = tuple((path, occurrence(path)) for path in named)
        if search_stamp != self._search_stamp:
            # The filesystem finder otherwise caches search-directory contents by mtime alone.
            metadata.MetadataPathFinder.invalidate_caches()
            self._entries = tuple(metadata.distributions(path=named))
            if tuple((path, occurrence(path)) for path in named) != search_stamp:
                raise EnvironmentMismatch("Distribution search paths changed during discovery")
            self._search_stamp = search_stamp
        versions = {}
        for entry in self._entries:
            pair = self._version(entry, fresh=fresh)
            if pair is not None:
                versions.setdefault(*pair)
        if versions != self._versions:
            self._versions = versions
        return self._versions

    def _version(self, entry: metadata.Distribution, *, fresh: bool) -> tuple[str, str] | None:
        path = entry._path if isinstance(entry, metadata.PathDistribution) else None
        if not isinstance(path, Path):
            # Non-filesystem providers own their metadata access and have no individual file stamps.
            return _distribution_version(entry)
        entries = (path, path / "METADATA") if path.name.endswith(".dist-info") else (path, path / "PKG-INFO")
        stamp = tuple(occurrence(item) for item in entries)
        if not fresh and (parsed := self._parsed.get(path)) is not None and parsed[0] == stamp:
            return parsed[1]
        pair = _distribution_version(entry)
        if tuple(occurrence(item) for item in entries) != stamp:
            raise EnvironmentMismatch(f"Distribution metadata changed while reading {path}")
        self._parsed[path] = stamp, pair
        return pair


@lru_cache(maxsize=16)
def _read_lock(path: Path, stamp: FileStamp) -> tuple[str, dict[str, frozenset[str]]]:
    data = path.read_bytes()
    if occurrence(path) != stamp:
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
    stamp = occurrence(path)
    if stamp is None:
        return EnvironmentSnapshot(installed, None)
    digest, candidates = _read_lock(path, stamp)
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
