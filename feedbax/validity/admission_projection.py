"""The data-only basis for deferred receipt observation."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import metadata
from pathlib import Path
import sys
from typing import Literal

from packaging.utils import canonicalize_name

from feedbax.contracts.base import StrictModel
from feedbax.contracts.code_validity import CodeTrace, RegistryBinding
from feedbax.validity.occurrence import FileStamp, occurrence
from feedbax.validity.source import package_spec


class FileObservation(StrictModel):
    path: Path
    resolved: Path | None = None
    identity: FileStamp | Literal["unreadable"] | None = None

    @classmethod
    def read(cls, path: Path) -> FileObservation:
        try:
            resolved = path.resolve()
            identity = occurrence(resolved)
            return cls(path=path, resolved=resolved, identity=identity)
        except OSError:
            return cls(path=path, identity="unreadable")

    def current(self) -> bool:
        return self == self.read(self.path)


type PluginEntrypoint = tuple[str, str, str, str]


class AdmissionProjection(StrictModel):
    roots: dict[str, Path]
    registry: tuple[RegistryBinding, ...]
    search_paths: tuple[Path, ...]
    entrypoints: tuple[PluginEntrypoint, ...]
    declarations: dict[str, tuple[FileObservation, ...]]

    @classmethod
    def capture(
        cls, roots: dict[str, Path], registry: tuple[RegistryBinding, ...], owners: Mapping[str, str],
    ) -> AdmissionProjection:
        paths = tuple(dict.fromkeys(Path(path).absolute() for path in sys.path))
        return cls(
            roots=roots, registry=registry, search_paths=paths,
            entrypoints=plugin_entrypoints(paths),
            declarations=declaration_files(roots, registry, owners),
        )

    def current(self, traces: tuple[CodeTrace, ...]) -> bool:
        distributions = {binding.distribution for trace in traces for binding in trace.registry}
        current = plugin_entrypoints(self.search_paths)
        if self.entrypoints != current:
            return False
        return all(
            file.current()
            for distribution in distributions for file in self.declarations.get(distribution, ())
        )

def plugin_entrypoints(paths: tuple[Path, ...]) -> tuple[PluginEntrypoint, ...]:
    return tuple(sorted(
        (canonicalize_name(dist.name), dist.version, entry.name, entry.value)
        for dist in metadata.distributions(path=[str(path) for path in paths])
        for entry in dist.entry_points
        if entry.group == "feedbax.plugins" and dist.name and dist.version
    ))


def declaration_files(
    roots: dict[str, Path], registry: tuple[RegistryBinding, ...], owners: Mapping[str, str],
) -> dict[str, tuple[FileObservation, ...]]:
    paths: dict[str, set[Path]] = {}
    for declaration in registry:
        if (source := declaration.source_path) is not None and (root := roots.get(source.scope)) is not None:
            paths.setdefault(declaration.distribution, set()).add(root / source.path)
    for package, distribution in owners.items():
        if (spec := package_spec(package)) is not None and spec.has_location and spec.origin is not None:
            paths.setdefault(distribution, set()).add(Path(spec.origin).absolute())
    return {dist: tuple(FileObservation.read(path) for path in sorted(files)) for dist, files in paths.items()}
