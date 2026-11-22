"""Discover declarations from installed package entry points."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from importlib import import_module, metadata
from inspect import isabstract, isfunction
from pkgutil import iter_modules
from types import ModuleType
from typing import Any

from packaging.utils import canonicalize_name

from .declarations import CallableDeclaration, DistributionSource, callable_declaration

PLUGIN_ENTRY_POINT_GROUP = "feedbax.plugins"


class BootstrapErrorCode(StrEnum):
    LOAD = "load"
    INVALID_REGISTRATION = "invalid_registration"
    DUPLICATE_PLUGIN = "duplicate_plugin_identity"
    NAMESPACE_COLLISION = "namespace_collision"
    REGISTRATION_FAILURE = "registration_failure"


class BootstrapError(RuntimeError):
    def __init__(self, code: BootstrapErrorCode, message: str, *, plugin_id: str | None = None):
        super().__init__(message)
        self.code = code
        self.plugin_id = plugin_id


@dataclass(frozen=True)
class PackageDeclarations:
    """The concrete classes and declared functions each module of one package defines."""

    source: DistributionSource
    classes: tuple[type, ...]
    functions: tuple[tuple[Any, CallableDeclaration], ...]

    def subclasses[T](self, base: type[T], *excluded: type) -> tuple[type[T], ...]:
        return tuple(
            value for value in self.classes
            if value is not base and value not in excluded and issubclass(value, base)
        )


def installed_packages() -> tuple[DistributionSource, ...]:
    packages = []
    distributions = set()
    for entry in metadata.entry_points(group=PLUGIN_ENTRY_POINT_GROUP):
        dist = entry.dist
        package = entry.value
        if dist is None or not dist.name or not dist.version:
            raise BootstrapError(
                BootstrapErrorCode.INVALID_REGISTRATION, f"Missing distribution metadata for entry point {entry.name!r}",
            )
        if any(not segment.isidentifier() for segment in package.split(".")):
            raise BootstrapError(
                BootstrapErrorCode.INVALID_REGISTRATION,
                f"Entry point {entry.name!r} from {dist.name!r} must name a package, got {package!r}",
                plugin_id=dist.name,
            )
        name = canonicalize_name(dist.name)
        if name in distributions:
            raise BootstrapError(
                BootstrapErrorCode.DUPLICATE_PLUGIN,
                f"Distribution {dist.name!r} must supply exactly one {PLUGIN_ENTRY_POINT_GROUP!r} entry point",
                plugin_id=dist.name,
            )
        packages.append(DistributionSource(package, dist.name, dist.version))
        distributions.add(name)

    if "feedbax" not in distributions:
        raise BootstrapError(
            BootstrapErrorCode.LOAD,
            f"The Feedbax distribution does not expose its {PLUGIN_ENTRY_POINT_GROUP!r} entry point",
            plugin_id="feedbax",
        )

    ordered = tuple(sorted(packages, key=lambda item: item.package))
    roots: dict[str, DistributionSource] = {}
    for source in ordered:
        segments = source.package.split(".")
        for length in range(1, len(segments) + 1):
            prefix = ".".join(segments[:length])
            if previous := roots.get(prefix):
                raise BootstrapError(
                    BootstrapErrorCode.DUPLICATE_PLUGIN,
                    f"Distribution {source.distribution!r} scans overlapping package {source.package!r} "
                    f"with {previous.distribution!r} ({previous.package!r})",
                    plugin_id=source.distribution,
                )
        roots[source.package] = source
    return ordered


def _load_module(name: str, source: DistributionSource) -> ModuleType:
    try:
        return import_module(name)
    except (Exception, SystemExit) as exc:
        raise BootstrapError(
            BootstrapErrorCode.LOAD,
            f"Cannot import module {name!r} from {source.distribution!r} {source.version}: {exc}",
            plugin_id=source.distribution,
        ) from exc


def _package_modules(package: ModuleType, source: DistributionSource) -> Iterator[ModuleType]:
    yield package
    children = iter_modules(package.__path__, package.__name__ + ".")
    for child in sorted(children, key=lambda item: item.name):
        module = _load_module(child.name, source)
        if child.ispkg:
            yield from _package_modules(module, source)
        else:
            yield module


@lru_cache(maxsize=1)
def scan_packages(sources: tuple[DistributionSource, ...]) -> tuple[PackageDeclarations, ...]:
    scans = []
    for source in sources:
        root = _load_module(source.package, source)
        if not hasattr(root, "__path__"):
            raise BootstrapError(
                BootstrapErrorCode.INVALID_REGISTRATION,
                f"Distribution {source.distribution!r} must name a package, got module {source.package!r}",
                plugin_id=source.distribution,
            )
        classes = []
        functions = []
        seen = set()
        for module in _package_modules(root, source):
            for value in tuple(vars(module).values()):
                if not (isinstance(value, type) or isfunction(value)):
                    continue
                if value.__module__ != module.__name__ or id(value) in seen:
                    continue
                seen.add(id(value))
                if isfunction(value):
                    if (declaration := callable_declaration(value)) is not None:
                        functions.append((value, declaration))
                elif not isabstract(value):
                    classes.append(value)
        scans.append(PackageDeclarations(source, tuple(classes), tuple(functions)))
    return tuple(scans)


def installed_declarations() -> tuple[PackageDeclarations, ...]:
    return scan_packages(installed_packages())
