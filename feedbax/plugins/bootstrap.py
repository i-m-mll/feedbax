"""Discover declarations from installed package entry points."""

from __future__ import annotations

from collections.abc import Iterator
from contextvars import ContextVar
from dataclasses import dataclass
import traceback
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
importing_declarations = ContextVar("importing_declarations", default=False)


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


@dataclass(frozen=True)
class PackageLoadFailure:
    """A package whose modules did not all import, so none of its declarations are registered."""

    source: DistributionSource
    module: str
    error: str
    """The exception, with the file and line it was raised at."""

    def bootstrap_error(self) -> BootstrapError:
        return BootstrapError(
            BootstrapErrorCode.LOAD,
            f"Cannot import module {self.module!r} from {self.source.distribution!r} {self.source.version}: {self.error}",
            plugin_id=self.source.distribution,
        )


class _ImportFailed(Exception):
    def __init__(self, module: str, error: BaseException):
        super().__init__(module)
        self.module = module
        self.error = error


def _described(error: BaseException) -> str:
    text = "".join(traceback.format_exception_only(error)).strip()
    if isinstance(error, SyntaxError):
        return text
    frames = traceback.extract_tb(error.__traceback__)
    return f"{text} ({frames[-1].filename}, line {frames[-1].lineno})" if frames else text


def _load_module(name: str) -> ModuleType:
    token = importing_declarations.set(True)
    try:
        return import_module(name)
    except (Exception, SystemExit) as exc:
        raise _ImportFailed(name, exc) from exc
    finally:
        importing_declarations.reset(token)


def _package_modules(package: ModuleType) -> Iterator[ModuleType]:
    yield package
    children = iter_modules(package.__path__, package.__name__ + ".")
    for child in sorted(children, key=lambda item: item.name):
        module = _load_module(child.name)
        if child.ispkg:
            yield from _package_modules(module)
        else:
            yield module


@dataclass(frozen=True)
class PackageScan:
    packages: tuple[PackageDeclarations, ...]
    failures: tuple[PackageLoadFailure, ...]

    def require(self) -> tuple[PackageDeclarations, ...]:
        """The declarations, for a process that must not run with a package missing."""
        if self.failures:
            raise self.failures[0].bootstrap_error()
        return self.packages


def _scan(source: DistributionSource) -> PackageDeclarations:
    root = _load_module(source.package)
    if not hasattr(root, "__path__"):
        raise BootstrapError(
            BootstrapErrorCode.INVALID_REGISTRATION,
            f"Distribution {source.distribution!r} must name a package, got module {source.package!r}",
            plugin_id=source.distribution,
        )
    classes = []
    functions = []
    seen = set()
    for module in _package_modules(root):
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
    return PackageDeclarations(source, tuple(classes), tuple(functions))


@lru_cache(maxsize=1)
def scan_packages(sources: tuple[DistributionSource, ...]) -> PackageScan:
    """Scan each package; one whose module fails to import is reported and contributes nothing."""
    scans = []
    failures = []
    for source in sources:
        try:
            scans.append(_scan(source))
        except _ImportFailed as failed:
            if source.distribution == "feedbax":
                raise PackageLoadFailure(source, failed.module, _described(failed.error)).bootstrap_error() from failed.error
            failures.append(PackageLoadFailure(source, failed.module, _described(failed.error)))
    return PackageScan(tuple(scans), tuple(failures))


_tolerating_failures = False


def tolerate_package_failures() -> None:
    """Let this process run without the packages that fail to import, which it reports.

    Studio's backend restarts on every save of a package it serves, so a typo in one module must not stop it.
    A worker or a command-line process keeps failing on such a package.
    """
    global _tolerating_failures
    _tolerating_failures = True


def installed_scan() -> PackageScan:
    scan = scan_packages(installed_packages())
    if not _tolerating_failures:
        scan.require()
    return scan


def installed_declarations() -> tuple[PackageDeclarations, ...]:
    return installed_scan().packages
