"""Typed, transactional discovery for the single ``feedbax.plugins`` protocol."""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import importlib.metadata
import inspect
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from contextvars import ContextVar
from enum import StrEnum
from types import MappingProxyType, ModuleType
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

if TYPE_CHECKING:
    from .application import ApplicationRegistryBundle

PLUGIN_ENTRY_POINT_GROUP = "feedbax.plugins"
PLUGIN_PROTOCOL_VERSION = "feedbax.plugin.v1"
PLUGIN_DECLARATION_SCHEMA_ID = "feedbax.plugin.declaration"
PLUGIN_DECLARATION_SCHEMA_VERSION = "feedbax.plugin.declaration.v1"
BOOTSTRAP_CONTEXT_VERSION = "feedbax.bootstrap.v1"

T = TypeVar("T")
_BOOTSTRAP_LINEAGE: ContextVar[frozenset[int]] = ContextVar(
    "feedbax_bootstrap_lineage", default=frozenset()
)


class BootstrapErrorCode(StrEnum):
    LOAD = "load"
    UNSUPPORTED_PROTOCOL = "unsupported_protocol"
    DUPLICATE_PLUGIN = "duplicate_plugin_identity"
    MISSING_DEPENDENCY = "missing_dependency"
    MISSING_FAMILY = "missing_family"
    DEPENDENCY_CYCLE = "dependency_cycle"
    NAMESPACE_COLLISION = "namespace_collision"
    REGISTRATION_FAILURE = "registration_failure"
    REENTRANCY = "reentrancy"
    INVALID_REGISTRATION = "invalid_registration"


class BootstrapError(RuntimeError):
    """Attributable, machine-classifiable bootstrap failure."""

    def __init__(self, code: BootstrapErrorCode, message: str, *, plugin_id: str | None = None):
        super().__init__(message)
        self.code = code
        self.plugin_id = plugin_id


@dataclass(frozen=True)
class RegistryKey(Generic[T]):
    """Typed field selector for one declared registry family."""

    family: str
    attribute: str
    expected_type: type[T]
    protocol_version: str = "1"
    registered_keys: Callable[[T], Iterable[str]] = lambda _value: ()

    def __post_init__(self) -> None:
        if not all((self.family, self.attribute, self.protocol_version)):
            raise BootstrapError(BootstrapErrorCode.INVALID_REGISTRATION, "invalid registry key")
        if not isinstance(self.expected_type, type) or not callable(self.registered_keys):
            raise BootstrapError(BootstrapErrorCode.INVALID_REGISTRATION, "invalid registry key")


@dataclass(frozen=True)
class PluginDependency:
    plugin_id: str
    version: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.plugin_id, str) or not self.plugin_id:
            raise BootstrapError(BootstrapErrorCode.INVALID_REGISTRATION, "invalid dependency")
        if self.version is not None and (not isinstance(self.version, str) or not self.version):
            raise BootstrapError(BootstrapErrorCode.INVALID_REGISTRATION, "invalid dependency")


@dataclass(frozen=True)
class FamilyRequirement:
    family: str
    protocol_version: str = "1"

    def __post_init__(self) -> None:
        if not isinstance(self.family, str) or not self.family or not self.protocol_version:
            raise BootstrapError(BootstrapErrorCode.INVALID_REGISTRATION, "invalid family")


@dataclass(frozen=True)
class PluginDeclaration:
    """Strict versioned plugin identity and dependency declaration."""

    plugin_id: str
    version: str
    schema_id: str = PLUGIN_DECLARATION_SCHEMA_ID
    schema_version: str = PLUGIN_DECLARATION_SCHEMA_VERSION
    plugin_protocol_version: str = PLUGIN_PROTOCOL_VERSION
    supported_context_versions: tuple[str, ...] = (BOOTSTRAP_CONTEXT_VERSION,)
    dependencies: tuple[PluginDependency, ...] = ()
    families: tuple[FamilyRequirement, ...] = ()

    def __post_init__(self) -> None:
        strings = (self.plugin_id, self.version, self.schema_id, self.schema_version)
        if any(not isinstance(value, str) or not value for value in strings):
            raise BootstrapError(BootstrapErrorCode.INVALID_REGISTRATION, "invalid declaration")
        if not isinstance(self.dependencies, tuple) or not all(
            isinstance(item, PluginDependency) for item in self.dependencies
        ):
            raise BootstrapError(BootstrapErrorCode.INVALID_REGISTRATION, "invalid dependencies")
        if not isinstance(self.families, tuple) or not all(
            isinstance(item, FamilyRequirement) for item in self.families
        ):
            raise BootstrapError(BootstrapErrorCode.INVALID_REGISTRATION, "invalid families")
        if not isinstance(self.supported_context_versions, tuple) or not all(
            isinstance(item, str) and item for item in self.supported_context_versions
        ):
            raise BootstrapError(BootstrapErrorCode.INVALID_REGISTRATION, "invalid contexts")


RegistrationCallable = Callable[["RegistrationContext"], object | Awaitable[object]]


@dataclass(frozen=True)
class PluginRegistration:
    declaration: PluginDeclaration
    register: RegistrationCallable

    def __post_init__(self) -> None:
        if not isinstance(self.declaration, PluginDeclaration) or not callable(self.register):
            raise BootstrapError(
                BootstrapErrorCode.INVALID_REGISTRATION, "invalid plugin registration"
            )


@dataclass(frozen=True)
class PluginSource:
    registration: PluginRegistration
    entry_point_name: str
    entry_point_value: str
    distribution: str | None = None
    distribution_version: str | None = None
    fingerprint: str | None = None


@dataclass(frozen=True)
class PluginProvenance:
    plugin_id: str
    plugin_version: str
    distribution: str | None
    distribution_version: str | None
    fingerprint: str
    entry_point_name: str
    entry_point_value: str
    family_protocols: Mapping[str, str]
    registration_order: int
    registered_keys: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True)
class BootstrapState:
    """Caller-owned completed bundle; no process-global state is published."""

    bundle: ApplicationRegistryBundle
    provenance: tuple[PluginProvenance, ...]
    context_version: str = BOOTSTRAP_CONTEXT_VERSION

    def registry(self, key: RegistryKey[T]) -> T:
        return _registry_from_bundle(self.bundle, key)


def _registry_from_bundle(bundle: ApplicationRegistryBundle, key: RegistryKey[T]) -> T:
    value = getattr(bundle, key.attribute, None)
    if not isinstance(value, key.expected_type):
        raise BootstrapError(
            BootstrapErrorCode.MISSING_FAMILY,
            f"bundle field {key.attribute!r} is not {key.expected_type.__name__} "
            f"for family {key.family!r}",
        )
    return cast(T, value)


class RegistrationContext:
    """One isolated, exactly-once bootstrap execution context."""

    def __init__(
        self,
        bundle_factory: Callable[[], ApplicationRegistryBundle],
        registry_keys: Iterable[RegistryKey[Any]],
        *,
        context_version: str = BOOTSTRAP_CONTEXT_VERSION,
    ) -> None:
        keys = tuple(registry_keys)
        if len({key.family for key in keys}) != len(keys):
            raise ValueError("registry family names must be unique")
        bundle = bundle_factory()
        for key in keys:
            _registry_from_bundle(bundle, key)
        self._bundle = bundle
        self.context_version = context_version
        self._keys = MappingProxyType({key.family: key for key in keys})
        self._task: asyncio.Task[BootstrapState] | None = None
        self._owner: asyncio.Task[Any] | None = None
        self._active_plugin: str | None = None
        self._active_families: frozenset[str] = frozenset()
        self._sealed = False

    @property
    def families(self) -> Mapping[str, RegistryKey[Any]]:
        return self._keys

    def registry(self, key: RegistryKey[T]) -> T:
        if self._sealed:
            raise BootstrapError(
                BootstrapErrorCode.REGISTRATION_FAILURE, "bootstrap context is sealed"
            )
        if self._bundle is None:
            raise BootstrapError(
                BootstrapErrorCode.REGISTRATION_FAILURE, "bootstrap context is failed"
            )
        configured = self._keys.get(key.family)
        if configured is not key or configured.protocol_version != key.protocol_version:
            raise BootstrapError(
                BootstrapErrorCode.MISSING_FAMILY,
                f"family {key.family!r} with protocol {key.protocol_version!r} is unavailable",
                plugin_id=self._active_plugin,
            )
        if self._active_plugin is not None and key.family not in self._active_families:
            raise BootstrapError(
                BootstrapErrorCode.MISSING_FAMILY,
                f"plugin {self._active_plugin!r} did not declare family {key.family!r}",
                plugin_id=self._active_plugin,
            )
        return _registry_from_bundle(self._bundle, key)


def _installed_entry_points(group: str) -> tuple[Any, ...]:
    try:
        return tuple(importlib.metadata.entry_points(group=group))
    except TypeError:
        points = importlib.metadata.entry_points()
        select = getattr(points, "select", None)
        if callable(select):
            return tuple(select(group=group))
        get = getattr(points, "get", None)
        return tuple(get(group, ())) if callable(get) else ()


def _fingerprint(*values: str | None) -> str:
    return hashlib.sha256("\0".join(value or "" for value in values).encode()).hexdigest()


def _typed_registration(value: object, origin: str) -> PluginRegistration:
    if not isinstance(value, PluginRegistration):
        raise BootstrapError(
            BootstrapErrorCode.INVALID_REGISTRATION,
            f"{origin} must load PluginRegistration; legacy hook modules are unsupported",
        )
    return value


def _module_registration(module: ModuleType, origin: str) -> PluginRegistration:
    value = getattr(module, "PLUGIN_REGISTRATION", None)
    return _typed_registration(value, origin)


def discover_plugin_registrations(
    *,
    entry_points: Iterable[Any] | None = None,
    modules: Sequence[str] = (),
    entry_point_group: str = PLUGIN_ENTRY_POINT_GROUP,
) -> tuple[PluginSource, ...]:
    """Load declarations only; registration executes later in canonical order."""
    sources: list[PluginSource] = []
    for point in (
        entry_points if entry_points is not None else _installed_entry_points(entry_point_group)
    ):
        name, value = str(point.name), str(point.value)
        try:
            registration = _typed_registration(point.load(), f"entry-point:{name}={value}")
        except BootstrapError:
            raise
        except Exception as exc:
            raise BootstrapError(
                BootstrapErrorCode.LOAD, f"failed to load {name!r}: {exc}"
            ) from exc
        distribution = getattr(point, "dist", None)
        distribution_name = getattr(distribution, "name", None)
        distribution_version = getattr(distribution, "version", None)
        sources.append(
            PluginSource(
                registration,
                name,
                value,
                distribution_name,
                distribution_version,
                _fingerprint(distribution_name, distribution_version, name, value),
            )
        )
    for module_name in modules:
        try:
            registration = _module_registration(importlib.import_module(module_name), module_name)
        except BootstrapError:
            raise
        except Exception as exc:
            raise BootstrapError(
                BootstrapErrorCode.LOAD, f"failed to import module {module_name!r}: {exc}"
            ) from exc
        sources.append(
            PluginSource(
                registration,
                module_name,
                f"{module_name}:PLUGIN_REGISTRATION",
                fingerprint=_fingerprint("module", module_name, registration.declaration.version),
            )
        )
    return tuple(sources)


def _validated_order(
    sources: Sequence[PluginSource], context: RegistrationContext
) -> tuple[PluginSource, ...]:
    by_id: dict[str, PluginSource] = {}
    for source in sources:
        declaration = source.registration.declaration
        plugin_id = declaration.plugin_id
        if (
            declaration.schema_id != PLUGIN_DECLARATION_SCHEMA_ID
            or declaration.schema_version != PLUGIN_DECLARATION_SCHEMA_VERSION
            or declaration.plugin_protocol_version != PLUGIN_PROTOCOL_VERSION
            or context.context_version not in declaration.supported_context_versions
        ):
            raise BootstrapError(
                BootstrapErrorCode.UNSUPPORTED_PROTOCOL,
                f"plugin {plugin_id!r} has an unsupported declaration or protocol version",
                plugin_id=plugin_id,
            )
        if plugin_id in by_id:
            raise BootstrapError(
                BootstrapErrorCode.DUPLICATE_PLUGIN,
                f"duplicate plugin identity {plugin_id!r}",
                plugin_id=plugin_id,
            )
        by_id[plugin_id] = source
        dependency_ids = [item.plugin_id for item in declaration.dependencies]
        if len(dependency_ids) != len(set(dependency_ids)):
            raise BootstrapError(
                BootstrapErrorCode.INVALID_REGISTRATION,
                f"plugin {plugin_id!r} declares a dependency more than once",
                plugin_id=plugin_id,
            )
        for requirement in declaration.families:
            key = context.families.get(requirement.family)
            if key is None:
                raise BootstrapError(
                    BootstrapErrorCode.MISSING_FAMILY,
                    f"plugin {plugin_id!r} requires missing family {requirement.family!r}",
                    plugin_id=plugin_id,
                )
            if key.protocol_version != requirement.protocol_version:
                raise BootstrapError(
                    BootstrapErrorCode.UNSUPPORTED_PROTOCOL,
                    f"plugin {plugin_id!r} requires unsupported {requirement.family!r} protocol",
                    plugin_id=plugin_id,
                )
    outgoing = {plugin_id: set() for plugin_id in by_id}
    indegree = {plugin_id: 0 for plugin_id in by_id}
    for plugin_id, source in by_id.items():
        for dependency in source.registration.declaration.dependencies:
            target = by_id.get(dependency.plugin_id)
            if target is None or (
                dependency.version is not None
                and target.registration.declaration.version != dependency.version
            ):
                raise BootstrapError(
                    BootstrapErrorCode.MISSING_DEPENDENCY,
                    f"plugin {plugin_id!r} has unsatisfied dependency {dependency.plugin_id!r}",
                    plugin_id=plugin_id,
                )
            outgoing[dependency.plugin_id].add(plugin_id)
            indegree[plugin_id] += 1
    ready = sorted(key for key, count in indegree.items() if count == 0)
    ordered: list[PluginSource] = []
    while ready:
        plugin_id = ready.pop(0)
        ordered.append(by_id[plugin_id])
        for dependant in sorted(outgoing[plugin_id]):
            indegree[dependant] -= 1
            if not indegree[dependant]:
                ready.append(dependant)
                ready.sort()
    if len(ordered) != len(sources):
        raise BootstrapError(BootstrapErrorCode.DEPENDENCY_CYCLE, "plugin dependency cycle")
    return tuple(ordered)


def _keys(context: RegistrationContext) -> dict[str, tuple[str, ...]]:
    assert context._bundle is not None
    return {
        family: tuple(
            sorted(map(str, key.registered_keys(_registry_from_bundle(context._bundle, key))))
        )
        for family, key in context.families.items()
    }


async def _execute(
    context: RegistrationContext,
    registrations: Sequence[PluginSource | PluginRegistration] | None,
    entry_points: Iterable[Any] | None,
    modules: Sequence[str],
) -> BootstrapState:
    context._owner = asyncio.current_task()
    provenance: list[PluginProvenance] = []
    try:
        sources = (
            discover_plugin_registrations(entry_points=entry_points, modules=modules)
            if registrations is None
            else tuple(
                item
                if isinstance(item, PluginSource)
                else PluginSource(item, item.declaration.plugin_id, "explicit")
                for item in registrations
            )
        )
        for order, source in enumerate(_validated_order(sources, context)):
            declaration = source.registration.declaration
            before = _keys(context)
            context._active_plugin = declaration.plugin_id
            context._active_families = frozenset(item.family for item in declaration.families)
            try:
                result = source.registration.register(context)
                if inspect.isawaitable(result):
                    await result
            except BootstrapError:
                raise
            except Exception as exc:
                code = (
                    BootstrapErrorCode.NAMESPACE_COLLISION
                    if "already registered" in str(exc) or "duplicate" in str(exc).lower()
                    else BootstrapErrorCode.REGISTRATION_FAILURE
                )
                raise BootstrapError(
                    code,
                    f"plugin {declaration.plugin_id!r} registration failed: {exc}",
                    plugin_id=declaration.plugin_id,
                ) from exc
            after = _keys(context)
            changed = {
                family: tuple(sorted(set(values) - set(before[family])))
                for family, values in after.items()
                if values != before[family]
            }
            provenance.append(
                PluginProvenance(
                    declaration.plugin_id,
                    declaration.version,
                    source.distribution,
                    source.distribution_version,
                    source.fingerprint or _fingerprint(source.entry_point_value),
                    source.entry_point_name,
                    source.entry_point_value,
                    MappingProxyType(
                        {item.family: item.protocol_version for item in declaration.families}
                    ),
                    order,
                    MappingProxyType(changed),
                )
            )
        assert context._bundle is not None
        context._bundle.seal()
        return BootstrapState(context._bundle, tuple(provenance), context.context_version)
    except BaseException:
        failed_bundle = context._bundle
        context._bundle = None
        if failed_bundle is not None:
            failed_bundle.seal()
        raise
    finally:
        context._active_plugin = None
        context._active_families = frozenset()
        context._owner = None
        context._sealed = True


async def bootstrap_application(
    context: RegistrationContext,
    *,
    registrations: Sequence[PluginSource | PluginRegistration] | None = None,
    entry_points: Iterable[Any] | None = None,
    modules: Sequence[str] = (),
) -> BootstrapState:
    """Execute once per context; concurrent callers await the same result."""
    lineage = _BOOTSTRAP_LINEAGE.get()
    if id(context) in lineage or context._owner is asyncio.current_task():
        raise BootstrapError(
            BootstrapErrorCode.REENTRANCY,
            "same-context bootstrap is re-entrant",
            plugin_id=context._active_plugin,
        )
    if context._task is None:
        token = _BOOTSTRAP_LINEAGE.set(lineage | {id(context)})
        try:
            context._task = asyncio.create_task(
                _execute(context, registrations, entry_points, modules)
            )
        finally:
            _BOOTSTRAP_LINEAGE.reset(token)
    return await asyncio.shield(context._task)
