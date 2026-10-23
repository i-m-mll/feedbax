"""Application-process composition around the typed plugin bootstrap."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, TypeVar

from .application import APPLICATION_REGISTRY_KEYS, new_application_registry_bundle
from .bootstrap import (
    BootstrapError,
    BootstrapErrorCode,
    BootstrapState,
    PluginSource,
    RegistrationContext,
    RegistryFamilyRegistration,
    bootstrap_application,
    discover_plugin_registrations,
)

T = TypeVar("T")


def _extension_family_providers(
    sources: Sequence[PluginSource],
) -> tuple[RegistryFamilyRegistration[Any], ...]:
    """Validate and collect one canonical runtime provider per extension family."""
    core_keys = {key.family: key for key in APPLICATION_REGISTRY_KEYS}
    providers: dict[str, RegistryFamilyRegistration[Any]] = {}
    seen_registrations: set[int] = set()
    for source in sources:
        registration = source.registration
        if id(registration) in seen_registrations:
            continue
        seen_registrations.add(id(registration))
        requirements = {
            item.family: item.protocol_version for item in registration.declaration.families
        }
        if len(requirements) != len(registration.declaration.families):
            raise BootstrapError(
                BootstrapErrorCode.INVALID_REGISTRATION,
                f"plugin {registration.declaration.plugin_id!r} declares a family more than once",
                plugin_id=registration.declaration.plugin_id,
            )
        for provider in registration.registry_families:
            key = provider.key
            required_protocol = requirements.get(key.family)
            if required_protocol is None:
                raise BootstrapError(
                    BootstrapErrorCode.INVALID_REGISTRATION,
                    f"plugin {registration.declaration.plugin_id!r} provides undeclared "
                    f"family {key.family!r}",
                    plugin_id=registration.declaration.plugin_id,
                )
            if required_protocol != key.protocol_version:
                raise BootstrapError(
                    BootstrapErrorCode.UNSUPPORTED_PROTOCOL,
                    f"plugin {registration.declaration.plugin_id!r} provides family "
                    f"{key.family!r} with protocol {key.protocol_version!r}, not "
                    f"declared protocol {required_protocol!r}",
                    plugin_id=registration.declaration.plugin_id,
                )
            if key.family in core_keys:
                raise BootstrapError(
                    BootstrapErrorCode.INVALID_REGISTRATION,
                    f"plugin {registration.declaration.plugin_id!r} duplicates core family "
                    f"{key.family!r}",
                    plugin_id=registration.declaration.plugin_id,
                )
            existing = providers.get(key.family)
            if existing is not None:
                if existing.key is key:
                    detail = "duplicate provider"
                elif existing.key.protocol_version != key.protocol_version:
                    detail = "incompatible protocol"
                elif existing.key.expected_type is not key.expected_type:
                    detail = "incompatible registry type"
                else:
                    detail = "incompatible canonical registry key"
                code = (
                    BootstrapErrorCode.UNSUPPORTED_PROTOCOL
                    if detail == "incompatible protocol"
                    else BootstrapErrorCode.INVALID_REGISTRATION
                )
                raise BootstrapError(
                    code,
                    f"extension family {key.family!r} has {detail}",
                    plugin_id=registration.declaration.plugin_id,
                )
            providers[key.family] = provider
    return tuple(providers.values())


async def compose_application(
    *,
    modules: Sequence[str] = (),
    entry_points: Iterable[Any] | None = None,
    local_component_source: Path | None = Path.home() / ".feedbax" / "components",
) -> BootstrapState:
    """Discover and bootstrap one isolated application registry state."""
    sources = discover_plugin_registrations(entry_points=entry_points, modules=modules)
    family_providers = _extension_family_providers(sources)
    context = RegistrationContext(
        lambda: new_application_registry_bundle(local_component_source=local_component_source),
        APPLICATION_REGISTRY_KEYS,
        registry_families=family_providers,
    )
    return await bootstrap_application(
        context,
        registrations=sources,
    )


async def bootstrap_and_dispatch(
    dispatch: Callable[[BootstrapState], T],
    *,
    modules: Sequence[str] = (),
    local_component_source: Path | None = Path.home() / ".feedbax" / "components",
) -> T:
    """Bootstrap once, then invoke a synchronous process dispatcher."""
    state = await compose_application(
        modules=modules,
        local_component_source=local_component_source,
    )
    return dispatch(state)
