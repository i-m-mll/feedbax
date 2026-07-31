"""Typed registrations exported through the single ``feedbax.plugins`` group."""

from __future__ import annotations

from feedbax.plugins import (
    FamilyRequirement,
    PluginDeclaration,
    PluginDependency,
    PluginRegistration,
    RegistrationContext,
)

from .family import FIXTURE_RECORDS


_FOUNDATION_PLUGIN_ID = "feedbax_external_conformance.foundation"
_DEPENDENT_PLUGIN_ID = "feedbax_external_conformance.dependent"


def _register_foundation(context: RegistrationContext) -> None:
    context.registry(FIXTURE_RECORDS).register("foundation")


def _register_dependent(context: RegistrationContext) -> None:
    context.registry(FIXTURE_RECORDS).register("dependent")


FOUNDATION_PLUGIN_REGISTRATION = PluginRegistration(
    declaration=PluginDeclaration(
        plugin_id=_FOUNDATION_PLUGIN_ID,
        version="1",
        families=(FamilyRequirement(FIXTURE_RECORDS.family),),
    ),
    register=_register_foundation,
)

DEPENDENT_PLUGIN_REGISTRATION = PluginRegistration(
    declaration=PluginDeclaration(
        plugin_id=_DEPENDENT_PLUGIN_ID,
        version="1",
        dependencies=(PluginDependency(_FOUNDATION_PLUGIN_ID, "1"),),
        families=(FamilyRequirement(FIXTURE_RECORDS.family),),
    ),
    register=_register_dependent,
)


__all__ = ["DEPENDENT_PLUGIN_REGISTRATION", "FOUNDATION_PLUGIN_REGISTRATION"]
