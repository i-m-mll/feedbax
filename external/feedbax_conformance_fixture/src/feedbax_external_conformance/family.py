"""Fixture-owned registry family for the unified bootstrap conformance case."""

from __future__ import annotations

from collections.abc import MutableSequence
from dataclasses import dataclass

from feedbax.plugins import (
    APPLICATION_REGISTRY_KEYS,
    ApplicationRegistryBundle,
    RegistrationContext,
    RegistryKey,
)
from feedbax.plugins.application import new_application_registry_bundle


class FixtureRecordRegistry:
    """Small external family with the same mutation/sealing shape as Feedbax registries."""

    def __init__(self) -> None:
        self._records: list[str] = []
        self._sealed = False

    def register(self, record: str) -> None:
        if self._sealed:
            raise RuntimeError("fixture record registry is sealed")
        if record in self._records:
            raise ValueError(f"fixture record already registered: {record!r}")
        self._records.append(record)

    def keys(self) -> tuple[str, ...]:
        return tuple(self._records)

    def seal(self) -> None:
        self._sealed = True


@dataclass(frozen=True)
class FixtureApplicationRegistryBundle(ApplicationRegistryBundle):
    """Application bundle extended externally without changing Feedbax's loader."""

    fixture_records: FixtureRecordRegistry

    def seal(self) -> None:
        super().seal()
        self.fixture_records.seal()


FIXTURE_RECORDS = RegistryKey(
    family="feedbax_external_conformance.fixture_records",
    attribute="fixture_records",
    expected_type=FixtureRecordRegistry,
    registered_keys=lambda registry: registry.keys(),
)


def new_fixture_registration_context(
    *, registry_sink: MutableSequence[FixtureRecordRegistry] | None = None
) -> RegistrationContext:
    """Build one fresh application context containing the external family."""

    def factory() -> FixtureApplicationRegistryBundle:
        base = new_application_registry_bundle(local_component_source=None)
        registry = FixtureRecordRegistry()
        if registry_sink is not None:
            registry_sink.append(registry)
        return FixtureApplicationRegistryBundle(
            **base.__dict__,
            fixture_records=registry,
        )

    return RegistrationContext(factory, (*APPLICATION_REGISTRY_KEYS, FIXTURE_RECORDS))


__all__ = [
    "FIXTURE_RECORDS",
    "FixtureApplicationRegistryBundle",
    "FixtureRecordRegistry",
    "new_fixture_registration_context",
]
