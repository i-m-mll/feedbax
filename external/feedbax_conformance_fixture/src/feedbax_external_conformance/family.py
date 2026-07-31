"""Fixture-owned registry family for the unified bootstrap conformance case."""

from __future__ import annotations

from feedbax.plugins import RegistryKey

EXTERNAL_DYNAMIC_COMPONENT = "feedbax_external_conformance.VariableFanIn"


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


FIXTURE_RECORDS = RegistryKey(
    family="feedbax_external_conformance.fixture_records",
    attribute="fixture_records",
    expected_type=FixtureRecordRegistry,
    registered_keys=lambda registry: registry.keys(),
)


__all__ = [
    "FIXTURE_RECORDS",
    "EXTERNAL_DYNAMIC_COMPONENT",
    "FixtureRecordRegistry",
]
