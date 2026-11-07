"""Frozen historical wire endpoints for native component owners."""

from typing import Any, Mapping

from feedbax.component_registry.history import Change, History, LegacyEndpoint
from feedbax.contracts.migrations import UnsupportedComponentMigration

from .affine_composer import AffineValueComposer

AFFINE_VALUE_COMPOSER_SCHEMA_VERSION_V1 = "feedbax.component.affine_value_composer.v1"
AFFINE_VALUE_COMPOSER_SCHEMA_VERSION_V2 = "feedbax.component.affine_value_composer.v2"
AFFINE_VALUE_COMPOSER_SCHEMA_VERSION_V3 = "feedbax.component.affine_value_composer.v3"


def _affine_v1_evidence(params: Mapping[str, Any]) -> bool:
    return params.get("schema_version") == AFFINE_VALUE_COMPOSER_SCHEMA_VERSION_V1 or (
        "schema_version" not in params and "label" in params
    )


def _affine_v2_evidence(params: Mapping[str, Any]) -> bool:
    return params.get("schema_version") == AFFINE_VALUE_COMPOSER_SCHEMA_VERSION_V2


def _migrate_affine_v1(params: dict[str, Any]) -> dict[str, Any]:
    if "schema_version" in params and not _affine_v1_evidence(params):
        raise UnsupportedComponentMigration(
            f"Conflicting AffineValueComposer v1 marker: {params['schema_version']!r}"
        )
    return {
        **{name: value for name, value in params.items() if name != "label"},
        "schema_version": AFFINE_VALUE_COMPOSER_SCHEMA_VERSION_V2,
    }


def _migrate_affine_v2(params: dict[str, Any]) -> dict[str, Any]:
    if "schema_version" in params and not _affine_v2_evidence(params):
        raise UnsupportedComponentMigration(
            f"Conflicting AffineValueComposer v2 marker: {params['schema_version']!r}"
        )
    return {name: value for name, value in params.items() if name != "schema_version"}


HISTORIES = {
    AffineValueComposer: History(
        Change(2, migrate=_migrate_affine_v1),
        Change(3, migrate=_migrate_affine_v2),
        legacy=(
            LegacyEndpoint(
                "AffineValueComposer",
                "feedbax.component.affine_value_composer.v1",
                1,
                matches_unversioned=_affine_v1_evidence,
            ),
            LegacyEndpoint(
                "AffineValueComposer",
                "feedbax.component.affine_value_composer.v2",
                2,
                matches_unversioned=_affine_v2_evidence,
            ),
            LegacyEndpoint("AffineValueComposer", "feedbax.component.affine_value_composer.v3", 3),
        ),
    ),
}
