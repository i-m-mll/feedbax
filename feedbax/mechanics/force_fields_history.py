"""Frozen force-field wire endpoints and parameter transformations."""

from collections.abc import Mapping
from typing import Any
from feedbax.component_registry.history import Change, History, LegacyEndpoint
from .force_fields import CurlField, ThresholdLatchedForce

THRESHOLD_LATCHED_FORCE_SCHEMA_VERSION_V1 = "feedbax.component.threshold_latched_force.v1"
THRESHOLD_LATCHED_FORCE_SCHEMA_VERSION_V2 = "feedbax.component.threshold_latched_force.v2"
THRESHOLD_LATCHED_FORCE_SCHEMA_VERSION = "feedbax.component.threshold_latched_force.v3"


def _threshold_v1_evidence(params: Mapping[str, Any]) -> bool:
    selector = params.get("state_selector")
    return (
        isinstance(selector, Mapping)
        and "path" in selector
        and "kind" not in selector
        and "lateral_force" not in params
    )


def _threshold_v2_evidence(params: Mapping[str, Any]) -> bool:
    selector = params.get("state_selector")
    return isinstance(selector, Mapping) and "kind" in selector and "label" in params


def _migrate_threshold_latched_force_v1(params: dict[str, Any]) -> dict[str, Any]:
    migrated = dict(params)
    selector = dict(migrated["state_selector"])
    selector["kind"] = "fixed"
    migrated["state_selector"] = selector
    migrated["lateral_force"] = 0.0
    return migrated


def _migrate_threshold_latched_force_v2(params: dict[str, Any]) -> dict[str, Any]:
    return {name: value for name, value in params.items() if name != "label"}


def _has_legacy_label(params):
    return "label" in params


def _drop_label(params):
    return {name: value for name, value in params.items() if name != "label"}


def _threshold_v3_evidence(params):
    selector = params.get("state_selector")
    return (
        isinstance(selector, Mapping)
        and "kind" in selector
        and "lateral_force" in params
        and "label" not in params
    )


def _curl_v2_evidence(params):
    return set(params) <= {"scale", "active", "amplitude"}


HISTORIES = {
    CurlField: History(
        Change(2, migrate=_drop_label),
        legacy=(
            LegacyEndpoint("CurlField", "1", 1, _has_legacy_label),
            LegacyEndpoint("CurlField", "2", 2, _curl_v2_evidence),
        ),
    ),
    ThresholdLatchedForce: History(
        Change(2, migrate=_migrate_threshold_latched_force_v1),
        Change(3, migrate=_migrate_threshold_latched_force_v2),
        legacy=(
            LegacyEndpoint(
                "ThresholdLatchedForce",
                THRESHOLD_LATCHED_FORCE_SCHEMA_VERSION_V1,
                1,
                _threshold_v1_evidence,
            ),
            LegacyEndpoint(
                "ThresholdLatchedForce",
                THRESHOLD_LATCHED_FORCE_SCHEMA_VERSION_V2,
                2,
                _threshold_v2_evidence,
            ),
            LegacyEndpoint("ThresholdLatchedForce", THRESHOLD_LATCHED_FORCE_SCHEMA_VERSION, 3),
        ),
    ),
}
