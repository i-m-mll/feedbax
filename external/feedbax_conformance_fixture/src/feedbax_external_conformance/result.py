"""Versioned machine-readable fixture result."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator


RESULT_SCHEMA_ID = "feedbax.external_conformance.result"
_RESULT_SCHEMA_VERSION_V1 = f"{RESULT_SCHEMA_ID}.v1"
RESULT_SCHEMA_VERSION_V2 = f"{RESULT_SCHEMA_ID}.v2"
RESULT_SCHEMA_VERSION_V7 = f"{RESULT_SCHEMA_ID}.v7"
RESULT_SCHEMA_VERSION_V8 = f"{RESULT_SCHEMA_ID}.v8"
RESULT_SCHEMA_VERSION_V9 = f"{RESULT_SCHEMA_ID}.v9"
RESULT_SCHEMA_VERSION_V10 = f"{RESULT_SCHEMA_ID}.v10"
RESULT_SCHEMA_VERSION = f"{RESULT_SCHEMA_ID}.v11"
REJECTED_UNSHIPPED_SCHEMA_VERSIONS = (
    f"{RESULT_SCHEMA_ID}.v3",
    f"{RESULT_SCHEMA_ID}.v4",
    f"{RESULT_SCHEMA_ID}.v5",
    f"{RESULT_SCHEMA_ID}.v6",
)
V2_REQUIRED_CASE_IDS = (
    "ordered_registration",
    "component_registration_and_migration",
    "value_identity",
    "material_dependencies",
    "staged_exact_parent_migration",
    "public_lifecycle_recovery",
)
V2_REQUIRED_CASE_ID_SET = frozenset(V2_REQUIRED_CASE_IDS)
REQUIRED_CASE_IDS = (
    "ordered_registration",
    "unified_plugin_bootstrap",
    "external_driver_plugin",
    "component_registration_and_migration",
    "dynamic_component_ports",
    "value_identity",
    "component_param_array_values",
    "material_dependencies",
    "staged_exact_parent_migration",
    "resolved_evaluation_row_projection",
    "public_lifecycle_recovery",
)
REQUIRED_CASE_ID_SET = frozenset(REQUIRED_CASE_IDS)
RESULT_SCHEMA_MIGRATION_TABLE = {
    _RESULT_SCHEMA_VERSION_V1: (
        f"migrate to {RESULT_SCHEMA_VERSION_V2} by adding unbound protocol role slots; "
        f"then reject for {RESULT_SCHEMA_VERSION}"
    ),
    RESULT_SCHEMA_VERSION_V2: (
        "reject; protected v2 contains neither current projection nor array-value evidence"
    ),
    "v3-v6": (
        "reject; unshipped broad projection evidence is not reinterpreted as the "
        "current narrowed resolver-handle contract"
    ),
    RESULT_SCHEMA_VERSION_V7: (
        "reject; shipped v7 contains no component_param_array_values evidence"
    ),
    RESULT_SCHEMA_VERSION_V8: ("reject; shipped v8 contains no unified_plugin_bootstrap evidence"),
    RESULT_SCHEMA_VERSION_V9: ("reject; shipped v9 contains no dynamic_component_ports evidence"),
    RESULT_SCHEMA_VERSION_V10: ("reject; shipped v10 contains no external_driver_plugin evidence"),
}


class ProtocolRoleSlots(BaseModel):
    """Unratified slots reserved for later stability-policy bindings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    current: Literal[None]
    minimum: Literal[None]


class LifecycleResult(BaseModel):
    """Truthful state of the clean-wheel production lifecycle case."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["pass", "blocked"]
    reason_code: str | None = None


class ConformanceResult(BaseModel):
    """Strict current result emitted by the external fixture."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["feedbax.external_conformance.result"] = RESULT_SCHEMA_ID
    schema_version: Literal["feedbax.external_conformance.result.v11"] = RESULT_SCHEMA_VERSION
    status: Literal["pass", "blocked"]
    feedbax_version: str = Field(min_length=1)
    feedbax_install_root: str = Field(min_length=1)
    fixture_install_root: str = Field(min_length=1)
    protocol_roles: ProtocolRoleSlots
    cases: dict[str, StrictBool]
    lifecycle: LifecycleResult

    @model_validator(mode="after")
    def _validate_outcome(self) -> "ConformanceResult":
        observed = frozenset(self.cases)
        if observed != REQUIRED_CASE_ID_SET:
            raise ValueError(
                "external conformance cases must exactly match the v11 contract: "
                f"missing={sorted(REQUIRED_CASE_ID_SET - observed)!r}, "
                f"extra={sorted(observed - REQUIRED_CASE_ID_SET)!r}"
            )
        if not all(self.cases.values()):
            raise ValueError("every required external conformance case must pass")
        if self.status != self.lifecycle.status:
            raise ValueError("result and lifecycle status must match")
        if self.status == "blocked" and not self.lifecycle.reason_code:
            raise ValueError("blocked lifecycle requires a reason_code")
        if self.status == "pass" and self.lifecycle.reason_code is not None:
            raise ValueError("passing lifecycle must not carry a reason_code")
        return self


def load_result(payload: ConformanceResult | dict[str, Any]) -> ConformanceResult:
    """Load v11 while preserving explicit decisions for older incomplete evidence."""

    if isinstance(payload, ConformanceResult):
        return ConformanceResult.model_validate(payload.model_dump(mode="json"))
    data = dict(payload)
    if data.get("schema_id") != RESULT_SCHEMA_ID:
        raise ValueError(
            f"unsupported external conformance result schema_id: {data.get('schema_id')!r}"
        )
    version = data.get("schema_version")
    if version == _RESULT_SCHEMA_VERSION_V1:
        if "protocol_roles" in data:
            raise ValueError(
                "external conformance result v1 did not define protocol_roles; "
                "remove the ambiguous field before migration"
            )
        data["schema_version"] = RESULT_SCHEMA_VERSION_V2
        data["protocol_roles"] = {"current": None, "minimum": None}
        version = RESULT_SCHEMA_VERSION_V2
    if version == RESULT_SCHEMA_VERSION_V2:
        raise ValueError(
            "external conformance result v2 cannot migrate to v11: protected v2 "
            "contains neither resolved_evaluation_row_projection, "
            "component_param_array_values, nor unified_plugin_bootstrap evidence"
        )
    if version in REJECTED_UNSHIPPED_SCHEMA_VERSIONS:
        raise ValueError(
            f"external conformance result {version!r} is rejected: unshipped v3-v6 "
            "projection evidence is not the narrowed v7 resolver-handle contract"
        )
    if version == RESULT_SCHEMA_VERSION_V7:
        raise ValueError(
            "external conformance result v7 cannot migrate to v11: v7 contains no "
            "component_param_array_values evidence"
        )
    if version == RESULT_SCHEMA_VERSION_V8:
        raise ValueError(
            "external conformance result v8 cannot migrate to v11: v8 contains no "
            "unified_plugin_bootstrap evidence"
        )
    if version == RESULT_SCHEMA_VERSION_V9:
        raise ValueError(
            "external conformance result v9 cannot migrate to v11: v9 contains no "
            "dynamic_component_ports evidence"
        )
    if version == RESULT_SCHEMA_VERSION_V10:
        raise ValueError(
            "external conformance result v10 cannot migrate to v11: v10 contains no "
            "external_driver_plugin evidence"
        )
    if version != RESULT_SCHEMA_VERSION:
        raise ValueError(
            "unsupported external conformance result schema_version: "
            f"{version!r}; expected {RESULT_SCHEMA_VERSION!r}; "
            f"migration table={RESULT_SCHEMA_MIGRATION_TABLE!r}"
        )
    return ConformanceResult.model_validate(data)


__all__ = [
    "ConformanceResult",
    "REJECTED_UNSHIPPED_SCHEMA_VERSIONS",
    "REQUIRED_CASE_IDS",
    "REQUIRED_CASE_ID_SET",
    "RESULT_SCHEMA_ID",
    "RESULT_SCHEMA_MIGRATION_TABLE",
    "RESULT_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION_V2",
    "RESULT_SCHEMA_VERSION_V7",
    "RESULT_SCHEMA_VERSION_V8",
    "RESULT_SCHEMA_VERSION_V9",
    "RESULT_SCHEMA_VERSION_V10",
    "V2_REQUIRED_CASE_IDS",
    "V2_REQUIRED_CASE_ID_SET",
    "load_result",
]
