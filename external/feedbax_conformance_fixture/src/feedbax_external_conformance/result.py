"""Versioned machine-readable fixture result."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator


RESULT_SCHEMA_ID = "feedbax.external_conformance.result"
RESULT_SCHEMA_VERSION_V1 = f"{RESULT_SCHEMA_ID}.v1"
RESULT_SCHEMA_VERSION_V2 = f"{RESULT_SCHEMA_ID}.v2"
RESULT_SCHEMA_VERSION_V3 = f"{RESULT_SCHEMA_ID}.v3"
RESULT_SCHEMA_VERSION = f"{RESULT_SCHEMA_ID}.v4"
V2_REQUIRED_CASE_IDS = (
    "ordered_registration",
    "component_registration_and_migration",
    "value_identity",
    "material_dependencies",
    "staged_exact_parent_migration",
    "public_lifecycle_recovery",
)
REQUIRED_CASE_IDS = (
    *V2_REQUIRED_CASE_IDS[:-1],
    "typed_evaluation_row_projection",
    V2_REQUIRED_CASE_IDS[-1],
)
V3_REQUIRED_CASE_IDS = REQUIRED_CASE_IDS
_REQUIRED_CASE_ID_SET = frozenset(REQUIRED_CASE_IDS)
RESULT_SCHEMA_MIGRATION_TABLE = {
    RESULT_SCHEMA_VERSION_V1: (
        f"migrate to {RESULT_SCHEMA_VERSION_V2} by adding unbound protocol role slots; "
        f"then reject for {RESULT_SCHEMA_VERSION}"
    ),
    RESULT_SCHEMA_VERSION_V2: ("reject; v2 contains no typed_evaluation_row_projection evidence"),
    RESULT_SCHEMA_VERSION_V3: (
        "reject; v3 row-projection evidence did not require a resolver-issued "
        "state-materialization receipt"
    ),
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
    schema_version: Literal["feedbax.external_conformance.result.v4"] = RESULT_SCHEMA_VERSION
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
        if observed != _REQUIRED_CASE_ID_SET:
            raise ValueError(
                "external conformance cases must exactly match the v4 contract: "
                f"missing={sorted(_REQUIRED_CASE_ID_SET - observed)!r}, "
                f"extra={sorted(observed - _REQUIRED_CASE_ID_SET)!r}"
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
    """Load v4; reject older evidence that cannot prove the resolver receipt."""
    if isinstance(payload, ConformanceResult):
        return ConformanceResult.model_validate(payload.model_dump(mode="json"))
    data = dict(payload)
    if data.get("schema_id") != RESULT_SCHEMA_ID:
        raise ValueError(
            f"unsupported external conformance result schema_id: {data.get('schema_id')!r}"
        )
    version = data.get("schema_version")
    if version == RESULT_SCHEMA_VERSION_V1:
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
            "external conformance result v2 cannot migrate to v3: "
            "v2 contains no typed_evaluation_row_projection evidence"
        )
    if version == RESULT_SCHEMA_VERSION_V3:
        raise ValueError(
            "external conformance result v3 cannot migrate to v4: "
            "v3 typed_evaluation_row_projection evidence did not require a "
            "resolver-issued state-materialization receipt"
        )
    if version != RESULT_SCHEMA_VERSION:
        raise ValueError(
            "unsupported external conformance result schema_version: "
            f"{version!r}; expected {RESULT_SCHEMA_VERSION!r}; "
            f"migration table={RESULT_SCHEMA_MIGRATION_TABLE!r}"
        )
    return ConformanceResult.model_validate(data)


__all__ = [
    "RESULT_SCHEMA_ID",
    "RESULT_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION_V1",
    "RESULT_SCHEMA_VERSION_V2",
    "RESULT_SCHEMA_VERSION_V3",
    "RESULT_SCHEMA_MIGRATION_TABLE",
    "REQUIRED_CASE_IDS",
    "V2_REQUIRED_CASE_IDS",
    "V3_REQUIRED_CASE_IDS",
    "ConformanceResult",
    "LifecycleResult",
    "ProtocolRoleSlots",
    "load_result",
]
