"""Output-only evidence for the provider-free governed shadow launch."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import Field, field_validator

from feedbax.contracts.manifest import StrictModel


SHADOW_LAUNCH_EVIDENCE_SCHEMA_ID = "feedbax.orchestration.shadow_launch_evidence"
SHADOW_LAUNCH_EVIDENCE_SCHEMA_VERSION = f"{SHADOW_LAUNCH_EVIDENCE_SCHEMA_ID}.v1"


class ShadowLaunchRowEvidence(StrictModel):
    """The one authenticated native continuation transition exercised for a row."""

    row_id: str = Field(min_length=1)
    planned_run_id: str = Field(min_length=1)
    segment_completed_batches: Literal[1] = 1
    payload_binding_status: Literal["verified"] = "verified"


class ShadowLaunchEvidence(StrictModel):
    """Provider-free evidence that is deliberately not provider readiness authority."""

    schema_id: Literal["feedbax.orchestration.shadow_launch_evidence"] = (
        SHADOW_LAUNCH_EVIDENCE_SCHEMA_ID
    )
    schema_version: Literal["feedbax.orchestration.shadow_launch_evidence.v1"] = (
        SHADOW_LAUNCH_EVIDENCE_SCHEMA_VERSION
    )
    evidence_kind: Literal["provider_free_shadow_launch"] = "provider_free_shadow_launch"
    provider_readiness: Literal["not_evaluated"] = "not_evaluated"
    run_set_id: str = Field(min_length=1)
    bundle_sha256: str
    exercised_through_stage: Literal["COLLECT"] = "COLLECT"
    rows: tuple[ShadowLaunchRowEvidence, ...] = Field(min_length=1)

    @field_validator("bundle_sha256")
    @classmethod
    def _validate_bundle_sha256(cls, value: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("bundle_sha256 must be a lowercase sha256 digest")
        return value
