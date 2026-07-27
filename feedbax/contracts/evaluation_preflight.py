"""Typed pre-output cardinality and retained-storage evaluation contracts."""

from __future__ import annotations

from pydantic import Field, model_validator

from feedbax.contracts.manifest import StrictModel


EVALUATION_OUTPUT_PREFLIGHT_POLICY_SCHEMA_ID = "feedbax.spec.evaluation_output_preflight_policy"
EVALUATION_OUTPUT_PREFLIGHT_POLICY_SCHEMA_VERSION = (
    "feedbax.spec.evaluation_output_preflight_policy.v1"
)
EVALUATION_OUTPUT_PREFLIGHT_EVIDENCE_SCHEMA_ID = (
    "feedbax.orchestration.evaluation_output_preflight_evidence"
)
EVALUATION_OUTPUT_PREFLIGHT_EVIDENCE_SCHEMA_VERSION = (
    "feedbax.orchestration.evaluation_output_preflight_evidence.v1"
)


class EvaluationOutputPreflightPolicy(StrictModel):
    """Authored expectations required to budget retained evaluation output."""

    schema_id: str = EVALUATION_OUTPUT_PREFLIGHT_POLICY_SCHEMA_ID
    schema_version: str = EVALUATION_OUTPUT_PREFLIGHT_POLICY_SCHEMA_VERSION
    expected_resolved_row_count: int = Field(ge=1, strict=True)
    retained_bytes_per_resolved_row: int = Field(ge=1, strict=True)
    retained_bytes_per_resolved_row_source: str = Field(min_length=1)
    planned_repetitions: int = Field(ge=1, strict=True)
    required_free_space_reserve_bytes: int = Field(ge=0, strict=True)

    @model_validator(mode="after")
    def _validate_schema(self) -> "EvaluationOutputPreflightPolicy":
        if self.schema_id != EVALUATION_OUTPUT_PREFLIGHT_POLICY_SCHEMA_ID:
            raise ValueError("unsupported evaluation output preflight policy schema_id")
        if self.schema_version != EVALUATION_OUTPUT_PREFLIGHT_POLICY_SCHEMA_VERSION:
            raise ValueError("unsupported evaluation output preflight policy schema_version")
        return self


class EvaluationOutputPreflightEvidence(StrictModel):
    """Observed cardinality and filesystem-capacity decision retained in a bundle."""

    schema_id: str = EVALUATION_OUTPUT_PREFLIGHT_EVIDENCE_SCHEMA_ID
    schema_version: str = EVALUATION_OUTPUT_PREFLIGHT_EVIDENCE_SCHEMA_VERSION
    expected_resolved_row_count: int = Field(ge=1, strict=True)
    resolved_row_count: int = Field(ge=1, strict=True)
    retained_bytes_per_resolved_row: int = Field(ge=1, strict=True)
    retained_bytes_per_resolved_row_source: str = Field(min_length=1)
    planned_repetitions: int = Field(ge=1, strict=True)
    estimated_retained_bytes: int = Field(ge=1, strict=True)
    required_free_space_reserve_bytes: int = Field(ge=0, strict=True)
    required_free_bytes: int = Field(ge=1, strict=True)
    observed_free_bytes: int = Field(ge=0, strict=True)
    output_root: str = Field(min_length=1)
    observed_filesystem_path: str = Field(min_length=1)
    observed_filesystem_device: int = Field(ge=0, strict=True)

    @model_validator(mode="after")
    def _validate_evidence(self) -> "EvaluationOutputPreflightEvidence":
        if self.schema_id != EVALUATION_OUTPUT_PREFLIGHT_EVIDENCE_SCHEMA_ID:
            raise ValueError("unsupported evaluation output preflight evidence schema_id")
        if self.schema_version != EVALUATION_OUTPUT_PREFLIGHT_EVIDENCE_SCHEMA_VERSION:
            raise ValueError("unsupported evaluation output preflight evidence schema_version")
        if self.resolved_row_count != self.expected_resolved_row_count:
            raise ValueError("resolved row count does not match the authored expectation")
        expected_estimate = (
            self.resolved_row_count
            * self.retained_bytes_per_resolved_row
            * self.planned_repetitions
        )
        if self.estimated_retained_bytes != expected_estimate:
            raise ValueError("estimated_retained_bytes does not match retained-write inputs")
        if (
            self.required_free_bytes
            != self.estimated_retained_bytes + self.required_free_space_reserve_bytes
        ):
            raise ValueError("required_free_bytes does not equal estimate plus reserve")
        if self.observed_free_bytes < self.required_free_bytes:
            raise ValueError("observed_free_bytes does not satisfy required_free_bytes")
        return self


__all__ = [
    "EVALUATION_OUTPUT_PREFLIGHT_EVIDENCE_SCHEMA_ID",
    "EVALUATION_OUTPUT_PREFLIGHT_EVIDENCE_SCHEMA_VERSION",
    "EVALUATION_OUTPUT_PREFLIGHT_POLICY_SCHEMA_ID",
    "EVALUATION_OUTPUT_PREFLIGHT_POLICY_SCHEMA_VERSION",
    "EvaluationOutputPreflightEvidence",
    "EvaluationOutputPreflightPolicy",
]
