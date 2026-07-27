"""Governed terminal evidence for one evaluation-matrix executor invocation."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from feedbax.contracts.manifest import StrictModel


EVALUATION_LIFECYCLE_EVIDENCE_SCHEMA_ID = (
    "feedbax.orchestration.evaluation_lifecycle_evidence"
)
EVALUATION_LIFECYCLE_EVIDENCE_SCHEMA_VERSION = (
    "feedbax.orchestration.evaluation_lifecycle_evidence.v1"
)
EVALUATION_SHADOW_LAUNCH_EVIDENCE_SCHEMA_ID = (
    "feedbax.orchestration.evaluation_shadow_launch_evidence"
)
EVALUATION_SHADOW_LAUNCH_EVIDENCE_SCHEMA_VERSION = (
    "feedbax.orchestration.evaluation_shadow_launch_evidence.v1"
)


class EvaluationLifecycleRowOutcome(StrictModel):
    """One authored matrix row retained inside the governed executor outcome."""

    row_id: str = Field(min_length=1)
    manifest_id: str = Field(min_length=1)
    manifest_path: str = Field(min_length=1)
    status: Literal["completed"] = "completed"
    diagnostic_schema_ids: tuple[str, ...] = ()


class EvaluationLifecycleEvidence(StrictModel):
    """Ordered whole-matrix result collected by the common orchestration lifecycle."""

    schema_id: Literal["feedbax.orchestration.evaluation_lifecycle_evidence"] = (
        EVALUATION_LIFECYCLE_EVIDENCE_SCHEMA_ID
    )
    schema_version: Literal[
        "feedbax.orchestration.evaluation_lifecycle_evidence.v1"
    ] = EVALUATION_LIFECYCLE_EVIDENCE_SCHEMA_VERSION
    executor_family: Literal["evaluation-matrix"] = "evaluation-matrix"
    orchestration_row_id: str = Field(min_length=1)
    ordered_row_ids: tuple[str, ...] = Field(min_length=1)
    outcomes: tuple[EvaluationLifecycleRowOutcome, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _bind_ordered_outcomes(self) -> "EvaluationLifecycleEvidence":
        observed = tuple(item.row_id for item in self.outcomes)
        if observed != self.ordered_row_ids:
            raise ValueError("evaluation lifecycle outcomes must preserve ordered_row_ids")
        if len(observed) != len(set(observed)):
            raise ValueError("evaluation lifecycle row ids must be unique")
        return self


class EvaluationShadowLaunchEvidence(StrictModel):
    """Provider-free traversal evidence for the evaluation executor family."""

    schema_id: Literal["feedbax.orchestration.evaluation_shadow_launch_evidence"] = (
        EVALUATION_SHADOW_LAUNCH_EVIDENCE_SCHEMA_ID
    )
    schema_version: Literal[
        "feedbax.orchestration.evaluation_shadow_launch_evidence.v1"
    ] = EVALUATION_SHADOW_LAUNCH_EVIDENCE_SCHEMA_VERSION
    evidence_kind: Literal["provider_free_shadow_launch"] = "provider_free_shadow_launch"
    provider_readiness: Literal["not_evaluated"] = "not_evaluated"
    run_set_id: str = Field(min_length=1)
    bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    exercised_through_stage: Literal["TEARDOWN"] = "TEARDOWN"
    lifecycle: EvaluationLifecycleEvidence


__all__ = [
    "EVALUATION_LIFECYCLE_EVIDENCE_SCHEMA_ID",
    "EVALUATION_LIFECYCLE_EVIDENCE_SCHEMA_VERSION",
    "EVALUATION_SHADOW_LAUNCH_EVIDENCE_SCHEMA_ID",
    "EVALUATION_SHADOW_LAUNCH_EVIDENCE_SCHEMA_VERSION",
    "EvaluationLifecycleEvidence",
    "EvaluationLifecycleRowOutcome",
    "EvaluationShadowLaunchEvidence",
]
