"""Versioned authority for finalizing compact products across evaluation matrices."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from feedbax.contracts.manifest import ArtifactRef, StrictModel


EVALUATION_COMPACT_PRODUCT_UNION_SCHEMA_ID = "feedbax.spec.evaluation_compact_product_union"
EVALUATION_COMPACT_PRODUCT_UNION_SCHEMA_VERSION = "feedbax.spec.evaluation_compact_product_union.v1"
EVALUATION_COMPACT_PRODUCT_UNION_EVIDENCE_SCHEMA_ID = (
    "feedbax.orchestration.evaluation_compact_product_union_evidence"
)
EVALUATION_COMPACT_PRODUCT_UNION_EVIDENCE_SCHEMA_VERSION = (
    "feedbax.orchestration.evaluation_compact_product_union_evidence.v1"
)


class EvaluationCompactProductUnionSource(StrictModel):
    """One explicitly ordered governed matrix source for a compact product union."""

    cohort_key: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    matrix_intent_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    consumer_id: str = Field(min_length=1)
    consumer_version: str = Field(min_length=1)
    leaf_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    compact_product_schema_id: str = Field(min_length=1)
    compact_product_schema_version: str = Field(min_length=1)
    compact_product_role: str = Field(min_length=1)
    ordered_row_ids: tuple[str, ...] = Field(min_length=1)
    compaction_evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    terminal_checkpoint_schema_id: str = Field(min_length=1)
    terminal_checkpoint_schema_version: str = Field(min_length=1)
    terminal_checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    terminal_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    terminal_product_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _validate_versioned_identities(self) -> "EvaluationCompactProductUnionSource":
        if self.compact_product_schema_id == self.compact_product_schema_version:
            raise ValueError("compact union source product schema must be versioned")
        if self.terminal_checkpoint_schema_id == self.terminal_checkpoint_schema_version:
            raise ValueError("compact union source checkpoint schema must be versioned")
        if len(self.ordered_row_ids) != len(set(self.ordered_row_ids)):
            raise ValueError("compact union source ordered row ids must be unique")
        return self


class EvaluationCompactProductUnion(StrictModel):
    """Content-addressable union intent over whole, distinct matrix identities."""

    schema_id: Literal["feedbax.spec.evaluation_compact_product_union"] = (
        EVALUATION_COMPACT_PRODUCT_UNION_SCHEMA_ID
    )
    schema_version: Literal["feedbax.spec.evaluation_compact_product_union.v1"] = (
        EVALUATION_COMPACT_PRODUCT_UNION_SCHEMA_VERSION
    )
    consumer_id: str = Field(min_length=1)
    consumer_version: str = Field(min_length=1)
    output_schema_id: str = Field(min_length=1)
    output_schema_version: str = Field(min_length=1)
    output_role: str = Field(min_length=1)
    output_logical_name: str = Field(min_length=1)
    sources: tuple[EvaluationCompactProductUnionSource, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def _validate_declared_sources(self) -> "EvaluationCompactProductUnion":
        if self.output_schema_id == self.output_schema_version:
            raise ValueError("compact union output schema must be versioned")
        cohort_keys = [source.cohort_key for source in self.sources]
        matrix_hashes = [source.matrix_intent_hash for source in self.sources]
        if len(cohort_keys) != len(set(cohort_keys)):
            raise ValueError("compact union cohort keys must be unique")
        if len(matrix_hashes) != len(set(matrix_hashes)):
            raise ValueError("compact union matrix identities must be unique")
        if any(
            source.consumer_id != self.consumer_id
            or source.consumer_version != self.consumer_version
            for source in self.sources
        ):
            raise ValueError("compact union sources must use the declared consumer identity")
        return self


class EvaluationCompactProductUnionSourceEvidence(StrictModel):
    """Verified materialized source retained in declared union order."""

    cohort_key: str = Field(min_length=1)
    matrix_intent_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    ordered_row_ids: tuple[str, ...] = Field(min_length=1)
    terminal_checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    terminal_product: ArtifactRef


class EvaluationCompactProductUnionEvidence(StrictModel):
    """Terminal publication and provider-free lifecycle proof for one union."""

    schema_id: Literal["feedbax.orchestration.evaluation_compact_product_union_evidence"] = (
        EVALUATION_COMPACT_PRODUCT_UNION_EVIDENCE_SCHEMA_ID
    )
    schema_version: Literal[
        "feedbax.orchestration.evaluation_compact_product_union_evidence.v1"
    ] = EVALUATION_COMPACT_PRODUCT_UNION_EVIDENCE_SCHEMA_VERSION
    union_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sources: tuple[EvaluationCompactProductUnionSourceEvidence, ...] = Field(min_length=2)
    terminal_product: ArtifactRef
    terminal_manifest: ArtifactRef
    provider_readiness: Literal["not_evaluated"] = "not_evaluated"
    completed_stages: tuple[Literal["UNION", "COLLECT", "CERTIFY", "TEARDOWN"], ...] = (
        "UNION",
        "COLLECT",
        "CERTIFY",
        "TEARDOWN",
    )

    @model_validator(mode="after")
    def _validate_terminal_lifecycle(self) -> "EvaluationCompactProductUnionEvidence":
        if self.completed_stages != ("UNION", "COLLECT", "CERTIFY", "TEARDOWN"):
            raise ValueError("compact union lifecycle must complete through teardown")
        cohort_keys = [source.cohort_key for source in self.sources]
        if len(cohort_keys) != len(set(cohort_keys)):
            raise ValueError("compact union evidence sources must be unique")
        return self


__all__ = [
    "EVALUATION_COMPACT_PRODUCT_UNION_EVIDENCE_SCHEMA_ID",
    "EVALUATION_COMPACT_PRODUCT_UNION_EVIDENCE_SCHEMA_VERSION",
    "EVALUATION_COMPACT_PRODUCT_UNION_SCHEMA_ID",
    "EVALUATION_COMPACT_PRODUCT_UNION_SCHEMA_VERSION",
    "EvaluationCompactProductUnion",
    "EvaluationCompactProductUnionEvidence",
    "EvaluationCompactProductUnionSource",
    "EvaluationCompactProductUnionSourceEvidence",
]
