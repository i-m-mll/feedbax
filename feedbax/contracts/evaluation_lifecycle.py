"""Governed terminal evidence for one evaluation-matrix executor invocation."""

from __future__ import annotations

import json
from typing import Literal

from pydantic import Field, JsonValue, model_validator

from feedbax.contracts.manifest import (
    AUTHENTICATED_MANIFEST_REF_SCHEMA_ID,
    AUTHENTICATED_MANIFEST_REF_SCHEMA_VERSION,
    ArtifactRef,
    ParentRef,
    StrictModel,
    canonical_json_bytes,
)


EVALUATION_LIFECYCLE_EVIDENCE_SCHEMA_ID = "feedbax.orchestration.evaluation_lifecycle_evidence"
EVALUATION_LIFECYCLE_EVIDENCE_SCHEMA_VERSION = (
    "feedbax.orchestration.evaluation_lifecycle_evidence.v1"
)
EVALUATION_SHADOW_LAUNCH_EVIDENCE_SCHEMA_ID = (
    "feedbax.orchestration.evaluation_shadow_launch_evidence"
)
EVALUATION_SHADOW_LAUNCH_EVIDENCE_SCHEMA_VERSION = (
    "feedbax.orchestration.evaluation_shadow_launch_evidence.v2"
)
EVALUATION_SHADOW_LAUNCH_EVIDENCE_SCHEMA_VERSION_V1 = (
    "feedbax.orchestration.evaluation_shadow_launch_evidence.v1"
)
EVALUATION_MATRIX_BATCH_PLAN_SCHEMA_ID = "feedbax.spec.evaluation_matrix_batch_plan"
EVALUATION_MATRIX_BATCH_PLAN_SCHEMA_VERSION_V1 = "feedbax.spec.evaluation_matrix_batch_plan.v1"
EVALUATION_MATRIX_BATCH_PLAN_SCHEMA_VERSION_V2 = "feedbax.spec.evaluation_matrix_batch_plan.v2"
EVALUATION_MATRIX_BATCH_PLAN_SCHEMA_VERSION_V3 = "feedbax.spec.evaluation_matrix_batch_plan.v3"
EVALUATION_MATRIX_BATCH_PLAN_SCHEMA_VERSION = "feedbax.spec.evaluation_matrix_batch_plan.v4"
EVALUATION_BATCH_COMPACTION_EVIDENCE_SCHEMA_ID = (
    "feedbax.orchestration.evaluation_batch_compaction_evidence"
)
EVALUATION_BATCH_COMPACTION_EVIDENCE_SCHEMA_VERSION_V1 = (
    "feedbax.orchestration.evaluation_batch_compaction_evidence.v1"
)
EVALUATION_BATCH_COMPACTION_EVIDENCE_SCHEMA_VERSION_V2 = (
    "feedbax.orchestration.evaluation_batch_compaction_evidence.v2"
)
EVALUATION_BATCH_COMPACTION_EVIDENCE_SCHEMA_VERSION = (
    "feedbax.orchestration.evaluation_batch_compaction_evidence.v3"
)
EVALUATION_BATCH_MERGE_CHECKPOINT_SCHEMA_ID = (
    "feedbax.orchestration.evaluation_batch_merge_checkpoint"
)
EVALUATION_BATCH_MERGE_CHECKPOINT_SCHEMA_VERSION_V1 = (
    "feedbax.orchestration.evaluation_batch_merge_checkpoint.v1"
)
EVALUATION_BATCH_MERGE_CHECKPOINT_SCHEMA_VERSION_V2 = (
    "feedbax.orchestration.evaluation_batch_merge_checkpoint.v2"
)
EVALUATION_BATCH_MERGE_CHECKPOINT_SCHEMA_VERSION = (
    "feedbax.orchestration.evaluation_batch_merge_checkpoint.v3"
)
EVALUATION_WORKER_TOPOLOGY_EVIDENCE_SCHEMA_ID = (
    "feedbax.orchestration.evaluation_worker_topology_evidence"
)
EVALUATION_WORKER_TOPOLOGY_EVIDENCE_SCHEMA_VERSION = (
    "feedbax.orchestration.evaluation_worker_topology_evidence.v1"
)
EVALUATION_MATRIX_ORDERED_UNION_EVIDENCE_SCHEMA_ID = (
    "feedbax.orchestration.evaluation_matrix_ordered_union_evidence"
)
EVALUATION_MATRIX_ORDERED_UNION_EVIDENCE_SCHEMA_VERSION = (
    "feedbax.orchestration.evaluation_matrix_ordered_union_evidence.v1"
)
EVALUATION_COLLECTION_OUTPUTS = (
    "evaluation-matrix-result.json",
    "evaluation-worker-topology.json",
    "evaluation-batch-compaction.json",
    "evaluation-batch-compaction",
)


def _require_canonical_json_parameters(parameters: dict[str, JsonValue]) -> None:
    try:
        json.dumps(
            parameters,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("consumer parameters must be canonical JSON") from exc


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
    schema_version: Literal["feedbax.orchestration.evaluation_lifecycle_evidence.v1"] = (
        EVALUATION_LIFECYCLE_EVIDENCE_SCHEMA_VERSION
    )
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


class EvaluationMatrixBatchUnit(StrictModel):
    """One ordered subset assigned inside the governed public matrix harness."""

    batch_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    ordered_row_ids: tuple[str, ...] = Field(min_length=1)
    required_leaf_ids: tuple[str, ...] | None = None

    @model_validator(mode="after")
    def _unique_rows(self) -> "EvaluationMatrixBatchUnit":
        if len(self.ordered_row_ids) != len(set(self.ordered_row_ids)):
            raise ValueError("evaluation batch unit row ids must be unique")
        if self.required_leaf_ids is not None and len(self.required_leaf_ids) != len(
            set(self.required_leaf_ids)
        ):
            raise ValueError("evaluation batch unit required leaf ids must be unique")
        return self


class EvaluationBatchConsumerDeclaration(StrictModel):
    """Authored terminal compact leaf required before raw-cache reclamation."""

    leaf_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    consumer_id: str = Field(min_length=1)
    consumer_version: str = Field(min_length=1)
    terminal_analysis_type: str = Field(min_length=1)
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    accepted_evaluation_state_schema_ids: tuple[str, ...] = Field(min_length=1)
    compact_product_schema_id: str = Field(min_length=1)
    compact_product_schema_version: str = Field(min_length=1)
    compact_product_role: str = Field(min_length=1)
    merge_state_schema_id: str = Field(min_length=1)
    merge_state_schema_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_versions_and_schemas(self) -> "EvaluationBatchConsumerDeclaration":
        _require_canonical_json_parameters(self.parameters)
        if self.compact_product_schema_version == self.compact_product_schema_id:
            raise ValueError("compact product schema_version must be a versioned identity")
        if self.merge_state_schema_version == self.merge_state_schema_id:
            raise ValueError("merge state schema_version must be a versioned identity")
        if len(self.accepted_evaluation_state_schema_ids) != len(
            set(self.accepted_evaluation_state_schema_ids)
        ):
            raise ValueError("accepted evaluation-state schema ids must be unique")
        return self


class EvaluationMatrixBatchPlan(StrictModel):
    """Versioned partition of one matrix identity across persistent workers."""

    schema_id: Literal["feedbax.spec.evaluation_matrix_batch_plan"] = (
        EVALUATION_MATRIX_BATCH_PLAN_SCHEMA_ID
    )
    schema_version: Literal["feedbax.spec.evaluation_matrix_batch_plan.v4"] = (
        EVALUATION_MATRIX_BATCH_PLAN_SCHEMA_VERSION
    )
    matrix_intent_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    batches: tuple[EvaluationMatrixBatchUnit, ...] = Field(min_length=1)
    consumers: tuple[EvaluationBatchConsumerDeclaration, ...] = ()

    @model_validator(mode="after")
    def _unique_batches_and_rows(self) -> "EvaluationMatrixBatchPlan":
        batch_ids = [item.batch_id for item in self.batches]
        if len(batch_ids) != len(set(batch_ids)):
            raise ValueError("evaluation batch plan batch ids must be unique")
        row_ids = [row_id for item in self.batches for row_id in item.ordered_row_ids]
        if len(row_ids) != len(set(row_ids)):
            raise ValueError("evaluation batch plan row ids must be globally unique")
        leaf_ids = [item.leaf_id for item in self.consumers]
        if len(leaf_ids) != len(set(leaf_ids)):
            raise ValueError("evaluation batch consumer leaf ids must be unique")
        declared = set(leaf_ids)
        for batch in self.batches:
            unknown = set(batch.required_leaf_ids or ()) - declared
            if unknown:
                raise ValueError(
                    f"evaluation batch {batch.batch_id!r} names unknown required leaves: "
                    f"{sorted(unknown)!r}"
                )
            if self.consumers and not batch.required_leaf_ids:
                raise ValueError(
                    f"evaluation batch {batch.batch_id!r} requires no compact terminal leaves"
                )
        used = {leaf_id for batch in self.batches for leaf_id in (batch.required_leaf_ids or ())}
        unused = declared - used
        if unused:
            raise ValueError(f"evaluation batch consumer leaves are unused: {sorted(unused)!r}")
        return self


class EvaluationBatchLeafAcknowledgement(StrictModel):
    """Verified compact fragment and authored-order merge transition for one leaf."""

    leaf_id: str = Field(min_length=1)
    consumer_id: str = Field(min_length=1)
    consumer_version: str = Field(min_length=1)
    terminal_analysis_type: str = Field(min_length=1)
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    compact_product_schema_id: str = Field(min_length=1)
    compact_product_schema_version: str = Field(min_length=1)
    compact_product_role: str = Field(min_length=1)
    fragment: ArtifactRef
    prior_merge_state_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    merge_state: ArtifactRef
    reused_verified_fragment: bool = False

    @model_validator(mode="after")
    def _validate_parameters(self) -> "EvaluationBatchLeafAcknowledgement":
        _require_canonical_json_parameters(self.parameters)
        return self


class EvaluationBatchMergeCheckpoint(StrictModel):
    """Durable identity envelope for one exactly-once authored merge transition."""

    schema_id: Literal["feedbax.orchestration.evaluation_batch_merge_checkpoint"] = (
        EVALUATION_BATCH_MERGE_CHECKPOINT_SCHEMA_ID
    )
    schema_version: Literal["feedbax.orchestration.evaluation_batch_merge_checkpoint.v3"] = (
        EVALUATION_BATCH_MERGE_CHECKPOINT_SCHEMA_VERSION
    )
    matrix_intent_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    batch: EvaluationMatrixBatchUnit
    declaration: EvaluationBatchConsumerDeclaration
    parent_authorities: tuple[ParentRef, ...] = Field(min_length=1)
    acknowledgement: EvaluationBatchLeafAcknowledgement

    @model_validator(mode="after")
    def _bind_authenticated_batch(self) -> "EvaluationBatchMergeCheckpoint":
        if len(self.parent_authorities) != len(self.batch.ordered_row_ids):
            raise ValueError("merge checkpoint requires one parent authority per authored row")
        parent_ids = [parent.id for parent in self.parent_authorities]
        if len(parent_ids) != len(set(parent_ids)):
            raise ValueError("merge checkpoint parent authorities must be unique")
        for parent in self.parent_authorities:
            if (
                parent.kind != "EvaluationRunManifest"
                or parent.role != "evaluation_run"
                or parent.metadata.get("ref_schema_id") != AUTHENTICATED_MANIFEST_REF_SCHEMA_ID
                or parent.metadata.get("ref_schema_version")
                != AUTHENTICATED_MANIFEST_REF_SCHEMA_VERSION
                or not isinstance(parent.metadata.get("manifest_sha256"), str)
                or len(parent.metadata["manifest_sha256"]) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in parent.metadata["manifest_sha256"]
                )
                or not isinstance(parent.metadata.get("size_bytes"), int)
                or parent.metadata["size_bytes"] < 0
            ):
                raise ValueError("merge checkpoint parent authority is not authenticated")
        acknowledgement = self.acknowledgement
        if (
            acknowledgement.leaf_id != self.declaration.leaf_id
            or acknowledgement.consumer_id != self.declaration.consumer_id
            or acknowledgement.consumer_version != self.declaration.consumer_version
            or acknowledgement.terminal_analysis_type
            != self.declaration.terminal_analysis_type
            or canonical_json_bytes(acknowledgement.parameters)
            != canonical_json_bytes(self.declaration.parameters)
            or acknowledgement.compact_product_schema_id
            != self.declaration.compact_product_schema_id
            or acknowledgement.compact_product_schema_version
            != self.declaration.compact_product_schema_version
            or acknowledgement.compact_product_role != self.declaration.compact_product_role
        ):
            raise ValueError("merge checkpoint acknowledgement drifted from its declaration")
        return self


class EvaluationBatchReclamationEvidence(StrictModel):
    """Fail-closed proof that one batch became safe to reclaim."""

    batch_id: str = Field(min_length=1)
    batch_index: int = Field(ge=0)
    ordered_row_ids: tuple[str, ...] = Field(min_length=1)
    leaf_acknowledgements: tuple[EvaluationBatchLeafAcknowledgement, ...] = Field(min_length=1)
    removed_cache_manifest_ids: tuple[str, ...] = Field(min_length=1)
    removed_cache_bytes: int = Field(ge=0)


class EvaluationBatchCompactionEvidence(StrictModel):
    """Ordered terminal proof for compact fragments, merge state, and reclamation."""

    schema_id: Literal["feedbax.orchestration.evaluation_batch_compaction_evidence"] = (
        EVALUATION_BATCH_COMPACTION_EVIDENCE_SCHEMA_ID
    )
    schema_version: Literal["feedbax.orchestration.evaluation_batch_compaction_evidence.v3"] = (
        EVALUATION_BATCH_COMPACTION_EVIDENCE_SCHEMA_VERSION
    )
    matrix_intent_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    ordered_batch_ids: tuple[str, ...]
    declared_leaf_ids: tuple[str, ...]
    required_leaf_ids_by_batch: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    reclamations: tuple[EvaluationBatchReclamationEvidence, ...] = ()
    terminal_products: tuple[ArtifactRef, ...] = ()

    @model_validator(mode="after")
    def _validate_complete_order(self) -> "EvaluationBatchCompactionEvidence":
        if len(self.ordered_batch_ids) != len(set(self.ordered_batch_ids)):
            raise ValueError("compaction ordered batch ids must be unique")
        if len(self.declared_leaf_ids) != len(set(self.declared_leaf_ids)):
            raise ValueError("compaction declared leaf ids must be unique")
        if self.declared_leaf_ids:
            if tuple(item.batch_id for item in self.reclamations) != self.ordered_batch_ids:
                raise ValueError("compaction reclamations must preserve authored batch order")
            for item in self.reclamations:
                expected = set(self.required_leaf_ids_by_batch.get(item.batch_id, ()))
                observed = [ack.leaf_id for ack in item.leaf_acknowledgements]
                if len(observed) != len(set(observed)) or set(observed) != expected:
                    raise ValueError("each reclaimed batch must acknowledge every declared leaf")
        elif self.reclamations or self.terminal_products:
            raise ValueError("compaction output requires declared terminal leaves")
        return self


class EvaluationMatrixOrderedUnionEvidence(StrictModel):
    """Aggregate proof that batch outcomes reconstruct one authored matrix."""

    schema_id: Literal["feedbax.orchestration.evaluation_matrix_ordered_union_evidence"] = (
        EVALUATION_MATRIX_ORDERED_UNION_EVIDENCE_SCHEMA_ID
    )
    schema_version: Literal["feedbax.orchestration.evaluation_matrix_ordered_union_evidence.v1"] = (
        EVALUATION_MATRIX_ORDERED_UNION_EVIDENCE_SCHEMA_VERSION
    )
    matrix_intent_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    ordered_row_ids_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ordered_batch_ids: tuple[str, ...] = Field(min_length=1)
    ordered_row_ids: tuple[str, ...] = Field(min_length=1)


class EvaluationWorkerProcessEvidence(StrictModel):
    """Batches observed in one persistent matrix-harness worker process."""

    pid: int = Field(gt=0)
    ordered_batch_ids: tuple[str, ...] = Field(min_length=1)


class EvaluationWorkerTopologyEvidence(StrictModel):
    """Process-reuse proof for governed evaluation batch execution."""

    schema_id: Literal["feedbax.orchestration.evaluation_worker_topology_evidence"] = (
        EVALUATION_WORKER_TOPOLOGY_EVIDENCE_SCHEMA_ID
    )
    schema_version: Literal["feedbax.orchestration.evaluation_worker_topology_evidence.v1"] = (
        EVALUATION_WORKER_TOPOLOGY_EVIDENCE_SCHEMA_VERSION
    )
    requested_worker_count: int = Field(gt=0)
    batch_count: int = Field(gt=0)
    processes: tuple[EvaluationWorkerProcessEvidence, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _bind_batches(self) -> "EvaluationWorkerTopologyEvidence":
        batch_ids = [
            batch_id for process in self.processes for batch_id in process.ordered_batch_ids
        ]
        if len(batch_ids) != self.batch_count or len(batch_ids) != len(set(batch_ids)):
            raise ValueError("evaluation worker topology must cover every batch exactly once")
        if len(self.processes) > self.requested_worker_count:
            raise ValueError("observed more evaluation workers than requested")
        pids = [process.pid for process in self.processes]
        if len(pids) != len(set(pids)):
            raise ValueError("evaluation worker topology process ids must be unique")
        return self


class EvaluationShadowLaunchEvidence(StrictModel):
    """Provider-free traversal evidence for the evaluation executor family."""

    schema_id: Literal["feedbax.orchestration.evaluation_shadow_launch_evidence"] = (
        EVALUATION_SHADOW_LAUNCH_EVIDENCE_SCHEMA_ID
    )
    schema_version: Literal["feedbax.orchestration.evaluation_shadow_launch_evidence.v2"] = (
        EVALUATION_SHADOW_LAUNCH_EVIDENCE_SCHEMA_VERSION
    )
    evidence_kind: Literal["provider_free_shadow_launch"] = "provider_free_shadow_launch"
    provider_readiness: Literal["not_evaluated"] = "not_evaluated"
    run_set_id: str = Field(min_length=1)
    bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    exercised_through_stage: Literal["TEARDOWN"] = "TEARDOWN"
    lifecycles: tuple[EvaluationLifecycleEvidence, ...] = Field(min_length=1)
    ordered_union: EvaluationMatrixOrderedUnionEvidence
    worker_topology: EvaluationWorkerTopologyEvidence


__all__ = [
    "EVALUATION_COLLECTION_OUTPUTS",
    "EVALUATION_LIFECYCLE_EVIDENCE_SCHEMA_ID",
    "EVALUATION_LIFECYCLE_EVIDENCE_SCHEMA_VERSION",
    "EVALUATION_MATRIX_BATCH_PLAN_SCHEMA_ID",
    "EVALUATION_MATRIX_BATCH_PLAN_SCHEMA_VERSION",
    "EVALUATION_MATRIX_BATCH_PLAN_SCHEMA_VERSION_V1",
    "EVALUATION_MATRIX_BATCH_PLAN_SCHEMA_VERSION_V2",
    "EVALUATION_BATCH_COMPACTION_EVIDENCE_SCHEMA_ID",
    "EVALUATION_BATCH_COMPACTION_EVIDENCE_SCHEMA_VERSION",
    "EVALUATION_BATCH_COMPACTION_EVIDENCE_SCHEMA_VERSION_V1",
    "EVALUATION_BATCH_MERGE_CHECKPOINT_SCHEMA_ID",
    "EVALUATION_BATCH_MERGE_CHECKPOINT_SCHEMA_VERSION",
    "EVALUATION_BATCH_MERGE_CHECKPOINT_SCHEMA_VERSION_V1",
    "EVALUATION_MATRIX_ORDERED_UNION_EVIDENCE_SCHEMA_ID",
    "EVALUATION_MATRIX_ORDERED_UNION_EVIDENCE_SCHEMA_VERSION",
    "EVALUATION_SHADOW_LAUNCH_EVIDENCE_SCHEMA_ID",
    "EVALUATION_SHADOW_LAUNCH_EVIDENCE_SCHEMA_VERSION",
    "EVALUATION_SHADOW_LAUNCH_EVIDENCE_SCHEMA_VERSION_V1",
    "EVALUATION_WORKER_TOPOLOGY_EVIDENCE_SCHEMA_ID",
    "EVALUATION_WORKER_TOPOLOGY_EVIDENCE_SCHEMA_VERSION",
    "EvaluationBatchCompactionEvidence",
    "EvaluationBatchConsumerDeclaration",
    "EvaluationBatchLeafAcknowledgement",
    "EvaluationBatchMergeCheckpoint",
    "EvaluationBatchReclamationEvidence",
    "EvaluationLifecycleEvidence",
    "EvaluationLifecycleRowOutcome",
    "EvaluationMatrixBatchPlan",
    "EvaluationMatrixBatchUnit",
    "EvaluationMatrixOrderedUnionEvidence",
    "EvaluationShadowLaunchEvidence",
    "EvaluationWorkerProcessEvidence",
    "EvaluationWorkerTopologyEvidence",
]
