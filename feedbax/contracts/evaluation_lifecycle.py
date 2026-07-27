"""Governed terminal evidence for one evaluation-matrix executor invocation."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from feedbax.contracts.manifest import StrictModel


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
EVALUATION_MATRIX_BATCH_PLAN_SCHEMA_VERSION = "feedbax.spec.evaluation_matrix_batch_plan.v1"
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
    "evaluation",
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

    @model_validator(mode="after")
    def _unique_rows(self) -> "EvaluationMatrixBatchUnit":
        if len(self.ordered_row_ids) != len(set(self.ordered_row_ids)):
            raise ValueError("evaluation batch unit row ids must be unique")
        return self


class EvaluationMatrixBatchPlan(StrictModel):
    """Versioned partition of one matrix identity across persistent workers."""

    schema_id: Literal["feedbax.spec.evaluation_matrix_batch_plan"] = (
        EVALUATION_MATRIX_BATCH_PLAN_SCHEMA_ID
    )
    schema_version: Literal["feedbax.spec.evaluation_matrix_batch_plan.v1"] = (
        EVALUATION_MATRIX_BATCH_PLAN_SCHEMA_VERSION
    )
    matrix_intent_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    batches: tuple[EvaluationMatrixBatchUnit, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_batches_and_rows(self) -> "EvaluationMatrixBatchPlan":
        batch_ids = [item.batch_id for item in self.batches]
        if len(batch_ids) != len(set(batch_ids)):
            raise ValueError("evaluation batch plan batch ids must be unique")
        row_ids = [row_id for item in self.batches for row_id in item.ordered_row_ids]
        if len(row_ids) != len(set(row_ids)):
            raise ValueError("evaluation batch plan row ids must be globally unique")
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
    "EVALUATION_MATRIX_ORDERED_UNION_EVIDENCE_SCHEMA_ID",
    "EVALUATION_MATRIX_ORDERED_UNION_EVIDENCE_SCHEMA_VERSION",
    "EVALUATION_SHADOW_LAUNCH_EVIDENCE_SCHEMA_ID",
    "EVALUATION_SHADOW_LAUNCH_EVIDENCE_SCHEMA_VERSION",
    "EVALUATION_SHADOW_LAUNCH_EVIDENCE_SCHEMA_VERSION_V1",
    "EVALUATION_WORKER_TOPOLOGY_EVIDENCE_SCHEMA_ID",
    "EVALUATION_WORKER_TOPOLOGY_EVIDENCE_SCHEMA_VERSION",
    "EvaluationLifecycleEvidence",
    "EvaluationLifecycleRowOutcome",
    "EvaluationMatrixBatchPlan",
    "EvaluationMatrixBatchUnit",
    "EvaluationMatrixOrderedUnionEvidence",
    "EvaluationShadowLaunchEvidence",
    "EvaluationWorkerProcessEvidence",
    "EvaluationWorkerTopologyEvidence",
]
