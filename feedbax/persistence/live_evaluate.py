"""Keep a scored checkpoint and its exact worker evidence together."""

from __future__ import annotations

from typing import TYPE_CHECKING

from feedbax.contracts.artifact_custody import ContentRetentionTarget, ContentRootRecord
from feedbax.contracts.effective_success import read_success_object
from feedbax.contracts.live_evaluate import (
    LiveEvaluationPoint, ScoredEvaluationPoint, require_evaluation_coverage, rollout_slot,
)
from feedbax.persistence.content_retention import require_active_root
from feedbax.validity.evidence import StagedEvidenceMissing, read_execution_evidence, read_staged_evidence

if TYPE_CHECKING:
    from feedbax.contracts.code_validity import ReceiptEvidence
    from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider


def read_live_evaluation_evidence(
    store: ImmutableArtifactBlobProvider, point: LiveEvaluationPoint, *, active_root: ContentRootRecord | None = None,
) -> ReceiptEvidence:
    result = point.result
    if not isinstance(result, ScoredEvaluationPoint):
        raise ValueError("Failed Evaluate point has no result evidence")
    root = require_active_root(store, point.root_id) if active_root is None else active_root
    if root.root_id != point.root_id or root.targets != (
        ContentRetentionTarget(ref_kind="object", object=result.states.object_ref),
    ):
        raise ValueError("Live Evaluate point lacks its exact active states")
    try:
        evidence = read_staged_evidence(store, point.evaluation_job_id, result.evidence, scope=point.root_id)
    except StagedEvidenceMissing:
        if active_root is None:
            raise
        from feedbax.persistence.effective_success import SuccessView

        # A view selected while active can finish reading after the Store adopts its proof.
        ref = SuccessView(store).attempt_ref(point.evaluation_job_id)
        if ref is None:
            raise
        success = read_success_object(ref, store.get_object)
        if success.attempt != point.evaluation_job_id or success.trace is None or (
            result.evidence not in success.trace.prefixes
        ):
            raise
        evidence = read_execution_evidence(store, result.evidence)
    slots = tuple(rollout_slot(result.member.coordinate, replica) for replica in result.member.replica_ids or ("0",))
    if evidence.attempt != point.evaluation_job_id or result.member.coordinate != point.checkpoint.completed_updates:
        raise ValueError("Live Evaluate evidence differs from its scored checkpoint and replica slots")
    require_evaluation_coverage(evidence.trace.computation or evidence.trace.computation_unavailable, slots)
    return evidence
