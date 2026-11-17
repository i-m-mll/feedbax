"""Keep a scored checkpoint and its exact worker evidence together."""

from __future__ import annotations

from typing import TYPE_CHECKING

from feedbax.contracts.artifact_custody import ContentRetentionTarget
from feedbax.contracts.live_evaluate import (
    LiveEvaluationPoint, ScoredEvaluationPoint, require_evaluation_coverage, rollout_slot,
)
from feedbax.persistence.content_retention import require_active_root
from feedbax.validity.evidence import evidence_objects, read_execution_evidence

if TYPE_CHECKING:
    from feedbax.contracts.code_validity import ReceiptEvidence
    from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider


def live_evaluation_targets(
    store: ImmutableArtifactBlobProvider, point: LiveEvaluationPoint,
) -> tuple[ReceiptEvidence, tuple[ContentRetentionTarget, ...]]:
    result = point.result
    if not isinstance(result, ScoredEvaluationPoint):
        raise ValueError("Failed Evaluate point has no result evidence")
    evidence = read_execution_evidence(store, result.evidence)
    slots = tuple(rollout_slot(result.member.coordinate, replica) for replica in result.member.replica_ids or ("0",))
    if evidence.attempt != point.evaluation_job_id or result.member.coordinate != point.checkpoint.completed_updates:
        raise ValueError("Live Evaluate evidence differs from its scored checkpoint and replica slots")
    require_evaluation_coverage(evidence.trace.computation, slots)
    refs = dict.fromkeys((result.states.object_ref, result.evidence, *evidence_objects(store, evidence.trace)))
    return evidence, tuple(ContentRetentionTarget(ref_kind="object", object=ref) for ref in refs)


def read_live_evaluation_evidence(
    store: ImmutableArtifactBlobProvider, point: LiveEvaluationPoint,
) -> ReceiptEvidence:
    evidence, targets = live_evaluation_targets(store, point)
    if require_active_root(store, point.root_id).targets != targets:
        raise ValueError("Live Evaluate point lacks exact states and worker evidence in Store custody")
    return evidence
