"""Keep a scored checkpoint and its exact worker evidence together."""

from __future__ import annotations

from typing import TYPE_CHECKING

from feedbax.contracts.artifact_custody import ContentRetentionTarget, ContentRootRecord
from feedbax.contracts.live_evaluate import (
    LiveEvaluationPoint, ScoredEvaluationPoint, require_evaluation_coverage, rollout_slot,
)
from feedbax.persistence.content_retention import require_active_root
from feedbax.validity.evidence import execution_evidence_targets, read_execution_evidence

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
    return evidence, (
        ContentRetentionTarget(ref_kind="object", object=result.states.object_ref),
        *execution_evidence_targets(store, result.evidence, evidence),
    )


def read_live_evaluation_evidence(
    store: ImmutableArtifactBlobProvider, point: LiveEvaluationPoint, *, active_root: ContentRootRecord | None = None,
) -> ReceiptEvidence:
    evidence, targets = live_evaluation_targets(store, point)
    root = require_active_root(store, point.root_id) if active_root is None else active_root
    if root.root_id != point.root_id or root.targets != targets:
        raise ValueError("Live Evaluate point lacks exact states and worker evidence in Store custody")
    return evidence
