"""Publication of the SISU exemplar through the common custody protocols."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from feedbax.contracts.publication import (
    ArtifactRecord,
    BlobRef,
    CheckpointSet,
    CheckpointSlot,
    ExactRef,
    ProvenanceEdge,
    PublicationReceipt,
    PublicationRequest,
    PublicationService,
    artifact_record,
    canonical_bytes,
    checkpoint_set_id,
)
from feedbax.workflow.plan import WorkflowPlan


SISU_ARTIFACT_CHAIN = (
    "training",
    "continuation",
    "evaluation",
    "analysis",
    "figure",
    "report",
)


def workflow_plan_ref(plan: WorkflowPlan) -> ExactRef:
    """Return the exact typed reference for one canonical workflow plan."""
    raw = canonical_bytes(plan.document())
    return ExactRef(
        domain="workflow_plan",
        identity=plan.identity,
        bytes=BlobRef.from_bytes(raw),
    )


@dataclass(frozen=True, slots=True)
class ArtifactPayload:
    """Bytes and declared meaning for one exemplar output."""

    data: bytes
    media_type: str
    schema_id: str
    schema_version: str


@dataclass(frozen=True, slots=True)
class CheckpointPayload:
    """Exact byte payloads needed to construct one resumable checkpoint."""

    progress: Mapping[str, int | float | str]
    prng_state: bytes
    slots: Mapping[str, tuple[str, str, str, str, bytes]]


def _stage_artifact_chain(
    service: PublicationService,
    *,
    study_id: str,
    payloads: Mapping[str, ArtifactPayload],
) -> tuple[ArtifactRecord, ...]:
    supplied = set(payloads)
    expected = set(SISU_ARTIFACT_CHAIN)
    if supplied != expected:
        missing = ", ".join(sorted(expected - supplied)) or "none"
        extra = ", ".join(sorted(supplied - expected)) or "none"
        raise ValueError(
            f"SISU publication requires its complete artifact chain; missing={missing}; extra={extra}"
        )
    records: list[ArtifactRecord] = []
    for role in SISU_ARTIFACT_CHAIN:
        payload = payloads[role]
        blob = service.stage(payload.data)
        records.append(
            artifact_record(
                logical_id=f"{study_id}.{role}",
                role=role,
                media_type=payload.media_type,
                payload_schema_id=payload.schema_id,
                payload_schema_version=payload.schema_version,
                blobs=(blob,),
                dimensions={"study": study_id},
            )
        )
    return tuple(records)


def _stage_checkpoint(
    service: PublicationService,
    *,
    payload: CheckpointPayload,
    training_program_id: str,
    graph: ExactRef,
    experiment: ExactRef,
    continuation: str,
    parent: ExactRef | None,
) -> CheckpointSet:
    prng_state = service.stage(payload.prng_state)
    slots = tuple(
        CheckpointSlot(
            name=name,
            state_type=state_type,
            array_structure_id=array_structure_id,
            codec_schema_id=codec_schema_id,
            codec_schema_version=codec_schema_version,
            blob=service.stage(data),
        )
        for name, (
            state_type,
            array_structure_id,
            codec_schema_id,
            codec_schema_version,
            data,
        ) in sorted(payload.slots.items())
    )
    values = {
        "training_program_id": training_program_id,
        "graph": graph,
        "experiment": experiment,
        "progress": dict(payload.progress),
        "prng_state": prng_state,
        "slots": slots,
        "continuation": continuation,
        "parent": parent,
    }
    return CheckpointSet(checkpoint_id=checkpoint_set_id(**values), **values)


def publish_sisu_artifact_chain(
    service: PublicationService,
    *,
    idempotency_key: str,
    study_id: str,
    training_program_id: str,
    workflow_plan: ExactRef,
    graph: ExactRef,
    experiment: ExactRef,
    payloads: Mapping[str, ArtifactPayload],
    trained_checkpoint: CheckpointPayload,
    continued_checkpoint: CheckpointPayload,
) -> PublicationReceipt:
    """Publish the complete training-through-report SISU chain atomically."""
    records = _stage_artifact_chain(service, study_id=study_id, payloads=payloads)
    trained = _stage_checkpoint(
        service,
        payload=trained_checkpoint,
        training_program_id=training_program_id,
        graph=graph,
        experiment=experiment,
        continuation="fork",
        parent=None,
    )
    continued = _stage_checkpoint(
        service,
        payload=continued_checkpoint,
        training_program_id=training_program_id,
        graph=graph,
        experiment=experiment,
        continuation="resume",
        parent=trained.exact_ref,
    )
    edges: list[ProvenanceEdge] = [
        ProvenanceEdge(
            relation="produced_by",
            subject=record.exact_ref,
            object=workflow_plan,
        )
        for record in records
    ]
    edges.extend(
        ProvenanceEdge(
            relation="derived_from",
            subject=current.exact_ref,
            object=previous.exact_ref,
        )
        for previous, current in zip(records, records[1:])
    )
    edges.extend(
        (
            ProvenanceEdge(
                relation="derived_from",
                subject=trained.exact_ref,
                object=records[0].exact_ref,
            ),
            ProvenanceEdge(
                relation="resumed_from",
                subject=continued.exact_ref,
                object=trained.exact_ref,
            ),
            ProvenanceEdge(
                relation="derived_from",
                subject=records[1].exact_ref,
                object=continued.exact_ref,
            ),
        )
    )
    return service.publish(
        PublicationRequest(
            idempotency_key=idempotency_key,
            artifacts=records,
            provenance=tuple(edges),
            checkpoints=(trained, continued),
        )
    )


__all__ = [
    "ArtifactPayload",
    "CheckpointPayload",
    "SISU_ARTIFACT_CHAIN",
    "publish_sisu_artifact_chain",
    "workflow_plan_ref",
]
