"""Verify why a planned producer has no product in a gathered input."""

from feedbax.contracts.artifact_custody import ContentObjectRef
from feedbax.contracts.effective_success import (
    EmptyInputExclusion, ExcludedProduct, MetricExclusion, read_success_object, read_success_request,
    success_content_ref,
)
from feedbax.contracts.request_refs import ConditionalProduct, ProductRef, RequestId
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.train_metrics import passes_final_metric_condition


def exclusion_objects(
    store: ImmutableArtifactBlobProvider, proof: ExcludedProduct, request_id: RequestId,
) -> tuple[ContentObjectRef, ...]:
    """Authenticate the excluded producer, retaining the exact definition and decision evidence."""
    pending = [(proof, request_id)]
    known = {}
    objects = {}
    while pending:
        item, expected = pending.pop()
        request = item.request
        if expected != request.id:
            raise ValueError("Exclusion proof names a different producer request")
        if request.id in known:
            if item != known[request.id]:
                raise ValueError("Conflicting exclusion proofs for the same producer")
            continue
        known[request.id] = item
        refs = (request.object, *request.documents)
        decoded = read_success_request(request, store.get_object)
        match item.cause:
            case MetricExclusion(train=receipt) if decoded.kind in {"train", "evaluate"}:
                conditions = [
                    member.reference for members in decoded.inputs.values() for member in members
                    if isinstance(member.reference, ConditionalProduct)
                ]
                if len(conditions) != 1:
                    raise ValueError("Excluded run has no single final metric condition")
                producer = read_success_object(receipt, store.get_object)
                if passes_final_metric_condition(store, producer, conditions[0]):
                    raise ValueError("Excluded run's producer actually met the final metric condition")
                history = success_content_ref(producer.products["metrics"][0].content)
                assert isinstance(history, ContentObjectRef)
                refs += (receipt, producer.request.object, *producer.request.documents, history)
            case EmptyInputExclusion(role=role, members=excluded) if decoded.kind in {"analyze", "render", "report"}:
                members = decoded.inputs.get(role, ())
                if len(members) != len(excluded):
                    raise ValueError("Exclusion does not cover an entire gathered input")
                for member in members:
                    if member.key not in excluded or not isinstance(member.reference, ProductRef):
                        raise ValueError("Exclusion does not cover an entire gathered input")
                    pending.append((excluded[member.key], member.reference.request_id))
            case _:
                raise ValueError("Exclusion proof differs from the producer's request kind")

        objects.update((ref.sha256, ref) for ref in refs)
    return tuple(objects.values())
