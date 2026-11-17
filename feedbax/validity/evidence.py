"""Store custody for recorded execution evidence and its immutable inputs."""

from __future__ import annotations

from collections.abc import Iterator
from hashlib import sha256
from typing import TYPE_CHECKING

from feedbax.contracts.artifact_custody import ContentObjectRef, ContentRetentionTarget, ContentRootRecord
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.code_validity import CodeTrace, ReceiptEvidence
from feedbax.persistence.content_retention import read_active_roots

if TYPE_CHECKING:
    from feedbax.contracts.effective_success import Success
    from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider


def read_execution_evidence(store: ImmutableArtifactBlobProvider, ref: ContentObjectRef) -> ReceiptEvidence:
    return ReceiptEvidence.model_validate_json(store.get_object(ref))


def evidence_objects(store: ImmutableArtifactBlobProvider, trace: CodeTrace) -> Iterator[ContentObjectRef]:
    pending = [trace]
    seen_prefixes: set[str] = set()
    while pending:
        segment = pending.pop()
        yield from (source.content for source in segment.sources)
        if (computation := segment.computation) is not None:
            yield from (start.binding for start in computation.starts if start.binding is not None)
            for program in computation.programs:
                for lowered in (program.reference, program.execution):
                    if lowered is not None and lowered.raw_program is not None:
                        yield lowered.raw_program
            if computation.prefix is not None:
                yield computation.prefix.parameters
        for prefix in segment.prefixes:
            if prefix.sha256 not in seen_prefixes:
                seen_prefixes.add(prefix.sha256)
                yield prefix
                pending.append(read_execution_evidence(store, prefix).trace)


def _root_id(attempt: str) -> str:
    return f"execution-evidence:{attempt}"


def _root_targets(
    store: ImmutableArtifactBlobProvider, ref: ContentObjectRef, evidence: ReceiptEvidence,
) -> tuple[ContentRetentionTarget, ...]:
    unique: dict[str, ContentObjectRef] = {}
    for item in (ref, *evidence_objects(store, evidence.trace)):
        if unique.setdefault(item.sha256, item) != item:
            raise ValueError("Conflicting execution evidence content identities")
    return tuple(
        ContentRetentionTarget(ref_kind="object", object=item)
        for item in unique.values()
    )


def stage_evidence(store: ImmutableArtifactBlobProvider, evidence: ReceiptEvidence) -> ContentObjectRef:
    with store._exclusive_lock():
        ref = store.store_object(canonical_json_v2_bytes(evidence.model_dump(mode="json")))
        store.register_root(ContentRootRecord(
            root_id=_root_id(evidence.attempt), root_kind="run",
            targets=_root_targets(store, ref, evidence),
        ))
    return ref


def read_staged_evidence(
    store: ImmutableArtifactBlobProvider, attempt: str, ref: ContentObjectRef,
) -> ReceiptEvidence:
    evidence = read_execution_evidence(store, ref)
    (root,) = read_active_roots(store, (_root_id(attempt),))
    if evidence.attempt != attempt or root.targets != _root_targets(store, ref, evidence):
        raise ValueError("Execution evidence differs from the recorded attempt and its Store root")
    return evidence


def release_staged_evidence(store: ImmutableArtifactBlobProvider, attempt: str) -> None:
    store.release_root(_root_id(attempt))


def verify_execution_replay(receipt: Success, expected: ContentObjectRef | None) -> None:
    if receipt.trace is None or receipt.provenance is None:
        if expected is None:
            return
    elif expected is not None:
        saved = ReceiptEvidence(attempt=receipt.attempt, trace=receipt.trace, provenance=receipt.provenance)
        data = canonical_json_v2_bytes(saved.model_dump(mode="json"))
        if (sha256(data).hexdigest(), len(data)) == (expected.sha256, expected.size_bytes):
            return
    raise ValueError("Published attempt has different execution evidence")
