"""Store custody for recorded execution evidence and its immutable inputs."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from hashlib import sha256
from typing import TYPE_CHECKING

from feedbax.contracts.artifact_custody import ContentObjectRef, ContentRetentionTarget, ContentTreeRef
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.code_validity import CodeTrace, ReceiptEvidence
from feedbax.contracts.store_closure import StoreClosureRef, read_store_closure_ref

if TYPE_CHECKING:
    from feedbax.contracts.effective_success import Success
    from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider


class StagedEvidenceMissing(ValueError):
    """An exact execution scope no longer has provisional Store custody."""


def read_execution_evidence(store: ImmutableArtifactBlobProvider, ref: ContentObjectRef) -> ReceiptEvidence:
    return ReceiptEvidence.model_validate_json(store.get_object(ref))


def evidence_objects(store: ImmutableArtifactBlobProvider, trace: CodeTrace) -> Iterator[ContentObjectRef]:
    from feedbax.persistence.receipt_content import tree_objects

    pending = [trace]
    seen_prefixes: set[str] = set()
    while pending:
        segment = pending.pop()
        yield from (source.content for source in segment.sources)
        if (computation := segment.computation) is not None:
            for reference in computation.content_references():
                if isinstance(reference, ContentTreeRef):
                    yield from tree_objects(store, reference)
                else:
                    yield reference
        for prefix in segment.prefixes:
            if prefix.sha256 not in seen_prefixes:
                seen_prefixes.add(prefix.sha256)
                yield prefix
                pending.append(read_execution_evidence(store, prefix).trace)


def _staged_ref(attempt: str) -> str:
    return f"execution/staged/{attempt}"


def _references(
    store: ImmutableArtifactBlobProvider, ref: ContentObjectRef, evidence: ReceiptEvidence,
) -> tuple[ContentObjectRef, ...]:
    unique: dict[str, ContentObjectRef] = {}
    for item in (ref, *evidence_objects(store, evidence.trace)):
        if unique.setdefault(item.sha256, item) != item:
            raise ValueError("Conflicting execution evidence content identities")
    return tuple(unique.values())


def execution_evidence_targets(
    store: ImmutableArtifactBlobProvider, ref: ContentObjectRef, evidence: ReceiptEvidence,
) -> tuple[ContentRetentionTarget, ...]:
    return tuple(ContentRetentionTarget(ref_kind="object", object=item) for item in _references(store, ref, evidence))


def _staged_value(store: ImmutableArtifactBlobProvider, ref: ContentObjectRef, evidence: ReceiptEvidence) -> dict:
    return StoreClosureRef(roots=_references(store, ref, evidence)).model_dump(mode="json")


def stage_evidence(
    store: ImmutableArtifactBlobProvider, evidence: ReceiptEvidence | Callable[[], ReceiptEvidence],
    *, scope: str | None = None,
) -> ContentObjectRef:
    with store._exclusive_lock():
        if callable(evidence):
            evidence = evidence()
        ref = store.store_object(canonical_json_v2_bytes(evidence.model_dump(mode="json")))
        if not store.compare_and_swap_ref(_staged_ref(scope or evidence.attempt), expected=None,
                                          replacement=_staged_value(store, ref, evidence)):
            raise ValueError("This execution already has staged evidence")
    return ref


def read_staged_evidence(
    store: ImmutableArtifactBlobProvider, attempt: str, ref: ContentObjectRef, *, scope: str | None = None,
) -> ReceiptEvidence:
    with store._exclusive_lock():
        staged = store.read_ref(_staged_ref(scope or attempt))
        if staged is None:
            raise StagedEvidenceMissing("The execution has no staged evidence")
        evidence = read_execution_evidence(store, ref)
        if evidence.attempt != attempt or staged != _staged_value(store, ref, evidence):
            raise ValueError("Execution evidence differs from the recorded attempt and its Store root")
    return evidence


def release_staged_evidence(
    store: ImmutableArtifactBlobProvider, attempt: str, *, scope: str | None = None,
) -> None:
    _release_ref(store, _staged_ref(scope or attempt))


def stage_checkpoint_evidence(
    store: ImmutableArtifactBlobProvider, attempt: str, *roots: ContentObjectRef | ContentTreeRef,
) -> None:
    """Retain checkpoint evidence until an authenticated durable owner takes custody."""
    name = f"execution/checkpoints/{attempt}"
    with store._exclusive_lock():
        expected = store.read_ref(name)
        retained = list(read_store_closure_ref(expected).roots) if expected is not None else []
        for root in roots:
            if root not in retained:
                retained.append(root)
        replacement = StoreClosureRef(roots=tuple(retained)).model_dump(mode="json")
        if not store.compare_and_swap_ref(name, expected=expected, replacement=replacement):
            raise ValueError("Staged checkpoint evidence moved during publication")


def release_checkpoint_evidence(store: ImmutableArtifactBlobProvider, attempt: str) -> None:
    _release_ref(store, f"execution/checkpoints/{attempt}")


def _release_ref(store: ImmutableArtifactBlobProvider, name: str) -> None:
    with store._exclusive_lock():
        expected = store.read_ref(name)
        if expected is not None and not store.compare_and_swap_ref(name, expected=expected, replacement=None):
            raise ValueError("Staged execution evidence moved before release")


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
