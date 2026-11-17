"""Store custody for recorded execution evidence and its immutable inputs."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from hashlib import sha256
from typing import TYPE_CHECKING

from feedbax.contracts.artifact_custody import ContentObjectRef, ContentTreeRef
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.code_validity import (
    CheckpointComputation, CheckpointEvidence, CheckpointEvidenceRoot, CodeTrace, ComputationUnavailable,
    ExecutionEvidenceRef, ObservationGeneration, ReceiptEvidence,
)
from feedbax.contracts.computation_validity import ComputationPrograms, ComputationTrace
from feedbax.contracts.store_closure import StoreClosureRef, read_store_closure_ref

if TYPE_CHECKING:
    from feedbax.contracts.effective_success import Success
    from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider


class StagedEvidenceMissing(ValueError):
    """An exact execution scope no longer has provisional Store custody."""


_OBSERVED_FIELDS = ("registry", "files", "environment", "indeterminate", "indeterminate_files")


def execution_evidence_identity(ref: ExecutionEvidenceRef) -> tuple:
    if isinstance(ref, CheckpointEvidenceRoot):
        return ref.evidence.sha256, ref.store_id, ref.attempt, ref.completed_updates
    return (ref.sha256,)


def _resolve_checkpoint(
    store: ImmutableArtifactBlobProvider, root: CheckpointEvidenceRoot,
) -> tuple[CheckpointEvidence, list[tuple[ContentObjectRef, ObservationGeneration]], ComputationTrace | None]:
    endpoint = CheckpointEvidence.model_validate_json(store.get_object(root.evidence))
    if (endpoint.store_id, endpoint.attempt, endpoint.completed_updates) != (
        root.store_id, root.attempt, root.completed_updates,
    ):
        raise ValueError("Checkpoint evidence differs from the exact Store, attempt or coordinate")
    generations = []
    index = None
    ref = endpoint.generation
    while ref is not None:
        generation = ObservationGeneration.model_validate_json(store.get_object(ref))
        if (generation.store_id, generation.attempt) != (root.store_id, root.attempt) or (
            index is not None and generation.index != index
        ):
            raise ValueError("Observation generation differs from its checkpoint ancestry")
        generations.append((ref, generation))
        index = generation.index - 1
        ref = generation.previous

    computation = None
    if isinstance(proof := endpoint.computation, CheckpointComputation):
        programs = ComputationPrograms.model_validate_json(store.get_object(proof.programs))
        computation = ComputationTrace(
            **dict(programs), completed_steps=proof.completed_steps, prefix=proof.prefix,
        )
    return endpoint, generations, computation


def read_execution_evidence(store: ImmutableArtifactBlobProvider, ref: ExecutionEvidenceRef) -> ReceiptEvidence:
    if isinstance(ref, ContentObjectRef):
        return ReceiptEvidence.model_validate_json(store.get_object(ref))
    endpoint, generations, computation = _resolve_checkpoint(store, ref)
    sources = {}
    observations = {}
    for _, generation in reversed(generations):
        sources.update((item.path, item) for item in generation.sources)
        observations.update((key, value) for key in _OBSERVED_FIELDS if (value := getattr(generation, key)) is not None)
    return ReceiptEvidence(attempt=endpoint.attempt, provenance=endpoint.provenance, trace=CodeTrace(
        sources=tuple(sorted(sources.values(), key=lambda item: (item.path.scope, item.path.path))),
        **observations,
        computation=computation,
        computation_unavailable=endpoint.computation if isinstance(endpoint.computation, ComputationUnavailable) else None,
        prefixes=endpoint.prefixes,
    ))


def evidence_objects(store: ImmutableArtifactBlobProvider, trace: CodeTrace) -> Iterator[ContentObjectRef]:
    yield from (source.content for source in trace.sources)
    yield from _computation_objects(store, trace.computation)
    yield from _prefix_objects(store, trace.prefixes)


def checkpoint_evidence_objects(
    store: ImmutableArtifactBlobProvider, root: CheckpointEvidenceRoot,
) -> Iterator[ContentObjectRef]:
    yield from _prefix_objects(store, (root,))


def _computation_objects(
    store: ImmutableArtifactBlobProvider, computation: ComputationTrace | None,
) -> Iterator[ContentObjectRef]:
    from feedbax.persistence.receipt_content import tree_objects

    if computation is not None:
        for reference in computation.content_references():
            if isinstance(reference, ContentTreeRef):
                yield from tree_objects(store, reference)
            else:
                yield reference


def _prefix_objects(
    store: ImmutableArtifactBlobProvider, prefixes: tuple[ExecutionEvidenceRef, ...],
) -> Iterator[ContentObjectRef]:
    pending = list(prefixes)
    seen_prefixes = set()
    while pending:
        ref = pending.pop()
        identity = execution_evidence_identity(ref)
        if identity in seen_prefixes:
            continue
        seen_prefixes.add(identity)
        if isinstance(ref, CheckpointEvidenceRoot):
            yield ref.evidence
            endpoint, generations, computation = _resolve_checkpoint(store, ref)
            for generation_ref, generation in generations:
                yield generation_ref
                yield from (source.content for source in generation.sources)
            if isinstance(endpoint.computation, CheckpointComputation):
                yield endpoint.computation.programs
            yield from _computation_objects(store, computation)
            pending.extend(endpoint.prefixes)
        else:
            yield ref
            segment = read_execution_evidence(store, ref).trace
            yield from (source.content for source in segment.sources)
            yield from _computation_objects(store, segment.computation)
            pending.extend(segment.prefixes)


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
