"""Bind exact Host publications as immutable model inputs."""

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from feedbax.contracts.artifact_custody import ContentLeaseRecord, ContentObjectRef, ContentRetentionTarget
from feedbax.contracts.effective_success import (
    Bindings, BoundMember, CheckpointModelSourceEvidence, CheckpointPublicationSource, Product, RequestObject,
    exact_model_checkpoint_input,
)
from feedbax.contracts.request_refs import BatchMember, EffectiveBindingId, RequestId
from feedbax.contracts.studio_training import PublishedTrainingCheckpoint
from feedbax.job_completion import load_bound_checkpoint_selection
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.checkpoint_publication import checkpoint_model_objects, model_product, read_retained_checkpoint
from feedbax.persistence.effective_success import SuccessView
from feedbax.persistence.method_checkpoint import method_checkpoint_files
from feedbax.persistence.canonical_train import BoundStudioTrainingJobInput


def _publication_files(publication: dict) -> dict[str, tuple[str, int]]:
    return {file["path"]: (file["sha256"], file["size_bytes"]) for file in publication["files"]}


def retain_checkpoint(
    store: ImmutableArtifactBlobProvider, checkpoint: PublishedTrainingCheckpoint, publication: dict,
) -> CheckpointPublicationSource:
    files = _publication_files(publication)
    pointer = checkpoint.pointer
    transaction = checkpoint.checkpoint_root / pointer.manifest_relative_path
    launch = checkpoint.checkpoint_root / "binding.json"
    transaction_path, launch_path = (path.relative_to(store.root).as_posix() for path in (transaction, launch))
    digest, size = files[transaction_path]
    launch_digest, launch_size = files[launch_path]
    if digest != pointer.manifest_sha256 or launch_digest != publication["binding_sha256"]:
        raise ValueError("Checkpoint transaction or Train binding is absent from the exact Host evidence")
    return CheckpointPublicationSource(
        pointer=pointer,
        transaction=store._store_file(transaction, sha256=digest, size_bytes=size),
        train_binding=store._store_file(launch, sha256=launch_digest, size_bytes=launch_size),
    )


def checkpoint_producer(launch: BoundStudioTrainingJobInput) -> RequestObject:
    canonical = launch.canonical_request
    if canonical is None:
        raise ValueError("Checkpoint lacks its canonical Train request")
    return RequestObject(id=launch.request_id, object=canonical.request, documents=canonical.documents)


def retain_checkpoint_model(
    store: ImmutableArtifactBlobProvider, checkpoint: PublishedTrainingCheckpoint, publication: dict,
    evidence: CheckpointPublicationSource, producer: RequestObject,
) -> Product:
    manifest, _ = read_retained_checkpoint(store, evidence, producer)
    files = _publication_files(publication)
    members = method_checkpoint_files(checkpoint.checkpoint_root, checkpoint.pointer, include_extra=False)
    for relative, ref in members.items():
        if relative.as_posix() == checkpoint.pointer.manifest_relative_path:
            continue
        path = checkpoint.checkpoint_root / relative
        if files.get(path.relative_to(store.root).as_posix()) != (ref.sha256, ref.size_bytes):
            raise ValueError("Checkpoint model object is absent from the exact Host evidence")
        store._store_file(path, sha256=ref.sha256, size_bytes=ref.size_bytes)
    return model_product(manifest)


@contextmanager
def bind_managed_inputs(
    view: SuccessView, request: RequestObject, *,
    origins: Mapping[RequestId, ContentObjectRef], checkpoint_bindings: Sequence[dict], project: str,
) -> Iterator[tuple[EffectiveBindingId, Bindings]]:
    exact = exact_model_checkpoint_input(view.read_request(request))
    if exact is None:
        if checkpoint_bindings:
            raise ValueError("Unexpected Host checkpoint binding")
        yield view.bind(request, origins=origins)
        return
    if len(checkpoint_bindings) != 1:
        raise ValueError("Analyze requires one exact Host checkpoint binding")
    _, member, reference = exact
    assert isinstance(reference.select, BatchMember)
    (binding,) = checkpoint_bindings
    store = view.store
    ((checkpoint, launch),) = load_bound_checkpoint_selection(
        binding, root=store.root, producer_request=reference.request_id,
        coordinates=(reference.select.batch,), project=project,
    )
    (publication,) = binding["checkpoint_selection"]["members"]
    producer = checkpoint_producer(launch)
    with store._exclusive_lock():
        evidence = retain_checkpoint(store, checkpoint, publication)
        product = retain_checkpoint_model(store, checkpoint, publication, evidence, producer)
        checkpoint_model = BoundMember(
            key=member.key, content=product.content, descriptor=product.descriptor,
            checkpoint_model=CheckpointModelSourceEvidence(producer=producer, publication=evidence),
        )
        refs = {ref.sha256: ref for ref in checkpoint_model_objects(store, member, checkpoint_model)}
        lease = store.acquire_lease(ContentLeaseRecord(
            lease_id=f"checkpoint-model:{uuid4().hex}",
            owner=request.id,
            targets=tuple(ContentRetentionTarget(ref_kind="object", object=ref) for ref in refs.values()),
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        ))
    try:
        yield view.bind(request, origins=origins, checkpoint_model=checkpoint_model)
    finally:
        store.release_lease(lease.lease_id)
