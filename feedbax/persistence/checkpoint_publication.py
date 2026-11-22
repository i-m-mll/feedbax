"""Read verified native checkpoint evidence from the portable success Store."""

from feedbax.contracts.artifact_custody import ContentObjectRef
from feedbax.contracts.effective_success import (
    BoundMember, CheckpointPublicationSource, Product, RequestObject, read_success_request,
)
from feedbax.contracts.method_checkpoint import MethodCheckpointManifest
from feedbax.contracts.model_checkpoint import MODEL_TREE_MEDIA_TYPE
from feedbax.contracts.request_refs import BatchMember, ContentRef, InputMember, ProductRef
from feedbax.contracts.request_specs import DocumentTrainingRequestSpec
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.method_checkpoint import parse_method_checkpoint_manifest
from feedbax.persistence.receipt_content import ClosureVisitors, success_content_objects
from feedbax.persistence.canonical_train import BoundStudioTrainingJobInput


def read_retained_checkpoint(
    store: ImmutableArtifactBlobProvider, source: CheckpointPublicationSource,
    producer: RequestObject,
) -> tuple[MethodCheckpointManifest, DocumentTrainingRequestSpec]:
    launch = BoundStudioTrainingJobInput.model_validate_json(store.get_object(source.train_binding))
    binding = launch.canonical_request
    request = read_success_request(producer, store.get_object)
    spec = request.spec
    manifest = parse_method_checkpoint_manifest(store.get_object(source.transaction), source.pointer)
    identity = manifest.identity
    if (
        binding is None or not isinstance(spec, DocumentTrainingRequestSpec)
        or (launch.request_id, binding.request, binding.documents)
        != (producer.id, producer.object, producer.documents)
        or launch.spec != spec
        or (identity.request_id, identity.method_key) != (request.id, spec.method)
    ):
        raise ValueError("Checkpoint publication differs from its authenticated Train definition")
    return manifest, spec


def model_product(manifest: MethodCheckpointManifest) -> Product:
    model = manifest.model
    descriptor = model.tree.descriptor
    return Product(
        name="model", content=ContentRef(
            sha256=descriptor.sha256, size=descriptor.size_bytes, media_type=MODEL_TREE_MEDIA_TYPE, tree=True,
        ), descriptor=model.encoding.model_dump(mode="json"),
    )


def checkpoint_model_objects(
    store: ImmutableArtifactBlobProvider, member: InputMember, bound: BoundMember,
    *, visitors: ClosureVisitors | None = None,
) -> tuple[ContentObjectRef, ...]:
    source = bound.checkpoint_model
    reference = member.reference
    if (
        source is None or bound.key != member.key or not isinstance(reference, ProductRef)
        or reference.role != "checkpoint" or not isinstance(reference.select, BatchMember)
        or reference.request_id != source.producer.id
        or reference.select.batch != source.publication.pointer.completed_updates
    ):
        raise ValueError("Model input differs from its authored checkpoint selection")
    refs = (
        source.publication.transaction, source.publication.train_binding,
        source.producer.object, *source.producer.documents,
    )
    if visitors is not None:
        for ref in refs:
            visitors.object(ref)
    manifest, _ = read_retained_checkpoint(store, source.publication, source.producer)
    product = model_product(manifest)
    if (product.content, product.descriptor) != (bound.content, bound.descriptor):
        raise ValueError("Model input differs from the retained checkpoint product")
    return (*refs, *success_content_objects(store, product, visitors=visitors))
