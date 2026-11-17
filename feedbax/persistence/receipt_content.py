"""Resolve and verify the canonical Store objects behind receipt content."""

from collections.abc import Callable
from dataclasses import dataclass

from feedbax.contracts.artifact_custody import ContentObjectRef, ContentTreeRef
from feedbax.contracts.checkpoint_projection import ContinuationEncoding, ContinuationOccurrence
from feedbax.contracts.dataset import DATASET_MEDIA_TYPE, DatasetProductRef
from feedbax.contracts.effective_success import Product, success_content_ref
from feedbax.contracts.model_checkpoint import (
    MODEL_TREE_ARRAY_PATH,
    MODEL_TREE_GRAPH_PATH,
    MODEL_TREE_MEDIA_TYPE,
    SemanticModelEncoding,
)
from feedbax.contracts.request_refs import ContentRef
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.store_damage import require_sound


@dataclass(frozen=True)
class ClosureVisitors:
    object: Callable[[ContentObjectRef], None]
    array: Callable[[ContentObjectRef, str], None]


def tree_objects(
    store: ImmutableArtifactBlobProvider,
    ref: ContentTreeRef,
    *,
    model_encoding: SemanticModelEncoding | None = None,
    visitors: ClosureVisitors | None = None,
) -> tuple[ContentObjectRef, ...]:
    if visitors is not None:
        visitors.object(ref.descriptor)
    tree = store.get_tree(ref)
    if model_encoding is not None:
        from feedbax.contracts.graph import GraphSpec
        from feedbax.persistence.array_container import open_array_container
        from feedbax.persistence.model_tree import model_tree_assets

        members = {file.relative_path: file.object for file in tree.files}
        if MODEL_TREE_GRAPH_PATH not in members:
            raise ValueError("Semantic model tree is missing its graph")
        if visitors is not None:
            visitors.object(members[MODEL_TREE_GRAPH_PATH])
        graph = GraphSpec.model_validate_json(store.get_object(members[MODEL_TREE_GRAPH_PATH]))
        model_tree_assets(graph, members)
        array = members[MODEL_TREE_ARRAY_PATH]
        digest = model_encoding.array_descriptor_sha256
        if visitors is not None:
            visitors.array(array, digest)
        else:
            with (
                store.open_owned_object(array) as path,
                open_array_container(path, expected_descriptor_sha256=digest),
            ):
                pass
    return (ref.descriptor, *(file.object for file in tree.files))


def success_content_objects(
    store: ImmutableArtifactBlobProvider, value: ContentRef | Product, *,
    visitors: ClosureVisitors | None = None,
) -> tuple[ContentObjectRef, ...]:
    content = value.content if isinstance(value, Product) else value
    ref = success_content_ref(content)
    if isinstance(ref, ContentObjectRef):
        if visitors is not None and content.media_type == DATASET_MEDIA_TYPE and isinstance(value, Product):
            descriptor = value.descriptor
            if not isinstance(descriptor, dict) or descriptor.keys() != {"descriptor_sha256"}:
                raise ValueError(f"Invalid dataset binding for {ref.object_id}")
            binding = DatasetProductRef.model_validate({"content": content, **descriptor})
            visitors.array(ref, binding.descriptor_sha256)
        return (ref,)
    encoding = None
    extra: tuple[ContentObjectRef, ...] = ()
    if content.media_type == MODEL_TREE_MEDIA_TYPE:
        if not isinstance(value, Product):
            raise ValueError("A semantic model binding requires a product array descriptor")
        descriptor = value.descriptor
        if isinstance(descriptor, dict) and "schema_version" in descriptor:
            projection = ContinuationEncoding.model_validate(descriptor)
            encoding = projection.model
            extra = tuple(slot.content for slot in projection.array_slots.values())
            if visitors is not None:
                for slot in projection.array_slots.values():
                    visitors.array(slot.content, slot.array_descriptor_sha256)
        else:
            encoding = SemanticModelEncoding.model_validate(descriptor)
    return (*tree_objects(store, ref, model_encoding=encoding, visitors=visitors), *extra)


def continuation_objects(
    occurrence: ContinuationOccurrence, *, visitors: ClosureVisitors | None = None,
) -> tuple[ContentObjectRef, ...]:
    slots = (occurrence.batch_counter, occurrence.optimizer, occurrence.prng)
    if visitors is not None:
        for slot in slots:
            visitors.array(slot.content, slot.array_descriptor_sha256)
    return tuple(slot.content for slot in slots)


def verify_success_content(
    store: ImmutableArtifactBlobProvider, value: ContentRef | Product
) -> None:
    for ref in success_content_objects(store, value):
        require_sound(store, ref)
        store.verify_object(ref)
