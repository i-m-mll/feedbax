"""Compare quick replay products in the format their consumer reads."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from feedbax.contracts.artifact_custody import ContentObjectRef, ContentTreeRef
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.dataset import DATASET_MEDIA_TYPE, DatasetProductRef
from feedbax.contracts.effective_success import Product, success_content_ref
from feedbax.contracts.figure_document import PLOTLY_MEDIA_TYPE
from feedbax.contracts.strict_json import strict_json_loads
from feedbax.persistence.dataset import open_stored_dataset

if TYPE_CHECKING:
    from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider


def _arrays_match(before, after) -> bool:
    old, new = np.asarray(before), np.asarray(after)
    if old.shape != new.shape or old.dtype != new.dtype:
        return False
    if np.array_equal(old, new):
        return True
    if old.dtype.kind != "f" or not (np.isfinite(old).all() and np.isfinite(new).all()):
        return False
    if not np.array_equal(old == 0, new == 0):
        return False
    tolerance = 8 * np.finfo(old.dtype).eps
    with np.errstate(over="ignore", under="ignore"):
        return bool(np.all(np.abs(old - new) <= tolerance * np.maximum(np.abs(old), np.abs(new))))


def _dataset_matches(store: ImmutableArtifactBlobProvider, before: Product, after: Product) -> bool:
    left = DatasetProductRef.model_validate(dict(content=before.content, **before.descriptor))
    right = DatasetProductRef.model_validate(dict(content=after.content, **after.descriptor))
    with open_stored_dataset(store, left) as old, open_stored_dataset(store, right) as new:
        return old.descriptor == new.descriptor and all(
            _arrays_match(old.data(name), new.data(name)) for name in old.descriptor.tensor_names
        )


def _document_matches(store: ImmutableArtifactBlobProvider, left: ContentObjectRef, right: ContentObjectRef) -> bool:
    return left == right or (
        canonical_json_v2_bytes(strict_json_loads(store.get_object(left)))
        == canonical_json_v2_bytes(strict_json_loads(store.get_object(right)))
    )


def light_product_matches(store: ImmutableArtifactBlobProvider, before: Product, after: Product) -> bool:
    if (
        before.name != after.name or before.content.media_type != after.content.media_type
        or before.content.tree != after.content.tree
    ):
        return False
    if before == after:
        return True
    media_type = before.content.media_type
    if media_type == DATASET_MEDIA_TYPE:
        return _dataset_matches(store, before, after)
    left, right = success_content_ref(before.content), success_content_ref(after.content)
    if isinstance(left, ContentTreeRef) and isinstance(right, ContentTreeRef):
        old, new = store.get_tree(left), store.get_tree(right)
        return old.directories == new.directories and len(old.files) == len(new.files) and all(
            a.relative_path == b.relative_path and (
                _document_matches(store, a.object, b.object) if a.relative_path.endswith(".json") else a.object == b.object
            )
            for a, b in zip(old.files, new.files, strict=True)
        )
    if isinstance(left, ContentObjectRef) and isinstance(right, ContentObjectRef) and media_type == PLOTLY_MEDIA_TYPE:
        return _document_matches(store, left, right)
    return False
