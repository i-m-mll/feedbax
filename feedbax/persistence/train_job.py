"""Retain and transfer exact Train bindings and progress for a stopped attempt."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import TYPE_CHECKING

from pydantic import Field, model_validator

from feedbax.contracts.artifact_custody import CONTENT_OBJECT_REF_PREFIX, ContentBinding, ContentObjectRef, StoreTransferProgress
from feedbax.contracts.base import StrictModel, sha256_bytes
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.checkpoints import CheckpointLatestPointer, CheckpointTransactionManifest
from feedbax.contracts.effective_success import Bindings, Product, RequestObject
from feedbax.contracts.model_checkpoint import MODEL_TREE_MEDIA_TYPE
from feedbax.contracts.publication import CheckpointSet
from feedbax.contracts.request_refs import ContentRef
from feedbax.contracts.store_closure import ReceiptRoot, StoreClosureRef, read_store_closure_ref
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider, content_object_relative_path
from feedbax.persistence.effective_success import SuccessView
from feedbax.persistence.receipt_content import success_content_objects

if TYPE_CHECKING:
    from feedbax.persistence.store_sync import StoreEndpoint


class TrainJobCheckpoint(StrictModel):
    pointer: CheckpointLatestPointer
    content: ContentBinding


class TrainJobEntry(StrictModel):
    attempt: str
    request: RequestObject
    inputs: Bindings
    checkpoints: tuple[TrainJobCheckpoint, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def exact_attempt(self) -> TrainJobEntry:
        coordinates = [checkpoint.pointer.completed_training_batches for checkpoint in self.checkpoints]
        if (
            not self.request.id.startswith("train:")
            or any(checkpoint.pointer.run_id != self.attempt for checkpoint in self.checkpoints)
            or any(value is None for value in coordinates)
            or len(set(coordinates)) != len(coordinates)
            or coordinates != sorted(value for value in coordinates if value is not None)
        ):
            raise ValueError("Train progress does not name ordered publications of the exact attempt")
        return self

    @property
    def recovery(self) -> TrainJobCheckpoint:
        return self.checkpoints[-1]

    @property
    def object_ref(self) -> ContentObjectRef:
        """The exact binding object named by the committed job ref."""
        raw = self._canonical_bytes()
        digest = sha256_bytes(raw)
        return ContentObjectRef(
            object_id=CONTENT_OBJECT_REF_PREFIX + digest, sha256=digest, size_bytes=len(raw),
        )

    def _canonical_bytes(self) -> bytes:
        return canonical_json_v2_bytes(self.model_dump(mode="json"))


def _job_ref(attempt: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_-]+", attempt) is None:
        raise ValueError("Train attempt must be a single identifier")
    return f"jobs/{attempt}"


def _origins(entry: TrainJobEntry) -> tuple[ContentObjectRef, ...]:
    return tuple({
        member.selection.receipt.sha256: member.selection.receipt
        for members in entry.inputs.values() for member in members if member.selection is not None
    }.values())


def _closure(ref: ContentObjectRef, entry: TrainJobEntry) -> StoreClosureRef:
    return StoreClosureRef(
        roots=(ref, entry.request.object, *entry.request.documents,
               *(ReceiptRoot(receipt=origin) for origin in _origins(entry)),
               *(checkpoint.content.tree for checkpoint in entry.checkpoints)),
        include_ancestry=True,
    )


def _verify_entry(store: ImmutableArtifactBlobProvider, entry: TrainJobEntry) -> None:
    SuccessView(store).verify_bindings(entry.request, entry.inputs)
    _verify_checkpoints(store, entry)


def _verify_checkpoints(store: ImmutableArtifactBlobProvider, entry: TrainJobEntry) -> None:
    for checkpoint in entry.checkpoints:
        content, pointer = checkpoint.content, checkpoint.pointer
        if content.root_kind != "checkpoint-custody" or content.binding_name != pointer.transaction_id:
            raise ValueError("Train recovery requires the exact full checkpoint binding")
        files = {file.relative_path: file.object for file in store.get_tree(content.tree).files}
        saved = CheckpointLatestPointer.model_validate_json(store.get_object(files["latest.json"]))
        manifest_ref = files[pointer.manifest_relative_path]
        if saved != pointer or manifest_ref.sha256 != pointer.manifest_sha256:
            raise ValueError("Train checkpoint tree differs from its exact published pointer")
        manifest = CheckpointTransactionManifest.model_validate_json(store.get_object(manifest_ref))
        if (
            manifest.run_id != pointer.run_id or manifest.transaction_id != pointer.transaction_id
            or manifest.completed_coordinate != pointer.completed_coordinate
            or manifest.completed_training_batches != pointer.completed_training_batches
            or manifest.content_integrity_digest.transaction_root_sha256 != pointer.transaction_root_sha256
        ):
            raise ValueError("Train checkpoint manifest differs from its exact published occurrence")
        parent = pointer.manifest_relative_path.rpartition("/")[0]
        checkpoint_set = CheckpointSet.model_validate_json(store.get_object(files[f"{parent}/checkpoint-set.json"]))
        recorded_parent = manifest.metadata.get("checkpoint_segment_parent_ref")
        if (
            checkpoint_set.transaction.identity != pointer.transaction_id
            or checkpoint_set.transaction.bytes.digest != manifest_ref.sha256
            or checkpoint_set.transaction.bytes.size_bytes != manifest_ref.size_bytes
            or (recorded_parent is not None and (
                checkpoint_set.parent is None or recorded_parent != checkpoint_set.parent.model_dump(mode="json")
            ))
        ):
            raise ValueError("Train checkpoint set differs from its authenticated transaction")
        for slot in manifest.slots:
            member = files[f"{parent}/{slot.relative_path}"]
            if (member.sha256, member.size_bytes) != (slot.sha256, slot.size_bytes):
                raise ValueError("Train checkpoint tree differs from its authenticated slot manifest")
            if slot.media_type == MODEL_TREE_MEDIA_TYPE:
                model = Product(name="model", descriptor=slot.metadata["semantic_model"], content=ContentRef(
                    sha256=member.sha256, size=member.size_bytes, media_type=MODEL_TREE_MEDIA_TYPE, tree=True,
                ))
                for ref in success_content_objects(store, model):
                    if files.get(f"{parent}/semantic-model/{content_object_relative_path(ref.sha256).as_posix()}") != ref:
                        raise ValueError("Train checkpoint tree omits authenticated model members")
        for member in files.values():
            store.verify_object(member)


def publish_train_job(store: ImmutableArtifactBlobProvider, entry: TrainJobEntry) -> ContentObjectRef:
    """Retain the stopped attempt's authenticated binding and exact checkpoint trees."""
    name = _job_ref(entry.attempt)
    with store._exclusive_lock():
        _verify_entry(store, entry)
        ref = store.store_object(entry._canonical_bytes())
        closure = _closure(ref, entry).model_dump(mode="json")
        current = store.read_ref(name)
        if current == closure:
            return ref
        if current is not None or not store.compare_and_swap_ref(name, expected=None, replacement=closure):
            raise ValueError("Train attempt already has different retained custody")
        return ref


def read_train_job(
    store: ImmutableArtifactBlobProvider, attempt: str, *, expected: ContentObjectRef | None = None,
) -> TrainJobEntry:
    """Read a committed job entry and authenticate the custody needed for Resume."""
    with store._exclusive_lock(create=False):
        raw = store.read_ref(_job_ref(attempt))
        if raw is None:
            raise FileNotFoundError(f"No collected checkpoint for Train attempt {attempt!r}")
        entry = _read_entry(store, read_store_closure_ref(raw), attempt, expected=expected)
        _verify_entry(store, entry)
        return entry


def _entry_ref(closure: StoreClosureRef, expected: ContentObjectRef | None) -> ContentObjectRef:
    ref = closure.roots[0]
    if not isinstance(ref, ContentObjectRef):
        raise ValueError("Train job closure does not start with its binding object")
    if expected is not None and ref != expected:
        raise ValueError("Train job entry differs from the observed Store reference")
    return ref


def _read_entry(
    store: ImmutableArtifactBlobProvider, closure: StoreClosureRef, attempt: str,
    *, expected: ContentObjectRef | None,
) -> TrainJobEntry:
    ref = _entry_ref(closure, expected)
    data = store.get_object(ref)
    entry = TrainJobEntry.model_validate_json(data)
    if (
        entry.attempt != attempt or closure != _closure(ref, entry)
        or data != entry._canonical_bytes()
    ):
        raise ValueError("Train job entry differs from its named closure")
    return entry


def collect_train_job(
    source: StoreEndpoint, destination: ImmutableArtifactBlobProvider,
    *, attempt: str, request_id: str, pointer: CheckpointLatestPointer,
    expected: ContentObjectRef | None = None,
    progress: Callable[[StoreTransferProgress], None] | None = None,
) -> TrainJobEntry:
    """Collect a stopped attempt's exact binding, checkpoints, and producer provenance."""
    from feedbax.persistence.store_sync import _collect_train_job

    return _collect_train_job(
        source, destination, attempt=attempt, request_id=request_id, pointer=pointer, expected=expected, progress=progress,
    )
