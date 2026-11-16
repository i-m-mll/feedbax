"""Sync a typed closure between two immutable Stores on one volume."""

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import errno
from pathlib import Path
import stat

from feedbax._secure_fs import close_directory_chain, open_directory_chain
from feedbax.contracts.artifact_custody import ArtifactBlobContainmentError, ContentObjectRef
from feedbax.contracts.effective_success import INDEX_REF, decode_success_index
from feedbax.contracts.request_refs import RequestId
from feedbax.contracts.store_closure import ReceiptRoot, StoreClosureRef, read_store_closure_ref
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.effective_success import PublishedSuccess, SuccessView
from feedbax.persistence.receipt_closure import ReceiptClosure


@dataclass(frozen=True)
class StoreSyncResult:
    closure: StoreClosureRef
    cloned: int
    reused: int


def _directory_identity(path: Path | str) -> tuple[int, int]:
    info = Path(path).lstat()
    if not stat.S_ISDIR(info.st_mode):
        raise NotADirectoryError(errno.ENOTDIR, "Expected a Store directory", str(path))
    return info.st_dev, info.st_ino


@contextmanager
def _locked_stores(
    source: ImmutableArtifactBlobProvider,
    destination: ImmutableArtifactBlobProvider,
) -> Iterator[None]:
    source_identity = _directory_identity(source.root)
    close_directory_chain(
        open_directory_chain(
            destination.root,
            create=True,
            error_factory=ArtifactBlobContainmentError,
            context="destination Store",
        )
    )
    destination_identity = _directory_identity(destination.root)
    if source_identity == destination_identity:
        raise ValueError("Store sync requires two distinct Store directories")
    if source_identity[0] != destination_identity[0]:
        raise OSError(errno.EXDEV, "Store sync requires both Stores on the same volume")

    with ExitStack() as locks:
        for identity, store in sorted(
            ((source_identity, source), (destination_identity, destination))
        ):
            locks.enter_context(store._exclusive_lock(create=store is destination))
            if _directory_identity(store.root) != identity:
                raise ValueError(f"Store directory changed while taking its lock: {store.root}")
        yield


def sync_store(
    source: ImmutableArtifactBlobProvider,
    destination: ImmutableArtifactBlobProvider,
    selection: StoreClosureRef | str,
    *,
    destination_ref: str,
    expected: StoreClosureRef | None = None,
) -> StoreSyncResult:
    """Publish a complete verified closure from committed receipts or a named packet ref.

    A ref replacement requires its expected value. Repeating the same transfer is
    safe after interruption, including when the requested ref already exists.
    """
    destination._ref_path(destination_ref)
    if destination_ref == INDEX_REF:
        raise ValueError("Store sync cannot publish to the successful-result index")

    with _locked_stores(source, destination):
        index = None
        if isinstance(selection, str):
            value = source.read_ref(selection)
            if value is None:
                raise FileNotFoundError(f"Source closure ref is missing: {selection}")
            closure = read_store_closure_ref(value)
        else:
            closure = selection
            if any(isinstance(root, ReceiptRoot) for root in closure.roots):
                index = decode_success_index(source.read_ref(INDEX_REF))

        replacement = closure.model_dump(mode="json")
        current = destination.read_ref(destination_ref)
        previous = expected.model_dump(mode="json") if expected is not None else None
        if current != replacement and current != previous:
            raise ValueError(f"Destination closure ref changed: {destination_ref}")

        objects = ReceiptClosure(source, index).walk(closure)
        cloned = sum(destination._clone_object_from(source, ref) for ref in objects)
        if not destination.compare_and_swap_ref(
            destination_ref, expected=current, replacement=replacement
        ):
            raise ValueError(f"Destination closure ref changed: {destination_ref}")
        return StoreSyncResult(closure, cloned, len(objects) - cloned)


def collect_success(
    source: ImmutableArtifactBlobProvider,
    destination: ImmutableArtifactBlobProvider,
    receipt: ContentObjectRef,
    *,
    attempt: str,
    request_id: RequestId,
) -> PublishedSuccess:
    """Transfer and admit an accepted result with its original receipt identity."""
    return _collect_success(source, destination, receipt, attempt, request_id, promote=True)


def collect_success_evidence(
    source: ImmutableArtifactBlobProvider,
    destination: ImmutableArtifactBlobProvider,
    receipt: ContentObjectRef,
    *,
    attempt: str,
    request_id: RequestId,
) -> PublishedSuccess:
    """Collect a producer by attempt, preserving the destination's current results."""
    return _collect_success(source, destination, receipt, attempt, request_id, promote=False)


def _collect_success(
    source: ImmutableArtifactBlobProvider,
    destination: ImmutableArtifactBlobProvider,
    receipt: ContentObjectRef,
    attempt: str,
    request_id: RequestId,
    *,
    promote: bool,
) -> PublishedSuccess:
    with _locked_stores(source, destination):
        index = decode_success_index(source.read_ref(INDEX_REF))
        closure = ReceiptClosure(source, index)
        committed = closure.receipt_at(receipt)
        if (committed.attempt, committed.request.id) != (attempt, request_id):
            raise ValueError("Source receipt differs from the expected attempt and request")
        for ref in closure.for_admission(receipt).objects:
            destination._clone_object_from(source, ref)
        return SuccessView(destination).admit_collected(
            receipt, attempt=attempt, request_id=request_id,
            supersession_times=index.supersession_times, promote=promote,
        )
