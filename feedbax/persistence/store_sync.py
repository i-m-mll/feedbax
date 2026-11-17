"""Transfer and publish exact typed closures between immutable Stores."""

from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime
import errno
import os
from pathlib import Path
import stat
import sys
from typing import Literal

from pydantic import Field, JsonValue, TypeAdapter

from feedbax._secure_fs import close_directory_chain, open_directory_chain
from feedbax.contracts.artifact_custody import (
    ArtifactBlobContainmentError,
    ArtifactBlobIntegrityError,
    ContentObjectRef,
    StoreTransferProgress,
)
from feedbax.contracts.base import StrictModel
from feedbax.contracts.method_checkpoint import CheckpointLatestPointer
from feedbax.contracts.effective_success import INDEX_REF, Success, decode_success_index
from feedbax.contracts.execution_target import AttemptTarget
from feedbax.contracts.request_refs import RequestId
from feedbax.contracts.store_closure import ReceiptRoot, StoreClosureRef, read_store_closure_ref
from feedbax.persistence.artifact_custody import (
    ImmutableArtifactBlobProvider,
    content_object_relative_path,
)
from feedbax.persistence.effective_success import PublishedSuccess, SuccessView
from feedbax.persistence.receipt_closure import ReceiptClosure
from feedbax.persistence.ssh_store import RemoteStore, SSHStoreConnection as SSHStoreConnection
from feedbax.persistence.train_job import (
    TrainJobEntry, _entry_ref, _job_ref, _origins, _read_entry, _verify_checkpoints, _verify_entry,
)

type StoreEndpoint = ImmutableArtifactBlobProvider | RemoteStore
_INCOMING = ".incoming"


@dataclass(frozen=True)
class StoreSyncResult:
    closure: StoreClosureRef
    cloned: int
    reused: int


class _ClosureSource(StrictModel):
    kind: Literal["closure"] = "closure"
    selection: StoreClosureRef | str


class _SuccessSource(StrictModel):
    kind: Literal["success"] = "success"
    receipt: ContentObjectRef
    attempt: str
    request_id: RequestId
    expected_target: AttemptTarget | None = Field(default=None, exclude_if=lambda target: target is None)


class _TrainSource(StrictModel):
    kind: Literal["stopped_train"] = "stopped_train"
    attempt: str
    request_id: RequestId
    pointer: CheckpointLatestPointer
    expected: ContentObjectRef | None = None


type _Selection = _ClosureSource | _SuccessSource | _TrainSource


class _Inventory(StrictModel):
    closure: StoreClosureRef
    objects: tuple[ContentObjectRef, ...]
    supersession_times: dict[str, datetime] = Field(default_factory=dict)


class _RefUpdate(StrictModel):
    name: str
    expected: StoreClosureRef | None = None


class _PublishRequest(StrictModel):
    inventory: _Inventory
    ref: _RefUpdate


def _directory_identity(path: Path | str) -> tuple[int, int]:
    info = Path(path).lstat()
    if not stat.S_ISDIR(info.st_mode):
        raise NotADirectoryError(errno.ENOTDIR, "Expected a Store directory", str(path))
    return info.st_dev, info.st_ino


def _open_directory(path: Path | str, *, create: bool, mode: int = 0o777):
    return open_directory_chain(
        path, create=create, error_factory=ArtifactBlobContainmentError,
        context="Store transfer directory", create_mode=mode,
    )


@contextmanager
def _locked_stores(
    source: StoreEndpoint, destination: ImmutableArtifactBlobProvider,
) -> Iterator[None]:
    close_directory_chain(_open_directory(destination.root, create=True))
    if isinstance(source, RemoteStore):
        with _locked_store(destination, create=True):
            yield
        return

    source_identity = _directory_identity(source.root)
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


@contextmanager
def _locked_store(store: ImmutableArtifactBlobProvider, *, create: bool) -> Iterator[None]:
    identity = _directory_identity(store.root)
    with store._exclusive_lock(create=create):
        if _directory_identity(store.root) != identity:
            raise ValueError(f"Store directory changed while taking its lock: {store.root}")
        yield


def _named_closure(store: ImmutableArtifactBlobProvider, name: str) -> StoreClosureRef:
    value = store.read_ref(name)
    if value is None:
        raise FileNotFoundError(f"Source closure ref is missing: {name}")
    return read_store_closure_ref(value)


def _selected_train(
    store: ImmutableArtifactBlobProvider, closure: StoreClosureRef, selection: _TrainSource,
) -> TrainJobEntry:
    entry = _read_entry(store, closure, selection.attempt, expected=selection.expected)
    if entry.request.id != selection.request_id or entry.recovery.pointer != selection.pointer:
        raise ValueError("Collected Train binding differs from the mirrored request or Stop checkpoint")
    return entry


def _inventory(store: ImmutableArtifactBlobProvider, selection: _Selection) -> _Inventory:
    if isinstance(selection, _ClosureSource):
        chosen = selection.selection
        if isinstance(chosen, str):
            closure = _named_closure(store, chosen)
            index = None
        else:
            closure = chosen
            index = (
                decode_success_index(store.read_ref(INDEX_REF))
                if any(isinstance(root, ReceiptRoot) for root in closure.roots) else None
            )
        return _Inventory(closure=closure, objects=ReceiptClosure(store, index).walk(closure))

    index = decode_success_index(store.read_ref(INDEX_REF))
    receipts = ReceiptClosure(store, index)
    if isinstance(selection, _SuccessSource):
        committed = receipts.receipt_at(selection.receipt)
        if (committed.attempt, committed.request.id) != (selection.attempt, selection.request_id):
            raise ValueError("Source receipt differs from the expected attempt and request")
        if selection.expected_target is not None and committed.target != selection.expected_target:
            raise ValueError("Source receipt differs from the expected execution target")
        closure = StoreClosureRef(roots=(ReceiptRoot(receipt=selection.receipt),), include_ancestry=True)
        origins = (selection.receipt,)
        objects = ()
    else:
        closure = _named_closure(store, _job_ref(selection.attempt))
        entry = _selected_train(store, closure, selection)
        _verify_entry(store, entry)
        origins = _origins(entry)
        objects = receipts.walk(closure)

    by_digest = {ref.sha256: ref for ref in objects}
    times = {}
    for receipt in origins:
        graph = receipts.for_admission(receipt)
        for ref in graph.objects:
            if by_digest.setdefault(ref.sha256, ref) != ref:
                raise ValueError(f"Conflicting object evidence: {ref.sha256}")
        for ref, result in graph.receipts:
            if result.supersedes is not None:
                times[ref.sha256] = index.supersession_times[ref.sha256]
    return _Inventory(
        closure=closure, objects=tuple(by_digest[digest] for digest in sorted(by_digest)),
        supersession_times=times,
    )


def _verify_source(store: ImmutableArtifactBlobProvider, inventory: _Inventory) -> None:
    for ref in inventory.objects:
        store.verify_observed_object(ref, verifier="SSH Store transfer source")


@contextmanager
def _transfer_inventory(
    source: StoreEndpoint, destination: ImmutableArtifactBlobProvider, selection: _Selection,
) -> Iterator[_Inventory]:
    remote = (
        _Inventory.model_validate_json(source.call("inventory", selection))
        if isinstance(source, RemoteStore) else None
    )
    with _locked_stores(source, destination):
        if isinstance(source, RemoteStore):
            assert remote is not None
            yield remote
        else:
            yield _inventory(source, selection)


def _incoming_path(store: ImmutableArtifactBlobProvider, ref: ContentObjectRef) -> Path:
    return Path(store.root) / _INCOMING / content_object_relative_path(ref.sha256)


def _receive(
    inventory: _Inventory, store: ImmutableArtifactBlobProvider, *, prepare: bool,
    verified: Callable[[ContentObjectRef], None] | None = None,
) -> tuple[ContentObjectRef, ...]:
    """Adopt completed staged files, returning only objects still needed."""
    missing = []
    for ref in inventory.objects:
        if store._reuse_object(ref, store._replacement_validator(ref)):
            store._settle_install(ref)
            if verified is not None:
                verified(ref)
            continue
        staged = _incoming_path(store, ref)
        try:
            store._store_file(staged, sha256=ref.sha256, size_bytes=ref.size_bytes)
        except FileNotFoundError:
            if not prepare:
                raise
            close_directory_chain(_open_directory(staged.parent, create=True, mode=0o700))
            missing.append(ref)
        except ArtifactBlobIntegrityError:
            _remove_incoming(staged)
            raise
        else:
            _remove_incoming(staged)
            if verified is not None:
                verified(ref)
    return tuple(missing)


def _remove_incoming(staged: Path) -> None:
    records = _open_directory(staged.parent, create=False)
    try:
        os.unlink(staged.name, dir_fd=records[-1][1])
    except FileNotFoundError:
        pass
    finally:
        close_directory_chain(records)


def _bring_objects(
    source: StoreEndpoint, destination: ImmutableArtifactBlobProvider, inventory: _Inventory,
    *, progress: Callable[[StoreTransferProgress], None] | None = None,
) -> int:
    if isinstance(source, RemoteStore):
        missing = _receive(inventory, destination, prepare=True)
        total_bytes = sum(ref.size_bytes for ref in inventory.objects)
        received = total_bytes - sum(ref.size_bytes for ref in missing)
        remaining = {ref.sha256 for ref in missing}

        def verified(ref: ContentObjectRef) -> None:
            nonlocal received
            if progress is not None and ref.sha256 in remaining:
                remaining.remove(ref.sha256)
                received += ref.size_bytes
                progress(StoreTransferProgress(verified_bytes=received, total_bytes=total_bytes))

        if progress is not None:
            progress(StoreTransferProgress(verified_bytes=received, total_bytes=total_bytes))
        source.connection.copy_objects(source.root, Path(destination.root) / _INCOMING, missing, pull=True)
        _receive(inventory, destination, prepare=False, verified=verified if progress is not None else None)
        return len(missing)
    return sum(destination._clone_object_from(source, ref) for ref in inventory.objects)


def _ref_before(
    store: ImmutableArtifactBlobProvider, closure: StoreClosureRef, update: _RefUpdate,
) -> dict[str, JsonValue] | None:
    if update.name == INDEX_REF or update.name.startswith("runs/"):
        raise ValueError("Store sync cannot publish successful results without admission")
    current = store.read_ref(update.name)
    previous = update.expected.model_dump(mode="json") if update.expected is not None else None
    if current != closure.model_dump(mode="json") and current != previous:
        raise ValueError(f"Destination closure ref changed: {update.name}")
    return current


def _publish(
    store: ImmutableArtifactBlobProvider, inventory: _Inventory, update: _RefUpdate,
    current: dict[str, JsonValue] | None,
) -> None:
    if not store.compare_and_swap_ref(
        update.name, expected=current, replacement=inventory.closure.model_dump(mode="json")
    ):
        raise ValueError(f"Destination closure ref changed: {update.name}")


def _verify_closure(store: ImmutableArtifactBlobProvider, inventory: _Inventory) -> None:
    if ReceiptClosure(store).walk(inventory.closure) != inventory.objects:
        raise ValueError("Transferred inventory differs from its exact Store closure")


def sync_store(
    source: StoreEndpoint,
    destination: StoreEndpoint,
    selection: StoreClosureRef | str,
    *,
    destination_ref: str,
    expected: StoreClosureRef | None = None,
) -> StoreSyncResult:
    """Publish a verified closure from committed receipts or a named packet ref."""
    selected = _ClosureSource(selection=selection)
    update = _RefUpdate(name=destination_ref, expected=expected)
    if isinstance(destination, RemoteStore):
        if isinstance(source, RemoteStore):
            raise ValueError("Store transfer requires one local endpoint")
        with _locked_store(source, create=False):
            inventory = _inventory(source, selected)
            _verify_source(source, inventory)
        request = _PublishRequest(inventory=inventory, ref=update)
        missing = TypeAdapter(tuple[ContentObjectRef, ...]).validate_json(destination.call("missing", request))
        object_refs = {ref.sha256: ref for ref in inventory.objects}
        if len({ref.sha256 for ref in missing}) != len(missing) or any(
            object_refs.get(ref.sha256) != ref for ref in missing
        ):
            raise ValueError("Destination asked for objects outside the exact Store closure")
        with _locked_store(source, create=False):
            destination.connection.copy_objects(Path(source.root), destination.root / _INCOMING, missing, pull=False)
        destination.call("publish", request)
        transferred = len(missing)
    else:
        with _transfer_inventory(source, destination, selected) as inventory:
            current = _ref_before(destination, inventory.closure, update)
            transferred = _bring_objects(source, destination, inventory)
            if isinstance(source, RemoteStore):
                _verify_closure(destination, inventory)
            _publish(destination, inventory, update, current)
    return StoreSyncResult(inventory.closure, transferred, len(inventory.objects) - transferred)


def import_packet(
    packet: ImmutableArtifactBlobProvider,
    destination: ImmutableArtifactBlobProvider,
    *, packet_ref: str, name: str,
) -> StoreSyncResult:
    """Transfer and admit one exact receipt selection without a Host job."""
    selection = _ClosureSource(selection=packet_ref)
    update = _RefUpdate(name=f"packets/{name}")
    with _transfer_inventory(packet, destination, selection) as inventory:
        (root,) = inventory.closure.roots
        if not isinstance(root, ReceiptRoot):
            raise ValueError("A receipt packet must name an exact receipt")
        receipt = ReceiptClosure(packet).receipt_at(root.receipt)
        current = _ref_before(destination, inventory.closure, update)
        transferred = _bring_objects(packet, destination, inventory)
        SuccessView(destination).admit_collected(
            root.receipt, attempt=receipt.attempt, request_id=receipt.request.id,
            supersession_times={}, promote=True, selection=inventory.closure,
        )
        _publish(destination, inventory, update, current)
    return StoreSyncResult(inventory.closure, transferred, len(inventory.objects) - transferred)


def collect_success(
    source: StoreEndpoint,
    destination: ImmutableArtifactBlobProvider,
    receipt: ContentObjectRef,
    *,
    attempt: str,
    request_id: RequestId,
    expected_target: AttemptTarget | None = None,
    verify: Callable[[Success], None] | None = None,
    progress: Callable[[StoreTransferProgress], None] | None = None,
) -> PublishedSuccess:
    """Transfer and admit an accepted result with its original receipt identity."""
    return _collect_success(
        source, destination, receipt, attempt, request_id,
        promote=True, expected_target=expected_target, verify=verify,
        progress=progress,
    )


def collect_success_evidence(
    source: StoreEndpoint,
    destination: ImmutableArtifactBlobProvider,
    receipt: ContentObjectRef,
    *,
    attempt: str,
    request_id: RequestId,
) -> PublishedSuccess:
    """Collect a producer by attempt, preserving the destination's current results."""
    return _collect_success(source, destination, receipt, attempt, request_id, promote=False)


def _collect_success(
    source: StoreEndpoint,
    destination: ImmutableArtifactBlobProvider,
    receipt: ContentObjectRef,
    attempt: str,
    request_id: RequestId,
    *,
    promote: bool,
    expected_target: AttemptTarget | None = None,
    verify: Callable[[Success], None] | None = None,
    progress: Callable[[StoreTransferProgress], None] | None = None,
) -> PublishedSuccess:
    selection = _SuccessSource(
        receipt=receipt, attempt=attempt, request_id=request_id, expected_target=expected_target,
    )
    with _transfer_inventory(source, destination, selection) as inventory:
        _bring_objects(source, destination, inventory, progress=progress)
        if verify is not None:
            verify(ReceiptClosure(destination).receipt_at(receipt))
        return SuccessView(destination).admit_collected(
            receipt, attempt=attempt, request_id=request_id,
            supersession_times=inventory.supersession_times, promote=promote,
            expected_target=expected_target,
        )


def _collect_train_job(
    source: StoreEndpoint, destination: ImmutableArtifactBlobProvider,
    *, attempt: str, request_id: RequestId, pointer: CheckpointLatestPointer,
    expected: ContentObjectRef | None,
    progress: Callable[[StoreTransferProgress], None] | None = None,
) -> TrainJobEntry:
    """Collect a receipt-less stopped attempt and publish its verified Resume custody."""
    selection = _TrainSource(attempt=attempt, request_id=request_id, pointer=pointer, expected=expected)
    with _transfer_inventory(source, destination, selection) as inventory:
        _entry_ref(inventory.closure, selection.expected)
        update = _RefUpdate(name=_job_ref(attempt))
        current = _ref_before(destination, inventory.closure, update)
        _bring_objects(source, destination, inventory, progress=progress)
        entry = _selected_train(destination, inventory.closure, selection)
        _verify_checkpoints(destination, entry)
        receipts = ReceiptClosure(destination)
        for origin in _origins(entry):
            receipt = receipts.receipt_at(origin)
            SuccessView(destination).admit_collected(
                origin, attempt=receipt.attempt, request_id=receipt.request.id,
                supersession_times=inventory.supersession_times, promote=False,
            )
        SuccessView(destination).verify_bindings(entry.request, entry.inputs)
        _publish(destination, inventory, update, current)
        return entry


def _serve(operation: str, root: Path, raw: bytes) -> bytes:
    store = ImmutableArtifactBlobProvider(root)
    if operation == "inventory":
        selection = TypeAdapter(_Selection).validate_json(raw)
        with store._exclusive_lock(create=False):
            inventory = _inventory(store, selection)
            _verify_source(store, inventory)
        return inventory.model_dump_json().encode()
    if operation not in {"missing", "publish"}:
        raise ValueError(f"Unknown Store transfer operation: {operation!r}")
    request = _PublishRequest.model_validate_json(raw)
    close_directory_chain(_open_directory(store.root, create=True))
    with store._exclusive_lock():
        current = _ref_before(store, request.inventory.closure, request.ref)
        if operation == "missing":
            missing = _receive(request.inventory, store, prepare=True)
            return TypeAdapter(tuple[ContentObjectRef, ...]).dump_json(missing)
        _receive(request.inventory, store, prepare=False)
        _verify_closure(store, request.inventory)
        _publish(store, request.inventory, request.ref, current)
        return b""


if __name__ == "__main__":
    _operation, _root = sys.argv[1:]
    sys.stdout.buffer.write(_serve(_operation, Path(_root), sys.stdin.buffer.read()))
