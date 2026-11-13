"""Measure an immutable Store from a read-only snapshot of its known roots."""

from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
import os
from pathlib import Path
import stat

from feedbax.contracts.artifact_custody import (
    CONTENT_MANIFEST_REF_SCHEMA_ID,
    CONTENT_OBJECT_REF_SCHEMA_ID,
    CONTENT_TREE_REF_SCHEMA_ID,
    ContentManifestRef,
    ContentObjectRef,
    ContentRetentionTarget,
    ContentTreeRef,
)
from feedbax.contracts.effective_success import (
    INDEX_REF,
    Success,
    decode_success_index,
)
from feedbax.contracts.dataset import DatasetProductRef
from feedbax.contracts.store_closure import (
    STORE_CLOSURE_REF_SCHEMA,
    ReceiptRoot,
    StoreClosureRef,
    read_store_closure_ref,
)
from feedbax.contracts.strict_json import strict_json_loads
from feedbax.persistence.artifact_custody import (
    ImmutableArtifactBlobProvider,
    _manifest_reference_key,
    _read_store_document,
)
from feedbax.persistence.content_retention import (
    ContentRetentionError,
    _inventory_stats,
    _read_index,
    _target_objects,
)
from feedbax.persistence.receipt_closure import ReceiptClosure
from feedbax.persistence.receipt_content import ClosureVisitors


FRESH_OBJECT_GRACE_SECONDS = 24 * 60 * 60
DEFAULT_SUPERSEDED_DAYS = 14


def superseded_days(value: object) -> int:
    if value is None or value == "none":
        return 0
    if type(value) is not int or value < 0:
        raise ValueError("settings.keep_superseded must be a nonnegative day count or 'none'")
    return value


@dataclass(frozen=True)
class RootObservation:
    """A typed root from an existing owner, optionally with exact supersession evidence."""

    name: str
    targets: tuple[ContentRetentionTarget, ...]
    superseded_at: datetime | None = None


@dataclass(frozen=True)
class Size:
    objects: int
    bytes: int


@dataclass(frozen=True)
class RootSize:
    name: str
    closure: Size
    shared: Size
    unshared: Size | None


@dataclass(frozen=True)
class Reachability:
    """Logical byte sizes. Fresh bytes count objects protected only by grace."""

    as_of: str
    keep_superseded_days: int
    fresh_grace_seconds: int
    inventory: Size
    retained: Size
    shared: Size
    fresh: Size
    eligible: Size | None
    eligible_digests: tuple[str, ...] | None
    unclassified: Size
    roots: tuple[RootSize, ...]
    gaps: tuple[str, ...]


def object_target(ref: ContentObjectRef) -> ContentRetentionTarget:
    return ContentRetentionTarget(ref_kind="object", object=ref)


def _success_roots(
    store: ImmutableArtifactBlobProvider,
    value: dict,
    gaps: list[str],
    *,
    include_history: bool = False,
    visitors: ClosureVisitors | None = None,
) -> Iterator[RootObservation]:
    index = decode_success_index(value)
    closure = ReceiptClosure(store, index)
    receipts: dict[str, tuple[ContentObjectRef, Success]] = {}
    for attempt in index.attempts:
        try:
            if visitors is not None:
                visitors.object(index.attempts[attempt])
            ref, receipt = closure.receipt_for_attempt(attempt)
            receipts[ref.sha256] = ref, receipt
        except (ValueError, OSError) as error:
            gaps.append(f"successful attempt {attempt}: {error}")

    def receipt_at(ref: ContentObjectRef) -> Success:
        found = receipts.get(ref.sha256)
        if found is None or found[0] != ref:
            raise ContentRetentionError(f"committed receipt is unavailable: {ref.sha256}")
        return found[1]

    def targets(ref: ContentObjectRef) -> tuple[ContentRetentionTarget, ...]:
        selection = StoreClosureRef(roots=(ReceiptRoot(receipt=ref),), include_ancestry=True)
        return tuple(object_target(item) for item in closure.walk(selection, visitors=visitors))

    current: dict[str, ContentObjectRef] = {}
    for key, ref in index.authored.items():
        try:
            if receipt_at(ref).request.id != key:
                raise ContentRetentionError("authored ref points to a different request")
            current[ref.sha256] = ref
        except ContentRetentionError as error:
            gaps.append(f"successful authored ref {key}: {error}")
    for key, ref in index.effective.items():
        try:
            if receipt_at(ref).effective != key:
                raise ContentRetentionError("effective ref points to different bindings")
            current[ref.sha256] = ref
        except ContentRetentionError as error:
            gaps.append(f"successful effective ref {key}: {error}")
    selected = (ref for ref, _ in receipts.values()) if include_history else current.values()
    for ref in selected:
        try:
            yield RootObservation(f"success:{receipt_at(ref).attempt}", targets(ref))
        except (ValueError, OSError) as error:
            gaps.append(f"success {ref.sha256}: {error}")

    predecessors: set[str] = set()
    for ref, successor in receipts.values():
        previous = successor.supersedes
        when = index.supersession_times.get(ref.sha256)
        if previous is None:
            if when is not None:
                gaps.append(f"success {successor.attempt}: timestamp has no predecessor")
            continue
        try:
            predecessor = receipt_at(previous)
            if predecessor.request.id != successor.request.id or previous == ref:
                raise ContentRetentionError("supersession differs from the authored request")
            if previous.sha256 in predecessors:
                raise ContentRetentionError("predecessor has more than one successor")
            predecessors.add(previous.sha256)
            if when is None and not include_history:
                gaps.append(
                    f"success {successor.attempt}: authoritative supersession time unavailable"
                )
            if not include_history:
                yield RootObservation(
                    f"superseded-success:{predecessor.attempt}",
                    targets(previous),
                    when,
                )
        except (ValueError, OSError) as error:
            gaps.append(f"success {successor.attempt}: {error}")


def _files(directory: Path) -> Iterator[Path]:
    try:
        mode = directory.lstat().st_mode
    except FileNotFoundError:
        return
    if not stat.S_ISDIR(mode):
        raise ContentRetentionError(f"expected a directory at {directory}")
    for path in sorted(directory.iterdir()):
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            yield from _files(path)
        else:
            yield path


def _store_refs(
    store: ImmutableArtifactBlobProvider,
    gaps: list[str],
    *,
    retention_gaps: list[str],
    include_history: bool = False,
    visitors: ClosureVisitors | None = None,
) -> Iterator[RootObservation]:
    root = store.root / "refs"
    for path in _files(root):
        name = path.relative_to(root).as_posix()
        try:
            if path != store._ref_path(name) or (record := _read_store_document(path)) is None:
                raise ContentRetentionError("ref changed while being read")
            value = record[1]
            if name == INDEX_REF:
                yield from _success_roots(
                    store, value, gaps, include_history=include_history, visitors=visitors,
                )
                continue
            if name == "plan" or name.startswith("runs/"):
                retention_gaps.append(f"ref {name}: plan or result root contract is unavailable")
            if value.get("schema_version") == STORE_CLOSURE_REF_SCHEMA:
                closure = ReceiptClosure(store).walk(read_store_closure_ref(value), visitors=visitors)
                yield RootObservation(f"ref:{name}", tuple(object_target(item) for item in closure))
                continue
            if value.keys() == {"content", "descriptor_sha256"}:
                dataset = DatasetProductRef.model_validate(value)
                if visitors is not None:
                    visitors.array(dataset.object_ref, dataset.descriptor_sha256)
                yield RootObservation(f"ref:{name}", (object_target(dataset.object_ref),))
                continue
            match value.get("schema_id"):
                case identity if identity == CONTENT_OBJECT_REF_SCHEMA_ID:
                    target = object_target(ContentObjectRef.model_validate(value))
                case identity if identity == CONTENT_TREE_REF_SCHEMA_ID:
                    target = ContentRetentionTarget(
                        ref_kind="tree",
                        tree=ContentTreeRef.model_validate(value),
                    )
                case identity if identity == CONTENT_MANIFEST_REF_SCHEMA_ID:
                    target = ContentRetentionTarget(
                        ref_kind="manifest",
                        manifest=ContentManifestRef.model_validate(value),
                    )
                    retention_gaps.append(
                        f"ref {name}: typed receipt closure and supersession are unavailable"
                    )
                case _:
                    raise ContentRetentionError("no known typed root contract")
            yield RootObservation(f"ref:{name}", (target,))
        except (ValueError, OSError) as error:
            gaps.append(f"ref {name}: {error}")

    manifest_root = store.root / "references" / "manifests" / "sha256"
    for path in _files(manifest_root):
        name = path.relative_to(store.root).as_posix()
        try:
            if path != store._manifest_reference_path(path.name) or len(path.name) != 64:
                raise ContentRetentionError("unexpected manifest ref path")
            raw = store._read_manifest_reference(path.name, required=True)
            ref = ContentManifestRef.model_validate(strict_json_loads(raw))
            if _manifest_reference_key(ref.kind, ref.id) != path.name:
                raise ContentRetentionError("manifest ref does not match its canonical identity")
            yield RootObservation(
                f"manifest:{ref.kind}/{ref.id}",
                (ContentRetentionTarget(ref_kind="manifest", manifest=ref),),
            )
            retention_gaps.append(f"manifest {ref.kind}/{ref.id}: legacy receipt closure is unavailable")
        except (ValueError, OSError) as error:
            gaps.append(f"{name}: {error}")


def _legacy_roots(
    store: ImmutableArtifactBlobProvider,
    now: datetime,
    gaps: list[str],
    *,
    retention_gaps: list[str],
) -> Iterator[RootObservation]:
    index = _read_index(store)
    if index.pending_gc is not None:
        gaps.append("legacy Store collection has a pending generation")
    for record in index.roots:
        if not record.released:
            if record.nonterminal or record.restartable or record.root_kind == "migration":
                retention_gaps.append(f"legacy root {record.root_id} is still open")
            yield RootObservation(f"legacy-root:{record.root_id}", record.targets)
    for record in index.holds:
        if not record.released:
            yield RootObservation(f"legacy-hold:{record.hold_id}", record.targets)
    for record in index.leases:
        if not record.released and (
            record.expiry_disposition != "release" or record.expires_at > now
        ):
            if record.expires_at <= now:
                retention_gaps.append(f"legacy lease {record.lease_id} expired without release")
            yield RootObservation(f"legacy-lease:{record.lease_id}", record.targets)


def store_roots(
    store: ImmutableArtifactBlobProvider,
    now: datetime,
    errors: list[str],
    *,
    retention_gaps: list[str],
    include_history: bool = False,
    visitors: ClosureVisitors | None = None,
) -> Iterator[RootObservation]:
    """Observe stored root authorities while the caller holds the Store lock."""
    if os.path.lexists(store.root / "jobs"):
        errors.append("Store jobs namespace has no supported root contract")
    readers: tuple[Callable[[], Iterator[RootObservation]], ...] = (
        lambda: _store_refs(
            store, errors, retention_gaps=retention_gaps, include_history=include_history, visitors=visitors,
        ),
        lambda: _legacy_roots(store, now, errors, retention_gaps=retention_gaps),
    )
    for read in readers:
        try:
            yield from read()
        except (ValueError, OSError) as error:
            errors.append(str(error))


def measure_store(
    store: ImmutableArtifactBlobProvider,
    *,
    now: datetime,
    keep_superseded_days: int,
    observations: tuple[RootObservation, ...] = (),
    gaps: tuple[str, ...] = (),
) -> Reachability:
    """Count immutable object sizes. Eligibility requires complete root evidence."""
    if now.utcoffset() is None:
        raise ValueError("the observation time must be timezone-aware")
    now = now.astimezone(UTC)
    window = superseded_days(keep_superseded_days) * FRESH_OBJECT_GRACE_SECONDS
    incomplete = list(gaps)
    closures: dict[str, set[str]] = {}
    with store._exclusive_lock(create=False):
        inventory, errors = _inventory_stats(store)
        incomplete.extend(errors)
        for root in store_roots(store, now, incomplete, retention_gaps=incomplete):
            _mark(store, inventory, closures, root, now, window, incomplete)
        for root in observations:
            _mark(store, inventory, closures, root, now, window, incomplete)

        owners = Counter(digest for closure in closures.values() for digest in closure)
        retained = set(owners)
        shared = {digest for digest, count in owners.items() if count > 1}
        other = inventory.keys() - retained
        grace_cutoff = now.timestamp() - FRESH_OBJECT_GRACE_SECONDS
        fresh = {digest for digest in other if inventory[digest].st_mtime > grace_cutoff}
        possible = other - fresh

        def size(digests) -> Size:
            return Size(len(digests), sum(inventory[digest].st_size for digest in digests))

        return Reachability(
            now.isoformat(),
            keep_superseded_days,
            FRESH_OBJECT_GRACE_SECONDS,
            size(inventory),
            size(retained),
            size(shared),
            size(fresh),
            None if incomplete else size(possible),
            None if incomplete else tuple(sorted(possible)),
            size(possible if incomplete else ()),
            tuple(
                RootSize(
                    name,
                    size(closure),
                    size(closure & shared),
                    None if incomplete else size(closure - shared),
                )
                for name, closure in sorted(closures.items())
            ),
            tuple(sorted(set(incomplete))),
        )


def _mark(
    store: ImmutableArtifactBlobProvider,
    inventory: dict[str, os.stat_result],
    closures: dict[str, set[str]],
    root: RootObservation,
    now: datetime,
    window: int,
    gaps: list[str],
) -> None:
    if root.superseded_at is not None:
        if root.superseded_at.utcoffset() is None:
            gaps.append(f"{root.name}: supersession time lacks a timezone")
        elif not window or (now - root.superseded_at).total_seconds() >= window:
            return
    closure = closures.setdefault(root.name, set())
    for target in root.targets:
        try:
            for ref in _target_objects(store, target):
                info = inventory.get(ref.sha256)
                if info is None or info.st_size != ref.size_bytes:
                    gaps.append(f"{root.name}: missing object or size mismatch: {ref.sha256}")
                else:
                    closure.add(ref.sha256)
        except (ValueError, OSError) as error:
            gaps.append(f"{root.name}: {error}")
