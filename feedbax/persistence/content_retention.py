"""Transactional retention and manual garbage collection for the canonical CAS."""

from __future__ import annotations

import hashlib
import os
import stat
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from feedbax.contracts.artifact_custody import (
    CONTENT_OBJECT_REF_PREFIX,
    ContentGcDecision,
    ContentGcReceipt,
    ContentGcRequest,
    ContentHoldRecord,
    ContentLeaseRecord,
    ContentPendingGc,
    ContentRetentionIndex,
    ContentRetentionTarget,
    ContentRootRecord,
)
from feedbax.contracts.canonical_json import canonical_json_v1_bytes
from feedbax.contracts.strict_json import strict_json_loads

if TYPE_CHECKING:
    from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider


class ContentRetentionError(ValueError):
    """Retention state is invalid, ambiguous, or conflicts with an existing record."""


def _index_path(provider: ImmutableArtifactBlobProvider) -> Path:
    return provider.root / "retention" / "index.json"


def _receipt_path(
    provider: ImmutableArtifactBlobProvider,
    receipt_id: str,
    stage: str,
) -> Path:
    return provider.root / "retention" / "receipts" / receipt_id / f"{stage}.json"


def _quarantine_path(
    provider: ImmutableArtifactBlobProvider,
    pending: ContentPendingGc,
    digest: str,
) -> Path:
    return provider.root / "retention" / "quarantine" / pending.domain / pending.receipt_id / digest


def _read_index(provider: ImmutableArtifactBlobProvider) -> ContentRetentionIndex:
    path = _index_path(provider)
    try:
        before = path.lstat()
    except FileNotFoundError:
        return ContentRetentionIndex()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ContentRetentionError("retention index must be one regular file")
    raw = path.read_bytes()
    after = path.lstat()
    if (before.st_dev, before.st_ino, before.st_size) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
    ):
        raise ContentRetentionError("retention index identity changed during read")
    try:
        payload = strict_json_loads(raw.decode("utf-8"), ref=str(path))
        from feedbax.contracts.migrations import migrate_structured_spec_payload

        migrated = migrate_structured_spec_payload(
            "ContentRetentionIndex",
            payload,
            path=str(path),
        )
        index = ContentRetentionIndex.model_validate(migrated.payload)
    except Exception as exc:
        raise ContentRetentionError(f"retention index is corrupt or unsupported: {exc}") from exc
    if raw != canonical_json_v1_bytes(payload):
        raise ContentRetentionError("retention index is not canonical")
    return index


def _ensure_store_directory(provider: ImmutableArtifactBlobProvider, path: Path) -> None:
    try:
        relative = path.relative_to(provider.root)
    except ValueError as exc:
        raise ContentRetentionError("retention path escapes the canonical content store") from exc
    current = provider.root
    for part in relative.parts:
        current = current / part
        try:
            current.mkdir()
        except FileExistsError:
            pass
        mode = current.lstat().st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise ContentRetentionError(f"retention path contains an unsafe component: {current}")


def _write_new_or_replace(
    provider: ImmutableArtifactBlobProvider,
    path: Path,
    data: bytes,
) -> None:
    _ensure_store_directory(provider, path.parent)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise ContentRetentionError("retention record write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _write_index(provider: ImmutableArtifactBlobProvider, index: ContentRetentionIndex) -> None:
    _write_new_or_replace(
        provider,
        _index_path(provider),
        canonical_json_v1_bytes(index.model_dump(mode="json")),
    )


def _write_receipt(provider: ImmutableArtifactBlobProvider, receipt: ContentGcReceipt) -> None:
    path = _receipt_path(provider, receipt.receipt_id, receipt.stage)
    data = canonical_json_v1_bytes(receipt.model_dump(mode="json"))
    if path.exists():
        if path.read_bytes() != data:
            raise ContentRetentionError("GC receipt identity drifted during recovery")
        return
    _write_new_or_replace(provider, path, data)


def _replace_record(
    index: ContentRetentionIndex, field: str, record: object, key: str
) -> ContentRetentionIndex:
    records = list(getattr(index, field))
    for position, current in enumerate(records):
        if getattr(current, key) == getattr(record, key):
            if current == record:
                return index
            records[position] = record
            break
    else:
        records.append(record)
    records.sort(key=lambda value: getattr(value, key))
    return index.model_copy(update={field: tuple(records), "generation": index.generation + 1})


def _target_digests(
    provider: ImmutableArtifactBlobProvider,
    target: ContentRetentionTarget,
) -> set[str]:
    if target.ref_kind == "object":
        assert target.object is not None
        provider.get_object(target.object)
        return {target.object.sha256}
    if target.ref_kind == "tree":
        assert target.tree is not None
        tree = provider.get_tree(target.tree)
        digests = {target.tree.descriptor.sha256}
        for member in tree.files:
            provider.get_object(member.object)
            digests.add(member.object.sha256)
        return digests
    assert target.manifest is not None
    resolved, _raw = provider.resolve_manifest(target.manifest.kind, target.manifest.id)
    if resolved != target.manifest:
        raise ContentRetentionError(
            f"manifest retention target identity drifted: {target.manifest.ref}"
        )
    return {target.manifest.object.sha256}


def _validate_targets(
    provider: ImmutableArtifactBlobProvider,
    targets: tuple[ContentRetentionTarget, ...],
) -> set[str]:
    digests: set[str] = set()
    for target in targets:
        digests.update(_target_digests(provider, target))
    return digests


def register_root(
    provider: ImmutableArtifactBlobProvider,
    root: ContentRootRecord,
) -> ContentRootRecord:
    """Register one exact durable root; a reused ID must carry identical meaning."""
    if root.released:
        raise ContentRetentionError("new roots must be active")
    if root.domain != "canonical":
        raise ContentRetentionError(
            "remote attempt caches are non-authoritative and cannot be roots"
        )
    with provider._exclusive_lock():
        index = _read_index(provider)
        if index.pending_gc is not None:
            raise ContentRetentionError("recover the pending GC generation before changing roots")
        _validate_targets(provider, root.targets)
        existing = next((item for item in index.roots if item.root_id == root.root_id), None)
        if existing is not None and existing != root:
            raise ContentRetentionError(
                f"root ID already has different exact content: {root.root_id}"
            )
        updated = _replace_record(index, "roots", root, "root_id")
        if updated != index:
            _write_index(provider, updated)
        return root


def set_run_state(
    provider: ImmutableArtifactBlobProvider,
    root_id: str,
    *,
    nonterminal: bool,
    restartable: bool,
) -> ContentRootRecord:
    """Update explicit run liveness without deriving it from elapsed time."""
    with provider._exclusive_lock():
        index = _read_index(provider)
        if index.pending_gc is not None:
            raise ContentRetentionError("recover the pending GC generation before changing roots")
        current = next((item for item in index.roots if item.root_id == root_id), None)
        if current is None or current.root_kind != "run" or current.released:
            raise ContentRetentionError(f"active run root is unavailable: {root_id}")
        updated_record = current.model_copy(
            update={"nonterminal": nonterminal, "restartable": restartable}
        )
        updated = _replace_record(index, "roots", updated_record, "root_id")
        if updated != index:
            _write_index(provider, updated)
        return updated_record


def release_root(provider: ImmutableArtifactBlobProvider, root_id: str) -> ContentRootRecord:
    """Explicitly release a terminal, non-restartable root."""
    with provider._exclusive_lock():
        index = _read_index(provider)
        if index.pending_gc is not None:
            raise ContentRetentionError("recover the pending GC generation before changing roots")
        current = next((item for item in index.roots if item.root_id == root_id), None)
        if current is None:
            raise ContentRetentionError(f"root is unavailable: {root_id}")
        if current.nonterminal or current.restartable:
            raise ContentRetentionError("nonterminal or restartable run roots cannot be released")
        released = current.model_copy(update={"released": True})
        updated = _replace_record(index, "roots", released, "root_id")
        if updated != index:
            _write_index(provider, updated)
        return released


def place_hold(
    provider: ImmutableArtifactBlobProvider,
    hold: ContentHoldRecord,
) -> ContentHoldRecord:
    """Place one explicit scientific or owner hold."""
    if hold.released:
        raise ContentRetentionError("new holds must be active")
    if hold.domain != "canonical":
        raise ContentRetentionError("remote attempt caches cannot become authoritative holds")
    with provider._exclusive_lock():
        index = _read_index(provider)
        if index.pending_gc is not None:
            raise ContentRetentionError("recover the pending GC generation before changing holds")
        _validate_targets(provider, hold.targets)
        existing = next((item for item in index.holds if item.hold_id == hold.hold_id), None)
        if existing is not None and existing != hold:
            raise ContentRetentionError(
                f"hold ID already has different exact content: {hold.hold_id}"
            )
        updated = _replace_record(index, "holds", hold, "hold_id")
        if updated != index:
            _write_index(provider, updated)
        return hold


def release_hold(provider: ImmutableArtifactBlobProvider, hold_id: str) -> ContentHoldRecord:
    """Release one exact hold idempotently."""
    return _release_record(provider, "holds", "hold_id", hold_id)


def acquire_lease(
    provider: ImmutableArtifactBlobProvider,
    lease: ContentLeaseRecord,
) -> ContentLeaseRecord:
    """Acquire an exact lease under the same lock used to mark GC candidates."""
    if lease.released:
        raise ContentRetentionError("new leases must be active")
    with provider._exclusive_lock():
        index = _read_index(provider)
        if index.pending_gc is not None:
            raise ContentRetentionError(
                "recover the pending GC generation before acquiring a lease"
            )
        _validate_targets(provider, lease.targets)
        existing = next((item for item in index.leases if item.lease_id == lease.lease_id), None)
        if existing is not None and existing != lease:
            same_identity = (
                existing.model_copy(
                    update={
                        "expires_at": lease.expires_at,
                        "expiry_disposition": lease.expiry_disposition,
                    }
                )
                == lease
            )
            if not existing.released and same_identity and existing.expiry_disposition == "release":
                return existing
            same_owner = existing.model_copy(update={"expires_at": lease.expires_at}) == lease
            if existing.released or not same_owner or lease.expires_at < existing.expires_at:
                raise ContentRetentionError(
                    f"lease ID already has different exact content: {lease.lease_id}"
                )
        updated = _replace_record(index, "leases", lease, "lease_id")
        if updated != index:
            _write_index(provider, updated)
        return lease


def renew_lease(
    provider: ImmutableArtifactBlobProvider,
    lease_id: str,
    *,
    expires_at: datetime,
) -> ContentLeaseRecord:
    """Renew one active lease; callers choose the timestamp and cadence."""
    if expires_at.tzinfo is None or expires_at.utcoffset() is None:
        raise ContentRetentionError("lease expiry must be timezone-aware")
    with provider._exclusive_lock():
        index = _read_index(provider)
        if index.pending_gc is not None:
            raise ContentRetentionError("recover the pending GC generation before renewing a lease")
        current = next((item for item in index.leases if item.lease_id == lease_id), None)
        if current is None or current.released:
            raise ContentRetentionError(f"active lease is unavailable: {lease_id}")
        renewed = current.model_copy(update={"expires_at": expires_at.astimezone(timezone.utc)})
        updated = _replace_record(index, "leases", renewed, "lease_id")
        if updated != index:
            _write_index(provider, updated)
        return renewed


def schedule_lease_release(
    provider: ImmutableArtifactBlobProvider,
    lease_id: str,
    *,
    expires_at: datetime,
) -> ContentLeaseRecord:
    """Atomically turn one blocking in-flight lease into a bounded grace lease."""
    if expires_at.tzinfo is None or expires_at.utcoffset() is None:
        raise ContentRetentionError("lease expiry must be timezone-aware")
    with provider._exclusive_lock():
        index = _read_index(provider)
        if index.pending_gc is not None:
            raise ContentRetentionError(
                "recover the pending GC generation before scheduling a lease release"
            )
        current = next((item for item in index.leases if item.lease_id == lease_id), None)
        if current is None or current.released:
            raise ContentRetentionError(f"active lease is unavailable: {lease_id}")
        if current.expiry_disposition == "release":
            return current
        scheduled = current.model_copy(
            update={
                "expires_at": expires_at.astimezone(timezone.utc),
                "expiry_disposition": "release",
            }
        )
        updated = _replace_record(index, "leases", scheduled, "lease_id")
        if updated != index:
            _write_index(provider, updated)
        return scheduled


def release_lease(provider: ImmutableArtifactBlobProvider, lease_id: str) -> ContentLeaseRecord:
    """Durably release one lease; expiry alone never releases it."""
    return _release_record(provider, "leases", "lease_id", lease_id)


def _release_record(provider, field: str, key: str, record_id: str):
    with provider._exclusive_lock():
        index = _read_index(provider)
        if index.pending_gc is not None:
            raise ContentRetentionError(
                "recover the pending GC generation before releasing custody"
            )
        current = next(
            (item for item in getattr(index, field) if getattr(item, key) == record_id), None
        )
        if current is None:
            raise ContentRetentionError(f"retention record is unavailable: {record_id}")
        released = current.model_copy(update={"released": True})
        updated = _replace_record(index, field, released, key)
        if updated != index:
            _write_index(provider, updated)
        return released


def _inventory(provider: ImmutableArtifactBlobProvider) -> tuple[dict[str, int], list[str]]:
    root = provider.root / "artifacts" / "sha256"
    if not root.exists():
        return {}, []
    root_mode = root.lstat().st_mode
    if stat.S_ISLNK(root_mode) or not stat.S_ISDIR(root_mode):
        return {}, ["canonical object namespace is not one directory"]
    objects: dict[str, int] = {}
    errors: list[str] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        current = path.lstat()
        if path.is_dir() and not stat.S_ISLNK(current.st_mode):
            if path.name == ".feedbax-materialization-staging" and any(path.iterdir()):
                errors.append(f"incomplete materialization staging directory: {relative}")
            continue
        digest = path.name
        expected = len(relative.parts) == 2 and relative.parts[0] == digest[:2]
        expected = expected and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)
        if not expected or not stat.S_ISREG(current.st_mode) or current.st_nlink != 1:
            errors.append(f"unsafe or unknown object-store entry: {relative}")
            continue
        raw = path.read_bytes()
        after = path.lstat()
        if (current.st_dev, current.st_ino, current.st_size) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
        ) or hashlib.sha256(raw).hexdigest() != digest:
            errors.append(f"object identity drift: {digest}")
            continue
        objects[digest] = len(raw)
    return objects, errors


def _live_closure(
    provider: ImmutableArtifactBlobProvider,
    index: ContentRetentionIndex,
    *,
    now: datetime,
) -> tuple[
    set[str],
    set[str],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    list[str],
]:
    live: set[str] = set()
    expired_release: set[str] = set()
    expired_lease_ids: list[str] = []
    errors: list[str] = []
    roots = tuple(item.root_id for item in index.roots if not item.released)
    leases = tuple(
        item.lease_id
        for item in index.leases
        if not item.released
        and not (item.expiry_disposition == "release" and item.expires_at <= now)
    )
    holds = tuple(item.hold_id for item in index.holds if not item.released)
    for record in (*index.roots, *index.leases, *index.holds):
        if record.released:
            continue
        if isinstance(record, ContentRootRecord):
            if record.nonterminal or record.restartable:
                errors.append(f"run root remains nonterminal or restartable: {record.root_id}")
            if record.root_kind == "migration":
                errors.append(f"migration root remains incomplete: {record.root_id}")
        if isinstance(record, ContentLeaseRecord) and record.expires_at <= now:
            if record.expiry_disposition == "block":
                errors.append(f"lease expired without explicit release: {record.lease_id}")
            else:
                expired_lease_ids.append(record.lease_id)
                try:
                    expired_release.update(_validate_targets(provider, record.targets))
                except Exception as exc:
                    errors.append(f"invalid expired retention record {record!r}: {exc}")
                continue
        try:
            live.update(_validate_targets(provider, record.targets))
        except Exception as exc:
            errors.append(f"invalid live retention record {record!r}: {exc}")
    return (
        live,
        expired_release,
        tuple(sorted(expired_lease_ids)),
        roots,
        leases,
        holds,
        errors,
    )


def _receipt(
    *,
    receipt_id: str,
    request: ContentGcRequest,
    generation: int,
    stage: str,
    roots: tuple[str, ...],
    leases: tuple[str, ...],
    holds: tuple[str, ...],
    decisions: tuple[ContentGcDecision, ...],
    errors: tuple[str, ...] = (),
) -> ContentGcReceipt:
    return ContentGcReceipt(
        receipt_id=receipt_id,
        domain=request.domain,
        generation=generation,
        dry_run=not request.destructive,
        stage=stage,
        root_ids=roots,
        lease_ids=leases,
        hold_ids=holds,
        byte_count=sum(item.size_bytes for item in decisions if item.decision != "live"),
        decisions=decisions,
        errors=errors,
    )


def collect_garbage(
    provider: ImmutableArtifactBlobProvider,
    request: ContentGcRequest | None = None,
    *,
    now: datetime | None = None,
) -> ContentGcReceipt:
    """Report candidates by default, or run one explicit guarded destructive generation."""
    request = request or ContentGcRequest()
    supplied_now = now or datetime.now(timezone.utc)
    if supplied_now.tzinfo is None or supplied_now.utcoffset() is None:
        raise ContentRetentionError("GC observation time must be timezone-aware")
    observed_at = supplied_now.astimezone(timezone.utc)
    receipt_id = uuid.uuid4().hex
    lock_path = provider.root / ".content-store.lock"
    if not lock_path.exists() or not _index_path(provider).exists():
        return _receipt(
            receipt_id=receipt_id,
            request=request,
            generation=0,
            stage="no-op",
            roots=(),
            leases=(),
            holds=(),
            decisions=(),
            errors=("retention state has not been initialized",),
        )
    with provider._exclusive_lock():
        try:
            index = _read_index(provider)
        except Exception as exc:
            return _receipt(
                receipt_id=receipt_id,
                request=request,
                generation=0,
                stage="no-op",
                roots=(),
                leases=(),
                holds=(),
                decisions=(),
                errors=(str(exc),),
            )
        if index.pending_gc is not None:
            if not request.destructive:
                return _receipt(
                    receipt_id=receipt_id,
                    request=request,
                    generation=index.generation,
                    stage="no-op",
                    roots=(),
                    leases=(),
                    holds=(),
                    decisions=(),
                    errors=("a destructive GC generation requires recovery",),
                )
            return _recover_locked(provider, index)
        objects, errors = _inventory(provider)
        (
            live,
            expired_release,
            expired_lease_ids,
            roots,
            leases,
            holds,
            live_errors,
        ) = _live_closure(
            provider,
            index,
            now=observed_at,
        )
        errors.extend(live_errors)
        decisions = tuple(
            ContentGcDecision(
                object_id=f"{CONTENT_OBJECT_REF_PREFIX}{digest}",
                sha256=digest,
                size_bytes=size,
                decision="live" if digest in live else "candidate",
                reason=(
                    "reachable from explicit custody"
                    if digest in live
                    else (
                        "time-bounded lease expired"
                        if digest in expired_release
                        else "unreferenced"
                    )
                ),
            )
            for digest, size in sorted(objects.items())
        )
        if errors:
            return _receipt(
                receipt_id=receipt_id,
                request=request,
                generation=index.generation,
                stage="no-op",
                roots=roots,
                leases=leases,
                holds=holds,
                decisions=decisions,
                errors=tuple(errors),
            )
        if not request.destructive:
            return _receipt(
                receipt_id=receipt_id,
                request=request,
                generation=index.generation,
                stage="reported",
                roots=roots,
                leases=leases,
                holds=holds,
                decisions=decisions,
            )
        if request.expected_generation != index.generation:
            return _receipt(
                receipt_id=receipt_id,
                request=request,
                generation=index.generation,
                stage="no-op",
                roots=roots,
                leases=leases,
                holds=holds,
                decisions=decisions,
                errors=("retention generation drifted after candidate reporting",),
            )
        candidates = tuple(item for item in decisions if item.decision == "candidate")
        pending = ContentPendingGc(
            receipt_id=receipt_id,
            domain=request.domain,
            generation=index.generation + 1,
            stage="marked",
            root_ids=roots,
            lease_ids=leases,
            expired_lease_ids=expired_lease_ids,
            hold_ids=holds,
            decisions=candidates,
        )
        marked_index = index.model_copy(
            update={"generation": pending.generation, "pending_gc": pending}
        )
        _write_index(provider, marked_index)
        _write_receipt(
            provider,
            _receipt(
                receipt_id=receipt_id,
                request=request,
                generation=pending.generation,
                stage="marked",
                roots=roots,
                leases=leases,
                holds=holds,
                decisions=candidates,
            ),
        )
        return _recover_locked(provider, marked_index)


def recover_garbage_collection(provider: ImmutableArtifactBlobProvider) -> ContentGcReceipt | None:
    """Resume one marked generation idempotently; return ``None`` when none exists."""
    with provider._exclusive_lock():
        index = _read_index(provider)
        if index.pending_gc is None:
            return None
        return _recover_locked(provider, index)


def _recover_locked(
    provider: ImmutableArtifactBlobProvider,
    index: ContentRetentionIndex,
) -> ContentGcReceipt:
    pending = index.pending_gc
    assert pending is not None
    request = ContentGcRequest(
        domain=pending.domain,
        destructive=True,
        expected_generation=pending.generation,
    )
    quarantined: list[ContentGcDecision] = []
    try:
        for candidate in pending.decisions:
            source = provider._canonical_path(candidate.sha256)
            destination = _quarantine_path(provider, pending, candidate.sha256)
            _ensure_store_directory(provider, destination.parent)
            if source.exists() and destination.exists():
                raise ContentRetentionError(
                    f"candidate exists in canonical and quarantine paths: {candidate.sha256}"
                )
            if source.exists():
                current = source.lstat()
                if (
                    not stat.S_ISREG(current.st_mode)
                    or current.st_nlink != 1
                    or current.st_size != candidate.size_bytes
                    or hashlib.sha256(source.read_bytes()).hexdigest() != candidate.sha256
                ):
                    raise ContentRetentionError(
                        f"candidate identity drift before quarantine: {candidate.sha256}"
                    )
                os.replace(source, destination)
            if not destination.exists():
                if pending.stage == "quarantined":
                    quarantined.append(candidate)
                    continue
                raise ContentRetentionError(f"marked candidate is missing: {candidate.sha256}")
            current = destination.lstat()
            if (
                not stat.S_ISREG(current.st_mode)
                or current.st_nlink != 1
                or current.st_size != candidate.size_bytes
                or hashlib.sha256(destination.read_bytes()).hexdigest() != candidate.sha256
            ):
                raise ContentRetentionError(
                    f"quarantined candidate identity drift: {candidate.sha256}"
                )
            quarantined.append(
                candidate.model_copy(
                    update={"decision": "quarantined", "reason": "generation-guarded quarantine"}
                )
            )
        quarantined_pending = pending.model_copy(
            update={"stage": "quarantined", "decisions": tuple(quarantined)}
        )
        quarantined_index = index.model_copy(update={"pending_gc": quarantined_pending})
        _write_index(provider, quarantined_index)
        quarantined_receipt = _receipt(
            receipt_id=pending.receipt_id,
            request=request,
            generation=pending.generation,
            stage="quarantined",
            roots=pending.root_ids,
            leases=pending.lease_ids,
            holds=pending.hold_ids,
            decisions=tuple(quarantined),
        )
        _write_receipt(provider, quarantined_receipt)
        deleted: list[ContentGcDecision] = []
        for candidate in quarantined:
            path = _quarantine_path(provider, quarantined_pending, candidate.sha256)
            if path.exists():
                current = path.lstat()
                if (
                    not stat.S_ISREG(current.st_mode)
                    or current.st_nlink != 1
                    or current.st_size != candidate.size_bytes
                    or hashlib.sha256(path.read_bytes()).hexdigest() != candidate.sha256
                ):
                    raise ContentRetentionError(
                        f"quarantined candidate identity drift before sweep: {candidate.sha256}"
                    )
        for candidate in quarantined:
            path = _quarantine_path(provider, quarantined_pending, candidate.sha256)
            if path.exists():
                path.unlink()
            deleted.append(
                candidate.model_copy(
                    update={"decision": "deleted", "reason": "quarantine identity verified"}
                )
            )
        expired_lease_ids = set(quarantined_pending.expired_lease_ids)
        completed = quarantined_index.model_copy(
            update={
                "leases": tuple(
                    lease.model_copy(update={"released": True})
                    if lease.lease_id in expired_lease_ids
                    else lease
                    for lease in quarantined_index.leases
                ),
                "pending_gc": None,
            }
        )
        _write_index(provider, completed)
        receipt = _receipt(
            receipt_id=pending.receipt_id,
            request=request,
            generation=pending.generation,
            stage="swept",
            roots=pending.root_ids,
            leases=pending.lease_ids,
            holds=pending.hold_ids,
            decisions=tuple(deleted),
        )
        _write_receipt(provider, receipt)
        return receipt
    except Exception as exc:
        return _receipt(
            receipt_id=pending.receipt_id,
            request=request,
            generation=pending.generation,
            stage="no-op",
            roots=pending.root_ids,
            leases=pending.lease_ids,
            holds=pending.hold_ids,
            decisions=tuple(quarantined) or pending.decisions,
            errors=(str(exc),),
        )


__all__ = [
    "ContentRetentionError",
    "acquire_lease",
    "collect_garbage",
    "place_hold",
    "recover_garbage_collection",
    "register_root",
    "release_hold",
    "release_lease",
    "release_root",
    "renew_lease",
    "schedule_lease_release",
    "set_run_state",
]
