"""Transactional retention and exact lease-driven collection for the canonical CAS."""

from __future__ import annotations

import hashlib
import os
import stat
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from feedbax.contracts.artifact_custody import (
    CONTENT_OBJECT_REF_PREFIX,
    ArtifactBlobCustodyError,
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


def _quarantine_directory(
    provider: ImmutableArtifactBlobProvider,
    pending: ContentPendingGc,
) -> Path:
    return provider.root / "retention" / "quarantine" / pending.domain / pending.receipt_id


def _automatic_job_path(provider: ImmutableArtifactBlobProvider, job_id: str) -> Path:
    return provider.root / "retention" / "automatic-collections" / f"{job_id}.json"


def _is_single_regular_file(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    return stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1


def _entry_exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


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


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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
        # An earlier attempt may have created the directory but failed to sync its parent.
        _sync_directory(current.parent)


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
        _sync_directory(path.parent)
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
    if _entry_exists(path):
        _ensure_store_directory(provider, path.parent)
        if not _is_single_regular_file(path) or path.read_bytes() != data:
            raise ContentRetentionError("GC receipt identity drifted during recovery")
        _sync_directory(path.parent)
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
        provider.verify_object(target.object)
        return {target.object.sha256}
    if target.ref_kind == "tree":
        assert target.tree is not None
        tree = provider.get_tree(target.tree)
        digests = {target.tree.descriptor.sha256}
        for member in tree.files:
            provider.verify_object(member.object)
            digests.add(member.object.sha256)
        return digests
    assert target.manifest is not None
    resolved = provider._resolve_manifest_ref(target.manifest.kind, target.manifest.id)
    provider.verify_object(resolved.object)
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


def require_active_root(
    provider: ImmutableArtifactBlobProvider,
    root_id: str,
) -> ContentRootRecord:
    """Verify one active canonical root and its retained objects without changing it."""
    with provider._exclusive_lock():
        index = _read_index(provider)
        if index.pending_gc is not None:
            raise ContentRetentionError("recover the pending GC generation before verifying roots")
        for root in index.roots:
            if root.root_id == root_id and not root.released and root.domain == "canonical":
                _validate_targets(provider, root.targets)
                return root
    raise ContentRetentionError(f"active retention root is unavailable: {root_id}")


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


def finalize_run_retention(
    provider: ImmutableArtifactBlobProvider,
    root_id: str,
    targets: tuple[ContentRetentionTarget, ...],
    *,
    lease: ContentLeaseRecord | None,
) -> str:
    """Atomically keep a terminal run root or transfer its objects to a release lease."""
    with provider._exclusive_lock():
        index = _read_index(provider)
        root, existing_lease = _match_run_retention(index, root_id, targets, lease)
        if _run_retention_is_final(root, lease, existing_lease):
            return root_id if lease is None else lease.lease_id
        if index.pending_gc is not None:
            raise ContentRetentionError("recover the pending GC generation before finalizing roots")
        if root.released or (existing_lease is not None and existing_lease.released):
            raise ContentRetentionError(f"released run root lacks its terminal lease: {root_id}")
        _validate_targets(provider, targets)
        terminal = root.model_copy(
            update={"nonterminal": False, "restartable": False, "released": lease is not None}
        )
        updated = _replace_record(index, "roots", terminal, "root_id")
        if lease is not None and existing_lease is None:
            updated = _replace_record(updated, "leases", lease, "lease_id")
        _write_index(provider, updated)
        return root_id if lease is None else lease.lease_id


def verify_finalized_run_retention(
    provider: ImmutableArtifactBlobProvider,
    root_id: str,
    targets: tuple[ContentRetentionTarget, ...],
    *,
    lease: ContentLeaseRecord | None,
) -> None:
    """Authenticate an existing terminal run record, even after its lease was collected."""
    with provider._exclusive_lock():
        root, existing_lease = _match_run_retention(_read_index(provider), root_id, targets, lease)
        if not _run_retention_is_final(root, lease, existing_lease):
            raise ContentRetentionError(f"run retention is not finalized: {root_id}")


def require_released_leases(
    provider: ImmutableArtifactBlobProvider,
    leases: Mapping[str, tuple[ContentRetentionTarget, ...]],
    *,
    owner: str,
) -> None:
    """Verify that each named lease previously authorized disposal of its exact objects."""
    with provider._exclusive_lock():
        by_id = {item.lease_id: item for item in _read_index(provider).leases}
        for lease_id, targets in leases.items():
            lease = by_id.get(lease_id)
            if (
                lease is None
                or not lease.released
                or lease.domain != "canonical"
                or lease.owner != owner
                or lease.expiry_disposition != "release"
                or lease.targets != targets
            ):
                raise ContentRetentionError(
                    f"exact released retention lease is unavailable: {lease_id}"
                )


def _match_run_retention(
    index: ContentRetentionIndex,
    root_id: str,
    targets: tuple[ContentRetentionTarget, ...],
    lease: ContentLeaseRecord | None,
) -> tuple[ContentRootRecord, ContentLeaseRecord | None]:
    root = next((item for item in index.roots if item.root_id == root_id), None)
    if (
        root is None
        or root.root_kind != "run"
        or root.domain != "canonical"
        or _target_identities(root.targets) != _target_identities(targets)
    ):
        raise ContentRetentionError(f"exact run retention root is unavailable: {root_id}")
    existing = None
    if lease is not None:
        if lease.released or lease.expiry_disposition != "release" or lease.targets != targets:
            raise ContentRetentionError("terminal run lease must release the exact run targets")
        existing = next((item for item in index.leases if item.lease_id == lease.lease_id), None)
        if existing is not None and (
            existing.owner != lease.owner
            or existing.domain != lease.domain
            or existing.expiry_disposition != lease.expiry_disposition
            or _target_identities(existing.targets) != _target_identities(targets)
        ):
            raise ContentRetentionError(f"terminal run lease identity drifted: {lease.lease_id}")
    return root, existing


def _target_identities(targets: tuple[ContentRetentionTarget, ...]) -> set[bytes]:
    return {canonical_json_v1_bytes(target.model_dump(mode="json")) for target in targets}


def _run_retention_is_final(
    root: ContentRootRecord,
    lease: ContentLeaseRecord | None,
    existing_lease: ContentLeaseRecord | None,
) -> bool:
    if root.nonterminal or root.restartable:
        return False
    if lease is None:
        return not root.released
    return root.released and existing_lease is not None


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


def release_owned_lease(
    provider: ImmutableArtifactBlobProvider, lease_id: str, *, owner: str
) -> ContentLeaseRecord | None:
    """Release only a lease belonging to the given owner, if one exists."""
    return _release_record(provider, "leases", "lease_id", lease_id, owner=owner)


def _release_record(provider, field: str, key: str, record_id: str, *, owner: str | None = None):
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
            if owner is not None:
                return None
            raise ContentRetentionError(f"retention record is unavailable: {record_id}")
        if owner is not None and current.owner != owner:
            raise ContentRetentionError(f"lease has a different owner: {record_id}")
        released = current.model_copy(update={"released": True})
        updated = _replace_record(index, field, released, key)
        if updated != index:
            _write_index(provider, updated)
        return released


def _inventory(provider: ImmutableArtifactBlobProvider) -> tuple[dict[str, int], list[str]]:
    root = provider.root / "artifacts" / "sha256"
    try:
        root_mode = root.lstat().st_mode
    except FileNotFoundError:
        return {}, []
    if not stat.S_ISDIR(root_mode):
        return {}, ["canonical object namespace is not one directory"]
    objects: dict[str, int] = {}
    errors: list[str] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        current = path.lstat()
        if stat.S_ISDIR(current.st_mode):
            if path.name == ".feedbax-materialization-staging" and any(path.iterdir()):
                errors.append(f"incomplete materialization staging directory: {relative}")
            continue
        digest = path.name
        expected = len(relative.parts) == 2 and relative.parts[0] == digest[:2]
        expected = expected and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)
        if not expected or not stat.S_ISREG(current.st_mode) or current.st_nlink != 1:
            errors.append(f"unsafe or unknown object-store entry: {relative}")
            continue
        try:
            provider._verify_blob(path, sha256=digest, size_bytes=current.st_size)
        except (ArtifactBlobCustodyError, OSError) as exc:
            errors.append(f"object identity drift: {digest}: {exc}")
            continue
        objects[digest] = current.st_size
    return objects, errors


def _live_closure(
    provider: ImmutableArtifactBlobProvider,
    index: ContentRetentionIndex,
    *,
    now: datetime,
    release_lease_ids: tuple[str, ...] = (),
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
    release_scope = set(release_lease_ids)
    known_lease_ids = {item.lease_id for item in index.leases}
    for lease_id in sorted(release_scope - known_lease_ids):
        errors.append(f"scoped release lease is unavailable: {lease_id}")
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
            elif not release_scope or record.lease_id in release_scope:
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
            release_lease_ids=request.release_lease_ids,
        )
        errors.extend(live_errors)
        decisions = tuple(
            ContentGcDecision(
                object_id=f"{CONTENT_OBJECT_REF_PREFIX}{digest}",
                sha256=digest,
                size_bytes=size,
                decision=(
                    "live"
                    if digest in live
                    or (request.release_lease_ids and digest not in expired_release)
                    else "candidate"
                ),
                reason=(
                    "reachable from explicit custody"
                    if digest in live
                    else (
                        "time-bounded lease expired"
                        if digest in expired_release
                        else (
                            "outside exact lease-release scope"
                            if request.release_lease_ids
                            else "unreferenced"
                        )
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
        return _recover_locked(provider, marked_index)


def recover_garbage_collection(provider: ImmutableArtifactBlobProvider) -> ContentGcReceipt | None:
    """Resume one marked generation idempotently; return ``None`` when none exists."""
    with provider._exclusive_lock():
        index = _read_index(provider)
        if index.pending_gc is None:
            return None
        return _recover_locked(provider, index)


def _active_release_leases(
    provider: ImmutableArtifactBlobProvider,
    lease_ids: tuple[str, ...],
) -> tuple[ContentLeaseRecord, ...]:
    with provider._exclusive_lock():
        index = _read_index(provider)
        if index.pending_gc is not None:
            _recover_locked(provider, index)
            index = _read_index(provider)
        by_id = {item.lease_id: item for item in index.leases}
        missing = [lease_id for lease_id in lease_ids if lease_id not in by_id]
        if missing:
            raise ContentRetentionError(
                f"automatic release leases are unavailable: {missing!r}"
            )
        active = tuple(by_id[lease_id] for lease_id in lease_ids if not by_id[lease_id].released)
        invalid = [
            lease.lease_id for lease in active if lease.expiry_disposition != "release"
        ]
        if invalid:
            raise ContentRetentionError(
                f"automatic release leases lack release disposition: {invalid!r}"
            )
        return active


def collect_release_leases(
    provider: ImmutableArtifactBlobProvider,
    lease_ids: tuple[str, ...],
    *,
    now: datetime | None = None,
) -> ContentGcReceipt | None:
    """Collect only objects made unreachable by the named expired release leases."""
    canonical_ids = tuple(sorted(set(lease_ids)))
    if canonical_ids != lease_ids or not canonical_ids:
        raise ContentRetentionError("lease IDs must be nonempty, unique, and canonically ordered")
    supplied_now = now or datetime.now(timezone.utc)
    if supplied_now.tzinfo is None or supplied_now.utcoffset() is None:
        raise ContentRetentionError("lease collection time must be timezone-aware")
    observed_at = supplied_now.astimezone(timezone.utc)
    for _attempt in range(8):
        active = _active_release_leases(provider, canonical_ids)
        if not active:
            return None
        future = [lease.lease_id for lease in active if lease.expires_at > observed_at]
        if future:
            raise ContentRetentionError(f"automatic release leases have not expired: {future!r}")
        request = ContentGcRequest(release_lease_ids=canonical_ids)
        report = collect_garbage(provider, request, now=observed_at)
        if report.errors:
            raise ContentRetentionError("; ".join(report.errors))
        result = collect_garbage(
            provider,
            request.model_copy(
                update={"destructive": True, "expected_generation": report.generation}
            ),
            now=observed_at,
        )
        if not result.errors:
            if not _active_release_leases(provider, canonical_ids):
                return result
            continue
        if result.errors != ("retention generation drifted after candidate reporting",):
            raise ContentRetentionError("; ".join(result.errors))
    raise ContentRetentionError("retention generation kept drifting during lease collection")


def run_scheduled_lease_collection(
    root: Path,
    lease_ids: tuple[str, ...],
    *,
    retry_seconds: float = 60.0,
    ready_path: Path | None = None,
    fallback_at: datetime | None = None,
) -> None:
    """Wait for named durable leases and retry exact collection until they are released."""
    from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider

    provider = ImmutableArtifactBlobProvider(root.absolute())
    while True:
        try:
            if not _index_path(provider).exists():
                return
            active = _active_release_leases(provider, lease_ids)
            if not active:
                return
            now = datetime.now(timezone.utc)
            next_expiry = min(lease.expires_at for lease in active)
            if ready_path is not None and not _is_single_regular_file(ready_path):
                if fallback_at is None:
                    raise ContentRetentionError("gated collection requires a fallback deadline")
                next_expiry = max(next_expiry, fallback_at)
            if next_expiry > now:
                poll_seconds = 1.0 if ready_path is not None else retry_seconds
                time.sleep(min(poll_seconds, (next_expiry - now).total_seconds()))
                continue
            due_ids = tuple(
                sorted(lease.lease_id for lease in active if lease.expires_at <= now)
            )
            collect_release_leases(provider, due_ids, now=now)
        except Exception as exc:
            print(f"automatic lease collection retry: {exc}", file=sys.stderr, flush=True)
            time.sleep(retry_seconds)


def launch_scheduled_lease_collection(
    root: Path,
    lease_ids: tuple[str, ...],
    *,
    ready_path: Path | None = None,
    fallback_at: datetime | None = None,
) -> int:
    """Launch one detached waiter for exact durable release leases."""
    canonical_ids = tuple(sorted(set(lease_ids)))
    if canonical_ids != lease_ids or not canonical_ids:
        raise ContentRetentionError("lease IDs must be nonempty, unique, and canonically ordered")
    provider = __import__(
        "feedbax.persistence.artifact_custody", fromlist=["ImmutableArtifactBlobProvider"]
    ).ImmutableArtifactBlobProvider(root.absolute())
    _active_release_leases(provider, canonical_ids)
    if (ready_path is None) != (fallback_at is None):
        raise ContentRetentionError("ready_path and fallback_at must be supplied together")
    if fallback_at is not None and (
        fallback_at.tzinfo is None or fallback_at.utcoffset() is None
    ):
        raise ContentRetentionError("automatic collection fallback must be timezone-aware")
    job_payload = {
        "root": str(provider.root),
        "lease_ids": list(canonical_ids),
        "ready_path": str(ready_path.absolute()) if ready_path is not None else None,
        "fallback_at": (
            fallback_at.astimezone(timezone.utc).isoformat()
            if fallback_at is not None
            else None
        ),
    }
    job_bytes = canonical_json_v1_bytes(job_payload)
    job_path = _automatic_job_path(provider, hashlib.sha256(job_bytes).hexdigest())
    if _is_single_regular_file(job_path):
        if job_path.read_bytes() != job_bytes:
            raise ContentRetentionError("automatic collection job identity drifted")
    elif job_path.exists():
        raise ContentRetentionError("automatic collection job path is unsafe")
    else:
        _write_new_or_replace(provider, job_path, job_bytes)
    log_path = provider.root / "retention" / "automatic-collection.log"
    _ensure_store_directory(provider, log_path.parent)
    if log_path.exists() and not _is_single_regular_file(log_path):
        raise ContentRetentionError("automatic collection log path is unsafe")
    log_descriptor = os.open(
        log_path,
        os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW,
        0o600,
    )
    log_metadata = os.fstat(log_descriptor)
    if not stat.S_ISREG(log_metadata.st_mode) or log_metadata.st_nlink != 1:
        os.close(log_descriptor)
        raise ContentRetentionError("automatic collection log identity is unsafe")
    with os.fdopen(log_descriptor, "ab", buffering=0) as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "feedbax.persistence.content_retention",
                "--automatic-release-job",
                str(job_path),
            ],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=True,
            close_fds=True,
        )
    return process.pid


def _verify_quarantine(
    provider: ImmutableArtifactBlobProvider,
    pending: ContentPendingGc,
    quarantine: Path,
) -> list[Path]:
    remaining = []
    for candidate in pending.decisions:
        path = quarantine / candidate.sha256
        if _entry_exists(path):
            provider._verify_blob(path, sha256=candidate.sha256, size_bytes=candidate.size_bytes)
            remaining.append(path)
        elif pending.stage == "marked":
            raise ContentRetentionError(f"marked candidate is missing: {candidate.sha256}")
    return remaining


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
    quarantine = _quarantine_directory(provider, pending)
    marked = pending.stage == "marked"
    try:
        if pending.decisions:
            _ensure_store_directory(provider, quarantine)
        if marked:
            _write_receipt(
                provider,
                _receipt(
                    receipt_id=pending.receipt_id,
                    request=request,
                    generation=pending.generation,
                    stage="marked",
                    roots=pending.root_ids,
                    leases=pending.lease_ids,
                    holds=pending.hold_ids,
                    decisions=pending.decisions,
                ),
            )
            source_directories: set[Path] = set()
            for candidate in pending.decisions:
                source = provider._canonical_path(candidate.sha256)
                destination = quarantine / candidate.sha256
                source_directories.add(source.parent)
                if _entry_exists(source):
                    if _entry_exists(destination):
                        raise ContentRetentionError(
                            f"candidate exists in canonical and quarantine paths: {candidate.sha256}"
                        )
                    provider._verify_blob(
                        source, sha256=candidate.sha256, size_bytes=candidate.size_bytes
                    )
                    os.replace(source, destination)
            for directory in sorted(source_directories):
                _sync_directory(directory)
            if pending.decisions:
                _sync_directory(quarantine)
            _verify_quarantine(provider, pending, quarantine)
            pending = pending.model_copy(
                update={
                    "stage": "quarantined",
                    "decisions": tuple(
                        candidate.model_copy(
                            update={
                                "decision": "quarantined",
                                "reason": "generation-guarded quarantine",
                            }
                        )
                        for candidate in pending.decisions
                    ),
                }
            )
            index = index.model_copy(update={"pending_gc": pending})
            _write_index(provider, index)
        _write_receipt(
            provider,
            _receipt(
                receipt_id=pending.receipt_id,
                request=request,
                generation=pending.generation,
                stage="quarantined",
                roots=pending.root_ids,
                leases=pending.lease_ids,
                holds=pending.hold_ids,
                decisions=pending.decisions,
            ),
        )
        # After the quarantine commit, a canonical name may contain a later publication.
        for path in _verify_quarantine(provider, pending, quarantine):
            path.unlink()
        if pending.decisions:
            _sync_directory(quarantine)

        receipt = _receipt(
            receipt_id=pending.receipt_id,
            request=request,
            generation=pending.generation,
            stage="swept",
            roots=pending.root_ids,
            leases=pending.lease_ids,
            holds=pending.hold_ids,
            decisions=tuple(
                candidate.model_copy(
                    update={"decision": "deleted", "reason": "quarantine identity verified"}
                )
                for candidate in pending.decisions
            ),
        )
        _write_receipt(provider, receipt)
        expired_lease_ids = set(pending.expired_lease_ids)
        _write_index(
            provider,
            index.model_copy(
                update={
                    "leases": tuple(
                        lease.model_copy(update={"released": True})
                        if lease.lease_id in expired_lease_ids
                        else lease
                        for lease in index.leases
                    ),
                    "pending_gc": None,
                }
            ),
        )
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
            decisions=pending.decisions,
            errors=(str(exc),),
        )


__all__ = [
    "ContentRetentionError",
    "acquire_lease",
    "collect_release_leases",
    "collect_garbage",
    "launch_scheduled_lease_collection",
    "place_hold",
    "recover_garbage_collection",
    "register_root",
    "release_hold",
    "release_lease",
    "release_owned_lease",
    "release_root",
    "renew_lease",
    "run_scheduled_lease_collection",
    "schedule_lease_release",
    "set_run_state",
]


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--automatic-release-job", type=Path, required=True)
    arguments = parser.parse_args()
    job_path = arguments.automatic_release_job.absolute()
    raw = job_path.read_bytes()
    payload = strict_json_loads(raw.decode("utf-8"), ref=str(job_path))
    if raw != canonical_json_v1_bytes(payload):
        raise ContentRetentionError("automatic collection job is not canonical")
    if not isinstance(payload, dict):
        raise ContentRetentionError("automatic collection job must be an object")
    if job_path.stem != hashlib.sha256(raw).hexdigest():
        raise ContentRetentionError("automatic collection job filename does not match its bytes")
    root_value = payload.get("root")
    if not isinstance(root_value, str) or not Path(root_value).is_absolute():
        raise ContentRetentionError("automatic collection job root is invalid")
    lease_ids = payload.get("lease_ids")
    if not isinstance(lease_ids, list) or not all(isinstance(item, str) for item in lease_ids):
        raise ContentRetentionError("automatic collection job lease IDs are invalid")
    if tuple(lease_ids) != tuple(sorted(set(lease_ids))):
        raise ContentRetentionError("automatic collection job lease IDs are not canonical")
    ready_value = payload.get("ready_path")
    fallback_value = payload.get("fallback_at")
    ready_path = Path(ready_value) if isinstance(ready_value, str) else None
    fallback_at = (
        datetime.fromisoformat(fallback_value) if isinstance(fallback_value, str) else None
    )
    run_scheduled_lease_collection(
        Path(root_value),
        tuple(lease_ids),
        ready_path=ready_path,
        fallback_at=fallback_at,
    )
    try:
        job_path.unlink()
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    _main()
