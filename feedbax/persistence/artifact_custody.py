"""Immutable, content-addressed custody for exact artifact bytes."""

from __future__ import annotations

import errno
import fcntl
import hashlib
import os
import stat
import threading
import unicodedata
import uuid
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Literal

from pydantic import ValidationError as PydanticValidationError

from feedbax._secure_fs import (
    SecurePathRigor,
    canonicalize_trusted_system_aliases,
    close_directory_chain,
    directory_open_flags,
    file_open_flags,
    open_directory_chain,
    open_existing_directory,
    open_existing_file,
    rename_no_replace,
    recheck_directory_chain,
    require_secure_path_capabilities,
)
from feedbax.contracts.artifact_custody import (
    IMMUTABLE_ARTIFACT_BLOB_PROVIDER_KIND,
    IMMUTABLE_ARTIFACT_BLOB_PROVIDER_SCHEMA_ID,
    IMMUTABLE_ARTIFACT_BLOB_PROVIDER_SCHEMA_VERSION,
    IMMUTABLE_ARTIFACT_BLOB_STORAGE_BACKEND,
    ArtifactBlobContainmentError,
    ArtifactBlobCustodyError,
    ArtifactBlobIntegrityError,
    ArtifactBlobReferenceError,
    CONTENT_MANIFEST_REF_PREFIX,
    CONTENT_OBJECT_REF_PREFIX,
    CONTENT_TREE_REF_PREFIX,
    ContentManifestConflictError,
    ContentManifestRef,
    ContentGcReceipt,
    ContentGcRequest,
    ContentHoldRecord,
    ContentLeaseRecord,
    ContentObjectRef,
    ContentRetentionTarget,
    ContentRootRecord,
    ContentTree,
    ContentTreeFile,
    ContentTreeRef,
    ImmutableArtifactBlobProviderSpec,
)
from feedbax.contracts.base import ArtifactRef
from feedbax.contracts.canonical_json import canonical_json_v1_bytes
from feedbax.contracts.strict_json import strict_json_loads


_ARTIFACT_ID_PREFIX = "artifact://sha256/"
_DIGEST_LENGTH = 64
_STORAGE_BACKEND = IMMUTABLE_ARTIFACT_BLOB_STORAGE_BACKEND
_READ_CHUNK_SIZE = 1024 * 1024
_RESERVED_METADATA_KEYS = frozenset(
    {
        "artifact_id",
        "original_uri",
        "path",
        "relative_path",
        "sha256",
        "size_bytes",
        "source_path",
        "storage_backend",
        "uri",
    }
)
_canonicalize_trusted_system_aliases = canonicalize_trusted_system_aliases
_store_lock_state = threading.local()


def _parse_content_object_id(object_id: object) -> str:
    if not isinstance(object_id, str) or not object_id.startswith(CONTENT_OBJECT_REF_PREFIX):
        raise ArtifactBlobReferenceError(
            "content object id must use content-object://sha256/<digest> syntax"
        )
    return _validate_digest(object_id.removeprefix(CONTENT_OBJECT_REF_PREFIX))


def _manifest_reference_key(kind: str, manifest_id: str) -> str:
    for field, value in (("kind", kind), ("manifest_id", manifest_id)):
        if not isinstance(value, str) or not value or "\x00" in value:
            raise ArtifactBlobReferenceError(
                f"manifest {field} must be a non-empty NUL-free string"
            )
    return hashlib.sha256(f"{kind}\0{manifest_id}".encode("utf-8")).hexdigest()


@contextmanager
def _exclusive_store_lock(root: Path) -> Iterator[None]:
    lock_key = str(root)
    held = getattr(_store_lock_state, "held", {})
    if lock_key in held:
        held[lock_key] += 1
        try:
            yield
        finally:
            held[lock_key] -= 1
        return
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".content-store.lock"
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        lock_stat = os.fstat(descriptor)
        path_stat = os.stat(lock_path, follow_symlinks=False)
        if (
            not stat.S_ISREG(lock_stat.st_mode)
            or lock_stat.st_nlink != 1
            or (lock_stat.st_dev, lock_stat.st_ino) != (path_stat.st_dev, path_stat.st_ino)
        ):
            raise ArtifactBlobContainmentError("content-store lock must be one stable regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        held[lock_key] = 1
        _store_lock_state.held = held
        try:
            yield
        finally:
            del held[lock_key]
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _parse_artifact_id(artifact_id: object) -> str:
    if not isinstance(artifact_id, str):
        raise ArtifactBlobReferenceError("artifact_id must be a string")
    if not artifact_id.startswith(_ARTIFACT_ID_PREFIX):
        raise ArtifactBlobReferenceError(
            "artifact_id must use canonical artifact://sha256/<digest> syntax"
        )
    digest = artifact_id.removeprefix(_ARTIFACT_ID_PREFIX)
    if len(digest) != _DIGEST_LENGTH or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ArtifactBlobReferenceError(
            "artifact_id must contain exactly 64 lowercase hexadecimal characters"
        )
    return digest


def _validate_digest(digest: object) -> str:
    if not isinstance(digest, str):
        raise ArtifactBlobReferenceError("artifact sha256 must be a string")
    if len(digest) != _DIGEST_LENGTH or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ArtifactBlobReferenceError(
            "artifact sha256 must contain exactly 64 lowercase hexadecimal characters"
        )
    return digest


def _validate_size(size_bytes: object, *, field: str) -> int:
    if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0:
        raise ArtifactBlobReferenceError(f"{field} must be a non-negative integer")
    return size_bytes


def _validate_explicit_root(root: object) -> Path:
    if not isinstance(root, (Path, str)):
        raise ArtifactBlobReferenceError("explicit_root must be an absolute path")
    raw_root = os.fspath(root)
    if not raw_root or "\0" in raw_root:
        raise ArtifactBlobReferenceError("explicit_root must be non-empty and NUL-free")
    root_path = Path(raw_root)
    if not root_path.is_absolute():
        raise ArtifactBlobReferenceError(
            "explicit_root must be absolute; cwd, user, and environment expansion are forbidden"
        )
    if ".." in root_path.parts:
        raise ArtifactBlobReferenceError("explicit_root must not contain lexical '..' components")
    return root_path


def _validate_explicit_destination(destination: object) -> Path:
    if not isinstance(destination, (Path, str)):
        raise ArtifactBlobReferenceError("materialization destination must be an absolute path")
    raw_destination = os.fspath(destination)
    if not raw_destination or "\0" in raw_destination:
        raise ArtifactBlobReferenceError(
            "materialization destination must be non-empty and NUL-free"
        )
    destination_path = Path(raw_destination)
    if not destination_path.is_absolute():
        raise ArtifactBlobReferenceError(
            "materialization destination must be absolute; cwd, user, and environment "
            "expansion are forbidden"
        )
    if ".." in destination_path.parts:
        raise ArtifactBlobReferenceError(
            "materialization destination must not contain lexical '..' components"
        )
    return destination_path


def _file_state(file_stat: os.stat_result) -> tuple[int, ...]:
    return (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_mode,
        file_stat.st_nlink,
        file_stat.st_size,
        file_stat.st_mtime_ns,
        file_stat.st_ctime_ns,
    )


_directory_flags = partial(
    directory_open_flags,
    error_factory=ArtifactBlobContainmentError,
)
_file_flags = partial(
    file_open_flags,
    rigor=SecurePathRigor.SINGLE_LINK_FILE_IDENTITY,
    error_factory=ArtifactBlobContainmentError,
)
_open_directory_chain = partial(
    open_directory_chain,
    error_factory=ArtifactBlobContainmentError,
    context="immutable blob custody directory",
)
_recheck_directory_chain = partial(
    recheck_directory_chain,
    error_factory=ArtifactBlobContainmentError,
    context="immutable blob custody directory",
)
_close_directory_chain = close_directory_chain


def _write_file_descriptor(file_descriptor: int, data: bytes) -> None:
    remaining = memoryview(data)
    while remaining:
        written = os.write(file_descriptor, remaining)
        if written <= 0:
            raise ArtifactBlobIntegrityError("materialization write made no progress")
        remaining = remaining[written:]


def _read_file_descriptor(file_descriptor: int) -> bytes:
    os.lseek(file_descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while chunk := os.read(file_descriptor, _READ_CHUNK_SIZE):
        chunks.append(chunk)
    return b"".join(chunks)


def _stream_file_descriptor(file_descriptor: int, *, target: int | None = None) -> tuple[str, int]:
    os.lseek(file_descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    size_bytes = 0
    while chunk := os.read(file_descriptor, _READ_CHUNK_SIZE):
        digest.update(chunk)
        size_bytes += len(chunk)
        if target is not None:
            _write_file_descriptor(target, chunk)
    return digest.hexdigest(), size_bytes


def _publish_materialized_file_no_replace(
    temporary_name: str,
    destination_name: str,
    *,
    temporary_parent_descriptor: int,
    parent_descriptor: int,
) -> None:
    rename_no_replace(
        temporary_name,
        destination_name,
        source_dir_fd=temporary_parent_descriptor,
        destination_dir_fd=parent_descriptor,
        error_factory=ArtifactBlobIntegrityError,
        context="immutable artifact publication",
    )


def _open_materialization_staging_container(
    *,
    parent_descriptor: int,
) -> int:
    """Open the fixed descriptor-pinned private staging container."""
    directory_name = ".feedbax-materialization-staging"
    try:
        os.mkdir(directory_name, mode=0o700, dir_fd=parent_descriptor)
    except FileExistsError:
        pass
    directory_descriptor: int | None = None
    try:
        directory_descriptor = open_existing_directory(
            directory_name,
            error_factory=ArtifactBlobContainmentError,
            context="materialization staging container",
            dir_fd=parent_descriptor,
        )
        descriptor_stat = os.fstat(directory_descriptor)
        path_stat = os.stat(
            directory_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(descriptor_stat.st_mode)
            or descriptor_stat.st_uid != os.geteuid()
            or stat.S_IMODE(descriptor_stat.st_mode) & 0o077
            or (descriptor_stat.st_dev, descriptor_stat.st_ino)
            != (path_stat.st_dev, path_stat.st_ino)
        ):
            raise ArtifactBlobContainmentError(
                "materialization staging container must be owned, mode-0700, and stable"
            )
        return directory_descriptor
    except Exception:
        if directory_descriptor is not None:
            os.close(directory_descriptor)
        raise


def _remove_private_materialization_staging_name(
    *,
    directory_descriptor: int,
    temporary_name: str,
) -> None:
    """Remove one unguessable name from the pinned private container."""
    try:
        os.unlink(temporary_name, dir_fd=directory_descriptor)
    except FileNotFoundError:
        pass
    os.close(directory_descriptor)


def _stage_blob(
    destination: Path,
    expected_identity: tuple[str, int],
    write_payload: Callable[[int], None],
) -> os.stat_result:
    records = _open_directory_chain(destination.parent, create=True)
    parent_descriptor = records[-1][1]
    temporary_name = f"payload-{uuid.uuid4().hex}"
    staging_descriptor: int | None = None
    temporary_descriptor: int | None = None
    final_descriptor: int | None = None
    try:
        staging_descriptor = _open_materialization_staging_container(
            parent_descriptor=parent_descriptor
        )
        temporary_descriptor = os.open(
            temporary_name,
            _file_flags(writable=True) | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=staging_descriptor,
        )
        write_payload(temporary_descriptor)
        os.fsync(temporary_descriptor)
        if _stream_file_descriptor(temporary_descriptor) != expected_identity:
            raise ArtifactBlobIntegrityError(f"staged blob failed verification: {destination}")
        try:
            _publish_materialized_file_no_replace(
                temporary_name,
                destination.name,
                temporary_parent_descriptor=staging_descriptor,
                parent_descriptor=parent_descriptor,
            )
        except FileExistsError:
            pass
        _remove_private_materialization_staging_name(
            directory_descriptor=staging_descriptor,
            temporary_name=temporary_name,
        )
        staging_descriptor = None
        final_descriptor, before = open_existing_file(
            destination.name,
            rigor=SecurePathRigor.SINGLE_LINK_FILE_IDENTITY,
            error_factory=ArtifactBlobIntegrityError,
            context=f"canonical blob {destination}",
            dir_fd=parent_descriptor,
        )
        stored_identity = _stream_file_descriptor(final_descriptor)
        after = os.fstat(final_descriptor)
        if _file_state(before) != _file_state(after) or stored_identity != expected_identity:
            raise ArtifactBlobIntegrityError(
                f"canonical blob bytes do not match content identity: {destination}"
            )
        path_stat = os.stat(destination.name, dir_fd=parent_descriptor, follow_symlinks=False)
        if (path_stat.st_dev, path_stat.st_ino) != (after.st_dev, after.st_ino):
            raise ArtifactBlobIntegrityError(
                f"canonical blob identity changed during publication: {destination}"
            )
        _recheck_directory_chain(records)
        return after
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise ArtifactBlobContainmentError(
                f"canonical blob path traverses a symlink or non-directory: {destination}"
            ) from exc
        raise
    finally:
        if temporary_descriptor is not None:
            os.close(temporary_descriptor)
        if final_descriptor is not None:
            os.close(final_descriptor)
        if staging_descriptor is not None:
            try:
                _remove_private_materialization_staging_name(
                    directory_descriptor=staging_descriptor,
                    temporary_name=temporary_name,
                )
            except OSError:
                os.close(staging_descriptor)
        _close_directory_chain(records)


def _stage_blob_bytes(data: bytes, destination: Path) -> os.stat_result:
    identity = hashlib.sha256(data).hexdigest(), len(data)
    return _stage_blob(
        destination, identity, lambda descriptor: _write_file_descriptor(descriptor, data)
    )


def _stage_blob_file(
    source_descriptor: int,
    destination: Path,
    expected_identity: tuple[str, int],
) -> os.stat_result:
    def write_payload(descriptor: int) -> None:
        if _stream_file_descriptor(source_descriptor, target=descriptor) != expected_identity:
            raise ArtifactBlobIntegrityError(f"artifact source changed during store: {destination}")

    return _stage_blob(destination, expected_identity, write_payload)


@dataclass(frozen=True, slots=True)
class ImmutableArtifactBlobProvider:
    """Explicit local provider for suffixless immutable artifact blobs."""

    root: Path | str
    storage_backend: str = _STORAGE_BACKEND

    def __post_init__(self) -> None:
        root_path = _validate_explicit_root(self.root)
        try:
            if stat.S_ISLNK(root_path.lstat().st_mode):
                raise ArtifactBlobContainmentError(
                    "immutable blob provider root must not be a symlink"
                )
        except FileNotFoundError:
            pass
        if self.storage_backend != _STORAGE_BACKEND:
            raise ArtifactBlobReferenceError(
                f"unsupported immutable blob storage backend: {self.storage_backend!r}"
            )
        require_secure_path_capabilities(
            SecurePathRigor.SINGLE_LINK_FILE_IDENTITY,
            error_factory=ArtifactBlobContainmentError,
            extra_dir_fd_operations=(os.mkdir, os.link, os.unlink),
            require_link_no_follow=True,
        )
        object.__setattr__(self, "root", _canonicalize_trusted_system_aliases(root_path))

    def _exclusive_lock(self) -> AbstractContextManager[None]:
        """Return the one store lock shared by references and retention state."""
        return _exclusive_store_lock(self.root)

    def register_root(self, root: ContentRootRecord) -> ContentRootRecord:
        """Register one explicit canonical retention root."""
        from feedbax.persistence.content_retention import register_root

        return register_root(self, root)

    def set_run_state(
        self,
        root_id: str,
        *,
        nonterminal: bool,
        restartable: bool,
    ) -> ContentRootRecord:
        """Record whether a run root remains nonterminal or restartable."""
        from feedbax.persistence.content_retention import set_run_state

        return set_run_state(
            self,
            root_id,
            nonterminal=nonterminal,
            restartable=restartable,
        )

    def release_root(self, root_id: str) -> ContentRootRecord:
        """Release one terminal non-restartable root."""
        from feedbax.persistence.content_retention import release_root

        return release_root(self, root_id)

    def place_hold(self, hold: ContentHoldRecord) -> ContentHoldRecord:
        """Place one explicit scientific or owner hold."""
        from feedbax.persistence.content_retention import place_hold

        return place_hold(self, hold)

    def release_hold(self, hold_id: str) -> ContentHoldRecord:
        """Release one explicit hold."""
        from feedbax.persistence.content_retention import release_hold

        return release_hold(self, hold_id)

    def acquire_lease(self, lease: ContentLeaseRecord) -> ContentLeaseRecord:
        """Acquire one exact lease under the store transaction lock."""
        from feedbax.persistence.content_retention import acquire_lease

        return acquire_lease(self, lease)

    def renew_lease(self, lease_id: str, *, expires_at: datetime) -> ContentLeaseRecord:
        """Renew one exact active lease using a caller-selected expiry."""
        from feedbax.persistence.content_retention import renew_lease

        return renew_lease(self, lease_id, expires_at=expires_at)

    def release_lease(self, lease_id: str) -> ContentLeaseRecord:
        """Durably release one lease; expiry is never implicit release."""
        from feedbax.persistence.content_retention import release_lease

        return release_lease(self, lease_id)

    def schedule_lease_release(
        self,
        lease_id: str,
        *,
        expires_at: datetime,
    ) -> ContentLeaseRecord:
        """Convert one in-flight lease to bounded automatic-release semantics."""
        from feedbax.persistence.content_retention import schedule_lease_release

        return schedule_lease_release(self, lease_id, expires_at=expires_at)

    def collect_garbage(
        self,
        request: ContentGcRequest | None = None,
        *,
        now: datetime | None = None,
    ) -> ContentGcReceipt:
        """Report candidates by default or execute one guarded manual sweep."""
        from feedbax.persistence.content_retention import collect_garbage

        return collect_garbage(self, request, now=now)

    def recover_garbage_collection(self) -> ContentGcReceipt | None:
        """Resume one interrupted marked generation idempotently."""
        from feedbax.persistence.content_retention import recover_garbage_collection

        return recover_garbage_collection(self)

    def store_object(self, data: bytes) -> ContentObjectRef:
        """Store one role-free exact byte object in the canonical CAS."""
        if not isinstance(data, bytes):
            raise TypeError("content object data must be bytes")
        digest = hashlib.sha256(data).hexdigest()
        destination = self._canonical_path(digest)
        self._validate_store_target(destination)
        with self._exclusive_lock():
            self._validate_store_target(destination)
            written = _stage_blob_bytes(data, destination)
            ref = ContentObjectRef(
                object_id=f"{CONTENT_OBJECT_REF_PREFIX}{digest}",
                sha256=digest,
                size_bytes=written.st_size,
            )
            if self.get_object(ref) != data:
                raise ArtifactBlobIntegrityError(
                    f"pre-existing content object differs from stored payload: {ref.object_id}"
                )
            return ref

    def _store_file(
        self,
        source_path: Path | str,
        *,
        sha256: str | None = None,
        size_bytes: int | None = None,
    ) -> ContentObjectRef:
        """Store exact file bytes through bounded chunks, checking any declared identity."""
        source = Path(source_path).absolute()
        source_records = _open_directory_chain(source.parent, create=False)
        source_descriptor: int | None = None
        try:
            source_parent = source_records[-1][1]
            source_descriptor, before = open_existing_file(
                source.name,
                rigor=SecurePathRigor.SINGLE_LINK_FILE_IDENTITY,
                error_factory=ArtifactBlobContainmentError,
                context=f"artifact source {source}",
                dir_fd=source_parent,
            )
            path_stat = os.stat(source.name, dir_fd=source_parent, follow_symlinks=False)
            if (path_stat.st_dev, path_stat.st_ino) != (before.st_dev, before.st_ino):
                raise ArtifactBlobIntegrityError(f"artifact source identity changed: {source}")
            identity = _stream_file_descriptor(source_descriptor)
            if sha256 is not None and identity[0] != _validate_digest(sha256):
                raise ArtifactBlobIntegrityError(f"artifact source SHA-256 mismatch: {source}")
            if size_bytes is not None and identity[1] != _validate_size(
                size_bytes, field="size_bytes"
            ):
                raise ArtifactBlobIntegrityError(f"artifact source size mismatch: {source}")
            if _file_state(before) != _file_state(os.fstat(source_descriptor)):
                raise ArtifactBlobIntegrityError(f"artifact source changed during hash: {source}")
            destination = self._canonical_path(identity[0])
            self._validate_store_target(destination)
            with self._exclusive_lock():
                self._validate_store_target(destination)
                if destination.exists():
                    destination_records = _open_directory_chain(destination.parent, create=False)
                    destination_descriptor: int | None = None
                    try:
                        destination_descriptor, written = open_existing_file(
                            destination.name,
                            rigor=SecurePathRigor.SINGLE_LINK_FILE_IDENTITY,
                            error_factory=ArtifactBlobIntegrityError,
                            context=f"canonical blob {destination}",
                            dir_fd=destination_records[-1][1],
                        )
                        if _stream_file_descriptor(destination_descriptor) != identity or (
                            _file_state(written) != _file_state(os.fstat(destination_descriptor))
                        ):
                            raise ArtifactBlobIntegrityError(
                                f"canonical blob bytes do not match content identity: {destination}"
                            )
                        _recheck_directory_chain(destination_records)
                    finally:
                        if destination_descriptor is not None:
                            os.close(destination_descriptor)
                        _close_directory_chain(destination_records)
                else:
                    written = _stage_blob_file(source_descriptor, destination, identity)
            if _file_state(before) != _file_state(os.fstat(source_descriptor)):
                raise ArtifactBlobIntegrityError(f"artifact source changed during store: {source}")
            path_after = os.stat(source.name, dir_fd=source_parent, follow_symlinks=False)
            if (path_after.st_dev, path_after.st_ino) != (before.st_dev, before.st_ino):
                raise ArtifactBlobIntegrityError(f"artifact source identity changed: {source}")
            _recheck_directory_chain(source_records)
            return ContentObjectRef(
                object_id=f"{CONTENT_OBJECT_REF_PREFIX}{identity[0]}",
                sha256=identity[0],
                size_bytes=written.st_size,
            )
        finally:
            if source_descriptor is not None:
                os.close(source_descriptor)
            _close_directory_chain(source_records)

    def get_object(
        self,
        object_ref: ContentObjectRef | str,
        *,
        size_bytes: int | None = None,
    ) -> bytes:
        """Read one pinned role-free object and verify its exact identity."""
        if isinstance(object_ref, str):
            digest = _parse_content_object_id(object_ref)
            if size_bytes is None:
                raise ArtifactBlobReferenceError(
                    "size_bytes is required when reading a raw content object id"
                )
            expected_size = _validate_size(size_bytes, field="size_bytes")
        elif isinstance(object_ref, ContentObjectRef):
            digest = _parse_content_object_id(object_ref.object_id)
            if object_ref.sha256 != digest:
                raise ArtifactBlobReferenceError(
                    "content object id digest does not match ContentObjectRef.sha256"
                )
            expected_size = object_ref.size_bytes
            if size_bytes is not None and size_bytes != expected_size:
                raise ArtifactBlobReferenceError(
                    "size_bytes override does not match ContentObjectRef.size_bytes"
                )
        else:
            raise ArtifactBlobReferenceError(
                "object_ref must be a ContentObjectRef or content object id"
            )
        data = self._read_canonical_blob(self._canonical_path(digest), expected_size)
        if hashlib.sha256(data).hexdigest() != digest:
            raise ArtifactBlobIntegrityError(f"content object SHA-256 mismatch: expected {digest}")
        return data

    def ingest_tree(self, source_root: Path | str) -> ContentTreeRef:
        """Ingest a canonical regular-file tree with identity independent of its role."""
        with self._exclusive_lock():
            return self._ingest_tree_locked(source_root)

    def _ingest_tree_locked(self, source_root: Path | str) -> ContentTreeRef:
        """Ingest a tree while publication and GC share one store transaction."""
        source = _canonicalize_trusted_system_aliases(_validate_explicit_root(source_root))
        try:
            source_stat = source.stat(follow_symlinks=False)
            resolved = source.resolve(strict=True)
        except OSError as exc:
            raise ArtifactBlobContainmentError("content-tree source root is unavailable") from exc
        if resolved != source or not stat.S_ISDIR(source_stat.st_mode):
            raise ArtifactBlobContainmentError(
                "content-tree source root must be a canonical non-symlink directory"
            )
        directories: list[str] = []
        files: list[ContentTreeFile] = []
        root_descriptor = open_existing_directory(
            source,
            error_factory=ArtifactBlobContainmentError,
            context="content-tree source root",
        )
        try:
            self._ingest_tree_directory(root_descriptor, (), directories, files)
            after = os.stat(source, follow_symlinks=False)
            opened_after = os.fstat(root_descriptor)
            if _file_state(source_stat) != _file_state(after) or _file_state(
                source_stat
            ) != _file_state(opened_after):
                raise ArtifactBlobIntegrityError("content-tree source changed during ingest")
        finally:
            os.close(root_descriptor)
        tree = ContentTree(
            directories=tuple(sorted(directories)),
            files=tuple(sorted(files, key=lambda entry: entry.relative_path)),
        )
        descriptor_bytes = canonical_json_v1_bytes(tree.model_dump(mode="json"))
        descriptor = self.store_object(descriptor_bytes)
        return ContentTreeRef(
            tree_id=f"{CONTENT_TREE_REF_PREFIX}{descriptor.sha256}",
            descriptor=descriptor,
        )

    def publish_rooted_tree(
        self,
        members: Mapping[str, Path | ContentObjectRef],
        *,
        root_id: str,
        root_kind: Literal["logical-ref", "run", "export", "import", "migration"],
        nonterminal: bool = False,
        restartable: bool = False,
    ) -> tuple[ContentTreeRef, ContentRootRecord]:
        """Store or reuse exact members, then publish and root their complete tree."""
        with self._exclusive_lock():
            files: list[ContentTreeFile] = []
            directories: set[str] = set()
            for relative_path, source in sorted(members.items()):
                if isinstance(source, ContentObjectRef):
                    self.get_object(source)
                    object_ref = source
                elif isinstance(source, Path):
                    object_ref = self.store_object(self._read_regular_source(source))
                else:
                    raise TypeError(
                        "rooted-tree members must be pathlib.Path or ContentObjectRef values"
                    )
                canonical_path = ContentTreeFile(
                    relative_path=relative_path,
                    object=object_ref,
                ).relative_path
                parent = PurePosixPath(canonical_path).parent
                while parent != PurePosixPath("."):
                    directories.add(parent.as_posix())
                    parent = parent.parent
                files.append(ContentTreeFile(relative_path=canonical_path, object=object_ref))

            tree = ContentTree(
                directories=tuple(sorted(directories)),
                files=tuple(files),
            )
            descriptor = self.store_object(canonical_json_v1_bytes(tree.model_dump(mode="json")))
            tree_ref = ContentTreeRef(
                tree_id=f"{CONTENT_TREE_REF_PREFIX}{descriptor.sha256}",
                descriptor=descriptor,
            )
            root = ContentRootRecord(
                root_id=root_id,
                root_kind=root_kind,
                targets=(ContentRetentionTarget(ref_kind="tree", tree=tree_ref),),
                nonterminal=nonterminal,
                restartable=restartable,
            )
            self.register_root(root)
            return tree_ref, root

    @staticmethod
    def _read_regular_source(source: Path) -> bytes:
        path = _canonicalize_trusted_system_aliases(_validate_explicit_root(source))
        try:
            before = path.stat(follow_symlinks=False)
            if path.resolve(strict=True) != path:
                raise ArtifactBlobContainmentError(
                    f"rooted-tree source must be canonical and non-symlink: {source}"
                )
            descriptor, opened = open_existing_file(
                path,
                rigor=SecurePathRigor.SINGLE_LINK_FILE_IDENTITY,
                error_factory=ArtifactBlobContainmentError,
                context=f"rooted-tree source {source}",
            )
            try:
                data = _read_file_descriptor(descriptor)
                after = os.fstat(descriptor)
            finally:
                os.close(descriptor)
            current = path.stat(follow_symlinks=False)
        except OSError as exc:
            raise ArtifactBlobContainmentError(
                f"rooted-tree source is not one stable regular file: {source}"
            ) from exc
        if (
            _file_state(before) != _file_state(opened)
            or _file_state(opened) != _file_state(after)
            or _file_state(before) != _file_state(current)
        ):
            raise ArtifactBlobIntegrityError(f"rooted-tree source changed during read: {source}")
        return data

    def get_tree(self, tree_ref: ContentTreeRef) -> ContentTree:
        """Resolve and revalidate one exact role-free content-tree descriptor."""
        if not isinstance(tree_ref, ContentTreeRef):
            raise ArtifactBlobReferenceError("tree_ref must be a ContentTreeRef")
        raw = self.get_object(tree_ref.descriptor)
        try:
            payload = strict_json_loads(raw.decode("utf-8"), ref=tree_ref.tree_id)
            tree = ContentTree.model_validate(payload)
        except Exception as exc:
            raise ArtifactBlobIntegrityError(
                f"content-tree descriptor is invalid: {tree_ref.tree_id}: {exc}"
            ) from exc
        expected = canonical_json_v1_bytes(tree.model_dump(mode="json"))
        if raw != expected:
            raise ArtifactBlobIntegrityError(
                f"content-tree descriptor is not canonical: {tree_ref.tree_id}"
            )
        return tree

    def bind_manifest(
        self,
        kind: str,
        manifest_id: str,
        raw_bytes: bytes,
    ) -> ContentManifestRef:
        """Pin one logical manifest identity to one exact immutable byte object."""
        return self.bind_manifests(((kind, manifest_id, raw_bytes),))[0]

    def bind_manifests(
        self,
        manifests: tuple[tuple[str, str, bytes], ...],
    ) -> tuple[ContentManifestRef, ...]:
        """Validate then publish a conflict-free batch of exact manifest references."""
        with self._exclusive_lock():
            proposed = self._prepare_manifest_bindings(manifests)
            if not proposed:
                return ()
            self._validate_manifest_bindings_locked(proposed)
            self._publish_manifest_bindings_locked(proposed)
        return tuple(ref for _key, ref, _bytes in proposed)

    def _prepare_manifest_bindings(
        self,
        manifests: tuple[tuple[str, str, bytes], ...],
    ) -> list[tuple[str, ContentManifestRef, bytes]]:
        """Build and validate manifest-reference records without publishing them."""
        proposed: list[tuple[str, ContentManifestRef, bytes]] = []
        seen: dict[tuple[str, str], str] = {}
        for kind, manifest_id, raw_bytes in manifests:
            key = _manifest_reference_key(kind, manifest_id)
            if not isinstance(raw_bytes, bytes):
                raise TypeError("manifest bytes must be bytes")
            try:
                payload = strict_json_loads(
                    raw_bytes.decode("utf-8"),
                    ref=f"manifest {(kind, manifest_id)!r}",
                )
            except Exception as exc:
                raise ArtifactBlobReferenceError(
                    f"manifest {(kind, manifest_id)!r} must contain strict UTF-8 JSON: {exc}"
                ) from exc
            if not isinstance(payload, Mapping) or (
                payload.get("kind"),
                payload.get("id"),
            ) != (kind, manifest_id):
                raise ArtifactBlobReferenceError(
                    "manifest bytes must state the exact kind and id being bound: "
                    f"expected={(kind, manifest_id)!r}"
                )
            object_ref = self.store_object(raw_bytes)
            logical = (kind, manifest_id)
            previous = seen.get(logical)
            if previous is not None and previous != object_ref.sha256:
                raise ContentManifestConflictError(
                    f"conflicting bytes for manifest {(kind, manifest_id)!r}: "
                    f"{previous} != {object_ref.sha256}"
                )
            seen[logical] = object_ref.sha256
            ref = ContentManifestRef(
                kind=kind,
                id=manifest_id,
                ref=f"{CONTENT_MANIFEST_REF_PREFIX}{kind}/{manifest_id}",
                object=object_ref,
            )
            proposed.append((key, ref, canonical_json_v1_bytes(ref.model_dump(mode="json"))))

        return proposed

    def _validate_manifest_bindings_locked(
        self,
        proposed: list[tuple[str, ContentManifestRef, bytes]],
    ) -> None:
        for key, ref, expected in proposed:
            existing = self._read_manifest_reference(key, required=False)
            if existing is not None and existing != expected:
                current = ContentManifestRef.model_validate(
                    strict_json_loads(existing.decode("utf-8"), ref=ref.ref)
                )
                raise ContentManifestConflictError(
                    f"manifest {(ref.kind, ref.id)!r} is already pinned to "
                    f"{current.object.sha256}, not {ref.object.sha256}"
                )

    def _publish_manifest_bindings_locked(
        self,
        proposed: list[tuple[str, ContentManifestRef, bytes]],
    ) -> None:
        for key, _ref, expected in proposed:
            _stage_blob_bytes(expected, self._manifest_reference_path(key))

    def resolve_manifest(
        self,
        kind: str,
        manifest_id: str,
    ) -> tuple[ContentManifestRef, bytes]:
        """Resolve one logical manifest only through its exact pinned store reference."""
        key = _manifest_reference_key(kind, manifest_id)
        raw_ref = self._read_manifest_reference(key, required=True)
        assert raw_ref is not None
        try:
            ref = ContentManifestRef.model_validate(
                strict_json_loads(raw_ref.decode("utf-8"), ref=f"manifest reference {key}")
            )
        except Exception as exc:
            raise ArtifactBlobIntegrityError(
                f"manifest reference record is invalid: {(kind, manifest_id)!r}: {exc}"
            ) from exc
        if raw_ref != canonical_json_v1_bytes(ref.model_dump(mode="json")):
            raise ArtifactBlobIntegrityError(
                f"manifest reference record is not canonical: {(kind, manifest_id)!r}"
            )
        if (ref.kind, ref.id) != (kind, manifest_id):
            raise ArtifactBlobIntegrityError(
                f"manifest reference key resolves a different logical identity: {ref.ref}"
            )
        return ref, self.get_object(ref.object)

    def _manifest_reference_path(self, key: str) -> Path:
        return self.root / "references" / "manifests" / "sha256" / key[:2] / key

    def _read_manifest_reference(self, key: str, *, required: bool) -> bytes | None:
        path = self._manifest_reference_path(key)
        try:
            path_stat = path.lstat()
        except FileNotFoundError:
            if required:
                raise FileNotFoundError(f"manifest reference is missing: {key}")
            return None
        if not stat.S_ISREG(path_stat.st_mode) or path_stat.st_nlink != 1:
            raise ArtifactBlobIntegrityError(f"manifest reference must be one regular file: {path}")
        return self._read_canonical_blob(path, path_stat.st_size)

    def _ingest_tree_directory(
        self,
        directory_descriptor: int,
        relative: tuple[str, ...],
        directories: list[str],
        files: list[ContentTreeFile],
    ) -> None:
        children = sorted(
            os.scandir(directory_descriptor), key=lambda entry: os.fsencode(entry.name)
        )
        for child in children:
            name = child.name
            if unicodedata.normalize("NFC", name) != name:
                raise ArtifactBlobContainmentError(
                    "content-tree member names must use NFC Unicode normalization"
                )
            before = child.stat(follow_symlinks=False)
            path = PurePosixPath(*relative, name).as_posix()
            if stat.S_ISDIR(before.st_mode):
                child_descriptor = open_existing_directory(
                    name,
                    error_factory=ArtifactBlobContainmentError,
                    context=f"content-tree directory {path}",
                    dir_fd=directory_descriptor,
                )
                directories.append(path)
                try:
                    self._ingest_tree_directory(
                        child_descriptor,
                        (*relative, name),
                        directories,
                        files,
                    )
                    after = os.fstat(child_descriptor)
                    current = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
                    if _file_state(before) != _file_state(after) or _file_state(
                        before
                    ) != _file_state(current):
                        raise ArtifactBlobIntegrityError(
                            f"content-tree directory changed during ingest: {path}"
                        )
                finally:
                    os.close(child_descriptor)
            elif stat.S_ISREG(before.st_mode):
                if before.st_nlink != 1:
                    raise ArtifactBlobContainmentError(
                        f"content-tree files must have exactly one hard link: {path}"
                    )
                descriptor, opened = open_existing_file(
                    name,
                    rigor=SecurePathRigor.SINGLE_LINK_FILE_IDENTITY,
                    error_factory=ArtifactBlobContainmentError,
                    context=f"content-tree file {path}",
                    dir_fd=directory_descriptor,
                )
                try:
                    data = _read_file_descriptor(descriptor)
                    after = os.fstat(descriptor)
                    current = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
                    if (
                        _file_state(before) != _file_state(opened)
                        or _file_state(opened) != _file_state(after)
                        or _file_state(before) != _file_state(current)
                    ):
                        raise ArtifactBlobIntegrityError(
                            f"content-tree file changed during ingest: {path}"
                        )
                finally:
                    os.close(descriptor)
                files.append(ContentTreeFile(relative_path=path, object=self.store_object(data)))
            else:
                raise ArtifactBlobContainmentError(
                    f"content-tree members must be regular files or directories: {path}"
                )

    def store_bytes(
        self,
        data: bytes,
        *,
        role: str,
        logical_name: str,
        media_type: str = "application/octet-stream",
        metadata: Mapping[str, Any] | None = None,
    ) -> ArtifactRef:
        """Store exact bytes once and return their canonical artifact reference."""
        if not isinstance(data, bytes):
            raise TypeError("immutable artifact data must be bytes")
        artifact_metadata = dict(metadata or {})
        self._reject_reserved_metadata(artifact_metadata)

        object_ref = self.store_object(data)
        digest = object_ref.sha256
        artifact_id = f"{_ARTIFACT_ID_PREFIX}{digest}"
        artifact = ArtifactRef(
            role=role,
            logical_name=logical_name,
            artifact_id=artifact_id,
            sha256=digest,
            size_bytes=object_ref.size_bytes,
            storage_backend=_STORAGE_BACKEND,
            uri=artifact_id,
            media_type=media_type,
            metadata=artifact_metadata,
        )
        stored_data = self.get_bytes(artifact)
        if stored_data != data:
            raise ArtifactBlobIntegrityError(
                f"pre-existing artifact bytes differ from stored payload: {artifact_id}"
            )
        return artifact

    def get_bytes(
        self,
        artifact: ArtifactRef | str,
        *,
        size_bytes: int | None = None,
    ) -> bytes:
        """Read and validate exact bytes from a canonical artifact reference."""
        digest, expected_size = self._validated_reference(artifact, size_bytes=size_bytes)
        data = self._read_canonical_blob(self._canonical_path(digest), expected_size)
        actual_digest = hashlib.sha256(data).hexdigest()
        if actual_digest != digest:
            raise ArtifactBlobIntegrityError(
                f"artifact sha256 mismatch: expected {digest}, found {actual_digest}"
            )
        return data

    def canonical_relative_path(
        self,
        artifact: ArtifactRef | str,
        *,
        size_bytes: int | None = None,
    ) -> Path:
        """Return the provider-owned relative location after validating exact bytes."""
        digest, expected_size = self._validated_reference(artifact, size_bytes=size_bytes)
        self.get_bytes(artifact, size_bytes=size_bytes)
        return Path("artifacts") / "sha256" / digest[:2] / digest

    def materialize(
        self,
        artifact: ArtifactRef | str,
        destination: Path | str,
        *,
        size_bytes: int | None = None,
    ) -> Path:
        """Copy validated bytes atomically to a new independent destination."""
        data = self.get_bytes(artifact, size_bytes=size_bytes)
        requested_destination = _validate_explicit_destination(destination)
        destination_path = _canonicalize_trusted_system_aliases(requested_destination)
        self._validate_materialization_destination(destination_path)

        records = _open_directory_chain(destination_path.parent, create=True)
        parent_descriptor = records[-1][1]
        destination_name = destination_path.name
        temporary_name = f"payload-{uuid.uuid4().hex}"
        staging_descriptor: int | None = None
        temporary_descriptor: int | None = None
        final_descriptor: int | None = None
        temporary_identity: tuple[int, int] | None = None
        try:
            try:
                os.stat(destination_name, dir_fd=parent_descriptor, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise FileExistsError(
                    f"materialization destination already exists: {destination_path}"
                )

            staging_descriptor = _open_materialization_staging_container(
                parent_descriptor=parent_descriptor
            )
            temporary_descriptor = os.open(
                temporary_name,
                _file_flags(writable=True) | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=staging_descriptor,
            )
            _write_file_descriptor(temporary_descriptor, data)
            os.fsync(temporary_descriptor)
            temporary_stat = os.fstat(temporary_descriptor)
            temporary_identity = (temporary_stat.st_dev, temporary_stat.st_ino)
            if (
                temporary_stat.st_size != len(data)
                or _read_file_descriptor(temporary_descriptor) != data
            ):
                raise ArtifactBlobIntegrityError(
                    f"materialization temporary bytes failed verification: {destination_path}"
                )
            os.close(temporary_descriptor)
            temporary_descriptor = None

            try:
                _publish_materialized_file_no_replace(
                    temporary_name,
                    destination_name,
                    temporary_parent_descriptor=staging_descriptor,
                    parent_descriptor=parent_descriptor,
                )
            except FileExistsError as exc:
                raise FileExistsError(
                    f"materialization destination already exists: {destination_path}"
                ) from exc
            _remove_private_materialization_staging_name(
                directory_descriptor=staging_descriptor,
                temporary_name=temporary_name,
            )
            staging_descriptor = None

            final_descriptor, final_stat_before = open_existing_file(
                destination_name,
                rigor=SecurePathRigor.SINGLE_LINK_FILE_IDENTITY,
                error_factory=ArtifactBlobIntegrityError,
                context=f"materialized artifact {destination_path}",
                dir_fd=parent_descriptor,
            )
            if (
                final_stat_before.st_dev,
                final_stat_before.st_ino,
            ) != temporary_identity or final_stat_before.st_size != len(data):
                raise ArtifactBlobIntegrityError(
                    f"materialized artifact identity is invalid: {destination_path}"
                )
            materialized_data = _read_file_descriptor(final_descriptor)
            final_stat_after = os.fstat(final_descriptor)
            if (
                _file_state(final_stat_before) != _file_state(final_stat_after)
                or materialized_data != data
                or hashlib.sha256(materialized_data).digest() != hashlib.sha256(data).digest()
            ):
                raise ArtifactBlobIntegrityError(
                    f"materialized artifact bytes failed verification: {destination_path}"
                )
            path_stat = os.stat(
                destination_name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if (path_stat.st_dev, path_stat.st_ino) != temporary_identity:
                raise ArtifactBlobIntegrityError(
                    f"materialized artifact path identity changed: {destination_path}"
                )
            _recheck_directory_chain(records)
            external_stat = os.stat(destination_path, follow_symlinks=False)
            if (external_stat.st_dev, external_stat.st_ino) != temporary_identity:
                raise ArtifactBlobContainmentError(
                    f"materialization destination mapping changed: {destination_path}"
                )
            self._validate_materialization_destination(destination_path)
            return requested_destination
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                raise ArtifactBlobContainmentError(
                    f"materialization traversed a symlink or non-directory: {destination_path}"
                ) from exc
            raise
        finally:
            if temporary_descriptor is not None:
                os.close(temporary_descriptor)
            if final_descriptor is not None:
                os.close(final_descriptor)
            if staging_descriptor is not None:
                try:
                    _remove_private_materialization_staging_name(
                        directory_descriptor=staging_descriptor,
                        temporary_name=temporary_name,
                    )
                except OSError:
                    # Preserve an exceptional private orphan for diagnosis.
                    # Never widen cleanup to a public destination name.
                    os.close(staging_descriptor)
            _close_directory_chain(records)

    def _canonical_path(self, digest: str) -> Path:
        return self.root / "artifacts" / "sha256" / digest[:2] / digest

    def _validated_reference(
        self,
        artifact: ArtifactRef | str,
        *,
        size_bytes: int | None,
    ) -> tuple[str, int]:
        if isinstance(artifact, str):
            digest = _parse_artifact_id(artifact)
            if size_bytes is None:
                raise ArtifactBlobReferenceError(
                    "size_bytes is required when reading a raw artifact identifier"
                )
            return digest, _validate_size(size_bytes, field="size_bytes")

        if not isinstance(artifact, ArtifactRef):
            raise ArtifactBlobReferenceError("artifact must be an ArtifactRef or artifact ID")
        digest = _parse_artifact_id(artifact.artifact_id)
        reference_digest = _validate_digest(artifact.sha256)
        if reference_digest != digest:
            raise ArtifactBlobReferenceError("artifact_id digest does not match ArtifactRef.sha256")
        uri_digest = _parse_artifact_id(artifact.uri)
        if artifact.uri != artifact.artifact_id or uri_digest != digest:
            raise ArtifactBlobReferenceError(
                "ArtifactRef.uri must equal its exact canonical artifact_id"
            )
        reference_size = _validate_size(artifact.size_bytes, field="ArtifactRef.size_bytes")
        if artifact.storage_backend != _STORAGE_BACKEND:
            raise ArtifactBlobReferenceError(
                "artifact storage_backend is unsupported by this custody provider"
            )
        if not isinstance(artifact.metadata, Mapping):
            raise ArtifactBlobReferenceError("ArtifactRef.metadata must be a mapping")
        self._reject_reserved_metadata(artifact.metadata)
        if size_bytes is not None:
            override_size = _validate_size(size_bytes, field="size_bytes")
            if override_size != reference_size:
                raise ArtifactBlobReferenceError(
                    "size_bytes override does not match ArtifactRef.size_bytes"
                )
        return digest, reference_size

    @staticmethod
    def _reject_reserved_metadata(metadata: Mapping[str, Any]) -> None:
        reserved = sorted(_RESERVED_METADATA_KEYS.intersection(metadata))
        if reserved:
            raise ArtifactBlobReferenceError(
                "artifact metadata contains reserved locator fields: " + ", ".join(reserved)
            )

    def _validate_store_target(self, target: Path) -> None:
        self._validate_lexical_containment(target)
        for parent in (
            self.root,
            self.root / "artifacts",
            self.root / "artifacts" / "sha256",
            target.parent,
        ):
            try:
                parent_stat = parent.lstat()
            except FileNotFoundError:
                continue
            if stat.S_ISLNK(parent_stat.st_mode):
                raise ArtifactBlobContainmentError(
                    f"custody store component must not be a symlink: {parent}"
                )
            if not stat.S_ISDIR(parent_stat.st_mode):
                raise ArtifactBlobContainmentError(
                    f"custody store component must be a directory: {parent}"
                )
        try:
            target_stat = target.lstat()
        except FileNotFoundError:
            return
        if stat.S_ISLNK(target_stat.st_mode):
            raise ArtifactBlobContainmentError(
                f"canonical artifact must not be a symlink: {target}"
            )
        if not stat.S_ISREG(target_stat.st_mode):
            raise ArtifactBlobIntegrityError(f"canonical artifact must be a regular file: {target}")

    def _validate_lexical_containment(self, target: Path) -> None:
        try:
            target.relative_to(self.root)
        except ValueError as exc:
            raise ArtifactBlobContainmentError(
                f"artifact path escapes custody root: {target}"
            ) from exc

    def _read_canonical_blob(self, target: Path, expected_size: int) -> bytes:
        self._validate_lexical_containment(target)
        try:
            records = _open_directory_chain(target.parent, create=False)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"artifact blob is missing: {target}") from exc
        parent_descriptor = records[-1][1]
        file_descriptor: int | None = None
        try:
            try:
                path_stat_before = os.stat(
                    target.name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError as exc:
                raise FileNotFoundError(f"artifact blob is missing: {target}") from exc
            if stat.S_ISLNK(path_stat_before.st_mode):
                raise ArtifactBlobContainmentError(
                    f"canonical artifact must not be a symlink: {target}"
                )
            file_descriptor, file_stat_before = open_existing_file(
                target.name,
                rigor=SecurePathRigor.SINGLE_LINK_FILE_IDENTITY,
                error_factory=ArtifactBlobIntegrityError,
                context=f"canonical artifact {target}",
                dir_fd=parent_descriptor,
            )
            if (file_stat_before.st_dev, file_stat_before.st_ino) != (
                path_stat_before.st_dev,
                path_stat_before.st_ino,
            ):
                raise ArtifactBlobIntegrityError(
                    f"canonical artifact identity changed before read: {target}"
                )
            if file_stat_before.st_size != expected_size:
                raise ArtifactBlobIntegrityError(
                    "artifact size mismatch: "
                    f"expected {expected_size}, found {file_stat_before.st_size}"
                )
            data = _read_file_descriptor(file_descriptor)
            file_stat_after = os.fstat(file_descriptor)
            if _file_state(file_stat_before) != _file_state(file_stat_after):
                raise ArtifactBlobIntegrityError(
                    f"canonical artifact identity changed during read: {target}"
                )
            path_stat_after = os.stat(
                target.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if _file_state(path_stat_before) != _file_state(path_stat_after):
                raise ArtifactBlobIntegrityError(
                    f"canonical artifact path changed during read: {target}"
                )
            _recheck_directory_chain(records)
            external_stat = os.stat(target, follow_symlinks=False)
            if (external_stat.st_dev, external_stat.st_ino) != (
                file_stat_after.st_dev,
                file_stat_after.st_ino,
            ):
                raise ArtifactBlobContainmentError(
                    f"canonical artifact path mapping changed during read: {target}"
                )
            if len(data) != expected_size:
                raise ArtifactBlobIntegrityError(
                    f"artifact read size mismatch: expected {expected_size}, found {len(data)}"
                )
            return data
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                raise ArtifactBlobContainmentError(
                    f"artifact path traverses a symlink or non-directory: {target}"
                ) from exc
            raise
        finally:
            if file_descriptor is not None:
                os.close(file_descriptor)
            _close_directory_chain(records)

    def _validate_materialization_destination(self, destination: Path) -> None:
        if not destination.name:
            raise ArtifactBlobReferenceError("materialization destination must name a file")
        custody_store = self.root / "artifacts" / "sha256"
        try:
            destination.relative_to(custody_store)
        except ValueError:
            return
        raise ArtifactBlobContainmentError(
            f"materialization destination must be outside custody CAS: {destination}"
        )


def open_immutable_artifact_blob_provider(
    spec: ImmutableArtifactBlobProviderSpec | Mapping[str, Any],
    *,
    explicit_root: Path | str,
) -> ImmutableArtifactBlobProvider:
    """Bind a portable provider spec to one explicit machine-local custody root."""
    if isinstance(spec, ImmutableArtifactBlobProviderSpec):
        provider_spec = spec
    elif isinstance(spec, Mapping):
        try:
            provider_spec = ImmutableArtifactBlobProviderSpec.model_validate(spec)
        except PydanticValidationError as exc:
            raise ArtifactBlobReferenceError(
                "unsupported immutable artifact blob provider spec; expected "
                f"schema_id={IMMUTABLE_ARTIFACT_BLOB_PROVIDER_SCHEMA_ID!r}, "
                "current_version="
                f"{IMMUTABLE_ARTIFACT_BLOB_PROVIDER_SCHEMA_VERSION!r}, "
                f"kind={IMMUTABLE_ARTIFACT_BLOB_PROVIDER_KIND!r}, "
                f"storage_backend={IMMUTABLE_ARTIFACT_BLOB_STORAGE_BACKEND!r}: {exc}"
            ) from exc
    else:
        raise ArtifactBlobReferenceError(
            "provider spec must be ImmutableArtifactBlobProviderSpec or a mapping"
        )

    root_path = _validate_explicit_root(explicit_root)
    if provider_spec.schema_id != IMMUTABLE_ARTIFACT_BLOB_PROVIDER_SCHEMA_ID:
        raise ArtifactBlobReferenceError(
            "unsupported immutable artifact blob provider schema_id; expected "
            f"{IMMUTABLE_ARTIFACT_BLOB_PROVIDER_SCHEMA_ID!r}; "
            f"current_version={IMMUTABLE_ARTIFACT_BLOB_PROVIDER_SCHEMA_VERSION!r}"
        )
    if provider_spec.schema_version != IMMUTABLE_ARTIFACT_BLOB_PROVIDER_SCHEMA_VERSION:
        raise ArtifactBlobReferenceError(
            "unsupported immutable artifact blob provider schema_version; "
            f"current_version={IMMUTABLE_ARTIFACT_BLOB_PROVIDER_SCHEMA_VERSION!r}"
        )
    if provider_spec.kind != IMMUTABLE_ARTIFACT_BLOB_PROVIDER_KIND:
        raise ArtifactBlobReferenceError(
            "unsupported immutable artifact blob provider kind; expected "
            f"{IMMUTABLE_ARTIFACT_BLOB_PROVIDER_KIND!r}; "
            f"current_version={IMMUTABLE_ARTIFACT_BLOB_PROVIDER_SCHEMA_VERSION!r}"
        )
    if provider_spec.config.storage_backend != IMMUTABLE_ARTIFACT_BLOB_STORAGE_BACKEND:
        raise ArtifactBlobReferenceError(
            "unsupported immutable artifact blob storage_backend; expected "
            f"{IMMUTABLE_ARTIFACT_BLOB_STORAGE_BACKEND!r}; "
            f"current_version={IMMUTABLE_ARTIFACT_BLOB_PROVIDER_SCHEMA_VERSION!r}"
        )
    return ImmutableArtifactBlobProvider(
        root=root_path,
        storage_backend=provider_spec.config.storage_backend,
    )


__all__ = [
    "ArtifactBlobContainmentError",
    "ArtifactBlobCustodyError",
    "ArtifactBlobIntegrityError",
    "ArtifactBlobReferenceError",
    "ContentManifestConflictError",
    "ContentManifestRef",
    "ContentObjectRef",
    "ContentTree",
    "ContentTreeFile",
    "ContentTreeRef",
    "ImmutableArtifactBlobProvider",
    "open_immutable_artifact_blob_provider",
]
