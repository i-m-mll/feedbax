"""Immutable, content-addressed custody for exact artifact bytes."""

from __future__ import annotations

import errno
import fcntl
import hashlib
import os
import re
import stat
import sys
import threading
import unicodedata
import uuid
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Iterator, Literal

from pydantic import JsonValue
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
from feedbax.contracts.canonical_json import canonical_json_v1_bytes, canonical_json_v2_bytes
from feedbax.contracts.strict_json import strict_json_loads
from feedbax.persistence._store_metadata import sync_directory, write_store_metadata
from feedbax.persistence.store_damage import (
    ObjectVerificationError, clear_damage, observe_failure, occurrence, read_damage, record_damage, require_sound,
)

if TYPE_CHECKING:
    from feedbax.contracts.project_io import TextProject


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
_REF_COMPONENT = re.compile(r"[a-z0-9_][a-z0-9_:-]*")


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
def _exclusive_store_lock(root: Path, *, create: bool = True) -> Iterator[None]:
    lock_key = str(root)
    held = getattr(_store_lock_state, "held", {})
    if lock_key in held:
        held[lock_key] += 1
        try:
            yield
        finally:
            held[lock_key] -= 1
        return
    if create:
        root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".content-store.lock"
    flags = (os.O_RDWR | os.O_CREAT) if create else os.O_RDONLY
    descriptor = os.open(lock_path, flags | os.O_NOFOLLOW, 0o600)
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


def content_object_relative_path(digest: str) -> Path:
    digest = _validate_digest(digest)
    return Path("artifacts", "sha256", digest[:2], digest)


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


def _write_staged_file(parent: int, name: str, *, write_payload: Callable[[int], None]) -> int:
    descriptor = os.open(
        name, _file_flags(writable=True) | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=parent
    )
    try:
        write_payload(descriptor)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _clone_staged_file(parent: int, name: str, *, source: int) -> int:
    if sys.platform == "darwin":
        import ctypes

        clone = ctypes.CDLL(None, use_errno=True).fclonefileat
        clone.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32)
        clone.restype = ctypes.c_int
        if clone(source, parent, os.fsencode(name), 0) != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), name)
        descriptor = os.open(name, _file_flags(), dir_fd=parent)
        try:
            os.fchmod(descriptor, 0o400)
            os.utime(descriptor, None)
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    request = getattr(fcntl, "FICLONE", None)
    if sys.platform != "linux" or request is None:
        raise OSError(errno.ENOTSUP, "Store sync requires a filesystem with file cloning")

    def clone_payload(descriptor: int) -> None:
        fcntl.ioctl(descriptor, request, source)

    return _write_staged_file(parent, name, write_payload=clone_payload)


def _remove_write_permissions(descriptor: int, info: os.stat_result) -> None:
    mode = stat.S_IMODE(info.st_mode)
    os.fchmod(descriptor, mode & ~0o222)


def _stage_blob(
    destination: Path,
    expected_identity: tuple[str, int],
    create_payload: Callable[[int, str], int],
    *,
    validate: Callable[[Path], None] | None = None,
    on_corrupt: Callable[[Exception], None] | None = None,
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
        temporary_descriptor = create_payload(staging_descriptor, temporary_name)
        if validate is not None:
            os.lseek(temporary_descriptor, 0, os.SEEK_SET)
            validate(_descriptor_path(temporary_descriptor))
        _remove_write_permissions(temporary_descriptor, os.fstat(temporary_descriptor))
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
            installed = True
        except FileExistsError:
            installed = False
        final_descriptor, before = open_existing_file(
            destination.name,
            rigor=SecurePathRigor.SINGLE_LINK_FILE_IDENTITY,
            error_factory=ArtifactBlobIntegrityError,
            context=f"canonical blob {destination}",
            dir_fd=parent_descriptor,
        )
        stored_identity = _stream_file_descriptor(final_descriptor)
        after = os.fstat(final_descriptor)
        path_stat = os.stat(destination.name, dir_fd=parent_descriptor, follow_symlinks=False)
        if _file_state(before) != _file_state(after) or _file_state(path_stat) != _file_state(after):
            raise ArtifactBlobIntegrityError(
                f"canonical blob identity changed during publication: {destination}"
            )
        _recheck_directory_chain(records)
        if stored_identity != expected_identity:
            if stored_identity[0] == expected_identity[0]:
                raise ArtifactBlobReferenceError(f"publication size differs from verified object: {destination}")
            error = ObjectVerificationError(
                f"canonical blob bytes do not match content identity: {destination}, found {stored_identity}",
                found_sha256=stored_identity[0], found_size=stored_identity[1],
            )
            if on_corrupt is None:
                raise error
            on_corrupt(error)
            incumbent = os.stat(destination.name, dir_fd=parent_descriptor, follow_symlinks=False)
            if _file_state(incumbent) != _file_state(after):
                raise ArtifactBlobIntegrityError(f"corrupt occupant changed before replacement: {destination}")
            os.replace(
                temporary_name, destination.name,
                src_dir_fd=staging_descriptor, dst_dir_fd=parent_descriptor,
            )
            installed = True
            after = os.fstat(temporary_descriptor)
            if _file_state(os.stat(destination.name, dir_fd=parent_descriptor, follow_symlinks=False)) != _file_state(after):
                raise ArtifactBlobIntegrityError(f"verified replacement changed during publication: {destination}")
        _recheck_directory_chain(records)
        if installed:
            os.fsync(parent_descriptor)
        if after.st_mode & 0o222:
            _remove_write_permissions(final_descriptor, after)
            os.fsync(final_descriptor)
            return os.fstat(final_descriptor)
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


def _stage_blob_bytes(
    data: bytes,
    destination: Path,
    *,
    identity: tuple[str, int] | None = None,
    validate: Callable[[Path], None] | None = None,
    on_corrupt: Callable[[Exception], None] | None = None,
) -> os.stat_result:
    if identity is None:
        identity = hashlib.sha256(data).hexdigest(), len(data)
    writer = partial(_write_staged_file, write_payload=partial(_write_file_descriptor, data=data))
    return _stage_blob(destination, identity, writer, validate=validate, on_corrupt=on_corrupt)


def _stage_blob_file(
    source_descriptor: int,
    destination: Path,
    expected_identity: tuple[str, int],
    *,
    on_corrupt: Callable[[Exception], None],
    validate: Callable[[Path], None] | None = None,
) -> os.stat_result:
    def write_payload(descriptor: int) -> None:
        if _stream_file_descriptor(source_descriptor, target=descriptor) != expected_identity:
            raise ArtifactBlobIntegrityError(f"artifact source changed during store: {destination}")

    writer = partial(_write_staged_file, write_payload=write_payload)
    return _stage_blob(destination, expected_identity, writer, validate=validate, on_corrupt=on_corrupt)


def _object_identity(
    object_ref: ContentObjectRef | str,
    *,
    size_bytes: int | None = None,
) -> tuple[str, int]:
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
    return digest, expected_size


def _content_ref(digest: str, size: int) -> ContentObjectRef:
    return ContentObjectRef(
        object_id=f"{CONTENT_OBJECT_REF_PREFIX}{digest}", sha256=digest, size_bytes=size,
    )


def _read_blob(target: Path, expected_size: int) -> bytes:
    with _open_blob_for_read(target, expected_size) as descriptor:
        data = _read_file_descriptor(descriptor)
    if len(data) != expected_size:
        raise ObjectVerificationError(
            f"artifact read size mismatch: expected {expected_size}, found {len(data)}: {target}",
            found_size=len(data),
        )
    return data


def _verify_blob(
    target: Path, *, sha256: str, size_bytes: int, destination_descriptor: int | None = None,
    freeze: bool = False,
) -> None:
    with _open_blob_for_read(target, None, freeze=freeze) as descriptor:
        actual_digest, actual_size = _stream_file_descriptor(
            descriptor, target=destination_descriptor
        )
        if actual_digest == sha256 and actual_size != size_bytes:
            raise ArtifactBlobReferenceError(
                f"reference size mismatch for {target}: expected {size_bytes}, found {actual_size}"
            )
        if actual_size != size_bytes:
            raise ObjectVerificationError(
                f"artifact read size mismatch: expected {size_bytes}, found {actual_size}: {target}",
                found_size=actual_size, found_sha256=actual_digest,
            )
        if actual_digest != sha256:
            raise ObjectVerificationError(
                f"artifact SHA-256 mismatch: expected {sha256}, found {actual_digest}: {target}",
                found_size=actual_size, found_sha256=actual_digest,
            )


@contextmanager
def _open_blob_for_read(
    target: Path, expected_size: int | None, *, freeze: bool = False,
) -> Iterator[int]:
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
        if expected_size is not None and file_stat_before.st_size != expected_size:
            raise ObjectVerificationError(
                "artifact size mismatch: "
                f"expected {expected_size}, found {file_stat_before.st_size}: {target}",
                found_size=file_stat_before.st_size,
            )
        yield file_descriptor
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
        if freeze and file_stat_after.st_mode & 0o222:
            _remove_write_permissions(file_descriptor, file_stat_after)
            os.fsync(file_descriptor)
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


def _encode_store_document(document: dict[str, JsonValue]) -> bytes:
    if not isinstance(document, dict):
        raise ArtifactBlobReferenceError("store metadata must be a JSON object")
    return canonical_json_v2_bytes(document)


def _read_store_document(path: Path) -> tuple[bytes, dict[str, JsonValue]] | None:
    try:
        with _open_blob_for_read(path, None) as descriptor:
            raw = _read_file_descriptor(descriptor)
    except FileNotFoundError:
        return None
    try:
        document = strict_json_loads(raw, ref=str(path))
        if raw == _encode_store_document(document):
            return raw, document
    except ValueError as exc:
        raise ArtifactBlobIntegrityError(f"invalid store metadata at {path}: {exc}") from exc
    raise ArtifactBlobIntegrityError(f"store metadata is not canonical JSON: {path}")


def _read_content_object(path: Path, digest: str, expected_size: int) -> bytes:
    try:
        data = _read_blob(path, expected_size)
    except ObjectVerificationError as error:
        # A size disagreement can come from the reference; verify before attributing damage.
        _verify_blob(path, sha256=digest, size_bytes=expected_size)
        raise ArtifactBlobIntegrityError(f"canonical object changed during read: {path}") from error
    actual = hashlib.sha256(data).hexdigest()
    if actual != digest:
        raise ObjectVerificationError(
            f"content object SHA-256 mismatch: expected {digest}, found {actual}: {path}",
            found_size=len(data), found_sha256=actual,
        )
    return data


def _descriptor_path(descriptor: int) -> Path:
    return Path("/dev/fd") / str(descriptor)


def verify_content_file(path: Path, object_ref: ContentObjectRef) -> None:
    """Verify a stable, single-link regular file against its pinned object identity."""
    digest, size_bytes = _object_identity(object_ref)
    _verify_blob(_validate_explicit_root(path), sha256=digest, size_bytes=size_bytes)


def copy_content_file(source: Path, destination: Path, object_ref: ContentObjectRef) -> None:
    """Verify the source while copying it to a new private file in a trusted directory."""
    digest, size_bytes = _object_identity(object_ref)
    source = _validate_explicit_root(source)
    descriptor = os.open(destination, _file_flags(writable=True) | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        try:
            _verify_blob(
                source, sha256=digest, size_bytes=size_bytes, destination_descriptor=descriptor
            )
            os.fchmod(descriptor, 0o600)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException:
        destination.unlink()
        raise


def read_content_tree_descriptor(path: Path, tree_ref: ContentTreeRef) -> ContentTree:
    """Read and parse a stable file containing a pinned canonical tree descriptor."""
    if not isinstance(tree_ref, ContentTreeRef):
        raise ArtifactBlobReferenceError("tree_ref must be a ContentTreeRef")
    digest, expected_size = _object_identity(tree_ref.descriptor)
    raw = _read_content_object(_validate_explicit_root(path), digest, expected_size)
    return decode_content_tree_descriptor(raw, tree_ref)


def decode_content_tree_descriptor(raw: bytes, tree_ref: ContentTreeRef) -> ContentTree:
    """Parse canonical content-tree bytes after checking their pinned identity."""
    if len(raw) != tree_ref.descriptor.size_bytes or hashlib.sha256(raw).hexdigest() != (
        tree_ref.descriptor.sha256
    ):
        raise ArtifactBlobIntegrityError(f"content-tree descriptor identity changed: {tree_ref.tree_id}")
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

    def _exclusive_lock(self, *, create: bool = True) -> AbstractContextManager[None]:
        """Return the one store lock shared by references and retention state."""
        return _exclusive_store_lock(self.root, create=create)

    @property
    def store_id(self) -> str:
        """Return the persistent ID, initializing it on first use."""
        with self._exclusive_lock():
            return self._ensure_store_id_locked()

    def _ensure_store_id_locked(self) -> str:
        path = self.root / "store.json"
        record = _read_store_document(path)
        if record is None:
            if os.path.lexists(self.root / "refs"):
                raise ArtifactBlobIntegrityError(f"store identity is missing: {path}")
            store_id = str(uuid.uuid4())
            write_store_metadata(
                self.root,
                path,
                _encode_store_document({"id": store_id}),
                error_factory=ArtifactBlobContainmentError,
            )
            return store_id

        _, document = record
        store_id = document.get("id")
        if len(document) == 1 and isinstance(store_id, str):
            try:
                if str(uuid.UUID(store_id, version=4)) == store_id:
                    return store_id
            except ValueError:
                pass
        raise ArtifactBlobIntegrityError(f"invalid store identity: {path}")

    def _ref_path(self, name: str) -> Path:
        parts = name.split("/")
        if not all(_REF_COMPONENT.fullmatch(part) for part in parts):
            raise ArtifactBlobReferenceError(f"invalid store ref name: {name!r}")
        return self.root.joinpath("refs", *parts)

    def read_ref(self, name: str) -> dict[str, JsonValue] | None:
        """Read a complete ref or return None when absent.

        A name consists of slash-separated lowercase path components. A colon
        inside a component is allowed for request IDs.
        """
        path = self._ref_path(name)
        with self._exclusive_lock():
            record = _read_store_document(path)
        return record[1] if record is not None else None

    def compare_and_swap_ref(
        self,
        name: str,
        *,
        expected: dict[str, JsonValue] | None,
        replacement: dict[str, JsonValue] | None,
    ) -> bool:
        """Atomically change a ref if its canonical value matches.

        None denotes absence or deletion. False means a stale expectation.
        Callers own the schema and object liveness of their ref values.
        """
        path = self._ref_path(name)
        previous = _encode_store_document(expected) if expected is not None else None
        new = _encode_store_document(replacement) if replacement is not None else None
        with self._exclusive_lock():
            record = _read_store_document(path)
            current = record[0] if record is not None else None
            if current != previous:
                return False
            if current == new:
                return True
            self._ensure_store_id_locked()
            if new is None:
                path.unlink()
                sync_directory(path.parent)
            else:
                write_store_metadata(
                    self.root, path, new, error_factory=ArtifactBlobContainmentError
                )
        return True

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

    def release_owned_lease(self, lease_id: str, *, owner: str) -> ContentLeaseRecord | None:
        """Release an owner's exact lease if it was acquired."""
        from feedbax.persistence.content_retention import release_owned_lease

        return release_owned_lease(self, lease_id, owner=owner)

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

    def store_object(
        self,
        data: bytes,
        *,
        expected: ContentObjectRef | None = None,
        validate: Callable[[Path], None] | None = None,
    ) -> ContentObjectRef:
        """Check the supplied identity and staged contents before publication."""
        if not isinstance(data, bytes):
            raise TypeError("content object data must be bytes")
        identity = hashlib.sha256(data).hexdigest(), len(data)
        if expected is not None and identity != _object_identity(expected):
            raise ArtifactBlobIntegrityError(f"content bytes do not match {expected.object_id}")
        digest, _ = identity
        destination = self._canonical_path(digest)
        self._validate_store_target(destination)
        with self._exclusive_lock():
            self._validate_store_target(destination)
            ref = _content_ref(*identity)
            _stage_blob_bytes(
                data, destination, identity=identity,
                validate=self._replacement_validator(ref, validate),
                on_corrupt=partial(self._record_publication_damage, ref),
            )
            self._settle_install(ref)
            return ref

    def _record_publication_damage(self, ref: ContentObjectRef, error: Exception) -> None:
        record_damage(self, ref, error, verifier="publication")

    def _settle_install(self, ref: ContentObjectRef) -> None:
        if read_damage(self, ref.sha256) is not None:
            sync_directory(self.object_path(ref).parent)
            clear_damage(self, ref.sha256)

    def _replacement_validator(
        self, ref: ContentObjectRef, validate: Callable[[Path], None] | None = None,
    ) -> Callable[[Path], None] | None:
        observation = read_damage(self, ref.sha256)
        if observation is None or observation.descriptor_sha256 is None:
            return validate

        def verify(path: Path) -> None:
            from feedbax.persistence.array_container import open_array_container

            if validate is not None:
                validate(path)
            # Opening /dev/fd on Darwin shares the staging descriptor's offset.
            # Each validator must begin at the start of those same staged bytes.
            if path.parent == Path("/dev/fd"):
                os.lseek(int(path.name), 0, os.SEEK_SET)
            with open_array_container(path, expected_descriptor_sha256=observation.descriptor_sha256) as reader:
                reader.verify_tensors()

        return verify

    def store_rooted_object(self, data: bytes, *, namespace: str) -> ContentObjectRef:
        """Store an imported object and retain its exact identity in one transaction."""
        with self._exclusive_lock():
            ref = self.store_object(data)
            self.register_root(ContentRootRecord(
                root_id=f"{namespace}:{ref.sha256}",
                root_kind="import",
                targets=(ContentRetentionTarget(ref_kind="object", object=ref),),
            ))
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
                ref = _content_ref(*identity)
                validate = self._replacement_validator(ref)
                if source != destination and not self._reuse_object(ref, validate):
                    _stage_blob_file(
                        source_descriptor, destination, identity,
                        on_corrupt=partial(self._record_publication_damage, ref),
                        validate=validate,
                    )
                if _file_state(before) != _file_state(os.fstat(source_descriptor)):
                    raise ArtifactBlobIntegrityError(f"artifact source changed during store: {source}")
                path_after = os.stat(source.name, dir_fd=source_parent, follow_symlinks=False)
                if (path_after.st_dev, path_after.st_ino) != (before.st_dev, before.st_ino):
                    raise ArtifactBlobIntegrityError(f"artifact source identity changed during store: {source}")
                _recheck_directory_chain(source_records)
                if source == destination:
                    if validate is not None:
                        validate(_descriptor_path(source_descriptor))
                    _verify_blob(destination, sha256=ref.sha256, size_bytes=ref.size_bytes, freeze=True)
                self._settle_install(ref)
                return ref
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
        digest, expected_size = _object_identity(object_ref, size_bytes=size_bytes)
        ref = _content_ref(digest, expected_size)
        require_sound(self, ref)
        before = occurrence(self, ref)
        try:
            return _read_content_object(self._canonical_path(digest), digest, expected_size)
        except (ObjectVerificationError, FileNotFoundError) as error:
            # A missing Store has no local observation authority.
            if (self.root / ".content-store.lock").exists():
                observe_failure(self, ref, before, error, verifier="object read")
            raise

    def object_path(self, object_ref: ContentObjectRef) -> Path:
        """Derive the canonical path for a pinned object without reading it."""
        digest, _ = _object_identity(object_ref)
        return self._canonical_path(digest)

    @contextmanager
    def open_owned_object(
        self, object_ref: ContentObjectRef, *, allow_damaged: bool = False,
    ) -> Iterator[Path]:
        """Borrow a verified store object's pinned file for a selective reader.

        Ingress establishes the byte identity. The borrow checks containment,
        size and stability without hashing payloads.
        """
        if not allow_damaged:
            require_sound(self, object_ref)
        with self._borrow_object(object_ref) as path:
            yield path

    @contextmanager
    def _borrow_object(self, object_ref: ContentObjectRef) -> Iterator[Path]:
        digest, size_bytes = _object_identity(object_ref)
        with _open_blob_for_read(self._canonical_path(digest), size_bytes) as descriptor:
            yield _descriptor_path(descriptor)

    def verify_object(
        self, object_ref: ContentObjectRef, *, verifier: str = "object verification",
    ) -> None:
        """Verify one pinned object in bounded chunks."""
        digest, size_bytes = _object_identity(object_ref)
        before = occurrence(self, object_ref)
        try:
            _verify_blob(self._canonical_path(digest), sha256=digest, size_bytes=size_bytes)
        except (ObjectVerificationError, FileNotFoundError) as error:
            if (self.root / ".content-store.lock").exists():
                observe_failure(self, object_ref, before, error, verifier=verifier)
            raise

    def verify_observed_object(self, object_ref: ContentObjectRef, *, verifier: str) -> None:
        """Verify bytes and any known descriptor failure without clearing the observation."""
        self.verify_object(object_ref, verifier=verifier)
        validate = self._replacement_validator(object_ref)
        if validate is not None:
            with self._borrow_object(object_ref) as path:
                validate(path)

    def _clone_object_from(
        self, source: ImmutableArtifactBlobProvider, object_ref: ContentObjectRef
    ) -> bool:
        """Clone into canonical custody, or reuse a verified object, with both Stores locked."""
        digest, size_bytes = _object_identity(object_ref)
        target = self._canonical_path(digest)
        source.verify_observed_object(object_ref, verifier="transfer source")
        validate = self._replacement_validator(object_ref)
        if self._reuse_object(object_ref, validate):
            self._settle_install(object_ref)
            return False

        self._validate_store_target(target)
        with _open_blob_for_read(source.object_path(object_ref), size_bytes) as descriptor:
            _stage_blob(
                target, (digest, size_bytes), partial(_clone_staged_file, source=descriptor),
                on_corrupt=partial(self._record_publication_damage, object_ref),
                validate=validate,
            )
        self._settle_install(object_ref)
        return True

    def _reuse_object(
        self, ref: ContentObjectRef, validate: Callable[[Path], None] | None,
    ) -> bool:
        """Reuse sound canonical bytes under the publication lock."""
        target = self.object_path(ref)
        try:
            _verify_blob(target, sha256=ref.sha256, size_bytes=ref.size_bytes, freeze=True)
        except (FileNotFoundError, ObjectVerificationError):
            # The shared writer verifies the occupant again before installing anything.
            return False
        if validate is not None:
            with self._borrow_object(ref) as path:
                validate(path)
        return True

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
        return self.store_tree(tree)

    def store_tree(self, tree: ContentTree) -> ContentTreeRef:
        """Store a canonical descriptor whose members already have exact content references."""
        descriptor = self.store_object(canonical_json_v1_bytes(tree.model_dump(mode="json")))
        return ContentTreeRef(tree_id=f"{CONTENT_TREE_REF_PREFIX}{descriptor.sha256}", descriptor=descriptor)

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
            tree_ref = self.store_tree(tree)
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
        return decode_content_tree_descriptor(self.get_object(tree_ref.descriptor), tree_ref)

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
        ref = self._resolve_manifest_ref(kind, manifest_id)
        return ref, self.get_object(ref.object)

    def _resolve_manifest_ref(self, kind: str, manifest_id: str) -> ContentManifestRef:
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
        return ref

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
        try:
            return self.get_object(_content_ref(digest, expected_size))
        except ObjectVerificationError as error:
            if error.found_sha256 is None:
                raise
            raise ArtifactBlobIntegrityError(
                f"artifact sha256 mismatch: expected {digest}, found {error.found_sha256}"
            ) from error

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
        return self.root / content_object_relative_path(digest)

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
        return _read_blob(target, expected_size)

    def _verify_blob(self, target: Path, *, sha256: str, size_bytes: int) -> None:
        self._validate_lexical_containment(target)
        _verify_blob(target, sha256=sha256, size_bytes=size_bytes)

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


def open_project_store(project: TextProject) -> ImmutableArtifactBlobProvider:
    """Bind the existing object owner to the store declared by a text project."""
    store = ImmutableArtifactBlobProvider(root=project.store_path)
    with store._exclusive_lock():
        store._ensure_store_id_locked()
    return store


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
