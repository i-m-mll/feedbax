"""Immutable, content-addressed custody for exact artifact bytes."""

from __future__ import annotations

import errno
import hashlib
import os
import stat
import sys
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from feedbax.contracts.artifact_custody import (
    IMMUTABLE_ARTIFACT_BLOB_PROVIDER_KIND,
    IMMUTABLE_ARTIFACT_BLOB_PROVIDER_SCHEMA_ID,
    IMMUTABLE_ARTIFACT_BLOB_PROVIDER_SCHEMA_VERSION,
    IMMUTABLE_ARTIFACT_BLOB_STORAGE_BACKEND,
    ArtifactBlobContainmentError,
    ArtifactBlobCustodyError,
    ArtifactBlobIntegrityError,
    ArtifactBlobReferenceError,
    ImmutableArtifactBlobProviderSpec,
)
from feedbax.contracts.base import ArtifactRef


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
_DirectoryRecord = tuple[Path, int, os.stat_result]


def _canonicalize_trusted_system_aliases(path: Path) -> Path:
    absolute_path = Path(os.path.abspath(path))
    if sys.platform != "darwin" or len(absolute_path.parts) < 2:
        return absolute_path
    alias_name = absolute_path.parts[1]
    expected = {
        "tmp": (Path("/private/tmp"), {"private/tmp", "/private/tmp"}),
        "var": (Path("/private/var"), {"private/var", "/private/var"}),
    }.get(alias_name)
    if expected is None:
        return absolute_path
    canonical_prefix, allowed_targets = expected
    alias_path = Path(absolute_path.anchor) / alias_name
    try:
        alias_stat = alias_path.lstat()
        alias_target = os.readlink(alias_path)
    except OSError:
        return absolute_path
    if not stat.S_ISLNK(alias_stat.st_mode) or alias_target not in allowed_targets:
        return absolute_path
    return canonical_prefix.joinpath(*absolute_path.parts[2:])


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


def _require_descriptor_capabilities() -> None:
    missing = [
        name for name in ("O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK") if not getattr(os, name, 0)
    ]
    dir_fd_functions = (os.open, os.mkdir, os.link, os.stat, os.unlink)
    supports_dir_fd = getattr(os, "supports_dir_fd", set())
    missing.extend(
        function.__name__ for function in dir_fd_functions if function not in supports_dir_fd
    )
    supports_follow_symlinks = getattr(os, "supports_follow_symlinks", set())
    if os.stat not in supports_follow_symlinks:
        missing.append("stat(follow_symlinks=False)")
    if os.link not in supports_follow_symlinks:
        missing.append("link(follow_symlinks=False)")
    if missing:
        raise ArtifactBlobContainmentError(
            "immutable blob custody requires descriptor-relative no-follow filesystem "
            "operations; unavailable: " + ", ".join(sorted(set(missing)))
        )


def _directory_flags() -> int:
    _require_descriptor_capabilities()
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)


def _file_flags(*, writable: bool = False) -> int:
    _require_descriptor_capabilities()
    flags = os.O_RDWR if writable else os.O_RDONLY
    return flags | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)


def _open_directory_chain(directory: Path, *, create: bool) -> list[_DirectoryRecord]:
    absolute_directory = Path(os.path.abspath(directory))
    anchor = Path(absolute_directory.anchor)
    if not anchor.anchor:
        raise ArtifactBlobContainmentError(
            f"directory must resolve to an absolute path: {directory}"
        )
    records: list[_DirectoryRecord] = []
    flags = _directory_flags()
    try:
        descriptor = os.open(anchor, flags)
        anchor_stat = os.fstat(descriptor)
        if not stat.S_ISDIR(anchor_stat.st_mode):
            raise ArtifactBlobContainmentError(f"path anchor is not a directory: {anchor}")
        records.append((anchor, descriptor, anchor_stat))
        current_path = anchor
        for component in absolute_directory.parts[1:]:
            current_path = current_path / component
            try:
                next_descriptor = os.open(component, flags, dir_fd=records[-1][1])
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(component, mode=0o777, dir_fd=records[-1][1])
                except FileExistsError:
                    pass
                next_descriptor = os.open(component, flags, dir_fd=records[-1][1])
            next_stat = os.fstat(next_descriptor)
            if not stat.S_ISDIR(next_stat.st_mode):
                os.close(next_descriptor)
                raise ArtifactBlobContainmentError(
                    f"path component is not a directory: {current_path}"
                )
            records.append((current_path, next_descriptor, next_stat))
    except OSError as exc:
        _close_directory_chain(records)
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise ArtifactBlobContainmentError(
                f"directory path traverses a symlink or non-directory: {directory}"
            ) from exc
        raise
    except Exception:
        _close_directory_chain(records)
        raise
    return records


def _recheck_directory_chain(records: list[_DirectoryRecord]) -> None:
    for path, descriptor, initial_stat in records:
        descriptor_stat = os.fstat(descriptor)
        try:
            path_stat = os.stat(path, follow_symlinks=False)
        except FileNotFoundError as exc:
            raise ArtifactBlobContainmentError(
                f"directory disappeared during custody operation: {path}"
            ) from exc
        expected_identity = (initial_stat.st_dev, initial_stat.st_ino)
        if (
            not stat.S_ISDIR(path_stat.st_mode)
            or (descriptor_stat.st_dev, descriptor_stat.st_ino) != expected_identity
            or (path_stat.st_dev, path_stat.st_ino) != expected_identity
        ):
            raise ArtifactBlobContainmentError(
                f"directory identity changed during custody operation: {path}"
            )


def _close_directory_chain(records: list[_DirectoryRecord]) -> None:
    for _, descriptor, _ in reversed(records):
        os.close(descriptor)


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


def _link_materialized_file(
    temporary_name: str,
    destination_name: str,
    *,
    temporary_parent_descriptor: int,
    parent_descriptor: int,
) -> None:
    os.link(
        temporary_name,
        destination_name,
        src_dir_fd=temporary_parent_descriptor,
        dst_dir_fd=parent_descriptor,
        follow_symlinks=False,
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
        directory_descriptor = os.open(
            directory_name,
            _directory_flags(),
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


def _stage_blob_bytes(data: bytes, destination: Path) -> os.stat_result:
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
        _write_file_descriptor(temporary_descriptor, data)
        os.fsync(temporary_descriptor)
        if _read_file_descriptor(temporary_descriptor) != data:
            raise ArtifactBlobIntegrityError(
                f"staged blob bytes failed verification: {destination}"
            )
        try:
            _link_materialized_file(
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
        final_descriptor = os.open(destination.name, _file_flags(), dir_fd=parent_descriptor)
        before = os.fstat(final_descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ArtifactBlobIntegrityError(
                f"canonical blob is not an unaliased regular file: {destination}"
            )
        stored = _read_file_descriptor(final_descriptor)
        after = os.fstat(final_descriptor)
        if _file_state(before) != _file_state(after) or stored != data:
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


@dataclass(frozen=True, slots=True)
class ImmutableArtifactBlobProvider:
    """Explicit local provider for suffixless immutable artifact blobs."""

    root: Path | str
    storage_backend: str = _STORAGE_BACKEND

    def __post_init__(self) -> None:
        root_path = _validate_explicit_root(self.root)
        if self.storage_backend != _STORAGE_BACKEND:
            raise ArtifactBlobReferenceError(
                f"unsupported immutable blob storage backend: {self.storage_backend!r}"
            )
        object.__setattr__(self, "root", _canonicalize_trusted_system_aliases(root_path))

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

        digest = hashlib.sha256(data).hexdigest()
        artifact_id = f"{_ARTIFACT_ID_PREFIX}{digest}"
        self._validate_store_target(self._canonical_path(digest))
        written = _stage_blob_bytes(data, self._canonical_path(digest))
        artifact = ArtifactRef(
            role=role,
            logical_name=logical_name,
            artifact_id=artifact_id,
            sha256=digest,
            size_bytes=written.st_size,
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
                _link_materialized_file(
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

            final_descriptor = os.open(
                destination_name,
                _file_flags(),
                dir_fd=parent_descriptor,
            )
            final_stat_before = os.fstat(final_descriptor)
            if (
                not stat.S_ISREG(final_stat_before.st_mode)
                or final_stat_before.st_nlink != 1
                or (final_stat_before.st_dev, final_stat_before.st_ino) != temporary_identity
                or final_stat_before.st_size != len(data)
            ):
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
            file_descriptor = os.open(
                target.name,
                _file_flags(),
                dir_fd=parent_descriptor,
            )
            file_stat_before = os.fstat(file_descriptor)
            if not stat.S_ISREG(file_stat_before.st_mode):
                raise ArtifactBlobIntegrityError(
                    f"canonical artifact must be a regular file: {target}"
                )
            if file_stat_before.st_nlink != 1:
                raise ArtifactBlobIntegrityError(
                    f"canonical artifact has mutable hard-link aliases: {target}"
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
    "ImmutableArtifactBlobProvider",
    "open_immutable_artifact_blob_provider",
]
