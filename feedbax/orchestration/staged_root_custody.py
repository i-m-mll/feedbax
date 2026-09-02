"""Versioned custody for explicit staged execution roots."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Literal

from pydantic import Field, field_validator, model_validator

from feedbax.contracts.artifact_custody import ImmutableArtifactBlobProviderSpec
from feedbax.contracts.base import StrictModel
from feedbax.contracts.staged_execution import (
    STAGED_CHECKPOINT_CUSTODY_BACKEND,
    STAGED_EXECUTION_DESCRIPTOR_SCHEMA_ID,
    STAGED_EXECUTION_DESCRIPTOR_SCHEMA_VERSION,
    StagedCheckpointCustodySpec,
    StagedExecutionDescriptor,
    validate_staged_binding_name,
)

if TYPE_CHECKING:
    from feedbax.analysis.execution_context import (
        StagedArtifactProviderRootBinding,
        StagedCheckpointCustodyRootBinding,
        StagedManifestRootBinding,
    )


STAGED_ROOT_CUSTODY_SCHEMA_ID = "feedbax.orchestration.staged_root_custody"
STAGED_ROOT_CUSTODY_SCHEMA_VERSION = "feedbax.orchestration.staged_root_custody.v1"
STAGED_ROOT_CUSTODY_REF_PREFIX = "staged-root://sha256/"
StagedRootKind = Literal["manifest-store", "artifact-provider", "checkpoint-custody"]


class StagedRootCustodyError(ValueError):
    """Raised when a staged execution root cannot be sealed or authenticated."""


class StagedRootFileRecord(StrictModel):
    """One canonical regular-file member of a sealed staged root."""

    relative_path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)

    @field_validator("relative_path")
    @classmethod
    def _validate_relative_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or "\\" in value
            or "\x00" in value
            or any(part in {"", ".", ".."} for part in value.split("/"))
        ):
            raise ValueError("staged-root members require canonical POSIX relative paths")
        return path.as_posix()


class StagedRootCustody(StrictModel):
    """Durable identity and exact file manifest for one staged execution root."""

    schema_id: Literal["feedbax.orchestration.staged_root_custody"] = STAGED_ROOT_CUSTODY_SCHEMA_ID
    schema_version: Literal["feedbax.orchestration.staged_root_custody.v1"] = (
        STAGED_ROOT_CUSTODY_SCHEMA_VERSION
    )
    binding_name: str = Field(min_length=1)
    root_kind: StagedRootKind
    custody_ref: str = Field(min_length=1)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    directories: tuple[str, ...] = ()
    files: tuple[StagedRootFileRecord, ...]
    artifact_provider: ImmutableArtifactBlobProviderSpec | None = None

    @field_validator("binding_name")
    @classmethod
    def _validate_binding_name(cls, value: str) -> str:
        return validate_staged_binding_name(value)

    @field_validator("directories")
    @classmethod
    def _validate_directories(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(StagedRootFileRecord._validate_relative_path(path) for path in value)
        if normalized != tuple(sorted(normalized)) or len(normalized) != len(set(normalized)):
            raise ValueError("staged-root directories must be unique and canonically ordered")
        return normalized

    @model_validator(mode="after")
    def _validate_identity(self) -> "StagedRootCustody":
        paths = [record.relative_path for record in self.files]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ValueError("staged-root files must be unique and canonically ordered")
        observed = staged_root_content_sha256(
            binding_name=self.binding_name,
            root_kind=self.root_kind,
            directories=self.directories,
            files=self.files,
        )
        if self.content_sha256 != observed:
            raise ValueError("staged-root content_sha256 does not match its file manifest")
        if self.custody_ref != f"{STAGED_ROOT_CUSTODY_REF_PREFIX}{observed}":
            raise ValueError("staged-root custody_ref does not match its content identity")
        if (self.root_kind == "artifact-provider") != (self.artifact_provider is not None):
            raise ValueError("artifact-provider roots require exactly one immutable provider spec")
        return self


@dataclass(frozen=True, slots=True)
class StagedRootSourceBinding:
    """One explicit local root to seal before orchestration acquisition."""

    name: str
    kind: StagedRootKind
    root: Path | str
    artifact_provider: ImmutableArtifactBlobProviderSpec | None = None


@dataclass(frozen=True, slots=True)
class StagedRootSnapshotBinding:
    """One content-addressed sealed root bound for STAGE_INPUTS."""

    name: str
    kind: StagedRootKind
    root: Path | str
    expected_root_identity: tuple[int, int]


@dataclass(frozen=True, slots=True)
class SealedStagedRoot:
    """A source root, its immutable local snapshot, and durable identity."""

    source_root: Path
    staging_root: Path
    custody: StagedRootCustody

    @property
    def binding(self) -> StagedRootSnapshotBinding:
        """Return the exact binding consumed by orchestration drivers."""
        return StagedRootSnapshotBinding(
            name=self.custody.binding_name,
            kind=self.custody.root_kind,
            root=self.staging_root,
            expected_root_identity=_directory_identity(self.staging_root),
        )


@dataclass(frozen=True, slots=True)
class MaterializedStagedRoot:
    """One authenticated staged root at its final execution location."""

    custody: StagedRootCustody
    root: Path


@dataclass(frozen=True, slots=True)
class StagedExecutionRootBindings:
    """Direct arguments for the existing public staged execution context."""

    descriptor: StagedExecutionDescriptor
    artifact_provider_bindings: tuple[StagedArtifactProviderRootBinding, ...]
    manifest_root_bindings: tuple[StagedManifestRootBinding, ...]
    checkpoint_custody_bindings: tuple[StagedCheckpointCustodyRootBinding, ...]


def staged_root_content_sha256(
    *,
    binding_name: str,
    root_kind: StagedRootKind,
    directories: Sequence[str] = (),
    files: Sequence[StagedRootFileRecord],
) -> str:
    """Return the canonical identity of one named, typed staged root."""
    payload = {
        "binding_name": binding_name,
        "root_kind": root_kind,
        "directories": list(directories),
        "files": [record.model_dump(mode="json", exclude_none=True) for record in files],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def seal_staged_root(
    binding: StagedRootSourceBinding,
    *,
    snapshot_parent: Path | str,
) -> SealedStagedRoot:
    """Copy and seal exactly one explicit regular-file root."""
    validate_staged_binding_name(binding.name)
    _validate_root_kind(binding.kind)
    if (binding.kind == "artifact-provider") != (binding.artifact_provider is not None):
        raise StagedRootCustodyError(
            "artifact-provider roots require exactly one immutable provider spec"
        )
    source_root = _canonical_explicit_root(binding.root, kind=binding.kind)
    parent = Path(snapshot_parent).expanduser().resolve()
    parent.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(f"{binding.kind}\0{binding.name}".encode("utf-8")).hexdigest()[:16]
    custody_parent = parent / key
    custody_parent.mkdir(parents=True, exist_ok=True)
    build_root = Path(tempfile.mkdtemp(prefix=".staged-root-", dir=custody_parent))
    published_identity: tuple[int, int] | None = None
    staging_root: Path | None = None
    try:
        directories, records = _copy_root(source_root, build_root)
        digest = staged_root_content_sha256(
            binding_name=binding.name,
            root_kind=binding.kind,
            directories=directories,
            files=records,
        )
        custody = StagedRootCustody(
            binding_name=binding.name,
            root_kind=binding.kind,
            custody_ref=f"{STAGED_ROOT_CUSTODY_REF_PREFIX}{digest}",
            content_sha256=digest,
            directories=directories,
            files=list(records),
            artifact_provider=binding.artifact_provider,
        )
        staging_root = custody_parent / digest
        if staging_root.exists():
            verify_staged_root_snapshot(custody, staging_root)
            _remove_tree(build_root)
        else:
            parent_descriptor = os.open(
                custody_parent,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | os.O_NOFOLLOW,
            )
            try:
                _publish_directory_no_replace(
                    parent_descriptor,
                    build_root.name,
                    staging_root.name,
                    expected_identity=os.stat(
                        build_root.name,
                        dir_fd=parent_descriptor,
                        follow_symlinks=False,
                    ),
                )
                published_identity = _directory_identity(staging_root)
            finally:
                os.close(parent_descriptor)
        _seal_tree(staging_root)
        verify_staged_root_snapshot(custody, staging_root)
        return SealedStagedRoot(source_root, staging_root, custody)
    except Exception:
        if build_root.exists():
            _remove_tree(build_root)
        if (
            published_identity is not None
            and staging_root is not None
            and staging_root.exists()
            and _directory_identity(staging_root) == published_identity
        ):
            _remove_tree(staging_root)
        raise


def verify_staged_root_snapshot(
    custody: StagedRootCustody,
    root: Path | str,
) -> tuple[StagedRootFileRecord, ...]:
    """Fail closed unless a root still contains exactly the sealed bytes."""
    root_path = _canonical_explicit_root(root, kind="sealed staged root")
    observed_directories, observed_files = _read_root(root_path)
    if custody.directories != observed_directories or custody.files != observed_files:
        raise StagedRootCustodyError(
            f"sealed staged root differs from custody manifest: {custody.binding_name!r}"
        )
    digest = staged_root_content_sha256(
        binding_name=custody.binding_name,
        root_kind=custody.root_kind,
        directories=observed_directories,
        files=observed_files,
    )
    if digest != custody.content_sha256:
        raise StagedRootCustodyError(
            f"sealed staged root digest mismatch: {custody.binding_name!r}"
        )
    return observed_files


def verify_staged_root_snapshot_binding(binding: StagedRootSnapshotBinding) -> Path:
    """Require one runtime binding to retain its authenticated directory object."""
    validate_staged_binding_name(binding.name)
    _validate_root_kind(binding.kind)
    root = _canonical_explicit_root(binding.root, kind="sealed staged root")
    if _directory_identity(root) != binding.expected_root_identity:
        raise StagedRootCustodyError(
            f"sealed staged-root binding was replaced: {(binding.kind, binding.name)!r}"
        )
    return root


def materialize_staged_root_snapshot(
    custody: StagedRootCustody,
    source_root: Path | str,
    destination: Path | str,
) -> MaterializedStagedRoot:
    """Authenticate, copy, and atomically publish one staged root."""
    source = _canonical_explicit_root(source_root, kind="sealed staged root")
    verify_staged_root_snapshot(custody, source)
    target = Path(destination).expanduser().resolve()
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(target):
        raise StagedRootCustodyError(f"staged-root destination already exists: {target}")
    build_root = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=parent))
    published_identity: tuple[int, int] | None = None
    try:
        directories, records = _copy_root(source, build_root)
        if directories != custody.directories or records != custody.files:
            raise StagedRootCustodyError(
                f"staged root changed during materialization: {custody.binding_name!r}"
            )
        verify_staged_root_snapshot(custody, source)
        parent_descriptor = os.open(
            parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | os.O_NOFOLLOW,
        )
        try:
            _publish_directory_no_replace(
                parent_descriptor,
                build_root.name,
                target.name,
                expected_identity=os.stat(
                    build_root.name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                ),
            )
            published_identity = _directory_identity(target)
        finally:
            os.close(parent_descriptor)
        _seal_tree(target)
        verify_staged_root_snapshot(custody, target)
        return MaterializedStagedRoot(custody=custody, root=target)
    except Exception:
        if build_root.exists():
            _remove_tree(build_root)
        if (
            published_identity is not None
            and target.exists()
            and _directory_identity(target) == published_identity
        ):
            _remove_tree(target)
        raise


def staged_execution_root_bindings(
    roots: Sequence[MaterializedStagedRoot],
) -> StagedExecutionRootBindings:
    """Project exact materialized roots onto the public execution-context API."""
    from feedbax.analysis.execution_context import (
        StagedArtifactProviderRootBinding,
        StagedCheckpointCustodyRootBinding,
        StagedManifestRootBinding,
    )

    ordered = sorted(
        roots,
        key=lambda item: (item.custody.root_kind, item.custody.binding_name),
    )
    artifact_specs: dict[str, ImmutableArtifactBlobProviderSpec] = {}
    checkpoint_specs: dict[str, StagedCheckpointCustodySpec] = {}
    artifact_bindings: list[StagedArtifactProviderRootBinding] = []
    manifest_bindings: list[StagedManifestRootBinding] = []
    checkpoint_bindings: list[StagedCheckpointCustodyRootBinding] = []
    seen: set[tuple[str, str]] = set()
    for item in ordered:
        custody = item.custody
        key = (custody.root_kind, custody.binding_name)
        if key in seen:
            raise StagedRootCustodyError(f"duplicate materialized staged root: {key!r}")
        seen.add(key)
        if custody.root_kind == "artifact-provider":
            assert custody.artifact_provider is not None
            artifact_specs[custody.binding_name] = custody.artifact_provider
            artifact_bindings.append(
                StagedArtifactProviderRootBinding(custody.binding_name, item.root)
            )
        elif custody.root_kind == "manifest-store":
            manifest_bindings.append(StagedManifestRootBinding(custody.binding_name, item.root))
        else:
            checkpoint_specs[custody.binding_name] = StagedCheckpointCustodySpec(
                backend=STAGED_CHECKPOINT_CUSTODY_BACKEND
            )
            checkpoint_bindings.append(
                StagedCheckpointCustodyRootBinding(custody.binding_name, item.root)
            )
    return StagedExecutionRootBindings(
        descriptor=StagedExecutionDescriptor(
            schema_id=STAGED_EXECUTION_DESCRIPTOR_SCHEMA_ID,
            schema_version=STAGED_EXECUTION_DESCRIPTOR_SCHEMA_VERSION,
            artifact_providers=artifact_specs,
            checkpoint_custody=checkpoint_specs,
        ),
        artifact_provider_bindings=tuple(artifact_bindings),
        manifest_root_bindings=tuple(manifest_bindings),
        checkpoint_custody_bindings=tuple(checkpoint_bindings),
    )


def _canonical_explicit_root(root: Path | str, *, kind: str) -> Path:
    supplied = Path(root).expanduser()
    if not supplied.is_absolute():
        raise StagedRootCustodyError(f"{kind} root must be absolute")
    lexical = Path(os.path.abspath(supplied))
    try:
        resolved = supplied.resolve(strict=True)
        root_stat = supplied.stat(follow_symlinks=False)
    except OSError as exc:
        raise StagedRootCustodyError(f"{kind} root is unavailable") from exc
    if lexical != resolved or not stat.S_ISDIR(root_stat.st_mode):
        raise StagedRootCustodyError(f"{kind} root must be a canonical non-symlink directory")
    return resolved


def _copy_root(
    source: Path,
    destination: Path,
) -> tuple[tuple[str, ...], tuple[StagedRootFileRecord, ...]]:
    directories: list[str] = []
    records: list[StagedRootFileRecord] = []
    root_descriptor = _open_directory(source)
    opened_before = os.fstat(root_descriptor)
    try:
        _copy_directory(root_descriptor, destination, (), directories, records)
        root_after = os.stat(source, follow_symlinks=False)
        opened_after = os.fstat(root_descriptor)
        if (
            _directory_state(root_after) != _directory_state(opened_before)
            or _directory_state(opened_after) != _directory_state(opened_before)
        ):
            raise StagedRootCustodyError("staged-root identity changed during snapshot")
    finally:
        os.close(root_descriptor)
    directories.sort()
    records.sort(key=lambda record: record.relative_path)
    return tuple(directories), tuple(records)


def _copy_directory(
    directory_descriptor: int,
    destination: Path,
    relative: tuple[str, ...],
    directories: list[str],
    records: list[StagedRootFileRecord],
) -> None:
    children = sorted(os.scandir(directory_descriptor), key=lambda entry: os.fsencode(entry.name))
    for child in children:
        name = child.name
        before = child.stat(follow_symlinks=False)
        target = destination.joinpath(*relative, name)
        if stat.S_ISDIR(before.st_mode):
            target.mkdir(mode=0o700)
            directories.append(PurePosixPath(*relative, name).as_posix())
            child_descriptor = os.open(
                name,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | os.O_NOFOLLOW,
                dir_fd=directory_descriptor,
            )
            try:
                opened = os.fstat(child_descriptor)
                if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                    raise StagedRootCustodyError("staged-root directory changed during snapshot")
                _copy_directory(
                    child_descriptor,
                    destination,
                    (*relative, name),
                    directories,
                    records,
                )
                completed = os.fstat(child_descriptor)
                current = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
                if (
                    _directory_state(completed) != _directory_state(before)
                    or _directory_state(current) != _directory_state(before)
                ):
                    raise StagedRootCustodyError(
                        "staged-root directory changed during snapshot"
                    )
            finally:
                os.close(child_descriptor)
        elif stat.S_ISREG(before.st_mode):
            if before.st_nlink != 1:
                raise StagedRootCustodyError("staged-root files must have one hard link")
            data = _read_regular_file(directory_descriptor, name, before)
            target.write_bytes(data)
            target.chmod(0o400)
            records.append(
                StagedRootFileRecord(
                    relative_path=PurePosixPath(*relative, name).as_posix(),
                    sha256=hashlib.sha256(data).hexdigest(),
                    size_bytes=len(data),
                )
            )
        else:
            raise StagedRootCustodyError(
                f"staged-root contains a symlink or unsupported member: {name!r}"
            )


def _read_root(
    root: Path,
) -> tuple[tuple[str, ...], tuple[StagedRootFileRecord, ...]]:
    temporary = Path(tempfile.mkdtemp(prefix=".staged-root-verify-", dir=root.parent))
    try:
        return _copy_root(root, temporary)
    finally:
        _remove_tree(temporary)


def _read_regular_file(
    parent_descriptor: int,
    name: str,
    before: os.stat_result,
) -> bytes:
    descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_descriptor)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise StagedRootCustodyError("staged-root file changed before read")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
        current = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        before_state = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        after_state = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        current_state = (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
            current.st_ctime_ns,
        )
        if before_state != after_state or after_state != current_state:
            raise StagedRootCustodyError("staged-root file changed during read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _open_directory(root: Path) -> int:
    try:
        return os.open(
            root,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | os.O_NOFOLLOW,
        )
    except OSError as exc:
        raise StagedRootCustodyError("staged-root directory is unsafe or unavailable") from exc


def _directory_identity(root: Path) -> tuple[int, int]:
    observed = root.stat(follow_symlinks=False)
    return observed.st_dev, observed.st_ino


def _directory_state(observed: os.stat_result) -> tuple[int, int, int, int]:
    return (
        observed.st_dev,
        observed.st_ino,
        observed.st_mtime_ns,
        observed.st_ctime_ns,
    )


def _validate_root_kind(kind: str) -> None:
    if kind not in {"manifest-store", "artifact-provider", "checkpoint-custody"}:
        raise StagedRootCustodyError(f"unsupported staged-root kind: {kind!r}")


def _publish_directory_no_replace(
    parent_descriptor: int,
    source_name: str,
    destination_name: str,
    *,
    expected_identity: os.stat_result,
) -> None:
    from feedbax.training.checkpoint_custody import publish_directory_no_replace

    publish_directory_no_replace(
        parent_descriptor,
        source_name,
        destination_name,
        expected_identity=expected_identity,
    )


def _seal_tree(root: Path) -> None:
    for directory, _subdirs, files in os.walk(root):
        directory_path = Path(directory)
        for filename in files:
            (directory_path / filename).chmod(0o400)
        directory_path.chmod(0o500)


def _remove_tree(root: Path) -> None:
    if not root.exists():
        return
    for directory, subdirs, files in os.walk(root):
        directory_path = Path(directory)
        directory_path.chmod(0o700)
        for name in [*subdirs, *files]:
            path = directory_path / name
            if not path.is_symlink():
                path.chmod(0o700 if path.is_dir() else 0o600)
    shutil.rmtree(root)


__all__ = [
    "MaterializedStagedRoot",
    "STAGED_ROOT_CUSTODY_REF_PREFIX",
    "STAGED_ROOT_CUSTODY_SCHEMA_ID",
    "STAGED_ROOT_CUSTODY_SCHEMA_VERSION",
    "SealedStagedRoot",
    "StagedExecutionRootBindings",
    "StagedRootCustody",
    "StagedRootCustodyError",
    "StagedRootFileRecord",
    "StagedRootKind",
    "StagedRootSnapshotBinding",
    "StagedRootSourceBinding",
    "materialize_staged_root_snapshot",
    "seal_staged_root",
    "staged_execution_root_bindings",
    "staged_root_content_sha256",
    "verify_staged_root_snapshot",
    "verify_staged_root_snapshot_binding",
]
