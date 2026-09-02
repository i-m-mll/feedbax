"""Shared contract models, references, provenance, and content hashing."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:  # pragma: no cover - Python 3.12 always has importlib.metadata.
    PackageNotFoundError = Exception  # type: ignore[assignment]
    version = None  # type: ignore[assignment]


DEFAULT_MANIFEST_ROOT_ENV = "FEEDBAX_RUNS_DIR"
AUTHENTICATED_MANIFEST_REF_SCHEMA_ID = "feedbax.ref.authenticated_manifest"
AUTHENTICATED_MANIFEST_REF_SCHEMA_VERSION = "feedbax.ref.authenticated_manifest.v1"

_AUTHENTICATED_MANIFEST_REF_PROFILE_DISCRIMINATORS = frozenset(
    {"ref_schema_id", "ref_schema_version"}
)
_AUTHENTICATED_MANIFEST_REF_PROFILE_KEYS = _AUTHENTICATED_MANIFEST_REF_PROFILE_DISCRIMINATORS | {
    "manifest_sha256",
    "size_bytes",
}


def feedbax_version() -> str:
    """Return the installed Feedbax package version, or a useful local fallback."""
    if version is None:
        return "unknown"
    try:
        return version("feedbax")
    except PackageNotFoundError:
        return "unknown"


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp with stable second precision."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def default_manifest_root() -> Path:
    """Return the root directory for local manifests and artifacts."""
    configured = os.environ.get(DEFAULT_MANIFEST_ROOT_ENV)
    if configured:
        return Path(configured).expanduser()
    return Path.cwd() / "feedbax_runs"


class StrictModel(BaseModel):
    """Base model for provider-contract records."""

    model_config = ConfigDict(extra="forbid")


class ArtifactRef(StrictModel):
    """Reference to a large output artifact stored outside a manifest."""

    role: str
    logical_name: str
    artifact_id: Optional[str] = None
    sha256: Optional[str] = None
    media_type: str = "application/octet-stream"
    size_bytes: Optional[int] = None
    storage_backend: str = "feedbax-local"
    uri: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ArrayStoreRef(StrictModel):
    """Reference to a role-addressed parameter/state array store."""

    role: Literal["params", "state", "optimizer", "history"]
    schema_version: str
    storage_backend: str
    logical_name: str
    artifact_id: Optional[str] = None
    sha256: Optional[str] = None
    uri: Optional[str] = None
    array_count: int
    roles: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ArtifactValidationRecord(StrictModel):
    """Validation outcome for a durable artifact or migration step."""

    name: str
    status: Literal["passed", "failed", "warning"]
    checked_at: datetime = Field(default_factory=utc_now)
    schema_version: Optional[str] = None
    details: dict[str, Any] = Field(default_factory=dict)


class ArtifactMigrationRecord(StrictModel):
    """Provenance for a schema-to-schema artifact migration."""

    migration_id: str
    source_schema_version: str
    target_schema_version: str
    applied_at: datetime = Field(default_factory=utc_now)
    tool: str = "feedbax"
    deterministic: bool = True
    validation: list[ArtifactValidationRecord] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EntrypointRef(StrictModel):
    """How a manifest-producing operation was invoked."""

    kind: str
    command: Optional[str] = None
    name: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParentRef(StrictModel):
    """Reference to an input spec, parent manifest, or parent artifact."""

    kind: str
    id: str
    role: Optional[str] = None
    uri: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def authenticated_manifest_ref_metadata(digest: str, size_bytes: int) -> dict[str, Any]:
    """Return the ref metadata one ``(sha256, size)`` custody profile authenticates.

    This is the producer half of :func:`authenticated_manifest_ref_profile`: a
    caller that already holds an authenticated profile — because a custody
    document or a compile lock stated it — states it as ref metadata here rather
    than assembling the four keys itself, so a producer can never emit a profile
    the reader would refuse.

    Raises:
        ValueError: The digest is not a lowercase SHA-256 or the size is not a
            non-negative integer. An invalid profile is refused at the producer
            rather than written and refused later.
    """
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"authenticated manifest ref digest {digest!r} is not a SHA-256")
    if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0:
        raise ValueError(f"authenticated manifest ref size {size_bytes!r} is not a byte count")
    return {
        "ref_schema_id": AUTHENTICATED_MANIFEST_REF_SCHEMA_ID,
        "ref_schema_version": AUTHENTICATED_MANIFEST_REF_SCHEMA_VERSION,
        "manifest_sha256": digest,
        "size_bytes": size_bytes,
    }


def authenticated_manifest_ref_profile(ref: ParentRef) -> tuple[str, int] | None:
    """Return one ref's authenticated byte profile, if it declares one.

    Partial or unsupported authenticated profiles raise rather than degrading to
    an unauthenticated manifest reference.
    """

    discriminators = _AUTHENTICATED_MANIFEST_REF_PROFILE_DISCRIMINATORS.intersection(ref.metadata)
    if not discriminators:
        return None
    present = _AUTHENTICATED_MANIFEST_REF_PROFILE_KEYS.intersection(ref.metadata)
    if present != _AUTHENTICATED_MANIFEST_REF_PROFILE_KEYS:
        missing = ", ".join(sorted(_AUTHENTICATED_MANIFEST_REF_PROFILE_KEYS - present))
        raise ValueError(f"Authenticated manifest ref {ref.id!r} is incomplete: {missing}")
    schema_id = ref.metadata["ref_schema_id"]
    schema_version = ref.metadata["ref_schema_version"]
    digest = ref.metadata["manifest_sha256"]
    size = ref.metadata["size_bytes"]
    if schema_id != AUTHENTICATED_MANIFEST_REF_SCHEMA_ID:
        raise ValueError(f"Unsupported authenticated manifest ref schema_id: {schema_id!r}")
    if schema_version != AUTHENTICATED_MANIFEST_REF_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported authenticated manifest ref schema_version: {schema_version!r}"
        )
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"Authenticated manifest ref {ref.id!r} has invalid SHA-256")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError(f"Authenticated manifest ref {ref.id!r} has invalid byte size")
    if ref.uri is not None:
        raise ValueError("Authenticated manifest refs must keep machine-local locators out of uri")
    return digest, size


class Provenance(StrictModel):
    """Shared provenance fields recorded on durable manifests."""

    source_repo: Optional[str] = None
    source_branch: Optional[str] = None
    source_commit: Optional[str] = None
    dirty: Optional[bool] = None
    entrypoint: Optional[EntrypointRef] = None
    issues: list[str] = Field(default_factory=list)
    parents: list[ParentRef] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class FileHashRef(StrictModel):
    """Deterministic content hash for one source or artifact file."""

    path: str
    sha256: str
    size_bytes: int
    role: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TreeHashEntry(StrictModel):
    """One file entry included in a deterministic tree hash."""

    path: str
    sha256: str
    size_bytes: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class TreeHashRef(StrictModel):
    """Deterministic hash for a directory tree and its member file hashes."""

    path: str
    sha256: str
    file_count: int
    total_size_bytes: int
    files: list[TreeHashEntry] = Field(default_factory=list)
    role: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a value using stable JSON for hashing."""
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", exclude_none=True)
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_hash_ref(
    path: Path | str,
    *,
    root: Path | str | None = None,
    role: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> FileHashRef:
    """Return a deterministic hash reference for one file."""
    file_path = Path(path)
    display_path = str(file_path if root is None else file_path.relative_to(Path(root)))
    stat = file_path.stat()
    return FileHashRef(
        path=display_path,
        sha256=sha256_file(file_path),
        size_bytes=stat.st_size,
        role=role,
        metadata=dict(metadata or {}),
    )


def tree_hash_ref(
    path: Path | str,
    *,
    root: Path | str | None = None,
    role: Optional[str] = None,
    include_files: bool = True,
    metadata: Optional[dict[str, Any]] = None,
) -> TreeHashRef:
    """Return a deterministic hash reference for regular files under a directory."""
    tree_path = Path(path)
    if not tree_path.is_dir():
        raise NotADirectoryError(tree_path)

    entries: list[TreeHashEntry] = []
    total_size = 0
    for file_path in sorted(candidate for candidate in tree_path.rglob("*") if candidate.is_file()):
        relative_path = str(file_path.relative_to(tree_path))
        stat = file_path.stat()
        total_size += stat.st_size
        entries.append(
            TreeHashEntry(
                path=relative_path,
                sha256=sha256_file(file_path),
                size_bytes=stat.st_size,
            )
        )
    digest_payload = [entry.model_dump(mode="json", exclude_none=True) for entry in entries]
    display_path = str(tree_path if root is None else tree_path.relative_to(Path(root)))
    return TreeHashRef(
        path=display_path,
        sha256=sha256_bytes(canonical_json_bytes(digest_payload)),
        file_count=len(entries),
        total_size_bytes=total_size,
        files=entries if include_files else [],
        role=role,
        metadata=dict(metadata or {}),
    )


def collect_git_provenance(cwd: Path | str | None = None) -> Provenance:
    """Collect best-effort local Git provenance without mutating repository state."""
    repo_cwd = Path(cwd) if cwd is not None else Path.cwd()

    def _git(*args: str) -> Optional[str]:
        try:
            proc = subprocess.run(
                ["git", *args],
                cwd=repo_cwd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return proc.stdout.strip() or None

    status = _git("status", "--porcelain")
    return Provenance(
        source_repo=_git("config", "--get", "remote.origin.url"),
        source_branch=_git("rev-parse", "--abbrev-ref", "HEAD"),
        source_commit=_git("rev-parse", "HEAD"),
        dirty=(bool(status) if status is not None else None),
    )
