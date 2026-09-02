"""Secure local content-addressed artifact storage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

from feedbax.contracts.base import (
    ArtifactRef,
    default_manifest_root,
    sha256_bytes,
)
from feedbax.contracts.retention_artifact_schema import (
    RETENTION_ARTIFACT_ROLE_SCHEMAS,
    retained_observables_to_json,
    retention_artifact_schema,
)


def _artifact_path(root: Path, digest: str) -> Path:
    return root / "artifacts" / "sha256" / digest[:2] / digest


_ARTIFACT_STREAM_CHUNK_BYTES = 1024 * 1024


def _file_content_identity(path: Path) -> tuple[str, int]:
    """Return the streamed ``(sha256, size_bytes)`` identity of a file."""
    digest = hashlib.sha256()
    size_bytes = 0
    with Path(path).open("rb") as stream:
        while chunk := stream.read(_ARTIFACT_STREAM_CHUNK_BYTES):
            digest.update(chunk)
            size_bytes += len(chunk)
    return digest.hexdigest(), size_bytes


DEFAULT_ARTIFACT_MEDIA_TYPE = "application/octet-stream"

ARTIFACT_MEDIA_TYPES_BY_EXTENSION: dict[str, str] = {
    "html": "text/html",
    "json": "application/json",
    "svg": "image/svg+xml",
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "pdf": "application/pdf",
    "npz": "application/x-npz",
}


def media_type_for_extension(
    extension: str,
    *,
    default: str = DEFAULT_ARTIFACT_MEDIA_TYPE,
) -> str:
    """Return the artifact media type registered for a file extension.

    This is the single source for extension-to-media-type mapping used when storing
    artifacts. The extension may be given with or without a leading dot and in any case.

    Args:
        extension: File extension such as `"png"`, `".PNG"`, or a `Path.suffix` value.
        default: Media type returned for an unregistered extension.
    """
    return ARTIFACT_MEDIA_TYPES_BY_EXTENSION.get(extension.lstrip(".").lower(), default)


def store_artifact(
    source_path: Path | str,
    *,
    root: Path | str | None = None,
    role: str,
    logical_name: Optional[str] = None,
    media_type: str = "application/octet-stream",
    metadata: Optional[dict[str, Any]] = None,
) -> ArtifactRef:
    """Copy an artifact into the local content-addressed store and return its ref.

    The published canonical bytes are read back and verified against the source
    content identity before the reference is returned, so the returned digest and
    size always describe the bytes actually stored. A source that changes during
    the copy, or a canonical destination that already holds different bytes,
    fails closed without overwriting the existing canonical file.
    """
    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(source)
    root_path = Path(root) if root is not None else default_manifest_root()
    expected_identity = _file_content_identity(source)
    data = source.read_bytes()
    if (sha256_bytes(data), len(data)) != expected_identity:
        from feedbax.persistence.artifact_custody import ArtifactBlobIntegrityError

        raise ArtifactBlobIntegrityError(f"artifact source bytes changed during store: {source}")
    artifact_metadata = dict(metadata or {})
    artifact_metadata.setdefault("original_uri", str(source))
    return store_bytes_artifact(
        data,
        root=root_path,
        role=role,
        logical_name=logical_name or source.name,
        media_type=media_type,
        metadata=artifact_metadata,
    )


def store_json_artifact(
    value: Any,
    *,
    root: Path | str | None = None,
    role: str,
    logical_name: str,
    metadata: Optional[dict[str, Any]] = None,
) -> ArtifactRef:
    """Write stable JSON into the local content-addressed store.

    The serialized bytes are published through the same verified byte store as
    :func:`store_bytes_artifact`, so the canonical file is read back and compared
    against the serialized payload, including when the canonical name already exists.
    """
    data = json.dumps(value, indent=2, sort_keys=True).encode() + b"\n"
    return store_bytes_artifact(
        data,
        root=root,
        role=role,
        logical_name=logical_name,
        media_type="application/json",
        metadata=metadata,
    )


def store_bytes_artifact(
    data: bytes,
    *,
    root: Path | str | None = None,
    role: str,
    logical_name: str,
    media_type: str = "application/octet-stream",
    metadata: Optional[dict[str, Any]] = None,
) -> ArtifactRef:
    """Atomically write opaque bytes into the local content-addressed store.

    The canonical name is published only after the exact temporary bytes are
    flushed and verified. Platforms without descriptor-relative, no-follow
    operations fail closed at the common BlobStore boundary.
    """
    if not isinstance(data, bytes):
        raise TypeError("artifact data must be bytes")
    from feedbax.persistence.publication import LocalBlobStore

    root_path = Path(root) if root is not None else default_manifest_root()
    blob = LocalBlobStore(Path(root_path).absolute()).stage(data)
    dest = _artifact_path(root_path, blob.digest)
    artifact_metadata = dict(metadata or {})
    artifact_metadata.setdefault("relative_path", str(dest.relative_to(root_path)))
    return ArtifactRef(
        role=role,
        logical_name=logical_name,
        artifact_id=f"artifact://sha256/{blob.digest}",
        sha256=blob.digest,
        media_type=media_type,
        size_bytes=blob.size_bytes,
        uri=str(dest),
        metadata=artifact_metadata,
    )


def _validate_retention_artifact_version(
    role: str,
    payload: dict[str, Any],
    *,
    path: str,
) -> dict[str, Any]:
    """Validate or stamp a governed retention artifact payload."""
    from feedbax.contracts.migrations import UnsupportedSpecVersion, default_spec_registry

    kind, expected_schema_id, current_version = retention_artifact_schema(role)
    schema_id = payload.get("schema_id")
    if schema_id is not None and schema_id != expected_schema_id:
        raise UnsupportedSpecVersion(
            "Unsupported retention artifact schema identity: "
            f"path={path!r}, role={role!r}, kind={kind!r}, "
            f"schema_id={schema_id!r}, expected={expected_schema_id!r}"
        )

    source_version = payload.get("schema_version")
    if source_version is not None and not isinstance(source_version, str):
        raise UnsupportedSpecVersion(
            "Retention artifact schema_version must be a string: "
            f"path={path!r}, role={role!r}, kind={kind!r}, "
            f"schema_version={source_version!r}"
        )
    if isinstance(source_version, str) and source_version and source_version != current_version:
        try:
            default_spec_registry.migrate(kind, payload, source_version=source_version)
        except UnsupportedSpecVersion as exc:
            raise UnsupportedSpecVersion(
                "Unsupported retention artifact schema version: "
                f"path={path!r}, role={role!r}, kind={kind!r}; {exc}"
            ) from exc

    stamped = dict(payload)
    stamped["schema_id"] = expected_schema_id
    stamped["schema_version"] = current_version
    return stamped


def _retention_artifact_payload(
    role: str,
    value: Any,
    *,
    path: str,
) -> dict[str, Any]:
    if role == "retained_observables":
        if (
            isinstance(value, dict)
            and ("schema_id" in value or "schema_version" in value)
            and "observables" in value
        ):
            payload = dict(value)
        else:
            payload = retained_observables_to_json(value)
    elif role == "retention_plan":
        if not isinstance(value, dict):
            raise TypeError(
                "retention_plan artifact payload must be a mapping: "
                f"path={path!r}, got={type(value).__name__}"
            )
        payload = dict(value)
    else:
        payload = value
    if not isinstance(payload, dict):
        raise TypeError(
            "retention artifact payload must be a mapping after schema wrapping: "
            f"path={path!r}, role={role!r}, got={type(payload).__name__}"
        )
    return _validate_retention_artifact_version(role, payload, path=path)


def _validate_retention_artifact_ref_metadata(data: dict[str, Any]) -> dict[str, Any]:
    artifacts = data.get("artifacts")
    if not isinstance(artifacts, list):
        return data
    normalized = dict(data)
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict):
            continue
        role = artifact.get("role")
        if role not in RETENTION_ARTIFACT_ROLE_SCHEMAS:
            continue
        metadata = artifact.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        missing = [
            key
            for key in ("schema_id", "schema_version")
            if not isinstance(metadata.get(key), str) or not metadata.get(key)
        ]
        if missing:
            from feedbax.contracts.migrations import UnsupportedSpecVersion

            raise UnsupportedSpecVersion(
                "Retention artifact ref is missing governed schema metadata: "
                f"path='artifacts/{index}/metadata', role={role!r}, missing={missing}"
            )
        _validate_retention_artifact_version(
            role,
            {
                "schema_id": metadata["schema_id"],
                "schema_version": metadata["schema_version"],
            },
            path=f"artifacts/{index}/metadata",
        )
    return normalized
