"""Secure local content-addressed artifact storage."""

from __future__ import annotations

from typing import Any

from feedbax.contracts.retention_artifact_schema import (
    RETENTION_ARTIFACT_ROLE_SCHEMAS,
    retained_observables_to_json,
    retention_artifact_schema,
)


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
