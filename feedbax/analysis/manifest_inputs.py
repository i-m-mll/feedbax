"""Portable authentication and local resolution for staged manifest inputs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path, PurePosixPath
import stat
from urllib.parse import unquote, urlsplit

from feedbax.contracts.manifest import (
    AnyManifest,
    ParentRef,
    load_manifest_bytes,
    safe_manifest_key,
)


AUTHENTICATED_MANIFEST_REF_SCHEMA_ID = "feedbax.ref.authenticated_manifest"
AUTHENTICATED_MANIFEST_REF_SCHEMA_VERSION = "feedbax.ref.authenticated_manifest.v1"

_PROFILE_DISCRIMINATORS = frozenset({"ref_schema_id", "ref_schema_version"})
_PROFILE_KEYS = _PROFILE_DISCRIMINATORS | {"manifest_sha256", "size_bytes"}
_MANIFEST_DIRECTORIES = {
    "EvaluationRunManifest": "evaluation_runs",
    "AnalysisRunManifest": "analysis_runs",
    "FigureManifest": "FigureManifest",
    "ReportManifest": "reports",
}


@dataclass(frozen=True)
class ResolvedManifestInput:
    """Authenticated manifest bytes resolved at one machine-local path."""

    ref: ParentRef
    manifest: AnyManifest
    path: Path
    raw_bytes: bytes


def _authenticated_profile(ref: ParentRef) -> tuple[str, int] | None:
    discriminators = _PROFILE_DISCRIMINATORS.intersection(ref.metadata)
    if not discriminators:
        return None
    present = _PROFILE_KEYS.intersection(ref.metadata)
    if present != _PROFILE_KEYS:
        missing = ", ".join(sorted(_PROFILE_KEYS - present))
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


def is_authenticated_manifest_ref(ref: ParentRef) -> bool:
    """Return whether *ref* declares the complete supported authentication profile.

    Partial and unknown profiles raise instead of silently degrading to legacy lookup.
    """

    return _authenticated_profile(ref) is not None


def authenticated_manifest_ref(
    manifest: AnyManifest,
    path: Path | str,
    role: str,
) -> ParentRef:
    """Create a portable ref authenticating the exact final bytes at *path*."""

    manifest_path = Path(path)
    raw = manifest_path.read_bytes()
    parsed = load_manifest_bytes(raw)
    if parsed.kind != manifest.kind or parsed.id != manifest.id:
        raise ValueError(
            f"Manifest bytes at {manifest_path} do not match {manifest.kind} {manifest.id!r}"
        )
    return ParentRef(
        kind=manifest.kind,
        id=manifest.id,
        role=role,
        uri=None,
        metadata={
            "ref_schema_id": AUTHENTICATED_MANIFEST_REF_SCHEMA_ID,
            "ref_schema_version": AUTHENTICATED_MANIFEST_REF_SCHEMA_VERSION,
            "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
        },
    )


def _canonical_locator(ref: ParentRef) -> str:
    try:
        directory = _MANIFEST_DIRECTORIES[ref.kind]
    except KeyError as exc:
        raise ValueError(
            f"Authenticated staged manifest kind {ref.kind!r} is not supported"
        ) from exc
    return f"manifests/{directory}/{safe_manifest_key(ref.id)}.json"


def _locator_parts(locator: Path | str) -> tuple[str, ...]:
    value = os.fspath(locator)
    if not value or "\\" in value:
        raise ValueError(f"Invalid manifest runtime locator: {value!r}")
    split = urlsplit(value)
    if split.scheme or split.netloc or split.query or split.fragment:
        raise ValueError(f"Manifest runtime locator must be a plain relative path: {value!r}")
    decoded = unquote(value)
    if decoded != value and ("\\" in decoded or "?" in decoded or "#" in decoded):
        raise ValueError(f"Invalid encoded manifest runtime locator: {value!r}")
    path = PurePosixPath(decoded)
    if path.is_absolute() or not path.parts:
        raise ValueError(f"Manifest runtime locator must be non-empty and relative: {value!r}")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Manifest runtime locator escapes its root: {value!r}")
    return tuple(path.parts)


def _read_regular_file(root: Path, parts: tuple[str, ...]) -> bytes:
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    file_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptors: list[int] = []
    try:
        current = os.open(root, directory_flags)
        descriptors.append(current)
        for component in parts[:-1]:
            current = os.open(component, directory_flags, dir_fd=current)
            descriptors.append(current)
        file_descriptor = os.open(parts[-1], file_flags, dir_fd=current)
        descriptors.append(file_descriptor)
        file_stat = os.fstat(file_descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise ValueError("Authenticated manifest locator does not name a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(file_descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Authenticated manifest product is missing: {'/'.join(parts)}"
        ) from exc
    except OSError as exc:
        raise ValueError(f"Authenticated manifest locator is unsafe: {'/'.join(parts)}") from exc
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def resolve_manifest_input(
    ref: ParentRef,
    manifest_root: Path | str,
    runtime_locator: Path | str | None = None,
) -> ResolvedManifestInput:
    """Resolve and authenticate one staged manifest ref before downstream effects."""

    profile = _authenticated_profile(ref)
    if profile is None:
        raise ValueError(f"Manifest ref {ref.id!r} makes no authenticated claim")
    digest, expected_size = profile
    parts = _locator_parts(
        runtime_locator if runtime_locator is not None else _canonical_locator(ref)
    )
    root = Path(manifest_root)
    raw = _read_regular_file(root, parts)
    if len(raw) != expected_size:
        raise ValueError(f"Authenticated manifest size mismatch for {ref.id!r}")
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f"Authenticated manifest SHA-256 mismatch for {ref.id!r}")
    manifest = load_manifest_bytes(raw)
    if manifest.kind != ref.kind:
        raise ValueError(
            f"Authenticated manifest kind mismatch: expected {ref.kind!r}, got {manifest.kind!r}"
        )
    if manifest.id != ref.id:
        raise ValueError(
            f"Authenticated manifest id mismatch: expected {ref.id!r}, got {manifest.id!r}"
        )
    return ResolvedManifestInput(
        ref=ref,
        manifest=manifest,
        path=root.joinpath(*parts),
        raw_bytes=raw,
    )
