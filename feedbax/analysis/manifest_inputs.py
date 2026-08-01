"""Portable authentication and local resolution for staged manifest inputs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path, PurePosixPath
import stat
from urllib.parse import unquote, urlsplit

from feedbax.contracts.manifest import (
    AUTHENTICATED_MANIFEST_REF_SCHEMA_ID,
    AUTHENTICATED_MANIFEST_REF_SCHEMA_VERSION,
    AnyManifest,
    ParentRef,
    authenticated_manifest_ref_profile,
    canonical_manifest_relative_path,
    load_manifest_bytes,
)


#: Manifest kinds admitted as authenticated staged inputs. Their durable
#: locations come from the canonical layout, never from a local copy of it.
_STAGED_MANIFEST_KINDS = frozenset(
    {
        "EvaluationRunManifest",
        "AnalysisRunManifest",
        "FigureManifest",
        "ReportManifest",
    }
)


@dataclass(frozen=True)
class ResolvedManifestInput:
    """Authenticated manifest bytes resolved at one machine-local path."""

    ref: ParentRef
    manifest: AnyManifest
    path: Path
    raw_bytes: bytes


def is_authenticated_manifest_ref(ref: ParentRef) -> bool:
    """Return whether *ref* declares the complete supported authentication profile.

    Partial and unknown profiles raise instead of silently degrading to legacy lookup.
    """

    return authenticated_manifest_ref_profile(ref) is not None


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
    if ref.kind not in _STAGED_MANIFEST_KINDS:
        raise ValueError(f"Authenticated staged manifest kind {ref.kind!r} is not supported")
    return canonical_manifest_relative_path(ref.kind, ref.id)


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
        if file_stat.st_nlink != 1:
            raise ValueError("Authenticated manifest locator must have exactly one hard link")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(file_descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        final_stat = os.fstat(file_descriptor)
        if final_stat.st_nlink != 1:
            raise ValueError("Authenticated manifest locator hard-link count changed during read")
        if (
            final_stat.st_dev,
            final_stat.st_ino,
            final_stat.st_size,
            final_stat.st_mtime_ns,
            final_stat.st_ctime_ns,
        ) != (
            file_stat.st_dev,
            file_stat.st_ino,
            file_stat.st_size,
            file_stat.st_mtime_ns,
            file_stat.st_ctime_ns,
        ):
            raise ValueError("Authenticated manifest identity changed during read")
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

    profile = authenticated_manifest_ref_profile(ref)
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


def resolve_contained_manifest_input(
    ref: ParentRef,
    manifest_root: Path | str,
    runtime_locator: Path | str,
) -> ResolvedManifestInput:
    """Resolve an explicitly located provider-free manifest within one supplied root."""

    parts = _locator_parts(runtime_locator)
    root = Path(manifest_root)
    raw = _read_regular_file(root, parts)
    manifest = load_manifest_bytes(raw)
    if manifest.kind != ref.kind or manifest.id != ref.id:
        raise ValueError("Contained manifest kind or id disagrees with ParentRef")
    return ResolvedManifestInput(
        ref=ref,
        manifest=manifest,
        path=root.joinpath(*parts),
        raw_bytes=raw,
    )
