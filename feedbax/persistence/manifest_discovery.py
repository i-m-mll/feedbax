"""Resolve manifest receipts from their canonical files."""

from __future__ import annotations

import os.path
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from feedbax.contracts.base import default_manifest_root
from feedbax.contracts.manifest import (
    MANIFEST_KIND_DIRECTORIES,
    AnyManifest,
    canonical_manifest_candidate_paths,
    load_manifest,
)
from feedbax.contracts.selection import ManifestPredicate
from feedbax.persistence.manifest_index import find_manifest_paths_by_id, iter_manifest_files


class ManifestLocationError(ValueError):
    """Raised when a manifest candidate's bytes contradict where they are stored.

    A locator that cannot be resolved and bytes filed under another kind's
    canonical directory are both custody failures, not cache misses: skipping
    either would let a different record answer for the requested identifier.
    """


class ManifestKindMismatch(ValueError):
    """Raised when the manifest holding a requested identifier is the wrong kind."""


def find_manifest_by_id(
    manifest_id: str,
    *,
    root: Path | str | None = None,
    expected_kind: str | None = None,
) -> tuple[AnyManifest, Path]:
    """Find one manifest by ID under a manifest root.

    Resolution is tiered and the canonical `(kind, id, root)` layout is
    authoritative. The SQLite manifest index is derived acceleration only: an
    absent index, a stale entry naming a path that no longer exists, and an
    entry naming bytes that now hold a different identifier all fall through to
    the canonical location and then to a full scan instead of raising.

    Tiers resolve in order — canonical paths, indexed paths, full scan — and the
    first tier yielding any match decides the answer. Duplicate matches inside a
    tier refuse, because one identifier must name one record.

    Args:
        manifest_id: The deterministic identifier one record must hold.
        root: Manifest root; defaults to :func:`default_manifest_root`.
        expected_kind: The ``BaseManifest.kind`` the caller requires. A caller
            that knows the kind states it, because an identifier alone does not:
            a same-identifier record of another kind is a corrupted or forged
            answer, and it refuses rather than being returned or silently
            skipped in favour of a lower tier.
    """
    root_path = Path(root) if root is not None else default_manifest_root()
    located = find_manifest_at_address(manifest_id, root=root_path, expected_kind=expected_kind)
    if located is not None:
        return located
    matches = _manifest_matches(
        manifest_id, iter_manifest_files(root_path), expected_kind=expected_kind
    )
    return _single_manifest_match(manifest_id, root_path, matches)


def find_manifest_at_address(
    manifest_id: str,
    *,
    root: Path | str | None = None,
    expected_kind: str | None = None,
) -> tuple[AnyManifest, Path] | None:
    """Probe canonical paths and any indexed legacy locator without scanning the root."""
    root_path = Path(root) if root is not None else default_manifest_root()
    return _find_manifest_at_address(manifest_id, root_path, expected_kind=expected_kind)


def _find_manifest_at_address(
    manifest_id: str,
    root_path: Path,
    *,
    expected_kind: str | None = None,
    scanned: Mapping[Path, AnyManifest] | None = None,
    unscanned_directories: frozenset[Path] = frozenset(),
) -> tuple[AnyManifest, Path] | None:
    canonical_paths = canonical_manifest_candidate_paths(manifest_id, root=root_path)
    if scanned is not None:
        canonical_paths = [
            path for path in canonical_paths
            if path in scanned or path.parent in unscanned_directories
        ]
    canonical = _manifest_matches(
        manifest_id,
        canonical_paths,
        expected_kind=expected_kind,
        scanned=scanned,
    )
    if canonical:
        return _single_manifest_match(manifest_id, root_path, canonical)

    manifest_dir = Path(os.path.abspath(root_path / "manifests"))
    indexed_paths = [
        path
        for path in find_manifest_paths_by_id(manifest_id, root=root_path)
        if path.suffix == ".json" and Path(os.path.abspath(path)).is_relative_to(manifest_dir)
    ]
    indexed = _manifest_matches(
        manifest_id, indexed_paths, expected_kind=expected_kind, scanned=scanned
    )
    return _single_manifest_match(manifest_id, root_path, indexed) if indexed else None


#: Canonical manifest-root subdirectory name to the single kind it durably stores.
_MANIFEST_KIND_BY_DIRECTORY: dict[str, str] = {
    directory: kind for kind, directory in MANIFEST_KIND_DIRECTORIES.items()
}


def _check_manifest_location(manifest: AnyManifest, manifest_path: Path) -> None:
    """Refuse bytes filed directly under another kind's canonical directory.

    The canonical `(kind, id, root)` layout is authoritative, so the directory a
    record sits in is a claim about what it is. Bytes whose own ``kind``
    contradicts that claim are misfiled or substituted, and admitting them would
    let one kind's canonical location answer for another's. Locations that are
    not a declared kind directory carry no such claim and are left to the
    caller's own kind check.
    """
    declared = _MANIFEST_KIND_BY_DIRECTORY.get(manifest_path.parent.name)
    if declared is None or manifest.kind == declared:
        return
    raise ManifestLocationError(
        f"Manifest {manifest.id!r} of kind {manifest.kind!r} is stored at {manifest_path}, "
        f"the canonical directory for {declared!r}"
    )


def _manifest_matches(
    manifest_id: str,
    candidate_paths: Sequence[Path],
    *,
    expected_kind: str | None = None,
    scanned: Mapping[Path, AnyManifest] | None = None,
) -> list[tuple[AnyManifest, Path]]:
    """Load candidate paths that actually hold *manifest_id*, skipping absent bytes.

    Absent bytes are a stale locator, not a custody failure, so they are skipped.
    Bytes that exist but cannot be parsed as a known manifest are a real custody
    failure and propagate. So is a locator that cannot be resolved: resolution
    is what proves two candidates are not the same file, and continuing past a
    failure would drop a candidate that may be the only true record — or admit
    the same one twice.
    """
    matches: list[tuple[AnyManifest, Path]] = []
    seen: set[Path] = set()
    for manifest_path in candidate_paths:
        try:
            resolved = manifest_path.resolve()
        except OSError as exc:
            raise ManifestLocationError(
                f"Manifest candidate {manifest_path} for {manifest_id!r} could not be "
                f"resolved, so it can be neither read nor ruled out: {exc}"
            ) from exc
        if resolved in seen:
            continue
        manifest = scanned.get(manifest_path) if scanned is not None else None
        if manifest is None:
            if not manifest_path.is_file():
                continue
            manifest = load_manifest(manifest_path)
        seen.add(resolved)
        if manifest.id != manifest_id:
            continue
        _check_manifest_location(manifest, manifest_path)
        if expected_kind is not None and manifest.kind != expected_kind:
            raise ManifestKindMismatch(
                f"Manifest {manifest_id!r} at {manifest_path} is a {manifest.kind!r}, "
                f"not the required {expected_kind!r}"
            )
        matches.append((manifest, manifest_path))
    return matches


def _single_manifest_match(
    manifest_id: str,
    root_path: Path,
    matches: list[tuple[AnyManifest, Path]],
) -> tuple[AnyManifest, Path]:
    if not matches:
        raise FileNotFoundError(f"Manifest {manifest_id!r} not found under {root_path}")
    if len(matches) > 1:
        paths = ", ".join(str(path) for _manifest, path in matches)
        raise ValueError(f"Manifest {manifest_id!r} is duplicated under {root_path}: {paths}")
    return matches[0]


def iter_manifests_by_kind(
    manifest_kind: str,
    *,
    root: Path | str | None = None,
) -> Iterator[tuple[AnyManifest, Path]]:
    """Resolve each discovered ID before filtering by kind."""
    root_path = Path(root) if root is not None else default_manifest_root()
    scanned, paths_by_id = _scan_manifest_files(root_path)
    unscanned_directories = frozenset(
        path
        for directory in MANIFEST_KIND_DIRECTORIES.values()
        if (path := root_path / "manifests" / directory).is_symlink()
    )

    relevant_ids = dict.fromkeys(
        manifest.id for manifest in scanned.values() if manifest.kind == manifest_kind
    )
    for manifest_id in relevant_ids:
        match = _find_manifest_at_address(
            manifest_id, root_path, scanned=scanned, unscanned_directories=unscanned_directories
        )
        if match is None:
            matches = _manifest_matches(manifest_id, paths_by_id[manifest_id], scanned=scanned)
            match = _single_manifest_match(manifest_id, root_path, matches)
        manifest, path = match
        if manifest.kind == manifest_kind:
            yield manifest, path


def find_manifests_in_scan(
    manifest_ids: set[str],
    *,
    root: Path,
    load: Callable[[Path], AnyManifest] = load_manifest,
) -> dict[str, tuple[AnyManifest, Path]]:
    """Find unresolved IDs in one full scan after address probes have missed."""
    if not manifest_ids:
        return {}
    scanned, paths_by_id = _scan_manifest_files(root, manifest_ids=manifest_ids, load=load)
    return {
        manifest_id: _single_manifest_match(
            manifest_id, root, _manifest_matches(manifest_id, paths, scanned=scanned)
        )
        for manifest_id, paths in paths_by_id.items()
    }


def _scan_manifest_files(
    root: Path,
    *,
    manifest_ids: set[str] | None = None,
    load: Callable[[Path], AnyManifest] = load_manifest,
) -> tuple[dict[Path, AnyManifest], dict[str, list[Path]]]:
    scanned: dict[Path, AnyManifest] = {}
    paths_by_id: dict[str, list[Path]] = {}
    for path in iter_manifest_files(root):
        manifest = load(path)
        if manifest_ids is None or manifest.id in manifest_ids:
            scanned[path] = manifest
            paths_by_id.setdefault(manifest.id, []).append(path)
    return scanned, paths_by_id


def manifest_record(manifest: AnyManifest, path: Path) -> dict[str, Any]:
    """Project a receipt and its locator into the run and selection row format."""
    payload = manifest.model_dump(mode="json", exclude_none=True)
    return {
        "id": manifest.id,
        "kind": manifest.kind,
        "schema_version": manifest.schema_version,
        "provider_version": manifest.provider_version,
        "created_at": manifest.created_at.isoformat(),
        "status": manifest.status,
        "path": str(path),
        "payload": payload,
    }


def manifest_records_by_kind(
    manifest_kind: str,
    *,
    root: Path | str | None = None,
) -> list[dict[str, Any]]:
    return _sorted_records(iter_manifests_by_kind(manifest_kind, root=root))


def manifest_records_for_predicate(
    predicate: ManifestPredicate,
    *,
    root: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Read the candidate receipts a predicate can match."""
    if not predicate.run_ids:
        return manifest_records_by_kind(predicate.manifest_kind, root=root)

    matches = []
    for manifest_id in dict.fromkeys(predicate.run_ids):
        try:
            manifest, path = find_manifest_by_id(manifest_id, root=root)
        except FileNotFoundError:
            continue
        if manifest.kind == predicate.manifest_kind:
            matches.append((manifest, path))
    return _sorted_records(matches)


def _sorted_records(matches: Iterable[tuple[AnyManifest, Path]]) -> list[dict[str, Any]]:
    return sorted(
        (manifest_record(manifest, path) for manifest, path in matches),
        key=lambda row: (row["created_at"], row["path"]),
        reverse=True,
    )
