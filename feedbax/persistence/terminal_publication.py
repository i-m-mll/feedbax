"""Publication of terminal manifests at their canonical receipts."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import os
from pathlib import Path

from feedbax.contracts.manifest import (
    AnyManifest,
    FigureManifest,
    ReportManifest,
    canonical_manifest_path,
    load_manifest,
    write_manifest,
)


@dataclass
class RenderPublicationCapture:
    """The exact receipt bytes observed before releasing the publication lock."""

    raw: bytes | None = None

    def require_bytes(self) -> bytes:
        if self.raw is None:
            raise RuntimeError("No render receipt was captured during publication")
        return self.raw


@contextmanager
def manifest_publication_lock(path: Path) -> Iterator[None]:
    """Lock a stable sibling inode across inspection, replacement and indexing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o666)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def read_publication_manifest(path: Path, *, kind: str, manifest_id: str) -> AnyManifest | None:
    """Read the current canonical receipt while holding its publication lock."""
    try:
        previous = load_manifest(path)
    except FileNotFoundError:
        return None
    if (previous.kind, previous.id) != (kind, manifest_id):
        raise ValueError(f"Manifest identity does not match its canonical path {path}")
    return previous


def publish_terminal_manifest_locked(
    manifest: AnyManifest,
    *,
    root: Path,
    previous: AnyManifest | None,
    index: bool = True,
) -> Path | None:
    """Publish under the receipt lock. A completed receipt can only be superseded by success."""
    if previous is not None and previous.status == "completed" and manifest.status != "completed":
        return None
    return write_manifest(manifest, root=root, index=index)


def publish_render_manifest(
    manifest: FigureManifest | ReportManifest,
    *,
    root: Path,
    capture: RenderPublicationCapture | None = None,
) -> Path | None:
    """Publish a render outcome. Return None when a failure would replace success."""
    path = canonical_manifest_path(manifest.kind, manifest.id, root=root)
    with manifest_publication_lock(path):
        previous = (
            read_publication_manifest(path, kind=manifest.kind, manifest_id=manifest.id)
            if manifest.status == "failed"
            else None
        )
        published_path = publish_terminal_manifest_locked(manifest, root=root, previous=previous)
        if capture is not None and published_path is not None:
            capture.raw = published_path.read_bytes()
        return published_path
