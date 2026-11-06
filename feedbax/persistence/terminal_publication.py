"""Publication of terminal figure and report receipts."""

from collections.abc import Iterator
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path

from feedbax.contracts.manifest import (
    FigureManifest,
    ReportManifest,
    canonical_manifest_path,
    load_manifest,
    write_manifest,
)


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


def publish_render_manifest(
    manifest: FigureManifest | ReportManifest, *, root: Path
) -> Path | None:
    """Publish a render outcome. Return None when a failure would replace success."""
    path = canonical_manifest_path(manifest.kind, manifest.id, root=root)
    with manifest_publication_lock(path):
        if manifest.status == "failed":
            try:
                previous = load_manifest(path)
            except FileNotFoundError:
                pass
            else:
                if (previous.kind, previous.id) != (manifest.kind, manifest.id):
                    raise ValueError(f"Manifest identity does not match its canonical path {path}")
                if previous.status == "completed":
                    return None
        return write_manifest(manifest, root=root)
