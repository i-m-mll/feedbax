"""Exact-byte, atomic publication of local documents."""

from __future__ import annotations

import os
from pathlib import Path
import uuid


def write_bytes_atomic(path: Path, data: bytes, *, mode: int = 0o600, replace: bool = True) -> None:
    """Publish a document after flushing its unique same-directory staging file.

    ``mode`` is filtered by the process umask, just as for ordinary file creation;
    replacement adopts the new file's permissions. Serialization belongs to the
    caller. Concurrent replacements are last-replacement-wins. With
    ``replace=False``, publication fails if the destination already exists.

    Success confirms file and directory sync, including every namespace ancestor.
    Unsupported directory sync raises rather than claiming durability. Errors
    after replacement can leave new bytes visible without confirmed durability.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(staging, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        try:
            stream = os.fdopen(descriptor, "wb")
        except BaseException:
            os.close(descriptor)
            raise
        with stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(staging, path)
        else:
            os.link(staging, path)
            staging.unlink()
        sync_parent_directories(path)
    finally:
        staging.unlink(missing_ok=True)


def sync_parent_directories(path: Path) -> None:
    """Sync a file's parent directory and all its resolved ancestors."""
    parent = path.parent.resolve(strict=True)
    for directory in (parent, *parent.parents):
        _fsync_directory(directory)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
