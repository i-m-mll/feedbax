"""Exact-byte, atomic replacement for mutable local documents."""

from __future__ import annotations

import errno
import os
from pathlib import Path
import uuid


def write_bytes_atomic(path: Path, data: bytes, *, mode: int = 0o600) -> None:
    """Replace a document after flushing its unique same-directory staging file.

    ``mode`` is filtered by the process umask, just as for ordinary file creation;
    replacement adopts the new file's permissions. Serialization belongs to the
    caller. Concurrent publishers are last-replacement-wins.

    Directory fsync is skipped on Windows and filesystems that explicitly report
    it unsupported. Other errors propagate, including directory fsync failure
    after replacement: the new bytes may be visible without confirmed durability.
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
        os.replace(staging, path)
        _fsync_directory(path.parent)
    finally:
        staging.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        try:
            os.fsync(descriptor)
        except OSError as error:
            if error.errno not in (errno.EINVAL, errno.ENOTSUP):
                raise
    finally:
        os.close(descriptor)
