"""Exact-byte, atomic publication of local documents."""

from __future__ import annotations

from collections.abc import Callable
import os
from pathlib import Path
import uuid


def write_bytes_atomic(
    path: Path,
    data: bytes,
    *,
    mode: int = 0o600,
    replace: bool = True,
    before_publish: Callable[[], None] | None = None,
    directory_fd: int | None = None,
) -> None:
    """Publish a document after flushing its unique same-directory staging file.

    ``mode`` is filtered by the process umask, just as for ordinary file creation;
    replacement adopts the new file's permissions. Serialization belongs to the
    caller. Concurrent replacements are last-replacement-wins. With
    ``replace=False``, publication fails if the destination already exists.

    Success confirms file and directory sync. Path-based writes sync every namespace
    ancestor. Descriptor-relative writes sync the open parent; the caller owns its
    publication in the containing directory. Unsupported directory sync raises.
    Errors after replacement can leave new bytes visible without confirmed durability.
    A caller may check a precondition after staging and immediately before publication.
    """
    temporary_name = f".{path.name}.{uuid.uuid4().hex}.tmp"
    if directory_fd is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        staging = path.with_name(temporary_name)
        destination = path
        location = {}
        publication = {}
    else:
        staging = Path(temporary_name)
        destination = Path(path.name)
        location = {"dir_fd": directory_fd}
        publication = {"src_dir_fd": directory_fd, "dst_dir_fd": directory_fd}
    descriptor = os.open(staging, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode, **location)
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
        if before_publish is not None:
            before_publish()
        if replace:
            os.replace(staging, destination, **publication)
        else:
            os.link(staging, destination, **publication)
            os.unlink(staging, **location)
        if directory_fd is None:
            sync_parent_directories(path)
        else:
            os.fsync(directory_fd)
    finally:
        try:
            os.unlink(staging, **location)
        except FileNotFoundError:
            pass


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
