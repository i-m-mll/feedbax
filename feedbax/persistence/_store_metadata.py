"""Atomic writes for mutable metadata under the project store lock."""

import os
import stat
import uuid
from collections.abc import Callable
from pathlib import Path


def sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def ensure_store_directory(
    root: Path, path: Path, *, error_factory: Callable[[str], Exception]
) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise error_factory("store metadata path escapes the canonical content store") from exc
    current = root
    for part in relative.parts:
        current = current / part
        try:
            current.mkdir()
        except FileExistsError:
            pass
        mode = current.lstat().st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise error_factory(f"store metadata path contains an unsafe component: {current}")
        # An earlier attempt may have created the directory but failed to sync its parent.
        sync_directory(current.parent)


def write_store_metadata(
    root: Path,
    path: Path,
    data: bytes,
    *,
    error_factory: Callable[[str], Exception],
) -> None:
    ensure_store_directory(root, path.parent, error_factory=error_factory)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise error_factory("store metadata write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
