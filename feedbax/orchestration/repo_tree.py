"""Sealed tree identity shared by the controller and venue bootstrap."""

import hashlib
import os
from pathlib import Path
from typing import Any


class RepoSnapshotError(RuntimeError):
    """Raised when governed repository bytes cannot be sealed safely."""


def hash_field(digest: Any, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def snapshot_tree_identity(root: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    entries = _snapshot_entries(root)
    for relative_bytes, path in entries:
        hash_field(digest, relative_bytes)
        if path.is_symlink():
            hash_field(digest, b"120000")
            hash_field(digest, os.fsencode(os.readlink(path)))
            continue
        mode = b"100755" if path.stat().st_mode & 0o111 else b"100644"
        hash_field(digest, mode)
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(len(chunk).to_bytes(8, "big"))
                digest.update(chunk)
        digest.update((0).to_bytes(8, "big"))
    return digest.hexdigest(), len(entries)


def _snapshot_entries(root: Path) -> list[tuple[bytes, Path]]:
    entries: list[tuple[bytes, Path]] = []

    def visit(directory: Path) -> None:
        children = sorted(os.scandir(directory), key=lambda entry: os.fsencode(entry.name))
        for child in children:
            path = Path(child.path)
            relative_bytes = os.fsencode(path.relative_to(root).as_posix())
            if child.is_symlink():
                entries.append((relative_bytes, path))
            elif child.is_dir(follow_symlinks=False):
                visit(path)
            elif child.is_file(follow_symlinks=False):
                entries.append((relative_bytes, path))
            else:
                raise RepoSnapshotError(f"sealed snapshot contains unsafe file type: {path}")

    visit(root)
    entries.sort(key=lambda item: item[0])
    return entries
