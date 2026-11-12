"""Exact file evidence for an authenticated local completion."""

import hashlib
import os
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import BinaryIO, Iterator

from feedbax._secure_fs import (
    SecurePathRigor,
    open_directory_chain,
    open_existing_directory,
    open_existing_file,
)


class CompletionFileError(ValueError):
    """A pinned completion file is absent, malformed or changed."""


@contextmanager
def completion_file(root: Path, relative: str) -> Iterator[tuple[BinaryIO, int]]:
    if not isinstance(relative, str) or any(
        part in {"", ".", ".."} or "\\" in part or "\x00" in part for part in relative.split("/")
    ):
        raise CompletionFileError(f"Invalid relative completion path: {relative!r}")
    parts = relative.split("/")
    with ExitStack() as stack:
        root_chain = open_directory_chain(
            root, create=False, error_factory=CompletionFileError, context="completion root"
        )
        for _, fd, _ in root_chain:
            stack.callback(os.close, fd)
        fd = root_chain[-1][1]
        for component in parts[:-1]:
            fd = open_existing_directory(
                component,
                error_factory=CompletionFileError,
                context="completion directory",
                dir_fd=fd,
            )
            stack.callback(os.close, fd)
        member, info = open_existing_file(
            parts[-1],
            rigor=SecurePathRigor.REGULAR_FILE_IDENTITY,
            error_factory=CompletionFileError,
            context="completion file",
            dir_fd=fd,
        )
        stream = stack.enter_context(os.fdopen(member, "rb"))
        yield stream, info.st_size
        after = os.fstat(stream.fileno())
        named = os.stat(parts[-1], dir_fd=fd, follow_symlinks=False)
        if (after.st_dev, after.st_ino) != (named.st_dev, named.st_ino) or (
            after.st_size,
            after.st_mtime_ns,
        ) != (info.st_size, info.st_mtime_ns):
            raise CompletionFileError(f"Completion file changed while open: {relative}")


def verify_completion_files(binding: dict) -> None:
    """Rehash the producer-attested immutable byte closure before delivery."""
    try:
        root = Path(binding["manifest_root"])
        files = binding["files"]
        if not root.is_absolute() or not isinstance(files, list) or not files:
            raise CompletionFileError("Completion requires an absolute root and file evidence.")
        paths = set()
        for item in files:
            relative = item["path"]
            if relative in paths or type(item["size_bytes"]) is not int:
                raise CompletionFileError("Invalid or duplicate completion file evidence.")
            paths.add(relative)
            with completion_file(root, relative) as (stream, size):
                if (
                    size != item["size_bytes"]
                    or hashlib.file_digest(stream, "sha256").hexdigest() != item["sha256"]
                ):
                    raise CompletionFileError(f"Completion file changed: {relative}")
    except (KeyError, TypeError, OSError) as error:
        raise CompletionFileError(f"Cannot verify completion files: {error}") from error
