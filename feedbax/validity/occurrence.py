"""Filesystem identity for source and environment observations."""

from os import stat_result
from pathlib import Path

type FileStamp = tuple[int, int, int, int, int]


def file_stamp(info: stat_result) -> FileStamp:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def occurrence(path: Path) -> FileStamp | None:
    try:
        return file_stamp(path.stat())
    except (FileNotFoundError, NotADirectoryError):
        return None
