"""Simulate out-of-band damage to read-only Store objects in integrity tests."""

from contextlib import contextmanager
from pathlib import Path
import stat


@contextmanager
def writable_store_object(path: Path):
    mode = stat.S_IMODE(path.stat().st_mode)
    path.chmod(mode | stat.S_IWUSR)
    try:
        yield path
    finally:
        path.chmod(mode)
