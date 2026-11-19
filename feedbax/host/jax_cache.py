"""The Feedbax-managed persistent JAX compilation cache.

Entries keep JAX's `<key>-cache` names. A write goes to a temporary file named with the
writer's pid and is renamed over the entry, so a reader sees a whole entry or none and
neither takes a lock. A hit sets the entry's modification time, which records its use.
Eviction is separate from writing: one process at a time scans the directory and removes
the least recently used entries beyond `JAX_CACHE_MAX_SIZE`.

This module is also loaded from the Host's installation into a project's interpreter, so
it imports only the standard library, its sibling `io`, and JAX.
"""

import contextlib
import fcntl
import importlib.abc
import importlib.util
import os
from pathlib import Path
import secrets
import sys

from .io import JAX_CACHE_MAX_SIZE

_ENTRY = "-cache"
_TEMPORARY = ".tmp"


def _entry_cache(directory: Path):
    from jax._src.compilation_cache_interface import CacheInterface

    class EntryCache(CacheInterface):
        def __init__(self, path: Path):
            self._path = path

        def get(self, key: str) -> bytes | None:
            entry = self._path / f"{key}{_ENTRY}"
            try:
                value = entry.read_bytes()
            except FileNotFoundError:
                return None
            with contextlib.suppress(FileNotFoundError):
                os.utime(entry)
            return value

        def put(self, key: str, value: bytes) -> None:
            entry = self._path / f"{key}{_ENTRY}"
            if entry.exists():
                return
            temporary = entry.with_name(f"{entry.name}.{os.getpid()}.{secrets.token_hex(4)}{_TEMPORARY}")
            try:
                temporary.write_bytes(value)
                os.replace(temporary, entry)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise

    return EntryCache(directory)


def install(directory: Path) -> None:
    """Make this process's JAX read and write the managed cache at `directory`."""
    import jax
    from jax._src import compilation_cache

    settings = {
        "jax_enable_compilation_cache": True,
        "jax_compilation_cache_dir": str(directory),
        "jax_persistent_cache_min_compile_time_secs": 0.0,
        "jax_persistent_cache_min_entry_size_bytes": 0,
        "jax_raise_persistent_cache_errors": True,
    }
    for name, value in settings.items():
        jax.config.update(name, value)
    compilation_cache.reset_cache()
    # The one private name: `_get_cache` returns this object instead of building JAX's own.
    compilation_cache._cache = _entry_cache(directory)


class _InstallAfterJaxImport(importlib.abc.MetaPathFinder):
    def __init__(self, directory: Path):
        self.directory = directory

    def find_spec(self, fullname, path, target=None):
        if fullname != "jax":
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(fullname)
        if spec is None or spec.loader is None:
            return spec
        execute = spec.loader.exec_module

        def execute_then_install(module) -> None:
            execute(module)
            install(self.directory)

        spec.loader.exec_module = execute_then_install
        return spec


def install_when_imported(directory: Path) -> None:
    """Install the cache now if JAX is imported, otherwise as soon as it is."""
    if "jax" in sys.modules:
        install(directory)
    else:
        sys.meta_path.insert(0, _InstallAfterJaxImport(directory))


def _live_temporary(name: str) -> bool:
    if not name.endswith(_TEMPORARY):
        return False
    try:
        pid = int(name.rsplit(".", 3)[-3])
        os.kill(pid, 0)
    except (ValueError, IndexError, ProcessLookupError):
        return False
    except PermissionError:
        return True
    return True


def evict(directory: Path, max_size: int = JAX_CACHE_MAX_SIZE) -> None:
    """Bring `directory` within `max_size`, unless another process is already evicting it.

    Removes every file that is neither an entry nor a live writer's temporary file, then
    entries from the least recently used.
    """
    with open(directory.with_name(directory.name + ".lock"), "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        entries = []
        total = 0
        with os.scandir(directory) as scan:
            for item in scan:
                if not item.is_file(follow_symlinks=False):
                    continue
                if item.name.endswith(_ENTRY):
                    try:
                        info = item.stat(follow_symlinks=False)
                    except FileNotFoundError:
                        continue
                    entries.append((info.st_mtime_ns, info.st_size, item.path))
                    total += info.st_size
                elif not _live_temporary(item.name):
                    Path(item.path).unlink(missing_ok=True)
        entries.sort()
        for _, size, path in entries:
            if total <= max_size:
                break
            Path(path).unlink(missing_ok=True)
            total -= size
