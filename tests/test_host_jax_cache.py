import os
from pathlib import Path
import subprocess
import sys

from feedbax.host import jax_cache
from tests.support.hang import HANG_SECONDS


_COMPILE = r"""
import sys
from pathlib import Path

from feedbax.host.jax_cache import install_when_imported

directory, order = Path(sys.argv[1]), sys.argv[2]
if order == "late":
    install_when_imported(directory)
import jax
import jax.monitoring
if order == "early":
    install_when_imported(directory)

events = []
jax.monitoring.register_event_listener(lambda name, **kwargs: events.append(name))
jax.jit(lambda x: x * 17.0 - 3.0).lower(2.0).compile()
print(sum(name == "/jax/compilation_cache/cache_hits" for name in events))
"""


def _compile(directory: Path, order: str) -> int:
    env = {key: value for key, value in os.environ.items() if not key.startswith("JAX_COMPILATION_CACHE")}
    result = subprocess.run(
        [sys.executable, "-c", _COMPILE, str(directory), order],
        env=env, capture_output=True, text=True, timeout=HANG_SECONDS, check=True,
    )
    return int(result.stdout)


def test_installed_cache_writes_one_whole_entry_and_another_process_hits_it(tmp_path):
    directory = tmp_path / "cache"
    directory.mkdir()

    assert _compile(directory, "late") == 0
    entries = sorted(path.name for path in directory.iterdir())
    assert len(entries) == 1 and entries[0].endswith("-cache"), entries

    entry = directory / entries[0]
    os.utime(entry, (1_000, 1_000))
    assert _compile(directory, "early") == 1
    assert sorted(path.name for path in directory.iterdir()) == entries
    assert entry.stat().st_mtime > 1_000, "the hit did not go through the Feedbax cache"


def test_eviction_keeps_recent_entries_and_removes_debris(tmp_path):
    directory = tmp_path / "cache"
    directory.mkdir()
    cache = jax_cache._entry_cache(directory)
    for index in range(4):
        (directory / f"k{index}-cache").write_bytes(b"x" * 100)
        os.utime(directory / f"k{index}-cache", (1_000 + index, 1_000 + index))
    assert cache.get("k0") == b"x" * 100
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    (directory / f"k9-cache.{dead.pid}.0000.tmp").write_bytes(b"partial")
    live = directory / f"k8-cache.{os.getpid()}.0000.tmp"
    live.write_bytes(b"writing")
    (directory / "k1-atime").write_bytes(b"t")
    (directory / ".lockfile").write_bytes(b"")

    jax_cache.evict(directory, max_size=250)

    assert sorted(path.name for path in directory.iterdir()) == sorted(["k0-cache", "k3-cache", live.name])
