from collections.abc import Iterator
import os
from pathlib import Path
import subprocess

import jax
import pytest


def _repo_cache_root() -> Path:
    """Return a cache root shared by Git worktrees for this checkout."""
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return repo_root / ".git" / "feedbax_test_cache"

    common_dir = Path(result.stdout.strip()).resolve()
    return common_dir / "feedbax_test_cache"


def _configure_jax_persistent_cache() -> None:
    if os.environ.get("FEEDBAX_DISABLE_JAX_COMPILATION_CACHE") == "1":
        return

    cache_dir = Path(
        os.environ.get(
            "FEEDBAX_JAX_COMPILATION_CACHE_DIR",
            str(_repo_cache_root() / "jax_compilation"),
        )
    ).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)

    min_compile_secs = float(
        os.environ.get("FEEDBAX_JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")
    )
    min_entry_size = int(os.environ.get("FEEDBAX_JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "0"))
    jax.config.update("jax_compilation_cache_dir", str(cache_dir))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", min_compile_secs)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", min_entry_size)


_configure_jax_persistent_cache()


@pytest.fixture
def enable_jax_x64() -> Iterator[None]:
    """Enable JAX x64 only for one test, then restore the prior global state."""
    previous = bool(jax.config.jax_enable_x64)
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        # Several precision tests used to set this at import time, polluting the
        # process-wide JAX config during collection and making later tests fail.
        jax.config.update("jax_enable_x64", previous)
