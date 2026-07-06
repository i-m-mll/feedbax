from collections.abc import Iterator
import hashlib
import os
from pathlib import Path
import subprocess

import jax
import pytest


_CACHE_SOURCE_PATHS = ("feedbax", "tests", "pyproject.toml", "uv.lock")
_CACHE_NAMESPACE_VERSION = b"source-fingerprint-v1"


def _git_output(repo_root: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    env = os.environ.copy()
    env["GIT_OPTIONAL_LOCKS"] = "0"
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        capture_output=True,
        check=False,
        env=env,
    )


def _source_cache_namespace(repo_root: Path) -> str:
    """Return a short namespace for compiled artifacts tied to tracked source state."""
    head = _git_output(repo_root, "rev-parse", "--verify", "HEAD")
    if head.returncode != 0:
        return "unknown-source"

    diff = _git_output(
        repo_root,
        "diff",
        "--no-ext-diff",
        "--binary",
        "HEAD",
        "--",
        *_CACHE_SOURCE_PATHS,
    )
    diff_bytes = diff.stdout if diff.returncode == 0 else b""
    head_bytes = head.stdout.strip()
    digest = hashlib.sha256(
        _CACHE_NAMESPACE_VERSION + b"\0" + head_bytes + b"\0" + diff_bytes
    ).hexdigest()[:16]
    head_prefix = head_bytes.decode(errors="replace")[:12]
    return f"{head_prefix}-{digest}"


def _cache_invocation_namespace() -> str:
    return os.environ.get("FEEDBAX_JAX_CACHE_INVOCATION_ID", f"pid-{os.getpid()}")


def _shared_test_cache_root(repo_root: Path) -> Path:
    override = os.environ.get("FEEDBAX_JAX_TEST_CACHE_ROOT")
    if override:
        return Path(override).expanduser()

    result = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
    )
    if result.returncode != 0:
        return repo_root / ".git" / "feedbax_test_cache"

    common_dir = Path(result.stdout.strip()).resolve()
    return common_dir / "feedbax_test_cache"


def _repo_cache_root(repo_root: Path | None = None) -> Path:
    """Return a test-invocation cache root grouped by tracked source state."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[1]
    shared_root = _shared_test_cache_root(repo_root)
    return shared_root / _source_cache_namespace(repo_root) / _cache_invocation_namespace()


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
