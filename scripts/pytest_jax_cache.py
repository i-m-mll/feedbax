"""Scope the JAX compilation cache to one pytest invocation by default."""

from contextlib import ExitStack
import os
from pathlib import Path

import jax
from jax.experimental.compilation_cache import compilation_cache
import pytest

from scripts.full_suite import owned_suite_temporary_root


_INVOCATION_CACHE_ROOT = pytest.StashKey[Path]()


@pytest.hookimpl(optionalhook=True)
def pytest_configure_node(node) -> None:
    """Give workers the controller-owned root without ownership of its cleanup."""
    if root := node.config.stash.get(_INVOCATION_CACHE_ROOT, None):
        node.workerinput["feedbax_jax_cache_root"] = str(root)


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    """Scope JAX cache configuration and any owned cache bytes to this invocation."""
    if os.environ.get("FEEDBAX_DISABLE_JAX_COMPILATION_CACHE") == "1":
        return

    scope = ExitStack()
    config.add_cleanup(scope.close)
    worker_input = getattr(config, "workerinput", {})
    worker_root = worker_input.get("feedbax_jax_cache_root")
    explicit = os.environ.get("FEEDBAX_JAX_COMPILATION_CACHE_DIR")
    if worker_root is None and explicit is not None:
        cache_dir = Path(explicit).expanduser()
    else:
        if worker_root is not None:
            root = Path(worker_root)
            process_name = worker_input["workerid"]
        else:
            root_env = {}
            if parent := os.environ.get("FEEDBAX_JAX_TEST_CACHE_ROOT"):
                parent_path = Path(parent).expanduser()
                parent_path.mkdir(parents=True, exist_ok=True)
                root_env["TMPDIR"] = str(parent_path)
            root = scope.enter_context(owned_suite_temporary_root(root_env or None))
            config.stash[_INVOCATION_CACHE_ROOT] = root
            process_name = "controller"
        cache_dir = root / process_name / "jax_compilation"
        patch = scope.enter_context(pytest.MonkeyPatch.context())
        # Child processes inherit their launching worker's cache, not the controller's.
        patch.setenv("FEEDBAX_JAX_COMPILATION_CACHE_DIR", str(cache_dir))
    cache_dir.mkdir(parents=True, exist_ok=True)

    min_compile_secs = float(
        os.environ.get("FEEDBAX_JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")
    )
    min_entry_size = int(os.environ.get("FEEDBAX_JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "0"))
    settings = {
        "jax_compilation_cache_dir": str(cache_dir),
        "jax_persistent_cache_min_compile_time_secs": min_compile_secs,
        "jax_persistent_cache_min_entry_size_bytes": min_entry_size,
    }
    # The initialized persistent cache retains its path across config updates.
    scope.callback(compilation_cache.reset_cache)
    for name, value in settings.items():
        scope.callback(jax.config.update, name, getattr(jax.config, name))
        jax.config.update(name, value)
    compilation_cache.reset_cache()
