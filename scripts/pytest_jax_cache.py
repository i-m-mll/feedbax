"""Configure the JAX compilation cache of a pytest invocation.

Test processes and the Hosts they start share one persistent cache across invocations and
worktrees: an entry is keyed by the lowered program and the jaxlib version, so it is a
function of its inputs, and the maximum size makes JAX take a file lock for every read and
write. Tests whose subject is the cache configure a directory of their own.
"""

from contextlib import ExitStack
import os
from pathlib import Path

import jax
from jax.experimental.compilation_cache import compilation_cache
import pytest

from feedbax.host.io import JAX_CACHE_ENV, JAX_CACHE_MAX_SIZE, private_directory
from scripts.full_suite import shared_cache_root


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    """Point this process and the Hosts it starts at the shared cache until cleanup."""
    if os.environ.get("FEEDBAX_DISABLE_JAX_COMPILATION_CACHE") == "1":
        return

    scope = ExitStack()
    config.add_cleanup(scope.close)
    if JAX_CACHE_ENV not in os.environ:
        shared = shared_cache_root(Path(config.rootpath)) / "jax_compilation"
        scope.enter_context(pytest.MonkeyPatch.context()).setenv(JAX_CACHE_ENV, str(shared))
    cache_dir = private_directory(Path(os.environ[JAX_CACHE_ENV]).expanduser())

    settings = {
        "jax_compilation_cache_dir": str(cache_dir),
        "jax_persistent_cache_min_compile_time_secs": 0.0,
        "jax_persistent_cache_min_entry_size_bytes": 0,
        "jax_compilation_cache_max_size": JAX_CACHE_MAX_SIZE,
        "jax_raise_persistent_cache_errors": True,
    }
    # The initialized persistent cache retains its path across config updates.
    scope.callback(compilation_cache.reset_cache)
    for name, value in settings.items():
        scope.callback(jax.config.update, name, getattr(jax.config, name))
        jax.config.update(name, value)
    compilation_cache.reset_cache()
