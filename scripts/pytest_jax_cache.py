"""Configure the JAX compilation cache of a pytest invocation.

Test processes and the Hosts they start share one Feedbax-managed cache across invocations
and worktrees: an entry is keyed by the lowered program and the jaxlib version, so it is a
function of its inputs. The controller evicts it once at the start of a session. Tests whose
subject is the cache configure a directory of their own.
"""

from contextlib import ExitStack
import os
from pathlib import Path

import jax
from jax._src import compilation_cache
import pytest

from feedbax.host import jax_cache
from feedbax.host.io import JAX_CACHE_ENV, private_directory
from scripts.full_suite import shared_cache_root


_RESTORED = (
    "jax_enable_compilation_cache",
    "jax_compilation_cache_dir",
    "jax_persistent_cache_min_compile_time_secs",
    "jax_persistent_cache_min_entry_size_bytes",
    "jax_raise_persistent_cache_errors",
)


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
    if not hasattr(config, "workerinput"):
        jax_cache.evict(cache_dir)

    scope.callback(compilation_cache.reset_cache)
    for name in _RESTORED:
        scope.callback(jax.config.update, name, getattr(jax.config, name))
    jax_cache.install(cache_dir)
