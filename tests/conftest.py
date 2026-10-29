# Native pool budgets must be set before importing JAX or NumPy.
# ruff: noqa: E402
from collections.abc import Iterator
import os
from pathlib import Path
import shutil
from contextlib import ExitStack

from scripts.full_suite import (
    auto_worker_count,
    configure_native_threads,
    owned_suite_temporary_root,
)

configure_native_threads()

import jax
import pytest

import feedbax
from feedbax.analysis.evaluation import EvaluationRecipeRegistry
from feedbax.analysis.specs import AnalysisRecipeRegistry
from feedbax.orchestration import revision as _revision
from feedbax.orchestration.repo_snapshot import REPO_SNAPSHOT_CACHE_DIR_ENV
from feedbax.plugins.application import (
    ApplicationRegistryBundle,
    new_application_registry_bundle,
)


@pytest.hookimpl(optionalhook=True)
def pytest_xdist_auto_num_workers(config: pytest.Config) -> int:
    return auto_worker_count()


@pytest.fixture
def application_registry_bundle() -> ApplicationRegistryBundle:
    """Return fresh caller-owned registries for explicit lifecycle tests."""
    return new_application_registry_bundle(local_component_source=None)


@pytest.fixture
def evaluation_registry() -> EvaluationRecipeRegistry:
    return EvaluationRecipeRegistry()


@pytest.fixture
def analysis_registry() -> AnalysisRecipeRegistry:
    return AnalysisRecipeRegistry()


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    """Own default cache bytes until the controller finishes, including collection errors."""
    if os.environ.get("FEEDBAX_DISABLE_JAX_COMPILATION_CACHE") == "1":
        return

    explicit = os.environ.get("FEEDBAX_JAX_COMPILATION_CACHE_DIR")
    if explicit is not None:
        cache_dir = Path(explicit).expanduser()
    else:
        scope = ExitStack()
        config.add_cleanup(scope.close)
        root_env = {}
        if parent := os.environ.get("FEEDBAX_JAX_TEST_CACHE_ROOT"):
            parent_path = Path(parent).expanduser()
            parent_path.mkdir(parents=True, exist_ok=True)
            root_env["TMPDIR"] = str(parent_path)
        root = scope.enter_context(owned_suite_temporary_root(root_env or None))
        cache_dir = root / "jax_compilation"
        patch = scope.enter_context(pytest.MonkeyPatch.context())
        # Workers and child processes share this invocation; only its controller owns cleanup.
        patch.setenv("FEEDBAX_JAX_COMPILATION_CACHE_DIR", str(cache_dir))
    cache_dir.mkdir(parents=True, exist_ok=True)

    min_compile_secs = float(
        os.environ.get("FEEDBAX_JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")
    )
    min_entry_size = int(os.environ.get("FEEDBAX_JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "0"))
    jax.config.update("jax_compilation_cache_dir", str(cache_dir))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", min_compile_secs)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", min_entry_size)


@pytest.fixture(scope="session", autouse=True)
def _isolated_repo_snapshot_cache(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Keep sealed test snapshots out of the per-checkout production snapshot cache.

    Tests that seal repo snapshots through a production entry point would otherwise share
    one content-addressed tree with real runs and with every other worker on the machine.
    Pytest's per-worker base temporary directory gives each xdist worker of each run its
    own parent, and prunes it on later runs.
    """
    cache_dir = tmp_path_factory.mktemp("repo-snapshot-cache", numbered=False)
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(REPO_SNAPSHOT_CACHE_DIR_ENV, str(cache_dir))
        yield
    # Sealed snapshot directories are read-only by design, which defeats pytest's own
    # temporary-directory pruning, so restore owner write permission before removal.
    for directory, _subdirectories, _files in os.walk(cache_dir, topdown=True):
        os.chmod(directory, 0o700)
    shutil.rmtree(cache_dir, ignore_errors=True)


_WORKING_PACKAGE_ROOT = Path(feedbax.__file__).resolve().parent


@pytest.fixture(autouse=True)
def _fresh_checkout_revision_cache() -> Iterator[None]:
    """Give every test the memoized revision state of a freshly started process.

    ``revision._resolve_checkout_revision`` memoizes per package directory because
    the commit that supplied an imported package is fixed once the interpreter has
    loaded it. A test is not a process: it redirects ``feedbax.__file__`` at a
    throwaway checkout it just built, and may commit to or rewrite that checkout
    while it runs. Clearing the cache on both sides of each test keeps that
    freedom, so no test can observe an answer another test resolved and none can
    leak one forward.

    This covers test boundaries only. A single test that resolves a checkout and
    *then* changes what Git would answer about it — initialising a repository over
    an installed wheel, committing, checking out — must call
    ``revision._reset_checkout_revision_cache()`` itself between the two, or it
    will assert against the answer it already took.
    """
    _revision._reset_checkout_revision_cache()
    yield
    _revision._reset_checkout_revision_cache()


@pytest.fixture(autouse=True)
def _tolerate_dirty_working_checkout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Report only the working checkout under test as clean to the provenance gate.

    Assembly now fails closed when the checkout supplying ``import feedbax`` has
    uncommitted changes, which is the normal state of a checkout someone is
    editing and running tests in. Tests that assemble a request are not the place
    that guarantee reproduces: they import the very source they are exercising.
    Only the real working package root is reported clean, so the tmp-checkout
    tests in ``tests/test_feedbax_revision_provenance.py`` and the assembly-gate
    tests still exercise genuine dirty detection.
    """
    original = _revision._feedbax_tree_is_dirty

    def _dirty(package_root: Path) -> bool:
        if package_root == _WORKING_PACKAGE_ROOT:
            return False
        return original(package_root)

    monkeypatch.setattr(_revision, "_feedbax_tree_is_dirty", _dirty)


_SUBPROCESS_DIRTY_TOLERANCE = '''\
"""Test-only startup hook: report the working checkout under test as clean.

Tests that run the real orchestration CLI in a child process exercise the
working checkout, which is routinely dirty while someone is editing it. The
in-process fixture cannot reach a child, so the same narrow tolerance is applied
here: only the real working package root is reported clean, and every other
path still goes through genuine Git dirty detection.
"""

from pathlib import Path

import feedbax
from feedbax.orchestration import revision as _revision

_WORKING_PACKAGE_ROOT = Path(feedbax.__file__).resolve().parent
_original_tree_is_dirty = _revision._feedbax_tree_is_dirty


def _tree_is_dirty(package_root):
    if package_root == _WORKING_PACKAGE_ROOT:
        return False
    return _original_tree_is_dirty(package_root)


_revision._feedbax_tree_is_dirty = _tree_is_dirty
'''


@pytest.fixture
def subprocess_dirty_tolerance():
    """Return an installer for the child-process counterpart of the tolerance fixture.

    Call it with a directory that is first on the child's ``PYTHONPATH``; it
    writes a ``sitecustomize`` module there, which Python imports at interpreter
    startup before the CLI runs.
    """

    def install(directory: Path) -> None:
        (directory / "sitecustomize.py").write_text(_SUBPROCESS_DIRTY_TOLERANCE, encoding="utf-8")

    return install


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
