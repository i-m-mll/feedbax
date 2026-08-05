from collections.abc import Iterator
import hashlib
import os
from pathlib import Path
import shutil
import subprocess

import jax
import pytest

import feedbax
from feedbax.analysis.evaluation import EvaluationRecipeRegistry
from feedbax.analysis.reports import ReportRecipeRegistry
from feedbax.analysis.specs import AnalysisRecipeRegistry
from feedbax.orchestration import revision as _revision
from feedbax.orchestration.repo_snapshot import REPO_SNAPSHOT_CACHE_DIR_ENV
from feedbax.plugins.application import (
    ApplicationRegistryBundle,
    new_application_registry_bundle,
)


_CACHE_SOURCE_PATHS = ("feedbax", "tests", "pyproject.toml", "uv.lock")
_CACHE_NAMESPACE_VERSION = b"source-fingerprint-v1"


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


@pytest.fixture
def report_registry() -> ReportRecipeRegistry:
    return ReportRecipeRegistry()


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
        (directory / "sitecustomize.py").write_text(
            _SUBPROCESS_DIRTY_TOLERANCE, encoding="utf-8"
        )

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
