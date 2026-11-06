import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


_PYTEST_IN_CALLER = r"""
import os
from pathlib import Path
import sys

from scripts.full_suite import configure_native_threads

configure_native_threads()

import jax
import pytest

caller_cache = Path(os.environ["CACHE_CALLER_DIR"])
caller_config = {
    "jax_compilation_cache_dir": str(caller_cache),
    "jax_persistent_cache_min_compile_time_secs": -1.0,
    "jax_persistent_cache_min_entry_size_bytes": 1,
}
for name, value in caller_config.items():
    jax.config.update(name, value)

expected_exit, warm_cache, *pytest_args = sys.argv[1:]
if warm_cache == "warm":
    jax.jit(lambda x: x * 5.0).lower(2.0).compile()
    assert list(caller_cache.glob("*-cache")), "caller cache did not initialize"

exit_code = pytest.main(pytest_args)
assert exit_code == int(expected_exit), (exit_code, expected_exit)
restored = {name: getattr(jax.config, name) for name in caller_config}
assert restored == caller_config, restored

before = set(caller_cache.glob("*-cache"))
jax.jit(lambda x: x * 13.0 - 7.0).lower(2.0).compile()
assert set(caller_cache.glob("*-cache")) - before, "post-pytest compilation missed caller cache"
"""


@pytest.mark.parametrize(
    ("outcome", "workers", "override", "returncode"),
    [
        ("success", "0", "default", 0),
        ("failure", "0", "root", 1),
        ("collection", "0", "default", 2),
        ("interrupt", "0", "default", 2),
        ("success", "2", "default", 0),
        ("success", "0", "exact", 0),
        ("success", "2", "exact", 0),
        ("success", "0", "disabled", 0),
    ],
)
def test_invocation_cache_lifecycle(tmp_path, outcome, workers, override, returncode):
    """Observe real processes, cache bytes, xdist inheritance, and cleanup boundaries."""
    repo = Path(__file__).resolve().parents[1]
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    cache_parent = tmp_path / "cache parent"
    cache_parent.mkdir()
    sentinel = cache_parent / "caller-owned"
    sentinel.write_bytes(b"keep")
    records = tmp_path / "records"
    records.mkdir()
    conftest = r"""
import json
import os
from pathlib import Path

import jax

def pytest_sessionstart(session):
    worker = os.environ.get("PYTEST_XDIST_WORKER", "controller")
    cache = jax.config.jax_compilation_cache_dir
    Path(os.environ["CACHE_RECORDS"], worker + ".json").write_text(json.dumps(cache))

def pytest_sessionfinish(session):
    if hasattr(session.config, "workerinput"):
        return
    records = Path(os.environ["CACHE_RECORDS"])
    live = {
        path.stem: Path(json.loads(path.read_text())).is_dir()
        for path in records.glob("*.json") if json.loads(path.read_text()) is not None
    }
    (records / "live_at_controller_finish").write_text(json.dumps(live))
"""
    (corpus / "conftest.py").write_text(conftest)
    if outcome == "collection":
        source = "raise RuntimeError('collection witness')"
    else:
        source = r"""
import os
from pathlib import Path
import signal
import subprocess
import sys
import jax
import jax.numpy as jnp

def test_cache():
    expected = os.environ.get("FEEDBAX_JAX_COMPILATION_CACHE_DIR")
    if expected:
        child = subprocess.check_output([
            sys.executable, "-c",
            "import os; print(os.environ['FEEDBAX_JAX_COMPILATION_CACHE_DIR'])",
        ], text=True, timeout=10).strip()
        assert child == expected
    else:
        expected = os.environ["CACHE_CALLER_DIR"]
    assert jax.config.jax_compilation_cache_dir == expected
    assert float(jax.jit(lambda x: x + 1)(jnp.array(2.0))) == 3
    assert any(Path(expected).glob("*-cache"))
    outcome = os.environ["CACHE_OUTCOME"]
    if outcome == "interrupt":
        signal.raise_signal(signal.SIGINT)
    assert outcome != "failure"
"""
    (corpus / "test_witness.py").write_text(source)
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(
            (
                "FEEDBAX_JAX_",
                "FEEDBAX_DISABLE_JAX_",
                "PYTEST_",
            )
        )
    }
    env.update(
        {
            "PYTHONPATH": str(repo),
            "TMPDIR": str(cache_parent),
            "CACHE_RECORDS": str(records),
            "CACHE_OUTCOME": outcome,
            "JAX_ENABLE_COMPILATION_CACHE": "true",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        }
    )
    if override == "root":
        env["FEEDBAX_JAX_TEST_CACHE_ROOT"] = str(cache_parent)
    elif override == "exact":
        env["FEEDBAX_JAX_COMPILATION_CACHE_DIR"] = str(cache_parent / "explicit")
        env["FEEDBAX_JAX_TEST_CACHE_ROOT"] = str(cache_parent / "unused")
    elif override == "disabled":
        env["FEEDBAX_DISABLE_JAX_COMPILATION_CACHE"] = "1"
    command = [
        sys.executable,
        "-c",
        _PYTEST_IN_CALLER,
        str(returncode),
        "warm" if (outcome, workers, override) == ("success", "0", "default") else "cold",
        "-p",
        "xdist.plugin",
        "-p",
        "scripts.pytest_jax_cache",
        "-n",
        workers,
        "--dist=each",
        "-q",
        "-o",
        "addopts=",
        "--basetemp",
        str(tmp_path / "pytest-temp"),
    ]
    caches = []
    repeats = 2 if (outcome, workers, override) == ("success", "0", "default") else 1
    for invocation_index in range(repeats):
        env["CACHE_CALLER_DIR"] = str(tmp_path / f"caller-cache-{invocation_index}")
        result = subprocess.run(
            command, cwd=corpus, env=env, capture_output=True, text=True, timeout=25, check=False
        )
        assert result.returncode == 0, result.stdout + result.stderr
        observed = {path.stem: json.loads(path.read_text()) for path in records.glob("*.json")}
        if override == "disabled":
            assert observed == {"controller": env["CACHE_CALLER_DIR"]}
        else:
            cache = Path(observed["controller"])
            caches.append(cache)
            live = json.loads((records / "live_at_controller_finish").read_text())
            assert live == {name: True for name in observed}
            if override == "exact":
                assert len(set(observed.values())) == 1, observed
                assert cache == cache_parent / "explicit"
                assert sum(path.stat().st_size for path in cache.rglob("*") if path.is_file()) > 0
            else:
                paths = {name: Path(value) for name, value in observed.items()}
                assert len(set(paths.values())) == len(paths)
                invocation = cache.parent.parent
                assert {path.parent.parent for path in paths.values()} == {invocation}
                assert all(path.parent.name == name for name, path in paths.items())
                assert not invocation.exists()
                assert not cache.exists()
                assert list(cache_parent.iterdir()) == [sentinel]
                retained = sum(
                    path.stat().st_size
                    for path in cache_parent.rglob("*")
                    if path.is_file() and path != sentinel
                )
                assert retained == 0
        if workers == "2":
            assert set(observed) == {"controller", "gw0", "gw1"}
    if len(caches) == 2:
        assert caches[0] != caches[1]
    assert sentinel.read_bytes() == b"keep"
