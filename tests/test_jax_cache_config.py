import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from tests.support.hang import HANG_SECONDS


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
    "jax_compilation_cache_max_size": -1,
}
for name, value in caller_config.items():
    jax.config.update(name, value)

expected_exit, *pytest_args = sys.argv[1:]
caller_host_cache = os.environ.get("FEEDBAX_HOST_JAX_CACHE_DIR")
exit_code = pytest.main(pytest_args)
assert exit_code == int(expected_exit), (exit_code, expected_exit)
restored = {name: getattr(jax.config, name) for name in caller_config}
assert restored == caller_config, restored
assert os.environ.get("FEEDBAX_HOST_JAX_CACHE_DIR") == caller_host_cache

before = set(caller_cache.glob("*-cache"))
jax.jit(lambda x: x * 13.0 - 7.0).lower(2.0).compile()
assert set(caller_cache.glob("*-cache")) - before, "post-pytest compilation missed caller cache"
"""

_CONFTEST = r"""
import json
import os
from pathlib import Path

import jax
import jax.monitoring

COUNTS = {"hits": 0, "misses": 0}

def _count(name, **kwargs):
    for kind in COUNTS:
        if name == f"/jax/compilation_cache/cache_{kind}":
            COUNTS[kind] += 1

jax.monitoring.register_event_listener(_count)

def pytest_sessionfinish(session):
    worker = os.environ.get("PYTEST_XDIST_WORKER", "controller")
    record = {"cache": jax.config.jax_compilation_cache_dir, **COUNTS}
    Path(os.environ["CACHE_RECORDS"], worker + ".json").write_text(json.dumps(record))
"""

_WITNESS = r"""
import os
import signal
import subprocess
import sys

import jax
import jax.numpy as jnp

def test_cache():
    host_cache = os.environ.get("FEEDBAX_HOST_JAX_CACHE_DIR")
    if host_cache is not None:
        child = subprocess.check_output([
            sys.executable, "-c", "import os; print(os.environ['FEEDBAX_HOST_JAX_CACHE_DIR'])",
        ], text=True, timeout=10).strip()
        assert child == host_cache == jax.config.jax_compilation_cache_dir
    assert float(jax.jit(lambda x: x * 3.0 + 1.0)(jnp.array(2.0))) == 7
    outcome = os.environ["CACHE_OUTCOME"]
    if outcome == "interrupt":
        signal.raise_signal(signal.SIGINT)
    assert outcome != "failure"
"""


@pytest.mark.parametrize(
    ("outcome", "workers", "override", "returncode"),
    [
        ("success", "0", "default", 0),
        ("success", "2", "default", 0),
        ("failure", "0", "default", 1),
        ("collection", "0", "default", 2),
        ("interrupt", "0", "default", 2),
        ("success", "0", "exact", 0),
        ("success", "0", "disabled", 0),
    ],
)
def test_shared_cache_lifecycle(tmp_path, outcome, workers, override, returncode):
    """Observe real processes: shared cache across invocations, xdist, Hosts, and caller restore."""
    repo = Path(__file__).resolve().parents[1]
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    subprocess.run(["git", "init", "-q", str(corpus)], check=True, timeout=10)
    records = tmp_path / "records"
    records.mkdir()
    (corpus / "conftest.py").write_text(_CONFTEST)
    (corpus / "test_witness.py").write_text(
        "raise RuntimeError('collection witness')" if outcome == "collection" else _WITNESS
    )
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("FEEDBAX_", "PYTEST_", "JAX_COMPILATION_CACHE"))
    }
    env.update(
        {
            "PYTHONPATH": str(repo),
            "CACHE_RECORDS": str(records),
            "CACHE_OUTCOME": outcome,
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        }
    )
    expected = corpus.resolve() / ".git" / "feedbax_test_cache" / "jax_compilation"
    if override == "exact":
        expected = tmp_path / "explicit"
        env["FEEDBAX_HOST_JAX_CACHE_DIR"] = str(expected)
    elif override == "disabled":
        env["FEEDBAX_DISABLE_JAX_COMPILATION_CACHE"] = "1"
    command = [
        sys.executable,
        "-c",
        _PYTEST_IN_CALLER,
        str(returncode),
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
    invocations = 2 if outcome == "success" else 1
    for invocation in range(invocations):
        for record in records.iterdir():
            record.unlink()
        env["CACHE_CALLER_DIR"] = str(tmp_path / f"caller-cache-{invocation}")
        result = subprocess.run(
            command, cwd=corpus, env=env, capture_output=True, text=True, timeout=HANG_SECONDS, check=False
        )
        assert result.returncode == 0, result.stdout + result.stderr
        observed = {path.stem: json.loads(path.read_text()) for path in records.glob("*.json")}
        if workers == "2":
            assert set(observed) == {"controller", "gw0", "gw1"}
        if override == "disabled":
            assert {record["cache"] for record in observed.values()} == {env["CACHE_CALLER_DIR"]}
            continue
        assert {record["cache"] for record in observed.values()} == {str(expected)}
        assert (expected / ".lockfile").exists() == (outcome != "collection")
        if outcome == "success" and invocation == 1:
            assert sum(record["misses"] for record in observed.values()) == 0, observed
            assert sum(record["hits"] for record in observed.values()) > 0, observed
