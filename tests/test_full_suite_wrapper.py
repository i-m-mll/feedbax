from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def load_full_suite_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "full_suite.py"
    spec = importlib.util.spec_from_file_location("feedbax_full_suite", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_fingerprint_key_changes_when_uv_lock_hash_changes() -> None:
    full_suite = load_full_suite_module()

    old = full_suite.SuiteFingerprint(
        payload={
            "schema_version": 1,
            "git_tree": "tree",
            "uv_lock_sha256": "old-lock",
            "python_version": "3.12.0",
            "jax_version": "0.5.2",
            "jaxlib_version": "0.5.1",
        },
        memo_allowed=True,
    )
    new = full_suite.SuiteFingerprint(
        payload={**old.payload, "uv_lock_sha256": "new-lock"},
        memo_allowed=True,
    )

    assert old.key != new.key


def test_green_memo_lookup_fails_closed_on_lockfile_drift(tmp_path: Path) -> None:
    full_suite = load_full_suite_module()

    old = full_suite.SuiteFingerprint(
        payload={
            "schema_version": 1,
            "git_tree": "tree",
            "uv_lock_sha256": "old-lock",
            "python_version": "3.12.0",
            "jax_version": "0.5.2",
            "jaxlib_version": "0.5.1",
        },
        memo_allowed=True,
    )
    new = full_suite.SuiteFingerprint(
        payload={**old.payload, "uv_lock_sha256": "new-lock"},
        memo_allowed=True,
    )

    full_suite.write_green_memo(tmp_path, old, ["pytest", "tests"])

    assert full_suite.has_green_memo(tmp_path, old)
    assert not full_suite.has_green_memo(tmp_path, new)


def test_unresolved_fingerprint_component_disables_memo() -> None:
    full_suite = load_full_suite_module()

    fingerprint = full_suite.SuiteFingerprint(
        payload={
            "schema_version": 1,
            "git_tree": None,
            "uv_lock_sha256": "lock",
            "python_version": "3.12.0",
            "jax_version": "0.5.2",
            "jaxlib_version": "0.5.1",
        },
        memo_allowed=False,
        refusal_reasons=("git tree hash is unavailable",),
    )

    assert not fingerprint.memo_allowed
    assert not full_suite.has_green_memo(Path("/tmp/does-not-matter"), fingerprint)


def test_pytest_command_uses_xdist_by_default(monkeypatch) -> None:
    full_suite = load_full_suite_module()

    monkeypatch.delenv("FEEDBAX_FULL_SUITE_DISABLE_XDIST", raising=False)
    monkeypatch.setattr(full_suite, "distribution_version", lambda name: "3.8.0")

    command = full_suite.pytest_command(["-q"])

    assert command[2:] == ["pytest", "tests", "-n", "auto", "-q"]


def test_jax_cache_env_exposes_base_root_without_exact_cache_dir(monkeypatch, tmp_path) -> None:
    full_suite = load_full_suite_module()
    cache_root = tmp_path / "cache"

    monkeypatch.delenv("FEEDBAX_JAX_COMPILATION_CACHE_DIR", raising=False)
    monkeypatch.delenv("FEEDBAX_JAX_TEST_CACHE_ROOT", raising=False)

    full_suite.configure_jax_cache_env(cache_root)

    assert "FEEDBAX_JAX_COMPILATION_CACHE_DIR" not in full_suite.os.environ
    assert full_suite.os.environ["FEEDBAX_JAX_TEST_CACHE_ROOT"] == str(cache_root)


def test_jax_cache_env_preserves_explicit_exact_cache_dir(monkeypatch, tmp_path) -> None:
    full_suite = load_full_suite_module()
    cache_root = tmp_path / "cache"

    monkeypatch.setenv("FEEDBAX_JAX_COMPILATION_CACHE_DIR", "/explicit/jax-cache")
    monkeypatch.delenv("FEEDBAX_JAX_TEST_CACHE_ROOT", raising=False)

    full_suite.configure_jax_cache_env(cache_root)

    assert full_suite.os.environ["FEEDBAX_JAX_COMPILATION_CACHE_DIR"] == "/explicit/jax-cache"
    assert "FEEDBAX_JAX_TEST_CACHE_ROOT" not in full_suite.os.environ
