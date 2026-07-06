from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
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


def run_git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)


def commit_all(repo: Path, message: str) -> None:
    run_git(repo, "add", ".")
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test User",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-m",
            message,
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


def make_full_suite_repo(repo: Path) -> Path:
    repo.mkdir()
    run_git(repo, "init")
    (repo / "feedbax").mkdir()
    (repo / "feedbax" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_placeholder.py").write_text("def test_placeholder():\n    pass\n")
    (repo / "pyproject.toml").write_text("[project]\nname = 'feedbax-test'\n")
    (repo / "uv.lock").write_text("lock\n", encoding="utf-8")
    commit_all(repo, "initial")
    return repo


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


def test_tracked_change_disables_memo_recording(monkeypatch, tmp_path: Path) -> None:
    full_suite = load_full_suite_module()
    repo = make_full_suite_repo(tmp_path / "repo")
    monkeypatch.setattr(full_suite, "distribution_version", lambda name: "0.0.0")

    (repo / "feedbax" / "__init__.py").write_text("# dirty\n", encoding="utf-8")

    fingerprint = full_suite.build_fingerprint(repo)

    assert not fingerprint.memo_allowed
    assert "git working tree is dirty" in fingerprint.refusal_reasons


def test_untracked_docs_file_does_not_block_memo_recording(monkeypatch, tmp_path: Path) -> None:
    full_suite = load_full_suite_module()
    repo = make_full_suite_repo(tmp_path / "repo")
    memo_dir = tmp_path / "memo"
    monkeypatch.setattr(full_suite, "distribution_version", lambda name: "0.0.0")

    docs_path = repo / "docs" / "scratch.md"
    docs_path.parent.mkdir()
    docs_path.write_text("# local notes\n", encoding="utf-8")

    fingerprint = full_suite.build_fingerprint(repo)

    assert fingerprint.memo_allowed, fingerprint.refusal_reasons
    full_suite.write_green_memo(memo_dir, fingerprint, ["pytest", "tests"])
    assert full_suite.has_green_memo(memo_dir, fingerprint)


def test_untracked_relevant_test_file_disables_memo_recording(
    monkeypatch, tmp_path: Path
) -> None:
    full_suite = load_full_suite_module()
    repo = make_full_suite_repo(tmp_path / "repo")
    monkeypatch.setattr(full_suite, "distribution_version", lambda name: "0.0.0")

    (repo / "tests" / "test_new.py").write_text("def test_new():\n    pass\n", encoding="utf-8")

    fingerprint = full_suite.build_fingerprint(repo)

    assert not fingerprint.memo_allowed
    assert "git working tree is dirty" in fingerprint.refusal_reasons
