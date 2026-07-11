from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import signal
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


def lock_helper_code() -> str:
    return """
import importlib.util
import os
from pathlib import Path
import sys
import time

script_path, lock_path, worktree, release_path = sys.argv[1:]
spec = importlib.util.spec_from_file_location("feedbax_full_suite_helper", script_path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

with module.FullSuiteLock(Path(lock_path), repo_root=Path(worktree)):
    print("acquired " + str(os.getpid()), flush=True)
    release = Path(release_path)
    while not release.exists():
        time.sleep(0.02)
"""


def start_lock_helper(
    script_path: Path,
    lock_path: Path,
    worktree: Path,
    release_path: Path,
) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [
            sys.executable,
            "-c",
            lock_helper_code(),
            str(script_path),
            str(lock_path),
            str(worktree),
            str(release_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


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


def test_shared_cache_root_is_common_across_worktrees(tmp_path: Path) -> None:
    full_suite = load_full_suite_module()
    repo = make_full_suite_repo(tmp_path / "repo")
    worktree = tmp_path / "other-worktree"
    run_git(repo, "worktree", "add", "-b", "test-other-worktree", str(worktree))

    assert full_suite.shared_cache_root(repo) == full_suite.shared_cache_root(worktree)


def test_full_suite_lock_refuses_and_reports_holder(tmp_path: Path) -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "full_suite.py"
    lock_path = tmp_path / "common.git" / "feedbax_test_cache" / "full_suite.lock"
    holder_release = tmp_path / "release-holder"
    contender_release = tmp_path / "release-contender"
    holder = start_lock_helper(
        script_path, lock_path, Path("/worktrees/holder"), holder_release
    )
    assert holder.stdout is not None
    assert holder.stdout.readline().startswith("acquired ")

    contender = start_lock_helper(
        script_path, lock_path, Path("/worktrees/contender"), contender_release
    )
    contender_stdout, contender_stderr = contender.communicate(timeout=5)
    assert contender.returncode != 0
    assert contender_stdout == ""
    assert "Full suite already running; active holder:" in contender_stderr
    assert "worktree=/worktrees/holder" in contender_stderr

    holder_release.touch()
    holder_stdout, holder_stderr = holder.communicate(timeout=5)
    assert holder.returncode == 0, holder_stderr
    assert holder_stdout == ""
    released_metadata = json.loads(lock_path.read_text(encoding="utf-8"))
    assert released_metadata["worktree"] == "/worktrees/holder"


def test_main_returns_temporary_failure_when_lock_is_held(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    full_suite = load_full_suite_module()
    repo_root = Path(__file__).resolve().parents[1]
    cache_root = tmp_path / "common.git" / "feedbax_test_cache"
    monkeypatch.setattr(full_suite, "shared_cache_root", lambda unused: cache_root)

    with full_suite.FullSuiteLock(cache_root / "full_suite.lock", repo_root=repo_root):
        result = full_suite.main(["--force", "--no-memo"])

    assert result == 75
    assert "Full suite already running; active holder:" in capsys.readouterr().err


def test_full_suite_lock_recovers_after_abnormal_holder_exit(tmp_path: Path) -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "full_suite.py"
    lock_path = tmp_path / "common.git" / "feedbax_test_cache" / "full_suite.lock"
    holder_release = tmp_path / "release-holder"
    contender_release = tmp_path / "release-contender"
    holder = start_lock_helper(
        script_path,
        lock_path,
        Path("/worktrees/interrupted-holder"),
        holder_release,
    )
    assert holder.stdout is not None
    assert holder.stdout.readline().startswith("acquired ")

    os.kill(holder.pid, signal.SIGKILL)
    holder.wait(timeout=5)
    contender = start_lock_helper(
        script_path, lock_path, Path("/worktrees/after-interrupt"), contender_release
    )
    assert contender.stdout is not None
    assert contender.stdout.readline().startswith("acquired ")
    contender_release.touch()
    _, contender_stderr = contender.communicate(timeout=5)
    assert contender.returncode == 0, contender_stderr

    # The kernel releases the advisory lock after SIGKILL. The next holder
    # safely replaces stale metadata before running; release retains that last
    # complete record so contenders never observe an intentional empty window.
    metadata = json.loads(lock_path.read_text(encoding="utf-8"))
    assert metadata["worktree"] == "/worktrees/after-interrupt"
