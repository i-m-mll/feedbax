from __future__ import annotations

import ast
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest


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

script_path, lock_path, worktree, release_path, repository = sys.argv[1:]
spec = importlib.util.spec_from_file_location("feedbax_full_suite_helper", script_path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

with module.FullSuiteLock(
    Path(lock_path),
    repo_root=Path(worktree),
    repository=repository,
    command=[repository, "scripts/full_suite.sh"],
):
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
    *,
    repository: str = "feedbax",
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
            repository,
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

    assert command[2:] == [
        "pytest",
        "tests",
        "-n",
        "auto",
        "-m",
        "not optional_mjx and not optional_ppo",
        "-q",
    ]


@pytest.mark.parametrize(
    ("flag", "profile", "expression"),
    [
        ("--include-mjx", "mjx", "not optional_ppo"),
        ("--include-ppo", "ppo", "not optional_mjx"),
        (
            "--include-optional",
            "all",
            "optional_mjx or optional_ppo or (not optional_mjx and not optional_ppo)",
        ),
    ],
)
def test_optional_profile_flags_select_expected_markers(
    monkeypatch, flag: str, profile: str, expression: str
) -> None:
    full_suite = load_full_suite_module()
    monkeypatch.setattr(full_suite, "distribution_version", lambda name: None)

    args, passthrough = full_suite.parse_args([flag, "-q"])
    command = full_suite.pytest_command(passthrough, suite_profile=args.suite_profile)

    assert args.suite_profile == profile
    assert command[-3:] == ["-m", expression, "-q"]


def test_green_memo_cannot_alias_different_test_selection(monkeypatch, tmp_path: Path) -> None:
    full_suite = load_full_suite_module()
    repo = make_full_suite_repo(tmp_path / "repo")
    monkeypatch.setattr(full_suite, "distribution_version", lambda name: "0.0.0")

    core = full_suite.build_fingerprint(repo)
    mjx = full_suite.build_fingerprint(repo, suite_profile="mjx")
    selected_node = full_suite.build_fingerprint(
        repo,
        pytest_args=("tests/test_rl_ppo.py::TestGAE::test_shapes",),
    )
    memo_dir = tmp_path / "memo"
    full_suite.write_green_memo(memo_dir, core, ["pytest", "tests"])

    assert full_suite.has_green_memo(memo_dir, core)
    assert not full_suite.has_green_memo(memo_dir, mjx)
    assert not full_suite.has_green_memo(memo_dir, selected_node)


def test_green_memo_cannot_alias_serial_runner(monkeypatch, tmp_path: Path) -> None:
    full_suite = load_full_suite_module()
    repo = make_full_suite_repo(tmp_path / "repo")
    memo_dir = tmp_path / "memo"
    monkeypatch.setattr(full_suite, "distribution_version", lambda name: "0.0.0")

    monkeypatch.delenv("FEEDBAX_FULL_SUITE_DISABLE_XDIST", raising=False)
    parallel_args = full_suite.xdist_args()
    parallel = full_suite.build_fingerprint(repo, runner_args=parallel_args)
    full_suite.write_green_memo(memo_dir, parallel, ["pytest", "tests", *parallel_args])

    monkeypatch.setenv("FEEDBAX_FULL_SUITE_DISABLE_XDIST", "1")
    serial_args = full_suite.xdist_args()
    serial = full_suite.build_fingerprint(repo, runner_args=serial_args)

    assert parallel_args == ["-n", "auto"]
    assert serial_args == []
    assert full_suite.has_green_memo(memo_dir, parallel)
    assert not full_suite.has_green_memo(memo_dir, serial)


def test_nonempty_pytest_addopts_disables_green_memo(monkeypatch, tmp_path: Path) -> None:
    full_suite = load_full_suite_module()
    repo = make_full_suite_repo(tmp_path / "repo")
    memo_dir = tmp_path / "memo"
    monkeypatch.setattr(full_suite, "distribution_version", lambda name: "0.0.0")

    monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
    canonical = full_suite.build_fingerprint(repo, runner_args=("-n", "auto"))
    full_suite.write_green_memo(memo_dir, canonical, ["pytest", "tests", "-n", "auto"])

    monkeypatch.setenv("PYTEST_ADDOPTS", "-k subset")
    fingerprint = full_suite.build_fingerprint(repo, runner_args=("-n", "auto"))

    assert full_suite.has_green_memo(memo_dir, canonical)
    assert not fingerprint.memo_allowed
    assert "PYTEST_ADDOPTS is nonempty" in fingerprint.refusal_reasons
    assert not full_suite.has_green_memo(memo_dir, fingerprint)


def _assigned_pytest_marker(source: str) -> str | None:
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "pytestmark" for target in node.targets
        ):
            continue
        value = node.value
        if isinstance(value, ast.Attribute):
            return value.attr
    return None


def _class_pytest_markers(source: str, class_name: str) -> set[str]:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                decorator.attr
                for decorator in node.decorator_list
                if isinstance(decorator, ast.Attribute)
            }
    raise AssertionError(f"class {class_name} not found")


def test_representative_nodes_have_expected_optional_markers() -> None:
    tests_dir = Path(__file__).parent
    mjx_source = (tests_dir / "test_mjx_plant.py").read_text(encoding="utf-8")
    ppo_source = (tests_dir / "test_batched_ppo.py").read_text(encoding="utf-8")
    backend_source = (tests_dir / "test_backend.py").read_text(encoding="utf-8")

    assert _assigned_pytest_marker(mjx_source) == "optional_mjx"
    assert _assigned_pytest_marker(ppo_source) == "optional_ppo"
    assert "optional_mjx" in _class_pytest_markers(backend_source, "TestMJXBackendState")
    assert "optional_mjx" not in _class_pytest_markers(backend_source, "TestDiffraxBackendState")


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


def test_untracked_relevant_test_file_disables_memo_recording(monkeypatch, tmp_path: Path) -> None:
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


def test_lock_directory_default_and_override(tmp_path: Path) -> None:
    full_suite = load_full_suite_module()
    default = full_suite.full_suite_lock_dir({})

    assert default.parent == Path(full_suite.tempfile.gettempdir())
    assert default.name == f"full-suite-lock-{os.getuid()}"
    assert full_suite.full_suite_lock_path({}) == default / "full-suite.lock"

    override = tmp_path / "shared"
    environ = {"FULL_SUITE_LOCK_DIR": str(override)}
    assert full_suite.full_suite_lock_dir(environ) == override
    assert full_suite.full_suite_lock_path(environ) == override / "full-suite.lock"


@pytest.mark.parametrize("holder_repository", ["feedbax", "rlrmp"])
def test_full_suite_lock_refuses_and_reports_holder(tmp_path: Path, holder_repository: str) -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "full_suite.py"
    lock_path = tmp_path / "shared-lock" / "full-suite.lock"
    holder_release = tmp_path / "release-holder"
    contender_release = tmp_path / "release-contender"
    holder_worktree = Path(f"/worktrees/{holder_repository}-holder")
    holder = start_lock_helper(
        script_path,
        lock_path,
        holder_worktree,
        holder_release,
        repository=holder_repository,
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
    assert f"repository={holder_repository}" in contender_stderr
    assert f"worktree={holder_worktree}" in contender_stderr
    assert f"command=['{holder_repository}', 'scripts/full_suite.sh']" in contender_stderr
    assert "pid=" in contender_stderr
    assert "started_at=" in contender_stderr

    holder_release.touch()
    holder_stdout, holder_stderr = holder.communicate(timeout=5)
    assert holder.returncode == 0, holder_stderr
    assert holder_stdout == ""
    released_metadata = json.loads(lock_path.read_text(encoding="utf-8"))
    assert released_metadata["schema_version"] == 1
    assert released_metadata["protocol_version"] == 1
    assert released_metadata["repository"] == holder_repository
    assert released_metadata["worktree"] == str(holder_worktree)
    assert released_metadata["command"] == [holder_repository, "scripts/full_suite.sh"]


def test_main_returns_temporary_failure_when_lock_is_held(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    full_suite = load_full_suite_module()
    repo_root = Path(__file__).resolve().parents[1]
    lock_dir = tmp_path / "shared-lock"
    monkeypatch.setenv("FULL_SUITE_LOCK_DIR", str(lock_dir))

    with full_suite.FullSuiteLock(
        lock_dir / "full-suite.lock",
        repo_root=repo_root,
        repository="rlrmp",
    ):
        result = full_suite.main(["--force", "--no-memo"])

    assert result == full_suite.LOCK_BUSY_EXIT
    stderr = capsys.readouterr().err
    assert "Full suite already running; active holder:" in stderr
    assert "repository=rlrmp" in stderr


def test_full_suite_lock_recovers_after_abnormal_holder_exit(tmp_path: Path) -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "full_suite.py"
    lock_path = tmp_path / "shared-lock" / "full-suite.lock"
    holder_release = tmp_path / "release-holder"
    contender_release = tmp_path / "release-contender"
    holder = start_lock_helper(
        script_path,
        lock_path,
        Path("/worktrees/interrupted-holder"),
        holder_release,
        repository="rlrmp",
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
    assert metadata["schema_version"] == 1
    assert metadata["protocol_version"] == 1
    assert metadata["repository"] == "feedbax"
    assert metadata["worktree"] == "/worktrees/after-interrupt"
