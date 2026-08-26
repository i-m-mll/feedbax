from __future__ import annotations

import ast
import importlib.util
import json
import os
from pathlib import Path
import signal
import shutil
import stat
import subprocess
import sys
import tomllib

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
    # Serial must force `-n 0`, not an empty runner list: the pyproject addopts
    # default enables `-n auto`, so only an explicit `-n 0` makes the disabled
    # path actually run serially and keeps the serial fingerprint truthful.
    assert serial_args == ["-n", "0"]
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


def _module_pytest_markers(source: str) -> set[str]:
    """Return every ``pytest.mark.<name>`` assigned to a module-level ``pytestmark``."""
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "pytestmark" for target in node.targets
        ):
            continue
        value = node.value
        elements = value.elts if isinstance(value, (ast.List, ast.Tuple)) else [value]
        return {
            element.attr for element in elements if isinstance(element, ast.Attribute)
        }
    return set()


def test_slow_tier_leaves_the_default_gate_and_returns_for_every_closeout_profile() -> None:
    """The `slow` tier is deselected per iteration and restored by this wrapper.

    The `slow` modules build a real wheel and install it, so they need a clean Git
    checkout plus a network or a warm `uv` cache. That is not a promise the gate
    that must pass on every invocation can make, so `addopts` deselects them. They
    are still load-bearing — `test_cold_start_conformance` is the honesty gate for
    the whole upstreaming program — so every profile this wrapper runs must bring
    them back. A command-line `-m` overrides `addopts`, which is exactly how that
    happens; this test fails if either half of the arrangement is edited away.
    """
    repo_root = Path(__file__).resolve().parents[1]
    addopts = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))[
        "tool"
    ]["pytest"]["ini_options"]["addopts"]

    assert "not slow" in addopts, (
        "the default gate must deselect the `slow` tier; the marker is declared for "
        "exactly this purpose and is inert unless `addopts` names it"
    )

    full_suite = load_full_suite_module()
    for profile, expression in full_suite.SUITE_MARKER_EXPRESSIONS.items():
        assert "slow" not in expression, (
            f"the `{profile}` closeout profile must not deselect `slow`: the wrapper is "
            "where the wheel-building tier is paid for"
        )

    tests_dir = repo_root / "tests"
    for module in ("test_cold_start_conformance.py", "test_feedbax_wheel_provenance.py"):
        markers = _module_pytest_markers((tests_dir / module).read_text(encoding="utf-8"))
        assert "slow" in markers, (
            f"{module} builds and installs a real wheel and must stay in the `slow` tier"
        )


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


@pytest.mark.parametrize("returncode", [0, 7])
def test_main_removes_sealed_temporary_tree_after_pytest_exit(
    monkeypatch, tmp_path: Path, returncode: int
) -> None:
    full_suite = load_full_suite_module()
    repo_root = Path(__file__).resolve().parents[1]
    caller_root = tmp_path / "caller temporary root with spaces"
    caller_root.mkdir()
    caller_sentinel = caller_root / "keep-me"
    caller_sentinel.write_text("caller-owned", encoding="utf-8")
    observed_root = None

    monkeypatch.setenv("TMPDIR", str(caller_root))
    monkeypatch.setenv("FULL_SUITE_LOCK_DIR", str(tmp_path / "lock"))
    monkeypatch.setenv("FEEDBAX_FULL_SUITE_DISABLE_XDIST", "1")
    monkeypatch.setattr(full_suite, "repo_root_from", lambda start: repo_root)
    monkeypatch.setattr(full_suite, "shared_cache_root", lambda root: tmp_path / "cache")
    monkeypatch.setattr(
        full_suite,
        "build_fingerprint",
        lambda *args, **kwargs: full_suite.SuiteFingerprint(
            payload={},
            memo_allowed=False,
        ),
    )

    def run_pytest(command, *, cwd, env, check):
        nonlocal observed_root
        observed_root = Path(env["TMPDIR"])
        assert cwd == repo_root
        assert check is False
        assert observed_root.parent == caller_root
        assert observed_root != caller_root
        assert (observed_root / full_suite.TEMP_ROOT_MARKER).is_file()
        sealed_snapshot = observed_root / "sealed repo snapshot"
        sealed_snapshot.mkdir()
        sealed_file = sealed_snapshot / "large-fixture.bin"
        sealed_file.write_bytes(b"fixture")
        sealed_file.chmod(0o444)
        sealed_snapshot.chmod(0o555)
        return subprocess.CompletedProcess(command, returncode)

    monkeypatch.setattr(full_suite.subprocess, "run", run_pytest)

    assert full_suite.main(["--force", "--no-memo"]) == returncode
    assert observed_root is not None
    assert not observed_root.exists()
    assert caller_sentinel.read_text(encoding="utf-8") == "caller-owned"


def test_main_removes_sealed_temporary_tree_after_interruption(
    monkeypatch, tmp_path: Path
) -> None:
    full_suite = load_full_suite_module()
    repo_root = Path(__file__).resolve().parents[1]
    caller_root = tmp_path / "caller root"
    caller_root.mkdir()
    observed_root = None

    monkeypatch.setenv("TMPDIR", str(caller_root))
    monkeypatch.setenv("FULL_SUITE_LOCK_DIR", str(tmp_path / "lock"))
    monkeypatch.setenv("FEEDBAX_FULL_SUITE_DISABLE_XDIST", "1")
    monkeypatch.setattr(full_suite, "repo_root_from", lambda start: repo_root)
    monkeypatch.setattr(full_suite, "shared_cache_root", lambda root: tmp_path / "cache")
    monkeypatch.setattr(
        full_suite,
        "build_fingerprint",
        lambda *args, **kwargs: full_suite.SuiteFingerprint(
            payload={},
            memo_allowed=False,
        ),
    )

    def interrupt_pytest(command, *, cwd, env, check):
        nonlocal observed_root
        observed_root = Path(env["TMPDIR"])
        sealed_snapshot = observed_root / "sealed-repo-snapshot"
        sealed_snapshot.mkdir()
        sealed_file = sealed_snapshot / "partial-fixture.bin"
        sealed_file.write_bytes(b"partial")
        sealed_file.chmod(0o444)
        sealed_snapshot.chmod(0o555)
        raise KeyboardInterrupt

    monkeypatch.setattr(full_suite.subprocess, "run", interrupt_pytest)

    with pytest.raises(KeyboardInterrupt):
        full_suite.main(["--force", "--no-memo"])

    assert observed_root is not None
    assert not observed_root.exists()
    assert caller_root.exists()


def test_temporary_root_cleanup_refuses_missing_ownership_marker(tmp_path: Path) -> None:
    full_suite = load_full_suite_module()
    caller_root = tmp_path / "caller-root"
    caller_root.mkdir()
    with pytest.raises(
        full_suite.TemporaryRootOwnershipError,
        match="refusing to remove unverified suite temporary root",
    ):
        with full_suite.owned_suite_temporary_root({"TMPDIR": str(caller_root)}) as owned_root:
            sealed_directory = owned_root / "sealed"
            sealed_directory.mkdir()
            sealed_file = sealed_directory / "read-only"
            sealed_file.write_text("caller-like", encoding="utf-8")
            sealed_file.chmod(0o444)
            sealed_directory.chmod(0o555)
            (owned_root / full_suite.TEMP_ROOT_MARKER).unlink()

    assert owned_root.is_dir()
    assert owned_root.parent == caller_root
    assert stat.S_IMODE(sealed_directory.stat().st_mode) == 0o555
    assert stat.S_IMODE(sealed_file.stat().st_mode) == 0o444
    sealed_directory.chmod(0o755)
    shutil.rmtree(owned_root)
