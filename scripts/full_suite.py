from __future__ import annotations

import argparse
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
import fcntl
import getpass
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import secrets
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
from types import TracebackType
from typing import Any, Iterator, Self, Sequence


SCHEMA_VERSION = 2
LOCK_PROTOCOL_VERSION = 1
LOCK_BUSY_EXIT = 75
LOCK_ENV_VAR = "FULL_SUITE_LOCK_DIR"
LOCK_FILENAME = "full-suite.lock"
LOCK_REPOSITORY = "feedbax"
TEMP_ROOT_MARKER = ".feedbax-full-suite-owner"
REQUIRED_FINGERPRINT_FIELDS = (
    "git_tree",
    "uv_lock_sha256",
    "python_version",
    "jax_version",
    "jaxlib_version",
)
EXECUTION_RELEVANT_UNTRACKED_PATHS = (
    "conftest.py",
    "feedbax/",
    "pyproject.toml",
    "pytest.ini",
    "scripts/",
    "setup.cfg",
    "tests/",
    "tox.ini",
    "uv.lock",
)
# None of these expressions deselect `slow`, and that is deliberate. The
# per-iteration default gate deselects it through `addopts` in `pyproject.toml`
# because the `slow` tests build a real wheel and install it, which needs a
# clean Git checkout plus a network or warm `uv` cache — resources a gate that
# must pass on every invocation cannot promise. These expressions are passed on
# the command line, and a command-line `-m` overrides `addopts`, so the closeout
# bar this wrapper runs restores the `slow` tier every time. `test_full_suite_wrapper`
# pins both halves of that relationship.
SUITE_MARKER_EXPRESSIONS = {
    "core": "not optional_mjx and not optional_ppo",
    "mjx": "not optional_ppo",
    "ppo": "not optional_mjx",
    "all": "optional_mjx or optional_ppo or (not optional_mjx and not optional_ppo)",
}


class FullSuiteLockBusy(RuntimeError):
    """Raised when another repository-wide full-suite run owns the lock."""


class TemporaryRootOwnershipError(RuntimeError):
    """Raised when suite ownership cannot be proved before cleanup."""


def _make_owned_tree_removable(root: Path) -> None:
    """Add only the owner permissions needed to remove entries at and below ``root``."""

    def add_owner_permissions(path: Path, permissions: int) -> None:
        status = path.lstat()
        if stat.S_ISLNK(status.st_mode):
            return
        path.chmod(stat.S_IMODE(status.st_mode) | permissions, follow_symlinks=False)

    directory_permissions = stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR
    add_owner_permissions(root, directory_permissions)
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        for directory in directories:
            add_owner_permissions(current_path / directory, directory_permissions)
        for filename in files:
            add_owner_permissions(current_path / filename, stat.S_IWUSR)


@contextmanager
def owned_suite_temporary_root(
    environ: dict[str, str] | None = None,
) -> Iterator[Path]:
    """Yield one marked child beneath the caller-selected temporary directory."""
    env = dict(os.environ) if environ is None else environ
    caller_root = env.get("TMPDIR")
    parent = Path(caller_root).expanduser() if caller_root else Path(tempfile.gettempdir())
    path = Path(tempfile.mkdtemp(prefix="feedbax-full-suite.", dir=parent)).resolve()
    identity = path.stat().st_dev, path.stat().st_ino
    token = secrets.token_hex(32)
    marker = path / TEMP_ROOT_MARKER
    try:
        marker.write_text(token, encoding="utf-8")
    except BaseException:
        shutil.rmtree(path)
        raise

    try:
        yield path
    finally:
        try:
            path_status = path.lstat()
            marker_status = marker.lstat()
            marker_token = marker.read_text(encoding="utf-8")
        except OSError as error:
            raise TemporaryRootOwnershipError(
                f"refusing to remove unverified suite temporary root: {path}"
            ) from error

        if (
            not stat.S_ISDIR(path_status.st_mode)
            or (path_status.st_dev, path_status.st_ino) != identity
            or not stat.S_ISREG(marker_status.st_mode)
            or marker_token != token
        ):
            raise TemporaryRootOwnershipError(
                f"refusing to remove unverified suite temporary root: {path}"
            )
        _make_owned_tree_removable(path)
        shutil.rmtree(path)


class FullSuiteLock(AbstractContextManager["FullSuiteLock"]):
    """Hold the machine-wide advisory lock used by participating test suites."""

    def __init__(
        self,
        path: Path,
        *,
        repo_root: Path,
        repository: str = LOCK_REPOSITORY,
        command: Sequence[str] | None = None,
    ) -> None:
        self.path = path
        self.repo_root = repo_root
        self.repository = repository
        self.command = list(sys.argv if command is None else command)
        self._handle: Any | None = None

    def _read_holder(self) -> str:
        assert self._handle is not None
        self._handle.seek(0)
        raw = self._handle.read().strip()
        if not raw:
            return "holder metadata unavailable"
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return f"unreadable holder metadata: {raw!r}"
        return ", ".join(
            f"{key}={data[key]}"
            for key in ("repository", "pid", "host", "started_at", "worktree", "command")
            if data.get(key) is not None
        )

    def _write_holder(self) -> None:
        assert self._handle is not None
        holder = {
            "schema_version": LOCK_PROTOCOL_VERSION,
            "protocol_version": LOCK_PROTOCOL_VERSION,
            "repository": self.repository,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "started_at": datetime.now(UTC).isoformat(),
            "worktree": str(self.repo_root),
            "command": self.command,
        }
        self._handle.seek(0)
        self._handle.truncate()
        json.dump(holder, self._handle, sort_keys=True)
        self._handle.write("\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def __enter__(self) -> Self:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+", encoding="utf-8")
        try:
            try:
                fcntl.flock(self._handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise FullSuiteLockBusy(
                    f"Full suite already running; active holder: {self._read_holder()}"
                ) from exc
            self._write_holder()
            print(f"Acquired full-suite lock: {self.path}", file=sys.stderr, flush=True)
            return self
        except BaseException:
            self._handle.close()
            self._handle = None
            raise

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._handle is None:
            return
        try:
            fcntl.flock(self._handle, fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None


def full_suite_lock_dir(environ: dict[str, str] | None = None) -> Path:
    """Return the shared, user-scoped full-suite lock directory."""
    env = dict(os.environ) if environ is None else environ
    if override := env.get(LOCK_ENV_VAR):
        return Path(override).expanduser()
    try:
        user_token = str(os.getuid())
    except AttributeError:  # pragma: no cover - exercised only on platforms without getuid.
        user_token = re.sub(r"[^A-Za-z0-9_.-]", "_", getpass.getuser()) or "unknown"
    return Path(tempfile.gettempdir()) / f"full-suite-lock-{user_token}"


def full_suite_lock_path(environ: dict[str, str] | None = None) -> Path:
    """Return the shared lock-file path used across participating repositories."""
    return full_suite_lock_dir(environ) / LOCK_FILENAME


@dataclass(frozen=True)
class SuiteFingerprint:
    payload: dict[str, Any]
    memo_allowed: bool
    refusal_reasons: tuple[str, ...] = ()

    @property
    def key(self) -> str:
        raw = json.dumps(self.payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()


def run_command(args: Sequence[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False)


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def distribution_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def repo_root_from(start: Path) -> Path:
    result = run_command(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=start,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "not inside a Git checkout")
    return Path(result.stdout.strip()).resolve()


def shared_cache_root(repo_root: Path) -> Path:
    result = run_command(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=repo_root,
    )
    if result.returncode != 0:
        return repo_root / ".git" / "feedbax_test_cache"

    common_dir = Path(result.stdout.strip()).resolve()
    return common_dir / "feedbax_test_cache"


def configure_jax_cache_env(cache_root: Path) -> None:
    """Expose the shared test-cache root without forcing one exact JAX cache dir."""
    if "FEEDBAX_JAX_COMPILATION_CACHE_DIR" in os.environ:
        return
    os.environ.setdefault("FEEDBAX_JAX_TEST_CACHE_ROOT", str(cache_root))


def is_execution_relevant_untracked_path(path: str) -> bool:
    return any(
        path == relevant_path.rstrip("/") or path.startswith(relevant_path)
        for relevant_path in EXECUTION_RELEVANT_UNTRACKED_PATHS
    )


def has_memo_blocking_status(status_stdout: str) -> bool:
    for record in status_stdout.split("\0"):
        if not record:
            continue
        status_code = record[:2]
        if status_code != "??":
            return True
        if is_execution_relevant_untracked_path(record[3:]):
            return True
    return False


def clean_tree_hash(repo_root: Path) -> tuple[str | None, tuple[str, ...]]:
    status = run_command(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        cwd=repo_root,
    )
    if status.returncode != 0:
        return None, ("git status failed",)
    if has_memo_blocking_status(status.stdout):
        return None, ("git working tree is dirty",)

    result = run_command(["git", "rev-parse", "HEAD^{tree}"], cwd=repo_root)
    if result.returncode != 0:
        return None, ("git tree hash is unavailable",)
    return result.stdout.strip(), ()


def build_fingerprint(
    repo_root: Path,
    *,
    suite_profile: str = "core",
    pytest_args: Sequence[str] = (),
    runner_args: Sequence[str] = (),
) -> SuiteFingerprint:
    tree_hash, git_refusals = clean_tree_hash(repo_root)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "git_tree": tree_hash,
        "uv_lock_sha256": sha256_file(repo_root / "uv.lock"),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "jax_version": distribution_version("jax"),
        "jaxlib_version": distribution_version("jaxlib"),
        "pytest_version": distribution_version("pytest"),
        "pytest_xdist_version": distribution_version("pytest-xdist"),
        "suite_profile": suite_profile,
        "pytest_args": list(pytest_args),
        "runner_args": list(runner_args),
    }

    refusal_reasons = list(git_refusals)
    if os.environ.get("PYTEST_ADDOPTS"):
        refusal_reasons.append("PYTEST_ADDOPTS is nonempty")
    for field in REQUIRED_FINGERPRINT_FIELDS:
        if not payload.get(field):
            refusal_reasons.append(f"{field} is unavailable")

    return SuiteFingerprint(
        payload=payload,
        memo_allowed=not refusal_reasons,
        refusal_reasons=tuple(refusal_reasons),
    )


def memo_file(memo_dir: Path, fingerprint: SuiteFingerprint) -> Path:
    return memo_dir / f"{fingerprint.key}.json"


def has_green_memo(memo_dir: Path, fingerprint: SuiteFingerprint) -> bool:
    if not fingerprint.memo_allowed:
        return False
    path = memo_file(memo_dir, fingerprint)
    if not path.is_file():
        return False

    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return False
    return data.get("status") == "green" and data.get("fingerprint") == fingerprint.payload


def write_green_memo(
    memo_dir: Path,
    fingerprint: SuiteFingerprint,
    command: Sequence[str],
) -> Path:
    memo_dir.mkdir(parents=True, exist_ok=True)
    path = memo_file(memo_dir, fingerprint)
    tmp_path = path.with_suffix(".tmp")
    data = {
        "status": "green",
        "recorded_at": datetime.now(UTC).isoformat(),
        "fingerprint": fingerprint.payload,
        "command": list(command),
    }
    tmp_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp_path.replace(path)
    return path


def xdist_args() -> list[str]:
    if os.environ.get("FEEDBAX_FULL_SUITE_DISABLE_XDIST") == "1":
        # Explicitly force serial. The pyproject addopts default enables
        # `-n auto`, so an empty runner list would still run in parallel; a
        # trailing `-n 0` overrides it (last `-n` wins) and keeps the serial
        # green-memo fingerprint honest about how the suite actually ran.
        return ["-n", "0"]
    if distribution_version("pytest-xdist") is None:
        return []
    return ["-n", os.environ.get("FEEDBAX_FULL_SUITE_XDIST_WORKERS", "auto")]


def pytest_command(
    pytest_args: Sequence[str],
    *,
    suite_profile: str = "core",
    runner_args: Sequence[str] | None = None,
) -> list[str]:
    marker_expression = SUITE_MARKER_EXPRESSIONS[suite_profile]
    resolved_runner_args = xdist_args() if runner_args is None else runner_args
    return [
        sys.executable,
        "-m",
        "pytest",
        "tests",
        *resolved_runner_args,
        "-m",
        marker_expression,
        *pytest_args,
    ]


def parse_args(argv: Sequence[str] | None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run the Feedbax full pytest suite with xdist, JAX cache, and green-tree memoization."
    )
    parser.add_argument(
        "--force", action="store_true", help="Run even when a matching green memo exists."
    )
    parser.add_argument("--no-memo", action="store_true", help="Disable memo lookup and recording.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the computed command and memo state."
    )
    parser.add_argument(
        "--print-fingerprint",
        action="store_true",
        help="Print the fail-closed memo fingerprint and exit.",
    )
    parser.add_argument(
        "--memo-dir",
        type=Path,
        default=None,
        help="Override the full-suite memo directory.",
    )
    profile = parser.add_mutually_exclusive_group()
    profile.add_argument(
        "--include-mjx",
        action="store_const",
        const="mjx",
        dest="suite_profile",
        help="Run the core suite plus optional MJX simulation tests.",
    )
    profile.add_argument(
        "--include-ppo",
        action="store_const",
        const="ppo",
        dest="suite_profile",
        help="Run the core suite plus optional PPO rollout and training tests.",
    )
    profile.add_argument(
        "--include-optional",
        action="store_const",
        const="all",
        dest="suite_profile",
        help="Run the core, MJX, and PPO test tiers.",
    )
    parser.set_defaults(suite_profile="core")
    namespace, pytest_args = parser.parse_known_args(argv)
    if pytest_args[:1] == ["--"]:
        pytest_args = pytest_args[1:]
    return namespace, pytest_args


def main(argv: Sequence[str] | None = None) -> int:
    args, passthrough = parse_args(argv)
    repo_root = repo_root_from(Path.cwd())
    cache_root = shared_cache_root(repo_root)
    memo_dir = args.memo_dir or Path(
        os.environ.get("FEEDBAX_FULL_SUITE_MEMO_DIR", str(cache_root / "full_suite_memo"))
    )
    configure_jax_cache_env(cache_root)

    runner_args = xdist_args()
    command = pytest_command(
        passthrough,
        suite_profile=args.suite_profile,
        runner_args=runner_args,
    )

    if args.print_fingerprint:
        fingerprint = build_fingerprint(
            repo_root,
            suite_profile=args.suite_profile,
            pytest_args=passthrough,
            runner_args=runner_args,
        )
        print(
            json.dumps(
                {
                    "key": fingerprint.key,
                    "memo_allowed": fingerprint.memo_allowed,
                    "refusal_reasons": fingerprint.refusal_reasons,
                    "fingerprint": fingerprint.payload,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.dry_run:
        fingerprint = build_fingerprint(
            repo_root,
            suite_profile=args.suite_profile,
            pytest_args=passthrough,
            runner_args=runner_args,
        )
        would_skip = not args.force and not args.no_memo and has_green_memo(memo_dir, fingerprint)
        print(
            json.dumps(
                {
                    "command": command,
                    "memo_allowed": fingerprint.memo_allowed,
                    "memo_dir": str(memo_dir),
                    "refusal_reasons": fingerprint.refusal_reasons,
                    "would_skip": would_skip,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    lock_path = full_suite_lock_path()
    try:
        with FullSuiteLock(lock_path, repo_root=repo_root):
            fingerprint = build_fingerprint(
                repo_root,
                suite_profile=args.suite_profile,
                pytest_args=passthrough,
                runner_args=runner_args,
            )
            would_skip = (
                not args.force and not args.no_memo and has_green_memo(memo_dir, fingerprint)
            )

            if fingerprint.refusal_reasons:
                reasons = "; ".join(fingerprint.refusal_reasons)
                print(f"Full-suite memo disabled: {reasons}", file=sys.stderr)

            if would_skip:
                print(f"Skipping full suite: green memo {memo_file(memo_dir, fingerprint)}")
                return 0

            print("Running:", " ".join(command))
            with owned_suite_temporary_root() as temporary_root:
                result = subprocess.run(
                    command,
                    cwd=repo_root,
                    env={**os.environ, "TMPDIR": str(temporary_root)},
                    check=False,
                )
            if result.returncode == 0 and not args.no_memo and fingerprint.memo_allowed:
                path = write_green_memo(memo_dir, fingerprint, command)
                print(f"Recorded full-suite green memo: {path}")
            return result.returncode
    except FullSuiteLockBusy as error:
        print(error, file=sys.stderr, flush=True)
        return LOCK_BUSY_EXIT


if __name__ == "__main__":
    raise SystemExit(main())
