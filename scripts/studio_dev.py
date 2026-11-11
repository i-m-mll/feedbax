"""Run Studio from a checkout against its own loopback backend and data home."""

from __future__ import annotations

import argparse
import errno
import fcntl
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import BinaryIO

from feedbax.desktop.release import AppPaths
from feedbax.host import HostClient

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
HOST = "127.0.0.1"
TEST_OWNER_LOCK = 0
TEST_BACKEND_LOCK = 1


def stop(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def exit_on_signal(signum: int, _frame: object) -> None:
    raise SystemExit(128 + signum)


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def wait_for_exit(process: subprocess.Popen, owner: int | None) -> int:
    if owner is None:
        return process.wait()
    while True:
        if not process_exists(owner):
            raise SystemExit(0)
        try:
            return process.wait(timeout=0.25)
        except subprocess.TimeoutExpired:
            continue


def wait_until(check: Callable[[], bool], description: str, *, timeout: float = 5) -> None:
    deadline = time.monotonic() + timeout
    while not check():
        if time.monotonic() >= deadline:
            raise TimeoutError(description)
        time.sleep(0.05)


def try_test_lock(lease: BinaryIO, mode: int, offset: int) -> bool:
    try:
        fcntl.lockf(lease, mode | fcntl.LOCK_NB, 1, offset)
    except OSError as error:
        if error.errno in {errno.EACCES, errno.EAGAIN}:
            return False
        raise
    return True


def stop_test_host(root: Path) -> None:
    if not root.exists():
        return
    host = HostClient(root)
    active = [job["id"] for job in host.jobs() if job["outcome"] is None]
    if active:
        host.stop(active, "force")
        wait_until(
            lambda: all(job["outcome"] is not None for job in host.jobs()),
            f"Studio test jobs did not settle in {root}",
        )
    pid = host.ping()["pid"]
    if type(pid) is not int or pid <= 1:
        raise RuntimeError(f"Invalid Studio test host PID in {root}")
    os.kill(pid, signal.SIGTERM)
    wait_until(lambda: not process_exists(pid), f"Studio test host did not exit in {root}")


def finish_test_home(home: Path, host_root: Path, lease: BinaryIO, *, remove: bool) -> None:
    wait_until(
        lambda: try_test_lock(lease, fcntl.LOCK_EX, TEST_BACKEND_LOCK),
        f"A restarted Studio backend is still using {home}",
        timeout=6,
    )
    stop_test_host(host_root)
    if remove:
        shutil.rmtree(home)


def claim_test_home(home: Path, paths: AppPaths, args: argparse.Namespace, scope: ExitStack) -> None:
    if paths.host.resolve() != paths.host:
        raise ValueError(f"Studio test host must be contained in {home}")
    lease_path = home / ".studio-test.lock"
    if args.backend_only:
        lease = scope.enter_context(lease_path.open("rb"))
        if not try_test_lock(lease, fcntl.LOCK_SH, TEST_BACKEND_LOCK) or try_test_lock(
            lease, fcntl.LOCK_SH, TEST_OWNER_LOCK
        ):
            raise RuntimeError(f"No active Studio test launcher owns {home}")
        return
    home.mkdir(mode=0o700, parents=True, exist_ok=not args.remove_test_home)
    lease = scope.enter_context(lease_path.open("a+b"))
    if not try_test_lock(lease, fcntl.LOCK_EX, TEST_OWNER_LOCK):
        raise RuntimeError(f"Another Studio test launcher owns {home}")
    scope.callback(finish_test_home, home, paths.host, lease, remove=args.remove_test_home)


def serve(args: argparse.Namespace, scope: ExitStack) -> int:
    if args.home is None:
        home = Path(scope.enter_context(TemporaryDirectory(prefix="feedbax-studio-"))).resolve()
    else:
        home = args.home.expanduser().resolve()
    paths = AppPaths.resolve(instance_root=home)
    if args.owner is not None:
        claim_test_home(home, paths, args, scope)
    paths.configure_studio(desktop=False)
    os.environ.update(
        FEEDBAX_REPO_SNAPSHOT_CACHE_DIR=str(paths.cache / "snapshots"),
        VITE_FEEDBAX_API_PORT=str(args.backend_port),
        FEEDBAX_VITE_CACHE_DIR=str(paths.cache / "vite"),
        XDG_CACHE_HOME=str(paths.cache),
    )
    os.environ.pop("FEEDBAX_WORKER_URL", None)
    os.environ.pop("VITE_FEEDBAX_E2E", None)
    print(f"Studio data home: {home}", flush=True)

    backend = [sys.executable, "-B", "-m", "uvicorn", "feedbax.web.app:app"]
    vite = ["bun", "--bun", "vite"]
    front = vite
    if args.preview:
        dist = str(paths.cache / "dist")
        build = subprocess.Popen([*vite, "build", "--outDir", dist, "--emptyOutDir"], cwd=WEB)
        scope.callback(stop, build)
        build_code = wait_for_exit(build, args.owner)
        if build_code:
            return build_code
        front = [*vite, "preview", "--outDir", dist]
    port = args.port or (3009 if args.preview else 3008)

    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((HOST, args.backend_port))
        fd = listener.fileno()
        api = subprocess.Popen([*backend, "--fd", str(fd)], cwd=ROOT, pass_fds=(fd,))
    scope.callback(stop, api)
    if args.backend_only:
        return wait_for_exit(api, args.owner)
    pid_file = paths.cache / "backend.pid"
    pid_file.write_text(f"{api.pid}\n")
    scope.callback(pid_file.unlink, missing_ok=True)
    web = subprocess.Popen([*front, "--host", HOST, "--port", str(port), "--strictPort"], cwd=WEB)
    scope.callback(stop, web)
    return wait_for_exit(web, args.owner)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, help="keep Studio data in this directory")
    parser.add_argument("--port", type=int, help="frontend port (default: 3008 or 3009 with --preview)")
    parser.add_argument("--backend-port", type=int, default=8000)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preview", action="store_true", help="build and serve the production frontend")
    mode.add_argument("--backend-only", action="store_true", help="serve the backend for an existing frontend")
    parser.add_argument(
        "--remove-test-home", action="store_true", help="remove a new test home after its jobs settle"
    )
    args = parser.parse_args()
    if args.backend_only and args.home is None:
        parser.error("--backend-only requires --home")
    raw_owner = os.environ.get("FEEDBAX_STUDIO_TEST_OWNER_PID")
    args.owner = None
    if raw_owner is not None:
        if not raw_owner.isdecimal() or int(raw_owner) <= 1:
            parser.error("FEEDBAX_STUDIO_TEST_OWNER_PID must identify a test runner process")
        args.owner = int(raw_owner)
    if args.owner is not None and args.home is None:
        parser.error("a test runner requires --home")
    if args.remove_test_home and (args.owner is None or args.backend_only):
        parser.error("--remove-test-home requires the main test launcher")

    signal.signal(signal.SIGINT, exit_on_signal)
    signal.signal(signal.SIGTERM, exit_on_signal)
    with ExitStack() as scope:
        try:
            return serve(args, scope)
        finally:
            if args.owner is not None:
                signal.signal(signal.SIGINT, signal.SIG_IGN)
                signal.signal(signal.SIGTERM, signal.SIG_IGN)


if __name__ == "__main__":
    sys.exit(main())
