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
from contextlib import ExitStack, suppress
from pathlib import Path
from tempfile import mkdtemp
from typing import BinaryIO

from feedbax.desktop.release import AppPaths
from feedbax.host import HostClient
from feedbax.host.io import OWNER_ENV
from feedbax.host.platform import end_with

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


_held: list[int] | None = None


def exit_on_signal(signum: int, _frame: object) -> None:
    if _held is not None:
        _held.append(signum)
        return
    raise SystemExit(128 + signum)


def spawn(scope: ExitStack, command: list[str], **options) -> subprocess.Popen:
    """Start a child that `scope` stops; a signal that arrives meanwhile ends the launcher once it is registered."""
    global _held
    _held = []
    try:
        process = subprocess.Popen(command, **options)
        scope.callback(stop, process)
    finally:
        held, _held = _held, None
    if held:
        raise SystemExit(128 + held[0])
    return process


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


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


def stop_isolated_host(root: Path) -> None:
    if root.exists():
        HostClient(root).shutdown("force")


def seed_jax_cache(shared: Path, host_root: Path) -> None:
    host_root.mkdir(mode=0o700, parents=True)
    subprocess.run(["cp", "-cR", f"{shared}/", str(host_root / "jax")], check=True)
    (host_root / "jax").chmod(0o700)


def publish_jax_cache(host_root: Path, shared: Path) -> None:
    """Link this home's JAX cache entries, which JAX names by content and Feedbax writes whole, into the shared cache."""
    for entry in (host_root / "jax").glob("*-cache"):
        try:
            os.link(entry, shared / entry.name)
        except FileExistsError:
            pass


def stop_host_workers(host_root: Path) -> None:
    """Stop the workers a stopped Host leaves running, which may still write to the home."""
    found = subprocess.run(["pgrep", "-f", str(host_root / "workers")], capture_output=True, text=True)
    pids = [int(pid) for pid in found.stdout.split()]
    for pid in pids:
        with suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)
    wait_until(lambda: not any(process_exists(pid) for pid in pids), f"Studio workers did not exit in {host_root}")


def finish_test_home(
    home: Path, host_root: Path, lease: BinaryIO, *, remove: bool, jax_cache: Path | None
) -> None:
    wait_until(
        lambda: try_test_lock(lease, fcntl.LOCK_EX, TEST_BACKEND_LOCK),
        f"A restarted Studio backend is still using {home}",
        timeout=6,
    )
    stop_isolated_host(host_root)
    stop_host_workers(host_root)
    if jax_cache is not None and (host_root / "jax").is_dir():
        publish_jax_cache(host_root, jax_cache)
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
    if args.jax_cache is not None:
        seed_jax_cache(args.jax_cache, paths.host)
    scope.callback(
        finish_test_home, home, paths.host, lease, remove=args.remove_test_home, jax_cache=args.jax_cache
    )


def serve(home: Path, args: argparse.Namespace, scope: ExitStack, backend_app: str) -> int:
    paths = AppPaths.resolve(instance_root=home)
    port = args.port or (3009 if args.preview or args.preview_from else 3008)
    backend_port = args.backend_port or port + 5000
    if args.owner is not None:
        claim_test_home(home, paths, args, scope)
    paths.configure_studio(desktop=False)
    os.environ.update(
        FEEDBAX_REPO_SNAPSHOT_CACHE_DIR=str(paths.cache / "snapshots"),
        VITE_FEEDBAX_API_PORT=str(backend_port),
        FEEDBAX_VITE_CACHE_DIR=str(paths.cache / "vite"),
        XDG_CACHE_HOME=str(paths.cache),
    )
    os.environ.pop("VITE_FEEDBAX_E2E", None)
    print(f"Studio data home: {home}", flush=True)
    # Saved-project reads observe the host without starting it, so the launcher starts it.
    HostClient(paths.host).ping()

    # The backend ends with this launcher however it ends, SIGKILL included.
    backend = [
        sys.executable, "-B", "-c",
        "import sys; from feedbax.host.platform import end_with; end_with(int(sys.argv.pop(1)));"
        " from uvicorn.main import main; main()",
        str(os.getpid()), backend_app,
    ]
    vite = ["bun", "--bun", "vite"]
    front = vite
    if args.preview_from:
        if not (args.preview_from / "index.html").is_file():
            raise ValueError(f"No Studio preview built at {args.preview_from}")
        front = [*vite, "preview", "--outDir", str(args.preview_from.resolve())]
    if args.preview:
        dist = str(paths.cache / "dist")
        # A walkthrough: its build stamps each element with where it is written, and its pages load the notes toolbar.
        os.environ["FEEDBAX_WALKTHROUGH"] = "1"
        build = spawn(scope, [*vite, "build", "--outDir", dist, "--emptyOutDir"], cwd=WEB)
        build_code = build.wait()
        if build_code:
            return build_code
        front = [*vite, "preview", "--outDir", dist]

    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((HOST, backend_port))
        fd = listener.fileno()
        api = spawn(scope, [*backend, "--fd", str(fd)], cwd=ROOT, pass_fds=(fd,))
    if args.backend_only:
        return api.wait()
    pid_file = paths.cache / "backend.pid"
    pid_file.write_text(f"{api.pid}\n")
    scope.callback(pid_file.unlink, missing_ok=True)
    web = spawn(scope, [*front, "--host", HOST, "--port", str(port), "--strictPort"], cwd=WEB)
    return web.wait()


def main(*, backend_app: str = "feedbax.web.app:app") -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, help="keep Studio data in this directory")
    parser.add_argument("--port", type=int, help="frontend port (default: 3008 or 3009 with --preview)")
    parser.add_argument("--backend-port", type=int, help="backend port (default: the frontend port plus 5000)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preview", action="store_true",
                      help="build and serve the production frontend for a walkthrough: source-stamped, with the notes toolbar")
    mode.add_argument("--preview-from", type=Path, help="serve the existing production frontend build")
    mode.add_argument("--backend-only", action="store_true", help="serve the backend for an existing frontend")
    parser.add_argument(
        "--remove-test-home", action="store_true", help="remove a new test home after its jobs settle"
    )
    parser.add_argument(
        "--jax-cache",
        type=Path,
        help="seed a new test home's JAX compilation cache from this directory and add its new entries back",
    )
    args = parser.parse_args()
    if args.backend_only and args.home is None:
        parser.error("--backend-only requires --home")
    raw_owner = os.environ.get(OWNER_ENV)
    args.owner = None
    if raw_owner is not None:
        if not raw_owner.isdecimal() or int(raw_owner) <= 1:
            parser.error(f"{OWNER_ENV} must identify a test runner process")
        args.owner = int(raw_owner)
    if args.owner is not None and args.home is None:
        parser.error("a test runner requires --home")
    if args.remove_test_home and (args.owner is None or args.backend_only):
        parser.error("--remove-test-home requires the main test launcher")
    if args.jax_cache is not None and (args.owner is None or args.backend_only or not args.jax_cache.is_dir()):
        parser.error("--jax-cache requires the main test launcher and an existing directory")

    signal.signal(signal.SIGINT, exit_on_signal)
    signal.signal(signal.SIGTERM, exit_on_signal)
    if args.owner is not None:
        end_with(args.owner)
    with ExitStack() as scope:
        if args.home is None:
            # Retain the home if cleanup fails and a detached worker may still need it.
            home = Path(mkdtemp(prefix="feedbax-studio-")).resolve()
        else:
            home = args.home.expanduser().resolve()
        try:
            return serve(home, args, scope, backend_app)
        finally:
            # A repeated interrupt does not cut the cleanup short; SIGTERM does.
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            scope.close()
            if args.home is None:
                stop_isolated_host(AppPaths.resolve(instance_root=home).host)
                shutil.rmtree(home)


if __name__ == "__main__":
    sys.exit(main())
