"""Run Studio from a checkout against its own loopback backend and data home."""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

from feedbax.desktop.release import AppPaths

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
HOST = "127.0.0.1"


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


def serve(args: argparse.Namespace, scope: ExitStack) -> int:
    if args.home is None:
        home = Path(scope.enter_context(TemporaryDirectory(prefix="feedbax-studio-"))).resolve()
    else:
        home = args.home.expanduser().resolve()
    paths = AppPaths.resolve(instance_root=home)
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
    if args.backend_only:
        os.execv(sys.executable, [*backend, "--host", HOST, "--port", str(args.backend_port)])

    vite = ["bun", "--bun", "vite"]
    front = vite
    if args.preview:
        dist = str(paths.cache / "dist")
        subprocess.run([*vite, "build", "--outDir", dist, "--emptyOutDir"], cwd=WEB, check=True)
        front = [*vite, "preview", "--outDir", dist]
    port = args.port or (3009 if args.preview else 3008)

    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((HOST, args.backend_port))
        fd = listener.fileno()
        api = subprocess.Popen([*backend, "--fd", str(fd)], cwd=ROOT, pass_fds=(fd,))
    scope.callback(stop, api)
    pid_file = paths.cache / "backend.pid"
    pid_file.write_text(f"{api.pid}\n")
    scope.callback(pid_file.unlink, missing_ok=True)
    web = subprocess.Popen([*front, "--host", HOST, "--port", str(port), "--strictPort"], cwd=WEB)
    scope.callback(stop, web)
    return web.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, help="keep Studio data in this directory")
    parser.add_argument("--port", type=int, help="frontend port (default: 3008 or 3009 with --preview)")
    parser.add_argument("--backend-port", type=int, default=8000)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preview", action="store_true", help="build and serve the production frontend")
    mode.add_argument("--backend-only", action="store_true", help="serve the backend for an existing frontend")
    args = parser.parse_args()
    if args.backend_only and args.home is None:
        parser.error("--backend-only requires --home")

    signal.signal(signal.SIGINT, exit_on_signal)
    signal.signal(signal.SIGTERM, exit_on_signal)
    with ExitStack() as scope:
        return serve(args, scope)


if __name__ == "__main__":
    sys.exit(main())
