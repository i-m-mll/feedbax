"""Installed Studio service and owner-only native control entrypoint."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import json
import os
import platform
import signal
import socket
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path

from feedbax.desktop.control import CONTROL_ROUTE, Endpoint, bind_studio, connect
from feedbax.desktop.release import SERVICE_PROTOCOL, AppPaths, Release
from feedbax.host import HostClient, HostError


@contextmanager
def _service_lock(directory: Path):
    fd = os.open(directory / "owner.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another Studio service owns this instance") from exc
        yield
    finally:
        os.close(fd)


def _register_stop(app, server) -> None:
    from fastapi import HTTPException

    lock = asyncio.Lock()

    @app.post(CONTROL_ROUTE + "/stop")
    async def stop(request: dict):
        disposition = request.get("disposition")
        if disposition not in ("defer", "checkpoint", "force"):
            raise HTTPException(422, "select defer, checkpoint, or force")
        async with lock:
            if server.should_exit:
                raise HTTPException(409, "Studio service is already stopping")
            result = await app.state.desktop_jobs.stop(disposition=disposition)
            if result["status"] == "ready_to_stop":
                server.should_exit = True
            return result


async def _serve_until_stopped(server, app, sockets, stop_requested: asyncio.Event) -> None:
    async def interrupt() -> None:
        await stop_requested.wait()
        while not server.started:
            await asyncio.sleep(0.05)
        result = await app.state.desktop_jobs.stop("interrupt", timeout_seconds=10)
        if result["status"] != "ready_to_stop":
            raise RuntimeError(f"Studio could not release its background runtime: {result}")
        server.should_exit = True

    serving = asyncio.create_task(server.serve(sockets=sockets))
    stopping = asyncio.create_task(interrupt())
    try:
        done, _ = await asyncio.wait((serving, stopping), return_when=asyncio.FIRST_COMPLETED)
        if stopping in done:
            await stopping
        await serving
    finally:
        for task in (serving, stopping):
            if not task.done():
                task.cancel()
        await asyncio.gather(serving, stopping, return_exceptions=True)


async def _serve(release: Release, paths: AppPaths, studio_socket: socket.socket) -> None:
    from hashlib import sha256
    from importlib.metadata import version

    import jax
    import jaxlib
    import uvicorn
    from starlette.staticfiles import StaticFiles

    from feedbax.desktop.activity import register_activity
    from feedbax.desktop.preferences import register_preferences
    from feedbax.desktop.transport import DesktopTransport
    from feedbax.orchestration.revision import assert_feedbax_revision_exact
    from feedbax.web.app import create_app

    revision = assert_feedbax_revision_exact(release.document["source_commit"])
    backend = jax.default_backend()
    if backend != "cpu" or platform.machine() != "arm64":
        raise RuntimeError("this Studio release requires the JAX CPU backend on Apple Silicon")
    runtime: dict[str, object] = {
        "python": str(Path(sys.executable).resolve(strict=True)),
        "python_version": platform.python_version(),
        "source_commit": revision,
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "backend": backend,
    }
    dependency_bytes = release.resource("dependencies").read_bytes()
    if sha256(dependency_bytes).hexdigest() != release.document["dependency_manifest_sha256"]:
        raise RuntimeError("Studio dependency manifest does not match the installed release")
    dependencies = json.loads(dependency_bytes)
    profile = dependencies.get("profile", "jax-cpu")
    if profile not in ("jax-cpu", "mjx-cpu"):
        raise RuntimeError(f"unknown Studio dependency profile: {profile!r}")
    runtime["profile"] = profile
    if profile == "mjx-cpu":
        distributions = {name: version(name) for name in ("mujoco-core", "mujoco-mjx")}
        numerical = {"jax": jax.__version__, "jaxlib": jaxlib.__version__, **distributions}
        if any(dependencies["wheels"][name]["version"] != actual for name, actual in numerical.items()):
            raise RuntimeError("Studio numerical distributions do not match the installed release")
        runtime["physics"] = {
            "provider": "mujoco.mjx",
            "implementation": "jax",
            "device": backend,
            "distributions": distributions,
        }
    app = create_app(desktop_job_root=paths.state / "jobs", desktop_release=release.contents)
    endpoint = Endpoint.create(studio_socket.getsockname()[1])
    transport = DesktopTransport(app, endpoint)
    class Server(uvicorn.Server):
        def capture_signals(self):
            return nullcontext()

    server = Server(
        uvicorn.Config(
            transport,
            access_log=False,
            proxy_headers=False,
            timeout_graceful_shutdown=5,
            lifespan="on",
        )
    )

    @app.get(CONTROL_ROUTE + "/health")
    async def health():
        return {
            "status": "ready" if not server.should_exit else "stopping",
            "protocol": SERVICE_PROTOCOL,
            "release": release.digest,
            "installed_path": str(release.contents),
            "origin": endpoint.origin,
            "generation": endpoint.generation,
            "pid": os.getpid(),
            "runtime": runtime,
        }

    transport.register_ticket_route(lambda: server.should_exit)
    register_activity(app, release.digest)

    @app.get("/api/desktop/jobs")
    async def jobs():
        return app.state.desktop_jobs.snapshot()

    register_preferences(app, paths.settings / "studio.json")

    _register_stop(app, server)

    app.mount("/", StaticFiles(directory=release.resource("studio"), html=True), name="studio")
    control_socket_path = paths.control / "service.sock"
    control_socket_path.unlink(missing_ok=True)
    stop_requested = asyncio.Event()
    loop = asyncio.get_running_loop()
    signals = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
    for signum in signals:
        loop.add_signal_handler(signum, stop_requested.set)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as control_socket:
        try:
            control_socket.bind(str(control_socket_path))
            control_socket_path.chmod(0o600)
            await _serve_until_stopped(server, app, [control_socket, studio_socket], stop_requested)
            if not server.started:
                raise RuntimeError("Studio failed before entering its ready state")
        finally:
            control_socket_path.unlink(missing_ok=True)
            for signum in signals:
                loop.remove_signal_handler(signum)


def _retire_host(release: Release, paths: AppPaths, disposition: str) -> dict:
    import time
    host = HostClient(paths.host)
    deadline = time.monotonic() + (7 if disposition == "interrupt" else 30)
    while True:
        try:
            result = host.retire_runtime(str(release.resource("python")), str(release.contents), disposition)
        except HostError as error:
            if error.code == "invalid_request":
                raise RuntimeError(
                    "The running Host does not support installed-runtime retirement. "
                    "Stop that Host before replacing or uninstalling Studio."
                ) from error
            raise
        if result["status"] in ("absent", "released"):
            return {"status": "ready_to_stop", "disposition": disposition}
        if result["status"] == "deferred":
            return result
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Studio could not release its background Host: {result}")
        time.sleep(0.1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contents", type=Path, required=True)
    parser.add_argument("action", choices=("serve", "paths", "health", "ticket", "stop", "retire-host"))
    parser.add_argument("--wait", type=float, default=0)
    parser.add_argument("--disposition", choices=("defer", "checkpoint", "force", "interrupt"), default="defer")
    project = parser.add_mutually_exclusive_group()
    project.add_argument("--project-folder")
    project.add_argument("--project-file")
    project.add_argument("--project-id")
    project.add_argument("--activity", action="store_true", default=None)
    parser.add_argument("--folder-identity", nargs=2, type=int, metavar=("DEVICE", "INODE"))
    args = parser.parse_args()
    if args.action != "ticket" and any(
        value is not None for value in (
            args.project_folder, args.project_id, args.project_file, args.folder_identity, args.activity,
        )
    ):
        parser.error("project or Activity selection requires ticket")
    if args.folder_identity is not None and args.project_folder is None:
        parser.error("--folder-identity requires --project-folder")
    if args.disposition == "interrupt" and args.action != "retire-host":
        parser.error("the interrupt disposition is reserved for runtime retirement")
    os.umask(0o077)
    release = Release.load(args.contents)
    if Path(sys.executable).resolve(strict=True) != release.resource("python"):
        raise RuntimeError("Studio must execute its installed Python runtime")
    paths = AppPaths.resolve(instance_root=release.fixture_root)
    paths.configure_studio()
    if args.action == "paths":
        storage_id = hashlib.sha256(os.fsencode(paths.settings)).digest()[:16]
        print(json.dumps({
            "logs": str(paths.logs),
            "settings": str(paths.settings),
            "control": str(paths.control),
            "projects": str(paths.web / "projects"),
            "storage_id": list(storage_id),
            "release": release.digest,
        }))
        return
    if args.action == "retire-host":
        with _service_lock(paths.control):
            print(json.dumps(_retire_host(release, paths, args.disposition)))
        return
    if args.action != "serve":
        print(json.dumps(connect(
            release, paths, seconds=args.wait, action=args.action, disposition=args.disposition,
            project_folder=args.project_folder, project_id=args.project_id,
            project_file=args.project_file, activity=args.activity,
            folder_identity=tuple(args.folder_identity) if args.folder_identity is not None else None,
        )))
        return
    os.environ["FEEDBAX_DESKTOP_RELEASE_ROOT"] = str(release.contents)
    os.chdir(paths.workspace)
    with _service_lock(paths.control), bind_studio(paths) as studio_socket:
        asyncio.run(_serve(release, paths, studio_socket))


if __name__ == "__main__":
    main()
