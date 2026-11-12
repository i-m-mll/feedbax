"""Owner-only endpoint discovery and the native client's local control protocol."""

from __future__ import annotations

import http.client
import json
import os
import secrets
import socket
import stat
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from feedbax.desktop.release import SERVICE_PROTOCOL, AppPaths, Release

CONTROL_ROUTE = "/_desktop"
_PORT = "port.json"


def write_private_json(path: Path, value: object) -> None:
    fd, filename = tempfile.mkstemp(prefix=".write-", dir=path.parent)
    temporary = Path(filename)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _read_private_json(path: Path) -> dict:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_mode & 0o077
            or info.st_size > 8192
        ):
            raise PermissionError(f"invalid private desktop service record: {path}")
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"invalid desktop service record: {path}")
    return value


@dataclass(frozen=True)
class Endpoint:
    port: int
    generation: str

    @property
    def origin(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @classmethod
    def create(cls, port: int) -> Endpoint:
        return cls(port, secrets.token_hex(16))


class _LocalConnection(http.client.HTTPConnection):
    def __init__(self, path: Path, *, timeout: float) -> None:
        super().__init__("localhost", timeout=timeout)
        self.path = path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(str(self.path))


def call(paths: AppPaths, action: str, *, body: dict | None = None, timeout: float = 5) -> dict:
    connection = _LocalConnection(paths.control / "service.sock", timeout=timeout)
    try:
        connection.request(
            "POST" if body is not None else "GET",
            f"{CONTROL_ROUTE}/{action}",
            body=json.dumps(body) if body is not None else None,
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        data = json.load(response)
        if response.status >= 400:
            raise ValueError(data.get("detail", f"local service HTTP {response.status}"))
        return data
    finally:
        connection.close()


def bind_studio(paths: AppPaths) -> socket.socket:
    port_path = paths.control / _PORT
    try:
        port = _read_private_json(port_path)["port"]
    except FileNotFoundError:
        port = 0
    else:
        if type(port) is not int or not 1024 <= port <= 65535:
            raise ValueError("invalid saved desktop Studio port")
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(("127.0.0.1", port))
        if not port:
            write_private_json(port_path, {"port": server_socket.getsockname()[1]})
    except BaseException:
        server_socket.close()
        raise
    return server_socket


def connect(
    release: Release, paths: AppPaths, *, seconds: float, action: str, disposition: str,
    project_folder: str | None = None, project_id: str | None = None,
) -> dict:
    deadline = time.monotonic() + seconds
    while True:
        try:
            health = call(paths, "health", timeout=1)
        except (FileNotFoundError, ConnectionError, TimeoutError):
            if time.monotonic() >= deadline:
                return {"status": "unavailable"}
            time.sleep(0.15)
            continue
        if (
            health["release"] != release.digest
            or health["protocol"] != SERVICE_PROTOCOL
            or health["installed_path"] != str(release.contents)
        ):
            return {"status": "release_mismatch", "service": health}
        if action == "ticket":
            body = {key: value for key, value in (
                ("project_folder", project_folder), ("project_id", project_id),
            ) if value is not None}
            result = call(paths, "ticket", body=body)
            token = result.pop("ticket")
            return {**health, **result, "url": f"{health['origin']}{CONTROL_ROUTE}/connect#{token}"}
        if action == "stop":
            return call(paths, "stop", body={"disposition": disposition}, timeout=50)
        return health
