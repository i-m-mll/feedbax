"""File and framing primitives for the host and detached workers."""

import hashlib
import json
import os
import secrets
import socket
import stat
from pathlib import Path

from .protocol import JsonObject

TOKEN_ENV = "FEEDBAX_PROCESS_IDENTITY"


def encode(value: JsonObject) -> bytes:
    return (json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n").encode()


def private_directory(path: Path) -> Path:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise PermissionError(f"host directory must be private to the current user: {path}")
    return path


def socket_path(root: Path) -> Path:
    # macOS sockaddr_un allows 103 path bytes. Hash only the address, not the state root.
    socket_dir = private_directory(Path(f"/tmp/feedbax-host-{os.getuid()}"))
    digest = hashlib.sha256(os.fsencode(root)).hexdigest()[:32]
    return socket_dir / (digest + ".sock")


def sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_json(path: Path, value: JsonObject, *, exclusive: bool = False) -> bool:
    """Publish complete JSON bytes at once. An exclusive write arbitrates accept/revoke."""
    temp = path.parent / f".{path.name}.{secrets.token_hex(8)}"
    try:
        with temp.open("xb") as stream:
            stream.write(encode(value))
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            try:
                os.link(temp, path)
            except FileExistsError:
                return False
        else:
            os.replace(temp, path)
        sync_directory(path.parent)
        return True
    finally:
        temp.unlink(missing_ok=True)


def append(path: Path, kind: str, data: JsonObject, *, sync: bool = False) -> None:
    """Append one event with one write. Sync publications and outcomes before returning."""
    data_bytes = encode({"kind": kind, "data": data})
    fd = os.open(path, os.O_WRONLY | os.O_APPEND)
    try:
        if os.write(fd, data_bytes) != len(data_bytes):
            raise OSError(f"short event write to {path}")
        if sync:
            os.fsync(fd)
    finally:
        os.close(fd)


def receive(sock: socket.socket, limit: int) -> JsonObject:
    with sock.makefile("rb") as stream:
        line = stream.readline(limit + 1)
    if len(line) > limit:
        raise ValueError("host message exceeds size limit")
    if not line.endswith(b"\n"):
        raise EOFError("host socket closed before a complete message")
    value = json.loads(line)
    if not isinstance(value, dict):
        raise TypeError("host messages must be JSON objects")
    return value
