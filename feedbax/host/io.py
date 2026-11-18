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
# The persistent JAX compilation cache of the Host's workers; `<root>/jax` when unset.
JAX_CACHE_ENV = "FEEDBAX_HOST_JAX_CACHE_DIR"
EVIDENCE_FILE = "execution-evidence.json"


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


def read_worker_result(path: Path, *, truncate_incomplete: bool = False) -> dict | None:
    """Find the last complete result, optionally removing a torn final append."""
    with path.open("r+b" if truncate_incomplete else "rb") as stream:
        tail_start = max(0, stream.seek(0, os.SEEK_END) - 2 * 1024 * 1024)
        stream.seek(tail_start)
        tail = stream.read()
        end = tail.rfind(b"\n")
        if end == -1:
            return None
        if truncate_incomplete and end != len(tail) - 1:
            stream.truncate(tail_start + end + 1)
        for raw in reversed(tail[:end].split(b"\n")):
            try:
                event = json.loads(raw)
            except (ValueError, UnicodeError):
                continue
            if isinstance(event, dict) and event.get("kind") == "result":
                value = event.get("data")
                return value if isinstance(value, dict) else None
    return None


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
