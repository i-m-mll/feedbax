"""Identify the package bytes supplying a local Host."""

from hashlib import sha256
from pathlib import Path
from typing import TypedDict


class HostSource(TypedDict):
    package: str
    sha256: str


def host_source() -> HostSource:
    package = Path(__file__).resolve().parents[1]
    identity = sha256()
    for path in sorted(package.rglob("*")):
        if path.suffix in {".py", ".json", ".yaml", ".yml", ".feedbax", ".sh"} and path.is_file():
            identity.update(path.relative_to(package).as_posix().encode())
            identity.update(b"\0")
            identity.update(sha256(path.read_bytes()).digest())
    return {"package": str(package), "sha256": identity.hexdigest()}
