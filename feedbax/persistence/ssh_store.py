"""SSH and batched rsync for a Store on an already connected venue."""

from dataclasses import dataclass, field
import math
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess
from tempfile import TemporaryFile
from typing import BinaryIO

from pydantic import BaseModel

from feedbax.contracts.artifact_custody import ContentObjectRef
from feedbax.persistence.artifact_custody import content_object_relative_path


@dataclass(frozen=True)
class SSHStoreConnection:
    """The venue's existing SSH destination and client options."""

    host: str
    ssh_argv: tuple[str, ...] = ("ssh",)
    rsync_executable: str = "rsync"
    timeout_seconds: float = 60

    def __post_init__(self) -> None:
        if (
            not self.host or self.host.startswith("-") or ":" in self.host or not self.ssh_argv
            or not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0
        ):
            raise ValueError("Expected an SSH destination and client argv")

    def run(self, argv: list[str], payload: bytes) -> bytes:
        return self._run(
            [*self.ssh_argv, self.host, shlex.join(argv)],
            input=payload, timeout=self.timeout_seconds,
        )

    def copy_objects(
        self, source: Path | PurePosixPath, destination: Path | PurePosixPath,
        objects: tuple[ContentObjectRef, ...], *, pull: bool,
    ) -> None:
        if not objects:
            return
        remote = source if pull else destination
        local = destination if pull else source
        remote_arg = f"{self.host}:{shlex.quote(remote.as_posix().rstrip('/') + '/')}"
        local_arg = local.as_posix().rstrip("/") + "/"
        endpoints = (remote_arg, local_arg) if pull else (local_arg, remote_arg)
        with TemporaryFile() as manifest:
            for ref in objects:
                manifest.write(content_object_relative_path(ref.sha256).as_posix().encode("ascii") + b"\n")
            manifest.seek(0)
            # Default rsync writes to temporary names and renames only completed files.
            self._run([
                self.rsync_executable, "--files-from=-", "--ignore-existing",
                f"--timeout={math.ceil(self.timeout_seconds)}",
                "-e", shlex.join(self.ssh_argv), "--", *endpoints,
            ], stdin=manifest, env={**os.environ, "RSYNC_OLD_ARGS": "1", "RSYNC_PROTECT_ARGS": "0"})

    @staticmethod
    def _run(
        argv: list[str], *, input: bytes | None = None, stdin: BinaryIO | None = None,
        timeout: float | None = None, env: dict[str, str] | None = None,
    ) -> bytes:
        try:
            result = subprocess.run(
                argv, capture_output=True, input=input, stdin=stdin, timeout=timeout, env=env,
            )
        except subprocess.TimeoutExpired as error:
            raise TimeoutError(f"Store transport {argv[0]} timed out") from error
        if result.returncode != 0:
            detail = result.stderr.decode(errors="replace").strip()
            raise OSError(f"Store transport {argv[0]} exited {result.returncode}: {detail}")
        return result.stdout


@dataclass(frozen=True)
class RemoteStore:
    """A Store directory reachable via the venue's SSH connection."""

    root: PurePosixPath
    connection: SSHStoreConnection
    python: str = field(default="python", kw_only=True)

    def __post_init__(self) -> None:
        root = PurePosixPath(self.root)
        if not root.is_absolute() or ".." in root.parts:
            raise ValueError("Remote Store requires an absolute path")
        object.__setattr__(self, "root", root)

    def call(self, operation: str, request: BaseModel) -> bytes:
        return self.connection.run(
            [self.python, "-m", "feedbax.persistence.store_sync", operation, str(self.root)],
            request.model_dump_json().encode(),
        )
