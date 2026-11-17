"""Supervise one venue from a trusted bootstrap, before Feedbax is installed."""

import argparse
import fcntl
import json
import os
import signal
import stat
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path


_POLL_SECONDS = 1
_TERMINATE_TIMEOUT = 10
_MAX_CONFIG_BYTES = 32 * 1024


def _absolute_path(value: str) -> Path:
    if not isinstance(value, str) or "\x00" in value or not Path(value).is_absolute():
        raise ValueError("Watchdog paths must be absolute.")
    return Path(value)


@dataclass(frozen=True, slots=True)
class _Config:
    ends_at_ms: int
    host_root: Path
    host_python: Path
    ready_file: Path
    terminate_argv: tuple[str, ...]

    @classmethod
    def load(cls, path: Path) -> "_Config":
        directory = path.parent.stat()
        if directory.st_uid != os.getuid() or directory.st_mode & 0o077:
            raise PermissionError("Watchdog state must be private to the venue bootstrap user.")
        with _private_file(path, os.O_RDONLY) as fd:
            raw = os.read(fd, _MAX_CONFIG_BYTES + 1)
        if len(raw) > _MAX_CONFIG_BYTES:
            raise ValueError("Watchdog configuration exceeds the size limit.")
        data = json.loads(raw)
        if not isinstance(data, dict) or data.keys() != {
            "ends_at_ms", "host_root", "host_python", "ready_file", "terminate_argv",
        }:
            raise ValueError("Invalid venue watchdog configuration fields.")
        deadline = data["ends_at_ms"]
        argv = data["terminate_argv"]
        if type(deadline) is not int or not 0 < deadline < 2**53:
            raise ValueError("The venue cutoff must be a Unix timestamp in integer milliseconds.")
        if not isinstance(argv, list) or not argv or any(
            not isinstance(arg, str) or "\x00" in arg for arg in argv
        ):
            raise ValueError("The venue requires a fixed, authenticated termination command.")
        _absolute_path(argv[0])
        return cls(
            deadline, _absolute_path(data["host_root"]), _absolute_path(data["host_python"]),
            _absolute_path(data["ready_file"]), tuple(argv),
        )


@contextmanager
def _private_file(path: Path, flags: int) -> Iterator[int]:
    fd = os.open(path, flags | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise PermissionError("Watchdog files must be private regular files.")
        yield fd
    finally:
        os.close(fd)


def _try_lock(fd: int, kind: int) -> bool:
    try:
        fcntl.flock(fd, kind | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


def _start_host(config: _Config) -> subprocess.Popen | None:
    try:
        ready = config.ready_file.lstat()
        if (
            not stat.S_ISREG(ready.st_mode) or ready.st_uid != os.getuid()
            or ready.st_mode & 0o077 or ready.st_size != 0
        ):
            raise PermissionError("The venue's Host readiness marker must be a private, empty file.")
        # Share the Host's singleton lock convention so a restarted watchdog
        # leaves an existing detached Host and its deliveries alone.
        fd = os.open(config.host_root / "host.lock", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            if not _try_lock(fd, fcntl.LOCK_SH):
                return None
        finally:
            os.close(fd)
        with (config.host_root / "host.log").open("ab") as log:
            return subprocess.Popen(
                [
                    str(config.host_python), "-B", "-m", "feedbax.host", "--root", str(config.host_root),
                    "serve", "--require-session-until-ms", str(config.ends_at_ms),
                ],
                stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True,
                env={name: value for name, value in os.environ.items() if name != "RUNPOD_API_KEY"},
            )
    except FileNotFoundError:
        return None
    except OSError as error:
        print(f"Venue Host start failed ({error.__class__.__name__}); retrying.", file=sys.stderr, flush=True)
        return None


def _record_cutoff(path: Path) -> None:
    try:
        with _private_file(path, os.O_WRONLY | os.O_CREAT) as fd:
            os.fsync(fd)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except OSError:
        print("Could not persist cutoff; proceeding with venue removal.", file=sys.stderr, flush=True)


def _request_termination(argv: tuple[str, ...]) -> bool:
    try:
        with subprocess.Popen(
            argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        ) as process:
            try:
                return process.wait(timeout=_TERMINATE_TIMEOUT) == 0
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
    except OSError:
        return False
    return False


def _supervise(config: _Config, *, cutoff: Path) -> None:
    deadline = config.ends_at_ms / 1000
    monotonic_deadline = time.monotonic() + (deadline - time.time())
    host = None
    while not cutoff.exists():
        remaining = min(deadline - time.time(), monotonic_deadline - time.monotonic())
        if remaining <= 0:
            break
        if host is None or host.poll() is not None:
            host = _start_host(config)
        time.sleep(min(_POLL_SECONDS, remaining))

    _record_cutoff(cutoff)
    while not _request_termination(config.terminate_argv):
        print("Venue termination was not confirmed; retrying.", file=sys.stderr, flush=True)
        time.sleep(_POLL_SECONDS)


def _run(config_path: Path, config: _Config, *, detach: bool) -> None:
    lock = config_path.with_suffix(".lock")
    if detach:
        with _private_file(config_path.with_suffix(".log"), os.O_WRONLY | os.O_CREAT | os.O_APPEND) as log:
            candidate = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "--config", str(config_path)],
                stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                start_new_session=True, close_fds=True,
            )
        deadline = time.monotonic() + 5
        with _private_file(lock, os.O_RDONLY | os.O_CREAT) as fd:
            while _try_lock(fd, fcntl.LOCK_SH):
                fcntl.flock(fd, fcntl.LOCK_UN)
                if config_path.with_suffix(".cutoff").exists() and candidate.poll() == 0:
                    return
                if candidate.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError("The detached venue watchdog did not start.")
                time.sleep(.02)
        return

    with _private_file(lock, os.O_RDWR | os.O_CREAT) as fd:
        if _try_lock(fd, fcntl.LOCK_EX):
            _supervise(config, cutoff=config_path.with_suffix(".cutoff"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--detach", action="store_true")
    options = parser.parse_args()
    try:
        config_path = options.config.absolute()
        cfg = _Config.load(config_path)
        _run(config_path, cfg, detach=options.detach)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
