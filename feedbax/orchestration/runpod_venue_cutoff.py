"""Arm the venue watchdog at RunPod boot using only the pod's own credential."""

import argparse
import fcntl
import hashlib
import http.client
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


_INTENT = re.compile(r"feedbax-session-[0-9a-f]{64}")
_POD_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_TEST_ORIGIN = re.compile(r"http://127\.0\.0\.1:([0-9]{1,5})")
_TEST_KEY = "feedbax-local-test-pod-key"
_MAX_PROVIDER_REPLY = 1024 * 1024
_HTTP_TIMEOUT = 3


class _GuardError(RuntimeError):
    pass


def _encode(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


def _private_root(root: Path) -> None:
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    _check_private_root(root)


def _check_private_root(root: Path) -> None:
    info = root.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise _GuardError("RunPod guard requires a private venue root.")


def _read_private(path: Path) -> bytes:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise _GuardError("RunPod guard state must be a private regular file.")
        return stream.read()


def _publish(path: Path, data: bytes) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent) as staged:
        staged.write(data)
        staged.flush()
        os.fsync(staged.fileno())
        try:
            os.link(staged.name, path)
        except FileExistsError:
            if _read_private(path) != data:
                raise _GuardError("Existing RunPod guard state identifies a different boot configuration.")
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def _unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    fields: dict[str, object] = {}
    for key, value in pairs:
        if key in fields:
            raise _GuardError("RunPod returned ambiguous pod identity.")
        fields[key] = value
    return fields


@dataclass(frozen=True)
class _Venue:
    workspace: Path
    intent: str
    ends_at_ms: int
    pod_id: str

    def __post_init__(self) -> None:
        if (
            not self.workspace.is_absolute() or ".." in self.workspace.parts
            or not _INTENT.fullmatch(self.intent) or not _POD_ID.fullmatch(self.pod_id)
            or not 0 < self.ends_at_ms < 2**53
        ):
            raise _GuardError("RunPod guard requires an exact pod intent, identity and cutoff.")

    @property
    def root(self) -> Path:
        return self.workspace / "feedbax-sessions" / self.pod_id

    @property
    def proof(self) -> dict[str, str | int]:
        return {"version": 1, "pod_id": self.pod_id, "intent": self.intent, "ends_at_ms": self.ends_at_ms}

    @property
    def paths(self) -> tuple[Path, Path, Path]:
        root = self.root
        return root / "cutoff_guard.py", root / "watchdog.py", root / "watchdog.json"

    def watchdog_policy(self, test_origin: str | None) -> bytes:
        guard, _, _ = self.paths
        command = [
            sys.executable, "-I", str(guard), "terminate", "--workspace", str(self.workspace),
            "--intent", self.intent, "--ends-at-ms", str(self.ends_at_ms), "--pod-id", self.pod_id,
        ]
        if test_origin is not None:
            command.extend(["--test-origin", test_origin])
        return _encode({
            "ends_at_ms": self.ends_at_ms, "host_root": str(self.root / "host"),
            "host_python": str(self.root / "venv" / "bin" / "python"), "ready_file": str(self.root / "ready"),
            "terminate_argv": command,
        })


def _pod_key(venue: _Venue) -> str:
    key = os.environ.get("RUNPOD_API_KEY", "")
    if os.environ.get("RUNPOD_POD_ID") != venue.pod_id or not re.fullmatch(r"[A-Za-z0-9._~+/-]+=*", key):
        raise _GuardError("RunPod boot did not supply its own pod identity and scoped credential.")
    return key


def _provider_call(venue: _Venue, method: str, key: str, test_origin: str | None) -> tuple[int, bytes]:
    if test_origin is None:
        connection = http.client.HTTPSConnection("api.runpod.io", timeout=_HTTP_TIMEOUT)
    else:
        match = _TEST_ORIGIN.fullmatch(test_origin)
        if match is None or not 0 < int(match[1]) <= 65535 or key != _TEST_KEY:
            raise _GuardError("The loopback RunPod test origin requires the synthetic test credential.")
        connection = http.client.HTTPConnection("127.0.0.1", int(match[1]), timeout=_HTTP_TIMEOUT)
    try:
        connection.request(method, "/v2/pods/" + venue.pod_id, headers={
            "Authorization": f"Bearer {key}", "Accept": "application/json",
        })
        response = connection.getresponse()
        body = response.read(_MAX_PROVIDER_REPLY + 1) if method == "GET" and response.status == 200 else b""
        if len(body) > _MAX_PROVIDER_REPLY:
            raise _GuardError("RunPod pod identity response exceeded its size limit.")
        return response.status, body
    except (OSError, http.client.HTTPException):
        raise _GuardError("RunPod control-plane request failed.") from None
    finally:
        connection.close()


def _own_pod_exists(venue: _Venue, key: str, test_origin: str | None) -> bool:
    status, raw = _provider_call(venue, "GET", key, test_origin)
    if status == 404:
        return False
    if status != 200:
        raise _GuardError(f"RunPod did not authorize an own-pod read (HTTP {status}).")
    try:
        pod = json.loads(raw, object_pairs_hook=_unique_fields)
    except ValueError:
        raise _GuardError("RunPod returned invalid pod identity.") from None
    if not isinstance(pod, dict) or (pod.get("id"), pod.get("name")) != (venue.pod_id, venue.intent):
        raise _GuardError("RunPod pod identity or acquisition intent does not match.")
    return True


def _proof_record(venue: _Venue) -> bytes:
    files = {path.name: hashlib.sha256(_read_private(path)).hexdigest() for path in venue.paths}
    return _encode(venue.proof | {"files": files})


def _require_watchdog(venue: _Venue) -> None:
    if time.time_ns() // 1_000_000 >= venue.ends_at_ms or (venue.root / "watchdog.cutoff").exists():
        raise _GuardError("The RunPod venue reached its cutoff.")
    fd = os.open(venue.root / "watchdog.lock", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        raise _GuardError("The RunPod venue watchdog is not active.")
    finally:
        os.close(fd)


def _arm(venue: _Venue, options: argparse.Namespace) -> None:
    key = _pod_key(venue)
    guard_source = globals().get("BUNDLED_GUARD")
    watchdog_source = globals().get("BUNDLED_WATCHDOG")
    if guard_source is None:
        guard_source = Path(__file__).read_bytes()
    if watchdog_source is None and options.watchdog_script is not None:
        watchdog_source = options.watchdog_script.read_bytes()
    if not isinstance(guard_source, bytes) or not isinstance(watchdog_source, bytes):
        raise _GuardError("RunPod boot requires the exact watchdog and guard sources.")

    root = venue.root
    _private_root(root)
    guard, watchdog, config = venue.paths
    with (root / "cutoff-arm.lock").open("ab") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _publish(guard, guard_source)
        _publish(watchdog, watchdog_source)
        _publish(config, venue.watchdog_policy(options.test_origin))
        subprocess.run(
            [sys.executable, "-I", str(watchdog), "--config", str(config), "--detach"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=8, check=True,
        )
        if not _own_pod_exists(venue, key, options.test_origin):
            raise _GuardError("RunPod did not find the booting pod.")
        _require_watchdog(venue)
        _publish(root / "cutoff-proof.json", _proof_record(venue))


def _check(venue: _Venue) -> None:
    _check_private_root(venue.root)
    if _read_private(venue.root / "cutoff-proof.json") != _proof_record(venue):
        raise _GuardError("RunPod watchdog proof differs from this venue or its boot configuration.")
    _require_watchdog(venue)
    print(_encode(venue.proof).decode())


def _terminate(venue: _Venue, test_origin: str | None) -> None:
    key = _pod_key(venue)
    _check_private_root(venue.root)
    if _read_private(venue.paths[-1]) != venue.watchdog_policy(test_origin):
        raise _GuardError("RunPod termination differs from the armed venue configuration.")
    if time.time_ns() // 1_000_000 < venue.ends_at_ms and not (venue.root / "watchdog.cutoff").exists():
        raise _GuardError("RunPod venue cutoff has not been reached.")
    if not _own_pod_exists(venue, key, test_origin):
        return
    status, _ = _provider_call(venue, "DELETE", key, test_origin)
    if status not in {204, 404}:
        raise _GuardError(f"RunPod did not confirm termination (HTTP {status}).")
    if _own_pod_exists(venue, key, test_origin):
        raise _GuardError("RunPod still reports the pod after the termination request.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    for action in ("arm", "check", "terminate"):
        command = actions.add_parser(action)
        command.add_argument("--workspace", required=True, type=Path)
        command.add_argument("--intent", required=True)
        command.add_argument("--ends-at-ms", required=True, type=int)
        if action != "arm":
            command.add_argument("--pod-id", required=True)
        if action != "check":
            command.add_argument("--test-origin")
        if action == "arm":
            command.add_argument("--watchdog-script", type=Path)
    options = parser.parse_args()
    try:
        pod_id = os.environ.get("RUNPOD_POD_ID", "") if options.action == "arm" else options.pod_id
        venue = _Venue(options.workspace, options.intent, options.ends_at_ms, pod_id)
        match options.action:
            case "arm":
                _arm(venue, options)
            case "check":
                _check(venue)
            case "terminate":
                _terminate(venue, options.test_origin)
    except _GuardError as error:
        parser.exit(1, f"{error}\n")
    except (OSError, ValueError, subprocess.SubprocessError):
        parser.exit(1, "RunPod venue guard could not verify or apply its boot configuration.\n")


if __name__ == "__main__":
    main()
