"""Process-group leader that records how its detached command exited."""

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from .io import TOKEN_ENV, append

BOOTSTRAP = """\
import sys, types
path, role, *arguments = sys.argv[1:]
package = types.ModuleType('_feedbax_host_runtime')
package.__path__ = [path]
sys.modules[package.__name__] = package
module = __import__(f'{package.__name__}.{role}', fromlist=['main'])
raise SystemExit(module.main(*arguments))
"""

HOST_FILES = Path(__file__).resolve().parent


def command(interpreter: str, role: str, *args: str) -> list[str]:
    # Load only the host bootstrap from our installation. The project interpreter
    # resolves Feedbax and the executor from its own environment.
    return [interpreter, "-c", BOOTSTRAP, str(HOST_FILES), role, *args]


def _outcome(log: Path, status: int, require_result: bool) -> dict:
    result = None
    with log.open("r+b") as stream:
        # The child's final append can be torn. Remove only the incomplete final line
        # before appending our terminal event so both records remain independently framed.
        tail_start = max(0, stream.seek(0, os.SEEK_END) - 2 * 1024 * 1024)
        stream.seek(tail_start)
        tail = stream.read()
        end = tail.rfind(b"\n")
        if end >= 0 and end != len(tail) - 1:
            stream.truncate(tail_start + end + 1)
        for raw in reversed(tail[:end].split(b"\n")):
            try:
                event = json.loads(raw)
            except (ValueError, UnicodeError):
                continue
            if isinstance(event, dict) and event.get("kind") == "result":
                result = event.get("data")
                break
    if status < 0:
        signum = signal.Signals(-status)
        interrupted = signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGKILL)
        return {
            "outcome": "interrupted" if interrupted else "failed",
            "signal": int(signum),
            "error": {
                "code": "worker_interrupted" if interrupted else "worker_signal",
                "message": f"Worker exited on {signum.name}.",
            },
        }
    if isinstance(result, dict):
        outcome = result.get("outcome")
        if (outcome in ("succeeded", "stopped") and status == 0) or outcome == "failed":
            return result
    if status == 0 and not require_result and result is None:
        return {"outcome": "succeeded"}
    if status == 0:
        return {
            "outcome": "failed",
            "exit": 0,
            "error": {
                "code": "worker_result_missing",
                "message": "Worker exited without a valid result.",
            },
        }
    return {
        "outcome": "failed",
        "exit": status,
        "error": {
            "code": "worker_exit",
            "message": f"Worker exited with status {status}.",
        },
    }


def main(control_fd: str, profile_path: str) -> int:
    control = int(control_fd)
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, lambda *_: None)
    read_end, write_end = os.pipe()
    child = subprocess.Popen(
        command(sys.executable, "worker", control_fd, str(write_end), profile_path),
        pass_fds=(control, write_end),
    )
    os.close(control)
    os.close(write_end)
    with os.fdopen(read_end, "rb") as report:
        line = report.readline()
    status = child.wait()
    if not line:
        return 0
    log = Path(json.loads(line)["event_log"])
    if not log.exists():
        return 0
    with log.open("rb") as stream:
        accepted = json.loads(stream.readline())
    if (
        accepted.get("kind") == "accepted"
        and accepted["data"].get("token") == os.environ[TOKEN_ENV]
    ):
        profile = json.loads(Path(profile_path).read_text())
        append(log, "terminal", _outcome(log, status, bool(profile["warm_entrypoint"])), sync=True)
    return 0
