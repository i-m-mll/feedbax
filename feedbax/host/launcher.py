"""Process-group leader that records how its detached command exited."""

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from .io import EVIDENCE_FILE, TOKEN_ENV, append, read_worker_result
from .python import python_argv

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
    return python_argv(interpreter, "-c", BOOTSTRAP, str(HOST_FILES), role, *args)


def _outcome(log: Path, status: int, require_result: bool) -> dict:
    result = read_worker_result(log, truncate_incomplete=True)
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


def _terminal(log: Path, status: int, require_result: bool) -> None:
    if not log.exists():
        return
    with log.open("rb") as stream:
        accepted = json.loads(stream.readline())
    if (
        accepted.get("kind") == "accepted"
        and accepted["data"].get("token") == os.environ[TOKEN_ENV]
    ):
        outcome = _outcome(log, status, require_result)
        if outcome["outcome"] == "succeeded" and (evidence := log.with_name(EVIDENCE_FILE)).is_file():
            outcome["execution_evidence"] = json.loads(evidence.read_bytes())
        append(log, "terminal", outcome, sync=True)


def main(control_fd: str, profile_path: str) -> int:
    profile = json.loads(Path(profile_path).read_text())
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
    log = None
    with os.fdopen(read_end, "rb") as report:
        for line in report:
            record = json.loads(line)
            if "event_log" in record:
                log = Path(record["event_log"])
            else:
                assert log is not None
                _terminal(log, record["status"], require_result=True)
                log = None
    status = child.wait()
    if log is not None:
        _terminal(log, status, bool(profile["warm_entrypoint"]))
    return 0
