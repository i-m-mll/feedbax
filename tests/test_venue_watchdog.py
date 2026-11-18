"""A detached watchdog supervises the existing venue Host and never extends cutoff."""

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from feedbax.host import HostClient, HostError
from feedbax.host import watchdog
from feedbax.host.protocol import SessionAccess
from feedbax.host.session import initialize_session
from tests.support.hang import HANG_SECONDS


def until(read):
    deadline = time.monotonic() + HANG_SECONDS
    while time.monotonic() < deadline:
        if value := read():
            return value
        time.sleep(0.02)
    pytest.fail("The venue watchdog did not reach the expected state.")


def prepare_watchdog(bootstrap: Path, *, ends_at_ms: int):
    bootstrap.mkdir(mode=0o700)
    script = bootstrap / "watchdog.py"
    shutil.copyfile(watchdog.__file__, script)
    calls = bootstrap / "termination-attempts"
    confirm = bootstrap / "termination-confirmed"
    command = f"""
import sys
import time
from pathlib import Path
calls = Path({str(calls)!r})
retried = calls.exists()
with calls.open("a") as log:
    print(time.time_ns() // 1_000_000, file=log)
if not retried:
    sys.exit(1)
Path({str(confirm)!r}).touch()
"""
    path = bootstrap / "watchdog.json"
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as config:
        json.dump({
            "ends_at_ms": ends_at_ms, "host_root": str(bootstrap / "host"),
            "host_python": sys.executable, "ready_file": str(bootstrap / "ready"),
            "terminate_argv": [sys.executable, "-c", command],
        }, config)
    return [sys.executable, str(script), "--config", str(path)], calls, confirm


def test_detached_watchdog_recovers_host_then_retries_termination(tmp_path):
    bootstrap = tmp_path / "venue"
    ends_at_ms = int(time.time() * 1000) + 4500
    argv, calls, confirmation = prepare_watchdog(bootstrap, ends_at_ms=ends_at_ms)
    subprocess.run([*argv, "--detach"], check=True, timeout=HANG_SECONDS, capture_output=True)
    subprocess.run([*argv, "--detach"], check=True, timeout=HANG_SECONDS, capture_output=True)
    assert not (bootstrap / "host").exists()
    assert not calls.exists()

    host_root = bootstrap / "host"
    token = "controller_" * 4
    initialize_session(
        host_root, session_id="venue", operator="tester", expires_at=ends_at_ms / 1000,
        adoption_token="adoption_" * 4, controller_token=token,
    )
    client = HostClient(host_root, session=SessionAccess("venue", 0, token))
    (bootstrap / "ready").touch(mode=0o600)

    def ping():
        try:
            result = client._call("ping", start_host=False)
            assert isinstance(result, dict)
            pid, stream_id = result["pid"], result["stream_id"]
            assert type(pid) is int and isinstance(stream_id, str)
            return pid, stream_id
        except HostError as error:
            assert error.code == "disconnected"
            return None

    pid = None
    try:
        pid, stream_id = until(ping)
        os.kill(pid, signal.SIGKILL)

        def replacement():
            result = ping()
            return result if result and result[0] != pid else None

        pid, replacement_stream = until(replacement)
        assert replacement_stream == stream_id
        assert not calls.exists()
        until(confirmation.exists)
        timestamps = [int(line) for line in calls.read_text().splitlines()]
        assert len(timestamps) == 2
        assert all(timestamp >= ends_at_ms for timestamp in timestamps)
    finally:
        running = ping()
        if running:
            os.kill(running[0], signal.SIGTERM)
            until(lambda: not client.address.exists())


def test_watchdog_cutoff_does_not_wait_for_ready_marker_or_host(tmp_path):
    bootstrap = tmp_path / "venue"
    ends_at_ms = time.time_ns() // 1_000_000 - 1
    argv, calls, confirmation = prepare_watchdog(bootstrap, ends_at_ms=ends_at_ms)
    subprocess.run(argv, check=True, timeout=HANG_SECONDS)
    assert confirmation.exists()
    assert len(calls.read_text().splitlines()) == 2
    assert not (bootstrap / "host").exists()
