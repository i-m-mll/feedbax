"""A detached watchdog supervises the existing venue Host and never extends cutoff."""

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

from feedbax.host import HostClient, HostError
from feedbax.host import watchdog
from feedbax.host.protocol import SessionAccess
from feedbax.host.session import initialize_session
from tests.support.hang import HANG_SECONDS
from tests.support.host import wait_until


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


def start_venue(bootstrap: Path, *, ends_at_ms: int):
    """A detached watchdog over a ready venue, and a ping of its Host that returns its pid and stream id."""
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

    return client, ping, calls, confirmation


def replacement_of(ping, pid: int):
    def read():
        result = ping()
        return result if result and result[0] != pid else None
    return read


def stop_host(client: HostClient, ping) -> None:
    if running := ping():
        os.kill(running[0], signal.SIGTERM)
        wait_until(lambda: not client.address.exists())


def test_detached_watchdog_restarts_a_killed_host_before_cutoff(tmp_path):
    bootstrap = tmp_path / "venue"
    client, ping, calls, _ = start_venue(bootstrap, ends_at_ms=int(time.time() * 1000) + 3_600_000)
    pid, stream_id = wait_until(ping)
    supervisor = psutil.Process(pid).ppid()
    assert str(bootstrap / "watchdog.py") in psutil.Process(supervisor).cmdline()
    try:
        os.kill(pid, signal.SIGKILL)
        replacement, replacement_stream = wait_until(replacement_of(ping, pid))
        assert replacement_stream == stream_id
        assert psutil.Process(replacement).ppid() == supervisor
        assert not calls.exists()
    finally:
        os.kill(supervisor, signal.SIGKILL)
        wait_until(lambda: not psutil.pid_exists(supervisor))
        stop_host(client, ping)


@pytest.mark.round(
    "The watchdog restarts a killed venue Host and, at a real cutoff some seconds later, retries"
    " termination until it is confirmed."
)
def test_detached_watchdog_recovers_host_then_retries_termination(tmp_path):
    # Fixed when the watchdog starts, the cutoff must cover the Host's first start and its restart,
    # each up to about 2.5 s on a loaded machine.
    ends_at_ms = int(time.time() * 1000) + 10_000
    client, ping, calls, confirmation = start_venue(tmp_path / "venue", ends_at_ms=ends_at_ms)
    try:
        pid, stream_id = wait_until(ping)
        os.kill(pid, signal.SIGKILL)
        _, replacement_stream = wait_until(replacement_of(ping, pid))
        assert replacement_stream == stream_id
        assert not calls.exists()
        wait_until(confirmation.exists)
        timestamps = [int(line) for line in calls.read_text().splitlines()]
        assert len(timestamps) == 2
        assert all(timestamp >= ends_at_ms for timestamp in timestamps)
    finally:
        stop_host(client, ping)


def test_watchdog_cutoff_does_not_wait_for_ready_marker_or_host(tmp_path):
    bootstrap = tmp_path / "venue"
    ends_at_ms = time.time_ns() // 1_000_000 - 1
    argv, calls, confirmation = prepare_watchdog(bootstrap, ends_at_ms=ends_at_ms)
    subprocess.run(argv, check=True, timeout=HANG_SECONDS)
    assert confirmation.exists()
    assert len(calls.read_text().splitlines()) == 2
    assert not (bootstrap / "host").exists()
