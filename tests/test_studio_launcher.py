import os
from pathlib import Path
import socket
import subprocess
import sys
from urllib.error import URLError
from urllib.request import urlopen

import psutil
import pytest

from feedbax.desktop.release import AppPaths
from feedbax.host import HostClient, HostError
from feedbax.host.io import OWNER_ENV
from feedbax.host.source import host_source
from scripts import studio_dev
from tests.support.hang import HANG_SECONDS
from tests.support.host import HELD_UNTIL_STOPPED, wait_until

REPO = Path(__file__).resolve().parents[1]


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def test_studio_refuses_a_host_with_different_source_bytes(host, monkeypatch):
    launched = host.ping()
    assert launched["source"] == host_source()
    studio_dev.require_matching_host(host)
    monkeypatch.setattr(studio_dev, "host_source", lambda: {**host_source(), "sha256": "0" * 64})

    with pytest.raises(SystemExit) as error:
        studio_dev.require_matching_host(host)
    assert f"Host PID {launched['pid']}" in str(error.value)
    assert str(REPO / "feedbax") in str(error.value)
    assert host.ping() == launched


def test_a_test_launcher_and_its_host_exit_with_the_test_runner(tmp_path):
    """The Studio e2e fixture's launch, whose runner is killed once it serves: no process it started survives."""
    home = tmp_path / "home"
    build = tmp_path / "dist"
    build.mkdir()
    (build / "index.html").write_text("<!doctype html>")
    runner = subprocess.Popen([sys.executable, "-c", HELD_UNTIL_STOPPED])
    backend = _free_port()
    with (tmp_path / "launcher.log").open("wb") as log:
        launcher = subprocess.Popen(
            [sys.executable, "-B", "-m", "scripts.studio_dev", "--home", str(home),
             "--port", str(_free_port()), "--backend-port", str(backend),
             "--remove-test-home", "--preview-from", str(build)],
            cwd=REPO, env=os.environ | {OWNER_ENV: str(runner.pid)}, stdout=log, stderr=log,
        )
    try:
        host = HostClient(AppPaths.resolve(instance_root=home).host)
        host.prohibit_start()

        def answering():
            try:
                return host.ping()["pid"]
            except HostError:
                return None

        def serving():
            try:
                with urlopen(f"http://127.0.0.1:{backend}/api/health", timeout=HANG_SECONDS) as response:
                    return response.status == 200
            except (URLError, OSError):
                return False

        wait_until(serving)
        launched = [answering(), *(child.pid for child in psutil.Process(launcher.pid).children(recursive=True))]
    finally:
        runner.kill()
        runner.wait()
    launcher.wait(timeout=HANG_SECONDS)
    assert wait_until(lambda: not any(psutil.pid_exists(pid) for pid in launched))
    assert not home.exists()
