import os
from pathlib import Path
import socket
import subprocess
import sys

import psutil

from feedbax.desktop.release import AppPaths
from feedbax.host import HostClient, HostError
from feedbax.host.io import OWNER_ENV
from tests.support.hang import HANG_SECONDS
from tests.support.host import wait_until

REPO = Path(__file__).resolve().parents[1]


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def test_a_test_launcher_and_its_host_exit_with_the_test_runner(tmp_path):
    """The Studio e2e fixture's launch, whose runner is killed: nothing it started is left in its home."""
    home = tmp_path / "home"
    build = tmp_path / "dist"
    build.mkdir()
    (build / "index.html").write_text("<!doctype html>")
    runner = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    with (tmp_path / "launcher.log").open("wb") as log:
        launcher = subprocess.Popen(
            [sys.executable, "-B", "-m", "scripts.studio_dev", "--home", str(home),
             "--port", str(_free_port()), "--backend-port", str(_free_port()),
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

        pid = wait_until(answering)
    finally:
        runner.kill()
        runner.wait()
    launcher.wait(timeout=HANG_SECONDS)
    wait_until(lambda: not psutil.pid_exists(pid))
    assert not home.exists()
    assert not [process.pid for process in psutil.process_iter(["cmdline"])
                if any(str(home) in part for part in process.info["cmdline"] or ())]
