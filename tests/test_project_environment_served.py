"""Studio opened on a project inside a uv project computes with that project's package, edits included."""

from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pytest

from feedbax.desktop.release import AppPaths
from feedbax.host import HostClient
from feedbax.host.io import OWNER_ENV
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.effective_success import SuccessView
from feedbax.project_folder import ProjectFolderLocation
from tests.support.environment_project import (
    OPERATION, PACKAGE_SOURCE, analysis_step, environment_project, trained_project,
)
from tests.support.hang import HANG_SECONDS
from tests.support.host import wait_until

REPO = Path(__file__).resolve().parents[1]
_SETTLED = {"succeeded", "failed", "cannot_run"}


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class _Studio:
    def __init__(self, port: int, project_id: str):
        self.base = f"http://127.0.0.1:{port}/api"
        self.project = f"{self.base}/projects/{project_id}"

    def get(self, url: str, **query):
        with urlopen(url + (f"?{urlencode(query)}" if query else ""), timeout=HANG_SECONDS) as response:
            return json.load(response)

    def answering(self) -> bool:
        try:
            return self.get(f"{self.base}/health")["status"] == "ready"
        except (URLError, OSError):
            return False

    def window(self) -> tuple[str, dict]:
        revision = self.get(self.project)["revision"]
        return revision, self.get(f"{self.project}/train-runs", revision=revision, window_map="true")["window_map"]

    def status(self, steps: tuple[str, ...]) -> dict[str, set[str]]:
        status: dict[str, set[str]] = {step: set() for step in steps}
        for row in self.window()[1]["rows"]:
            if row["key"]["step"] in status:
                status[row["key"]["step"]].add(row["status"])
        return status

    def run(self, *steps: str) -> dict[str, set[str]]:
        revision, window = self.window()
        body = json.dumps({
            "revision": revision, "token": f"served-{os.getpid()}-{'-'.join(steps)}",
            "run_keys": [row["key"] for row in window["rows"] if row["key"]["step"] in steps],
        }).encode()
        with urlopen(Request(f"{self.project}/train-runs", data=body, headers={"content-type": "application/json"}),
                     timeout=HANG_SECONDS) as response:
            assert response.status == 202
        return wait_until(lambda: self.settled(steps))

    def settled(self, steps: tuple[str, ...]) -> dict[str, set[str]] | None:
        status = self.status(steps)
        return status if all(found and found <= _SETTLED for found in status.values()) else None

    def operations(self) -> set[str]:
        catalog = self.get(f"{self.base}/projects/authoring/catalog")
        return {entry.get("key") for entry in catalog["registries"]["operations"]["entries"]}


@pytest.mark.round(
    "a fresh uv environment is synced, a Studio backend starts in it, and a model trains, evaluates and is analyzed "
    "by real workers; measured floor about 60 s on an M-series Mac with a warm uv and JAX cache",
)
def test_studio_opened_on_a_uv_project_runs_its_package(tmp_path):
    """Caught: the package resolving only when installed into Studio's own environment (issue 3ccdba2), a Host that
    needs the package to publish, opening by copying, stale plans after an edit, and a typo taking Studio down."""
    repository = tmp_path / "repository"
    folder = environment_project(repository, trained_project())
    home = tmp_path / "home"
    build = tmp_path / "dist"
    build.mkdir()
    (build / "index.html").write_text("<!doctype html>")
    backend = _free_port()
    log = (tmp_path / "launcher.log").open("wb")
    launcher = subprocess.Popen(
        [sys.executable, "-B", "-m", "scripts.studio_dev", "--home", str(home), "--port", str(_free_port()),
         "--backend-port", str(backend), "--preview-from", str(build), "--project", str(folder)],
        cwd=REPO, env=os.environ | {OWNER_ENV: str(os.getpid())}, stdout=log, stderr=log,
    )
    studio = _Studio(backend, ProjectFolderLocation.selected(folder).project_id)
    host = HostClient(AppPaths.resolve(instance_root=home).host)
    try:
        wait_until(lambda: studio.answering() or launcher.poll() is not None)
        assert launcher.poll() is None, (tmp_path / "launcher.log").read_text()
        assert studio.window()[1]["problems"] == []

        assert studio.run("train", "evaluate", "analysis") == {step: {"succeeded"} for step in ("train", "evaluate", "analysis")}
        host_pid = host.ping()["pid"]
        interpreter = (repository / ".venv" / "bin" / "python").absolute()
        jobs = host.jobs(project=ProjectFolderLocation.selected(folder).project_id)
        assert {Path(job["data"]["runtime"]) for job in jobs} == {interpreter}
        store = ImmutableArtifactBlobProvider(folder / "results")
        lock = sha256((repository / "uv.lock").read_bytes()).hexdigest()
        view = SuccessView(store)
        receipts = [view.read_attempt(job["id"]).receipt for job in jobs if job["state"] == "succeeded"]
        assert "environment-fixture" in {receipt.provider.distribution for receipt in receipts if receipt.provider}
        locks = {receipt.provenance.lock_sha256 for receipt in receipts if receipt.provenance}
        assert locks == {lock}, "results were recorded against another environment's lock"
        assert not list(home.rglob("*.feedbax")), "opening the project copied it under the home"

        source = repository / PACKAGE_SOURCE / "__init__.py"
        original = source.read_text()
        source.write_text(original + "\n\n" + original[original.index("@operation"):original.index("class Readout")]
                          .replace("def speed(", "def speed_again("))
        wait_until(lambda: f"{OPERATION}_again" in studio.operations())
        (folder / "steps" / "again.json").write_text(json.dumps(analysis_step(f"{OPERATION}_again", "evaluate")))
        assert studio.run("again") == {"again": {"succeeded"}}
        assert host.ping()["pid"] == host_pid

        edited = source.read_text()
        source.write_text(edited + "\ndef broken(:\n")
        problems = wait_until(lambda: studio.window()[1]["problems"])
        packaged = {problem.get("package") for problem in problems if problem["code"] != "blocked_input"}
        assert packaged == {"environment_fixture"}, problems
        assert len({(problem["code"], problem["detail"]) for problem in problems if problem["step"] is None}) == 1
        source.write_text(edited)
        wait_until(lambda: studio.answering() and not studio.window()[1]["problems"])
    finally:
        launcher.send_signal(signal.SIGTERM)
        launcher.wait(timeout=HANG_SECONDS)
        log.close()
    shutil.rmtree(repository / ".venv")
