"""Exercise the owner-only Activity path against the real Host and saved Plan."""

from __future__ import annotations

from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from feedbax.contracts.project_io import ProjectConfiguration, dump_text_project
from feedbax.desktop.__main__ import _register_stop
from feedbax.desktop.activity import register_activity
from feedbax.host import HostClient, Job
from feedbax.host.protocol import AdmissionItem
from feedbax.web.app import create_app
from tests.support import minimal_worker
from tests.support.managed_train_fixture import managed_train_project
from tests.test_desktop_project_open import _desktop_clients
from tests.support.host import (
    HELD_UNTIL_STOPPED, admit, held_until_released, job_row, private_host as host_fixture, restart, reusing_host, run_job,
    wait_until,
)
from tests.support.hang import HANG_SECONDS

host = host_fixture
ACTIVITY = "/_desktop/activity"


def _project(root: Path, *, batches: int) -> None:
    project = managed_train_project(batches=batches)
    for name, content in dump_text_project(project, ProjectConfiguration(project=root.name, store="results")).items():
        file = root / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(content)


def test_background_activity_lifecycle(host, tmp_path, monkeypatch):
    monkeypatch.setenv("FEEDBAX_HOST_ROOT", str(host.root))
    monkeypatch.setenv("FEEDBAX_ORCHESTRATION_ROOT", str(tmp_path / "orchestration"))
    monkeypatch.setenv("FEEDBAX_CONTROLLER_EVENT_LOG", str(tmp_path / "controller.jsonl"))
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    projects = tmp_path / "projects"
    _project(projects / "reach", batches=160)
    _project(projects / "long", batches=3000)
    release = Path(sys.executable).resolve().parent.parent
    app = create_app(projects_dir=projects, desktop_release=release)
    transport, control, browser = _desktop_clients(app)
    register_activity(app, "fixture")
    control.headers["X-Feedbax-Release"] = "fixture"
    server = SimpleNamespace(should_exit=False)
    _register_stop(app, server)
    origin = {"Origin": transport.endpoint.origin}
    initial_host = host.ping()["pid"]
    foreign_hold = tmp_path / "foreign-release"
    foreign = run_job(
        "foreign",
        held_until_released(foreign_hold),
        slot="light",
    )
    admit(host, "foreign", foreign)
    wait_until(lambda: job_row(host, foreign.id, "running"))

    def launch(client: TestClient, project: str) -> str:
        url = f"/api/projects/{project}"
        opened = client.get(f"{url}/open")
        assert opened.status_code == 200, opened.text
        result = client.post(f"{url}/train-runs", headers=origin, json={
            "token": project, "revision": opened.json()["snapshot"]["revision"],
            "run_keys": [{"variant": "default", "step": "train", "row": "default"}],
        })
        assert result.status_code == 202, result.text
        (item,) = result.json()["items"]
        assert item["disposition"] == "admitted"
        return item["job_id"]

    def exchange(client: TestClient, payload: dict, *, activity: bool = False) -> None:
        ticket = control.post("/_desktop/ticket", json=payload)
        assert ticket.status_code == 200, ticket.text
        granted = client.post("/_desktop/exchange", headers={
            **origin, "X-Feedbax-Ticket": ticket.json()["ticket"],
        })
        assert granted.status_code == 204, granted.text
        if activity:
            assert "x-feedbax-project" not in granted.headers
            assert granted.headers["x-feedbax-activity"] == "1"

    with TestClient(app):
        wrong_release = control.get(ACTIVITY, headers={"X-Feedbax-Release": "other"})
        assert wrong_release.status_code == 409
        assert control.get(ACTIVITY).json()["count"] == 0
        initial_cursor = control.get(ACTIVITY).json()["cursor"]
        assert control.post("/_desktop/ticket", json={"activity": True, "project_id": "reach"}).status_code == 422
        exchange(browser, {"project_id": "reach"})
        assert browser.get(ACTIVITY, headers=origin).status_code == 403
        assert browser.post(f"{ACTIVITY}/stop", headers=origin, json={"mode": "force"}).status_code == 403

        hold = tmp_path / "release"
        gate = run_job("unrelated", held_until_released(hold))
        host.policy(compute=1)
        admit(host, "gate", gate)
        wait_until(lambda: job_row(host, gate.id, "running"))
        completed = launch(browser, "reach")
        assert control.get(ACTIVITY).json()["count"] == 1
        browser.close()
        hold.touch()
        wait_until(lambda: job_row(host, completed, "running"))
        assert control.get(ACTIVITY).json()["count"] == 1

        requested = control.post(f"{ACTIVITY}/open").json()
        assert requested["pid"] is None
        gui = {"pid": 10, "project": "reach", "ready": True}
        intent = control.post(f"{ACTIVITY}/window", json=gui).json()["open"]
        assert isinstance(intent, int)
        assert control.post(f"{ACTIVITY}/window", json=gui).json()["open"] == intent
        assert control.post(f"{ACTIVITY}/ack/{intent}").status_code == 200
        assert control.post(f"{ACTIVITY}/window", json=gui).json()["open"] is None
        assert control.post(f"{ACTIVITY}/window", json=gui).json()["open"] is None
        assert control.post(f"{ACTIVITY}/open").json()["pid"] == 10

        reopened = TestClient(browser.app, base_url=transport.endpoint.origin)
        exchange(reopened, {"activity": True}, activity=True)
        assert {path.name for path in projects.iterdir()} == {"reach", "long"}
        jobs = reopened.get("/api/jobs").json()["jobs"]
        assert any(job["job_id"] == completed for job in jobs)
        finished = wait_until(lambda: job_row(host, completed, "succeeded"))
        assert finished["progress"]["completed_updates"] == 160
        terminal = control.get(ACTIVITY, params={"after": initial_cursor}).json()
        while terminal["terminal"] is None:
            assert terminal["cursor"] > initial_cursor
            initial_cursor = terminal["cursor"]
            terminal = control.get(ACTIVITY, params={"after": initial_cursor}).json()
        assert terminal["count"] == 0
        assert terminal["terminal"] == {
            "id": completed, "project": finished["project"], "name": "Train", "status": "succeeded",
        }
        assert control.get(ACTIVITY, params={"after": terminal["cursor"]}).json()["terminal"] is None
        assert control.get(ACTIVITY).json()["terminal"] is None

        stopping = launch(reopened, "long")
        wait_until(lambda: job_row(host, stopping, "running"))
        reply = control.post(f"{ACTIVITY}/stop", json={"mode": "checkpoint"})
        assert reply.status_code == 200 and reply.json()[0]["error"] is None, reply.text
        stopped = wait_until(lambda: job_row(host, stopping, "stopped"))
        assert stopped["checkpoint"]
        assert control.get(ACTIVITY).json()["count"] == 0

        def admit_probe(name, source):
            job = replace(
                run_job(name, source, project=stopped["project"], request=f"train:{name}"),
                data=stopped["data"],
            )
            host.admit(name, [AdmissionItem(job.id, job)], record={"target": job.data["target"]})
            return job

        probe = admit_probe("stubborn", HELD_UNTIL_STOPPED)
        wait_until(lambda: job_row(host, probe.id, "running"))
        assert control.post(f"{ACTIVITY}/stop", json={"mode": "force"}).status_code == 200
        wait_until(lambda: job_row(host, probe.id)["outcome"])
        assert job_row(host, probe.id)["stop_mode"] == "force"

        failed_probe = admit_probe("failed", "raise RuntimeError('failed')")
        wait_until(lambda: job_row(host, failed_probe.id, "failed"))
        observed = control.get(ACTIVITY, params={"after": terminal["cursor"]}).json()
        while observed["terminal"] is None:
            assert observed["cursor"] > terminal["cursor"]
            terminal = observed
            observed = control.get(ACTIVITY, params={"after": terminal["cursor"]}).json()
        assert observed["terminal"]["id"] == failed_probe.id
        assert observed["terminal"]["status"] == "failed"

        stubborn = admit_probe("quit-escalation", HELD_UNTIL_STOPPED)
        wait_until(lambda: job_row(host, stubborn.id, "running"))
        with ThreadPoolExecutor(max_workers=1) as executor:
            quitting = executor.submit(control.post, "/_desktop/stop", json={"disposition": "checkpoint"})
            wait_until(lambda: job_row(host, stubborn.id, "stopping"))
            assert not quitting.done()
            assert control.post(f"{ACTIVITY}/stop", json={"mode": "force"}).status_code == 200
            quit_service = quitting.result(timeout=HANG_SECONDS)
        assert quit_service.status_code == 200 and quit_service.json()["status"] == "ready_to_stop", quit_service.text
        assert server.should_exit and host.ping()["pid"] == initial_host
        assert job_row(host, foreign.id)["state"] == "running"
        foreign_hold.touch()
        assert control.post("/_desktop/stop", json={"disposition": "checkpoint"}).status_code == 409
        reopened.close()


def test_background_observation_does_not_start_an_idle_host(tmp_path, monkeypatch):
    root = tmp_path / "missing-host"
    monkeypatch.setenv("FEEDBAX_HOST_ROOT", str(root))
    app = create_app(projects_dir=tmp_path / "projects")
    transport, control, _ = _desktop_clients(app)
    register_activity(app, "fixture")
    control.headers["X-Feedbax-Release"] = "fixture"
    with TestClient(app):
        assert control.get(ACTIVITY).json() == {
            "after": None, "cursor": None, "count": 0, "terminal": None, "window": None,
        }
        assert control.post(f"{ACTIVITY}/stop", json={"mode": "checkpoint"}).json() == []
        assert not HostClient(root).address.exists()


@pytest.mark.round(
    "After a Host restart a real Train worker is imported ahead of use and the next Train runs "
    "in it."
)
@pytest.mark.parametrize("desktop_owned", [False, True], ids=["standalone", "desktop-owned"])
def test_warming_rewarms_idle_host_after_restart(host, tmp_path, monkeypatch, desktop_owned):
    monkeypatch.setenv("FEEDBAX_HOST_ROOT", str(host.root))
    monkeypatch.setenv("FEEDBAX_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("FEEDBAX_CONTROLLER_EVENT_LOG", str(tmp_path / "intents.jsonl"))
    if desktop_owned:
        monkeypatch.setenv("FEEDBAX_DESKTOP_RELEASE_ROOT", str(Path(sys.executable).resolve().parent.parent))
    else:
        monkeypatch.delenv("FEEDBAX_DESKTOP_RELEASE_ROOT", raising=False)
    projects = tmp_path / "projects"
    _project(projects / "reach", batches=2)
    app = create_app(projects_dir=projects)

    def profiles():
        return set((host.root / "workers").glob("*/profile.json"))

    with TestClient(app) as studio:
        assert studio.post("/api/jobs/warm").status_code == 202
        original = wait_until(profiles)

        restart(host)

        assert studio.post("/api/jobs/warm").status_code == 202
        replacement = wait_until(lambda: profiles() - original)
        assert len(replacement) == 1

        assert studio.post("/api/jobs/warm").status_code == 202
        assert profiles() - original == replacement

        opened = studio.get("/api/projects/reach/open")
        assert opened.status_code == 200, opened.text
        submitted = studio.post("/api/projects/reach/train-runs", json={
            "revision": opened.json()["snapshot"]["revision"], "token": "restart-train",
            "run_keys": [{"variant": "default", "step": "train", "row": "default"}],
        })
        assert submitted.status_code == 202, submitted.text
        item, = submitted.json()["items"]
        assert item["disposition"] == "admitted"
        job = item["job_id"]
        done = wait_until(lambda: job_row(host, job, "succeeded"))
        log = host.root / "deliveries" / job / str(done["delivery"]) / "events.jsonl"
        with log.open() as stream:
            acceptance = json.loads(stream.readline())
        profile, = replacement
        assert acceptance == {"kind": "accepted", "data": {"token": profile.parent.name}}


@pytest.mark.parametrize("desktop_owned", [False, True], ids=["standalone", "desktop-owned"])
def test_warming_after_restart_keeps_one_idle_worker_for_the_next_train(
    reusing_host, tmp_path, monkeypatch, desktop_owned,
):
    host = reusing_host
    monkeypatch.setenv("FEEDBAX_HOST_ROOT", str(host.root))
    monkeypatch.setenv("FEEDBAX_CONTROLLER_EVENT_LOG", str(tmp_path / "intents.jsonl"))
    if desktop_owned:
        monkeypatch.setenv("FEEDBAX_DESKTOP_RELEASE_ROOT", str(Path(sys.executable).resolve().parent.parent))
    else:
        monkeypatch.delenv("FEEDBAX_DESKTOP_RELEASE_ROOT", raising=False)
    minimal_worker.train_with(monkeypatch, tmp_path)
    projects = tmp_path / "projects"
    _project(projects / "reach", batches=2)
    app = create_app(projects_dir=projects)

    def workers():
        return set((host.root / "workers").glob("*/profile.json"))

    with TestClient(app) as studio:
        assert studio.post("/api/jobs/warm").status_code == 202
        original = wait_until(workers)
        restart(host)
        assert studio.post("/api/jobs/warm").status_code == 202
        (replacement,) = wait_until(lambda: workers() - original)
        assert studio.post("/api/jobs/warm").status_code == 202

        opened = studio.get("/api/projects/reach/open")
        assert opened.status_code == 200, opened.text
        submitted = studio.post("/api/projects/reach/train-runs", json={
            "revision": opened.json()["snapshot"]["revision"], "token": "restart-train",
            "run_keys": [{"variant": "default", "step": "train", "row": "default"}],
        })
        assert submitted.status_code == 202, submitted.text
        (item,) = submitted.json()["items"]
        job = item["job_id"]
        minimal_worker.release(tmp_path, job)
        done = wait_until(lambda: job_row(host, job, "succeeded"))
        log = host.root / "deliveries" / job / str(done["delivery"]) / "events.jsonl"
        with log.open() as stream:
            acceptance = json.loads(stream.readline())
        assert acceptance == {"kind": "accepted", "data": {"token": replacement.parent.name}}
        assert workers() - original == {replacement}, "The pool keeps a worker beside the one that served the Train."


def test_interrupted_owned_host_leaves_the_running_train_resumable_on_the_next_host(
    tmp_path, monkeypatch, private_hosts,
):
    from feedbax.host.io import append
    from feedbax.jobs import JobService
    from feedbax.web.services.desktop_jobs import StudioJobLifecycle
    from feedbax.application.bundle import new_application_registry_bundle

    registries = new_application_registry_bundle()
    runtime = os.path.realpath(sys.executable)
    release = Path(runtime).parent.parent
    monkeypatch.setenv("FEEDBAX_DESKTOP_RELEASE_ROOT", str(release))
    host = HostClient(private_hosts / "host")
    owner = host.ping()["pid"]
    attempt = replace(
        Job("attempt", "project", "train:attempt", minimal_worker.launch(tmp_path)),
        data={"run_key": "run", "stage_id": "train", "scenario_id": "base", "label": "Train",
              "coordinates": {}, "runtime": runtime},
    )
    try:
        admit(host, "attempt", attempt)
        def checkpointed():
            row = job_row(host, attempt.id, "running")
            return row if row and row["checkpoint"] else None

        before = wait_until(checkpointed)
        cursor = host.events()["head"]
        retiring = JobService(HostClient(host.root), registries)
        result = asyncio.run(StudioJobLifecycle(retiring, release).stop("interrupt", timeout_seconds=8))
        assert result["status"] == "ready_to_stop", result
        assert not host.address.exists()
        log = host.root / "deliveries" / attempt.id / str(before["delivery"]) / "events.jsonl"
        append(log, "terminal", {"outcome": "failed"}, sync=True)

        assert host.ping()["pid"] != owner
        interrupted = job_row(host, attempt.id, "interrupted")
        assert interrupted is not None and interrupted["checkpoint"] == before["checkpoint"]
        assert interrupted["outcome"]["error"]["code"] == "service_stopped"
        events = host.events(after=cursor)["events"]
        assert [event["job_id"] for event in events if event["kind"] == "terminal"] == [attempt.id]
        resumed = JobService(host, registries).resume(attempt.id, "resume")
        assert resumed.continues == attempt.id
        (successor,) = host.jobs(job_id=resumed.job_id, include_launch=True)
        assert successor["launch"]["payload"]["resume"] == before["checkpoint"]
        minimal_worker.release(tmp_path, resumed.job_id)
        wait_until(lambda: job_row(host, resumed.job_id, "succeeded"))
    finally:
        wait_until(lambda: host.retire_runtime(runtime, str(release), "force")["status"] == "absent")
        wait_until(lambda: not subprocess.run(["pgrep", "-f", str(host.root)], capture_output=True).stdout)


@pytest.mark.round(
    "An owned Host exits under a running real Train, and the Train resumes from its checkpoint "
    "on a new Host."
)
def test_background_interrupt_retires_owned_host_and_resumes_exact_train(tmp_path, monkeypatch, private_hosts):
    from feedbax.contracts.studio_training import PublishedTrainingCheckpoint
    from feedbax.host import HostError
    from feedbax.host.process import process_group_member_pids
    from feedbax.host.io import append
    from feedbax.jobs import JobService
    from feedbax.application.bundle import new_application_registry_bundle
    from feedbax.host.method_checkpoint import read_method_checkpoint
    from feedbax.web.services.project_folder import ProjectFolderService

    root = private_hosts / "host"
    runtime = os.path.realpath(sys.executable)
    release = Path(runtime).parent.parent
    monkeypatch.setenv("FEEDBAX_DESKTOP_RELEASE_ROOT", str(release))
    monkeypatch.setenv("FEEDBAX_HOST_ROOT", str(root))
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("FEEDBAX_ORCHESTRATION_ROOT", str(tmp_path / "orchestration"))
    monkeypatch.setenv("FEEDBAX_CONTROLLER_EVENT_LOG", str(tmp_path / "controller.jsonl"))
    projects = tmp_path / "projects"
    _project(projects / "reach", batches=1000)
    folders = ProjectFolderService(projects)
    app = create_app(projects_dir=projects, desktop_release=release)
    host = HostClient(root)
    owner = host.ping()["pid"]
    plan = folders.read_plan("reach", revision=None)
    try:
        with TestClient(app):
            jobs = app.state.job_service
            (run,) = plan.plan.runs
            (item,) = jobs.admit_plan(folders, plan, run_keys=(run.key,), token="first").items
            attempt = item.job_id

            def checkpointed():
                row = job_row(host, attempt, "running")
                return row if row and row["checkpoint"] else None

            before = wait_until(checkpointed)
            source = PublishedTrainingCheckpoint.model_validate(before["checkpoint"])
            model = read_method_checkpoint(source.checkpoint_root, source.pointer, components=new_application_registry_bundle().components).model
            assert model is not None
            model.provider.verify_object(model.replicas[0].arrays)
            before_cursor = host.events()["head"]
            result = asyncio.run(app.state.desktop_jobs.stop("interrupt", timeout_seconds=8))
            assert result["status"] == "ready_to_stop", result
            assert not host.address.exists()
            assert process_group_member_pids(before["worker_pid"]) == ((), None)
            with pytest.raises(HostError, match="No such file"):
                jobs.host.ping()
            assert not host.address.exists()
        log = root / "deliveries" / attempt / str(before["delivery"]) / "events.jsonl"
        append(log, "terminal", {"outcome": "failed"}, sync=True)
        assert host.retire_runtime(runtime, str(release), "interrupt") == {"status": "absent"}
        assert host.ping()["pid"] != owner
        interrupted = job_row(host, attempt, "interrupted")
        assert interrupted is not None and interrupted["checkpoint"] == before["checkpoint"]
        assert interrupted["outcome"]["error"]["code"] == "service_stopped"
        terminal = [
            event for event in host.events(after=before_cursor)["events"]
            if event["job_id"] == attempt and event["kind"] == "terminal"
        ]
        assert len(terminal) == 1
        resumed = JobService(host, new_application_registry_bundle()).resume(attempt, "resume")
        assert resumed.continues == attempt
        (successor,) = host.jobs(job_id=resumed.job_id, include_launch=True)
        assert successor["launch"]["payload"]["resume"] == before["checkpoint"]
        done = wait_until(lambda: job_row(host, resumed.job_id, "succeeded"))
        assert done["progress"]["completed_updates"] == 1000
    finally:
        wait_until(lambda: host.retire_runtime(runtime, str(release), "force")["status"] == "absent")


def test_service_signal_waits_for_runtime_release_and_exits_successfully(tmp_path):
    marker = tmp_path / "ready"
    completed = tmp_path / "released"
    with subprocess.Popen([sys.executable, "-c", """
import asyncio
import signal
import sys
from pathlib import Path
from types import SimpleNamespace
from feedbax.desktop.__main__ import _serve_until_stopped

ready, released = map(Path, sys.argv[1:])

class Server:
    started = False

    def __init__(self):
        self.exit = asyncio.Event()

    @property
    def should_exit(self):
        return self.exit.is_set()

    @should_exit.setter
    def should_exit(self, value):
        if value:
            self.exit.set()

    async def serve(self, *, sockets):
        self.started = True
        ready.touch()
        await self.exit.wait()
        assert released.exists()

class Runtime:
    async def stop(self, disposition, *, timeout_seconds):
        assert disposition == "interrupt"
        released.touch()
        return {"status": "ready_to_stop"}

async def run():
    stop = asyncio.Event()
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, stop.set)
    app = SimpleNamespace(state=SimpleNamespace(desktop_jobs=Runtime()))
    await _serve_until_stopped(Server(), app, [], stop)

asyncio.run(run())
""", str(marker), str(completed)]) as process:
        try:
            wait_until(marker.exists)
            process.terminate()
            assert process.wait(timeout=HANG_SECONDS) == 0 and completed.exists()
        finally:
            if process.poll() is None:
                process.kill()
