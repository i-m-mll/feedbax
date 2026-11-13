"""Exercise the owner-only Activity path against the real Host and saved Plan."""

from __future__ import annotations

from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import asyncio
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from feedbax.contracts.project_io import ProjectConfiguration, dump_text_project
from feedbax.desktop.__main__ import _register_stop
from feedbax.desktop.activity import register_activity
from feedbax.host import HostClient
from feedbax.web.app import create_app
from tests.support.managed_train_fixture import managed_train_project
from tests.test_desktop_project_open import _desktop_clients
from tests.test_local_job_host import admit, host as host_fixture, job_row, run_job, wait_until

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
    app = create_app(projects_dir=projects, desktop_job_root=tmp_path / "desktop-jobs", desktop_release=release)
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
        f"from pathlib import Path; from time import sleep\n"
        f"while not Path({str(foreign_hold)!r}).exists(): sleep(.02)",
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
        gate = run_job("unrelated", f"from pathlib import Path; from time import sleep\n"
                       f"while not Path({str(hold)!r}).exists(): sleep(.02)")
        host.policy(compute=1)
        admit(host, "gate", gate)
        wait_until(lambda: job_row(host, gate.id, "running"))
        completed = launch(browser, "reach")
        assert control.get(ACTIVITY).json()["count"] == 1
        browser.close()
        hold.touch()
        wait_until(lambda: job_row(host, completed, "running"), timeout=20)
        assert control.get(ACTIVITY).json()["count"] == 1

        requested = control.post(f"{ACTIVITY}/open").json()
        assert requested["pid"] is None
        assert control.get(f"{ACTIVITY}/window").json() == {"pid": None}
        gui = {"pid": 10, "project": "reach", "ready": True}
        intent = control.post(f"{ACTIVITY}/window", json=gui).json()["open"]
        assert isinstance(intent, int)
        assert control.post(f"{ACTIVITY}/window", json=gui).json()["open"] == intent
        assert control.post(f"{ACTIVITY}/ack/{intent}").status_code == 200
        assert control.post(f"{ACTIVITY}/window", json=gui).json()["open"] is None
        assert control.get(f"{ACTIVITY}/window").json() == {"pid": 10}
        assert control.post(f"{ACTIVITY}/window", json=gui).json()["open"] is None
        assert control.post(f"{ACTIVITY}/open").json()["pid"] == 10

        reopened = TestClient(browser.app, base_url=transport.endpoint.origin)
        exchange(reopened, {"activity": True}, activity=True)
        assert {path.name for path in projects.iterdir()} == {"reach", "long"}
        jobs = reopened.get("/api/jobs").json()["jobs"]
        assert any(job["job_id"] == completed for job in jobs)
        finished = wait_until(lambda: job_row(host, completed, "succeeded"), timeout=25)
        assert finished["progress"]["completed_updates"] == 160
        terminal = control.get(ACTIVITY, params={"after": initial_cursor}).json()
        while terminal["terminal"] is None:
            assert terminal["cursor"] > initial_cursor
            initial_cursor = terminal["cursor"]
            terminal = control.get(ACTIVITY, params={"after": initial_cursor}).json()
        assert terminal["count"] == 0
        assert terminal["terminal"] == {"id": completed, "project": "reach", "name": "train", "status": "succeeded"}
        assert control.get(ACTIVITY, params={"after": terminal["cursor"]}).json()["terminal"] is None
        assert control.get(ACTIVITY).json()["terminal"] is None

        stopping = launch(reopened, "long")
        wait_until(lambda: job_row(host, stopping, "running"), timeout=20)
        reply = control.post(f"{ACTIVITY}/stop", json={"mode": "checkpoint"})
        assert reply.status_code == 200 and reply.json()[0]["error"] is None, reply.text
        stopped = wait_until(lambda: job_row(host, stopping, "stopped"), timeout=20)
        assert stopped["checkpoint"]
        assert control.get(ACTIVITY).json()["count"] == 0

        probe = replace(run_job("stubborn", "import time; time.sleep(40)"), data=stopped["data"])
        admit(host, "force-probe", probe)
        wait_until(lambda: job_row(host, probe.id, "running"))
        assert control.post(f"{ACTIVITY}/stop", json={"mode": "force"}).status_code == 200
        wait_until(lambda: job_row(host, probe.id)["outcome"])
        assert job_row(host, probe.id)["stop_mode"] == "force"

        failed_probe = replace(run_job("failed", "raise RuntimeError('failed')"), data=stopped["data"])
        admit(host, "terminal-probe", failed_probe)
        wait_until(lambda: job_row(host, failed_probe.id, "failed"))
        observed = control.get(ACTIVITY, params={"after": terminal["cursor"]}).json()
        while observed["terminal"] is None:
            assert observed["cursor"] > terminal["cursor"]
            terminal = observed
            observed = control.get(ACTIVITY, params={"after": terminal["cursor"]}).json()
        assert observed["terminal"]["id"] == failed_probe.id
        assert observed["terminal"]["status"] == "failed"

        stubborn = replace(run_job("quit-escalation", "import time; time.sleep(40)"), data=stopped["data"])
        admit(host, "quit-escalation", stubborn)
        wait_until(lambda: job_row(host, stubborn.id, "running"))
        with ThreadPoolExecutor(max_workers=1) as executor:
            quitting = executor.submit(control.post, "/_desktop/stop", json={"disposition": "checkpoint"})
            wait_until(lambda: job_row(host, stubborn.id, "stopping"))
            assert not quitting.done()
            assert control.post(f"{ACTIVITY}/stop", json={"mode": "force"}).status_code == 200
            quit_service = quitting.result(timeout=12)
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
    _, control, _ = _desktop_clients(app)
    register_activity(app, "fixture")
    control.headers["X-Feedbax-Release"] = "fixture"
    with TestClient(app):
        assert control.get(ACTIVITY).json() == {
            "after": None, "cursor": None, "count": 0, "terminal": None, "window": None,
        }
        assert control.post(f"{ACTIVITY}/stop", json={"mode": "checkpoint"}).json() == []
        assert not HostClient(root).address.exists()


def test_background_interrupt_retires_owned_host_and_resumes_exact_train(tmp_path, monkeypatch):
    from feedbax.contracts.studio_training import PublishedTrainingCheckpoint
    from feedbax.host import HostError
    from feedbax.host.process import process_group_member_pids
    from feedbax.host.io import append
    from feedbax.jobs import JobService
    from feedbax.plugins.application import new_application_registry_bundle
    from feedbax.training.checkpoint_custody import open_published_model_slot
    from feedbax.web.services.project_folder import ProjectFolderService

    root = tmp_path / "host"
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
    app = create_app(projects_dir=projects, desktop_job_root=tmp_path / "jobs", desktop_release=release)
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

            before = wait_until(checkpointed, timeout=20)
            source = PublishedTrainingCheckpoint.model_validate(before["checkpoint"])
            model = open_published_model_slot(source.checkpoint_root, source.pointer)
            assert model is not None
            model.provider.verify_object(model.arrays)
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
        done = wait_until(lambda: job_row(host, resumed.job_id, "succeeded"), timeout=22)
        assert done["progress"]["completed_updates"] == 1000
    finally:
        wait_until(lambda: host.retire_runtime(runtime, str(release), "force")["status"] == "absent")
