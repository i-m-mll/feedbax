"""Studio job reads answer from held state while the host is busy."""

from __future__ import annotations

import os
import signal
import threading
import time
from dataclasses import replace
from threading import Event

from fastapi.testclient import TestClient

from feedbax.host import HostError
from feedbax.host.protocol import AdmissionItem
from feedbax.web.app import create_app
from tests.test_local_job_host import host as host_fixture, run_job, wait_until

host = host_fixture


def _studio_job(name: str, project: str, source: str):
    data = {"run_key": name, "stage_id": "train", "scenario_id": "default", "label": name, "coordinates": {}}
    return replace(run_job(name, source, project=project, request=f"train:{name}"), data=data)


def test_job_reads_do_not_wait_on_a_busy_host(host, tmp_path, monkeypatch):
    monkeypatch.setenv("FEEDBAX_HOST_ROOT", str(host.root))
    monkeypatch.setenv("FEEDBAX_CONTROLLER_EVENT_LOG", str(tmp_path / "intents.jsonl"))
    project = "busy"
    with TestClient(create_app(projects_dir=tmp_path / "projects")) as studio:
        assert studio.get("/api/jobs", params={"project": project}).json()["jobs"] == []
        client = studio.app.state.job_service.host
        released = Event()
        blocked = Event()
        reads = {name: getattr(client, name) for name in ("jobs", "jobs_snapshot")}

        def busy(read):
            def call(*args, **kwargs):
                blocked.set()
                released.wait(30)
                return read(*args, **kwargs)
            return call

        for name, read in reads.items():
            monkeypatch.setattr(client, name, busy(read))
        job = _studio_job("held", project, "import time; time.sleep(30)")
        client.admit("busy-host", [AdmissionItem(job.id, job)], record={})
        assert blocked.wait(10)

        started = time.monotonic()
        response = studio.get("/api/jobs", params={"project": project})
        elapsed = time.monotonic() - started
        assert response.status_code == 200, response.text
        assert response.json()["jobs"] == []
        assert elapsed < 0.5

        released.set()

        def started():
            jobs = studio.get("/api/jobs", params={"project": project}).json()["jobs"]
            return jobs if [row["status"] for row in jobs] == ["starting"] else None

        assert [row["job_id"] for row in wait_until(started)] == [job.id]


def _admit(client, token: str, *jobs) -> None:
    client.admit(token, [AdmissionItem(job.id, job) for job in jobs], record={})


def test_an_unreadable_record_is_listed_only_for_its_own_project(host, tmp_path, monkeypatch):
    monkeypatch.setenv("FEEDBAX_HOST_ROOT", str(host.root))
    monkeypatch.setenv("FEEDBAX_CONTROLLER_EVENT_LOG", str(tmp_path / "intents.jsonl"))
    sleep = "import time; time.sleep(30)"
    with TestClient(create_app(projects_dir=tmp_path / "projects")) as studio:
        client = studio.app.state.job_service.host
        invalid = replace(_studio_job("invalid", "a", sleep), request="unknown-kind")
        _admit(client, "invalid", invalid, _studio_job("readable", "a", sleep), _studio_job("first", "b", sleep))

        def read(project):
            return studio.get("/api/jobs", params={"project": project}).json()

        def project_a():
            snapshot = read("a")
            return snapshot if snapshot["jobs"] and snapshot["unreadable"] else None

        snapshot = wait_until(project_a)
        assert [job["job_id"] for job in snapshot["jobs"]] == ["readable"]
        (record,) = snapshot["unreadable"]
        assert (record["project_id"], record["job_id"], record["label"]) == ("a", "invalid", "invalid")
        assert "unknown-kind" in record["reason"]
        assert read("b")["unreadable"] == []
        assert [job["job_id"] for job in read("b")["jobs"]] == ["first"]
        assert [entry["job_id"] for entry in studio.get("/api/jobs").json()["unreadable"]] == ["invalid"]

        cursor = read("b")["cursor"]
        _admit(client, "second", _studio_job("second", "b", sleep))
        _, page = studio.app.state.job_activity.events(after=cursor, projects=("b",), wait=10, machine=True)
        assert "second" in {event.job_id for event in page.events}
        assert {job["job_id"] for job in read("b")["jobs"]} == {"first", "second"}


def test_the_follower_recovers_from_one_unexpected_error(host, tmp_path, monkeypatch):
    monkeypatch.setenv("FEEDBAX_HOST_ROOT", str(host.root))
    monkeypatch.setenv("FEEDBAX_CONTROLLER_EVENT_LOG", str(tmp_path / "intents.jsonl"))
    with TestClient(create_app(projects_dir=tmp_path / "projects")) as studio:
        assert studio.get("/api/jobs").status_code == 200
        service = studio.app.state.job_service
        page = service.activity_page
        failures = iter([RuntimeError("one bad event")])

        def once(**kwargs):
            if (failure := next(failures, None)) is not None:
                raise failure
            return page(**kwargs)

        monkeypatch.setattr(service, "activity_page", once)
        _admit(service.host, "wake", _studio_job("after", "p", "import time; time.sleep(30)"))

        def failing():
            response = studio.get("/api/jobs")
            return response if response.status_code == 503 else None

        assert "one bad event" in wait_until(failing).json()["detail"]["message"]

        def recovered():
            response = studio.get("/api/jobs", params={"project": "p"})
            return response.status_code == 200 and [job["job_id"] for job in response.json()["jobs"]] == ["after"]

        assert wait_until(recovered)


def test_reads_recover_as_soon_as_a_quiet_host_answers_again(host, tmp_path, monkeypatch):
    monkeypatch.setenv("FEEDBAX_HOST_ROOT", str(host.root))
    monkeypatch.setenv("FEEDBAX_CONTROLLER_EVENT_LOG", str(tmp_path / "intents.jsonl"))
    with TestClient(create_app(projects_dir=tmp_path / "projects")) as studio:
        service = studio.app.state.job_service
        job = _studio_job("quiet", "p", "import time; time.sleep(30)")
        _admit(service.host, "quiet", job)

        page = service.activity_page

        def settled():
            head = page(after=0, wait=0).head
            time.sleep(0.5)
            return head == page(after=0, wait=0).head == studio.get("/api/jobs").json()["cursor"]

        assert wait_until(settled)
        down = Event()
        down.set()

        def restarting(**kwargs):
            if down.is_set():
                raise HostError("disconnected", "host socket closed before a complete message")
            return page(**kwargs)

        monkeypatch.setattr(service, "activity_page", restarting)

        def failing():
            response = studio.get("/api/jobs")
            return response if response.status_code == 503 else None

        assert wait_until(failing).json()["detail"]["code"] == "disconnected"
        down.clear()
        started = time.monotonic()
        assert wait_until(lambda: studio.get("/api/jobs").status_code == 200)
        # No event wakes the follower here, so recovery within a follow wait means its first page was a probe.
        assert time.monotonic() - started < 1


def test_shutdown_ends_the_follower_before_the_host_can_be_restarted(host, tmp_path, monkeypatch):
    monkeypatch.setenv("FEEDBAX_HOST_ROOT", str(host.root))
    monkeypatch.setenv("FEEDBAX_CONTROLLER_EVENT_LOG", str(tmp_path / "intents.jsonl"))
    with TestClient(create_app(projects_dir=tmp_path / "projects")) as studio:
        assert studio.get("/api/jobs").status_code == 200
        follower = next(thread for thread in threading.enumerate() if thread.name == "studio-job-activity")
    assert not follower.is_alive()
    os.kill(host.ping()["pid"], signal.SIGTERM)
    wait_until(lambda: not host.address.exists())
    time.sleep(0.2)
    assert not host.address.exists()
