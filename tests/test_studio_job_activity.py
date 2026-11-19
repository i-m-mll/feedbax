"""Studio job reads answer from held state while the host is busy."""

from __future__ import annotations

import os
import signal
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event

from fastapi.testclient import TestClient

from feedbax.host import HostError
from feedbax.host.protocol import AdmissionItem
from feedbax.web.app import create_app
from tests.support.hang import HANG_SECONDS
from tests.support.host import HELD_UNTIL_STOPPED, private_host as host_fixture, run_job, wait_until

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
                released.wait()
                return read(*args, **kwargs)
            return call

        for name, read in reads.items():
            monkeypatch.setattr(client, name, busy(read))
        job = _studio_job("held", project, HELD_UNTIL_STOPPED)
        try:
            client.admit("busy-host", [AdmissionItem(job.id, job)], record={})
            assert blocked.wait(HANG_SECONDS)
            with ThreadPoolExecutor(max_workers=1) as pool:
                reading = pool.submit(studio.get, "/api/jobs", params={"project": project})
                try:
                    response = reading.result(timeout=HANG_SECONDS)
                finally:
                    released.set()
        finally:
            released.set()
        assert response.status_code == 200, response.text
        assert response.json()["jobs"] == []

        def started():
            jobs = studio.get("/api/jobs", params={"project": project}).json()["jobs"]
            return jobs if [row["status"] for row in jobs] == ["starting"] else None

        assert [row["job_id"] for row in wait_until(started)] == [job.id]


def _admit(client, token: str, *jobs) -> None:
    client.admit(token, [AdmissionItem(job.id, job) for job in jobs], record={})


def test_an_unreadable_record_is_listed_only_for_its_own_project(host, tmp_path, monkeypatch):
    monkeypatch.setenv("FEEDBAX_HOST_ROOT", str(host.root))
    monkeypatch.setenv("FEEDBAX_CONTROLLER_EVENT_LOG", str(tmp_path / "intents.jsonl"))
    sleep = HELD_UNTIL_STOPPED
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

        def second_listed():
            nonlocal cursor
            _, page = studio.app.state.job_activity.events(after=cursor, projects=("b",), wait=0.2, machine=True)
            cursor = page.cursor
            return "second" in {event.job_id for event in page.events}

        assert wait_until(second_listed)
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
        _admit(service.host, "wake", _studio_job("after", "p", HELD_UNTIL_STOPPED))

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
        job = _studio_job("quiet", "p", HELD_UNTIL_STOPPED)
        _admit(service.host, "quiet", job)

        page = service.activity_page
        wait_until(lambda: service.host.jobs(job_id=job.id)[0]["state"] == "running")
        assert wait_until(lambda: page(after=0, wait=0).head == studio.get("/api/jobs").json()["cursor"])
        down = Event()
        down.set()

        ended = Event()

        def restarting(**kwargs):
            if down.is_set():
                raise HostError("disconnected", "host socket closed before a complete message")
            if kwargs["wait"]:
                # A follow wait is held until the test ends, as one longer than the hang limit would be.
                ended.wait()
                kwargs["wait"] = 0
            return page(**kwargs)

        monkeypatch.setattr(service, "activity_page", restarting)
        try:
            def failing():
                response = studio.get("/api/jobs")
                return response if response.status_code == 503 else None

            assert wait_until(failing).json()["detail"]["code"] == "disconnected"
            down.clear()
            # No event wakes the follower here, so recovery while follow waits are held means its first page was a probe.
            assert wait_until(lambda: studio.get("/api/jobs").status_code == 200)
        finally:
            ended.set()


def test_shutdown_ends_the_follower_before_the_host_can_be_restarted(host, tmp_path, monkeypatch):
    monkeypatch.setenv("FEEDBAX_HOST_ROOT", str(host.root))
    monkeypatch.setenv("FEEDBAX_CONTROLLER_EVENT_LOG", str(tmp_path / "intents.jsonl"))
    with TestClient(create_app(projects_dir=tmp_path / "projects")) as studio:
        assert studio.get("/api/jobs").status_code == 200
        follower = next(thread for thread in threading.enumerate() if thread.name == "studio-job-activity")
    assert not follower.is_alive()
    os.kill(host.ping()["pid"], signal.SIGTERM)
    wait_until(lambda: not host.address.exists())
