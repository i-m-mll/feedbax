"""Studio job reads answer from held state while the host is busy."""

from __future__ import annotations

import time
from dataclasses import replace
from threading import Event

from fastapi.testclient import TestClient

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
