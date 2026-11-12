"""Project-folder observations on the Studio event stream."""

from __future__ import annotations

import asyncio
import json
from collections import deque

from fastapi import FastAPI
from fastapi.testclient import TestClient

from feedbax.contracts.jobs import StudioJobEvent, StudioJobEventHistory
from feedbax.contracts.studio_project import StudioProjectRevision
from feedbax.host import HostError
from feedbax.web.api import jobs
from feedbax.web.services.project_folder import ProjectFolderNotFound


def _job_page(head: int, cursor: int, *event_ids: int) -> tuple[int, StudioJobEventHistory]:
    return head, StudioJobEventHistory(
        cursor=cursor,
        oldest_cursor=0,
        events=[
            StudioJobEvent(
                cursor=event_id,
                job_id="job",
                project_id="training-project",
                request_id="request",
                kind="log",
                data={"line": f"update {event_id}"},
            )
            for event_id in event_ids
        ],
    )


class _JobEvents:
    def __init__(self, *pages: tuple[int, StudioJobEventHistory], on_poll=None) -> None:
        self.pages = deque(pages)
        self.on_poll = on_poll
        self.calls = []

    def events(self, *, after, projects, wait, machine):
        self.calls.append((after, projects, wait, machine))
        if self.on_poll is not None:
            self.on_poll(len(self.calls))
        if not self.pages:
            raise HostError("unavailable", "Stream ended")
        return self.pages.popleft()


class _FolderObservations:
    def __init__(self, folders: dict[str, StudioProjectRevision]) -> None:
        self.revisions = folders
        self.fingerprints = dict.fromkeys(folders, 0)

    def fingerprint(self, project_id):
        if project_id not in self.revisions:
            raise ProjectFolderNotFound(project_id)
        return self.fingerprints[project_id]

    def revision(self, project_id):
        return self.revisions[project_id]

    def write(self, project_id, **changes):
        self.fingerprints[project_id] += 1
        self.revisions[project_id] = self.revisions[project_id].model_copy(update=changes)


def _revision(project_id, revision=1, presentation_revision=2):
    return StudioProjectRevision(
        project_id=project_id,
        revision=f"{revision:064x}",
        presentation_revision=f"{presentation_revision:064x}",
    )


def _app(job_events, folders=None):
    app = FastAPI()
    app.include_router(jobs.router)
    app.state.job_service = job_events
    if folders is not None:
        app.state.project_folders = folders
    return app


def _sse_frames(text):
    return [
        dict(line.split(": ", 1) for line in frame.splitlines())
        for frame in text.strip().split("\n\n")
    ]


def test_folder_revisions_do_not_advance_the_job_cursor(monkeypatch):
    monkeypatch.setattr(jobs, "_FOLDER_OBSERVATION_INTERVAL", 0)
    first = _revision("first")
    second = _revision("second")
    folders = _FolderObservations({"first": first, "second": second})

    def edit_after_replay(call_count):
        if call_count == 3:
            folders.write("first", revision=f"{3:064x}")
        if call_count == 4:
            folders.write("first")
            folders.write("second", presentation_revision=f"{4:064x}")

    job_events = _JobEvents(
        _job_page(12, 10, 10),
        _job_page(13, 13, 12, 13),
        _job_page(14, 14, 14),
        _job_page(14, 14),
        on_poll=edit_after_replay,
    )
    client = TestClient(_app(job_events, folders))
    response = client.get(
        "/api/events?after=3&project=training-project&machine=true"
        "&folder=first&folder=missing&folder=second&folder=first",
        headers={"Last-Event-ID": "9"},
    )
    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store"
    frames = _sse_frames(response.text)
    assert [(frame.get("event"), frame["id"]) for frame in frames if "id" in frame] == [
        (None, "10"),
        (None, "12"),
        ("replay_complete", "12"),
        (None, "13"),
        (None, "14"),
        ("heartbeat", "14"),
    ]
    revisions = [frame for frame in frames if frame.get("event") == "project_revision"]
    assert all("id" not in frame for frame in revisions)
    assert frames.index(revisions[0]) > next(
        index for index, frame in enumerate(frames) if frame.get("event") == "replay_complete"
    )
    assert [json.loads(frame["data"]) for frame in revisions] == [
        first.model_dump(mode="json"),
        second.model_dump(mode="json"),
        folders.revisions["first"].model_dump(mode="json"),
        folders.revisions["second"].model_dump(mode="json"),
    ]
    assert [after for after, *_ in job_events.calls] == [9, 10, 13, 14, 14]
    assert all(
        projects == ("training-project",) and machine
        for _, projects, _, machine in job_events.calls
    )


def test_unsubscribed_job_stream_keeps_its_wait_and_rejects_path_queries():
    job_events = _JobEvents(_job_page(2, 2, 2))
    client = TestClient(_app(job_events))
    response = client.get("/api/events?after=1")
    assert response.status_code == 200
    assert [frame.get("event") for frame in _sse_frames(response.text)] == [None, "replay_complete"]
    assert job_events.calls == [(1, (), 0, False), (2, (), 5, False)]

    response = client.get("/api/events", params={"folder": "../outside"})
    assert response.status_code == 422
    assert len(job_events.calls) == 2


def test_disconnecting_during_a_folder_notice_stops_the_observation():
    folders = _FolderObservations({"first": _revision("first")})
    job_events = _JobEvents(_job_page(0, 0))
    app = _app(job_events, folders)

    async def run():
        received_notice = asyncio.Event()

        async def receive():
            await received_notice.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            if message.get("body", b"").startswith(b"event: project_revision\n"):
                received_notice.set()
                await asyncio.Future()

        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "method": "GET",
            "path": "/api/events",
            "query_string": b"folder=first",
            "headers": [],
            "http_version": "1.1",
            "scheme": "http",
            "server": ("testserver", 80),
        }
        await asyncio.wait_for(app(scope, receive, send), timeout=2)
        assert received_notice.is_set()

    asyncio.run(run())
    assert job_events.calls == [(0, (), 0, False)]
