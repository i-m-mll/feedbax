"""Exercise worker HTTP event delivery and driver custody."""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

import httpx
import pytest
import uvicorn
from starlette.types import Message, Receive, Scope, Send

import feedbax.web.worker.app as worker_app
from feedbax.orchestration.events import RunEventReader
from feedbax.contracts.studio_training import StudioTrainingAssemblySpec
from feedbax.orchestration.state import RunSetState
from feedbax.web.services.worker_driver import WorkerHttpDriver
from tests.test_orchestration_core import _bundle, _compiled_row

_TIMEOUT = 5.0


@dataclass
class _TrainingGates:
    next_batch: threading.Event = field(default_factory=threading.Event)
    status_completed: threading.Event = field(default_factory=threading.Event)
    finish: threading.Event = field(default_factory=threading.Event)
    finished: threading.Event = field(default_factory=threading.Event)


@pytest.fixture
def training_gates(monkeypatch) -> Iterator[_TrainingGates]:
    gates = _TrainingGates()

    def run_training(job: worker_app._Job, _bootstrap_state) -> None:
        worker_app._emit(
            job, {"type": "training_progress", "job_id": job.job_id, "program_step": 1}
        )
        if not gates.next_batch.wait(2 * _TIMEOUT):
            return
        worker_app._emit(
            job, {"type": "training_progress", "job_id": job.job_id, "program_step": 2}
        )
        worker_app._mark_job_terminal(job, worker_app.WorkerStatus.COMPLETED)
        gates.status_completed.set()
        if not gates.finish.wait(2 * _TIMEOUT):
            return
        worker_app._emit(
            job, {"type": "training_complete", "job_id": job.job_id, "batch": 2, "loss": 0.25}
        )
        gates.finished.set()

    monkeypatch.setattr(worker_app, "_run_training", run_training)
    yield gates
    gates.next_batch.set()
    gates.finish.set()


@pytest.fixture
def worker_url(training_gates: _TrainingGates) -> Iterator[str]:
    app = worker_app.create_app()

    async def observe_worker_stream(scope: Scope, receive: Receive, send: Send) -> None:
        async def send_with_slow_subscriber(message: Message) -> None:
            await send(message)
            if message["type"] == "http.response.body" and message.get("body", b"").startswith(b"data: "):
                await asyncio.to_thread(training_gates.finished.wait, _TIMEOUT)

        await app(scope, receive, send_with_slow_subscriber if scope.get("path") == "/jobs/slow/stream" else send)

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        server = uvicorn.Server(
            uvicorn.Config(
                observe_worker_stream,
                log_level="warning",
                ws="none",
                lifespan="on",
                timeout_graceful_shutdown=1,
            )
        )
        thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + _TIMEOUT
            while not server.started and thread.is_alive() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert server.started, "Worker HTTP server failed to start"
            yield f"http://127.0.0.1:{listener.getsockname()[1]}"
        finally:
            server.should_exit = True
            thread.join(timeout=_TIMEOUT)
            assert not thread.is_alive(), "Worker HTTP server failed to stop"


def _wait_for_driver_events(reader: RunEventReader, count: int) -> list[tuple[int, str]]:
    deadline = time.monotonic() + _TIMEOUT
    events = reader.read_all()
    while len(events) < count and time.monotonic() < deadline:
        time.sleep(0.01)
        events = reader.read_all()
    return [(event.seq, event.type) for event in events]


def test_driver_retains_worker_events_through_completion(training_gates, worker_url, tmp_path) -> None:
    row_id = "2026-07-12-a1b2c3d4-studio"
    payload = StudioTrainingAssemblySpec(total_batches=2).worker_payload()
    compiled = _compiled_row(row_id).model_copy(update={"payload": payload, "resolved_semantics": payload})
    bundle = _bundle(
        tmp_path, run_set_id="2026-07-12-a1b2c3d4", rows=[compiled], driver="worker-http",
    )
    row = bundle.rows[0]
    state = RunSetState(run_set_id=bundle.run_set_id)
    driver = WorkerHttpDriver(base_url=worker_url, request_timeout=2 * _TIMEOUT)
    reader = RunEventReader(bundle.run_set_dir / "events" / f"{row.row_id}.events.jsonl")

    try:
        assert driver.launch_row(bundle, row, state)["status"] == "running"
        assert _wait_for_driver_events(reader, 1) == [(0, "progress")]
        training_gates.next_batch.set()
        assert _wait_for_driver_events(reader, 2) == [(0, "progress"), (1, "progress")]
        assert training_gates.status_completed.wait(_TIMEOUT)
        response = httpx.get(f"{worker_url}/jobs/{row.row_id}/status", timeout=_TIMEOUT)
        assert response.raise_for_status().json()["status"] == "completed"
        assert driver.probe(bundle, row, state).status == "running"
        assert [event.seq for event in reader.read_all()] == [0, 1]
        training_gates.finish.set()
        assert _wait_for_driver_events(reader, 3) == [(0, "progress"), (1, "progress"), (2, "complete")]
        deadline = time.monotonic() + _TIMEOUT
        probe = driver.probe(bundle, row, state)
        while probe.status == "running" and time.monotonic() < deadline:
            time.sleep(0.01)
            probe = driver.probe(bundle, row, state)
        assert probe.status == "completed"
    finally:
        training_gates.next_batch.set()
        training_gates.finish.set()
        driver.teardown(bundle, state)


def test_worker_replays_retained_suffix_and_rejects_expired_explicit_cursor(
    monkeypatch, training_gates, worker_url
) -> None:
    monkeypatch.setattr(worker_app, "_EVENT_BUFFER_MAX", 2)
    training_gates.next_batch.set()
    training_gates.finish.set()
    with httpx.Client(base_url=worker_url, timeout=_TIMEOUT) as client:
        response = client.post(
            "/start", json={"job_id": "retained", "run_set_id": "run", "total_batches": 2}
        )
        assert response.status_code == 200
        assert training_gates.finished.wait(_TIMEOUT), "Fake training failed to finish"

        path = "/jobs/retained/stream"
        expired = client.get(path, params={"from_seq": 0})
        assert expired.status_code == 410
        assert expired.json()["detail"] == {
            "code": "worker_stream_cursor_expired",
            "requested_seq": 0,
            "oldest_seq": 1,
        }

        for params in ({}, {"from_seq": 1}):
            response = client.get(path, params=params)
            assert response.status_code == 200
            events = [
                json.loads(line.removeprefix("data: "))
                for line in response.iter_lines()
                if line.startswith("data: ")
            ]
            assert [(event["seq"], event["type"]) for event in events] == [
                (1, "progress"),
                (2, "complete"),
            ]

def test_live_worker_stream_reports_a_gap_when_it_falls_behind_retention(
    monkeypatch, training_gates, worker_url
) -> None:
    monkeypatch.setattr(worker_app, "_EVENT_BUFFER_MAX", 1)
    with httpx.Client(base_url=worker_url, timeout=_TIMEOUT) as client:
        response = client.post(
            "/start", json={"job_id": "slow", "run_set_id": "run", "total_batches": 2}
        )
        response.raise_for_status()
        with client.stream("GET", "/jobs/slow/stream") as stream:
            assert stream.status_code == 200
            lines = stream.iter_lines()
            assert json.loads(next(lines).removeprefix("data: "))["seq"] == 0
            assert next(lines) == ""

            training_gates.next_batch.set()
            training_gates.finish.set()
            assert next(lines) == "event: stream_gap"
            assert json.loads(next(lines).removeprefix("data: ")) == {
                "code": "worker_stream_cursor_expired",
                "requested_seq": 1,
                "oldest_seq": 2,
            }
            assert list(lines) == [""]
