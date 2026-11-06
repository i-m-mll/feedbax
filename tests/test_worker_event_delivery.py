"""Exercise delivery across the worker HTTP stream and Studio's WebSocket route."""

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
from fastapi.testclient import TestClient
from starlette.types import Message, Receive, Scope, Send
from starlette.websockets import WebSocketDisconnect

import feedbax.web.worker.app as worker_app
from feedbax.orchestration.assembly import assemble_run_bundle
from feedbax.orchestration.events import RunEventReader
from feedbax.orchestration.state import RunSetState
from feedbax.web.app import create_app as create_studio_app
from feedbax.web.services.training_service import TrainingService
from feedbax.web.services.worker_driver import WorkerHttpDriver
from feedbax.web.ws import training as training_ws

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
def initial_cursor_stream_closed() -> threading.Event:
    return threading.Event()


@pytest.fixture
def worker_url(
    initial_cursor_stream_closed: threading.Event, training_gates: _TrainingGates
) -> Iterator[str]:
    app = worker_app.create_app()

    async def observe_worker_stream(scope: Scope, receive: Receive, send: Send) -> None:
        async def send_with_slow_subscriber(message: Message) -> None:
            await send(message)
            if message["type"] == "http.response.body" and message.get("body", b"").startswith(
                b"data: "
            ):
                await asyncio.to_thread(training_gates.finished.wait, _TIMEOUT)

        try:
            await app(
                scope,
                receive,
                send_with_slow_subscriber if scope.get("path") == "/jobs/slow/stream" else send,
            )
        finally:
            if (
                scope["type"] == "http"
                and scope["path"].endswith("/stream")
                and scope["query_string"] == b"from_seq=0"
            ):
                initial_cursor_stream_closed.set()

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


def _studio_client() -> TestClient:
    app = create_studio_app()

    async def with_websocket_deadline(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "websocket":
            await app(scope, receive, send)
            return
        async with asyncio.timeout(2 * _TIMEOUT):
            await app(scope, receive, send)

    return TestClient(with_websocket_deadline)


def _studio_event(websocket, job_id: str) -> tuple[int, str]:
    event = websocket.receive_json()
    assert event["job_id"] == job_id
    return event["seq"], event["type"]


def _assert_studio_closed(websocket) -> None:
    with pytest.raises(WebSocketDisconnect) as exc:
        websocket.receive_json()
    assert exc.value.code == 1000


@pytest.fixture
def studio_service(monkeypatch, tmp_path, worker_url) -> TrainingService:
    monkeypatch.setenv("FEEDBAX_ORCHESTRATION_ROOT", str(tmp_path / "orchestration"))
    monkeypatch.delenv("FEEDBAX_WORKER_URL", raising=False)
    service = TrainingService()
    service.connect_remote(worker_url)
    monkeypatch.setattr(training_ws, "training_service", service)
    return service


def test_driver_and_studio_each_receive_all_events_with_websocket_resume(
    training_gates, worker_url, studio_service, initial_cursor_stream_closed
) -> None:
    request, context, registry = studio_service._build_worker_assembly_request(
        worker_start={"total_batches": 2}
    )
    bundle = assemble_run_bundle(
        request, run_set_id="2026-07-12-a1b2c3d4", context=context, registry=registry
    )
    row = bundle.rows[0]
    state = RunSetState(run_set_id=bundle.run_set_id)
    driver = WorkerHttpDriver(base_url=worker_url, request_timeout=2 * _TIMEOUT)
    reader = RunEventReader(bundle.run_set_dir / "events" / f"{row.row_id}.events.jsonl")
    path = f"/ws/training/{row.row_id}"
    studio_events = [(0, "training_progress"), (1, "training_progress"), (2, "training_complete")]

    try:
        assert driver.launch_row(bundle, row, state)["status"] == "running"
        assert _wait_for_driver_events(reader, 1) == [(0, "progress")]
        with _studio_client() as client:
            with client.websocket_connect(path) as continuous:
                assert _studio_event(continuous, row.row_id) == studio_events[0]
                with client.websocket_connect(f"{path}?from_seq=0") as interrupted:
                    first = _studio_event(interrupted, row.row_id)
                    assert first == studio_events[0]
                    interrupted.close()
                    assert initial_cursor_stream_closed.wait(_TIMEOUT), (
                        "Studio did not close the worker subscription after WebSocket disconnect"
                    )
                training_gates.next_batch.set()
                assert _studio_event(continuous, row.row_id) == studio_events[1]
                assert _wait_for_driver_events(reader, 2) == [(0, "progress"), (1, "progress")]

                with client.websocket_connect(f"{path}?from_seq={first[0] + 1}") as resumed:
                    assert _studio_event(resumed, row.row_id) == studio_events[1]
                    assert training_gates.status_completed.wait(_TIMEOUT)
                    response = httpx.get(f"{worker_url}/jobs/{row.row_id}/status", timeout=_TIMEOUT)
                    assert response.raise_for_status().json()["status"] == "completed"
                    assert driver.probe(bundle, row, state).status == "running"
                    assert [event.seq for event in reader.read_all()] == [0, 1]
                    training_gates.finish.set()
                    assert _studio_event(resumed, row.row_id) == studio_events[2]
                    _assert_studio_closed(resumed)
                assert _studio_event(continuous, row.row_id) == studio_events[2]
                _assert_studio_closed(continuous)

            with client.websocket_connect(path) as late:
                assert [_studio_event(late, row.row_id) for _ in studio_events] == studio_events
                _assert_studio_closed(late)

        deadline = time.monotonic() + _TIMEOUT
        probe = driver.probe(bundle, row, state)
        while probe.status == "running" and time.monotonic() < deadline:
            time.sleep(0.01)
            probe = driver.probe(bundle, row, state)
        assert probe.status == "completed"
        assert [(event.seq, event.type) for event in reader.read_all()] == [
            (0, "progress"),
            (1, "progress"),
            (2, "complete"),
        ]
    finally:
        training_gates.next_batch.set()
        training_gates.finish.set()
        driver.teardown(bundle, state)


def test_worker_replays_retained_suffix_and_rejects_expired_explicit_cursor(
    monkeypatch, training_gates, worker_url, studio_service
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

    with _studio_client() as client:
        with client.websocket_connect("/ws/training/retained?from_seq=0") as websocket:
            event = websocket.receive_json()
            assert event["job_id"] == "retained"
            assert event["type"] == "training_error"
            assert event["error"] == "Training worker event cursor expired."
            _assert_studio_closed(websocket)


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
