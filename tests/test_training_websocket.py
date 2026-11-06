import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

websockets = pytest.importorskip("starlette.websockets")
WebSocketState = websockets.WebSocketState
from feedbax.orchestration.events import RunEvent  # noqa: E402
from feedbax.web.app import create_app  # noqa: E402
from feedbax.web.services.training_service import TrainingService  # noqa: E402
from feedbax.web.worker.client import WorkerEventStreamError  # noqa: E402
from feedbax.web.worker.transport import DEFAULT_WORKER_LIMITS  # noqa: E402
from feedbax.web.ws import training  # noqa: E402


class FakeWebSocket:
    def __init__(self, *, fail_on_send: bool = False):
        self.application_state = WebSocketState.CONNECTING
        self.client_state = WebSocketState.CONNECTED
        self.sent: list[dict] = []
        self.closed = False
        self.fail_on_send = fail_on_send

    async def accept(self) -> None:
        self.application_state = WebSocketState.CONNECTED

    async def receive(self) -> dict:
        await asyncio.Future()

    async def send_json(self, payload: dict) -> None:
        if self.fail_on_send:
            raise RuntimeError("websocket.disconnect")
        self.sent.append(payload)

    async def close(self) -> None:
        self.closed = True
        self.application_state = WebSocketState.DISCONNECTED


def test_training_ws_sends_upstream_errors_and_closes(monkeypatch) -> None:
    async def stream_progress(job_id: str, *, from_seq: int | None = None):
        assert job_id == "job-1"
        if False:
            yield None
        raise ValueError("worker failed")

    websocket = FakeWebSocket()
    monkeypatch.setattr(training.training_service, "stream_progress", stream_progress)

    asyncio.run(training.training_ws(websocket, "job-1"))

    assert len(websocket.sent) == 1
    error = websocket.sent[0]
    assert error["type"] == "training_error"
    assert error["job_id"] == "job-1"
    assert error["error"] == "Training worker stream failed."
    assert error["batch"] == 0
    assert error["diagnostics"][0]["code"] == "internal"
    assert error["diagnostics"][0]["message"] == "Training worker stream failed."
    assert error["seq"] >= 0
    assert isinstance(error["emitted_at_ms"], int)
    assert error["schema_version"] == "feedbax.spec.studio.api_transport.v3"
    assert websocket.closed is True


def test_training_ws_sends_sanitized_worker_stream_error(monkeypatch) -> None:
    async def stream_progress(job_id: str, *, from_seq: int | None = None):
        assert job_id == "job-stream-failure"
        if False:
            yield None
        raise WorkerEventStreamError("Training worker event stream failed.") from ValueError(
            "upstream secret"
        )

    websocket = FakeWebSocket()
    monkeypatch.setattr(training.training_service, "stream_progress", stream_progress)

    asyncio.run(training.training_ws(websocket, "job-stream-failure"))

    assert len(websocket.sent) == 1
    error = websocket.sent[0]
    assert error["type"] == "training_error"
    assert error["job_id"] == "job-stream-failure"
    assert error["error"] == "Training worker event stream failed."
    assert error["diagnostics"][0]["message"] == "Training worker event stream failed."
    assert "secret" not in str(error)
    assert websocket.closed is True


def test_training_ws_send_disconnect_exits_without_error(monkeypatch) -> None:
    async def stream_progress(job_id: str, *, from_seq: int | None = None):
        assert job_id == "job-2"
        yield SimpleNamespace(raw={"type": "training_progress", "job_id": job_id})

    websocket = FakeWebSocket(fail_on_send=True)
    monkeypatch.setattr(training.training_service, "stream_progress", stream_progress)

    asyncio.run(training.training_ws(websocket, "job-2"))

    assert websocket.sent == []


def test_training_ws_refuses_event_over_shared_size_budget(monkeypatch) -> None:
    async def stream_progress(job_id: str, *, from_seq: int | None = None):
        yield SimpleNamespace(raw={"type": "training_progress", "job_id": job_id})

    websocket = FakeWebSocket()
    monkeypatch.setattr(training.training_service, "stream_progress", stream_progress)
    monkeypatch.setattr(
        training.training_service,
        "worker_limits",
        lambda: replace(DEFAULT_WORKER_LIMITS, event_bytes=8),
    )

    asyncio.run(training.training_ws(websocket, "job-budget"))

    assert len(websocket.sent) == 1
    assert websocket.sent[0]["type"] == "training_error"
    assert "size budget" in websocket.sent[0]["error"]


def test_training_ws_streams_events_over_real_websocket(monkeypatch) -> None:
    async def stream_progress(job_id: str, *, from_seq: int | None = None):
        assert job_id == "job-real"
        assert from_seq == 8
        yield SimpleNamespace(
            raw={
                "type": "training_progress",
                "job_id": job_id,
                "progress": 0.5,
            }
        )

    monkeypatch.setattr(training.training_service, "stream_progress", stream_progress)

    with TestClient(create_app()) as client:
        with client.websocket_connect("/ws/training/job-real?from_seq=8") as websocket:
            assert websocket.receive_json() == {
                "type": "training_progress",
                "job_id": "job-real",
                "progress": 0.5,
            }


def test_training_ws_cancels_idle_worker_stream_on_disconnect(monkeypatch) -> None:
    async def run() -> None:
        worker_waiting = asyncio.Event()
        worker_closed = asyncio.Event()

        async def stream_events(endpoint, job_id: str, *, from_seq: int | None = None):
            assert from_seq == 8
            try:
                worker_waiting.set()
                await asyncio.Future()
                yield {}
            finally:
                worker_closed.set()

        class DisconnectingWebSocket(FakeWebSocket):
            async def receive(self) -> dict:
                await worker_waiting.wait()
                self.client_state = WebSocketState.DISCONNECTED
                return {"type": "websocket.disconnect"}

        service = training.training_service
        monkeypatch.setattr(service, "_endpoint", object())
        monkeypatch.setattr("feedbax.web.worker.client.stream_events", stream_events)
        monkeypatch.setattr(service, "worker_limits", lambda: DEFAULT_WORKER_LIMITS)

        websocket = DisconnectingWebSocket()
        await asyncio.wait_for(training.training_ws(websocket, "job-idle", from_seq=8), timeout=1)
        assert worker_closed.is_set()
        assert websocket.sent == []

    asyncio.run(run())


def test_training_service_replays_durable_events_inclusively(monkeypatch, tmp_path) -> None:
    service = TrainingService()
    monkeypatch.setattr(service, "_job_ref_for", lambda _: object())
    monkeypatch.setattr(service, "_load_bundle", lambda _: SimpleNamespace(run_set_dir=tmp_path))
    events_dir = tmp_path / "events"
    events_dir.mkdir()
    events_path = events_dir / "job-durable.events.jsonl"
    events = [
        RunEvent(
            run_set_id="run-set",
            row_id="job-durable",
            seq=seq,
            emitted_at_ms=seq,
            type="training_progress",
        )
        for seq in range(3)
    ]
    events_path.write_text("".join(event.model_dump_json() + "\n" for event in events))

    async def replay() -> list[int]:
        return [
            event.raw["worker_seq"]
            async for event in service.stream_progress("job-durable", from_seq=1)
        ]

    assert asyncio.run(replay()) == [1, 2]


def test_training_ws_bounds_stalled_writes_and_closes_upstream(monkeypatch) -> None:
    async def run() -> None:
        worker_closed = False

        async def stream_events(endpoint, job_id: str, *, from_seq: int | None = None):
            nonlocal worker_closed
            try:
                yield {"type": "training_progress", "seq": 0}
            finally:
                worker_closed = True

        class StalledWebSocket(FakeWebSocket):
            async def send_json(self, payload: dict) -> None:
                if payload["type"] == "training_error":
                    assert worker_closed
                self.sent.append(payload)
                await asyncio.Future()

            async def close(self) -> None:
                self.closed = True
                await asyncio.Future()

        service = training.training_service
        monkeypatch.setattr(service, "_endpoint", object())
        monkeypatch.setattr("feedbax.web.worker.client.stream_events", stream_events)
        monkeypatch.setattr(
            service, "worker_limits", lambda: replace(DEFAULT_WORKER_LIMITS, write_seconds=0.01)
        )

        websocket = StalledWebSocket()
        await asyncio.wait_for(training.training_ws(websocket, "job-stalled"), timeout=1)
        assert worker_closed
        assert websocket.closed
        assert [event["type"] for event in websocket.sent] == [
            "training_progress",
            "training_error",
        ]

    asyncio.run(run())
