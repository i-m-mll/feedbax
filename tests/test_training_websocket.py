import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

websockets = pytest.importorskip("starlette.websockets")
WebSocketState = websockets.WebSocketState
from feedbax.web.app import create_app  # noqa: E402
from feedbax.web.worker.client import WorkerEventStreamError  # noqa: E402
from feedbax.web.worker.transport import DEFAULT_WORKER_LIMITS  # noqa: E402
from feedbax.web.ws import training  # noqa: E402


class FakeWebSocket:
    def __init__(self, *, fail_on_send: bool = False):
        self.application_state = WebSocketState.CONNECTING
        self.sent: list[dict] = []
        self.closed = False
        self.fail_on_send = fail_on_send

    async def accept(self) -> None:
        self.application_state = WebSocketState.CONNECTED

    async def send_json(self, payload: dict) -> None:
        if self.fail_on_send:
            raise RuntimeError("websocket.disconnect")
        self.sent.append(payload)

    async def close(self) -> None:
        self.closed = True
        self.application_state = WebSocketState.DISCONNECTED


def test_training_ws_sends_upstream_errors_and_closes(monkeypatch) -> None:
    async def stream_progress(job_id: str):
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
    async def stream_progress(job_id: str):
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
    async def stream_progress(job_id: str):
        assert job_id == "job-2"
        yield SimpleNamespace(raw={"type": "training_progress", "job_id": job_id})

    websocket = FakeWebSocket(fail_on_send=True)
    monkeypatch.setattr(training.training_service, "stream_progress", stream_progress)

    asyncio.run(training.training_ws(websocket, "job-2"))

    assert websocket.sent == []


def test_training_ws_refuses_event_over_shared_size_budget(monkeypatch) -> None:
    async def stream_progress(job_id: str):
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
    async def stream_progress(job_id: str):
        assert job_id == "job-real"
        yield SimpleNamespace(
            raw={
                "type": "training_progress",
                "job_id": job_id,
                "progress": 0.5,
            }
        )

    monkeypatch.setattr(training.training_service, "stream_progress", stream_progress)

    with TestClient(create_app()) as client:
        with client.websocket_connect("/ws/training/job-real") as websocket:
            assert websocket.receive_json() == {
                "type": "training_progress",
                "job_id": "job-real",
                "progress": 0.5,
            }
