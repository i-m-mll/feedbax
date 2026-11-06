import asyncio

import httpx
import pytest

from feedbax.web.worker import client as worker_client
from feedbax.web.worker.transport import WorkerEndpoint, WorkerEventCursorExpired


class _FailingAsyncClient:
    failure: Exception
    attempts = 0

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    async def __aenter__(self) -> "_FailingAsyncClient":
        type(self).attempts += 1
        raise self.failure

    async def __aexit__(self, *args: object) -> None:
        del args


class _TerminalResponse:
    status_code = 200
    headers: dict[str, str] = {}

    async def aiter_bytes(self):
        yield b'data: {"type": "training_complete", "job_id": "job-terminal", "seq": 4}\n'


class _TerminalStream:
    async def __aenter__(self) -> _TerminalResponse:
        return _TerminalResponse()

    async def __aexit__(self, *args: object) -> None:
        del args


class _TerminalAsyncClient:
    attempts = 0

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    async def __aenter__(self) -> "_TerminalAsyncClient":
        type(self).attempts += 1
        return self

    async def __aexit__(self, *args: object) -> None:
        del args

    def stream(self, *args: object, **kwargs: object) -> _TerminalStream:
        del args, kwargs
        return _TerminalStream()


class _AcceptedResponse:
    failure: Exception | None = None
    status_code = 200
    headers: dict[str, str] = {}

    async def aiter_bytes(self):
        if self.failure is not None:
            raise self.failure
        if False:
            yield b""


class _AcceptedStream:
    async def __aenter__(self) -> _AcceptedResponse:
        return _AcceptedResponse()

    async def __aexit__(self, *args: object) -> None:
        del args


class _AcceptedAsyncClient:
    attempts = 0

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    async def __aenter__(self) -> "_AcceptedAsyncClient":
        type(self).attempts += 1
        return self

    async def __aexit__(self, *args: object) -> None:
        del args

    def stream(self, *args: object, **kwargs: object) -> _AcceptedStream:
        del args, kwargs
        return _AcceptedStream()


async def _consume_stream() -> list[dict]:
    endpoint = WorkerEndpoint.create("http://127.0.0.1:8765", credential="secret-token")
    return [
        event
        async for event in worker_client.stream_events(
            endpoint,
            "job-secret",
        )
    ]


def test_stream_events_raises_sanitized_error_after_retry_exhaustion(monkeypatch) -> None:
    failure = httpx.ConnectError("connection exposed secret-token")
    _FailingAsyncClient.failure = failure
    _FailingAsyncClient.attempts = 0
    monkeypatch.setattr(worker_client.httpx, "AsyncClient", _FailingAsyncClient)
    monkeypatch.setattr(worker_client, "_MAX_RECONNECT_ATTEMPTS", 1)
    monkeypatch.setattr(worker_client, "_RECONNECT_DELAY", 0)

    with pytest.raises(worker_client.WorkerEventStreamError) as caught:
        asyncio.run(_consume_stream())

    assert str(caught.value) == "Training worker event stream failed."
    assert caught.value.__cause__ is failure
    assert _FailingAsyncClient.attempts == 2


def test_stream_events_raises_sanitized_error_for_unknown_failure(monkeypatch) -> None:
    failure = ValueError("parser exposed secret-token")
    _FailingAsyncClient.failure = failure
    _FailingAsyncClient.attempts = 0
    monkeypatch.setattr(worker_client.httpx, "AsyncClient", _FailingAsyncClient)

    with pytest.raises(worker_client.WorkerEventStreamError) as caught:
        asyncio.run(_consume_stream())

    assert str(caught.value) == "Training worker event stream failed."
    assert caught.value.__cause__ is failure
    assert _FailingAsyncClient.attempts == 1


def test_stream_events_returns_cleanly_after_terminal_event(monkeypatch) -> None:
    _TerminalAsyncClient.attempts = 0
    monkeypatch.setattr(worker_client.httpx, "AsyncClient", _TerminalAsyncClient)

    events = asyncio.run(_collect_terminal_stream())

    assert events == [{"type": "training_complete", "job_id": "job-terminal", "seq": 4}]
    assert _TerminalAsyncClient.attempts == 1


@pytest.mark.parametrize(
    ("failure", "cause_type"),
    [
        (httpx.ReadError("read exposed secret-token"), httpx.ReadError),
        (None, worker_client._WorkerEventStreamEnded),
    ],
    ids=["post-connect-read-failure", "non-terminal-clean-eof"],
)
def test_accepted_stream_exhausts_complete_attempts_with_backoff(
    monkeypatch,
    failure: Exception | None,
    cause_type: type[Exception],
) -> None:
    sleep_delays: list[float] = []
    original_sleep = asyncio.sleep

    async def record_sleep(delay: float) -> None:
        sleep_delays.append(delay)
        await original_sleep(0)

    _AcceptedResponse.failure = failure
    _AcceptedAsyncClient.attempts = 0
    monkeypatch.setattr(worker_client.httpx, "AsyncClient", _AcceptedAsyncClient)
    monkeypatch.setattr(worker_client.asyncio, "sleep", record_sleep)
    monkeypatch.setattr(worker_client, "_MAX_RECONNECT_ATTEMPTS", 2)
    monkeypatch.setattr(worker_client, "_RECONNECT_DELAY", 0.25)

    with pytest.raises(worker_client.WorkerEventStreamError) as caught:
        asyncio.run(asyncio.wait_for(_consume_stream(), timeout=0.5))

    assert str(caught.value) == "Training worker event stream failed."
    assert isinstance(caught.value.__cause__, cause_type)
    assert _AcceptedAsyncClient.attempts == 3
    assert sleep_delays == [0.25, 0.25]


async def _collect_terminal_stream() -> list[dict]:
    endpoint = WorkerEndpoint.create("http://127.0.0.1:8765")
    return [event async for event in worker_client.stream_events(endpoint, "job-terminal")]


class _InterruptedBody(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b'data: {"type": "training_progress", "seq": 8}\n'
        yield b'data: {"type": "training_progress", "seq"'
        raise httpx.ReadError("worker connection dropped")


def _mock_worker_responses(monkeypatch, *responses: httpx.Response) -> list[str]:
    client = httpx.AsyncClient
    pending = iter(responses)
    cursors = []

    def respond(request: httpx.Request) -> httpx.Response:
        cursors.append(request.url.params["from_seq"])
        return next(pending)

    monkeypatch.setattr(
        worker_client.httpx,
        "AsyncClient",
        lambda **kwargs: client(transport=httpx.MockTransport(respond), **kwargs),
    )
    return cursors


async def _collect_from_cursor() -> list[dict]:
    endpoint = WorkerEndpoint.create("http://127.0.0.1:8765")
    return [
        event async for event in worker_client.stream_events(endpoint, "job-resume", from_seq=8)
    ]


def test_stream_events_replays_from_last_complete_event(monkeypatch) -> None:
    cursors = _mock_worker_responses(
        monkeypatch,
        httpx.Response(200, stream=_InterruptedBody()),
        httpx.Response(
            200,
            content=(
                b'data: {"type": "training_progress", "seq": 8}\n'
                b'data: {"type": "training_complete", "seq": 9}\n'
            ),
        ),
    )
    monkeypatch.setattr(worker_client, "_RECONNECT_DELAY", 0)

    progress, resync, complete = asyncio.run(_collect_from_cursor())

    assert cursors == ["8", "9"]
    assert progress == {"type": "training_progress", "seq": 8}
    assert resync["type"] == "training_resync"
    assert resync["observed_worker_seq"] == 9
    assert "worker_seq" not in resync
    assert complete == {"type": "training_complete", "seq": 9}


@pytest.mark.parametrize(
    ("status_code", "content", "error_type", "message"),
    [
        (410, b"upstream secret", WorkerEventCursorExpired, "cursor expired"),
        (
            200,
            b'event: stream_gap\ndata: {"code": "worker_stream_cursor_expired", '
            b'"requested_seq": 8, "oldest_seq": 9}\n\n',
            WorkerEventCursorExpired,
            "cursor expired",
        ),
        (
            200,
            b'data: {"type": "training_complete", "seq": 9}\n',
            worker_client.WorkerEventStreamError,
            "sequence gap",
        ),
    ],
    ids=["expired-admission", "expired-live", "missing-sequence"],
)
def test_stream_events_surfaces_unavailable_cursor(
    monkeypatch, status_code, content, error_type, message
) -> None:
    cursors = _mock_worker_responses(monkeypatch, httpx.Response(status_code, content=content))

    with pytest.raises(error_type, match=message):
        asyncio.run(_collect_from_cursor())

    assert cursors == ["8"]


def test_stream_events_accepts_empty_terminal_resume(monkeypatch) -> None:
    cursors = _mock_worker_responses(monkeypatch, httpx.Response(204))

    assert asyncio.run(_collect_from_cursor()) == []
    assert cursors == ["8"]
