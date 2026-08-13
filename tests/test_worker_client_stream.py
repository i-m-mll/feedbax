import asyncio

import httpx
import pytest

from feedbax.web.worker import client as worker_client


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
    def raise_for_status(self) -> None:
        return None

    async def aiter_lines(self):
        yield 'data: {"type": "training_complete", "job_id": "job-terminal", "seq": 4}'


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


async def _consume_stream() -> list[dict]:
    return [
        event
        async for event in worker_client.stream_events(
            "http://worker.invalid/secret",
            "job-secret",
            auth_token="secret-token",
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

    assert events == [
        {"type": "training_complete", "job_id": "job-terminal", "seq": 4}
    ]
    assert _TerminalAsyncClient.attempts == 1


async def _collect_terminal_stream() -> list[dict]:
    return [
        event
        async for event in worker_client.stream_events("http://worker", "job-terminal")
    ]
