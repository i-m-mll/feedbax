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


class _AcceptedResponse:
    failure: Exception | None = None

    def raise_for_status(self) -> None:
        return None

    async def aiter_lines(self):
        if self.failure is not None:
            raise self.failure
        if False:
            yield ""


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
    return [
        event
        async for event in worker_client.stream_events("http://worker", "job-terminal")
    ]
