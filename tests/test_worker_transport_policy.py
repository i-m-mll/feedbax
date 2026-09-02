from __future__ import annotations

import asyncio
import ipaddress
from dataclasses import replace
from typing import Any

import pytest

import feedbax.web.worker.transport as transport
from feedbax.web.orchestration import gcp
from feedbax.web.orchestration.startup_script import make_startup_script
from feedbax.web.services.training_service import TrainingService
from feedbax.web.worker import client as worker_client
from feedbax.web.worker.transport import (
    DEFAULT_WORKER_LIMITS,
    SSEConsumptionBudget,
    WorkerConfigurationError,
    WorkerConsumptionError,
    WorkerEndpoint,
    WorkerResponseError,
    request_json,
)


PUBLIC_ADDRESS = ipaddress.ip_address("93.184.216.34")
PRIVATE_ADDRESS = ipaddress.ip_address("10.0.0.2")


class _AsyncResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        chunks: tuple[bytes, ...] = (b'{"ok": true}',),
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.chunks = chunks
        self.headers = headers or {}

    async def aiter_bytes(self):
        for chunk in self.chunks:
            yield chunk


class _AsyncStream:
    def __init__(self, response: _AsyncResponse) -> None:
        self.response = response

    async def __aenter__(self) -> _AsyncResponse:
        return self.response

    async def __aexit__(self, *args: object) -> None:
        del args


def _install_async_client(
    monkeypatch: pytest.MonkeyPatch,
    response: _AsyncResponse,
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    class Client:
        def __init__(self, **kwargs: Any) -> None:
            calls.append({"client": kwargs})

        async def __aenter__(self) -> "Client":
            return self

        async def __aexit__(self, *args: object) -> None:
            del args

        def stream(self, method: str, url: str, **kwargs: Any) -> _AsyncStream:
            calls.append({"method": method, "url": url, **kwargs})
            return _AsyncStream(response)

    monkeypatch.setattr(transport.httpx, "AsyncClient", Client)
    return calls


def test_loopback_development_endpoint_needs_no_credential() -> None:
    endpoint = WorkerEndpoint.create("http://127.0.0.1:8765")

    assert endpoint.local_loopback is True
    assert endpoint.origin == "http://127.0.0.1:8765"
    assert endpoint.authorization_headers() == {}


def test_remote_endpoint_uses_https_exact_allowlist_and_authorization_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolutions = 0

    def resolve(_host: str, _port: int):
        nonlocal resolutions
        resolutions += 1
        return (PUBLIC_ADDRESS,)

    monkeypatch.setattr(transport, "_resolve_addresses", resolve)
    endpoint = WorkerEndpoint.create(
        "https://worker.example:9443",
        credential="top-secret",
        allowed_origins=("https://worker.example:9443",),
    )
    calls = _install_async_client(monkeypatch, _AsyncResponse())

    assert asyncio.run(request_json(endpoint, "GET", "/health")) == {"ok": True}
    request = calls[1]
    assert request["url"] == "https://worker.example:9443/health"
    assert request["headers"] == {"Authorization": "Bearer top-secret"}
    assert "top-secret" not in request["url"]
    assert calls[0]["client"]["follow_redirects"] is False
    assert resolutions == 2
    assert "top-secret" not in repr(endpoint)


@pytest.mark.parametrize(
    ("url", "credential", "allowlist", "message"),
    [
        ("http://worker.example", "token", ("http://worker.example",), "require HTTPS"),
        ("https://worker.example", None, ("https://worker.example",), "credential"),
        ("https://worker.example", "token", (), "allowlist"),
        ("https://token@worker.example", "token", ("https://worker.example",), "must not"),
        ("ftp://worker.example", "token", ("ftp://worker.example",), "HTTP or HTTPS"),
    ],
)
def test_remote_endpoint_rejects_unsafe_or_incomplete_configuration(
    monkeypatch: pytest.MonkeyPatch,
    url: str,
    credential: str | None,
    allowlist: tuple[str, ...],
    message: str,
) -> None:
    monkeypatch.setattr(transport, "_resolve_addresses", lambda _host, _port: (PUBLIC_ADDRESS,))

    with pytest.raises(WorkerConfigurationError, match=message):
        WorkerEndpoint.create(url, credential=credential, allowed_origins=allowlist)


def test_remote_endpoint_rejects_private_resolution_and_rebinding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        transport,
        "_resolve_addresses",
        lambda _host, _port: (PRIVATE_ADDRESS,),
    )
    with pytest.raises(WorkerConfigurationError, match="forbidden network address"):
        WorkerEndpoint.create(
            "https://worker.example",
            credential="token",
            allowed_origins=("https://worker.example",),
        )

    resolutions = iter(((PUBLIC_ADDRESS,), (PRIVATE_ADDRESS,)))
    monkeypatch.setattr(transport, "_resolve_addresses", lambda _host, _port: next(resolutions))
    endpoint = WorkerEndpoint.create(
        "https://worker.example",
        credential="token",
        allowed_origins=("https://worker.example",),
    )
    with pytest.raises(WorkerConfigurationError, match="forbidden network address"):
        asyncio.run(request_json(endpoint, "GET", "/health"))


def test_redirects_and_oversize_json_are_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    endpoint = WorkerEndpoint.create("http://127.0.0.1:8765")
    _install_async_client(monkeypatch, _AsyncResponse(status_code=302))
    with pytest.raises(WorkerResponseError, match="redirects are not allowed"):
        asyncio.run(request_json(endpoint, "GET", "/health"))

    endpoint = replace(
        endpoint,
        limits=replace(DEFAULT_WORKER_LIMITS, response_bytes=4),
    )
    _install_async_client(monkeypatch, _AsyncResponse(chunks=(b'{"too": "large"}',)))
    with pytest.raises(WorkerConsumptionError, match="byte budget"):
        asyncio.run(request_json(endpoint, "GET", "/health"))


def test_sse_budget_caps_bytes_event_size_count_and_lifetime(monkeypatch) -> None:
    limits = replace(
        DEFAULT_WORKER_LIMITS,
        stream_bytes=32,
        event_bytes=8,
        event_count=1,
        stream_seconds=1,
    )
    budget = SSEConsumptionBudget(limits)
    assert budget.feed(b"data: {}\n") == (b"{}",)
    with pytest.raises(WorkerConsumptionError, match="event budget"):
        budget.feed(b"data: {}\n")

    oversize = SSEConsumptionBudget(limits)
    with pytest.raises(WorkerConsumptionError, match="size budget"):
        oversize.feed(b"data: 12345678901")

    expired = SSEConsumptionBudget(limits)
    monkeypatch.setattr(transport.time, "monotonic", lambda: expired.started_at + 2)
    with pytest.raises(WorkerConsumptionError, match="lifetime budget"):
        expired.check_lifetime()


def test_training_service_remote_config_is_server_allowlisted_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(transport, "_resolve_addresses", lambda _host, _port: (PUBLIC_ADDRESS,))
    service = TrainingService(allowed_remote_origins=("https://worker.example",))
    service.connect_remote("https://worker.example", "token")
    assert service.worker_url() == "https://worker.example"

    with pytest.raises(WorkerConfigurationError, match="allowlist"):
        TrainingService().connect_remote("https://worker.example", "token")
    with pytest.raises(WorkerConfigurationError, match="credential"):
        service.connect_remote("https://worker.example")

    monkeypatch.setenv("FEEDBAX_WORKER_ALLOWED_ORIGINS", "not-json")
    with pytest.raises(WorkerConfigurationError, match="JSON array"):
        TrainingService()


def test_checkpoint_download_refuses_oversize_response(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    endpoint = WorkerEndpoint.create("http://127.0.0.1:8765")
    endpoint = replace(
        endpoint,
        limits=replace(DEFAULT_WORKER_LIMITS, checkpoint_bytes=4),
    )
    _install_async_client(monkeypatch, _AsyncResponse(chunks=(b"12345",)))

    with pytest.raises(WorkerConsumptionError, match="byte budget"):
        asyncio.run(worker_client.download_checkpoint(endpoint, "job-a", str(tmp_path / "out")))
    assert not (tmp_path / "out").exists()


def test_gcp_metadata_and_worker_startup_keep_credentials_out_of_argv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[str] = []

    async def fake_run_gcloud(*args: str):
        captured.extend(args)
        return [{"name": "worker", "status": "PROVISIONING"}]

    monkeypatch.setattr(gcp, "_run_gcloud", fake_run_gcloud)
    config = gcp.InstanceConfig(
        project="project",
        zone="zone",
        auth_token="argv-secret",
    )
    asyncio.run(gcp.create_instance(config, "worker"))

    argv = " ".join(captured)
    assert "argv-secret" not in argv
    assert "--metadata-from-file=" in argv
    script = make_startup_script()
    assert "--auth-token" not in script
    assert "FEEDBAX_WORKER_AUTH_TOKEN" in script


def test_gcp_rejects_missing_or_unsupported_credentials() -> None:
    with pytest.raises(ValueError, match="explicit worker credential"):
        gcp.InstanceConfig(project="project", zone="zone")
    with pytest.raises(ValueError, match="Tailscale bootstrap is unsupported"):
        gcp.InstanceConfig(
            project="project",
            zone="zone",
            auth_token="token",
            ts_auth_key="tailscale-secret",
        )
