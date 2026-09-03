"""Canonical trust and consumption policy for Studio worker requests."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import math
import socket
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

if TYPE_CHECKING:
    import httpx


class WorkerTransportError(RuntimeError):
    """A redacted worker transport failure safe for user-visible diagnostics."""


class WorkerConfigurationError(WorkerTransportError, ValueError):
    """Worker transport configuration cannot satisfy the trust boundary."""


class WorkerResponseError(WorkerTransportError):
    """The worker returned a refused or unsuccessful response."""

    def __init__(self, status_code: int) -> None:
        if 300 <= status_code < 400:
            message = "Training worker redirects are not allowed"
        else:
            message = f"Training worker request failed with status {status_code}"
        super().__init__(message)
        self.status_code = status_code


class WorkerConsumptionError(WorkerTransportError):
    """A response exceeded the configured time or size budget."""


@dataclass(frozen=True)
class WorkerConsumptionLimits:
    """Finite budgets applied to every worker request and event stream."""

    connect_seconds: float = 3.0
    read_seconds: float = 10.0
    write_seconds: float = 10.0
    pool_seconds: float = 3.0
    request_seconds: float = 30.0
    checkpoint_seconds: float = 10 * 60.0
    stream_seconds: float = 6 * 60 * 60.0
    response_bytes: int = 2 * 1024 * 1024
    checkpoint_bytes: int = 2 * 1024 * 1024 * 1024
    stream_bytes: int = 256 * 1024 * 1024
    event_bytes: int = 256 * 1024
    event_count: int = 1_000_000

    def __post_init__(self) -> None:
        numeric = (
            self.connect_seconds,
            self.read_seconds,
            self.write_seconds,
            self.pool_seconds,
            self.request_seconds,
            self.checkpoint_seconds,
            self.stream_seconds,
            self.response_bytes,
            self.checkpoint_bytes,
            self.stream_bytes,
            self.event_bytes,
            self.event_count,
        )
        if any(not math.isfinite(float(value)) or value <= 0 for value in numeric):
            raise WorkerConfigurationError("Worker consumption limits must be positive")
        if self.event_bytes > self.stream_bytes:
            raise WorkerConfigurationError(
                "Worker event size cannot exceed the total stream byte budget"
            )

    def httpx_timeout(self) -> httpx.Timeout:
        import httpx

        return httpx.Timeout(
            connect=self.connect_seconds,
            read=self.read_seconds,
            write=self.write_seconds,
            pool=self.pool_seconds,
        )


DEFAULT_WORKER_LIMITS = WorkerConsumptionLimits()


def _canonical_origin(value: str) -> tuple[str, str, int, str]:
    if not isinstance(value, str) or not value.strip():
        raise WorkerConfigurationError("Worker URL must be a non-empty absolute origin")
    try:
        parsed = urlsplit(value.strip())
        port = parsed.port
    except ValueError as exc:
        raise WorkerConfigurationError("Worker URL is not a valid absolute origin") from exc
    scheme = parsed.scheme.lower()
    host = parsed.hostname
    if scheme not in {"http", "https"} or host is None:
        raise WorkerConfigurationError("Worker URL must use an explicit HTTP or HTTPS origin")
    if parsed.username is not None or parsed.password is not None:
        raise WorkerConfigurationError("Worker credentials must not appear in the URL")
    if parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
        raise WorkerConfigurationError("Worker URL must be an exact origin without path or query")
    try:
        host.encode("ascii")
    except UnicodeEncodeError as exc:
        raise WorkerConfigurationError("Worker origin host must be ASCII") from exc
    host = host.lower()
    if port is None:
        port = 443 if scheme == "https" else 80
    rendered_host = f"[{host}]" if ":" in host else host
    default_port = 443 if scheme == "https" else 80
    suffix = "" if port == default_port else f":{port}"
    return f"{scheme}://{rendered_host}{suffix}", host, port, scheme


def _resolve_addresses(
    host: str, port: int
) -> tuple[ipaddress.IPv4Address | ipaddress.IPv6Address, ...]:
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        try:
            records = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except OSError as exc:
            raise WorkerConfigurationError("Worker origin could not be resolved") from exc
        addresses = tuple(dict.fromkeys(ipaddress.ip_address(record[4][0]) for record in records))
    else:
        addresses = (literal,)
    if not addresses:
        raise WorkerConfigurationError("Worker origin did not resolve to an address")
    return addresses


def _is_safe_remote_address(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return bool(
        address.is_global
        and not address.is_loopback
        and not address.is_private
        and not address.is_link_local
        and not address.is_multicast
        and not address.is_unspecified
        and not address.is_reserved
    )


@dataclass(frozen=True)
class WorkerEndpoint:
    """Validated worker identity plus its credential and finite request policy."""

    origin: str
    host: str
    port: int
    local_loopback: bool
    allowed_origins: frozenset[str]
    credential: str | None = field(default=None, repr=False)
    limits: WorkerConsumptionLimits = DEFAULT_WORKER_LIMITS

    @classmethod
    def create(
        cls,
        base_url: str,
        *,
        credential: str | None = None,
        allowed_origins: Iterable[str] = (),
        limits: WorkerConsumptionLimits = DEFAULT_WORKER_LIMITS,
    ) -> "WorkerEndpoint":
        origin, host, port, scheme = _canonical_origin(base_url)
        normalized_allowlist = frozenset(_canonical_origin(item)[0] for item in allowed_origins)
        if credential is not None and (not credential or "\r" in credential or "\n" in credential):
            raise WorkerConfigurationError("Worker credential must be non-empty and single-line")

        addresses = _resolve_addresses(host, port)
        try:
            host_literal = ipaddress.ip_address(host)
        except ValueError:
            explicit_local_host = host == "localhost"
        else:
            explicit_local_host = host_literal.is_loopback
        if explicit_local_host:
            if not all(address.is_loopback for address in addresses):
                raise WorkerConfigurationError(
                    "Local worker origin resolved outside the loopback interface"
                )
        else:
            if scheme != "https":
                raise WorkerConfigurationError("Remote workers require HTTPS")
            if origin not in normalized_allowlist:
                raise WorkerConfigurationError(
                    "Remote worker origin is not in the configured exact allowlist"
                )
            if credential is None:
                raise WorkerConfigurationError("Remote workers require an explicit credential")
            if not all(_is_safe_remote_address(address) for address in addresses):
                raise WorkerConfigurationError(
                    "Remote worker origin resolved to a forbidden network address"
                )

        return cls(
            origin=origin,
            host=host,
            port=port,
            local_loopback=explicit_local_host,
            allowed_origins=normalized_allowlist,
            credential=credential,
            limits=limits,
        )

    def revalidate_address(self) -> None:
        """Resolve again immediately before a request to detect address drift."""
        addresses = _resolve_addresses(self.host, self.port)
        if self.local_loopback:
            if not all(address.is_loopback for address in addresses):
                raise WorkerConfigurationError(
                    "Local worker origin no longer resolves only to loopback"
                )
            return
        if not all(_is_safe_remote_address(address) for address in addresses):
            raise WorkerConfigurationError(
                "Remote worker origin resolved to a forbidden network address"
            )

    async def revalidate_address_async(self) -> None:
        await asyncio.to_thread(self.revalidate_address)

    def url(self, path: str) -> str:
        if not path.startswith("/") or "://" in path or "\r" in path or "\n" in path:
            raise ValueError("Worker request path must be an absolute local path")
        return self.origin + path

    def authorization_headers(self) -> dict[str, str]:
        if self.credential is None:
            return {}
        return {"Authorization": f"Bearer {self.credential}"}


def _check_response(response: httpx.Response) -> None:
    if response.status_code < 200 or response.status_code >= 300:
        raise WorkerResponseError(response.status_code)


def _check_content_length(response: httpx.Response, maximum: int) -> None:
    raw = response.headers.get("content-length")
    if raw is None:
        return
    try:
        length = int(raw)
    except ValueError as exc:
        raise WorkerConsumptionError("Training worker returned an invalid response length") from exc
    if length < 0 or length > maximum:
        raise WorkerConsumptionError("Training worker response exceeded its byte budget")


async def read_response_bytes(response: httpx.Response, maximum: int) -> bytes:
    _check_content_length(response, maximum)
    data = bytearray()
    async for chunk in response.aiter_bytes():
        data.extend(chunk)
        if len(data) > maximum:
            raise WorkerConsumptionError("Training worker response exceeded its byte budget")
    return bytes(data)


def read_response_bytes_sync(response: httpx.Response, maximum: int, deadline: float) -> bytes:
    _check_content_length(response, maximum)
    data = bytearray()
    for chunk in response.iter_bytes():
        data.extend(chunk)
        if len(data) > maximum:
            raise WorkerConsumptionError("Training worker response exceeded its byte budget")
        if time.monotonic() > deadline:
            raise WorkerConsumptionError("Training worker request exceeded its overall time budget")
    return bytes(data)


def decode_json_object(data: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkerTransportError("Training worker returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise WorkerTransportError("Training worker returned a non-object JSON response")
    return payload


async def request_json(
    endpoint: WorkerEndpoint,
    method: str,
    path: str,
    *,
    json_body: Mapping[str, Any] | None = None,
    params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Make one redirect-free, revalidated, bounded JSON request."""
    import httpx

    try:
        async with asyncio.timeout(endpoint.limits.request_seconds):
            await endpoint.revalidate_address_async()
            async with httpx.AsyncClient(
                timeout=endpoint.limits.httpx_timeout(),
                follow_redirects=False,
            ) as client:
                async with client.stream(
                    method,
                    endpoint.url(path),
                    json=json_body,
                    params=params,
                    headers=endpoint.authorization_headers(),
                ) as response:
                    _check_response(response)
                    data = await read_response_bytes(response, endpoint.limits.response_bytes)
        return decode_json_object(data)
    except (WorkerTransportError, WorkerConfigurationError):
        raise
    except TimeoutError as exc:
        raise WorkerConsumptionError(
            "Training worker request exceeded its overall time budget"
        ) from exc
    except httpx.HTTPError as exc:
        raise WorkerTransportError("Training worker request failed") from exc


def request_json_sync(
    endpoint: WorkerEndpoint,
    method: str,
    path: str,
    *,
    json_body: Mapping[str, Any] | None = None,
    params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Synchronous form of the canonical bounded JSON request."""
    import httpx

    deadline = time.monotonic() + endpoint.limits.request_seconds
    try:
        endpoint.revalidate_address()
        with httpx.stream(
            method,
            endpoint.url(path),
            json=json_body,
            params=params,
            headers=endpoint.authorization_headers(),
            timeout=endpoint.limits.httpx_timeout(),
            follow_redirects=False,
        ) as response:
            _check_response(response)
            data = read_response_bytes_sync(response, endpoint.limits.response_bytes, deadline)
        return decode_json_object(data)
    except (WorkerTransportError, WorkerConfigurationError):
        raise
    except httpx.HTTPError as exc:
        raise WorkerTransportError("Training worker request failed") from exc


@dataclass
class SSEConsumptionBudget:
    """Incremental byte, event-count, event-size, and lifetime enforcement."""

    limits: WorkerConsumptionLimits
    started_at: float = field(default_factory=time.monotonic)
    total_bytes: int = 0
    total_events: int = 0
    buffered: bytearray = field(default_factory=bytearray)

    def feed(self, chunk: bytes) -> tuple[bytes, ...]:
        self.check_lifetime()
        self.total_bytes += len(chunk)
        if self.total_bytes > self.limits.stream_bytes:
            raise WorkerConsumptionError("Training worker stream exceeded its byte budget")
        self.buffered.extend(chunk)
        if len(self.buffered) > self.limits.event_bytes + 8 and b"\n" not in self.buffered:
            raise WorkerConsumptionError("Training worker event exceeded its size budget")

        payloads: list[bytes] = []
        while True:
            newline = self.buffered.find(b"\n")
            if newline < 0:
                break
            line = bytes(self.buffered[:newline]).rstrip(b"\r")
            del self.buffered[: newline + 1]
            if not line.startswith(b"data:"):
                continue
            payload = line[len(b"data:") :].strip()
            if not payload:
                continue
            self._admit_event(payload)
            payloads.append(payload)
        if len(self.buffered) > self.limits.event_bytes + 8:
            raise WorkerConsumptionError("Training worker event exceeded its size budget")
        return tuple(payloads)

    def check_lifetime(self) -> None:
        if time.monotonic() - self.started_at > self.limits.stream_seconds:
            raise WorkerConsumptionError("Training worker stream exceeded its lifetime budget")

    def admit_event(self, payload: bytes) -> None:
        """Apply the same stream budget to one downstream WebSocket event."""
        self.check_lifetime()
        self.total_bytes += len(payload)
        if self.total_bytes > self.limits.stream_bytes:
            raise WorkerConsumptionError("Training worker stream exceeded its byte budget")
        self._admit_event(payload)

    def _admit_event(self, payload: bytes) -> None:
        if len(payload) > self.limits.event_bytes:
            raise WorkerConsumptionError("Training worker event exceeded its size budget")
        self.total_events += 1
        if self.total_events > self.limits.event_count:
            raise WorkerConsumptionError("Training worker stream exceeded its event budget")
