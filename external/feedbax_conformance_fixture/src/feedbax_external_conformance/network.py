"""Runner-process outbound TCP denial applied before Feedbax imports."""

from __future__ import annotations

from contextlib import contextmanager
import socket
from typing import Iterator


_DENIAL_MESSAGE = "outbound TCP is forbidden during external conformance execution"


def _assert_outbound_tcp_denied() -> None:
    """Prove the three runner-process TCP connection APIs reject."""

    def connect() -> None:
        with socket.socket() as candidate:
            candidate.connect(("127.0.0.1", 9))

    def connect_ex() -> None:
        with socket.socket() as candidate:
            candidate.connect_ex(("127.0.0.1", 9))

    operations = (connect, connect_ex, lambda: socket.create_connection(("127.0.0.1", 9)))
    for operation in operations:
        try:
            operation()
        except RuntimeError as exc:
            if str(exc) != _DENIAL_MESSAGE:
                raise
        else:
            raise AssertionError("runner-process outbound TCP denial canary failed")


@contextmanager
def network_denied() -> Iterator[None]:
    """Reject outbound TCP connects in this process for the fixture execution."""
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_create_connection = socket.create_connection

    def denied(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError(_DENIAL_MESSAGE)

    socket.socket.connect = denied  # type: ignore[method-assign]
    socket.socket.connect_ex = denied  # type: ignore[method-assign]
    socket.create_connection = denied
    try:
        _assert_outbound_tcp_denied()
        yield
    finally:
        socket.socket.connect = original_connect  # type: ignore[method-assign]
        socket.socket.connect_ex = original_connect_ex  # type: ignore[method-assign]
        socket.create_connection = original_create_connection


__all__ = ["network_denied"]
