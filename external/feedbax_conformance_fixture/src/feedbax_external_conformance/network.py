"""Process-local network denial used before importing the Feedbax execution stack."""

from __future__ import annotations

from contextlib import contextmanager
import socket
from typing import Iterator


@contextmanager
def network_denied() -> Iterator[None]:
    """Reject socket connections for the complete fixture execution."""
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_create_connection = socket.create_connection

    def denied(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("network access is forbidden during external conformance execution")

    socket.socket.connect = denied  # type: ignore[method-assign]
    socket.socket.connect_ex = denied  # type: ignore[method-assign]
    socket.create_connection = denied
    try:
        yield
    finally:
        socket.socket.connect = original_connect  # type: ignore[method-assign]
        socket.socket.connect_ex = original_connect_ex  # type: ignore[method-assign]
        socket.create_connection = original_create_connection


__all__ = ["network_denied"]
