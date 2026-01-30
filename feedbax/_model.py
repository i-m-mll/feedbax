"""Backward-compatibility helpers for model utilities."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps

from feedbax.graph import Component


# Typing alias for older code; new models should use `Component` directly.
AbstractModel = Component


def wrap_stateless_callable(callable: Callable):
    """Wrap a stateless callable to accept (input, state, *args, **kwargs)."""

    @wraps(callable)
    def wrapped(input, state, *args, **kwargs):
        return callable(input, *args, **kwargs)

    return wrapped


def wrap_stateless_keyless_callable(callable: Callable):
    """Wrap a stateless callable that also takes no key."""

    @wraps(callable)
    def wrapped(input, state, *args, key=None, **kwargs):
        return callable(input, *args, **kwargs)

    return wrapped
