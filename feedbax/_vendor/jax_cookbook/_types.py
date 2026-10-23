from typing import Any

from equinox import Module


def is_module(element: Any) -> bool:
    """Return `True` if `element` is an Equinox module."""
    return isinstance(element, Module)


def is_none(x: Any) -> bool:
    """Return `True` if `x` is `None`."""
    return x is None
