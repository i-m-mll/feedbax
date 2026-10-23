"""Small support helpers for model construction."""

from collections.abc import Callable
from itertools import chain, zip_longest
import inspect


def identity_func(x):
    """Return ``x`` unchanged."""
    return x


def n_positional_args(func: Callable) -> int:
    """Return the number of positional-or-keyword arguments of a function."""
    sig = inspect.signature(func)
    return sum(
        1
        for param in sig.parameters.values()
        if param.kind == param.POSITIONAL_OR_KEYWORD
    )


def interleave_unequal(*args):
    """Interleave sequences of different lengths."""
    return (x for x in chain.from_iterable(zip_longest(*args)) if x is not None)
