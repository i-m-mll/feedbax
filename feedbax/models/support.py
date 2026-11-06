"""Small support helpers for model construction."""

from itertools import chain, zip_longest


def identity_func(x):
    """Return ``x`` unchanged."""
    return x


def interleave_unequal(*args):
    """Interleave sequences of different lengths."""
    return (x for x in chain.from_iterable(zip_longest(*args)) if x is not None)
