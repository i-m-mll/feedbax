"""Small support helpers for model construction."""

from itertools import chain, zip_longest

from feedbax._vendor.jax_cookbook import identity as identity_func

__all__ = ["identity_func", "interleave_unequal"]


def interleave_unequal(*args):
    """Interleave sequences of different lengths."""
    return (x for x in chain.from_iterable(zip_longest(*args)) if x is not None)
