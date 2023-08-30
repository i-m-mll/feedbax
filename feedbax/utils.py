
from itertools import zip_longest, chain
import math

import jax.numpy as jnp


def exp_taylor(x: float, n: int):
    """First `n` terms of the Taylor series for `exp` at the origin."""
    return [(x ** i) / math.factorial(i) for i in range(n)]


def interleave_unequal(*args):
    """Interleave sequences of different lengths."""
    return (x for x in chain.from_iterable(zip_longest(*args)) if x is not None)
    

def normalize(*args, min=0, max=1):
    """Normalize each column of each input array to [min, max]"""
    return [(arr - jnp.min(arr, axis=0)) / (jnp.max(arr, axis=0) - jnp.min(arr, axis=0)) * (max - min) + min
            for arr in args]
    

_sincos_derivative_signs = jnp.array([(1, 1), (1, -1), (-1, -1), (-1, 1)]).reshape((4, 1, 1, 2))

def sincos_derivative_signs(i):
    """Return the signs of the i-th derivatives of sin and cos."""
    return _sincos_derivative_signs[-i]