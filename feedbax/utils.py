
import math

import jax.numpy as jnp


def exp_taylor(x: float, n: int):
    """First `n` terms of the Taylor series for `exp` at the origin.
    """
    return [(x ** i) / math.factorial(i) for i in range(n)]


_sincos_derivative_signs = jnp.array([(1, 1), (1, -1), (-1, -1), (-1, 1)]).reshape((4, 1, 1, 2))

def sincos_derivative_signs(i):
    return _sincos_derivative_signs[-i]