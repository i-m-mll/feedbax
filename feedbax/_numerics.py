"""Shared numerical primitives for JAX computations."""

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


def broadcast_vector(vector: jax.Array, ndim: int, *, axis: int) -> jax.Array:
    """Place a vector on one axis with singleton dimensions on the others."""
    shape = [1] * ndim
    shape[axis] = vector.shape[0]
    return vector.reshape(shape)


def sqrt_with_zero_grad(value: ArrayLike) -> jax.Array:
    """Take the exact square root with derivative zero at zero."""
    is_zero = value == 0
    # Keep the square-root derivative finite before masking the result.
    safe_root = jnp.sqrt(jnp.where(is_zero, 1, value))
    return jnp.where(is_zero, 0, safe_root)
