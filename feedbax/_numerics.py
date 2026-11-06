"""Shared numerical primitives for JAX computations."""

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


def sqrt_with_zero_grad(value: ArrayLike) -> jax.Array:
    """Take the exact square root with derivative zero at zero."""
    is_zero = value == 0
    # Keep the square-root derivative finite before masking the result.
    safe_root = jnp.sqrt(jnp.where(is_zero, 1, value))
    return jnp.where(is_zero, 0, safe_root)
