"""Numerical helpers for analysis modules."""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def squareform_pdist(xs: Float[Array, "points dims"], ord: int | str | None = 2):
    """Return the pairwise distance matrix between points in `xs`."""
    dist = lambda x1, x2: jnp.linalg.norm(x1 - x2, ord=ord)
    row_dist = lambda x: jax.vmap(dist, in_axes=(None, 0))(x, xs)
    return jax.lax.map(row_dist, xs)
