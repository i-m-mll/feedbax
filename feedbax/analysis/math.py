"""Numerical helpers for analysis modules."""

from collections.abc import Sequence
from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int


def squareform_pdist(xs: Float[Array, "points dims"], ord: int | str | None = 2):
    """Return the pairwise distance matrix between points in `xs`."""
    dist = lambda x1, x2: jnp.linalg.norm(x1 - x2, ord=ord)
    row_dist = lambda x: jax.vmap(dist, in_axes=(None, 0))(x, xs)
    return jax.lax.map(row_dist, xs)


def ravel_except_last(arr):
    return jnp.reshape(arr, (-1, arr.shape[-1]))


def center_and_rescale(arr, axis=0):
    arr_centered = arr - jnp.nanmean(arr, axis=axis)
    return arr_centered / jnp.nanmax(arr_centered, axis=axis)


def _expand_boundary_for_comparison(
    boundary_vals: jnp.ndarray,
    target_ndim: int,
    axis: int,
) -> jnp.ndarray:
    expanded = jnp.expand_dims(boundary_vals, axis=axis)
    for i in range(axis + 1, target_ndim):
        expanded = jnp.expand_dims(expanded, axis=i)
    return expanded


def dynamic_slice_with_padding(
    array: Array,
    slice_end_idxs: Int[Array, "..."],
    axis: int,
    slice_start_idxs: Optional[Int[Array, "..."]] = None,
    pad_value: float = jnp.nan,
) -> Array:
    """Slice `array` along `axis` with per-prefix bounds and padding outside."""
    if axis < 0:
        axis = array.ndim + axis

    if slice_start_idxs is None:
        slice_start_idxs = jnp.zeros_like(slice_end_idxs)

    axis_indices = jnp.arange(array.shape[axis])
    idx_broadcast_shape = [1] * array.ndim
    idx_broadcast_shape[axis] = array.shape[axis]
    axis_indices_expanded = axis_indices.reshape(idx_broadcast_shape)

    masks = [
        op(
            axis_indices_expanded,
            _expand_boundary_for_comparison(slice_bound, array.ndim, axis),
        )
        for slice_bound, op in zip(
            [slice_start_idxs, slice_end_idxs],
            [jnp.greater_equal, jnp.less],
        )
    ]

    return jnp.where(jnp.logical_and(*masks), array, pad_value)
