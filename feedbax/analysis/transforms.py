"""Array transform decorators used by analysis code."""

from collections.abc import Callable, Sequence
import inspect
from functools import wraps
from typing import Any, Optional

import jax.numpy as jnp
import jax.tree as jt
from jaxtyping import Array, Shaped


def nan_bypass(
    func: Optional[Callable[..., Any]] = None,
    *,
    axis: int = 0,
    argnums: int | Sequence[int] | None = None,
    filler: float = 0.0,
):
    """Temporarily fill NaN slices before applying a function, then restore NaNs."""
    if func is not None and callable(func):
        return nan_bypass(axis=axis, argnums=argnums, filler=filler)(func)

    def decorator(f: Callable[..., Any]):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if argnums is None:
                argnums_tuple = tuple(range(len(args)))
            elif isinstance(argnums, int):
                argnums_tuple = (argnums,)
            else:
                argnums_tuple = tuple(argnums)

            def _row_has_nan(arr: jnp.ndarray) -> jnp.ndarray:
                red_axes = tuple(i for i in range(arr.ndim) if i != axis)
                return jnp.any(jnp.isnan(arr), axis=red_axes)

            nan_mask = jnp.zeros(args[argnums_tuple[0]].shape[axis], dtype=bool)
            for i in argnums_tuple:
                nan_mask = nan_mask | _row_has_nan(args[i])

            def _replace_rows(arr: jnp.ndarray) -> jnp.ndarray:
                bmask = jnp.expand_dims(
                    nan_mask,
                    axis=tuple(ax for ax in range(arr.ndim) if ax != axis),
                )
                return jnp.where(bmask, jnp.zeros_like(arr) + filler, arr)

            safe_args = list(args)
            for i in argnums_tuple:
                safe_args[i] = _replace_rows(args[i])

            out = f(*safe_args, **kwargs)

            def _restore_rows(arr: jnp.ndarray) -> jnp.ndarray:
                bmask = jnp.expand_dims(
                    nan_mask,
                    axis=tuple(ax for ax in range(arr.ndim) if ax != axis),
                )
                return jnp.where(bmask, jnp.full_like(arr, jnp.nan), arr)

            return jt.map(_restore_rows, out)

        return wrapper

    return decorator


def batch_reshape(
    func: Optional[Callable[[Shaped[Array, "batch *n"]], Shaped[Array, "batch *m"]]] = None,
    *,
    n_nonbatch: int | Sequence[int] = 1,
):
    """Decorate a function to collapse leading batch axes and restore them."""
    def decorator(f):
        n_params = len(inspect.signature(f).parameters)

        if isinstance(n_nonbatch, int):
            n_nonbatch_tuple = (n_nonbatch,) * n_params
        elif isinstance(n_nonbatch, Sequence):
            assert len(n_nonbatch) == n_params, (
                "if n_nonbatch is a sequence it must have the same length "
                "as the number of parameters of func"
            )
            n_nonbatch_tuple = tuple(n_nonbatch)

        @wraps(f)
        def wrapper(*args):
            batch_shapes = {arr.shape[:-n] for arr, n in zip(args, n_nonbatch_tuple)}
            assert len(batch_shapes) == 1, "all input arrays must have the same batch shape"
            batch_shape = batch_shapes.pop()
            collapsed_args = tuple(
                arr.reshape((-1, *arr.shape[-n:])) for arr, n in zip(args, n_nonbatch_tuple)
            )
            result = f(*collapsed_args)
            return jt.map(
                lambda arr: arr.reshape((*batch_shape, *arr.shape[1:])),
                result,
            )

        return wrapper

    if func is None:
        return decorator
    return decorator(func)
