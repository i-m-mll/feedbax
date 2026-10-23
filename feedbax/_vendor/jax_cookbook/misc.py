import copy
import logging
from collections.abc import (
    Hashable,
    Iterable,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
    Set,
)
from functools import wraps
from itertools import chain, zip_longest
from typing import Any, Iterable, Optional, Tuple, TypeVar, Union, cast

import jax
import jax.numpy as jnp
import jax.tree as jt
from jax import lax
from jaxtyping import Array

logger = logging.getLogger(__name__)


"""The signs of the i-th derivatives of cos and sin.

TODO: infinite cycle
"""
SINCOS_GRAD_SIGNS = jnp.array([(1, 1), (1, -1), (-1, -1), (-1, 1)])


T1 = TypeVar("T1")
T2 = TypeVar("T2")
TupleT = TypeVar("TupleT", bound=tuple)


class StrAlwaysLT(str):
    def __lt__(self, other):
        return True

    def __gt__(self, other):
        return False

    # def __repr__(self):
    #     return self.replace("'", "")


def interleave_unequal(*args):
    """Interleave sequences of different lengths."""
    return (x for x in chain.from_iterable(zip_longest(*args)) if x is not None)


def unzip2(xys: Iterable[Tuple[T1, T2]]) -> Tuple[Tuple[T1, ...], Tuple[T2, ...]]:
    """Unzip sequence of length-2 tuples into two tuples.

    Taken from `jax._src.util`.
    """
    # Note: we deliberately don't use zip(*xys) because it is lazily evaluated,
    # is too permissive about inputs, and does not guarantee a length-2 output.
    xs: MutableSequence[T1] = []
    ys: MutableSequence[T2] = []
    for x, y in xys:
        xs.append(x)
        ys.append(y)
    return tuple(xs), tuple(ys)


def get_unique_label(label: str, invalid_labels: Union[Sequence[str], Set[str]]) -> str:
    """Get a unique string from a base string, while avoiding certain strings.

    Simply appends consecutive integers to the string until a unique string is
    found.
    """
    i = 0
    label_ = label
    while label_ in invalid_labels:
        label_ = f"{label}_{i}"
        i += 1
    return label_


def unique_generator(
    seq: Sequence[T1], replace_duplicates: bool = False, replace_value: Any = None
) -> Iterable[Optional[T1]]:
    """Yields the first occurrence of sequence entries, in order.

    If `replace_duplicates` is `True`, replaces duplicates with `replace_value`.
    """
    seen = set()
    for item in seq:
        if id(item) not in seen:
            seen.add(id(item))
            yield item
        elif replace_duplicates:
            yield replace_value


def nested_dict_update(dict_, *args, make_copy: bool = True):
    """Source: https://stackoverflow.com/a/3233356/23918276"""
    if make_copy:
        dict_ = copy.deepcopy(dict_)
    for arg in args:
        for k, v in arg.items():
            if isinstance(v, Mapping):
                dict_[k] = nested_dict_update(
                    dict_.get(k, type(v)()),
                    v,
                    make_copy=make_copy,
                )
            else:
                dict_[k] = v
    return dict_


def crop_to_shortest(*, axis: int):
    """Decorator that equalises the length of all array arguments along *axis*.

    When the wrapped function is called, each positional argument that has a
    ``shape`` attribute is inspected; the minimum length along *axis* is
    computed, and every such argument is sliced to this length with
    ``jax.lax.slice_in_dim``.  Non-array arguments (scalars, objects, etc.) are
    passed through untouched.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Determine minimal size along the axis among array args.
            sizes = [a.shape[axis] for a in args if hasattr(a, "shape")]
            if not sizes:
                # No array arguments; nothing to crop.
                return func(*args, **kwargs)

            min_len = min(sizes)

            def _crop(a):
                if hasattr(a, "shape") and a.shape[axis] != min_len:
                    return jax.lax.slice_in_dim(a, 0, min_len, axis=axis)
                return a

            cropped_args = tuple(_crop(a) for a in args)
            return func(*cropped_args, **kwargs)

        return wrapper

    return decorator


def construct_tuple_like(cls: type[TupleT], elems: Iterable[Any]) -> TupleT:
    # Convert to list to avoid consuming iterators multiple times
    elems_list = list(elems)

    # NamedTuple/collections.namedtuple provide _make(iterable)
    make = getattr(cls, "_make", None)
    if callable(make):
        return cast(TupleT, make(elems_list))

    # Try tuple-style construction first (works for tuple and its subclasses)
    try:
        return cast(TupleT, cls(elems_list))
    except TypeError:
        pass

    # Try unpacked args (for custom tuple-like classes)
    try:
        return cast(TupleT, cls(*elems_list))
    except TypeError as e:
        raise TypeError(
            f"Cannot construct {cls.__name__} from elements; tried cls(elems) and cls(*elems)."
        ) from e


K = TypeVar("K", bound=Hashable)


def _clone_like_base(base: Mapping[K, Any]) -> MutableMapping[K, Any]:
    # Prefer preserving the concrete type *if* .copy() exists and returns something mutable.
    copy_m = getattr(base, "copy", None)
    if callable(copy_m):
        out = copy_m()
        if isinstance(out, MutableMapping):
            return out
    # Fallback: plain dict (always mutable)
    return dict(base)


#! TODO: Make this work for dict nodes in arbitrary PyTrees, not just pure-dict trees
def deep_merge(base: Mapping[K, Any], over: Mapping[K, Any]) -> MutableMapping[K, Any]:
    """Overlay `over` onto `base`, preserving base node types where possible.
    Purely structural: JAX PyTree–friendly (runs on host during trace)."""
    out = _clone_like_base(base)
    for k, v in over.items():
        bv = out.get(k)
        if isinstance(bv, Mapping) and isinstance(v, Mapping):
            out[k] = deep_merge(bv, v)
        else:
            out[k] = v
    return out


def split_by(x, sizes, axis=0):
    """Partition an array into given sizes along the specified axis."""
    if x.shape[axis] != sum(sizes):
        raise ValueError(f"Cannot split axis of size {x.shape[axis]} into sizes {sizes}")
    split_indices = jnp.cumsum(jnp.array(sizes))[:-1]
    return jnp.split(x, split_indices, axis=axis)


def _fname(f: object) -> str:
    return getattr(f, "__name__", f.__class__.__name__)


def moving_avg(x, K):  # x: [T]
    # simple causal K-window mean, shape -> [T - K + 1]
    c = jnp.cumsum(jnp.pad(x, (1, 0)))  # prefix sum with 0 at start
    wsum = c[K:] - c[:-K]
    return wsum / K


def softmin(values, tau, axis=-1, keepdims=False):
    m = jnp.min(values, axis=axis, keepdims=True)
    out = -tau * jnp.log(jnp.sum(jnp.exp(-(values - m) / tau), axis=axis, keepdims=True)) + m
    return out if keepdims else jnp.squeeze(out, axis=axis)


def mse(x, y):
    """Mean squared error."""
    return jt.map(
        lambda x, y: jnp.mean((x - y) ** 2),
        x,
        y,
    )


def nan_safe_mse(
    preds: Array,
    targets: Array,
) -> Array:
    """
    Calculates MSE safely for gradients when targets have NaN entries.

    Assumes that if `pred` has NaN entries, then `target` will also have NaNs in the same rows.

    Computes a mask of the NaN entries in `targets`, replaces NaNs with zeros,
    proceeds with MSE calculation, then masks the NaN entries out of the result
    prior to aggregation.
    """
    valid_mask = ~jnp.isnan(targets)
    targets_cleaned = jnp.nan_to_num(targets, nan=0.0)
    squared_errors = (preds - targets_cleaned) ** 2
    masked_squared_errors = jnp.where(valid_mask, squared_errors, 0.0)
    sum_of_squared_errors = jnp.sum(masked_squared_errors)
    num_valid_elements = jnp.sum(valid_mask)
    return sum_of_squared_errors / jnp.maximum(num_valid_elements, 1.0)


def window_take(
    arr: jnp.ndarray,
    idxs: jnp.ndarray,  # shape (N,), int32/64
    bounds: tuple[int, int],  # (lo, hi), hi exclusive; length = hi - lo (must be > 0)
    *,
    axis_p: int,  # slice-along axis
    axis_q: int,  # per-entry axis (len N)
    mode: str = "pad",  # "pad" or "clip"
    pad_value=0,
):
    """Return, for each index along axis_q, a length-(hi-lo) slice along axis_p
    starting at (idx + lo). Output has same ndim as arr, with axis_p length = hi-lo.
    """
    lo, hi = bounds
    L = int(hi - lo)
    if L <= 0:
        raise ValueError("hi - lo must be > 0")
    if axis_p == axis_q:
        raise ValueError("axis_p and axis_q must be different")

    ndim = arr.ndim
    p = axis_p % ndim
    q = axis_q % ndim
    if arr.shape[q] != idxs.shape[0]:
        raise ValueError("idxs length must match arr.shape[axis_q]")

    # Edge handling
    if mode == "pad":
        pad_left = max(0, -lo)
        pad_right = max(0, hi - 1)
        pad_width = [(0, 0)] * ndim
        pad_width[p] = (pad_left, pad_right)
        arrX = jnp.pad(arr, pad_width, constant_values=pad_value)
        starts = (idxs + lo + pad_left).astype(jnp.int32)
    elif mode == "clip":
        P = arr.shape[p]
        starts = jnp.clip(idxs + lo, 0, P - L).astype(jnp.int32)
        arrX = arr
    else:
        raise ValueError("mode must be 'pad' or 'clip'")

    # After vmapping over axis_q, that axis is removed inside the mapped fn.
    # Adjust the slice axis index accordingly.
    axis_p_in_mapped = p - (1 if p > q else 0)

    def slice_one(a_i, s_i):
        # a_i has arr with axis_q removed
        return lax.dynamic_slice_in_dim(a_i, s_i, L, axis=axis_p_in_mapped)

    # Map arr over its axis_q and idxs over its axis 0
    out = jax.vmap(slice_one, in_axes=(q, 0), out_axes=0)(arrX, starts)
    # Put the mapped axis back where axis_q originally was
    out = jnp.moveaxis(out, 0, q)
    return out
