"""For"""

# Modified by Feedbax: internal imports use the bundled package namespace.

from collections.abc import Callable, Sequence
from functools import wraps
from typing import Any, Iterable, Optional, Union

import equinox as eqx
import jax.tree as jt
from jaxtyping import PyTree

from feedbax._vendor.jax_cookbook._types import is_none


def vmap_multi(
    func: Callable,
    in_axes_sequence: Iterable[PyTree[Union[int, Optional[Callable[[Any], int]]]]],
    out_axes_sequence: Optional[
        Iterable[PyTree[Union[int, Optional[Callable[[Any], int]]]]]
    ] = None,
    vmap_func: Callable = eqx.filter_vmap,
):
    """Given a sequence of `in_axes`, construct a nested vmap of `func`.

    Arguments:
        func: Function to be transformed.
        in_axes_sequence: Sequence of `in_axes` specifications, as they would be passed to `jax.vmap`.
            For example, to successively map over axes 0, 1, and 2 of a single array argument,
            pass `in_axes_sequence=(0, 0, 0)`.
        out_axes_sequence: Optional sequence of `out_axes` specifications, one per vmap level.
        vmap_func: Transformation function to use.
    """
    func_v = func

    if out_axes_sequence is None:
        for in_axes in in_axes_sequence:
            func_v = vmap_func(func_v, in_axes=in_axes)
    else:
        in_axes_list = list(in_axes_sequence)
        out_axes_list = list(out_axes_sequence)
        if len(in_axes_list) != len(out_axes_list):
            raise ValueError(
                "in_axes_sequence and out_axes_sequence must have the same length"
            )
        for in_axes, out_axes in zip(in_axes_list, out_axes_list):
            func_v = vmap_func(func_v, in_axes=in_axes, out_axes=out_axes)

    return func_v


def unkwarg_key(func):
    """Converts a final `key` kwarg into an initial positional arg.

    This is useful because many Equinox modules take `key` as a kwarg, and transformations
    such as `equinox.filter_vmap` don't like kwargs -- but sometimes we want to transform over `key`
    anyway.
    """

    @wraps(func)
    def wrapper(key, *args):
        return func(*args, key=key)

    return wrapper


def natural_to_sequential_axes(axes: Sequence[Optional[int]]) -> list[Optional[int]]:
    """
    Given a flat sequence of natural‐index axes (ints or None),
    produce the corresponding sequential axes by subtracting,
    for each element, the number of *earlier* axes that were
    both non‐None and strictly less than it.

    Examples
    --------
    >>> natural_to_sequential([3, 0])
    [3, 0]

    >>> natural_to_sequential([4, 3, 2, 1])
    [4, 3, 2, 1]

    >>> natural_to_sequential([3, 5, 1])
    # at j=0: 3 (no preds <3)
    # at j=1: 5-1=4 (one pred 3<5)
    # at j=2: 1-0=1 (no pred<1)
    [3, 4, 1]
    """
    non_none_axes = [a for a in axes if a is not None]
    if len(non_none_axes) != len(set(non_none_axes)):
        raise ValueError(f"Natural indices must be unique; got: {axes!r}.")

    seq: list[Optional[int]] = []
    for j, a_j in enumerate(axes):
        if a_j is None:
            seq.append(None)
        else:
            # count earlier axes that have been peeled away
            peeled = sum(1 for b in axes[:j] if b is not None and b < a_j)
            seq.append(a_j - peeled)
    return seq


class MultiVmapAxes:
    """Mark that an argument should be vmapped over *multiple* levels.

    Parameters
    ----------
    *axes
        A sequence where **each element** is an `in_axes` spec *for one
        nested vmap level*.  Elements may themselves be arbitrary
        PyTrees (dict/list/tuple) of ints/None.
    natural_indices : bool, default ``False``
        If ``True`` the elements of ``axes`` must be *flat* integers or
        ``None``.  They are interpreted in **natural** coordinates – i.e.
        axis `a_i` of the *original* array – and automatically shifted to
        sequential indices via `a_i - i`.
    """

    axes: tuple[Any, ...]
    natural_indices: bool

    def __init__(self, *axes: Any, natural_indices: bool = False):
        if len(axes) == 1 and isinstance(axes[0], (list, tuple)):
            axes = tuple(axes[0])  # allow passing a list/tuple directly
        if natural_indices:
            # All axes must be scalar int/None
            for ax in axes:
                if not (ax is None or isinstance(ax, int)):
                    raise TypeError(
                        "Natural‑index MultiVmapAxes accepts only int/None elements; "
                        f"got element {ax!r}."
                    )
        if len(axes) == 0:
            raise ValueError("MultiVmapAxes must contain at least one axis spec")
        self.axes = tuple(axes)
        self.natural_indices = bool(natural_indices)

    def sequential_axes(self) -> list[Any]:
        """Return a list ready to be slotted per‑level into `in_axes`."""
        if not self.natural_indices:
            return list(self.axes)
        return natural_to_sequential_axes(self.axes)

    # Treat the *wrapper itself* as a PyTree **leaf** so that the recursion
    # sees it as atomic and defers to `sequential_axes` later.
    def tree_flatten(self):
        return (), None  # no children – it's a leaf

    @classmethod
    def tree_unflatten(cls, _: Any, __: Any):  # pragma: no cover – unused
        raise TypeError("MultiVmapAxes should never be unflattened by JAX")

    def __repr__(self) -> str:  # pragma: no cover
        mode = "natural" if self.natural_indices else "sequential"
        return f"MultiVmapAxes({self.axes}, mode={mode})"

    def __iter__(self):  # pragma: no cover – discourage but allow unpacking
        return iter(self.axes)


AxisSpec = Union[int, None, MultiVmapAxes, PyTree[int]]


def expand_axes_spec(spec: AxisSpec) -> list[Any]:
    """
    Expand a JAX-style in_axes spec (which may contain MultiVmapAxes)
    into a list of per-level in_axes PyTrees.
    """
    # 1) Flatten the user spec, capturing the treedef
    leaves, treedef = jt.flatten(spec, is_leaf=is_none)

    # 2) Expand each leaf to its own schedule list
    leaf_schedules: list[list[Optional[int]]] = []
    for leaf in leaves:
        if isinstance(leaf, MultiVmapAxes):
            sched = leaf.sequential_axes()
        else:
            # single‐level: run at level 0 only
            sched = [leaf]
        leaf_schedules.append(sched)

    # 3) Pad all schedules to the same number of levels
    max_levels = max(len(s) for s in leaf_schedules)
    for i, sched in enumerate(leaf_schedules):
        if len(sched) < max_levels:
            leaf_schedules[i] = sched + [None] * (max_levels - len(sched))

    # 4) Reconstruct each level's in_axes PyTree
    per_level_specs: list[Any] = []
    for lvl in range(max_levels):
        lvl_leaves = [sched[lvl] for sched in leaf_schedules]
        per_level_specs.append(treedef.unflatten(lvl_leaves))

    return per_level_specs
