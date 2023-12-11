"""Utility functions.

:copyright: Copyright 2023 by MLL <mll@mll.bio>.
:license: Apache 2.0, see LICENSE for details.
"""

import dataclasses
import inspect
from itertools import zip_longest, chain
from functools import partial
import logging 
import math
import os
from pathlib import Path, PosixPath
from shutil import rmtree
import subprocess
from time import perf_counter
from typing import Callable, Concatenate, Dict, Iterable, List, Optional, Tuple, TypeVar, TypeVarTuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Float, Array, PyTree


logger = logging.getLogger(__name__)


"""The signs of the i-th derivatives of cos and sin.

TODO: infinite cycle
"""
SINCOS_GRAD_SIGNS = jnp.array([(1, 1), (1, -1), (-1, -1), (-1, 1)])


T1 = TypeVar("T1")
T2 = TypeVar("T2")
Ts = TypeVarTuple("Ts")


class Timer:
    """Context manager for timing code blocks.
    
    From https://stackoverflow.com/a/69156219
    """
    def __init__(self):
        self.times = []
    
    def __enter__(self, printout=False):
        self.start_time = perf_counter()
        self.printout = printout
        return self

    def __exit__(self, *args, **kwargs):
        self.time = perf_counter() - self.start_time
        self.times.append(self.time)
        self.readout = f'Time: {self.time:.3f} seconds'
        if self.printout:
            print(self.readout)
            
    start = __enter__
    stop = __exit__


def angle_between_vectors(v1, v2):
    """Return the signed angle between two 2-vectors."""
    return jnp.arctan2(
        v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0], 
        v1[..., 0] * v2[..., 0] + v1[..., 1] * v2[..., 1],
    )   


def vector_angle(v):
    """Return the angle of a 2-vector.
    
    This is a hardcoded special case of `angle_between_vectors`, 
    where `v2=(1, 0)`.
    """
    return jnp.arctan2(v[..., 1], v[..., 0])


def datacls_flatten(datacls):
    """Flatten function for dataclasses as PyTrees."""
    field_names, field_values = zip(*[
        (field_.name, getattr(datacls, field_.name))
        for field_ in dataclasses.fields(datacls)
    ])
                        
    children, aux = tuple(field_values), tuple(field_names)
    return children, aux


def datacls_unflatten(cls, aux, children):
    """Unflatten function for dataclasses as PyTrees."""
    field_names, field_values = aux, children
    datacls = object.__new__(cls)
    for name, value in zip(field_names, field_values):
        object.__setattr__(datacls, name, value)
    return datacls 


def delete_contents(path: Union[str, Path]):
    """Delete all subdirectories and files of `path`."""
    for p in Path(path).iterdir():
        if p.is_dir():
            rmtree(p)
        elif p.is_file():
            p.unlink()
            
            
def _dirname_of_this_module():
    """Return the directory containing this module."""
    return os.path.dirname(os.path.abspath(__file__))
    

def exp_taylor(x: float, n: int):
    """First `n` terms of the Taylor series for `exp` at the origin."""
    return [(x ** i) / math.factorial(i) for i in range(n)]


def get_model_ensemble(
    get_model: Callable[[jax.Array, *Ts], eqx.Module], 
    n_replicates: int, 
    *args: *Ts, 
    key: jax.Array
) -> eqx.Module:
    """Helper to vmap model generation over a set of PRNG keys.
    
    TODO: 
    - Rename. This works for stuff other than `get_model`. It's basically
      a helper to split key, then vmap function.
    """
    keys = jr.split(key, n_replicates)
    get_model_ = partial(get_model, *args)
    return eqx.filter_vmap(get_model_)(keys)


def git_commit_id(path: Optional[str | PosixPath] = None) -> str:
    """Get the ID of the currently checked-out commit in the repo at `path`.

    If no `path` is given, returns the commit ID for the repo containing this
    module.

    Based on <https://stackoverflow.com/a/57683700>
    """
    if path is None:
        path = _dirname_of_this_module()

    commit_id = subprocess.check_output(["git", "describe", "--always"],
                                        cwd=path).strip().decode()

    return commit_id


def n_positional_args(func: Callable) -> int:
    """Get the number of positional arguments of a function."""
    sig = inspect.signature(func)
    return sum(1 for param in sig.parameters.values() 
               if param.kind == param.POSITIONAL_OR_KEYWORD)


def interleave_unequal(*args):
    """Interleave sequences of different lengths."""
    return (x for x in chain.from_iterable(zip_longest(*args)) 
            if x is not None)
    

def internal_grid_points(
    bounds: Float[Array, "bounds=2 ndim=2"], 
    n: int = 2
):
    """Generate an even grid of points inside the given bounds.
    
    e.g. if bounds=((0, 0), (9, 9)) and n=2 the return value will be
    Array([[3., 3.], [6., 3.], [3., 6.], [6., 6.]]).
    """
    ticks = jax.vmap(lambda b: jnp.linspace(*b, n + 2)[1:-1])(bounds.T)
    points = jnp.vstack(jax.tree_map(jnp.ravel, jnp.meshgrid(*ticks))).T
    return points


def filter_spec_leaves(tree, leaf_func):
    """Get a filter spec for tree leaves matching `leaf_func`.
    
    `leaf_func` should take `tree` and return leaves from `tree` to filter `True`.
    """
    filter_spec = jax.tree_util.tree_map(lambda _: False, tree)
    filter_spec = eqx.tree_at(
        leaf_func, filter_spec, replace_fn=lambda x: True,
    )
    return filter_spec


def normalize(
    tree: PyTree,
    min: float = 0, 
    max: float = 1, 
    axis: int = 0,
):
    """Normalize each input array to [min, max] along the given axis.
    
    Defaults to normalizing columns
    """
    def arr_norm(arr):
        arr_min = jnp.min(arr, axis=axis)
        arr_max = jnp.max(arr, axis=axis)
        return min + (max - min) * (arr - arr_min) / (arr_max - arr_min)
    
    return jax.tree_map(arr_norm, tree)


def device_put_all(tree: PyTree, device=jax.devices()[0]):
    """Put all array leaves of `tree` on the default device.
    
    TODO: I'm not sure this is actually useful for anything.
    """
    arrays, other = eqx.partition(tree, eqx.is_array)
    committed = jax.tree_map(
        lambda x: jax.device_put(x, device), 
        arrays,
    )
    return eqx.combine(committed, other)


@jax.named_scope("fbx.tree_get_idx")
def tree_get_idx(tree: PyTree, idx: int):
    """Retrieve the `idx`-th element of each array leaf of `tree`.
    
    Any non-array leaves are returned unchanged.
    """
    arrays, other = eqx.partition(tree, eqx.is_array)
    values = jax.tree_map(lambda xs: xs[idx], arrays)
    return eqx.combine(values, other)


@jax.named_scope("fbx.tree_get_idx")
def tree_take(tree: PyTree, idx: int, axis: int):
    """Take elements from the specified axis of each array leaf of `tree`.
    
    Any non-array leaves are returned unchanged.
    """
    arrays, other = eqx.partition(tree, eqx.is_array)
    values = jax.tree_map(lambda xs: jnp.take(xs, idx, axis=axis), arrays)
    return eqx.combine(values, other)


@jax.named_scope("fbx.tree_set_idx")
def tree_set_idx(
    tree: PyTree, 
    vals, 
    idx: int
):
    """Update the `idx`-th element of each array leaf of `tree`.
    
    `vals` should be a pytree with the same structure as `tree`,
    except that each leaf should be missing the first dimension of 
    the corresponding leaf in `tree`.
    
    Non-array leaves are simply replaced by their matching leaves in `vals`.
    """
    arrays = eqx.filter(tree, eqx.is_array)
    vals_update, other_update = eqx.partition(
        vals, jax.tree_map(lambda x: x is not None, arrays)
    )
    arrays_update = jax.tree_map(
        lambda xs, x: xs.at[idx].set(x), arrays, vals_update
    )
    return eqx.combine(arrays_update, other_update)


# def tree_set_idxs(tree, vals, idxs):
#     """Update the `idx`-th element of each array leaf of `tree`.
    
#     Similar to `tree_set_idx` but takes a PyTree of indices.
#     """
#     arrays = eqx.filter(tree, eqx.is_array)
#     vals_update, other_update = eqx.partition(
#         vals, jax.tree_map(lambda x: x is not None, arrays)
#     )
#     arrays_update = jax.tree_map(
#         lambda xs, x, idx: xs.at[idx].set(x), arrays, vals_update, idxs
#     )
#     return eqx.combine(arrays_update, other_update)

def random_split_like_tree(rng_key, target=None, treedef=None):
    """Generate a split of PRNG keys with a target PyTree structure.
    
    See https://github.com/google/jax/discussions/9508#discussioncomment-2144076
    """
    if treedef is None:
        treedef = jax.tree_structure(target)
    keys = jax.random.split(rng_key, treedef.num_leaves)
    return jax.tree_unflatten(treedef, keys)


def tree_stack(trees):
    """Stack the leaves of each tree in `trees`.
    
    All trees should have the same structure.
    
    See https://gist.github.com/willwhitney/dd89cac6a5b771ccff18b06b33372c75?permalink_comment_id=4634557#gistcomment-4634557
    """
    return jax.tree_util.tree_map(lambda *v: jnp.stack(v), *trees)


def tree_sum_squares(tree):
    """Sum the sums of squares of the leaves of a PyTree.
    
    Useful for (say) doing weight decay on model parameters.
    """
    return jax.tree_util.tree_reduce(
        lambda x, y: x + y, 
        jax.tree_map(lambda x: jnp.sum(x ** 2), tree)
    )


def tree_sum_n_features(tree):
    """Sum the sizes of the last dimensions of all leaves."""
    return jax.tree_util.tree_reduce(
        lambda x, y: x + y, 
        jax.tree_map(lambda x: x.shape[-1], tree)
    )


def corners_2d(bounds: Float[Array, "2 xy=2"]):    
    """Generate the corners of a rectangle from its bounds."""
    xy = jax.tree_map(jnp.ravel, jnp.meshgrid(*bounds.T))
    return jnp.vstack(xy)


def padded_bounds(x, p=0.2):
    """Return the lower and upper bounds of `x` with `p` percent padding."""
    bounds = jnp.array([jnp.min(x), jnp.max(x)])
    padding = (p * jnp.diff(bounds)).item()
    return bounds + jnp.array([-padding, padding])
    
# 
# jax.debug.print(''.join([f"{s.shape}\t{p}\n" 
#                             for p, s in jax.tree_util.tree_leaves_with_path(state)]))


def tree_pformat_indent(tree: PyTree, indent: int = 4, **kwargs) -> str:
    """Pretty format a PyTree, but indent all lines with `indent` spaces."""
    indent_str = " " * indent
    pformat_str = eqx.tree_pformat(tree, **kwargs)
    return indent_str + pformat_str.replace("\n", "\n{indent_str}")


def unzip2(
    xys: Iterable[Tuple[T1, T2]]
) -> Tuple[Tuple[T1, ...], Tuple[T2, ...]]:
    """Unzip sequence of length-2 tuples into two tuples.
    
    Taken from `jax._src.util`.
    """
    # Note: we deliberately don't use zip(*xys) because it is lazily evaluated,
    # is too permissive about inputs, and does not guarantee a length-2 output.
    xs: List[T1] = []
    ys: List[T2] = []
    for x, y in xys:
        xs.append(x)
        ys.append(y)
    return tuple(xs), tuple(ys)