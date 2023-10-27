"""Utility functions.

:copyright: Copyright 2023 by MLL <mll@mll.bio>.
:license: Apache 2.0, see LICENSE for details.
"""

import dataclasses
from itertools import zip_longest, chain
from functools import partial
import logging 
import math
import os
from pathlib import Path, PosixPath
from shutil import rmtree
import subprocess
from time import perf_counter
from typing import Callable, Concatenate, Dict, Optional, Union, ParamSpec

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Float, Array, PyTree


logger = logging.getLogger(__name__)


"""The signs of the i-th derivatives of cos and sin.

TODO: infinite cycle
"""
SINCOS_GRAD_SIGNS = jnp.array([(1, 1), (1, -1), (-1, -1), (-1, 1)])


P = ParamSpec('P')


class catchtime:
    """Context manager for timing code blocks.
    
    From https://stackoverflow.com/a/69156219
    """
    def __enter__(self, printout=False):
        self.start = perf_counter()
        self.printout = printout
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.start
        self.readout = f'Time: {self.time:.3f} seconds'
        if self.printout:
            print(self.readout)


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
            
            
def dirname_of_this_module():
    return os.path.dirname(os.path.abspath(__file__))
    

def exp_taylor(x: float, n: int):
    """First `n` terms of the Taylor series for `exp` at the origin."""
    return [(x ** i) / math.factorial(i) for i in range(n)]


def get_model_ensemble(
    get_model: Callable[Concatenate[jrandom.PRNGKeyArray, P], eqx.Module], 
    n_replicates: int, 
    *, 
    key: jrandom.PRNGKeyArray, 
    **kwargs,  # can't type this with P 
) -> eqx.Module:
    """Helper to vmap model generation over a set of PRNG keys."""
    keys = jrandom.split(key, n_replicates)
    get_model_ = partial(get_model, **kwargs)
    return eqx.filter_vmap(get_model_)(keys)


def git_commit_id(path: Optional[str | PosixPath] = None) -> str:
    """Get the ID of the currently checked-out commit in the repo at `path`.

    If no `path` is given, returns the commit ID for the repo containing this
    module.

    Based on <https://stackoverflow.com/a/57683700>
    """
    if path is None:
        path = dirname_of_this_module()

    commit_id = subprocess.check_output(["git", "describe", "--always"],
                                        cwd=path).strip().decode()

    return commit_id


def interleave_unequal(*args):
    """Interleave sequences of different lengths."""
    return (x for x in chain.from_iterable(zip_longest(*args)) 
            if x is not None)
    

def internal_grid_points(bounds, n=2):
    """Generate an even grid of points inside the given bounds.
    
    e.g. if bounds=((0, 9), (0, 9)) and n=2 the return value will be
    Array([[3., 3.], [6., 3.], [3., 6.], [6., 6.]]).
    """
    ticks = jax.vmap(lambda b: jnp.linspace(*b, n + 2)[1:-1])(bounds)
    points = jnp.vstack(jax.tree_map(jnp.ravel, jnp.meshgrid(*ticks))).T
    return points


def filter_spec_leaves(tree, leaf_func):
    """Get a filter spec for tree leaves matching `leaf_func`.
    
    `leaf_func` should take `tree` and return leaves from `tree` to filter `True`.
    """
    filter_spec = jax.tree_util.tree_map(lambda _: False, tree)
    filter_spec = eqx.tree_at(
        leaf_func, filter_spec, replace_fn=lambda x: True 
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


def corners_2d(bounds: Float[Array, "xy=2 2"]):    
    """Generate the corners of a rectangle from its bounds."""
    xy = jax.tree_map(jnp.ravel, jnp.meshgrid(*bounds))
    return jnp.vstack(xy)


def twolink_workspace_test(workspace: Float[Array, "xy=2 bounds=2"], twolink):
    """Tests whether a rectangular workspace is reachable by the two-link arm."""
    r = sum(twolink.l)
    lengths = jnp.sum(corners_2d(workspace) ** 2, axis=0) ** 0.5
    if jnp.any(lengths > r):
        return False 
    return True


def padded_bounds(x, p=0.2):
    """Return the lower and upper bounds of `x` with `p` percent padding."""
    bounds = jnp.array([jnp.min(x), jnp.max(x)])
    padding = (p * jnp.diff(bounds)).item()
    return bounds + jnp.array([-padding, padding])
    
# 
# jax.debug.print(''.join([f"{s.shape}\t{p}\n" 
#                             for p, s in jax.tree_util.tree_leaves_with_path(state)]))


