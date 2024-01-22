"""Utility functions.

:copyright: Copyright 2023 by MLL <mll@mll.bio>.
:license: Apache 2.0, see LICENSE for details.
"""

from abc import abstractmethod
from collections import OrderedDict
from collections.abc import MutableMapping, Sequence
import dataclasses
import dis
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
from typing import Any, Callable, Concatenate, Dict, Iterable, List, Optional, Tuple, TypeVar, TypeVarTuple, Union

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

KT = TypeVar("KT")
VT = TypeVar("VT")
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
        return self        # Apparently you're supposed to only yield the key

    def __exit__(self, *args, **kwargs):
        self.time = perf_counter() - self.start_time
        self.times.append(self.time)
        self.readout = f'Time: {self.time:.3f} seconds'
        if self.printout:
            print(self.readout)
            
    start = __enter__
    stop = __exit__


class AbstractTransformedOrderedDict(MutableMapping[KT, VT]):
    """Base for `OrderedDict`s which transform keys when getting and setting items.
    
    It stores the original keys, and otherwise behaves (e.g. when iterating)
    as though they are the true keys.
    
    This is useful when we want to use a certain type of object as a key, but 
    it would not be hashed properly by `OrderedDict`, so we need to transform
    it into something else. In particular, by replacing `_key_transform` with 
    a code parsing function, a subclass of this class can take lambdas as keys.
    
    See `feedbax.task.InitSpecDict` for an example.
       
    Based on https://stackoverflow.com/a/3387975
    
    TODO: 
    - I'm not sure how the typing should work. I guess `VT` might need to correspond
      to a tuple of the original key and the value.
    """
    
    def __init__(self, *args, **kwargs):
        self.store = OrderedDict()
        self.update(OrderedDict(*args, **kwargs))
            
    def __getitem__(self, key):
        return self.store[self._key_transform(key)][1]

    def __setitem__(self, key, value):
        self.store[self._key_transform(key)] = (key, value)
    
    def __delitem__(self, key):
        del self.store[self._key_transform(key)]
        
    def __iter__(self):
        # Apparently you're supposed to only yield the key
        for key in self.store:
            yield self.store[key][0]

    def __len__(self):
        return len(self.store)
    
    def __repr__(self):
        # Make a pretty representation of the lambdas
        items_str = ', '.join(
            f"(lambda state: state{'.' if k else ''}{k}, {v})" 
            for k, (_, v) in self.store.items()
        )
        return f"{type(self).__name__}([{items_str}])"

    @abstractmethod
    def _key_transform(self, key):
        ...

    def tree_flatten(self):
        """The same flatten function used by JAX for `dict`"""
        return unzip2(sorted(self.items()))[::-1]

    @classmethod
    def tree_unflatten(cls, keys, values):
        return cls(zip(keys, values))


def get_where_str(where_func: Callable) -> str:
    """
    Given a function that accesses a tree of attributes of a single parameter, 
    return a string repesenting the attributes.
    
    This is useful for getting a unique string representation of a substate 
    of an `AbstractState` or `AbstractModel` object, as defined by a `where`
    function, so we can compare two such functions and see if they refer to 
    the same substate.
    
    TODO:
    - I'm not sure it's best practice to introspect on bytecode like this.
    """
    bytecode = dis.Bytecode(where_func)
    return '.'.join(instr.argrepr for instr in bytecode
                    if instr.opname == "LOAD_ATTR")


def angle_between_vectors(v2, v1):
    """Return the signed angle between two 2-vectors."""
    return jnp.arctan2(
        v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0], 
        v1[..., 0] * v2[..., 0] + v1[..., 1] * v2[..., 1],
    )   


def tree_index(tree: PyTree, index: int):
    """Returns the same PyTree, indexing all of its array leaves.
    """
    models_arrays, models_other = eqx.partition(tree, eqx.is_array)
    return eqx.combine(
        jax.tree_map(lambda x: x[index], models_arrays),
        models_other,
    )

def mask_diagonal(array):
    """Set the diagonal of (the last two dimensions of) `array` to zero."""
    mask = 1 - jnp.eye(array.shape[-1])
    return array * mask


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
    
    TODO:
    - Is this really the best way to do this?
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
    
    TODO:
    - Get rid of `tree_get_idx` and just give this a default `axis=0`.
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


def get_unique_label(label: str, invalid_labels: Sequence[str]) -> str:
    """Get a unique string from a base string, while avoiding certain strings."""
    i = 0
    label_ = label
    while label_ in invalid_labels:
        label_ = f"{label}_{i}" 
        i += 1
    return label_


def tree_map_unzip(
    f: Callable[..., Tuple[Any, ...]], 
    tree: PyTree, 
    *rest, 
    is_leaf: Optional[Callable[[Any], bool]] = None,
):
    """Map a function that returns a tuple over a PyTree, unzipping the results.
    
    For example, for a function `f(x) -> (y, z)`, we can do 
    `ys, zs = tree_map_unzip(f, xs)`, whereas with a normal `tree_map` we'd get
    a PyTree of tuples `(y, z)`. That is, we return a tuple of PyTrees instead 
    of a PyTree of tuples.
    """
    results = jax.tree_map(f, tree, *rest, is_leaf=is_leaf)
    results_flat, treedef = jax.tree_flatten(
        results, 
        is_leaf=lambda x: isinstance(x, tuple)
    )
    results_unzip = zip(*results_flat)
    return tuple(jax.tree_unflatten(treedef, x) for x in results_unzip)


def tree_call(tree, *args, **kwargs):
    """Calls a tree's callable leaves and returns a tree of their return values.
    
    Every callable receives identical `*args, **kwargs`.
    
    Non-callable leaves are passed through as-is.
    """
    callables, other_values = eqx.partition(
        tree, 
        lambda x: isinstance(x, Callable)
    )
    callables_values = jax.tree_map(
        lambda x: x(*args, **kwargs),
        callables,
    )
    return eqx.combine(callables_values, other_values)