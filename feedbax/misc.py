"""Tools which did not belong any particular other place.

:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0, see LICENSE for details.
"""

from collections.abc import (
    Callable,
    Hashable,
    Iterable,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
    Set,
)
import copy
import difflib
import dis
import functools
import hashlib
import importlib
import inspect
import json
from itertools import zip_longest, chain
import logging
from operator import attrgetter
import os
from pathlib import Path, PosixPath
import pkgutil
import platform
import re
from shutil import rmtree
import signal
import subprocess
import textwrap
from time import perf_counter
import types
from copy import deepcopy
from dataclasses import dataclass, fields
from datetime import datetime
from functools import wraps
from types import GeneratorType, ModuleType
from typing import Any, Optional, Tuple, TypeVar, Union, get_origin

import equinox as eqx
from equinox import Module
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import jax._src.pretty_printer as pp
import jax.tree_util as jtu
import jax_cookbook.tree as jtree
import numpy as np
import pandas as pd
# Lazy imports to avoid circular dependency - imported locally in functions that need them
# from feedbax.intervene import AbstractIntervenor, CurlFieldParams, FixedFieldParams
from jax_cookbook import is_type
from jax_cookbook._func import wrap_to_accept_var_kwargs
from jaxtyping import Array, ArrayLike, Float, Int, PyTree, Shaped

from feedbax._progress import _tqdm_write
# Lazy import to avoid circular dependency
# from feedbax.config.yaml import get_yaml_loader


logger = logging.getLogger(__name__)


"""The signs of the i-th derivatives of cos and sin.

TODO: infinite cycle
"""
SINCOS_GRAD_SIGNS = jnp.array([(1, 1), (1, -1), (-1, -1), (-1, 1)])


T1 = TypeVar("T1")
T2 = TypeVar("T2")


class Timer:
    """Context manager for timing code blocks.

    Derived from https://stackoverflow.com/a/69156219
    """

    def __init__(self):
        self.times = []

    def __enter__(self, printout=False):
        self.start_time = perf_counter()
        self.printout = printout
        return self  # Apparently you're supposed to only yield the key

    def __exit__(self, *args, **kwargs):
        self.time = perf_counter() - self.start_time
        self.times.append(self.time)
        self.readout = f"Time: {self.time:.3f} seconds"
        if self.printout:
            print(self.readout)

    start = __enter__
    stop = __exit__


class TqdmLoggingHandler(logging.StreamHandler):
    """Avoid tqdm progress bar interruption by logger's output to console.

    Source: https://stackoverflow.com/a/67257516
    """

    # see logging.StreamHandler.eval method:
    # https://github.com/python/cpython/blob/d2e2534751fd675c4d5d3adc208bf4fc984da7bf/Lib/logging/__init__.py#L1082-L1091
    # and tqdm.write method:
    # https://github.com/tqdm/tqdm/blob/f86104a1f30c38e6f80bfd8fb16d5fcde1e7749f/tqdm/std.py#L614-L620

    def emit(self, record):
        try:
            msg = self.format(record)
            _tqdm_write(msg, end=self.terminator)
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)


class StrAlwaysLT(str):

    def __lt__(self, other):
        return True

    def __gt__(self, other):
        return False

    # def __repr__(self):
    #     return self.replace("'", "")


class BatchInfo(eqx.Module):
    size: int
    current: int
    total: int
    start: int = 0

    @property
    def progress(self) -> float:
        return self.current / self.total

    @property
    def run_progress(self) -> float:
        return (self.current - self.start) / (self.total - self.start)


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


def git_commit_id(
    path: Optional[str | PosixPath] = None,
    module: Optional[ModuleType] = None,
) -> str:
    """Get the ID of the currently checked-out commit in the repo at `path`.

    If no `path` or `module` is given, returns the commit ID for the repo
    containing this function definition.

    Derived from <https://stackoverflow.com/a/57683700>
    """
    if path is None:
        if module is None:
            path = _dirname_of_this_module()
        else:
            path = Path(module.__file__).absolute().parent

    commit_id = (
        subprocess.check_output(["git", "describe", "--always"], cwd=path)
        .strip()
        .decode()
    )

    return commit_id


def identity_func(x):
    """The identity function."""
    return x


def n_positional_args(func: Callable) -> int:
    """Get the number of positional arguments of a function."""
    sig = inspect.signature(func)
    return sum(
        1
        for param in sig.parameters.values()
        if param.kind == param.POSITIONAL_OR_KEYWORD
    )


def interleave_unequal(*args):
    """Interleave sequences of different lengths."""
    return (x for x in chain.from_iterable(zip_longest(*args)) if x is not None)


def corners_2d(bounds: Float[Array, "2 xy=2"]):
    """Generate the corners of a rectangle from its bounds."""
    xy = jt.map(jnp.ravel, jnp.meshgrid(*bounds.T))
    return jnp.vstack(xy)


def indent_str(s: str, indent: int = 4) -> str:
    """Pretty format a PyTree, but indent all lines with `indent` spaces."""
    indent_str = " " * indent
    return indent_str + s.replace("\n", f"\n{indent_str}")


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


def get_unique_label(label: str, invalid_labels: Sequence[str] | Set[str]) -> str:
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


def highlight_string_diff(obj1, obj2):
    """Given two objects, give a string that highlights the differences in
    their string representations.

    This can be useful for identifying slight differences in large PyTrees.

    Source: https://stackoverflow.com/a/76946768
    """
    str1 = repr(obj1)
    str2 = repr(obj2)

    matcher = difflib.SequenceMatcher(None, str1, str2)

    str2_new = ""
    i = 0
    for m in matcher.get_matching_blocks():
        if m.b > i:
            str2_new += str2[i : m.b]
        str2_new += f"\033[91m{str2[m.b:m.b + m.size]}\033[0m"
        i = m.b + m.size

    return str2_new.replace('\\n', '\n')


def print_trees_side_by_side(tree1, tree2, column_width=60, separator='|'):
    """Given two PyTrees, print their pretty representations side-by-side."""
    strs1 = eqx.tree_pformat(tree1).split('\n')
    strs2 = eqx.tree_pformat(tree2).split('\n')

    def wrap_text(text, width):
        return textwrap.wrap(text, width) or ['']

    wrapped1 = [wrap_text(s, column_width) for s in strs1]
    wrapped2 = [wrap_text(s, column_width) for s in strs2]

    for w1, w2 in zip_longest(wrapped1, wrapped2, fillvalue=['']):
        max_lines = max(len(w1), len(w2))

        for i in range(max_lines):
            line1 = w1[i] if i < len(w1) else ''
            line2 = w2[i] if i < len(w2) else ''
            print(f"{line1:<{column_width}} {separator} {line2:<{column_width}}")


def unique_generator(
    seq: Sequence[T1],
    replace_duplicates: bool = False,
    replace_value: Any = None
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


def is_module(element: Any) -> bool:
    """Return `True` if `element` is an Equinox module."""
    return isinstance(element, Module)


def is_none(x):
    return x is None


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


# def _simple_module_pprint(name, *children, **kwargs):
#     return bracketed(
#         pp.text(name),
#         kwargs['indent'],
#         [tree_pp(child, **kwargs) for child in children],
#         '(',
#         ')'
#     )


def _get_where_str(where_func: Callable) -> str:
    """
    Returns a string representation of the (nested) attributes accessed by a function.

    Only works for functions that take a single argument, and return the argument
    or a single (nested) attribute accessed from the argument.
    """
    bytecode = dis.Bytecode(where_func)
    return ".".join(instr.argrepr for instr in bytecode if instr.opname == "LOAD_ATTR")


class _NodeWrapper:
    def __init__(self, value):
        self.value = value


class NodePath:
    def __init__(self, path):
        self.path = path

    def __iter__(self):
        return iter(self.path)


def where_func_to_paths(where: Callable, tree: PyTree):
    """
    Similar to `_get_where_str`, but:

    - returns node paths, not strings;
    - works for `where` functions that return arbitrary PyTrees of nodes;
    - works for arbitrary node access (e.g. dict keys, sequence indices)
      and not just attribute access.

    Limitations:

    - requires a PyTree argument;
    - assumes the same object does not appear as multiple nodes in the tree;
    - if `where` specifies a node that is a subtree, it cannot also specify a node
      within that subtree.

    See [this issue](https://github.com/i-m-mll/feedbax/issues/14).
    """
    tree = eqx.tree_at(where, tree, replace_fn=lambda x: _NodeWrapper(x))
    id_tree = jtu.tree_map(id, tree, is_leaf=lambda x: isinstance(x, _NodeWrapper))
    node_ids = where(id_tree)

    paths_by_id = {node_id: path for path, node_id in jtu.tree_leaves_with_path(
        jtu.tree_map(
            lambda x: x if x in jt.leaves(node_ids) else None,
            id_tree,
        )
    )}

    paths = jtu.tree_map(lambda node_id: NodePath(paths_by_id[node_id]), node_ids)

    return paths


class _WhereStrConstructor:

    def __init__(self, label: str = ""):
        self.label = label

    def __getitem__(self, key: Any):
        if isinstance(key, str):
            key = f"'{key}'"
        elif isinstance(key, type):
            key = key.__name__
        return _WhereStrConstructor("".join([self.label, f"[{key}]"]))

    def __getattr__(self, name: str):
        sep = "." if self.label else ""
        return _WhereStrConstructor(sep.join([self.label, name]))


def _get_where_str_constructor_label(x: _WhereStrConstructor) -> str:
    return x.label


def where_func_to_attr_str_tree(where: Callable) -> PyTree[str]:
    """Also similar to `_get_where_str` and `where_func_to_paths`, but:

    - Avoids complicated logic of parsing bytecode, or traversing pytrees;
    - Works for `where` functions that return arbitrary PyTrees of node references;
    - Runs significantly (10+ times) faster than the other solutions.
    """

    try:
        return jt.map(_get_where_str_constructor_label, where(_WhereStrConstructor()))
    except TypeError:
        raise TypeError("`where` must return a PyTree of node references")


def attr_str_tree_to_where_func(tree: PyTree[str]) -> Callable:
    """Reverse transformation to `where_func_to_labels`.

    Takes a PyTree of strings describing attribute accesses, and returns a function
    that returns a PyTree of attributes.
    """
    getters = jt.map(lambda s: attrgetter(s), tree)

    def where_func(obj):
        return jt.map(lambda g: g(obj), getters)

    return where_func


def nan_bypass(
    func: Optional[Callable[..., Any]] = None,
    *,
    axis: int = 0,
    argnums: int | Sequence[int] | None = None,
    filler: float = 0.0,          
):
    """
    Decorator that temporarily fills slices along `axis` which contain *any*
    NaN in the positional arguments indexed by `argnums`, applies the wrapped
    function to the remaining data, and finally re-inserts NaNs.

    Args:
    axis: axis along which rows/segments are considered
    argnums: which positional arguments to check for NaNs.
        - int      → single argument
        - sequence → those indices
        - None     → all positional arguments
    filler: the value used to overwrite NaN rows before calling `func`
    """
    # allow decorator to be used with/without parentheses
    if func is not None and callable(func):
        return nan_bypass(axis=axis, argnums=argnums, filler=filler)(func)

    def decorator(f: Callable[..., Any]):
        @wraps(f)
        def wrapper(*args, **kwargs):
            # 1. Resolve which arguments we have to inspect
            if argnums is None:
                argnums_tuple = tuple(range(len(args)))
            elif isinstance(argnums, int):
                argnums_tuple = (argnums,)
            else:
                argnums_tuple = tuple(argnums)

            # 2. Build a boolean mask of slices that contain *any* NaN
            def _row_has_nan(arr: jnp.ndarray) -> jnp.ndarray:
                red_axes = tuple(i for i in range(arr.ndim) if i != axis)
                return jnp.any(jnp.isnan(arr), axis=red_axes)

            nan_mask = jnp.zeros(args[argnums_tuple[0]].shape[axis], dtype=bool)
            for i in argnums_tuple:
                nan_mask = nan_mask | _row_has_nan(args[i])

            # 3. Broadcast mask to each target arg & replace with fillers
            def _replace_rows(arr: jnp.ndarray) -> jnp.ndarray:
                # make mask broadcastable to arr
                bmask = jnp.expand_dims(
                    nan_mask,
                    axis=tuple(ax for ax in range(arr.ndim) if ax != axis),
                )
                return jnp.where(bmask, jnp.zeros_like(arr) + filler, arr)

            safe_args = list(args)
            for i in argnums_tuple:
                safe_args[i] = _replace_rows(args[i])

            # 4. Call the original function
            out = f(*safe_args, **kwargs)

            # 5. Re-insert NaNs in the output on masked rows
            def _restore_rows(arr: jnp.ndarray) -> jnp.ndarray:
                bmask = jnp.expand_dims(
                    nan_mask,
                    axis=tuple(ax for ax in range(arr.ndim) if ax != axis),
                )
                nan_arr = jnp.full_like(arr, jnp.nan)
                return jnp.where(bmask, nan_arr, arr)

            return jt.map(_restore_rows, out)

        return wrapper

    return decorator


def batch_reshape(
    func: Optional[Callable[[Shaped[Array, "batch *n"]], Shaped[Array, "batch *m"]]] = None,
    *,
    n_nonbatch: int | Sequence[int] = 1,
):
    """Decorate a function to collapse its input array to a single batch dimension, and uncollapse the result.

    !!! Example
        Decorate `sklearn.decomposition.PCA.transform` so that it works on arrays with multiple
        batch dimensions.

        ```python
        # Generate some 30-dimension data points with two batch dims, and do 2-dim PCA
        n = 30
        data = np.random.random((10, 20, n))
        pca = PCA(n_components=2).fit(data.reshape(-1, data.shape[-1]))

        # Generate some more data we want to project, with an arbitrary number of batches
        more_data = np.random.rand((5, 10, 15, n))
        more_data_pc = batch_reshape(pca.transform)(more_data)  # (5, 10, 15, 2)
        ```

    !!! Note
        I originally added the `n_nonbatch` parameter in order to vmap a function over different arrays
        with different numbers of batch dimensions. Here's a trivial example.

        ```python
        key = jr.PRNGKeyArray(0)
        n = 5

        def func(arr: Shaped[Array, 'n']):
            # Do something; here we'll just add a trailing singleton dimensions
            return arr[..., None]

        arrays = (
            jr.normal(key, 10, 20, n),
            jr.normal(key, 50, 5, 16, n),
            jr.normal(key, 2, n),
        )

        # results is the same as arrays but each array has final singleton
        result = tree_map(batch_reshape(jax.vmap(func)), arrays)
        ```

        However, it turns out you can do this more easily with `jnp.vectorize`.

    Arguments:
        func: A function whose input and output arrays have a single batch dimension.
        n_nonbatch: The number of final axes that should not be collapsed and reformed. May be specified
            separately for each parameter.
    """
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
            batch_shapes = set([arr.shape[:-n] for arr, n in zip(args, n_nonbatch_tuple)])
            assert len(batch_shapes) == 1, (
                "all input arrays must have the same batch shape"
            )
            batch_shape = batch_shapes.pop()

            # Reshape to collapse batch dimensions
            collapsed_args = tuple(
                arr.reshape((-1, *arr.shape[-n:])) for arr, n in zip(args, n_nonbatch_tuple)
            )

            result = f(*collapsed_args)

            # Reshape back to original batch structure
            return jt.map(
                lambda arr: arr.reshape((*batch_shape, *arr.shape[1:])),
                result,
            )

        return wrapper

    if func is None:
        # Called with parentheses: @batch_reshape() or @batch_reshape(n_nonbatch=2)
        return decorator
    else:
        # Called without parentheses: @batch_reshape
        return decorator(func)


def unkwargkey(f):
    """Converts a final `key` kwarg into an initial positional arg.

    This is useful because many Equinox modules take `key` as a kwarg, and Equinox
    transformations don't like kwargs -- but sometimes we want to transform over `key`
    anyway.
    """
    @wraps(f)
    def wrapper(key, *args):
        return f(*args, key=key)
    return wrapper


def batched_outer(
    x: Shaped[Array, "*batch n"],
    y: Shaped[Array, "*batch n"],
) -> Shaped[Array, "*batch n n"]:
    """Returns the outer product of the final dimension of an array."""
    return jnp.einsum('...i,...j->...ij', x, y)


def exponential_smoothing(
    x: Float[Array, "*"],
    alpha: float,
    init_window_size: int = 1,
    axis: int = -1,
):
    """
    Return the exponential moving average (EMA) of an array along the specified axis.

    Arguments:
        x: Input array
        alpha: Smoothing factor between 0 and 1.
        init_window_size: Initialize the EMA with the average of this many values
            from the beginning of the axis.
        axis: Axis along which to perform the EMA.

    Returns:
        Array with the same shape as x, containing the EMA along the specified axis.
    """

    # assert init_window_size > 0, "init_window_size must be greater than 0"

    alpha = jnp.clip(alpha, 0, 1)  # type: ignore

    # Take the average of the first `init_window_size` values as the initial EMA
    init_value = jnp.mean(jnp.take(x, jnp.arange(init_window_size), axis=axis), axis=axis)

    # Move the axis we're operating on to be the first dimension
    x_moved = jnp.moveaxis(x, axis, 0)

    def scan_fn(carry, x_t):
        ema = (1 - alpha) * carry + alpha * x_t
        return ema, ema

    _, ema = jax.lax.scan(scan_fn, init_value, x_moved)

    # Move the axis back to its original position
    return jnp.moveaxis(ema, 0, axis)


# ============================================================================
# Content merged from feedbax._experiments.misc
# ============================================================================


def delete_all_files_in_dir(dir_path: Path):
    """Delete all files in a directory."""
    if not dir_path.exists() or not dir_path.is_dir():
        raise ValueError(f"Directory {dir_path} does not exist or is not a directory.")

    for item in dir_path.iterdir():
        if item.is_file():
            item.unlink()


def dict_str(d, value_format=".2f"):
    """A string representation of a dict that is more filename-friendly than `str` or `repr`."""
    format_string = f"{{k}}-{{v:{value_format}}}"
    return "-".join(format_string.format(k=k, v=v) for k, v in d.items())


def get_datetime_str():
    return datetime.now().strftime("%Y%m%d-%Hh%M")


def get_gpu_memory(gpu_idx=0):
    """Returns the available memory (in MB) on a GPU. Depends on `nvidia-smi`.

    Source: https://stackoverflow.com/a/59571639
    """
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = subprocess.check_output(command.split()).decode("ascii").split("\n")[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values[gpu_idx]


def with_caller_logger(func):
    """
    Decorator that provides the caller's logger to the wrapped function.

    Wrapped functions should accept a `logger: logging.Logger` keyword argument.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # If logger is not provided in kwargs, get the caller's logger
        if "logger" not in kwargs:
            caller_module = None
            caller_frame = inspect.currentframe()
            if caller_frame is not None:
                caller_module = inspect.getmodule(caller_frame.f_back)
            if caller_module is not None:
                kwargs["logger"] = logging.getLogger(caller_module.__name__)
            else:
                kwargs["logger"] = logging.getLogger(func.__module__)

        # Call the original function with the resolved logger
        return func(*args, **kwargs)

    return wrapper


def discard(*args, **kwargs) -> None:
    """A no-op function that accepts any arguments and does nothing.

    This is useful when we're doing a tree map with side effects, and don't need the results to take up memory.
    """
    return None


@with_caller_logger
def get_name_of_callable(
    func: Callable,
    # return_lambda_id: bool = False,
    logger: logging.Logger = logger,
) -> str:
    """
    Returns the name of a callable object, handling different types appropriately.

    Args:
        func: The callable object whose name is to be retrieved.

    Returns:
        A string representing the callable's name or identifier.
    """
    func_name = getattr(func, "__name__", None)

    # Handle lambdas
    if func_name == "<lambda>":
        lambda_loc_str = location_for_log(func)
        logger.debug(f"Assigned name 'lambda' to lambda function at {lambda_loc_str}")
        return "lambda"

    # Handle partial functions
    elif isinstance(func, functools.partial):
        return get_name_of_callable(func.func, logger=logger)

    # Handle method objects (bound or unbound)
    elif inspect.ismethod(func):
        # For bound methods, include class name
        if hasattr(func, "__self__"):
            return f"{func.__self__.__class__.__name__}.{func.__name__}"
        return func.__name__

    # Handle callable class instances
    elif callable(func) and not isinstance(
        func, (types.FunctionType, types.BuiltinFunctionType, type)
    ):
        class_name = func.__class__.__name__
        logger.warning(
            f"Generating name for instance of callable class '{class_name}'. "
            f"Note that instance attributes/state are not captured by this name."
        )
        return class_name

    # Regular functions, built-in functions, and classes
    else:
        if func_name is not None:
            return func.__name__
        else:
            return repr(func)


def camel_to_snake(s: str):
    """Convert camel case to snake case."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()


def snake_to_camel(s: str):
    """Convert snake case to camel case."""
    return "".join(word.title() for word in s.split("_"))


def load_yaml(path: Path) -> dict:
    """Load a YAML file."""
    from feedbax.config.yaml import get_yaml_loader
    yaml = get_yaml_loader(typ="safe")
    with open(path, "r") as f:
        return yaml.load(f)


def load_from_json(path):
    with open(path, "r") as jsonf:
        return json.load(jsonf)


def write_to_json(tree, file_path):
    arrays, other = eqx.partition(tree, eqx.is_array)
    lists = jt.map(lambda arr: arr.tolist(), arrays)
    serializable = eqx.combine(other, lists)

    with open(file_path, "w") as jsonf:
        json.dump(serializable, jsonf, indent=4)


def get_field_amplitude(intervenor_params):
    from feedbax.intervene import CurlFieldParams, FixedFieldParams
    if isinstance(intervenor_params, FixedFieldParams):
        return jnp.linalg.norm(intervenor_params.field, axis=-1)
    elif isinstance(intervenor_params, CurlFieldParams):
        return jnp.abs(intervenor_params.amplitude)
    else:
        raise ValueError(f"Unknown intervenor parameters type: {type(intervenor_params)}")


def vector_with_gaussian_length(key, shape=()):
    key1, key2 = jr.split(key)

    angle = jr.uniform(key1, shape, minval=-jnp.pi, maxval=jnp.pi)
    length = jr.normal(key2, shape)

    vector = length * jnp.array([jnp.cos(angle), jnp.sin(angle)])
    return vector.T


#! TODO Separate version-getting logic from conditional logging logic
@with_caller_logger
def log_version_info(
    *args: ModuleType,
    git_modules: Optional[Sequence[ModuleType]] = None,
    python_version: bool = True,
    logger: logging.Logger = logger,
    level: int = logging.DEBUG,
) -> dict[str, str]:
    version_info: dict[str, str] = {}

    log_strs = []
    if python_version:
        python_ver = platform.python_version()
        version_info["python"] = python_ver
        log_strs.append(f"python version: {python_ver}")

    for package in args:
        version = package.__version__
        version_info[package.__name__] = version
        log_strs.append(f"{package.__name__} version: {version}")

    if git_modules:
        for module in git_modules:
            commit = git_commit_id(module=module)
            version_info[f"{module.__name__} commit"] = commit
            log_strs.append(f"{module.__name__} commit: {commit}")

    for s in log_strs:
        logger.log(level, s)

    return version_info


def round_to_list(xs: Array, n: int = 5):
    """Rounds floats to a certain number of decimals when casting an array to a list.

    This is useful when (e.g.) using `jnp.linspace` to get a sequence of numbers which
    will be used as keys of a dict, where we want to avoid small floating point variations
    being present in the keys.
    """
    return [round(x, n) for x in xs.tolist()]


def create_arr_df(arr, col_names=None):
    """Convert a numpy/JAX array into a dataframe of values, with additional columns
    giving the indices of the values in the array.

    If the array has complex dtype, split the real and imaginary components
    into separate columns.
    """
    if col_names is None:
        col_names = [f"dim_{i}" for i in range(len(arr.shape))]

    # Get all indices including the eigenvalue dimension
    indices = np.indices(arr.shape)

    if np.iscomplexobj(arr):
        data_cols = {"real": arr.real.flatten(), "imag": arr.imag.flatten()}
    else:
        data_cols = {"value": arr.flatten()}

    # Create the base dataframe
    df = pd.DataFrame(data_cols)

    # Add all dimension indices
    for i, idx_array in enumerate(indices):
        df[col_names[i]] = idx_array.flatten()

    return df


def squareform_pdist(xs: Float[Array, "points dims"], ord: int | str | None = 2):
    """Return the pairwise distance matrix between points in `x`.

    In the case of `ord=2`, this should be equivalent to:

        ```python
        from scipy.spatial.distance import pdist, squareform

        squareform(pdist(x, metric='euclidean'))
        ```

    However, note that the values for `ord` are those supported
    by `jax.numpy.linalg.norm`. This provides fewer metrics than those
    supported by `scipy.spatial.distance.pdist`.
    """
    dist = lambda x1, x2: jnp.linalg.norm(x1 - x2, ord=ord)
    row_dist = lambda x: jax.vmap(dist, in_axes=(None, 0))(x, xs)
    return jax.lax.map(row_dist, xs)


def take_model(*args, **kwargs):
    """Performs `jtree.take` on a feedbax model.

    It is currently necessary to use this in place of `jtree.take` when
    the model contains intervenors with arrays, since those arrays may
    not have the same batch (e.g. replicate) dimensions as the other
    model arrays.
    """
    from feedbax.intervene import AbstractIntervenor
    return jtree.filter_wrap(
        lambda x: not is_type(AbstractIntervenor)(x),
        is_leaf=is_type(AbstractIntervenor),
    )(jtree.take)(*args, **kwargs)


def get_dataclass_fields(
    obj: Any,
    exclude: tuple[str, ...] = (),
    include_internal: bool = False,
) -> dict[str, Any]:
    """Get the fields of a dataclass object as a dictionary."""
    return {
        field.name: getattr(obj, field.name)
        for field in fields(obj)
        if field.name not in exclude
        and (include_internal or not field.metadata.get("internal", False))
    }


def filename_join(strs, joinwith="__"):
    """Helper for formatting filenames from lists of strings."""
    return joinwith.join(s for s in strs if s)


def is_json_serializable(value):
    """Recursive helper function for isinstance-based checking"""
    json_types = (str, int, float, bool, type(None))

    if isinstance(value, json_types):
        return True
    elif isinstance(value, Mapping):
        return all(isinstance(k, str) and is_json_serializable(v) for k, v in value.items())
    elif isinstance(value, (list, tuple)) and not isinstance(value, GeneratorType):
        return all(is_json_serializable(item) for item in value)
    return False


def get_constant_input_fn(x, n_steps: int, n_trials: int):
    return lambda trial_spec, key: (jnp.full((n_trials, n_steps - 1), x, dtype=float))


def copy_delattr(obj: Any, *attr_names: str):
    """Return a deep copy of an object, with some attributes removed."""
    obj = deepcopy(obj)
    for attr_name in attr_names:
        delattr(obj, attr_name)
    return obj


def take_non_nan(arr, axis=1):
    # Create tuple of axes to reduce over (all axes except the specified one)
    reduce_axes = tuple(i for i in range(arr.ndim) if i != axis)
    has_nan = jnp.any(jnp.isnan(arr), axis=reduce_axes)
    valid_cols = jnp.where(~has_nan)[0]
    return jnp.take(arr, valid_cols, axis=axis)


def vectors_to_2d_angles(vectors):
    return jnp.arctan2(vectors[..., 1], vectors[..., 0])


def map_fn_over_tree(func, is_leaf: Optional[Callable] = None):
    """Partially applies `jt.map`, for use in functional expressions."""

    @functools.wraps(func)
    def map_fn(tree, *rest):
        return jt.map(func, tree, *rest, is_leaf=is_leaf)

    return map_fn


def normalize(arr, axis=-1):
    return arr / jnp.linalg.norm(arr, axis=axis, keepdims=True)


def ravel_except_last(arr):
    return jnp.reshape(arr, (-1, arr.shape[-1]))


def center_and_rescale(arr, axis=0):
    arr_centered = arr - jnp.nanmean(arr, axis=axis)
    arr_rescaled = arr_centered / jnp.nanmax(arr_centered, axis=axis)
    return arr_rescaled


def _expand_boundary_for_comparison(
    boundary_vals: jnp.ndarray,
    target_ndim: int,
    axis: int,
) -> jnp.ndarray:
    """Expands boundary_vals for broadcasting with target_array's axis_indices."""
    # Assumes boundary_vals.shape matches target_array.shape[:axis_of_comparison]
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
    """
    Slices target_array along 'axis' using [start, end) ranges, padding outside.
    slice_params.shape[:-1] should match target_array.shape[:axis].
    """
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

    final_mask = jnp.logical_and(*masks)

    return jnp.where(final_mask, array, pad_value)


def get_all_module_names(package_obj, exclude_private: bool = True):
    """Get the names of all modules in a package.

    Names include the full package path, e.g. `"some_library.subpackage.module_name"`,
    even if `package_obj` is `some_library.subpackage`.
    """
    names = []
    if not hasattr(package_obj, "__path__") or not hasattr(package_obj, "__name__"):
        return tuple()  # Not a valid package object to inspect

    # The prefix ensures names are fully qualified relative to the initial package
    prefix = package_obj.__name__ + "."

    for module_info in pkgutil.walk_packages(package_obj.__path__, prefix):
        is_private = module_info.name.startswith("_") or "._" in module_info.name
        if not module_info.ispkg and not (exclude_private and is_private):
            names.append(module_info.name)

    return tuple(names)


def exclude_unshared_keys_and_identical_values(list_of_dicts):
    """Filter dicts in a list to exclude unshared keys, and keys with identical values."""
    if not list_of_dicts:
        return []

    common_keys = set(list_of_dicts[0].keys())
    for d in list_of_dicts[1:]:
        common_keys.intersection_update(d.keys())

    keys_to_exclude = {
        key
        for key in common_keys
        if all(d[key] == list_of_dicts[0][key] for d in list_of_dicts[1:])
    }

    return [
        {k: v for k, v in original_dict.items() if k not in keys_to_exclude}
        for original_dict in list_of_dicts
    ]


def batch_index(arr, idxs):
    """
    Given a batched array of indices, take the elements of `arr` at those indices.

    If `arr` has shape `(*batch, x, ...)` and `idxs` has shape `(*batch)`, then this
    indexes axis `x` of `arr` at the scalar indices specified by `idxs`. This does not
    work for arbitrary slices over `x`, as the result would be ragged.
    """
    n_final_axes = len(arr.shape) - len(idxs.shape)
    final_axes = tuple(-i for i in range(1, n_final_axes + 1))
    return jnp.take_along_axis(arr, jnp.expand_dims(idxs, axis=final_axes), axis=final_axes[-1])


def get_md5_hexdigest(content):
    """Returns the MD5 hexdigest of an object."""
    return hashlib.md5(str(content).encode()).hexdigest()


class GracefulStopRequested(Exception):
    """Custom exception for graceful stopping."""

    pass


class GracefulInterruptHandler:
    """Context manager and decorator for graceful keyboard interrupt handling.

    Usage as context manager:
    ```python
    with GracefulInterruptHandler() as interrupt_handler:
        @interrupt_handler
        def sensitive_operation():
            # ... long running operation
            pass

        for item in items:
            sensitive_operation()
    ```

    The handler will:
    - First Ctrl-C during sensitive operation: Wait for completion, then stop
    - First Ctrl-C outside sensitive operation: Stop immediately
    - Second Ctrl-C anywhere: Abort immediately like normal
    """

    def __init__(
        self,
        sensitive_msg: Optional[str] = None,
        stop_msg: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Args:
            sensitive_msg: Message shown when interrupt occurs during sensitive operation
            stop_msg: Message shown when stopping gracefully after operation completes
            logger: Logger to use for messages (defaults to print if None)
        """
        self.stop_requested = False
        self.in_sensitive_operation = False
        self.original_handler = None

        self.sensitive_msg = (
            sensitive_msg or "Ctrl-C caught: will exit after current operation completes..."
        )
        self.stop_msg = stop_msg or "Operation completed, stopping as requested..."
        self.logger = logger

    def _log_message(self, message: str):
        """Log message using logger or print."""
        if self.logger:
            self.logger.info(message)
        else:
            print(f"\n{message}")

    def __enter__(self):
        self.original_handler = signal.signal(signal.SIGINT, self._signal_handler)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.signal(signal.SIGINT, self.original_handler)

    def _signal_handler(self, signum, frame):
        if self.stop_requested:
            # Second Ctrl-C: restore default and abort immediately
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            raise KeyboardInterrupt
        else:
            self.stop_requested = True
            if self.in_sensitive_operation:
                self._log_message(self.sensitive_msg)
            else:
                self._log_message("Ctrl-C caught: stopping...")
                raise KeyboardInterrupt

    def __call__(self, func):
        """Use as decorator."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if self.stop_requested:
                raise GracefulStopRequested()

            self.in_sensitive_operation = True
            try:
                result = func(*args, **kwargs)
                # Check again after completion
                if self.stop_requested:
                    self._log_message(self.stop_msg)
                    raise GracefulStopRequested()
                return result
            finally:
                self.in_sensitive_operation = False

        return wrapper


def location_inspect(fn) -> tuple[str, str, int]:
    """
    Returns a tuple ("<module>", "<filename>", <start_line>)
    using inspect for best‐effort source lookup.
    Falls back to code‐object attrs if necessary.
    """
    mod = fn.__module__
    try:
        # inspect.getsourcefile may return None in some REPLs,
        # so fall back to getfile
        srcfile = inspect.getsourcefile(fn) or inspect.getfile(fn)
        src_lines, start = inspect.getsourcelines(fn)
        return mod, srcfile, start
    except (OSError, TypeError):
        # fallback to code‐object
        return mod, fn.__code__.co_filename, fn.__code__.co_firstlineno


def path_delim(p: Path | str) -> str:
    return f"{PATH_DELIM}{str(p)}{PATH_DELIM}"


def location_for_log(fn) -> str:
    """
    Returns a string representation of the location of a function.
    """
    mod, srcfile, start = location_inspect(fn)
    return f"{mod} ({path_delim(srcfile)}:{start})"


def find_indices(arr, values: Array | Sequence[ArrayLike]):
    """Find the indices of `values` in `arr`."""

    def find_single_value(value):
        return jnp.where(arr == value, size=1)

    # Vectorize this function across all values
    return jax.vmap(find_single_value)(jnp.array(values))


def rms(x: Array, axis: int = -1) -> Array:
    """Returns the root mean square of `x` along `axis`."""
    return jnp.sqrt(jnp.mean(x**2, axis=axis))


def field_names(datacls) -> tuple[str, ...]:
    return tuple(datacls.__dataclass_fields__)


def get_origin_type(type_):
    """Get the origin type of a generic type, or the type itself if not generic."""
    origin = get_origin(type_)
    return origin if origin is not None else type_


PATH_DELIM = "`"


def unit_circle_points(n):
    """Generate N evenly spaced points on a unit circle."""
    angles = jnp.linspace(0, 2 * jnp.pi, n, endpoint=False)
    z = jnp.exp(1j * angles)
    return jnp.column_stack([z.real, z.imag])


def deep_merge(
    base: Mapping[str, Any], over: Mapping[str, Any], ignore_none: bool = True
) -> dict[str, Any]:
    out: dict[str, Any] = dict(base)
    for k, v in over.items():
        if v is None and ignore_none:
            continue
        bv = out.get(k)
        if isinstance(v, Mapping) and isinstance(bv, Mapping):
            out[k] = deep_merge(bv, v)
        else:
            out[k] = v
    return out


@dataclass
class _OptionalCallableFieldConverter[T]:
    """Converter for dataclass fields that may be callables, constant values, or `None`.

    By default, insists that callables are keyword-only, and wraps them to accept `**kwargs`
    if they do not already.
    """

    name: str
    kw_only: bool = True
    wrap_for_var_kwargs: bool = True

    def __call__(self, x: T | Callable[..., T] | None) -> Optional[Callable[..., T]]:
        if x is None:
            return None
        if isinstance(x, Callable):
            if self.kw_only:
                sig = inspect.signature(x)
                if any(
                    p.kind is not inspect.Parameter.KEYWORD_ONLY for p in sig.parameters.values()
                ):
                    raise ValueError(
                        f"Callables assigned to {self.name} cannot have positional parameters."
                    )
            if self.wrap_for_var_kwargs:
                return wrap_to_accept_var_kwargs(x)
            else:
                return x
        else:
            return lambda **kwargs: x
