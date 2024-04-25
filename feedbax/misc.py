"""Tools which did not belong any particular other place.

:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0, see LICENSE for details.
"""

from collections.abc import (
    Callable,
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
import inspect
from itertools import zip_longest, chain
import logging
import os
from pathlib import Path, PosixPath
from shutil import rmtree
import subprocess
from time import perf_counter
from typing import Any, Optional, Tuple, TypeAlias, TypeVar, Union

import equinox as eqx
from equinox import Module
from equinox._pretty_print import tree_pp, bracketed
import jax
import jax.numpy as jnp
import jax._src.pretty_printer as pp
import jax.tree_util as jtu
from jaxtyping import Float, Array, PyTree

from feedbax._progress import _tqdm_write


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


def git_commit_id(path: Optional[str | PosixPath] = None) -> str:
    """Get the ID of the currently checked-out commit in the repo at `path`.

    If no `path` is given, returns the commit ID for the repo containing this
    module.

    Derived from <https://stackoverflow.com/a/57683700>
    """
    if path is None:
        path = _dirname_of_this_module()

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
    xy = jax.tree_map(jnp.ravel, jnp.meshgrid(*bounds.T))
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


def _simple_module_pprint(name, *children, **kwargs):
    return bracketed(
        pp.text(name),
        kwargs['indent'],
        [tree_pp(child, **kwargs) for child in children],
        '(',
        ')'
    )


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


def where_func_to_paths(where, tree):
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

    paths_by_id = {leaf_id: path for path, leaf_id in jtu.tree_leaves_with_path(
        jtu.tree_map(
            lambda x: x if x in jax.tree_leaves(node_ids) else None,
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


def where_func_to_labels(where: Callable) -> PyTree[str]:
    """Also similar to `_get_where_str` and `where_func_to_paths`, but:
    
    - Avoids complicated logic of parsing bytecode, or traversing pytrees;
    - Works for `where` functions that return arbitrary PyTrees of node references;
    - Runs significantly (10+ times) faster than the other solutions.
    """

    try:
        return jax.tree_map(_get_where_str_constructor_label, where(_WhereStrConstructor()))
    except TypeError:
        raise TypeError("`where` must return a PyTree of node references")