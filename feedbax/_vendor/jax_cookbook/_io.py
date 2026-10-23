"""Facilities for saving and loading of experimental setups.

TODO:
- Could provide a simple interface to show the most recently saved files in
  a directory

:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

import json
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

import equinox as eqx
import jax.random as jr
import jax.tree as jt
import numpy as np
from jaxtyping import Array, PyTree

from ._func import is_type
from .misc import nested_dict_update
from .tree import filter_wrap

logger = logging.getLogger(__name__)


def save_old(
    path: str | Path,
    tree: PyTree[eqx.Module],
    hyperparameters: Optional[dict] = None,
    dump_func: Callable = lambda hps: json.dumps(hps, sort_keys=True),
) -> None:
    """Save a PyTree to disk along with hyperparameters used to generate it.

    Assumes none of the hyperparameters are JAX arrays, as these are not
    JSON serialisable.

    Based on the Equinox serialisation [example](https://docs.kidger.site/equinox/examples/serialisation/).

    Arguments:
        path: The path of the file to be saved. Note that the file at this path
            will be overwritten if it exists.
        tree: The PyTree to save. Its structure should match the return
            type of a function `setup_func`, which will be passed to `load`.
        hyperparameters: A dictionary of arguments for
            `setup_func` that were used to generate the PyTree, and upon
            loading, will be used to regenerate an appropriate skeleton to
            populate with the saved values from `tree`.
        sort_keys: Whether to sort the hyperparameters by key before serialising.
            This ensures that if we load and reserialise the file, it won't hash
            differently due to key order changes.
    """
    with open(path, "wb") as f:
        hyperparameter_str = dump_func(hyperparameters)
        f.write((hyperparameter_str + "\n").encode())
        eqx.tree_serialise_leaves(f, tree)

    filesize = os.path.getsize(path)

    logger.info(f"Wrote PyTree to {path} ({filesize / 1024**2:.1f} MiB)")


def json_dump(f: Any, hps: dict) -> None:
    json_str = json.dumps(hps, sort_keys=True)
    f.write((json_str + "\n").encode())


def save(
    path: str | Path,
    tree: PyTree[eqx.Module],
    hyperparameters: Optional[dict] = None,
    dump_fn: Callable = json_dump,
) -> None:
    """Save a PyTree to disk along with hyperparameters used to generate it.

    Assumes none of the hyperparameters are JAX arrays, as these are not
    JSON serialisable.

    Based on the Equinox serialisation [example](https://docs.kidger.site/equinox/examples/serialisation/).

    Arguments:
        path: The path of the file to be saved. Note that the file at this path
            will be overwritten if it exists.
        tree: The PyTree to save. Its structure should match the return
            type of a function `setup_func`, which will be passed to `load`.
        hyperparameters: A dictionary of arguments for
            `setup_func` that were used to generate the PyTree, and upon
            loading, will be used to regenerate an appropriate skeleton to
            populate with the saved values from `tree`.
        dump_fn: Function that writes hyperparameters to the open file handle. It must
            write bytes when the file is opened in binary mode.
    """
    with open(path, "wb") as f:
        dump_fn(f, hyperparameters)
        eqx.tree_serialise_leaves(f, tree)

    filesize = os.path.getsize(path)

    logger.info(f"Wrote PyTree to {path} ({filesize / 1024**2:.1f} MiB)")


def load(
    path: Path | str,
    setup_func: Callable[..., PyTree[Any, "T"]],
    **kwargs,
) -> PyTree[Any, "T"]:
    """Setup a PyTree from stored data and hyperparameters.

    Arguments:
        path: The path of the file to be loaded.
        setup_func: A function that returns a PyTree of the same structure
            as the PyTree that was saved to `path`, and which may take as
            arguments `hyperparameters` which `save` may have saved to the same
            file. It must take a keyword argument `key`.
    """
    tree, _ = load_with_hyperparameters(path=path, setup_func=setup_func, **kwargs)
    return tree


def load_with_hyperparameters(
    path: Path | str,
    setup_func: Callable[..., PyTree[Any, "T"]],
    missing_hyperparameters: Optional[dict[str, Any]] = None,
    **kwargs,
) -> tuple[PyTree[Any, "T"], dict[str, Any]]:
    """Setup a PyTree from stored data and hyperparameters.

    Arguments:
        path: The path of the file to be loaded.
        setup_func: A function that returns a PyTree of the same structure
            as the PyTree that was saved to `path`, and which may take as
            arguments `hyperparameters` which `save` may have saved to the same
            file. It must take a keyword argument `key`.
        missing_hyperparameters: A dictionary of hyperparameters, whose structure must
            contain all the leaves of the loaded hyperparameter dictionary, but may
            possess additional leaves if the signature of `setup_func` has been expanded
            since save time, so that we may call it properly here. Note that these
            additional parameters should not affect the structure of the saved PyTrees,
            or deserialisation will fail.
    """

    with open(path, "rb") as f:
        hyperparameters = json.loads(f.readline().decode())
        if hyperparameters is None:
            hyperparameters = dict()
        elif missing_hyperparameters is not None:
            hyperparameters = nested_dict_update(
                missing_hyperparameters, hyperparameters
            )
        tree = setup_func(**hyperparameters, key=jr.PRNGKey(0))
        tree = eqx.tree_deserialise_leaves(f, tree, **kwargs)

    return tree, hyperparameters


# TODO: Could also process non-array non-builtin datatypes, like `np.int64`
@filter_wrap(lambda leaf: is_type(Array, np.ndarray)(leaf))
def arrays_to_lists(tree: PyTree[Array | np.ndarray]) -> PyTree[list]:
    """Make JSON serialisable by converting all array leaves to lists."""
    return jt.map(lambda x: x.tolist(), tree)
