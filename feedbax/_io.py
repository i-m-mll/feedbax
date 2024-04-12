"""Facilities for saving and loading of experimental setups.

TODO:
- Could provide a simple interface to show the most recently saved files in
  a directory

:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from collections.abc import Callable
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Optional

import jax.random as jr
from jaxtyping import PyTree
import equinox as eqx


from feedbax.misc import git_commit_id, nested_dict_update


logger = logging.getLogger(__name__)


def save(
    path: str | Path,
    tree: PyTree[eqx.Module],
    hyperparameters: Optional[dict] = None,
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
    """
    with open(path, "wb") as f:
        hyperparameter_str = json.dumps(hyperparameters)
        f.write((hyperparameter_str + "\n").encode())
        eqx.tree_serialise_leaves(f, tree)


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
    tree, _ = load_with_hyperparameters(
        path=path, setup_func=setup_func, **kwargs
    )
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
            hyperparameters = nested_dict_update(missing_hyperparameters, hyperparameters)
        tree = setup_func(**hyperparameters, key=jr.PRNGKey(0))
        tree = eqx.tree_deserialise_leaves(f, tree, **kwargs)

    return tree, hyperparameters


def _save_with_datetime_and_commit_id(
    tree: PyTree[eqx.Module],
    hyperparameters: Optional[dict] = None,
    path: Optional[str | Path] = None,
    save_dir: str | Path = Path("."),
    suffix: Optional[str] = None,
) -> None:
    """Save a PyTree to disk along with hyperparameters used to generate it.

    If a path is not specified, a filename will be generated from the current
    time and the commit ID of the `feedbax` repository, and the file will be
    saved in `save_dir`, which defaults to the current working directory.

    Assumes none of the hyperparameters are JAX arrays, as these are not
    JSON serializable.

    Based on https://docs.kidger.site/equinox/examples/serialisation/
    """

    if path is None:
        timestr = datetime.today().strftime("%Y%m%d-%H%M%S")
        commit_id = git_commit_id()
        name = f"{timestr}_{commit_id}"
        if suffix is not None:
            name += f"_{suffix}"
        path = Path(save_dir) / f"{name}.eqx"

    with open(path, "wb") as f:
        hyperparameter_str = json.dumps(hyperparameters)
        f.write((hyperparameter_str + "\n").encode())
        eqx.tree_serialise_leaves(f, tree)