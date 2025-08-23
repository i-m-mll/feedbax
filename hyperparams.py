import logging
from collections.abc import Callable, Sequence
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, Optional, TypeVar

import equinox as eqx
import jax.tree as jt
import jax_cookbook.tree as jtree
import yaml
from jax_cookbook import anyf, is_type, where_attr_strs_to_func
from jaxtyping import ArrayLike, PyTree

from feedbax_experiments.config import STRINGS, load_config
from feedbax_experiments.constants import get_iterations_to_save_model_parameters
from feedbax_experiments.misc import copy_delattr
from feedbax_experiments.tree_utils import (
    tree_level_labels,
)
from feedbax_experiments.types import (
    LDict,
    TaskModelPair,
    TreeNamespace,
    dict_to_namespace,
    is_dict_with_int_keys,
)

# We use LDict labels to identify levels in task-model pair trees
# The label format is expected to be double-underscore separated parts that map to hyperparameter paths
# For example, "train__method" maps to hps.train.method and "train__pert__std" maps to hps.train.pert.std
T = TypeVar("T")
NT = TypeVar("NT", bound=SimpleNamespace)
DT = TypeVar("DT", bound=dict)


logger = logging.getLogger(__name__)


class _Placeholder: ...


def _get_key(hps: TreeNamespace, attr_str: str):
    attr_strs = attr_str.split(".")
    obj = hps
    for s in attr_strs:
        obj = getattr(obj, s, _Placeholder)
        if obj is _Placeholder:
            return None
    return obj


def _key_not_none(hps: TreeNamespace, attr_str: str):
    return _get_key(hps, attr_str) is not None


def set_dependent_hps(
    hps: TreeNamespace, config_type: Optional[Literal["training", "analysis"]] = None
) -> TreeNamespace:
    """Calculate and add any hyperparameters, which depend on other hyperparameters."""
    # Avoid in-place modification
    hps = deepcopy(hps)

    if config_type == "training":
        hps.intervention_scaleup_batches = [
            hps.n_batches_baseline,
            hps.n_batches_baseline + hps.n_scaleup_batches,
        ]
        hps.n_batches = hps.n_batches_baseline + hps.n_batches_condition
        hps.save_model_parameters = get_iterations_to_save_model_parameters(hps.n_batches)

    return hps


def cast_hps(
    hps: TreeNamespace, config_type: Optional[Literal["training", "analysis"]] = None
) -> TreeNamespace:
    """Cast any hyperparameters to their appropriate types."""
    hps = deepcopy(hps)

    if config_type is not None:
        train_where_attr_str = {"training": "where", "analysis": "train.where"}[config_type]
        train_where_where = where_attr_strs_to_func(train_where_attr_str)
        # train_where_key = train_where_attr_str.replace('.', STRINGS.hps_level_label_sep)

        if _key_not_none(hps, train_where_attr_str):
            # Wrap in an LDict so it doesn't get flattened by `flatten_hps`
            hps = eqx.tree_at(
                lambda hps: _get_key(hps, train_where_attr_str),
                hps,
                # Use the same key for simplicity in `flatten_hps`
                LDict.of("train__where")(train_where_where(hps)),
            )

    return hps


def load_hps(
    name: str, config_type: Optional[Literal["training", "analysis"]] = None
) -> TreeNamespace:
    """Given a path to a YAML config file, load it and convert to a PyTree of hyperparameters.

    If the path is not found, pass it as the experiment id to try to get a default config.
    So you can pass e.g. `"1-1"` to load the default hyperparameters for analysis module 1-1.
    Note that this is like treating `config_path` as a local path to a YAML file in
    `feedbax_experiments.config`.
    """
    # Load the defaults and update with the user-specified config
    config = load_config(name, config_type)
    return config_to_hps(config, config_type)


def config_to_hps(
    config: dict, config_type: Optional[Literal["training", "analysis"]] = None
) -> TreeNamespace:
    """Convert a config dict to a TreeNamespace of hyperparameters."""
    # ? Move this after `cast_hps` and exclude all `LDict`
    # Convert to a (nested) namespace instead of a dict, for attribute access
    hps = dict_to_namespace(config, to_type=TreeNamespace, exclude=is_dict_with_int_keys)
    # Make corrections and add in any derived values
    hps = set_dependent_hps(hps, config_type)
    hps = cast_hps(hps, config_type)
    return hps


def promote_hps(hps: TreeNamespace, *keys: str) -> TreeNamespace:
    """Remove the `model` attribute, and bring its own attributes out to the top level."""
    hps = deepcopy(hps)
    # Bring out the parameters under the `model` key; i.e. "model" won't appear in their flattened keys
    for key in keys:
        subtree = getattr(hps, key, None)
        if subtree is not None:
            hps.__dict__.update(subtree.__dict__)
            delattr(hps, key)
    return hps


def flatten_hps(
    hps: TreeNamespace,
    prefix: Optional[str] = None,
    is_leaf: Optional[Callable] = anyf(is_type(list), is_type(LDict)),
    ldict_to_dict: bool = True,
    join_with: str = STRINGS.hps_level_label_sep,
) -> TreeNamespace:
    """Flatten the hyperparameter namespace, joining keys with underscores."""
    hps = deepcopy(hps)

    values = jt.leaves(hps, is_leaf=is_leaf)

    # TODO: More general function that inverts all relevant operations in `cast_hps`?
    if ldict_to_dict:
        values = [dict(v) if isinstance(v, LDict) else v for v in values]

    keys = jt.leaves(jtree.labels(hps, join_with=join_with, is_leaf=is_leaf))

    if prefix is not None:
        keys = [join_with.join([prefix, k]) for k in keys]

    return TreeNamespace(**dict(zip(keys, values)))


def update_hps_given_tree_path(
    hps: TreeNamespace, path: tuple, labels: Sequence[str]
) -> TreeNamespace:
    """
    Update hyperparameters based on the path of a task-model pair in the training PyTree.

    Args:
        hps: The base hyperparameters
        path: Path to a leaf in the task-model pair tree
        labels: LDict labels for each level in the tree

    Returns:
        Updated hyperparameters with values from the path
    """
    hps = deepcopy(hps)
    for node_key, label in zip(path, labels):
        # Split the label to get the path into `hps`
        # For example: "train__method" -> ["train", "method"]
        parts = label.split(STRINGS.hps_level_label_sep)

        if not parts:
            continue

        # Navigate to the nested attribute and assign
        obj = hps
        for part in parts[:-1]:
            obj = getattr(obj, part)

        # Set the final attribute value
        last_part = parts[-1]
        setattr(obj, last_part, node_key.key)

    return hps


def fill_out_hps(
    hps_common: TreeNamespace, task_model_pairs: PyTree[TaskModelPair, "T"]
) -> PyTree[TreeNamespace, "T"]:
    """Given a common set of hyperparameters and a tree of task-model pairs, create a matching tree of
    pair-specific hyperparameters.

    This works because `task_model_pairs` is a tree of dicts, where each level of the tree is a different
    dict subtype, and where the keys are the values of hyperparameters. Each dict subtype has a fixed
    mapping to a particular
    """
    level_labels = tree_level_labels(task_model_pairs, is_leaf=is_type(TaskModelPair))

    # TODO: Use `jt.map_with_path` if updating to new JAX version
    return jt.map(
        lambda _, path: update_hps_given_tree_path(
            hps_common,
            path,
            level_labels,
        ),
        task_model_pairs,
        jtree.key_tuples(task_model_pairs, is_leaf=is_type(TaskModelPair)),
        is_leaf=is_type(TaskModelPair),
    )


def take_train_histories_hps(hps: TreeNamespace) -> TreeNamespace:
    """Selects specific hyperparameters from a TreeNamespace structure."""
    return TreeNamespace(
        train=TreeNamespace(
            n_batches=hps.train.n_batches,
            batch_size=hps.train.batch_size,
            where=hps.train.where,
            save_model_parameters=hps.train.save_model_parameters,
        ),
        model=TreeNamespace(
            n_replicates=hps.model.n_replicates,
        ),
    )


def flat_key_to_where_func(key: str, sep: str = STRINGS.hps_level_label_sep) -> Callable:
    """Convert a flattened hyperparameter key to a where-function."""
    where_str = key.replace(sep, ".")
    return where_attr_strs_to_func(where_str)


def use_train_hps_when_none(hps: TreeNamespace) -> TreeNamespace:
    """Replace any unspecified evaluation params with matching loading (training) params"""
    hps_train = hps.train
    hps_other = copy_delattr(hps, "train")
    hps = hps_other.update_none_leaves(hps_train)
    hps.train = hps_train
    return hps
