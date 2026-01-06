import fnmatch
import json
import os
import time
from collections.abc import Callable, Sequence
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any, Literal, Optional

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import jax_cookbook.tree as jtree
from feedbax.loss import AbstractLoss
from feedbax.noise import Multiplicative, Normal
from feedbax.task import SimpleReaches
from feedbax.xabdeef.losses import simple_reach_loss
from ipyfilechooser import FileChooser
from IPython.display import display
from ipywidgets import HTML
from jax_cookbook import is_module, is_type
from jaxtyping import PRNGKeyArray, PyTree
from sqlalchemy.orm import Session

from feedbax._experiments.config import PATHS
from feedbax._experiments.constants import (
    TASK_EVAL_PARAMS,
)
from feedbax._experiments.database import (
    get_model_record,
    load_tree_with_hps,
    load_tree_without_hps,
    record_to_hps_train,
)
from feedbax._experiments.misc import (
    take_model,
)
from feedbax._experiments.tree_utils import (
    at_path,
    subdict,
)
from feedbax._experiments.types import LDict, TaskModelPair, TreeNamespace


def get_latest_matching_file(directory: str, pattern: str) -> Optional[str]:
    """
    Returns the filename of the latest file in the given directory that matches the given pattern.

    The 'latest' file is determined by sorting the filenames in descending order.

    Arguments:
        directory: The directory path to search in.
        pattern: The pattern to match filenames against (e.g., 'A-*.json').

    Returns:
        The filename of the latest matching file, or None if no match is found.

    Raises:
        OSError: If there's an error reading the directory.
    """
    try:
        all_files = os.listdir(directory)
    except OSError as e:
        print(f"Error reading directory {directory}: {e}")
        return None

    matching_files = fnmatch.filter(all_files, pattern)

    if not matching_files:
        return None

    sorted_files = sorted(matching_files, reverse=True)

    return sorted_files[0]


def display_model_filechooser(
    path,
    filter_pattern="*.eqx",
):
    """Display a file chooser interface for the files at `path` whose names satisfy `filter_pattern`.

    The default filename is the one that sorts last.
    """
    fc = FileChooser(path)
    fc.filter_pattern = filter_pattern
    fc.title = "Select model file:"
    params_widget = HTML("")

    default_filename = get_latest_matching_file(path, fc.filter_pattern)
    if default_filename is not None:
        fc.default_filename = default_filename

    def display_params(path, html_widget):
        with open(path, "r") as f:
            params = json.load(f)
        params_str = eqx.tree_pformat(
            params, truncate_leaf=lambda x: isinstance(x, list) and len(x) > 10
        )
        html_widget.value = "<pre>" + params_str.replace(":\n", ":") + "</pre>"

    def display_params_callback(fc: Optional[FileChooser]):
        if fc is None:
            return
        if fc.selected is None:
            raise RuntimeError("")
        return display_params(
            fc.selected.replace("trained_models.eqx", "hyperparameters.json"),
            params_widget,
        )

    fc.register_callback(display_params_callback)

    display(fc, params_widget)

    return fc


def wait_for_value(variable, timeout: float = 3600):
    end_time = time.monotonic() + timeout
    while variable is None:
        if time.monotonic() > end_time:
            return False  # Timeout occurred
        time.sleep(0.1)
    return True


def choose_model_file(filter_pattern="*.eqx", timeout: float = 3600) -> str:
    """Displays a file chooser in the model directory until"""
    fc = display_model_filechooser(PATHS.models, filter_pattern=filter_pattern)

    if wait_for_value(fc, timeout=timeout):
        assert fc.selected is not None
        return fc.selected
    else:
        return f"{fc.default_path}/{fc.default_filename}"


def find_unique_filepath(path: str | Path, search_string: str) -> Optional[Path]:
    """
    Returns the path of the unique file in a directory whose filename contains a given string.

    Arguments:
        directory: The path to the directory to search in.
        search_string: The string to search for in filenames.

    Returns:
        The path of the unique file if found, None otherwise.
    """
    # Convert directory to Path object if it's a string
    dir_path = Path(path) if isinstance(path, str) else path

    matching_files = [
        filename
        for filename in dir_path.iterdir()
        if filename.is_file() and search_string.lower() in filename.name.lower()
    ]

    if len(matching_files) == 1:
        return matching_files[0]
    elif len(matching_files) == 0:
        print(f"No files found containing '{search_string}'.")
        return None
    else:
        print(f"Multiple files found containing '{search_string}':")
        for file in matching_files:
            print(file.name)
        return None


def set_model_noise(
    model,
    noise_stds: dict[Literal["feedback", "motor"] | str, Optional[float]],
    enable_noise: bool = True,
):
    """Change the system noise strength of a model."""
    get_noise_fns = dict(
        feedback=lambda std: Normal(std=std),
        motor=lambda std: Multiplicative(Normal(std=std)) + Normal(std=1.8 * std),
    )

    noise_fns = jt.map(
        lambda std, get_noise_fn: get_noise_fn(std),
        noise_stds,
        get_noise_fns,
    )

    wheres = dict(
        feedback=lambda model: model.step.feedback_channels[0].noise_func,
        motor=lambda model: model.step.efferent_channel.noise_func,
    )

    pairs, LeafTuple = jtree.zip_named(
        noise_func=noise_fns,
        where=wheres,
        is_leaf=is_module,
    )

    for noise_fn, where in jt.leaves(pairs, is_leaf=is_type(LeafTuple)):
        model = eqx.tree_at(where, model, noise_fn)

    if enable_noise:
        model = eqx.tree_at(
            lambda model: (
                model.step.feedback_channels[0].add_noise,
                model.step.efferent_channel.add_noise,
            ),
            model,
            (True, True),
        )

    return model


def setup_models_only(task_model_pair_setup_fn, *args, **kwargs):
    """Given a function that returns task-model pairs, just get the models."""
    task_model_pairs = task_model_pair_setup_fn(*args, **kwargs)
    _, models = jtree.unzip(task_model_pairs)
    return models


def setup_tasks_only(task_model_pair_setup_fn, *args, **kwargs):
    """Given a function that returns task-model pairs, just get the tasks."""
    task_model_pairs = task_model_pair_setup_fn(*args, **kwargs)
    tasks, _ = jtree.unzip(task_model_pairs)
    return tasks


def convert_tasks_to_small(tasks):
    """Given a PyTree of tasks, return a matching PyTree where each task uses the small set of validation trials."""
    return jt.map(
        lambda task: eqx.tree_at(
            lambda task: tuple(getattr(task, k) for k in TASK_EVAL_PARAMS["small"]),
            task,
            tuple(TASK_EVAL_PARAMS["small"].values()),
        ),
        tasks,
        is_leaf=is_module,
    )


# When excluding models based on performance measures aside from loss, these are the ones we'll consider
MEASURES_TO_RATE = ("end_pos_error",)


def setup_replicate_info(models, hps, *, key):
    """Returns a skeleton PyTree for loading the replicate info"""

    def models_tree_with_value(value):
        return jt.map(
            lambda _: value,
            models,
            is_leaf=is_module,
        )

    def get_measure_dict(value):
        return dict.fromkeys(
            ("best_total_loss",) + MEASURES_TO_RATE,
            models_tree_with_value(value),
        )

    # For each piece of replicate info, we need a PyTree with the same structure as the model PyTree
    return {
        info_label: models_tree_with_value(value)
        for info_label, value in dict(
            best_save_idx=jnp.zeros(hps.model.n_replicates, dtype=int),
            best_saved_iteration_by_replicate=[0] * hps.model.n_replicates,
            losses_at_best_saved_iteration=jnp.zeros(hps.model.n_replicates, dtype=float),
            losses_at_final_saved_iteration=jnp.zeros(hps.model.n_replicates, dtype=float),
            readout_norm=jnp.zeros(hps.model.n_replicates, dtype=float),
        ).items()
    } | dict(
        best_replicates=get_measure_dict(0),
        included_replicates=get_measure_dict(jnp.ones(hps.model.n_replicates, dtype=bool)),
    )


# TODO: Update docstring
def query_and_load_model(
    db_session: Session,
    setup_task_model_pair: Callable,
    params_query: dict[str, Any],
    noise_stds: Optional[dict[Literal["feedback", "motor"] | str, Optional[float]]] = None,
    surgeries: Optional[dict[tuple, Any]] = None,
    tree_inclusions: Optional[dict[type, Optional[Any | Sequence | Callable]]] = None,
    exclude_underperformers_by: Optional[str] = None,
    exclude_method: Literal["nan", "remove", "best-only"] = "nan",
    return_task: bool = False,
):
    """Query the models table in the project database and return the loaded and processed model( replicates).

    Arguments:
        db_session: The SQLAlchemy database session
        setup_task_model_pair: The function used to setup the task-model PyTree for this
            part of the project.
        params_query: The parameters used to query the records of the model table of the database.
            If more than one record matches, an error is raised.
        noise_stds:
        surgeries: Specifies model surgeries to perform. For example, passing
            `{('step', 'feedback_channels', 0, 'noise_func', 'std'): 0.1}` means to set the value of
            the feedback noise std to 0.1.
        tree_inclusions: Optionally, rules by which to include parts of dict nodes in the loaded
            PyTree of models. Each rule's key is a dict node type in the PyTree,
            and the respective values are the node key(s) which should be kept, or a callable that
            returns true for the keys to be kept. If `None`, all nodes are kept as-is.
        exclude_underperformers_by: An optional key of a performance measure evaluated in
            `post_training` by which to exclude model replicates. Excluded replicates will have
            their parameters replaced with NaN in the arrays of the returned PyTree.
        exclude_method: Whether to index-out the included replicates ('remove'), replace their
            model parameters with NaN ('nan'), or return only the single best replicate
            ('best-only').

    Returns:
        model: The model PyTree
        model_info: The object mapping to the model's database record
        replicate_info: A dict of information about the model replicates
        n_replicates_included: The number of replicates not excluded (made NaN) from the model arrays
        hps_train: Training hyperparameters extracted from database record
    """
    exclude_method_ = exclude_method.lower()

    model_info = get_model_record(
        db_session,
        has_replicate_info=True,
        **params_query,
    )

    if model_info is None:
        raise ValueError(f"No model found in database given parameters: {params_query}")

    assert model_info.replicate_info_path is not None, (
        "Model record's replicate_info_path is None, but has_replicate_info==True"
    )

    # Extract training hyperparameters from database record first
    hps_train = record_to_hps_train(model_info)

    model: eqx.Module
    model = load_tree_without_hps(
        model_info.path,
        hps_train,
        partial(setup_models_only, setup_task_model_pair),
    )
    #! Since `setup_models_only` merely discards the tasks after generation,
    #! we should be able to return `(task, model), hps` from `load_tree_with_hps`.
    #! Why not?

    #! TODO: If not too difficult, store seed/key value in the `model_record` so we
    #! obtain the identical task, down to the trials.
    replicate_info, _ = load_tree_with_hps(
        model_info.replicate_info_path,
        partial(setup_replicate_info, model),
    )

    task = None
    if return_task:
        task = setup_tasks_only(setup_task_model_pair, hps_train, key=jr.PRNGKey(0))

    n_replicates_included = model_info.model__n_replicates

    if surgeries is not None:
        for path, value in surgeries.items():
            model = jt.map(
                lambda m: eqx.tree_at(at_path(path), m, value),
                model,
                is_leaf=is_module,
            )

    if noise_stds is not None:
        # NOTE: Map over `model` as if it is type `PyTree[eqx.Module]`. This may be
        #       vestigial but it's also more general, and isn't costly, so I am leaving it as-is.
        model = jt.map(
            partial(
                set_model_noise,
                noise_stds=noise_stds,
                enable_noise=True,
            ),
            model,
            is_leaf=is_module,
        )

    if tree_inclusions is not None:
        for dict_type, inclusion in tree_inclusions.items():
            if inclusion is not None:
                replace_fn = lambda d, inclusion: subdict(d, inclusion)

                if isinstance(inclusion, Callable):
                    # Callables always result in sequence-like inclusions
                    inclusion = [
                        x for x in getattr(model_info, "inclusion", []) if all(inclusion(x))
                    ]
                elif isinstance(inclusion, str) or not isinstance(inclusion, Sequence):
                    # If not a Callable and not a Sequence, then assume we've
                    # been given a single key to include
                    replace_fn = lambda d, inclusion: d[inclusion]

                model, replicate_info = jt.map(
                    lambda d: replace_fn(d, inclusion),
                    (model, replicate_info),
                    is_leaf=is_type(dict_type),
                )

    if exclude_underperformers_by is not None:
        included_replicates = replicate_info["included_replicates"][exclude_underperformers_by]
        best_replicate = replicate_info["best_replicates"][exclude_underperformers_by]

        if exclude_method_ == "nan":

            def include_fn_nan(model, included, best):
                return jtree.array_set_scalar(model, jnp.nan, jnp.where(~included)[0])

            include_fn = include_fn_nan
        elif exclude_method_ == "remove":

            def include_fn_remove(model, included, best):
                return take_model(model, jnp.where(included)[0])

            include_fn = include_fn_remove
        elif exclude_method_ == "best-only":

            def include_fn_best(model, included, best):
                return take_model(model, best)

            include_fn = include_fn_best
        else:
            raise ValueError(f"Invalid exclude_method '{exclude_method_}'")

        model = jt.map(
            include_fn,
            model,
            included_replicates,
            best_replicate,
            is_leaf=is_module,
        )

        # print("\nReplicates included in analysis for each training condition:")
        # eqx.tree_pprint(jt.map(lambda x: jnp.where(x)[0], included_replicates), short_arrays=False)

        n_replicates_included = jt.map(lambda x: jnp.sum(x).item(), included_replicates)

        if any(n < 1 for n in jt.leaves(n_replicates_included)):
            raise ValueError("No replicates met inclusion criteria for at least one model variant")

    if return_task:
        tree = (task, model)
    else:
        tree = model

    return tree, model_info, replicate_info, n_replicates_included, hps_train
