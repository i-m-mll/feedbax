import logging
from collections.abc import Mapping, Sequence
from copy import deepcopy
from functools import partial
from operator import not_
from pathlib import Path
from types import NoneType
from typing import Literal, Optional, TypeVar

import equinox as eqx
import feedbax
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import jax_cookbook.tree as jtree
import numpy as np
import optax
from feedbax._io import arrays_to_lists
from feedbax.loss import AbstractLoss
from feedbax.misc import attr_str_tree_to_where_func
from feedbax.task import AbstractTask
from feedbax.train import TaskTrainer
from feedbax.xabdeef.losses import simple_reach_loss
from jax_cookbook import is_type
from jaxtyping import Array, PRNGKeyArray, PyTree
from sqlalchemy.orm import Session

import feedbax_experiments
from feedbax_experiments.database import (
    ModelRecord,
    db_session,
    get_db_session,
    get_record,
    save_model_and_add_record,
)
from feedbax_experiments.hyperparams import config_to_hps, flatten_hps
from feedbax_experiments.misc import (
    GracefulInterruptHandler,
    GracefulStopRequested,
    log_version_info,
)
from feedbax_experiments.plugins import EXPERIMENT_REGISTRY
from feedbax_experiments.tree_utils import pp
from feedbax_experiments.types import TaskModelPair, TreeNamespace, namespace_to_dict

from .loss import get_readout_norm_loss
from .post_training import process_model_post_training

# TODO: Move to config
LOG_STEP = 500


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


T = TypeVar("T")


def setup_trainer(
    hps: TreeNamespace,
) -> TaskTrainer:
    """Given the training hyperparameters, return a trainer object and loss function."""
    optimizer_class = partial(
        optax.adamw,
        weight_decay=hps.weight_decay,
    )

    schedule = make_delayed_cosine_schedule(
        hps.learning_rate_0,
        hps.constant_lr_iterations,
        hps.n_batches_baseline + hps.n_batches_condition,
        hps.cosine_annealing_alpha,
    )

    trainer = TaskTrainer(
        optimizer=optax.inject_hyperparams(optimizer_class)(
            learning_rate=schedule,
        ),
        checkpointing=True,
    )

    return trainer


def train_pair(
    trainer: TaskTrainer,
    pair: TaskModelPair,
    n_batches: int,
    task_baseline: Optional[AbstractTask] = None,
    n_batches_baseline: int = 0,
    *,
    key: PRNGKeyArray,
    **kwargs,
):
    """Given a trainer instance and a task-model pair, train the model for a given number of batches."""
    key0, key1 = jr.split(key, 2)

    if n_batches_baseline > 0 and task_baseline is not None:
        pretrained, pretrain_history, opt_state = trainer(
            task_baseline,
            pair.model,
            n_batches=n_batches_baseline,
            run_label="Baseline training",
            key=key0,
            **kwargs,
        )
    else:
        pretrained = pair.model
        pretrain_history = None
        opt_state = None

    trained, train_history, _ = trainer(
        pair.task,
        pretrained,
        opt_state=opt_state,
        n_batches=n_batches,
        idx_start=n_batches_baseline,
        run_label="Condition training",
        key=key1,
        # disable_progress=False,
        **kwargs,
    )

    if pretrain_history is None:
        train_history_all = train_history
    else:
        train_history_all = jtree.concatenate([pretrain_history, train_history])

    return trained, train_history_all


def where_strs_to_fns(where_strs: Sequence[str] | dict[int, Sequence[str]]):
    if isinstance(where_strs, Mapping):
        return {
            i: attr_str_tree_to_where_func(strs)
            # TODO: Let the user pass a single sequence, instead of a dict of them
            for i, strs in where_strs.items()
        }
    elif isinstance(where_strs, Sequence):
        return attr_str_tree_to_where_func(where_strs)
    else:
        raise ValueError("`where_strs` must be a sequence or dict of sequences")


def prepare_to_train(
    expt_key: str,
    config: dict,
):
    hps = config_to_hps(config, config_type="training") | dict(expt_name=expt_key)

    with db_session(autocommit=False) as db:
        training_status, model_record = check_model_training_status(db, hps)

    return training_status, hps, model_record


def train_and_save_from_config(
    expt_key: str,
    config: dict,
    untrained_only: bool = True,
    finish_incomplete_postprocessing: bool = True,
    postprocess: bool = True,
    force_postprocess: bool = False,
    n_std_exclude: int = 2,  # re: postprocessing
    save_figures: bool = True,  # re: postprocessing
    fig_dump_path: Optional[Path] = None,
    fig_dump_formats: Sequence[str] = ("html",),
    fig_dump_params: Optional[dict | PyTree] = None,
    version_info: Optional[dict] = None,
    *,
    key: PRNGKeyArray,
):
    """Given config and experiment name, execute the respective training run.

    Args:
        config: Training configuration dictionary
        expt_name: Name of the training experiment
    """
    # Convert config dict to hyperparameters namespace
    training_status, hps, model_record = prepare_to_train(expt_key, config)

    # Handle forced re-postprocessing of already-postprocessed models
    if force_postprocess and training_status == "postprocessed":
        assert model_record is not None, (
            "`prepare_to_train` should have returned a model record given training status is "
            "'postprocessed'"
        )
        with db_session(autocommit=False) as db:
            logger.info(
                f"Force re-postprocessing already-postprocessed model for experiment {expt_key}"
            )
            process_model_post_training(
                db,
                model_record,
                n_std_exclude,
                process_all=True,
                save_figures=save_figures,
                dump_path=fig_dump_path,
                dump_formats=fig_dump_formats,
                dump_params=fig_dump_params,
            )
        return None, None, model_record

    #! Why is this in here, and not in the more general `train_and_save`
    if untrained_only:
        if training_status == "postprocessed":
            logger.info(f"Skipping training of already-trained model for experiment {expt_key}")
            #! TODO: Load already-trained model?
            return None, None, model_record

        if finish_incomplete_postprocessing and training_status == "not_postprocessed":
            assert model_record is not None, (
                "`prepare_to_train` should have returned a model record given training status is "
                "'not_postprocessed'"
            )
            with db_session(autocommit=False) as db:
                logger.info("Post-processing model that was trained but not yet post-processed")
                # Post-process any models for which there is only a non-post-processed record
                process_model_post_training(
                    db,
                    model_record,
                    n_std_exclude,
                    process_all=True,
                    save_figures=save_figures,
                    dump_path=fig_dump_path,
                    dump_formats=fig_dump_formats,
                    dump_params=fig_dump_params,
                )

    return train_and_save(
        hps,
        postprocess=postprocess,
        n_std_exclude=n_std_exclude,
        save_figures=save_figures,
        fig_dump_path=fig_dump_path,
        fig_dump_formats=fig_dump_formats,
        fig_dump_params=fig_dump_params,
        version_info=version_info,
        key=key,
    )


def train_and_save(
    hps: TreeNamespace,
    postprocess: bool = True,
    n_std_exclude: int = 2,  # re: postprocessing
    save_figures: bool = True,  # re: postprocessing
    fig_dump_path: Optional[Path] = None,
    fig_dump_formats: Sequence[str] = ("html",),
    fig_dump_params: Optional[dict | PyTree] = None,
    run_number: Optional[int] = None,
    version_info: Optional[dict] = None,
    *,
    key: PRNGKeyArray,
):
    training_module = EXPERIMENT_REGISTRY.get_training_module(hps.expt_name)
    key_init, key_train, key_eval = jr.split(key, 3)
    task_model_pair = training_module.setup_task_model_pair(hps, key=key_init)
    trainer = setup_trainer(hps)

    # Get loss update function if training module provides one
    if hasattr(training_module, 'get_loss_update_func'):
        loss_update_func, loss_update_start_iteration = training_module.get_loss_update_func(hps)
    else:
        loss_update_func, loss_update_start_iteration = None, 0

    ## Train and save all the models.
    with GracefulInterruptHandler(
        sensitive_msg="Keyboard interrupt caught: will exit cleanly after current model is trained...",
        stop_msg="Finished training and processing model, stopping as requested.",
        logger=logger,
    ) as interrupt_handler:

        @interrupt_handler
        def train_and_save_pair(pair, hps):
            trained_model, train_history = train_pair(
                trainer,
                pair,
                hps.n_batches,
                key=key_train,
                ensembled=True,
                # loss_func=loss_fn,
                # task_baseline=task_baseline,  #! TODO: Specify param(s) in config
                where_train=where_strs_to_fns(dict(hps.where)),
                batch_size=hps.batch_size,
                log_step=LOG_STEP,
                save_model_parameters=hps.save_model_parameters,
                state_reset_iterations=hps.state_reset_iterations,
                loss_update_func=loss_update_func,
                loss_update_start_iteration=loss_update_start_iteration,
                # disable_tqdm=True,
            )
            with db_session(autocommit=False) as db:
                model_record = save_model_and_add_record(
                    db,
                    trained_model,
                    hps,
                    train_history=train_history,
                    version_info=version_info,
                )
                if postprocess:
                    process_model_post_training(
                        db,
                        model_record,
                        n_std_exclude,
                        process_all=True,
                        save_figures=save_figures,
                        dump_path=fig_dump_path,
                        dump_formats=fig_dump_formats,
                        dump_params=fig_dump_params,
                        run_number=run_number,
                    )
            return trained_model, train_history, model_record

        try:
            trained_model, train_history, model_record = train_and_save_pair(task_model_pair, hps)
        except GracefulStopRequested:
            raise KeyboardInterrupt

    return trained_model, train_history, model_record


def concat_save_iterations(iterations: Array, n_batches_seq: Sequence[int]):
    total_batches = np.cumsum([0] + list(n_batches_seq))
    return jnp.concatenate(
        [iterations[iterations < n] + total for n, total in zip(n_batches_seq, total_batches)]
    )


def partition_by_training_status(
    db_session: Session,
    all_hps_train: PyTree[TreeNamespace, "T"],
) -> tuple[
    PyTree[Optional[ModelRecord], "T"], PyTree[Optional[ModelRecord], "T"], PyTree[bool, "T"]
]:
    """Partition a set of hyperparameters into those that correspond to models that have already
    been trained and post-processed, those that have been trained but not post-processed, and those
    that have not been trained."""
    all_hps_train = arrays_to_lists(all_hps_train)

    def get_query_hps(hps: TreeNamespace, **kwargs) -> TreeNamespace:
        hps = deepcopy(hps)
        hps.is_path_defunct = False
        for k, v in kwargs.items():
            setattr(hps, k, v)
        return hps

    # Get records for models that have already been trained and post-processed
    postprocessed = jt.map(
        lambda hps: get_record(
            db_session,
            ModelRecord,
            enforce_unique=False,
            **namespace_to_dict(flatten_hps(get_query_hps(hps, postprocessed=True))),
        ),
        all_hps_train,
        is_leaf=is_type(TreeNamespace),
    )

    # Get models that have not been postprocessed
    not_postprocessed = jt.map(
        lambda hps, is_pp: (
            get_record(
                db_session,
                ModelRecord,
                **namespace_to_dict(flatten_hps(get_query_hps(hps, postprocessed=False))),
            )
            if not is_pp
            else None
        ),
        all_hps_train,
        postprocessed,
        is_leaf=is_type(TreeNamespace),
    )

    untrained = jt.map(
        lambda x, y: x is None and y is None,
        postprocessed,
        not_postprocessed,
        is_leaf=is_type(ModelRecord, NoneType),
    )

    return postprocessed, not_postprocessed, untrained


def check_model_training_status(
    db_session: Session,
    hps: TreeNamespace,
) -> tuple[str, Optional[ModelRecord]]:
    """Check whether a model with the given hyperparameters has been trained and/or
    post-processed.

    Returns one of "untrained", "not_postprocessed", or "postprocessed".
    """
    postprocessed, not_postprocessed, is_untrained = partition_by_training_status(db_session, hps)

    results = dict(
        untrained=is_untrained,
        not_postprocessed=not_postprocessed is not None,
        postprocessed=postprocessed is not None,
    )

    assert sum(results.values()) == 1, (
        "Inconsistency in `partition_by_training_status` result: model cannot be more than one of: "
        "untrained, not_postprocessed, postprocessed"
    )

    status = next(k for k, v in results.items() if v)

    model_record = postprocessed or not_postprocessed or None

    return status, model_record


def make_delayed_cosine_schedule(init_lr, constant_steps, total_steps, alpha=0.001):
    """Returns an Optax schedule that starts with constant learning rate, then cosine anneals."""
    constant_schedule = optax.constant_schedule(init_lr)

    cosine_schedule = optax.cosine_decay_schedule(
        init_value=init_lr,
        decay_steps=max(0, total_steps - constant_steps),
        alpha=alpha,
    )
    return optax.join_schedules(
        schedules=[constant_schedule, cosine_schedule], boundaries=[constant_steps]
    )


def bernoulli_active(p: float):
    def active_fn(trial_spec, batch_info, key):
        return jr.bernoulli(key, p=p)

    return active_fn


def always_active(_: float):
    _one = jnp.array(1.0)

    def active_fn(trial_spec, batch_info, key):
        return _one

    return active_fn
