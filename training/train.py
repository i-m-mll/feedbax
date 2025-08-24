import logging
from collections.abc import Mapping, Sequence
from copy import deepcopy
from functools import partial
from types import NoneType
from typing import Optional, TypeVar

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
    get_db_session,
    get_record,
    save_model_and_add_record,
)
from feedbax_experiments.hyperparams import config_to_hps, flatten_hps
from feedbax_experiments.misc import (
    GracefulInterruptHandler,
    GracefulStopRequested,
    load_module_from_package,
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


def train_setup(
    train_hps: TreeNamespace,
) -> tuple[TaskTrainer, AbstractLoss]:
    """Given the training hyperparameters, return a trainer object and loss function."""
    optimizer_class = partial(
        optax.adamw,
        weight_decay=train_hps.weight_decay,
    )

    schedule = make_delayed_cosine_schedule(
        train_hps.learning_rate_0,
        train_hps.constant_lr_iterations,
        train_hps.n_batches_baseline + train_hps.n_batches_condition,
        train_hps.cosine_annealing_alpha,
    )

    trainer = TaskTrainer(
        optimizer=optax.inject_hyperparams(optimizer_class)(
            learning_rate=schedule,
        ),
        checkpointing=True,
    )

    loss_func = simple_reach_loss()

    if all(
        getattr(train_hps, k, None) is not None
        for k in ("readout_norm_loss_weight", "readout_norm_value")
    ):
        readout_norm_loss = train_hps.readout_norm_loss_weight * get_readout_norm_loss(
            train_hps.readout_norm_value
        )
        loss_func = loss_func + readout_norm_loss

    return trainer, loss_func


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
        **kwargs,
    )

    if pretrain_history is None:
        train_history_all = train_history
    else:
        train_history_all = jtree.concatenate([pretrain_history, train_history])

    return trained, train_history_all


def where_strs_to_funcs(where_strs: Sequence[str] | dict[int, Sequence[str]]):
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


def train_and_save_models(
    config: dict,
    expt_name: str,
    untrained_only: bool = True,
    postprocess: bool = True,
    n_std_exclude: int = 2,  # re: postprocessing
    save_figures: bool = True,  # re: postprocessing
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
    hps = config_to_hps(config, config_type="training")

    db_session = get_db_session()

    key_init, key_train, key_eval = jr.split(key, 3)

    # User specifies which variant to run using the `id` key
    training_module = load_module_from_package(expt_name, registry=EXPERIMENT_REGISTRY)

    # `all_hps` is a tree of pair-specific hps
    task_model_pairs, all_hps_train = jtree.unzip(
        training_module.get_train_pairs(hps | dict(expt_name=expt_name), key_init)
    )

    if untrained_only:
        task_model_pairs = skip_already_trained(
            db_session,
            task_model_pairs,
            all_hps_train,
            n_std_exclude,
            save_figures,
        )

    if not any(jt.leaves(task_model_pairs, is_leaf=is_type(TaskModelPair))):
        logger.info("No models to train. Exiting.")
        # return jt.map(lambda _: None, task_model_pairs, is_leaf=is_type(TaskModelPair))
        return None, None, None

    # TODO: Also get `trainer`, `loss_func`, ... as trees like `task_model_pairs`
    # Otherwise certain hyperparameters (e.g. learning rate) will be constant
    # when the user might expect them to vary due to their config file.
    trainer, loss_func = train_setup(hps)

    ## Train and save all the models.
    # TODO: Is this correct? Or should we pass the task for the respective training method?
    task_baseline: AbstractTask = jt.leaves(task_model_pairs, is_leaf=is_type(TaskModelPair))[
        0
    ].task

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
                key=key_train,  #! Use the same PRNG key for all training runs
                ensembled=True,
                loss_func=loss_func,
                task_baseline=task_baseline,
                where_train=where_strs_to_funcs(dict(hps.where)),
                batch_size=hps.batch_size,
                log_step=LOG_STEP,
                save_model_parameters=hps.save_model_parameters,
                state_reset_iterations=hps.state_reset_iterations,
                # disable_tqdm=True,
            )
            model_record = save_model_and_add_record(
                db_session,
                trained_model,
                hps,
                train_history=train_history,
                version_info=version_info,
            )
            if postprocess:
                process_model_post_training(
                    db_session,
                    model_record,
                    n_std_exclude,
                    process_all=True,
                    save_figures=save_figures,
                )
            return trained_model, train_history, model_record

        try:
            trained_models, train_histories, model_records = jtree.unzip(
                jtree.map_tqdm(
                    # TODO: Could return already-trained models instead of None; would need to return
                    # `already_trained` bool from `skip_already_trained`
                    lambda pair, hps: train_and_save_pair(pair, hps) if pair is not None else None,
                    task_model_pairs,
                    all_hps_train,
                    label="Training all pairs",
                    is_leaf=is_type(TaskModelPair, NoneType),
                )
            )
        except GracefulStopRequested:
            raise KeyboardInterrupt

    return trained_models, train_histories, model_records


def concat_save_iterations(iterations: Array, n_batches_seq: Sequence[int]):
    total_batches = np.cumsum([0] + list(n_batches_seq))
    return jnp.concatenate(
        [iterations[iterations < n] + total for n, total in zip(n_batches_seq, total_batches)]
    )


def skip_already_trained(
    db_session: Session,
    task_model_pairs: PyTree[TaskModelPair, "T"],
    all_hps_train: PyTree[dict, "T"],
    n_std_exclude: int,
    save_figures: bool,
    post_process: bool = True,
):
    """Replace leaves in the tree of training pairs with None, where those models were already trained."""
    all_hps_train = arrays_to_lists(all_hps_train)

    def get_query_hps(hps: TreeNamespace, **kwargs) -> TreeNamespace:
        hps = deepcopy(hps)
        hps.is_path_defunct = False
        for k, v in kwargs.items():
            setattr(hps, k, v)
        return hps

    # Get records for models that have already been trained and post-processed
    records = jt.map(
        lambda hps: get_record(
            db_session,
            ModelRecord,
            enforce_unique=False,
            **namespace_to_dict(flatten_hps(get_query_hps(hps, postprocessed=True))),
        ),
        all_hps_train,
        is_leaf=is_type(TreeNamespace),
    )

    record_exists = jt.map(
        lambda x: x is not None,
        records,
        is_leaf=lambda x: x is None or isinstance(x, ModelRecord),
    )

    if post_process:
        # Get models that have not been post-processed
        records_not_pp = jt.map(
            lambda hps: get_record(
                db_session,
                ModelRecord,
                **namespace_to_dict(flatten_hps(get_query_hps(hps, postprocessed=False))),
            ),
            all_hps_train,
            is_leaf=is_type(TreeNamespace),
        )

        # Post-process any models for which there is only a non-post-processed record
        jt.map(
            lambda record_not_pp, record: (
                process_model_post_training(
                    db_session,
                    record_not_pp,
                    n_std_exclude,
                    process_all=True,
                    save_figures=save_figures,
                )
                if record_not_pp is not None and record is None
                else None
            ),
            records_not_pp,
            records,
            is_leaf=is_type(ModelRecord),
        )

    pairs_to_skip, task_model_pairs = eqx.partition(
        task_model_pairs,
        record_exists,
        is_leaf=is_type(TaskModelPair),
    )

    pairs_to_skip_flat = jt.leaves(pairs_to_skip, is_leaf=is_type(TaskModelPair))
    if len(pairs_to_skip_flat) > 0:
        logger.info(
            f"Skipping training of {len(pairs_to_skip_flat)} models whose hyperparameters "
            "match models already in the database"
        )

    return task_model_pairs


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
