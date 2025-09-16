import argparse
import logging
from collections.abc import Callable
from functools import partial
from typing import Any, Sequence, TypeVar
from typing import Literal as L

import equinox as eqx
import feedbax.plotly as fbp
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import jax_cookbook.tree as jtree
import numpy as np
import plotly.graph_objects as go
from feedbax.misc import attr_str_tree_to_where_func
from feedbax.train import TaskTrainerHistory, WhereFunc, init_task_trainer_history
from feedbax.xabdeef.losses import simple_reach_loss
from jax_cookbook import anyf, is_module, is_type
from jaxtyping import Array, Int, PyTree
from rich.logging import RichHandler
from rich.progress import Progress
from sqlalchemy.orm import Session

from feedbax_experiments.analysis.aligned import (
    get_aligned_vars,
    get_reach_origins_directions,
)
from feedbax_experiments.analysis.state_utils import (
    get_pos_endpoints,
    vmap_eval_ensemble,
)
from feedbax_experiments.database import (
    MODEL_RECORD_BASE_ATTRS,
    EvaluationRecord,
    ModelRecord,
    add_evaluation,
    add_evaluation_figure,
    check_model_files,
    get_db_session,
    load_tree_with_hps,
    query_model_records,
    save_model_and_add_record,
)
from feedbax_experiments.misc import log_version_info
from feedbax_experiments.plugins import EXPERIMENT_REGISTRY
from feedbax_experiments.setup_utils import (
    setup_models_only,
    setup_tasks_only,
)
from feedbax_experiments.training.loss import get_reach_loss, get_readout_norm_loss
from feedbax_experiments.types import LDict, TreeNamespace

logger = logging.getLogger(__name__)


# Number of trials to evaluate when deciding which replicates to exclude
N_TRIALS_VAL = 5


T = TypeVar("T")


def setup_train_histories(
    models_tree,
    hps_train: TreeNamespace,
    *,
    key,
) -> dict[float, TaskTrainerHistory]:
    """Returns a skeleton PyTree for the training histories (losses, parameter history, etc.)

    Note that `init_task_trainer_history` depends on `task` to infer:

    1) The number and name of loss function terms;
    2) The structure of trial specs, in case `save_trial_specs is not None`.

    Here, neither of these are a concern since 1) we are always using the same
    loss function for each set of saved/loaded models in this project, 2) `save_trial_specs is None`.
    """
    # Assume that where funcs may be lists (normally defined as tuples, but retrieved through sqlite JSON)
    where_train = jt.map(
        attr_str_tree_to_where_func,
        hps_train.where,
        is_leaf=is_type(list),
    )

    loss_fn = get_reach_loss(hps_train)

    return jt.map(
        lambda models: init_task_trainer_history(
            loss_fn,
            hps_train.n_batches,
            hps_train.model.n_replicates,
            ensembled=True,
            ensemble_random_trials=False,
            save_model_parameters=jnp.array(hps_train.save_model_parameters),
            save_trial_specs=None,
            batch_size=hps_train.batch_size,
            loss_func_validation=loss_fn,
            model=models,
            where_train=dict(where_train),
        ),
        models_tree,
        is_leaf=is_module,
    )


def load_data(model_record: ModelRecord):
    """Loads models, hyperparameters and training histories from files."""
    # Load model and associated data
    expt_name = str(model_record.expt_name)

    if not model_record.path.exists() or not model_record.train_history_path.exists():
        logger.error(f"Model or training history file not found for {model_record.hash}")
        return

    training_module = EXPERIMENT_REGISTRY.get_training_module(expt_name)

    #! TODO: Setup `tasks` as well and use these to infer the loss function for the train
    #! history structure, to make this project-independent.
    models, hps = load_tree_with_hps(
        model_record.path,
        partial(
            setup_models_only,
            training_module.setup_task_model_pair,
        ),
    )
    logger.debug("Loaded models")

    train_histories, train_history_hps = load_tree_with_hps(
        model_record.train_history_path,
        partial(setup_train_histories, models),
    )
    logger.debug("Loaded train histories")

    return (
        models,
        hps,
        train_histories,
        train_history_hps,
    )


# TODO: map over this entire function and use `jtree.unzip` on the result,
# instead of mapping multiple times inside this function
def get_best_iterations_and_losses(
    train_histories: PyTree[TaskTrainerHistory], save_model_parameters: Array, n_replicates: int
):
    """Computes best iterations and corresponding losses for each replicate."""
    best_save_idx = jt.map(
        lambda history: jnp.argmin(
            history.loss.total[save_model_parameters],
            axis=0,
        ),
        train_histories,
        is_leaf=is_module,
    )

    best_saved_iterations = jt.map(
        lambda idx: save_model_parameters[idx].tolist(),
        best_save_idx,
    )

    losses_at_best_saved_iteration = jt.map(
        lambda history, saved_iterations: (
            history.loss.total[jnp.array(saved_iterations), jnp.arange(n_replicates)]
        ),
        train_histories,
        best_saved_iterations,
        is_leaf=is_module,
    )

    return best_save_idx, best_saved_iterations, losses_at_best_saved_iteration


def get_best_and_included(measure, n_std_exclude=2):
    best_idx = jnp.argmin(measure).item()
    bound = (measure[best_idx] + n_std_exclude * measure.std()).item()
    included = measure < bound
    return best_idx, included


def end_position_error(pos, eval_reach_length=1, last_n_steps=10):
    # Since the data is aligned, the goal is always at the same position
    goal_pos = jnp.array([eval_reach_length, 0])
    error = jnp.mean(jnp.linalg.norm(pos[..., -last_n_steps:, :] - goal_pos, axis=-1), axis=-1)
    return error


def get_measures_to_rate(model, task, hps):
    states = vmap_eval_ensemble(jr.PRNGKey(0), hps, model, task)

    origins, directions = get_reach_origins_directions(task, model, hps)
    aligned_pos = get_aligned_vars(
        states.mechanics.effector.pos - origins[..., None, :],
        directions,
    )
    end_pos_errors = jt.map(
        partial(end_position_error, eval_reach_length=task.eval_reach_length),
        aligned_pos,
    )
    mean_end_pos_errors = jt.map(
        lambda x: jnp.mean(x, axis=(0, -1)),  # eval & condition, but not replicate
        end_pos_errors,
    )

    return dict(
        end_pos_error=mean_end_pos_errors,
    )


def _get_most_recent_idxs(idxs: Sequence[int], max_idx: int) -> Any:
    """Returns the value for the largest key less than or equal to `idx`."""
    keys = jnp.array(sorted(idxs))
    key_idxs = np.searchsorted(keys, np.arange(max_idx + 1), side="right")
    return keys[key_idxs - 1]


def get_best_models(
    model_record: ModelRecord,
    models: PyTree[eqx.Module, "T"],
    train_histories: PyTree[TaskTrainerHistory],
    save_model_parameters: Array,
    best_save_idx: PyTree[Int[Array, " replicate"]],
    n_replicates: int,
    where_train: WhereFunc | dict[str, WhereFunc],
) -> PyTree[eqx.Module, "T"]:
    """Serializes models with the best parameters for each replicate and training condition."""
    # Get a function that returns the `where_fn` used on a given iteration
    if isinstance(where_train, dict):
        where_train_idxs = _get_most_recent_idxs(
            [int(k) for k in where_train.keys()],
            model_record.n_batches,
        )

        def get_where_train_for_idx(idx):
            return where_train[str(where_train_idxs[idx])]

        get_where_train = get_where_train_for_idx
    else:

        def get_where_train_fixed(idx):
            return where_train

        get_where_train = get_where_train_fixed

    # TODO: If any model parameters were trainable at the end of the training run,
    # but were not trainable at the time of the best iteration, then this will keep
    # the final parameters. However, we should probably keep the value of these parameters
    # at the best iteration, even though they were not trainable then, since perhaps they
    # became trainable later (i.e. resulting in the final values) and this may have
    # affected the final loss.
    # TODO: Similarly, I think this might fail if two replicates differ in whether a parameter
    # was trainable at the best iteration; we'll end up trying to do a `jtree.stack` on pytrees
    # where some array leaves are sometimes `None`. The solution to this is the same as the
    # solution above: we need to select the best version of parameters that were trainable
    # *at any point*
    best_saved_parameters = jtree.stack(
        [
            # Select the best parameters for each replicate, for all train histories
            jt.map(
                lambda train_history, best_idxs: jtree.take_multi(
                    # Filter out the parameters that were not trainable at the best iteration
                    eqx.filter(
                        train_history.model_parameters,
                        jtree.filter_spec_leaves(
                            train_history.model_parameters,
                            get_where_train(save_model_parameters[int(best_idxs[i])]),
                        ),
                        is_leaf=is_module,
                    ),
                    [int(best_idxs[i]), i],
                    [0, 1],
                ),
                train_histories,
                best_save_idx,
                is_leaf=is_module,
            )
            for i in range(n_replicates)
        ]
    )

    models_with_best_parameters = eqx.combine(models, best_saved_parameters)

    return models_with_best_parameters


# TODO
#! This no longer works because the model records are kept and post-processed individually.
#! It probably makes sense to move this to an analysis script since it involves loading
#! a spread of training conditions (e.g. train_pert_std, as here)
# def get_replicate_distribution_figure(
#     measure: LDict[float, Shaped[Array, 'replicates']],
#     yaxis_title="",
# ) -> go.Figure:

#     n_replicates = len(jt.leaves(measure)[0])

#     df = pd.DataFrame(measure).reset_index().melt(id_vars='index')
#     df["index"] = df["index"].astype(str)

#     fig = go.Figure()

#     strips = px.scatter(
#         df,
#         x='variable',
#         y='value',
#         color="index",
#         color_discrete_sequence=px.colors.qualitative.Plotly,
#         # stripmode='overlay',
#     )

#     strips.update_traces(
#         marker_size=10,
#         marker_symbol='circle-open',
#         marker_line_width=3,
#     )

#     violins = [
#         go.Violin(
#             x=[train_pert_std] * n_replicates,
#             y=data,
#             # box_visible=True,
#             line_color='black',
#             meanline_visible=True,
#             fillcolor='lightgrey',
#             opacity=0.6,
#             name=f"{train_pert_std}",
#             showlegend=False,
#             spanmode='hard',
#         )
#         for train_pert_std, data in measure.items()
#     ]

#     fig.add_traces(violins)
#     fig.add_traces(strips.data)

#     fig.update_layout(
#         xaxis_type='category',
#         width=800,
#         height=500,
#         xaxis_title="Train disturbance std.",
#         yaxis_title=yaxis_title,
#         # xaxis_range=[-0.5, len(disturbance_stds) + 0.5],
#         # xaxis_tickvals=np.linspace(0,1.2,4),
#         # yaxis_type='log',
#         violingap=0.1,
#         # showlegend=False,
#         legend_title='Replicate',
#         legend_tracegroupgap=4,
#         # violinmode='overlay',
#         barmode='overlay',
#         # boxmode='group',
#     )

#     return fig


def get_train_history_figures(
    histories: PyTree[TaskTrainerHistory, "T"],
    best_saved_iteration_by_replicate,
) -> PyTree[go.Figure, "T"]:
    def get_figure(history, best_save_iterations):
        fig = fbp.loss_history(history.loss)
        text = "Best iter. by replicate: " + ", ".join(str(idx) for idx in best_save_iterations)
        fig.add_annotation(
            dict(
                text=text,
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=1,
            )
        )
        return fig

    return jt.map(
        lambda history, best_save_iterations: get_figure(history, best_save_iterations),
        histories,
        best_saved_iteration_by_replicate,
        is_leaf=is_type(TaskTrainerHistory),
    )


class FigFuncSpec(eqx.Module):
    func: Callable[..., go.Figure | LDict[float, go.Figure]]
    args: tuple[PyTree, ...]


def save_training_figures(
    db_session: Session,
    eval_info: EvaluationRecord,
    train_histories,
    replicate_info,
):
    # Specify the figure-generating functions and their arguments
    fig_specs: dict[str, FigFuncSpec] = dict(
        loss_history=FigFuncSpec(
            func=get_train_history_figures,
            args=(train_histories, replicate_info["best_saved_iteration_by_replicate"]),
        ),
        # loss_dist_over_replicates_best=FigFuncSpec(
        #     func=partial(
        #         get_replicate_distribution_figure,
        #         yaxis_title=f"Best batch total loss",
        #     ),
        #     args=(replicate_info['losses_at_best_saved_iteration'],),
        # ),
        # loss_dist_over_replicates_final=FigFuncSpec(
        #     func=partial(
        #         get_replicate_distribution_figure,
        #         yaxis_title=f"Final batch total loss",
        #     ),
        #     args=(replicate_info['losses_at_final_saved_iteration'],),
        # ),
        # readout_norm=FigFuncSpec(
        #     func=partial(
        #         get_replicate_distribution_figure,
        #         yaxis_title=f"Frobenius norm of readout weights",
        #     ),
        #     args=(replicate_info['readout_norm'],),
        # ),
    )

    # TODO
    # DON'T Evaluate all of them at the "train__pert__std" level
    all_figs = {
        fig_label: jt.map(
            fig_spec.func,
            *fig_spec.args,
            is_leaf=LDict.is_of("train__pert__std"),
        )
        for fig_label, fig_spec in fig_specs.items()
    }

    def save_and_add_figure(fig, plot_id, variant_label, train_std):
        fig_parameters = dict()

        if variant_label:
            fig_parameters |= dict(variant_label=variant_label)

        if train_std:
            fig_parameters |= dict(train__pert__std=float(train_std))

        add_evaluation_figure(
            db_session,
            eval_info,
            fig,
            plot_id,
            # TODO: let the user specify which formats to save
            save_formats=["png"],
            **fig_parameters,
        )

    is_leaf = anyf(LDict.is_of("train__pert__std"), is_type(go.Figure))

    # Save and add records for each figure
    # TODO: BUT DON'T MAP OVER "train__pert__std"
    for plot_id, figs in all_figs.items():
        # Some training notebooks use multiple training methods, and some don't. And some figure functions
        # return one figure per training condition, while others are summaries. Thus we need to descend
        # to the "train__pert__std" or `go.Figure` level first, and whatever the label is down to that level, will
        # label the training method (variant). Then we can descend to the `go.Figure` level, and whatever
        # label is constructed here will either be the training std (if we originally descended to "train__pert__std"),
        # or nothing.
        jt.map(
            # Map over each set (i.e. training variant) of disturbance train stds
            lambda fig_set, variant_label: jt.map(
                lambda fig, train_std: save_and_add_figure(fig, plot_id, variant_label, train_std),
                # TODO: WHY is jtree.labels here?
                fig_set,
                jtree.labels(fig_set, join_with="_", is_leaf=is_type(go.Figure)),
                is_leaf=is_type(go.Figure),
            ),
            figs,
            jtree.labels(figs, join_with="_", is_leaf=is_leaf),
            is_leaf=is_leaf,
        )

        logger.info(f"Saved {plot_id} figure set")


def compute_replicate_info(
    model_record,
    models,
    tasks,
    train_histories,
    save_model_parameters,
    n_replicates,
    n_std_exclude,
    where_train,
    hps,
):
    best_save_idx, best_saved_iterations, losses_at_best_saved_iteration = (
        get_best_iterations_and_losses(train_histories, save_model_parameters, n_replicates)
    )

    # Rate the best total loss, but also some other measures
    measures = dict(
        best_total_loss=losses_at_best_saved_iteration,
        **get_measures_to_rate(models, tasks, hps),
    )

    best_replicates, included_replicates = jtree.unzip(
        jt.map(
            partial(get_best_and_included, n_std_exclude=n_std_exclude),
            measures,
        )
    )

    losses_at_final_saved_iteration = jt.map(
        lambda history: history.loss.total[-1],
        train_histories,
        is_leaf=is_module,
    )

    # Create models with best parameters
    best_models = get_best_models(
        model_record,
        models,
        train_histories,
        save_model_parameters,
        best_save_idx,
        n_replicates,
        where_train,
    )

    readout_norm = jt.map(
        lambda model: jnp.linalg.norm(model.step.net.readout.weight, axis=(-2, -1), ord="fro"),
        best_models,
        is_leaf=is_module,
    )

    replicate_info = dict(
        best_save_idx=best_save_idx,
        best_saved_iteration_by_replicate=best_saved_iterations,
        losses_at_best_saved_iteration=losses_at_best_saved_iteration,
        losses_at_final_saved_iteration=losses_at_final_saved_iteration,
        best_replicates=best_replicates,
        included_replicates=included_replicates,
        readout_norm=readout_norm,
    )

    return replicate_info, best_models


def process_model_post_training(
    session: Session,
    model_record: ModelRecord,
    n_std_exclude: float,
    process_all: bool = True,
    save_figures: bool = True,
) -> ModelRecord | None:
    """Process a single model record, adding a new record with best parameters and replicate info."""

    if model_record.has_replicate_info or (model_record.postprocessed and not process_all):
        logger.info(
            f"Model {model_record.hash[:7]} has been processed previously and process_all is false; skipping"
        )
        return

    expt_name = str(model_record.expt_name)

    where_train = jt.map(
        attr_str_tree_to_where_func,
        model_record.where,
        is_leaf=is_type(list),
    )
    # where_train = attr_str_tree_to_where_func(tuple(set(jt.leaves(model_record.where))))
    n_replicates = int(model_record.model__n_replicates)
    save_model_parameters = jnp.array(model_record.save_model_parameters)

    all_data = load_data(model_record)

    #! `tasks` is just a single task, and `models` just a single model;
    #! since each `model_record` is now associated with a single model-task pair
    #! TODO: Simply what follows, including the functions called.
    if all_data is None:
        return
    else:
        (
            models,
            hps,
            train_histories,
            train_history_hyperparams,
        ) = all_data

    # `vmap_eval_ensemble` will look for this when we try to evaluate the states, however it is not
    # part of the model hyperparameters
    hps.eval_n = N_TRIALS_VAL

    training_module = EXPERIMENT_REGISTRY.get_training_module(expt_name)

    # Get respective validation tasks for each model
    tasks = setup_tasks_only(
        training_module.setup_task_model_pair,
        hps,
        key=jr.PRNGKey(0),
    )

    # Compute replicate info
    #! TODO: Don't pass things that can be obtained from `hps`: `where_train`, `n_replicates`, `save_model_parameters`
    replicate_info, best_models = compute_replicate_info(
        model_record,
        models,
        tasks,
        train_histories,
        save_model_parameters,
        n_replicates,
        n_std_exclude,
        where_train,
        hps,
    )

    try:
        # Save new model file with best parameters and get new record
        record_hyperparameters = {
            key: getattr(model_record, key)
            for key in model_record.__table__.columns.keys()
            if key not in MODEL_RECORD_BASE_ATTRS
        }

        new_record = save_model_and_add_record(
            session,
            best_models,
            hps | dict(n_std_exclude=n_std_exclude, postprocessed=True),
            train_history=train_histories,
            replicate_info=replicate_info,
            # Assume we want to keep the version info from training, not post-training
            version_info=model_record.version_info,
        )

    except Exception as e:
        # If anything fails, rollback and restore original record
        session.rollback()
        logger.error(f"Failed to process model {model_record.hash}: {e}")
        raise

    # ? Do we really need to make one of these, here? Or should training figures have their own table?
    eval_info = add_evaluation(
        session,
        models=model_record,
        eval_parameters=dict(
            n_evals=N_TRIALS_VAL,
            # n_std_exclude=n_std_exclude,  # Not relevant to the figures that are generated?
        ),
        expt_name=f"{model_record.expt_name}__post_training",
    )

    if save_figures:
        # Save training figures
        save_training_figures(
            session,
            eval_info,
            train_histories,
            replicate_info,
        )

    session.commit()
    logger.info(f"Post-training processing finished for model {model_record.hash}")
    return new_record


def main(
    n_std_exclude: float = 2.0,
    process_all: bool = False,
    save_figures: bool = True,
):
    """Process all models in database."""
    session = get_db_session()

    # Get all model records
    check_model_files(session)  # Mark any records with missing model files
    model_records = query_model_records(session)
    logger.info(f"Found {len(model_records)} model records")

    with Progress() as progress:
        task = progress.add_task("Processing...", total=len(model_records))
        for model_record in model_records:
            try:
                process_model_post_training(
                    session,
                    model_record,
                    n_std_exclude,
                    process_all=process_all,
                    save_figures=save_figures,
                )
                progress.update(task, advance=1)
            except Exception as e:
                logger.error(f"Skipping model {model_record.hash} due to error: {e}")
                raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-training processing of models.")
    parser.add_argument(
        "--n_std_exclude",
        default=2,
        type=float,
        help="Mark replicates this many stds above the best as to-be-excluded",
    )
    parser.add_argument(
        "--process_all",
        action="store_true",
        help="Reprocess all models, even if they already have replicate info",
    )
    parser.add_argument("--no_figs", action="store_true", help="Do not save training figures")
    args = parser.parse_args()

    main(
        args.n_std_exclude,
        args.process_all,
        not args.no_figs,
    )
