from functools import partial

import jax.numpy as jnp
import jax.tree as jt
import jax_cookbook.tree as jtree
from feedbax.intervene import add_intervenors, schedule_intervenor
from jax_cookbook import is_module, is_type

from rlrmp.analysis.aligned import (
    ALL_MEASURES,
    DEFAULT_VARSET,
    MEASURE_LABELS,
    VAR_LEVEL_LABEL,
    AlignedVars,
    get_aligned_trajectories_node,
)
from rlrmp.analysis.analysis import FigIterCtx
from rlrmp.analysis.disturbance import PLANT_INTERVENOR_LABEL, PLANT_PERT_FUNCS
from rlrmp.analysis.effector import EffectorTrajectories
from rlrmp.analysis.func import ApplyFuncs
from rlrmp.analysis.profiles import Profiles
from rlrmp.analysis.state_utils import get_best_replicate, vmap_eval_ensemble
from rlrmp.analysis.violins import Violins
from rlrmp.plot import (
    get_violins,
    set_axes_bounds_equal,
    set_axes_bounds_equal_traj2D,
    set_axis_bounds_equal,
)
from rlrmp.tree_utils import lohi, subdict
from rlrmp.types import LDict

COLOR_FUNCS = dict()


def setup_eval_tasks_and_models(task_base, models_base, hps):
    try:
        disturbance = PLANT_PERT_FUNCS[hps.pert.type]
    except KeyError:
        raise ValueError(f"Unknown perturbation type: {hps.pert.type}")

    # Insert the disturbance field component into each model
    models = jt.map(
        lambda models: add_intervenors(
            models,
            lambda model: model.step.mechanics,
            # The first key is the model stage where to insert the disturbance field;
            # `None` means prior to the first stage.
            # The field parameters will come from the task, so use an amplitude 0.0 placeholder.
            {None: {PLANT_INTERVENOR_LABEL: disturbance(0.0)}},
        ),
        models_base,
        is_leaf=is_module,
    )

    # Assume a sequence of amplitudes is provided, as in the default config
    pert_amps = hps.pert.amp
    # Construct tasks with different amplitudes of disturbance field
    all_tasks, all_models = jtree.unzip(
        jt.map(
            lambda pert_amp: schedule_intervenor(
                task_base,
                models,
                lambda model: model.step.mechanics,
                disturbance(pert_amp),
                label=PLANT_INTERVENOR_LABEL,
                default_active=False,
            ),
            LDict.of("pert__amp")(dict(zip(pert_amps, pert_amps))),
        )
    )

    all_hps = jt.map(lambda _: hps, all_tasks, is_leaf=is_module)

    return all_tasks, all_models, all_hps, None


# We aren't vmapping over any other variables, so this is trivial.
eval_func = vmap_eval_ensemble


"""Labels of measures to include in the analysis."""
MEASURE_KEYS = (
    "max_parallel_vel_forward",
    "max_lateral_vel_left",
    "max_lateral_vel_right",
    "max_lateral_distance_left",
    "sum_lateral_distance",
    "end_position_error",
    # "end_velocity_error",
    "max_parallel_force_forward",
    "sum_parallel_force",
    "max_lateral_force_right",
    "sum_lateral_force_abs",
    "max_net_force",
    "sum_net_force",
)


MEASURE_FUNCS = subdict(ALL_MEASURES, MEASURE_KEYS)


i_eval = 0  # For single-eval plots


DEPENDENCIES = {
    "measures": (
        ApplyFuncs(
            funcs=MEASURE_FUNCS,
            inputs=ApplyFuncs.Ports(input=AlignedVars()),
            is_leaf=LDict.is_of(VAR_LEVEL_LABEL),
        )
        # Discard the varset; only keep the aligned vars
        .after_transform(lambda results: results["full"], dependency_names="input")
    )
}


def measure_violin_params_fn(fig_params, ctx: FigIterCtx):
    return fig_params | dict(
        yaxis_title=MEASURE_LABELS[ctx.key],
    )


measure_violins_base = Violins(inputs=Violins.Ports(input="measures")).map_figs_at_level(
    "measure",
    dependency_name="input",
    fig_params_fn=measure_violin_params_fn,
)


# PyTree levels:
# State batch shape: (eval, replicate, condition)
ANALYSES = {
    "effector_trajectories_by_condition": (
        # By condition, all evals for the best replicate only
        EffectorTrajectories(
            colorscale_axis=1,
            colorscale_key="reach_condition",
        )
        .after_transform(get_best_replicate)
        .then_transform_figs(
            partial(set_axis_bounds_equal, "y", padding_factor=0.1),
        )
        # .with_fig_params()
    ),
    "effector_trajectories_by_replicate": (
        # By replicate, single eval
        EffectorTrajectories(
            colorscale_axis=0,
            colorscale_key="replicate",
        )
        .after_indexing(0, i_eval, axis_label="eval")
        .with_fig_params(
            scatter_kws=dict(line_width=1),
        )
    ),
    "effector_trajectories_single": (
        # Single eval for a single replicate
        EffectorTrajectories(
            colorscale_axis=0,
            colorscale_key="reach_condition",
        )
        .after_transform(get_best_replicate)
        .after_indexing(0, i_eval, axis_label="eval")
        .with_fig_params(
            curves_mode="markers+lines",
            ms=3,
            scatter_kws=dict(line_width=0.75),
            mean_scatter_kws=dict(line_width=0),
        )
    ),
    "plot--aligned_trajectories-by_pert_amp": (
        get_aligned_trajectories_node(colorscale_key="pert__amp")
        .after_transform(get_best_replicate)
        .after_getitem_at_level("task_variant", "small")
        .then_transform_figs(set_axes_bounds_equal_traj2D)
    ),
    "aligned_trajectories_by_train_std": (
        get_aligned_trajectories_node(colorscale_key="train__pert__std")
        .after_transform(get_best_replicate)
        .after_getitem_at_level("task_variant", "small")
        .then_transform_figs(set_axes_bounds_equal_traj2D)
    ),
    "profiles": (
        Profiles(varset=DEFAULT_VARSET)
        .after_transform(get_best_replicate)
        .after_level_to_bottom("train__pert__std", dependency_name="vars")
        # .after_transform(
        #     lambda tree, **kws: move_ldict_level_above("var", "train__pert__std", tree),
        #     dependency_names="vars",
        # )
    ),
    "plot--measures": measure_violins_base,
    "plot--measures_lohi_train_std": (
        measure_violins_base.after_transform(lohi, level="train__pert__std")
    ),
    "plot--measures_lohi_train_std_and_pert_amp": (
        measure_violins_base.after_transform(lohi, level=["train__pert__std", "pert__amp"])
    ),
}
