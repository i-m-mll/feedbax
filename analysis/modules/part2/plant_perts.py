from collections import namedtuple
from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import jax.tree as jt
import jax_cookbook.tree as jtree
import numpy as np
from feedbax.intervene import add_intervenors, schedule_intervenor
from jax_cookbook import MultiVmapAxes, is_module, is_type
from jaxtyping import Array, Float
from lark import Tree

from rlrmp.analysis import tangling
from rlrmp.analysis.aligned import (
    ALL_MEASURES,
    DEFAULT_VARSET,
    MEASURE_LABELS,
    VAR_LEVEL_LABEL,
    AlignedVars,
    add_aligned_position_endpoints,
    get_aligned_trajectories_node,
    get_varset_labels,
)
from rlrmp.analysis.analysis import CallWithDeps, Data, ExpandTo, FigIterCtx
from rlrmp.analysis.disturbance import PLANT_INTERVENOR_LABEL, PLANT_PERT_FUNCS
from rlrmp.analysis.eig import eig, svd
from rlrmp.analysis.func import ApplyFuncs, ApplyFunctional, make_argwise_functional
from rlrmp.analysis.grad import Hessians, Jacobians, PerJacobianBlock, spectral_norm
from rlrmp.analysis.pca import PCAResults, StatesPCA
from rlrmp.analysis.plot import ScatterPlots
from rlrmp.analysis.profiles import Profiles
from rlrmp.analysis.state_utils import (
    get_best_replicate,
    get_constant_task_input_fn,
    vmap_eval_ensemble,
)
from rlrmp.analysis.tangling import Tangling
from rlrmp.analysis.violins import Violins
from rlrmp.colors import ColorscaleSpec
from rlrmp.misc import rms
from rlrmp.plot import set_axes_bounds_equal, set_axes_bounds_equal_traj2D, set_axis_bounds_equal
from rlrmp.tree_utils import getitem_at_level, ldict_level_to_top, subdict
from rlrmp.types import (
    LDict,
    TreeNamespace,
)

N_PCA = 10
PCA_START_STEP = 0
PCA_END_STEP = 100
TANGLING_AGG_T_SLICE = slice(1, None)
GRADS_TIMESTEP_STRIDE = 5


COLOR_FUNCS = dict(
    sisu=ColorscaleSpec(
        sequence_func=lambda hps: hps.sisu,
        colorscale="thermal",
    ),
)


eval_func = vmap_eval_ensemble


def setup_eval_tasks_and_models(task_base, models_base, hps):
    try:
        disturbance = PLANT_PERT_FUNCS[hps.pert.type]
    except KeyError:
        raise ValueError(f"Unknown disturbance type: {hps.pert.type}")

    pert_amps = hps.pert.amp

    # Tasks with varying plant perturbation amplitude
    tasks_by_amp, _ = jtree.unzip(
        jt.map(  # over disturbance amplitudes
            lambda pert_amp: schedule_intervenor(  # (implicitly) over train stds
                task_base,
                jt.leaves(models_base, is_leaf=is_module)[0],
                lambda model: model.step.mechanics,
                disturbance(pert_amp),
                label=PLANT_INTERVENOR_LABEL,
                default_active=False,
            ),
            LDict.of("pert__amp")(
                dict(zip(pert_amps, pert_amps)),
            ),
        )
    )

    # Add plant perturbation module (placeholder with amp 0.0) to all loaded models
    models_by_std = jt.map(
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

    # Also vary tasks by SISU
    tasks = LDict.of("sisu")(
        {
            sisu: jt.map(
                lambda task: task.add_input(
                    name="sisu",
                    input_fn=get_constant_task_input_fn(
                        sisu,
                        hps.model.n_steps - 1,
                        task.n_validation_trials,
                    ),
                ),
                tasks_by_amp,
                is_leaf=is_module,
            )
            for sisu in hps.sisu
        }
    )

    # The outer levels of `models` have to match those of `tasks`
    models, hps = jtree.unzip(jt.map(lambda _: (models_by_std, hps), tasks, is_leaf=is_module))

    return tasks, models, hps, None


MEASURE_KEYS = (
    "max_parallel_vel_forward",
    # "max_lateral_vel_signed",
    # "max_lateral_vel_left",
    # "max_lateral_vel_right",  # -2
    "largest_lateral_distance",
    # "max_lateral_distance_left",
    # "sum_lateral_distance",
    "sum_lateral_distance_abs",
    "end_position_error",
    # "end_velocity_error",  # -1
    "max_parallel_force_forward",
    # "sum_parallel_force",  # -2
    # "max_lateral_force_right",  # -1
    "sum_lateral_force_abs",
    "max_net_force",
    "sum_net_force",
)


measure_funcs = subdict(ALL_MEASURES, MEASURE_KEYS)
measure_labels = MEASURE_LABELS

# this has a single batch axis for replicates
rnn_funcs = Data.models(where=lambda model: model.step.net.hidden)
# these have batch shape (evals, replicates, reach conditions, time)
rnn_inputs = Data.states(where=lambda states: states.net.input)
rnn_states = Data.states(where=lambda states: states.net.hidden)


def measure_violin_params_fn(fig_params, ctx: FigIterCtx):
    return fig_params | dict(
        yaxis_title=measure_labels[ctx.key],
    )


measure_violins_base = Violins(inputs=Violins.Ports(input="measures")).map_figs_at_level(
    "measure",
    dependency_name="input",
    fig_params_fn=measure_violin_params_fn,
)


def measure_violins(group_var: str, x_var: str):
    return measure_violins_base.after_rearrange_levels(
        [..., group_var, x_var],
        dependency_name="input",
    )


def get_state_pcs(pca_results, states):
    return jt.map(
        lambda pca_results_by_std, states_by_std: LDict.of("train__pert__std")(
            {
                std: pca_results_by_std[std].batch_transform(states_by_std[std])
                for std in pca_results_by_std
            }
        ),
        pca_results,
        ldict_level_to_top("train__pert__std", states, is_leaf=is_module),
        is_leaf=LDict.is_of("train__pert__std"),
    )


DEPENDENCIES = {
    # "measures": (
    #     ApplyFuncs(
    #         funcs=measure_funcs,
    #         inputs=ApplyFuncs.Ports(input=AlignedVars()),
    #         is_leaf=LDict.is_of(VAR_LEVEL_LABEL),
    #     )
    #     # Discard the varset; only keep the aligned vars
    #     .after_transform(lambda results: results['full'], dependency_names="input")
    # ),
    # "hidden_states_pca": (
    #     StatesPCA(
    #         n_components=N_PCA,
    #         where_states=lambda states: states.net.hidden,
    #         aggregate_over_labels=('pert__amp', 'sisu')
    #     )
    #     .after_transform(get_best_replicate)
    #     .after_transform(lambda x: x['full'])
    #     .after_indexing(-2, np.arange(PCA_START_STEP, PCA_END_STEP), axis_label="timestep")
    # ),
    # "tangling": (
    #     Tangling(
    #         variant="small",
    #         inputs=Tangling.Ports(
    #             state=Data.states(where=lambda states: states.net.hidden),
    #         ),
    #     )
    #     .after_transform(get_best_replicate)
    #     .after_transform(
    #         # Pull in the PCA results and use them to transform the hidden states
    #         CallWithDeps("hidden_states_pca")(get_state_pcs),
    #         dependency_names="state",
    #     )
    # ),
    # **{  # "jacobians" and "hessians"
    #     cls.__name__.lower(): (
    #         cls(
    #             inputs=cls.Ports(
    #                 funcs=rnn_funcs,
    #                 func_args=(rnn_inputs, rnn_states),
    #             ),
    #         )
    #         .after_transform_inputs(partial(getitem_at_level, "task_variant", "small"))
    #         .after_subdict_at_level("sisu", [-3, 0, 1, 3])
    #         .after_subdict_at_level("train__pert__std", [0, 1.5])
    #         .after_indexing(
    #             -2,
    #             lambda shape: jnp.arange(0, shape[-2], GRADS_TIMESTEP_STRIDE),
    #             axis_label="timestep",
    #             dependency_name="func_args",
    #         )
    #         .vmap(in_axes={
    #             'funcs': MultiVmapAxes(None, 0, None, None),
    #             'func_args': MultiVmapAxes(0, 1, 2, 3),
    #         })
    #         .estimate_memory()
    #     )
    #     for cls in (Jacobians, Hessians)
    # },
    # "effector_trajectories_by_condition": (
    #     # By condition, all evals for the best replicate only
    #     EffectorTrajectories(
    #         colorscale_axis=1,
    #         colorscale_key="reach_condition",
    #     )
    #     .after_transform(get_best_replicate)  # By default has `axis=1` for replicates
    # ),
}


tangling_violins = (
    Violins(
        inputs=Violins.Ports(input="tangling"),
    )
    .with_fig_params(
        yaxis_title="RMS tangling",
        violinmode="group",
    )
    .after_transform(
        lambda tangling: jt.map(lambda x: rms(x[..., TANGLING_AGG_T_SLICE]), tangling),
        dependency_names="input",
    )
    .then_transform_figs(partial(set_axis_bounds_equal, "y"), level="train__pert__std")
)


GradArgs = namedtuple("GradArgs", ["input", "state"])


RNN_INPUT_CHANNELS = dict(sisu=1, goal_pos=2, goal_vel=2, fb_pos=2, fb_vel=2)
RNNInputs = namedtuple("RNNInputs", RNN_INPUT_CHANNELS.keys())


def split_by(x, sizes, axis=0):
    return jnp.split(x, np.cumsum(list(sizes))[:-1], axis=axis)


def jac_u_reducer(jac_u: Array):
    """Compute desired functions of the input Jacobian."""
    # It might be necessary to perform this analysis separately for different components of the
    # input; e.g. position and velocity do not have comparable units/scaling, so one of them
    # may dominate any computed norms. Instead, we should compare them individually across
    # conditions.
    split_jac = RNNInputs(*split_by(jac_u, RNN_INPUT_CHANNELS.values()))

    def _get_measures(jac_part):
        part_svd = svd(jac_part)
        max_singval_idx = jnp.argmax(part_svd.vals, axis=0)
        max_sing = jtree.take(part_svd, max_singval_idx)
        return TreeNamespace(
            max_sing=max_sing,
        )

    # Also: Multiply C = WJ_u, where W is the readout
    # Then we can SVD of C (split into input types) to examine the input-output gains.
    # However:
    # 1. Without scaling, it only makes sense to compare *changes*, e.g. "between these conditions,
    #    did the input-output gain wrt position increase more than wrt velocity?"
    # 2. If we want to ask questions like "is position or velocity more influential on the output,
    #    in this condition?" then we need to normalize C by the standard deviations of the inputs.
    #    We also should do such normalization if we want a single number describing how sensitive
    #    the output is to the inputs altogether, e.g. the spectral norm of full C.
    # 3. Technically, we do not have access to the readout here. But also we do not want to
    #    keep the full Jacobians and do the analysis later. So how can we pull in the readouts?

    return jt.map(_get_measures, split_jac)


def jac_x_reducer(x: Array):
    """Compute desired functions of the state Jacobian."""
    jac_x_eig = eig(x)
    spectral_radius = jnp.max(jnp.abs(jac_x_eig.vals), axis=0)

    # others:
    # 1. dispersion of eigenvalue angle
    # 2. difference between largest and 90th percentile eigenvalue radius (to get a sense of fatness
    #    of tails)
    # 3. mean/SD of log of eigenvalue radius (log -> multiplicative growth/decay per step)
    # 4. finally, keep the eigenvalues but not the eigenvectors (space limitations)

    return TreeNamespace(
        spectral_radius=spectral_radius,
        # jac_x_eig=jac_x_eig,
    )


# State PyTree structure: ['sisu', 'pert__amp', 'train__pert__std']
ANALYSES = {
    # "jac_functions": (
    #     ApplyFunctional(
    #         inputs=ApplyFunctional.Ports(
    #             funcs=rnn_funcs,
    #             func_args=GradArgs(rnn_inputs, rnn_states),
    #         ),
    #         functional=make_argwise_functional(
    #             per=PerJacobianBlock(reducer=GradArgs(
    #                 input=jac_u_reducer,
    #                 state=jac_x_reducer,
    #             )),
    #         ),
    #     )
    #     .after_transform_inputs(partial(getitem_at_level, "task_variant", "small"))
    #     .after_subdict_at_level("sisu", [-3, 0, 1, 3])
    #     .after_subdict_at_level("train__pert__std", [0, 1.5])
    #     .vmap(in_axes={
    #         'funcs': MultiVmapAxes(None, 0, None, None),
    #         'func_args': MultiVmapAxes(0, 1, 2, 3),
    #     })
    #     .estimate_memory()
    # ),
    # "plot--tangling-by_train_std": tangling_violins.after_rearrange_levels(
    #     [..., 'sisu', 'pert__amp'], dependency_name="input"
    # ),
    # "plot--tangling-by_pert_amp": tangling_violins.after_rearrange_levels(
    #     [..., 'sisu', 'train__pert__std'], dependency_name="input"
    # ),
    "plot--aligned_trajectories-by_sisu": (
        get_aligned_trajectories_node(colorscale_key="sisu")
        .after_getitem_at_level("task_variant", "small")
        .map_figs_at_level("train__pert__std", dependency_name="input")
        .then_transform_figs(set_axes_bounds_equal_traj2D)
    ),
    "plot--aligned_trajectories-by_train_std": (
        get_aligned_trajectories_node(colorscale_key="train__pert__std")
        .after_getitem_at_level("task_variant", "small")
        .map_figs_at_level("sisu", dependency_name="input")
        .then_transform_figs(set_axes_bounds_equal_traj2D)
    ),
    # "plot--profiles-by_train_std": (
    #     Profiles(varset=DEFAULT_VARSET)
    #     .after_transform(get_best_replicate)
    #     .after_level_to_bottom('train__pert__std', dependency_name="vars")
    # ),
    # "plot--profiles-by_sisu": (
    #     Profiles(varset=DEFAULT_VARSET)
    #     .after_transform(get_best_replicate)
    #     .after_level_to_bottom('sisu', dependency_name="vars")
    # ),
    # "plot--measures-by_pert_amp": measure_violins("sisu", "train__pert__std"),
    # "plot--measures-by_train_std": measure_violins("sisu", "pert__amp"),
    # "plot--measures-train_std_by_pert_amp": measure_violins("train__pert__std", "sisu"),
}
