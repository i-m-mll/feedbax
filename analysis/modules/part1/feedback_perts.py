from collections.abc import Callable
from types import MappingProxyType, SimpleNamespace
from typing import ClassVar, Literal, Optional
import jax.numpy as jnp
import jax.tree as jt 

import equinox as eqx
from jax_cookbook import is_type, is_module, is_none
import jax_cookbook.tree as jtree

from feedbax.intervene import schedule_intervenor
import feedbax.plotly as fbp


# from rlrmp.analysis import measures
from rlrmp.analysis import AbstractAnalysis
from rlrmp.analysis.aligned import ALL_MEASURES, MEASURE_LABELS, VAR_LEVEL_LABEL, AlignedVars
from rlrmp.analysis.analysis import FigIterCtx
from rlrmp.analysis.disturbance import FB_INTERVENOR_LABEL, get_pert_amp_vmap_eval_func, task_with_pert_amp
from rlrmp.analysis.func import ApplyFuncs
from rlrmp.analysis.state_utils import vmap_eval_ensemble
from rlrmp.analysis.violins import Violins
from rlrmp.tree_utils import subdict
from rlrmp.types import AnalysisInputData, LDict, unflatten_dict_keys
from rlrmp.perturbations import feedback_impulse


COLOR_FUNCS = dict()


#! TODO: Move; these are redundant with 2-2
PERT_VAR_NAMES = ('fb_pos', 'fb_vel')
COORD_NAMES = ('x', 'y')
I_IMPULSE_AMP_PLOT = -1  # The largest amplitude perturbation

components_plot: Literal['xy', 'aligned'] = 'aligned'
components_labels = dict(
    xy=COORD_NAMES,
    aligned=(r'\parallel', r'\bot')
)
components_names = dict(
    xy=COORD_NAMES,
    aligned=('parallel', 'lateral'),
)


def _setup_rand(task_base, models_base, hps):
    """Impulses in random directions, i.e. uniform angles about the effector."""
    all_tasks, all_models = jtree.unzip(jt.map(
        lambda feedback_var_idx: schedule_intervenor(
            task_base, models_base,
            lambda model: model.step.feedback_channels[0],  # type: ignore
            feedback_impulse(  
                hps.model.n_steps,
                1.0,  # Will be varied later
                hps.pert.duration,
                feedback_var_idx,   
                hps.pert.start_step,
            ),
            default_active=False,
            stage_name="update_queue",
            label=FB_INTERVENOR_LABEL,
        ),
        LDict.of("pert__var")(dict(fb_pos=0, fb_vel=1)),
        is_leaf=is_type(tuple),
    ))

    # Get the perturbation directions, for later:
    #? I think these values are equivalent to `line_vec` in the functions in `state_utils`
    impulse_directions = jt.map(
        lambda task: task.validation_trials.intervene[FB_INTERVENOR_LABEL].arrays[:, hps.pert.start_step],
        all_tasks,
        is_leaf=is_module,
    )
    return all_tasks, all_models, impulse_directions


def _setup_xy(task_base, models_base, hps):
    """Impulses only in the x and y directions."""
    feedback_var_idxs = LDict.of("pert__var")(
        dict(zip(PERT_VAR_NAMES, range(len(PERT_VAR_NAMES))))
    )
    coord_idxs = dict(zip(COORD_NAMES, range(len(COORD_NAMES))))
    
    impulse_xy_conditions = LDict.of("pert__var").fromkeys(PERT_VAR_NAMES, dict.fromkeys(COORD_NAMES))
    impulse_xy_conditions_keys = jtree.key_tuples(
        impulse_xy_conditions, keys_to_strs=True, is_leaf=is_none,
    )

    all_tasks, all_models = jtree.unzip(jt.map(
        lambda ks: schedule_intervenor(
            task_base, models_base,
            lambda model: model.step.feedback_channels[0],  # type: ignore
            feedback_impulse(
                hps.model.n_steps,
                1.0, 
                hps.pert.duration,
                feedback_var_idxs[ks[0]],  
                hps.pert.start_step,
                feedback_dim=coord_idxs[ks[1]],  
            ),
            default_active=False,
            stage_name="update_queue",
        ),
        impulse_xy_conditions_keys,
        is_leaf=is_type(tuple),
    ))

    impulse_directions = jt.map(
        lambda task, ks: jnp.zeros(
            (task.n_validation_trials, 2)
        # ).at[:, coord_idxs[ks[1]]].set(copysign(1, impulse_amplitude[ks[0]])),
        # Assume x-y impulses are in the positive direction.
        ).at[:, coord_idxs[ks[1]]].set(1),
        all_tasks, impulse_xy_conditions_keys,
        is_leaf=is_module,
    )
    
    return all_tasks, all_models, impulse_directions


SETUP_FUNCS_BY_DIRECTION = dict(
    rand=_setup_rand,
    xy=_setup_xy,
)


def setup_eval_tasks_and_models(task_base, models_base, hps):

    impulse_end_step = hps.pert.start_step + hps.pert.duration
    impulse_time_idxs = slice(hps.pert.start_step, impulse_end_step)


    all_tasks, all_models, impulse_directions = SETUP_FUNCS_BY_DIRECTION[hps.pert.direction](
        task_base, models_base, hps
    )
    
    impulse_amplitudes = jt.map(
        lambda max_amp: jnp.linspace(0, max_amp, hps.pert.n_amps + 1)[1:],
        LDict.of("pert__var").from_ns(hps.pert.amp_max),
    )

    # For the example trajectories and aligned profiles, we'll only plot one of the impulse amplitudes. 
    impulse_amplitude_plot = {
        pert_var: v[I_IMPULSE_AMP_PLOT] for pert_var, v in impulse_amplitudes.items()
    }
    
    all_hps = jt.map(
        lambda amps: hps | unflatten_dict_keys(dict(pert__amps=amps)), 
        impulse_amplitudes,
    )
    
    extras = SimpleNamespace(
        impulse_directions=impulse_directions,
        impulse_time_idxs=impulse_time_idxs,
        impulse_amplitude_plot=impulse_amplitude_plot,
    )
    
    return all_tasks, all_models, all_hps, extras

    
eval_func = get_pert_amp_vmap_eval_func(lambda hps: hps.pert.amps, FB_INTERVENOR_LABEL)   


MEASURE_KEYS = [
    "max_net_force",
    "max_parallel_force_reverse",
    "sum_net_force",
    "max_parallel_vel_forward",
    "max_parallel_vel_reverse",
    "max_lateral_vel_left",
    "max_lateral_vel_right",
    "max_deviation",
    "sum_deviation",
]


MEASURE_FUNCS = subdict(ALL_MEASURES, MEASURE_KEYS)


def measure_violin_params_fn(fig_params, ctx: FigIterCtx):
    return fig_params | dict(
        yaxis_title=MEASURE_LABELS[ctx.key],
    )


DEPENDENCIES = {
    "measures": (
        ApplyFuncs(
            funcs=MEASURE_FUNCS,
            inputs=ApplyFuncs.Ports(input=AlignedVars()),
            is_leaf=LDict.is_of(VAR_LEVEL_LABEL),
        )
        # Discard the varset; only keep the aligned vars
        .after_transform(lambda results: results['full'], dependency_names="input")
    )    
}


ANALYSES = {
    "plot--measures": Violins(
        inputs=Violins.Ports(input="measures")
        .map_figs_at_level(
            "measure", 
            dependency_name="input", 
            fig_params_fn=measure_violin_params_fn,
        )
    )
    
    # "effector_single_eval": (
    #     Effector_SingleEval(
    #         variant="full",
    #         #! TODO: This doesn't result in the impulse amplitude *values* showing up in the legend!
    #         #! (could try to access `colorscale_key` from `hps`, in `Effector_SingleEval`)
    #         colorscale_key='pert__amp',
    #     ).with_fig_params(legend_title="Impulse amplitude")
    # ),
}