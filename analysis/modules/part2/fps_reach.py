"""
Analysis of fixed point (FP) structure of the network during reaching.

This analysis finds fixed points during simple reaching tasks, including:
1. Goal-goal fixed points (steady state)
2. Init-goal fixed points (initial state)
3. Fixed points along the trajectory during reaching
"""

from collections.abc import Callable, Sequence
from functools import partial
from types import MappingProxyType, SimpleNamespace
from typing import ClassVar, Optional

import equinox as eqx
from equinox import Module
import jax
import jax.numpy as jnp
import jax.tree as jt

from jax_cookbook import is_module, MultiVmapAxes
import jax_cookbook.tree as jtree
from jaxtyping import Array, PyTree
import numpy as np

from rlrmp.analysis import AbstractAnalysis
from rlrmp.analysis.analysis import _DummyAnalysis, Data, ExpandTo, LiteralInput
from rlrmp.analysis.fps import FixedPoints
from rlrmp.analysis.pca import StatesPCA
from rlrmp.misc import get_constant_input_fn
from rlrmp.analysis.state_utils import get_best_replicate, vmap_eval_ensemble
from rlrmp.tree_utils import take_replicate
from rlrmp.types import TreeNamespace
from rlrmp.types import LDict
# from rlrmp.analysis.fps_tmp2 import (
#     # ReachFPs,
#     ReachFPsInPCSpace,
#     ReachTrajectoriesInPCSpace,
#     ReachDirectionTrajectories,
# )
from rlrmp.analysis.disturbance import PLANT_INTERVENOR_LABEL, PLANT_PERT_FUNCS
from feedbax.intervene import add_intervenors, schedule_intervenor


"""Specify any additional colorscales needed for this analysis. 
These will be included in the `colors` kwarg passed to `AbstractAnalysis` methods
"""
COLOR_FUNCS: dict[str, Callable[[TreeNamespace], Sequence]] = dict(
)


def setup_eval_tasks_and_models(task_base: Module, models_base: LDict[float, Module], hps: TreeNamespace):
    """Set up tasks with plant perturbations and varying SISU."""
    try:
        disturbance = PLANT_PERT_FUNCS[hps.pert.type]
    except KeyError:
        raise ValueError(f"Unknown disturbance type: {hps.pert.type}")
    
    pert_amps = hps.pert.amp  # Using amp as perturbation amplitude for this analysis
    
    # Tasks with varying plant perturbation amplitude 
    tasks_by_amp, _ = jtree.unzip(jt.map( # over disturbance amplitudes
        lambda pert_amp: schedule_intervenor(  # (implicitly) over train stds
            task_base, jt.leaves(models_base, is_leaf=is_module)[0],
            lambda model: model.step.mechanics,
            disturbance(pert_amp),
            label=PLANT_INTERVENOR_LABEL,
            default_active=False,
        ),
        LDict.of("pert__amp")(
            dict(zip(pert_amps, pert_amps)),
        )
    ))
    
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
    tasks = LDict.of('sisu')({
        sisu: jt.map(
            lambda task: task.add_input(
                name="sisu",
                input_fn=get_constant_input_fn(
                    sisu, 
                    hps.model.n_steps, 
                    task.n_validation_trials,
                ),
            ),
            tasks_by_amp,
            is_leaf=is_module,
        )
        for sisu in hps.sisu
    })
    
    # The outer levels of `models` have to match those of `tasks`
    models, hps = jtree.unzip(jt.map(
        lambda _: (models_by_std, hps), tasks, is_leaf=is_module
    ))
    
    return tasks, models, hps, None


# Unlike `feedback_perts`, we don't need to vmap over impulse amplitude 
eval_func: Callable = vmap_eval_ensemble
 
# goals_pos = task.validation_trials.targets["mechanics.effector.pos"].value[:, -1]


N_PCA = 30
PCA_START_STEP = 50  # Start PCA from midway through trial
PCA_END_STEP = 100
# How many timesteps before and after the state whose FP will be computed,
# for which states will be taken as candidates for the FP.
CANDIDATE_TIMESTEP_RADIUS = 1


def call_rnn_func(rnn_func, inputs, h) -> Callable:
    """Get the RNN function for reaching tasks."""
    return rnn_func(inputs, h)


def extract_timestep_windows(
    candidates: jnp.ndarray,
    timestep_radius: int = CANDIDATE_TIMESTEP_RADIUS,
) -> jnp.ndarray:
    E, R, C, T, S = candidates.shape
    W = 2 * timestep_radius + 1

    # 1) pad the time axis so that windows at t=0 and t=T-1 still have full W length:
    pad_width = ((0,0), (0,0), (0,0), (timestep_radius, timestep_radius), (0,0))
    padded = jnp.pad(candidates, pad_width, mode='edge')  # shape (E, R, C, T+2*rad, S)

    # 2) for each t in [0..T-1], grab the slice padded[..., t:t+W, :]
    def window_at_t(t):
        start = (0, 0, 0, t, 0)
        size  = (E, R, C, W, S)
        return jax.lax.dynamic_slice(padded, start, size)  # → (E, R, C, W, S)
    
    # vmapped over t gives shape (T, E, R, C, W, S)
    windows = jax.vmap(window_at_t)(jnp.arange(T))
    windows = windows.transpose(1, 4, 2, 3, 0, 5)    # (E, W, R, C, T, S)
    return windows.reshape(E * W, R, C, T, S)


def prepare_candidates(candidates: PyTree[Array]) -> Array:
    """Get a small window of candidates around each timestep."""
    # in: (evals, replicates, reach conditions, timesteps, state dims)
    # out: (evals * timestep window, replicates, reach conditions, state dims)
    return jt.map(
        partial(extract_timestep_windows, timestep_radius=CANDIDATE_TIMESTEP_RADIUS),
        candidates,
    )
    

def prepare_candidates_simple(candidates: PyTree[Array]) -> Array:
    """Just use the full trajectory as candidates, every time."""
    E, R, C, T, S = candidates.shape
    candidates = candidates.transpose(0, 3, 1, 2, 4)  # (E, T, R, C, S)
    return candidates.reshape(E * T, R, C, S)
    
    
DEPENDENCIES = {
    "states_pca": (
        StatesPCA(n_components=N_PCA, where_states=lambda states: states.net.hidden)
        .after_transform(get_best_replicate)
        .after_indexing(-2, np.arange(PCA_START_STEP, PCA_END_STEP), axis_label="timestep")
    ),
}

rnn_funcs = Data.models(where=lambda model: model.step.net.hidden)
rnn_inputs = Data.states(where=lambda states: states.net.input)
rnn_states = Data.states(where=lambda states: states.net.hidden)

# State PyTree structure: ['sisu', 'pert_amp', 'train__pert__std']
# Array batch shape: (evals, replicates, reach conditions)
ANALYSES = {
    "reach_fp_results": (
        FixedPoints(
            inputs=FixedPoints.Ports(
                funcs=ExpandTo(
                    "func_args", 
                    LiteralInput(call_rnn_func),
                    where=lambda func_args: func_args[0],
                    is_leaf=is_module,
                ),
                func_args=(rnn_funcs, rnn_states),
                #! TODO: Check how the candidates were actually constructed in the notebook
                candidates=rnn_states,
            )
        )
        # .after_transform(prepare_candidates, dependency_names="candidates")
        .after_transform(prepare_candidates_simple, dependency_names="candidates")
        .vmap(in_axes={
            # (evals, replicate, condition, timestep)
            'func_args': (
                MultiVmapAxes(None, 0, None, None), 
                MultiVmapAxes(0, 1, 2, 3)
            ), 
            # 'candidates': MultiVmapAxes(None, 1, 2, 3),
            'candidates': MultiVmapAxes(None, 1, 2, None),
        })

    ) 
    # "plot--fps_PC": (),
    # "plot--hidden_and_fp_trajs_PC": (),
    # "plot--compare_sisu_trajs_PC": (),
    
    # "tangling": (),
    # "jacs": (),
    # "hessians": (),
    # # eig -> square matrices, svd -> non-square
    # "jacs-eig": (),
    # "jacs-svd": (),
    # "hessians-eig": (),
    # "hessians-svd": (),  

    # # Plot. Each of these should probably use the same subclass
    # # (i.e. PC traj plotting + color)
    # "plot--jacs_PC": (),
    # "plot--hessians_PC": (),
    # "plot--tangling_PC": (),
    
    # # Measures: global tangling, Lyapunov, ...
    # "measures": ()
}


