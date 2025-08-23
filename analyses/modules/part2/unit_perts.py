from ast import In
from collections.abc import Callable, Mapping
from functools import partial
from types import MappingProxyType
from typing import Any, Dict, Optional, Sequence
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
from equinox import Module, field
from feedbax.intervene import (
    NetworkConstantInput,
    TimeSeriesParam,
    add_intervenors,
    schedule_intervenor,
)
from feedbax.task import AbstractTask
from jax_cookbook import MultiVmapAxes, is_module, is_none, is_type
from jaxtyping import ArrayLike, PRNGKeyArray, PyTree

from feedbax_experiments.analysis.aligned import (
    DEFAULT_VARSET,
    VAR_LEVEL_LABEL,
    AlignedVars,
    get_trivial_reach_directions,
)
from feedbax_experiments.analysis.analysis import (
    AbstractAnalysis,
    AbstractAnalysisPorts,
    Data,
    FigIterCtx,
    InputOf,
    NoPorts,
    SinglePort,
)
from feedbax_experiments.analysis.disturbance import (
    PLANT_INTERVENOR_LABEL,
    PLANT_PERT_FUNCS,
    get_pert_amp_vmap_eval_func,
)
from feedbax_experiments.analysis.effector import EffectorTrajectories
from feedbax_experiments.analysis.network import UnitPreferences
from feedbax_experiments.analysis.profiles import Profiles
from feedbax_experiments.analysis.regression import Regression
from feedbax_experiments.analysis.state_utils import (
    get_best_replicate,
    get_constant_task_input_fn,
    get_segment_trials_func,
    get_symmetric_accel_decel_epochs,
    vmap_eval_ensemble,
)
from feedbax_experiments.colors import ColorscaleSpec
from feedbax_experiments.config import PLOTLY_CONFIG
from feedbax_experiments.constants import POS_ENDPOINTS_ALIGNED
from feedbax_experiments.misc import unit_circle_points
from feedbax_experiments.plot import add_endpoint_traces, get_violins, set_axes_bounds_equal
from feedbax_experiments.tree_utils import (
    first,
    ldict_level_keys,
    move_ldict_level_above,
    subdict,
    tree_level_labels,
)
from feedbax_experiments.types import (
    AnalysisInputData,
    LDict,
    TreeNamespace,
    VarSpec,
)

COLOR_FUNCS = dict(
    sisu=ColorscaleSpec(
        sequence_func=lambda hps: hps.sisu,
        colorscale="thermal",
    ),
    stim_amp=ColorscaleSpec(
        sequence_func=lambda hps: hps.pert.unit.amp,
        colorscale="viridis",
    ),
    pert__amp=ColorscaleSpec(
        sequence_func=lambda hps: hps.pert.plant.amp,
        colorscale="viridis",
    ),
)


UNIT_STIM_INTERVENOR_LABEL = "UnitStim"

SCALE_UNIT_STIM_BY_READOUT_VECTOR_LENGTH = False

# UNIT_STIM_IDX = 1

PLANT_PERT_LABELS = {0: "no curl", 1: "curl"}
PLANT_PERT_STYLES = dict(line_dash={0: "dot", 1: "solid"})

SISU_LABELS = {0: -2, 1: 0, 2: 2}
SISU_STYLES = dict(line_dash={0: "dot", 1: "dash", 2: "solid"})


def unit_stim(hps):
    idxs = slice(
        hps.pert.unit.start_step,
        hps.pert.unit.start_step + hps.pert.unit.duration,
    )
    trial_mask = jnp.zeros((hps.model.n_steps - 1,), bool).at[idxs].set(True)

    return NetworkConstantInput.with_params(
        # active=True,
        active=TimeSeriesParam(trial_mask),
    )


def schedule_unit_stim(*, tasks, models, hps):
    tasks, models = jtree.unzip(
        jt.map(
            lambda task, model, hps: schedule_intervenor(
                task,
                model,
                lambda model: model.step.net,
                # unit_stim(unit_idx, hps=hps),
                unit_stim(hps),
                default_active=False,
                stage_name=None,  # None -> before RNN forward pass; 'hidden' -> after
                label=UNIT_STIM_INTERVENOR_LABEL,
            ),
            tasks,
            models,
            hps,
            is_leaf=is_module,
        )
    )
    return tasks, models


def setup_eval_tasks_and_models(task_base, models_base, hps):
    try:
        disturbance = PLANT_PERT_FUNCS[hps.pert.plant.type]
    except KeyError:
        raise ValueError(f"Unknown disturbance type: {hps.pert.plant.type}")

    pert_amps = hps.pert.plant.amp

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

    # Schedule unit stim for a placeholder unit (0)
    tasks, models = schedule_unit_stim(tasks=tasks, models=models, hps=hps)

    return tasks, models, hps, None


def task_with_scaled_unit_stim(model, task, unit_idx, stim_amp_base, hidden_size, intervenor_label):
    """Scale the magnitude of unit stim based on the length of the unit's readout vector."""
    if SCALE_UNIT_STIM_BY_READOUT_VECTOR_LENGTH:
        readout_vector_length = jnp.linalg.norm(model.step.net.readout.weight[..., unit_idx])
        stim_amp = stim_amp_base / readout_vector_length
    else:
        stim_amp = stim_amp_base

    return eqx.tree_at(
        lambda task: (
            task.intervention_specs.validation[intervenor_label].intervenor.params.scale,
            task.intervention_specs.validation[intervenor_label].intervenor.params.unit_spec,
        ),
        task,
        (
            stim_amp,
            jnp.full(hidden_size, jnp.nan).at[unit_idx].set(1.0),
        ),
        is_leaf=is_none,
    )


def get_task_eval_func(task_base, hps, unit_idx, stim_amp_base):
    def task_eval_func(model, key):
        task = task_with_scaled_unit_stim(
            model,
            task_base,
            unit_idx,
            stim_amp_base,
            hps.train.model.hidden_size,
            UNIT_STIM_INTERVENOR_LABEL,
        )
        return task.eval(model, key=key)

    return task_eval_func


def eval_func(key_eval, hps, models, task):
    states = eqx.filter_vmap(
        lambda stim_amp: eqx.filter_vmap(
            lambda unit_idx: eqx.filter_vmap(
                lambda key_eval: eqx.filter_vmap(get_task_eval_func(task, hps, unit_idx, stim_amp))(
                    models, jr.split(key_eval, hps.train.model.n_replicates)
                )
            )(jr.split(key_eval, hps.eval_n))
        )(jnp.arange(hps.train.model.hidden_size))
    )(jnp.array(hps.pert.unit.amp))

    return states


MEASURE_KEYS = ()


# def get_unit_stim_origins_directions(task, models, hps):
#     origins = task.validation_trials.inits["mechanics.effector"].pos
#     directions = jnp.broadcast_to(jnp.array([1., 0.]), origins.shape)
#     return origins, directions


def get_impulse_vrect_kws(hps):
    return dict(
        x0=hps.pert.unit.start_step,
        x1=hps.pert.unit.start_step + hps.pert.unit.duration,
        fillcolor="grey",
        opacity=0.2,
        line_width=0,
        name="Perturbation",
    )


# def move_var_above_train_pert_std(tree, **kwargs):
#     return move_ldict_level_above('var', 'train__pert__std', tree)


def rearrange_profile_vars(tree, **kwargs):
    tree = move_ldict_level_above("train__pert__std", "pert__amp", tree)
    tree = move_ldict_level_above("var", "pert__amp", tree)
    return tree


unit_idxs_profiles_plot = jnp.array([2, 83, 95, 97, 43, 48])  # jnp.arange(8)


def segment_stim_epochs(states, *, hps_common, **kwargs):
    start_step = hps_common.pert.unit.start_step
    end_step = start_step + hps_common.pert.unit.duration

    return jt.map(
        lambda states: jt.map(
            lambda idxs: jt.map(
                lambda arr: arr[..., idxs, :],
                states,
            ),
            LDict.of("epoch")(
                {
                    "pre": slice(0, start_step),
                    "peri": slice(start_step, end_step),
                    "post": slice(end_step, None),
                }
            ),
        ),
        states,
        is_leaf=is_module,
    )


def transform_profile_vars(states_by_var, **kwargs):
    return LDict.of("var")(
        dict(
            deviation=jnp.linalg.norm(states_by_var["pos"], axis=-1, keepdims=True),
            angle=jnp.arctan2(states_by_var["pos"][..., 1], states_by_var["pos"][..., 0])[
                ..., None
            ],
            speed=jnp.linalg.norm(states_by_var["vel"], axis=-1, keepdims=True),
        )
    )


def max_deviation_after_stim(states_by_var, *, hps_common, **kwargs):
    deviation = jnp.linalg.norm(states_by_var["pos"], axis=-1)
    pert_end = hps_common.pert.unit.start_step + hps_common.pert.unit.duration
    ts = jnp.arange(pert_end, hps_common.model.n_steps)
    return jnp.max(deviation[..., ts], axis=-1)


class UnitStimRegressionFiguresPorts(AbstractAnalysisPorts):
    """Input ports for UnitStimRegressionFigures analysis."""

    regression_results: Optional[InputOf[Any]] = None
    unit_fb_gains: Optional[InputOf[Any]] = None


class UnitStimRegressionFigures(AbstractAnalysis[UnitStimRegressionFiguresPorts]):
    Ports = UnitStimRegressionFiguresPorts
    inputs: UnitStimRegressionFiguresPorts = eqx.field(
        default_factory=UnitStimRegressionFiguresPorts,
        converter=UnitStimRegressionFiguresPorts.converter,
    )

    variant: Optional[str] = "full"

    def make_figs(
        self,
        data: AnalysisInputData,
        *,
        regression_results,
        unit_fb_gains,
        replicate_info,
        **kwargs,
    ) -> PyTree[go.Figure]:
        regression_weights = regression_results[0].weight
        feature_labels = regression_results[1]

        figs = {}

        # ## Plot single regressor weights against each other
        fig = go.Figure(
            layout=dict(
                xaxis_title="SISU regression weight",
                yaxis_title="Curl amp. regression weight",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=regression_weights[:, 0, 1],
                y=regression_weights[:, 0, 2],
                mode="markers",
                hovertemplate="unit %{pointNumber}: (%{x}, %{y})",
            )
        )

        figs[(feature_labels[1], feature_labels[2])] = fig

        ## Plot SISU weight against feedback input weights (input idxs 5,6 & 7,8)
        # fig = go.Figure()
        # # Model should not depend on SISU or pert amp; select 0 for each
        # models = data.models["full"][0][0][1.5]
        # models_best_replicate = get_best_replicate(
        #     models, replicate_info=replicate_info[1.5], axis=0
        # )
        # input_weights = models_best_replicate.step.net.hidden.weight_ih

        # feedback_slices = {
        #     "pos": slice(5, 7),
        #     "vel": slice(7, 9),
        # }
        # gate_slices = {
        #     "reset": slice(0, 100),
        #     "update": slice(100, 200),
        #     "candidate": slice(200, 300),
        # }

        # # TODO: Make into nested LDict
        # for fb_var, idxs in feedback_slices.items():
        #     fig = go.Figure(
        #         layout=dict(
        #             xaxis_title="SISU regression weight",
        #             yaxis_title=f"{fb_var} feedback input weight",
        #         )
        #     )
        #     weight_amp = jnp.linalg.norm(input_weights[..., idxs], axis=-1)

        #     for gate, gate_idxs in gate_slices.items():
        #         fig.add_trace(
        #             go.Scatter(
        #                 x=regression_weights[:, 0, 1],
        #                 y=weight_amp[gate_idxs],
        #                 mode="markers",
        #                 name=gate,
        #             )
        #         )

        #     label = f"{fb_var}-fb-input-weight_vs_SISU-reg-weight"
        #     fig.write_html(f"{label}.html")
        #     fig.write_image(f"{label}.webp")

        return LDict.of("comparison")(figs)


class InstantFBResponsePorts(AbstractAnalysisPorts):
    rnn_cells: InputOf[Callable] = Data.models(where=lambda m: m.step.net.hidden)


class InstantFBResponse(AbstractAnalysis[InstantFBResponsePorts]):
    """Plot the instantaneous feedback response of the network."""

    Ports = InstantFBResponsePorts
    inputs: InstantFBResponsePorts = eqx.field(default_factory=Ports, converter=Ports.converter)

    n_directions: int = 24
    # n_samples: int = 1000
    fb_pert_amp: PyTree[float] = 0.5
    input_idxs: PyTree[slice | ArrayLike] = field(
        default_factory=lambda: LDict.of("fb_var")(
            pos=slice(5, 7),
            vel=slice(7, 9),
        )
    )

    def compute(self, data, *, rnn_cells, hps_common, **kwargs):
        fb_perts = self.fb_pert_amp * unit_circle_points(self.n_directions)
        pert_step = hps_common.pert.unit.start_step

        def _compute_single(rnn_cell, states):
            base_net_state = first(states, is_leaf=is_module).net
            # TODO: Repeat computation independently over multiple steady-state steps, and average.
            init_state = base_net_state.hidden[..., pert_step - 1, :]
            base_input = base_net_state.input[..., pert_step, :]

            def _single_pert(fb_pert):
                def _per_input(idxs):
                    perturbed_input = base_input.at[..., idxs].add(fb_pert)
                    return rnn_cell(perturbed_input, init_state) - init_state

                return jt.map(_per_input, self.input_idxs)

            return eqx.filter_vmap(_single_pert)(fb_perts)

        return jt.map(
            _compute_single,
            rnn_cells,
            data.states,
            is_leaf=is_module,
        )


VARSET = subdict(DEFAULT_VARSET, ("pos", "vel"))


def aggregate_unit_gains(fb_gains):
    return fb_gains


DEPENDENCIES = {
    "aligned_vars_trivial": AlignedVars(
        varset=VARSET,
        # Bypass alignment; keep aligned with x-y axes
        directions_func=get_trivial_reach_directions,
    ),
    "unit_fb_gains": (
        InstantFBResponse(
            n_directions=24,
            fb_pert_amp=0.5,
        )
        # There's only one reach condition atm
        .after_indexing(-3, 0, axis_label="condition", dependency_name="data.states")
        # Computation based on steady-state period; stim vars not relevant.
        .after_indexing(1, 0, axis_label="stim_unit_idx", dependency_name="data.states")
        .after_indexing(0, 0, axis_label="stim_amp", dependency_name="data.states")
        .vmap(
            in_axes={"rnn_cells": MultiVmapAxes(0, None), "data.states": MultiVmapAxes(1, 2)}
        )  # replicates
        .then_transform_result(aggregate_unit_gains)
    ),
}


def dashed_fig_params_fn(fig_params, ctx: FigIterCtx):
    return fig_params | dict(
        scatter_kws=dict(
            line_dash=SISU_STYLES["line_dash"][ctx.idx],
            legendgroup=SISU_LABELS[ctx.idx],
            legendgrouptitle_text=f"SISU: {SISU_LABELS[ctx.idx]}",
        ),
    )


# PyTree structure: [sisu, pert__amp, train__pert__std]
# Array batch shape: [stim_amp, stim_unit_idx, eval, replicate, condition]
ANALYSES = {
    "unit_stim_profiles": (
        Profiles(
            variant="full",
            vrect_kws_func=get_impulse_vrect_kws,
            coord_labels=None,
            inputs=Profiles.Ports(
                vars="aligned_vars_trivial",
            ),
        )
        .after_transform(partial(get_best_replicate, axis=3))
        .after_indexing(
            1, unit_idxs_profiles_plot, axis_label="unit_stim_idx"
        )  #! Only make figures for a few stim units
        .after_transform(
            transform_profile_vars, level="var", dependency_names="vars"
        )  # e.g. positions to deviations
        .after_unstacking(1, "unit_stim_idx", above_level="pert__amp", dependency_name="vars")
        # .after_indexing(0, 1, axis_label="stim_amp")  #! Only make figures for unit stim condition
        .after_unstacking(0, "stim_amp", above_level="pert__amp", dependency_name="vars")
        .after_transform(
            rearrange_profile_vars, dependency_names="vars"
        )  # Plot pert amp. on same figure
        .combine_figs_by_level(  # Also plot SISU on same figure, with different line styles
            level="sisu",
            fig_params_fn=dashed_fig_params_fn,
        )
        .with_fig_params(
            layout_kws=dict(
                width=500,
                height=350,
            ),
        )
    ),
    # "unit_stim_regression": (
    #     Regression(
    #         variant="full",
    #         inputs=Regression.Ports(
    #             regressor_tree="aligned_vars_trivial",
    #         ),
    #     )
    #     .after_transform(partial(get_best_replicate, axis=3))
    #     .after_indexing(0, 1, axis_label="stim_amp")  #! Only do regression for stim condition
    #     .after_transform(lambda subtree, **kwargs: subtree[1.5], level="train__pert__std")  #! Only for trained on perturbations
    #     # .after_transform(transform_profile_vars, level='var', dependency_name="regressor_tree")  #
    #     # e.g. positions to deviations
    #     .after_transform(max_deviation_after_stim, level="var", dependency_names="regressor_tree")
    #     .vmap(in_axes={"regressor_tree": 0})
    # ),
    # "unit_stim_regression_figures": UnitStimRegressionFigures(
    #     inputs=UnitStimRegressionFigures.Ports(
    #         regression_results="unit_stim_regression",
    #     ),
    # ),
    #! TODO: Replace with `get_aligned_trajectories_node`
    # "aligned_effector_trajectories": (
    #     AlignedEffectorTrajectories(
    #         variant="full",
    #         custom_inputs={
    #             "aligned_vars": "aligned_vars_trivial",
    #         },
    #     )
    #     .after_transform(partial(get_best_replicate, axis=3))
    # ),
    # "unit_preferences": (
    #     # Result shape: [stim_amp, unit_stim_idx, unit_idx, feature]
    #     UnitPreferences(
    #         variant="full",
    #         feature_fn=lambda task, states: states.efferent.output,
    #     )
    #     .after_transform(partial(get_best_replicate, axis=3))
    #     .after_transform(
    #         # get_segment_trials_func(get_symmetric_accel_decel_epochs),
    #         segment_stim_epochs,
    #         dependency_names="states",
    #     )
    #     .vmap_over_states(axes=[0, 1])  # Compute preferences separately for stim vs. nostim, and for each stim unit
    # ),
}
