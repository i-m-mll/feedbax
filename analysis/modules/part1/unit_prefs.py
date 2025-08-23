from functools import partial
from typing import Optional

import jax.tree as jt
from jaxtyping import Array, Float
import numpy as np

from feedbax.intervene import add_intervenors, schedule_intervenor
from jax_cookbook import is_module
import jax_cookbook.tree as jtree

from rlrmp.analysis.disturbance import PLANT_PERT_FUNCS
from rlrmp.analysis.network import UnitPreferences
from rlrmp.analysis.state_utils import get_best_replicate, vmap_eval_ensemble
from rlrmp.analysis.disturbance import PLANT_INTERVENOR_LABEL
from rlrmp.analysis.state_utils import get_symmetric_accel_decel_epochs
from rlrmp.analysis.state_utils import get_segment_trials_func
from rlrmp.misc import vectors_to_2d_angles
from rlrmp.types import LDict


COLOR_FUNCS = {}


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
    all_tasks, all_models = jtree.unzip(jt.map(
        lambda pert_amp: schedule_intervenor(
            task_base, models,
            lambda model: model.step.mechanics,
            disturbance(pert_amp),
            label=PLANT_INTERVENOR_LABEL,
            default_active=False,
        ),
        LDict.of("pert__amp")(
            dict(zip(pert_amps, pert_amps))
        ),
    ))

    all_hps = jt.map(lambda _: hps, all_tasks, is_leaf=is_module)

    return all_tasks, all_models, all_hps, None


eval_func = vmap_eval_ensemble


def get_goal_positions(task, states):
    targets = task.validation_trials.targets["mechanics.effector.pos"].value
    return targets[..., -1:, :]


def get_control_forces(task, states):
    return states.efferent.output


ts = np.arange(0, 20)


ANALYSES = {
    # "unit_prefs_goal_positions": (
    #     UnitPreferences(feature_fn=get_goal_positions)
    #     .after_transform(get_best_replicate)
    #     .after_transform(
    #         get_segment_trials_func(get_symmetric_accel_decel_epochs),
    #         dependency_names="states",
    #     )
    #     # .after_indexing(-2, ts, axis_label="timestep")
    #     # .and_transform_results(map_fn_over_tree(vectors_to_2d_angles))
    # ),
    "unit_prefs_control_forces": (
        UnitPreferences(
            feature_fn=get_control_forces,
        )
        .after_transform(get_best_replicate)
        .after_transform(
            get_segment_trials_func(get_symmetric_accel_decel_epochs),
            dependency_names="states",
        )
        # .after_indexing(-2, ts, axis_label="timestep")
        # .and_transform_results(map_fn_over_tree(vectors_to_2d_angles))
    ),
    "unit_prefs_goal_positions": (
        UnitPreferences(
            feature_fn=get_goal_positions,
        )
        .after_transform(get_best_replicate)
        .after_transform(
            get_segment_trials_func(get_symmetric_accel_decel_epochs),
            dependency_names="states",
        )
        # .after_indexing(-2, ts, axis_label="timestep")
        # .and_transform_results(map_fn_over_tree(vectors_to_2d_angles))
    ),
}


#! Can visualize the unit preference regression using `planar_regression`, below.
#! Here is an example of how to do this, given `data, common_inputs, all_results`,
#! in the case of goal position preference.
#! I have not included this in `UnitPreferences.make_figs` at this time. 
# import jax.numpy as jnp
# import numpy as np
# from rlrmp.analysis.state_utils import get_best_replicate

# unit_plot = 0
# ts = np.arange(15)
# ts_late = np.arange(15, 30)
# states_b = get_best_replicate(data.states['full'][0], replicate_info=common_inputs['replicate_info'])
# hidden = states_b[0].net.hidden
# hidden_early = hidden[..., ts, :]
# hidden_late = hidden[..., ts_late, :]
# targets = data.tasks['full'][0].validation_trials.targets["mechanics.effector.pos"].value
# targets_early = targets[:, ts]
# targets_late = targets[:, ts_late]
# pref_dirs = list(all_results.values())[0]['full'][0][0]
# pref_dirs = jt.map(lambda pd: pd / jnp.linalg.norm(pd, axis=-1, keepdims=True), pref_dirs)

# fig = planar_regression(targets_early, hidden_early[0, ..., unit_plot], pref_dirs['accel'][unit_plot])

import plotly.graph_objects as go
import numpy as np
from jaxtyping import Array, Float
import feedbax.plotly as fbp


def plot_planar_regression(
    X: Float[Array, "... ndim=2"], 
    y, 
    weights, 
    labels=dict(x='x', y='y', z='Unit activity'),
    colorscale='phase',
):
    """Create a 3D plot of data points with a vector representing weights."""
    # Create a Plotly figure
    fig = go.Figure()
    
    colors = fbp.sample_colorscale_unique(colorscale, X.shape[0])
    
    # Add scatter3d traces for each dataset
    for X_, y_, color in zip(X, y, colors):
        fig.add_trace(go.Scatter3d(
            x=X_[:, 0],
            y=X_[:, 1],
            z=y_,
            mode='markers',
            marker=dict(
                size=5,
                color=color,
            ),
            showlegend=False
        ))
    
    # Add the weights vector as a line
    # Scale the vector to length 0.5 to match the original
    vector_length = 0.5
    magnitude = np.sqrt(weights[0]**2 + weights[1]**2)
    if magnitude > 0:
        scale = vector_length / magnitude
        scaled_x = weights[0] * scale
        scaled_y = weights[1] * scale
        
        fig.add_trace(go.Scatter3d(
            x=[0, scaled_x],
            y=[0, scaled_y],
            z=[0, 0],  # No z-component in the original
            mode='lines',
            line=dict(
                color='black',
                width=5
            ),
            showlegend=False
        ))
    
    # Configure the layout
    fig.update_layout(
        scene=dict(
            xaxis_title=labels['x'],
            yaxis_title=labels['y'],
            zaxis_title=labels['z'],
            aspectmode='auto'
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )

    return fig

