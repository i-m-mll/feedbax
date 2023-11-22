# ---
# jupyter:
#   jupytext:
#     formats: ipynb,scripts//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: fx
#     language: python
#     name: python3
# ---

# %% [markdown]
# Simple feedback model with a single-layer RNN controlling a two-link arm to reach from a starting position to a target position. 

# %%
LOG_LEVEL = "INFO"
NB_PREFIX = "nb10"
DEBUG = False
ENABLE_X64 = False
N_DIM = 2  # TODO: not here


# %%
# %load_ext autoreload
# %autoreload 2

# %%
import os
import logging
from pathlib import Path
import sys
from typing import Optional

from IPython import get_ipython

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

# redirect stderr (e.g. warnings) to file
stderr_log = sys.stderr = open(f'log/stderr_{NB_PREFIX}.log', 'w')
get_ipython().log.handlers[0].stream = stderr_log 
get_ipython().log.setLevel(LOG_LEVEL)

# %%
import equinox as eqx
import jax
import jax.numpy as jnp 
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import optax 

import feedbax.loss as fbl
from feedbax.model import SimpleFeedback
from feedbax.iterate import Iterator
from feedbax.mechanics import Mechanics 
from feedbax.mechanics.muscle import (
    ActivationFilter,
    TodorovLiVirtualMuscle, 
) 
from feedbax.mechanics.muscled_arm import TwoLinkMuscled 
from feedbax.networks import RNNCellWithReadoutAndInput
from feedbax.task import RandomReaches
from feedbax.trainer import TaskTrainer, save, load
from feedbax.xabdeef.losses import simple_reach_loss


from feedbax.plot import (
    plot_loss, 
    plot_2D_joint_positions,
    plot_pos_vel_force_2D,
    plot_activity_heatmap,
)

# %%
os.environ["FEEDBAX_DEBUG"] = str(DEBUG)
logging.getLogger("jax").setLevel(logging.INFO)

jax.config.update("jax_debug_nans", DEBUG)
jax.config.update('jax_disable_jit', DEBUG)
jax.config.update("jax_enable_x64", ENABLE_X64)

# not sure if this will work or if I need to use the env variable version
#jax.config.update("jax_traceback_filtering", DEBUG)  

plt.style.use('dark_background')

# %%
model_dir = Path("../models/")


# %% [markdown]
# Define the model.

# %%
def get_model(
    task,
    dt: float = 0.05, 
    rnn_input_size: int = 25,
    hidden_size: int = 50, 
    n_steps: int = 50, 
    feedback_delay: int = 0, 
    tau: float = 0.01, 
    out_nonlinearity=jax.nn.sigmoid,
    clip_states=False,
    key: Optional[jr.PRNGKeyArray] = None,
):
    if key is None:
        # in case we just want a skeleton model, e.g. for deserializing
        key = jr.PRNGKey(0)
    
    system = TwoLinkMuscled(
        muscle_model=TodorovLiVirtualMuscle(), 
        activator=ActivationFilter(
            tau_act=tau,  
            tau_deact=tau,
        )
    )
    mechanics = Mechanics(system, dt, clip_states=clip_states)
    
    # this time we need to specifically request joint angles
    # and angular velocities, and not muscle activations
    feedback_leaves_func = lambda mechanics_state: (
        mechanics_state.system.theta,
        mechanics_state.system.d_theta,
        mechanics_state.effector.pos,
        mechanics_state.effector.vel,        
    )
    
    # automatically determine network input size
    input_size = SimpleFeedback.get_nn_input_size(
        task, mechanics, feedback_leaves_func
    )
    
    net = RNNCellWithReadoutAndInput(
        input_size, 
        rnn_input_size,
        hidden_size, 
        system.control_size, 
        out_nonlinearity=out_nonlinearity, 
        key=key,
    )
    body = SimpleFeedback(
        net, 
        mechanics, 
        delay=feedback_delay, 
        feedback_leaves_func=feedback_leaves_func, 
    )

    return Iterator(body, n_steps)


# %%
seed = 5566

n_steps = 50
dt = 0.05 
feedback_delay_steps = 0
workspace = ((-0.15, 0.15), 
             (0.20, 0.50))
hidden_size  = 50
learning_rate = 0.05

loss_term_weights = dict(
    effector_position=1.,
    effector_final_velocity=0.1,
    effector_straight_path=1e-3,
    nn_output=1e-4,
    nn_activity=0.,
)

hyperparams = dict(
    seed=seed,
    n_steps=n_steps,
    workspace=workspace,
    loss_term_weights=loss_term_weights,
    dt=dt,
    hidden_size=hidden_size,
    feedback_delay_steps=feedback_delay_steps,
)


# %%
def setup(
    seed, 
    n_steps, 
    workspace,
    loss_term_weights,
    dt, 
    hidden_size,
    feedback_delay_steps,    
):

    key = jr.PRNGKey(seed)

    discount = fbl.power_discount(n_steps, 6)
    loss_func = fbl.CompositeLoss(
        dict(
            effector_position=fbl.EffectorPositionLoss(discount=discount),
            effector_final_velocity=fbl.EffectorFinalVelocityLoss(),
            effector_straight_path=fbl.EffectorStraightPathLoss(),
            nn_output=fbl.NetworkOutputLoss(),
            nn_activity=fbl.NetworkActivityLoss(),
        ),
        weights=loss_term_weights,
    )

    task = RandomReaches(
        loss_func=loss_func,
        workspace=workspace, 
        n_steps=n_steps,
        eval_grid_n=2,
        eval_n_directions=8,
        eval_reach_length=0.05,
    )

    model = get_model(
        task,
        dt=dt,
        hidden_size=hidden_size,
        n_steps=n_steps,
        feedback_delay=feedback_delay_steps,
        tau=0.01,
        out_nonlinearity=jax.nn.sigmoid,
        key=key, 
    )
    
    return model, task


# %%
model, task = setup(**hyperparams)

trainer = TaskTrainer(
    optimizer=optax.inject_hyperparams(optax.adam)(
        learning_rate=learning_rate
    ),
    checkpointing=True,
)

# %%
trainable_leaves_func = lambda model: (
    model.step.net.input_layer,
    model.step.net.cell.weight_hh, 
    model.step.net.cell.weight_ih, 
    model.step.net.cell.bias,
)

model, losses, loss_terms, learning_rates = trainer(
    task=task, 
    model=model,
    n_batches=10_000, 
    batch_size=500, 
    log_step=100,
    trainable_leaves_func=trainable_leaves_func,
    key=jr.PRNGKey(seed + 1),
)

plot_loss(losses, loss_terms)
plt.show()

# %% [markdown]
# Save the trained model to file

# %%
model_path = save(
    (model, task),
    hyperparams, 
    save_dir=model_dir, 
    suffix=NB_PREFIX,
)

model_path

# %% [markdown]
# If we didn't just save a model, we can try to load one

# %%
try:
    model_path
    model, task
except NameError:
    model_path = '../models/model_20231121-104123_eb3d3cc_nb10.eqx'
    model, task = load(model_path, setup)

# %% [markdown]
# Evaluate on a centre-out task

# %%
loss, loss_terms, states = task.eval(model, key=jr.PRNGKey(0))

# %%
# fig = make_eval_plot(states[1], states[2], workspace)
trial_specs, _ = task.trials_validation
goal_states = jax.tree_map(lambda x: x[:, -1], trial_specs.target)

# %%
plot_pos_vel_force_2D(
    states,
    endpoints=(trial_specs.init.pos, goal_states.pos), 
    force_labels=('Biarticular controls', 'Flexor', 'Extensor'), 
    cmap='plasma', 
    workspace=task.workspace,
);

# %% [markdown]
# Plot entire arm trajectory for an example direction

# %%
idx = 0

# %%
# convert all joints to Cartesian since I only saved the EE state

# vmap twice, over trials and time; `forward_kinematics` applies to single points
forward_kinematics = model.step.mechanics.system.forward_kinematics
xy_pos = jax.vmap(jax.vmap(forward_kinematics, in_axes=0), in_axes=1)(
    states.mechanics.system
).pos

# #? we can't just swap `in_axes` above; it causes a vmap shape error with 
# axis 2 of the arrays in `states.mechanics.system`, which includes 
# the (unused, in this case) muscle activation state
xy_pos = jnp.swapaxes(xy_pos, 0, 1)

# %%
ax = plot_2D_joint_positions(xy_pos[idx], add_root=True)
plt.show()

# %% [markdown]
# Network hidden activities over time for the same example reach direction

# %%
# semilogx is interesting in this case without a GO cue
# ...helps to visualize the initial activity
fig, ax = plt.subplots(1, 1)
ax.semilogx(states.hidden[idx])
ax.set_xlabel('Time step')
ax.set_ylabel('Hidden unit activity')
plt.show()

# %% [markdown]
# Heatmap of network activity over time for an example direction

# %%
plot_activity_heatmap(states.hidden[2], cmap='viridis')

# %%
