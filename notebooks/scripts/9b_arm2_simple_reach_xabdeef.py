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
NB_PREFIX = "nb9"
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
import jax.random as jrandom
import matplotlib.pyplot as plt
import numpy as np
import optax 

from feedbax.model import SimpleFeedback
from feedbax.iterate import Iterator, SimpleIterator
import feedbax.loss as fbl
from feedbax.mechanics import Mechanics 
from feedbax.mechanics.arm import TwoLink
from feedbax.networks import RNNCell, RNNCellWithReadout
from feedbax.task import RandomReaches
from feedbax.trainer import TaskTrainer, save, load
from feedbax.xabdeef.losses import simple_reach_loss

from feedbax.plot import (
    plot_losses, 
    plot_2D_joint_positions,
    plot_pos_vel_force_2D,
    plot_activity_heatmap,
)

# %%
os.environ["FEEDBAX_DEBUG"] = str(DEBUG)
logging.getLogger("jax").setLevel(logging.INFO)

jax.config.update("jax_debug_nans", DEBUG)
jax.config.update("jax_enable_x64", ENABLE_X64)

# not sure if this will work or if I need to use the env variable version
#jax.config.update("jax_traceback_filtering", DEBUG)  

plt.style.use('dark_background')

# %%
# paths

# training checkpoints
chkpt_dir = Path("/tmp/feedbax-checkpoints")
chkpt_dir.mkdir(exist_ok=True)

# tensorboard
tb_logdir = Path("runs")

model_dir = Path("../models/")


# %%
def get_model(
    task,
    key=None,
    dt=0.05, 
    hidden_size=50, 
    n_steps=50, 
    feedback_delay=0, 
    out_nonlinearity=lambda x: x,
):
    if key is None:
        # in case we just want a skeleton model, e.g. for deserializing
        key = jrandom.PRNGKey(0)

    key1, key2, key3 = jrandom.split(key, 3)

    system = TwoLink()  
    mechanics = Mechanics(system, dt, clip_states=True)
    
    # unlike in the pointmass example, the effector is different
    # from the system configuration, so we include both 
    # (by default only `mechanics_state.system` is included)
    feedback_leaves_func = lambda mechanics_state: (
        mechanics_state.system,
        mechanics_state.effector,         
    )
    
    input_size = SimpleFeedback.get_nn_input_size(
        task, mechanics, feedback_leaves_func
    )

    net = RNNCellWithReadout(
        input_size, 
        hidden_size, 
        system.control_size,
        out_nonlinearity=out_nonlinearity,
        key=key1,
    )

    body = SimpleFeedback(
        net, 
        mechanics, 
        delay=feedback_delay,  
        feedback_leaves_func=feedback_leaves_func,
        key=key3,
    )
    
    return SimpleIterator(body, n_steps)


# %%
seed = 5566

n_steps = 50
dt = 0.05 
feedback_delay_steps = 0
workspace = ((-0.15, 0.20), 
             (0.15, 0.50))
hidden_size  = 50

loss_term_weights = dict(
    effector_position=1.,
    effector_final_velocity=1.,
    nn_output=1e-5,
    nn_activity=1e-6,
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

    key = jrandom.PRNGKey(seed)

    loss_func = simple_reach_loss(
        n_steps, 
        loss_term_weights,
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
        key, 
        dt=dt,
        hidden_size=hidden_size,
        n_steps=n_steps,
        feedback_delay=feedback_delay_steps,
    )
    
    return model, task


# %%
model, task = setup(**hyperparams)

trainer = TaskTrainer(
    optimizer=optax.inject_hyperparams(optax.adam)(
        learning_rate=2e-2
    ),
    chkpt_dir=chkpt_dir,
    checkpointing=True,
)

# %%
trainable_leaves_func = lambda model: model.step.net

model, losses, learning_rates = trainer(
    task=task, 
    model=model,
    n_batches=10_000, 
    batch_size=500, 
    log_step=100,
    trainable_leaves_func=trainable_leaves_func,
    key=jrandom.PRNGKey(seed + 1),
)

plot_losses(losses)
plt.show()

# %% [markdown]
# Save the model and task to file along with the hyperparameters needed to set them up again on loading

# %%
model_path = save(
    (model, task),
    hyperparams, 
    save_dir=model_dir, 
    suffix=NB_PREFIX
)

model_path

# %% [markdown]
# If we didn't just save a model, we can try to load one. Note that this depends on `setup` being the same.

# %%
try:
    model_path
    model, task
except NameError:
    model_path = "../models/model_20231026-100544_b4a92ad_nb12.eqx"
    model, task = load(model_path, setup)

# %% [markdown]
# Evaluate on a centre-out task

# %%
losses, states = task.eval(model, key=jrandom.PRNGKey(0))

# %%
trial_specs, _ = task.trials_validation
goal_states = jax.tree_map(lambda x: x[:, -1], trial_specs.target)

plot_pos_vel_force_2D(
    states,
    endpoints=(
        trial_specs.init['mechanics']['effector'].pos, 
        goal_states.pos
    )
)

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

# %%
