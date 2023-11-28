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
# Simple feedback model with a single-layer RNN controlling a point mass to reach from a starting position to a target position. 
#
# Components:
#
# - Point mass module
# - RNN module
# - Feedback loop 
#     - RNN call plus a single step of diffrax integration
# - Loss function
#     - quadratic in position near final state
#     - quadratic in controls
# - Generate reach endpoints
#     - uniformly sampled in a rectangular workspace
#     - i.i.d. start and end (variable magnitude)

# %%
LOG_LEVEL = "INFO"
NB_PREFIX = "nb8"
DEBUG = False
DISABLE_JIT = False
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
from typing import Optional, Callable

from IPython import get_ipython

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

# redirect stderr (e.g. warnings) to file
stderr_log = sys.stderr = open(f'log/stderr_{NB_PREFIX}.log', 'w')
get_ipython().log.handlers[0].stream = stderr_log 
get_ipython().log.setLevel(LOG_LEVEL)

# %%
import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp 
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import optax 

from feedbax.model import SimpleFeedback
from feedbax.iterate import Iterator
import feedbax.loss as fbl
from feedbax.mechanics import Mechanics 
from feedbax.mechanics.linear import point_mass
from feedbax.networks import RNNCellWithReadout, RNNCellWithReadoutAndInput
from feedbax.plot import plot_losses, plot_pos_vel_force_2D
from feedbax.task import RandomReaches
from feedbax.trainer import TaskTrainer, save, load
from feedbax.utils import tree_take 

# %%
os.environ["FEEDBAX_DEBUG"] = str(DEBUG)
logging.getLogger("jax").setLevel(logging.INFO)

jax.config.update("jax_debug_nans", DEBUG)
jax.config.update('jax_disable_jit', DISABLE_JIT)
jax.config.update("jax_enable_x64", ENABLE_X64)

# not sure if this will work or if I need to use the env variable version
#jax.config.update("jax_traceback_filtering", DEBUG)  

plt.style.use('dark_background')

# %%
model_dir = Path("../models/")


# %%
def get_model(
    task,
    dt: float = 0.05, 
    mass: float = 1., 
    rnn_input_size: int = 50,
    hidden_size: int = 50, 
    readout_hidden_size: int = 25,
    n_steps: int = 100, 
    feedback_delay: int = 0,
    out_nonlinearity=lambda x: x,
    key: Optional[jr.PRNGKeyArray] = None,
):
    if key is None:
        # in case we just want a skeleton model, e.g. for deserializing
        key = jr.PRNGKey(0)
    
    system = point_mass(mass=mass, n_dim=N_DIM)
    mechanics = Mechanics(system, dt, solver=diffrax.Euler)
    
    # automatically determine network input size
    input_size = SimpleFeedback.get_nn_input_size(
        task, mechanics
    )
    
    net = RNNCellWithReadoutAndInput(
        input_size,
        rnn_input_size,
        hidden_size,
        system.control_size, 
        readout_type=eqx.nn.Linear, #two_layer_readout,
        out_nonlinearity=out_nonlinearity, 
        key=key,
    )
    
    body = SimpleFeedback(
        net, 
        mechanics, 
        feedback_delay
    )
    
    return Iterator(body, n_steps)


# %%
seed = 5566

n_steps = 100
dt = 0.1
feedback_delay_steps = 5
workspace = ((-1., -1.),
             (1., 1.))
hidden_size  = 50
learning_rate = 0.01

loss_term_weights = dict(
    effector_position=1.,
    effector_final_velocity=1.,
    nn_output=1e-5,
    nn_activity=1e-5,
)

# hyperparams dict + setup function isn't strictly necessary,
# but it makes model saving and loading more sensible
hyperparams = dict(
    seed=seed,
    n_steps=n_steps, 
    loss_term_weights=loss_term_weights, 
    workspace=workspace, 
    dt=dt, 
    hidden_size=hidden_size, 
    feedback_delay_steps=feedback_delay_steps, 
)


# %%
def setup(
    seed, 
    n_steps, 
    loss_term_weights,
    workspace,
    dt,
    hidden_size,
    feedback_delay_steps,
):
    """Set up the model and the task."""
    key = jr.PRNGKey(seed)

    # these assume a particular PyTree structure to the states returned by the model
    # which is why we simply instantiate them 
    loss_func = fbl.CompositeLoss(
        dict(
            # these assume a particular PyTree structure to the states returned by the model
            # which is why we simply instantiate them 
            effector_position=fbl.EffectorPositionLoss(
                discount_func=lambda n_steps: fbl.power_discount(n_steps, 6)),
            effector_final_velocity=fbl.EffectorFinalVelocityLoss(),
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
        eval_reach_length=0.5,
    )

    model = get_model(
        task,
        dt=dt,
        hidden_size=hidden_size,
        n_steps=n_steps,
        feedback_delay=feedback_delay_steps,
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
n_batches = 10_000
batch_size = 500
key_train = jr.PRNGKey(seed + 1)

trainable_leaves_func = lambda model: (
    model.step.net.input_layer.weight,
    model.step.net.input_layer.bias,
    model.step.net.cell.weight_hh, 
    model.step.net.cell.weight_ih, 
    model.step.net.cell.bias,
    model.step.net.readout,
)

model, losses, learning_rates = trainer(
    task=task, 
    model=model,
    n_batches=n_batches, 
    batch_size=batch_size, 
    log_step=200,
    trainable_leaves_func=trainable_leaves_func,
    key=key_train,
)

plot_losses(losses)

# %% [markdown]
# Save the trained model to file, along with the task and the hyperparams needed to set them up again

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
    model_path = "../models/model_20231120-155247_0244319_nb8.eqx"
    model, task = load(model_path, setup)

# %% [markdown]
# Evaluate on a centre-out task

# %%
losses, states = task.eval(model, key=jr.PRNGKey(0))

# %%
trial_specs, _ = task.trials_validation

plot_pos_vel_force_2D(
    states,
    endpoints=(
        trial_specs.init['mechanics']['effector'].pos, 
        trial_specs.goal.pos
    ),
)
plt.show()

# %%
from feedbax.intervene import EffectorCurlForceField
from feedbax.model import add_intervenors

model_ = eqx.tree_at(
    lambda model: model.step,
    model,
    add_intervenors(model.step, [EffectorCurlForceField(0.15)]),
)

# %%
losses, states = task.eval(model_, key=jr.PRNGKey(0))

# %%
(losses, states), trials, aux = task.eval_train_batch(
    model, 
    batch_size=10,
    key=jr.PRNGKey(0), 
)

# %%
init_states, target_states, _ = trials
goal_states = jax.tree_map(lambda x: x[:, -1], target_states)
plot_pos_vel_force_2D(
    states,
    endpoints=(init_states.pos, goal_states.pos),
)
plt.show()

# %%
plt.plot(jnp.sum(states.mechanics.system.vel ** 2, -1).T, '-')
plt.show()

# %%
