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
from functools import partial
import logging
import os
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

from feedbax.context import SimpleFeedback
import feedbax.loss as fbl
from feedbax.mechanics import Mechanics 
from feedbax.mechanics.muscle import (
    ActivationFilter,
    TodorovLiVirtualMuscle, 
) 
from feedbax.mechanics.muscled_arm import TwoLinkMuscled 
from feedbax.networks import RNN
from feedbax.recursion import Recursion
from feedbax.task import RandomReaches
from feedbax.trainer import TaskTrainer


from feedbax.plot import (
    plot_loglog_losses, 
    plot_2D_joint_positions,
    plot_states_forces_2d,
    plot_activity_heatmap,
)

# %%
os.environ["FEEDBAX_DEBUG"] = str(DEBUG)
logging.getLogger("jax").setLevel(logging.INFO)

jax.config.update("jax_debug_nans", DEBUG)
jax.config.update("jax_enable_x64", ENABLE_X64)

# not sure if this will work or if I need to use the env variable version
#jax.config.update("jax_traceback_filtering", DEBUG)  

# %%
# paths

# training checkpoints
chkpt_dir = Path("/tmp/feedbax-checkpoints")
chkpt_dir.mkdir(exist_ok=True)

# tensorboard
tb_logdir = Path("runs")

model_dir = Path("../models/")


# %% [markdown]
# Define the model.

# %%
def get_model(
        key=None,
        dt=0.05, 
        n_hidden=50, 
        n_steps=50, 
        feedback_delay=0, 
        tau=0.01, 
        out_nonlinearity=jax.nn.sigmoid,
):
    if key is None:
        # in case we just want a skeleton model, e.g. for deserializing
        key = jrandom.PRNGKey(0)
    key1, key2 = jrandom.split(key)
    
    system = TwoLinkMuscled(
        muscle_model=TodorovLiVirtualMuscle(), 
        activator=ActivationFilter(
            tau_act=tau,  
            tau_deact=tau,
        )
    )
    mechanics = Mechanics(system, dt)
    
    feedback_leaves_func = lambda mechanics_state: (
        mechanics_state.system.theta,
        mechanics_state.system.d_theta,
        mechanics_state.effector,         
    )
    
    # joint state feedback + effector state + target state
    n_input = system.twolink.state_size + 2 * N_DIM + 2 * N_DIM
    cell = eqx.nn.GRUCell(n_input, n_hidden, key=key1)
    net = RNN(cell, system.control_size, out_nonlinearity=out_nonlinearity, key=key2)
    body = SimpleFeedback(
        net, 
        mechanics, 
        delay=feedback_delay,  
    )

    model = Recursion(
        body, 
        n_steps,
        feedback_leaves_func,
    )
    
    return model 


# %% [markdown]
# Train the model.

# %%
seed = 5566
key = jrandom.PRNGKey(seed)

n_steps = 50
dt = 0.05 
feedback_delay = 0
workspace = jnp.array([[-0.15, 0.15], 
                       [0.20, 0.50]])
n_hidden  = 50
out_nonlinearity = jax.nn.sigmoid
learning_rate = 0.05

# #! these assume a particular PyTree structure to the states returned by the model
# #! which is why we simply instantiate them 
discount = jnp.linspace(1. / n_steps, 1., n_steps) ** 6
loss_func = fbl.CompositeLoss(
    (
        fbl.EffectorPositionLoss(discount=discount),
        fbl.EffectorFinalVelocityLoss(),
        fbl.ControlLoss(),
        fbl.NetworkActivityLoss(),
    ),
    weights=(1, 0.1, 1e-4, 0.)
)

task = RandomReaches(
    loss_func=loss_func,
    workspace=workspace, 
    n_steps=n_steps,
    eval_grid_n=2,
    eval_n_directions=8,
    eval_reach_length=0.05,
)

get_model = partial(
    get_model,
    dt=dt,
    n_hidden=n_hidden,
    n_steps=n_steps,
    feedback_delay=feedback_delay,
    tau=0.01,
    out_nonlinearity=out_nonlinearity,
)

trainer = TaskTrainer(
    optimizer=optax.inject_hyperparams(optax.adam)(
        learning_rate=learning_rate
    ),
    chkpt_dir=chkpt_dir,
    checkpointing=True,
)

# %%
keys_model = jrandom.split(key, 3)
models = eqx.filter_vmap(get_model)(keys_model)
eqx.filter_vmap(lambda x, y: eqx.combine(x, y), in_axes=(0, None))(
    *eqx.partition(models, eqx.is_array)
)

# %%
trainable_leaves_func = lambda model: (
    model.step.net.cell.weight_hh, 
    model.step.net.cell.weight_ih, 
    model.step.net.cell.bias
)

model, losses, losses_terms = trainer.model_ensemble(
    task=task, 
    get_model=get_model,
    n_model_replicates=3,
    n_batches=1000, 
    batch_size=500, 
    log_step=1,
    trainable_leaves_func=trainable_leaves_func,
    key=key,
)

# %%
plt.loglog(losses.T)

# %%
losses_terms_df = jax.tree_map(
    lambda losses: pd.DataFrame(losses.T, index=range(n_replicates)).melt(
        var_name='Time step', 
        value_name='Loss'
    ),
    dict(losses_terms, total=losses),
)

fig, ax = plt.subplots()
ax.set(xscale='log', yscale='log')
for label, df in losses_terms_df.items():
    sns.lineplot(data=df, x='Time step', y='Loss', errorbar='sd', label=label, ax=ax)
plt.show()

# %%
plot_loglog_losses(losses, losses_terms)
plt.show()

# %% [markdown]
# Evaluate on a centre-out task

# %%
evaluate, eval_endpoints = get_evaluate_func(models, workspace, term_weights=term_weights)
loss, loss_terms, states = eqx.filter_vmap(evaluate)(models)

# %%
fig, _ = plot_states_forces_2d(
                    states[1][0][0], states[1][1][0], states[2][0], eval_endpoints[..., :2], 
                    cmap='plasma', workspace=workspace
)

# %%
# plot entire arm trajectory for an example direction
# convert all joints to Cartesian since I only saved the EE state
xy_pos = eqx.filter_vmap(nlink_angular_to_cartesian)(
    models.step.mechanics.system.twolink, 
    states[0].reshape(-1, 2), 
    states[1].reshape(-1, 2)
)[0].reshape(states[0].shape[0], -1, 2, 2)

# %%
ax = plot_2D_joint_positions(xy_pos[0], add_root=True)
plt.show()

# %% [markdown]
# ## Debugging stuff

# %% [markdown]
# ### Get jaxpr for the loss function

# %%
key = jrandom.PRNGKey(5566)
batch_size = 10
init_state, target_state = uniform_endpoints(key, batch_size, N_DIM, workspace)

filter_spec = jax.tree_util.tree_map(lambda _: False, trained)
filter_spec = eqx.tree_at(
    lambda tree: (tree.step.net.cell.weight_hh, 
                    tree.step.net.cell.weight_ih, 
                    tree.step.net.cell.bias),
    filter_spec,
    replace=(True, True, True)
)     

diff_model, static_model = eqx.partition(trained, filter_spec)

grad_fn = eqx.filter_jit(eqx.filter_value_and_grad(loss_fn, has_aux=True))
jaxpr, _, _ = eqx.filter_make_jaxpr(grad_fn)(diff_model, static_model, init_state, target_state)

# %% [markdown]
# The result is absolutely huge. I guess I should try to repeat this for the un-vmapped model.

# %%
jaxpr
