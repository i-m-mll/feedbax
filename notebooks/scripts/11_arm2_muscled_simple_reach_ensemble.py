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
from pathlib import Path
import sys
from typing import Optional

import diffrax as dfx
import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp 
import jax.random as jrandom
import logging
import matplotlib.pyplot as plt
import numpy as np
import optax 
import pandas as pd
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter
import tqdm
from tqdm import tqdm

from feedbax.mechanics.arm import (
    nlink_angular_to_cartesian, 
    twolink_effector_pos_to_angles
)
from feedbax.mechanics.muscle import (
    ActivationFilter,
    LillicrapScottVirtualMuscle,
    TodorovLiVirtualMuscle, 
)    
from feedbax.mechanics.muscled_arm import TwoLinkMuscled 
from feedbax.mechanics.system import System
from feedbax.networks import RNN
from feedbax.plot import (
    plot_loglog_losses, 
    plot_2D_joint_positions,
    plot_states_forces_2d,
    plot_activity_heatmap
)
from feedbax.task import centreout_endpoints, uniform_endpoints
from feedbax.utils import (
    delete_contents,
    internal_grid_points,
    tree_get_idx, 
    tree_set_idx, 
    tree_sum_squares,
)

# %% [markdown]
# Simple feedback model with a single-layer RNN controlling a two-link arm to reach from a starting position to a target position. 

# %%
DEBUG = False
jax.config.update("jax_debug_nans", False)
jax.config.update("jax_enable_x64", False)

N_DIM = 2

# %%
# paths

# training checkpoints
chkpt_dir = Path("/tmp/jax-checkpoints")
chkpt_dir.mkdir(exist_ok=True)

# tensorboard
tb_logdir = Path(".runs")

model_dir = Path("../models/")


# %% [markdown]
# Define the model components.

# %%
class Mechanics(eqx.Module):
    system: System 
    dt: float = eqx.field(static=True)
    term: dfx.AbstractTerm = eqx.field(static=True)
    solver: Optional[dfx.AbstractSolver] 
    
    def __init__(self, system, dt, solver=None):
        self.system = system
        self.term = dfx.ODETerm(self.system.vector_field)
        if solver is None:
            self.solver = dfx.Tsit5()
        else:
            self.solver = solver
        self.dt = dt        
    
    def __call__(self, state, args):
        inputs, solver_state = args 
        # using (0, dt) for (tprev, tnext) seems fine if there's no t dependency in the system
        state, _, _, solver_state, _ = self.solver.step(
            self.term, 0, self.dt, state, inputs, solver_state, made_jump=False
        )
        # # #! I don't even return solver state, so apparently it's not important
        return state
    
    def init_solver_state(self, state):
        args = inputs_empty = jnp.zeros((self.system.control_size,))
        return self.solver.init(self.term, 0, self.dt, state, args)
    

class SimpleFeedback(eqx.Module):
    """Simple feedback loop with a single RNN and single mechanical system."""
    net: eqx.Module  
    mechanics: Mechanics 
    delay: int = eqx.field(static=True)
    
    def __init__(self, net, mechanics, delay=0):
        self.net = net
        self.mechanics = mechanics
        self.delay = delay + 1  # indexing: delay=0 -> storage len=1, idx=-1
    
    def __call__(self, state, args):
        mechanics_state, _, _, hidden, solver_state = state
        inputs, feedback_state = args
        
        # mechanics state feedback plus task inputs (e.g. target state)
        control, hidden = self.net((inputs, feedback_state), hidden)
        
        mechanics_state = self.mechanics(mechanics_state, (control, solver_state))
        
        ee_state = tuple(arr[:, -1] for arr in nlink_angular_to_cartesian(
            self.mechanics.system.twolink, mechanics_state[0], mechanics_state[1]
        ))
        
        return mechanics_state, ee_state, control, hidden, solver_state
    
    def init_state(self, mechanics_state): 
        
        # # #! how to avoid this here?
        ee_state = tuple(arr[:, -1] for arr in nlink_angular_to_cartesian(
            self.mechanics.system.twolink, mechanics_state[0], mechanics_state[1]
        ))
        
        return (
            mechanics_state, 
            ee_state, 
            jnp.zeros((self.net.out_size,)),
            jnp.zeros((self.net.hidden_size,)),
            self.mechanics.init_solver_state(mechanics_state),   
        )
    

class Recursion(eqx.Module):
    """"""
    step: eqx.Module 
    n_steps: int = eqx.field(static=True)
    
    def __init__(self, step, n_steps):
        self.step = step
        self.n_steps = n_steps        
        
    def _body_func(self, i, x):
        states, args = x 
        # this seems to work, but I'm worried it will break on non-array leaves later
        state = tree_get_idx(states, i)
        
        # # #! this ultimately shouldn't be here, but costs less memory than a `SimpleFeedback`-based storage hack:
        # # #! could put the concatenate inside of `Network`? & pass any pytree of inputs
        feedback = (
            tree_get_idx(states[0][:2], i - self.step.delay),  # omit muscle activation
            tree_get_idx(states[1], i - self.step.delay),
        )
        args = (args[0], feedback)  
        
        state = self.step(state, args)
        
        states = tree_set_idx(states, state, i + 1)
        return states, args
    
    def __call__(self, state, args):
        init_state = self.step.init_state(state)
        
        # # #! part of the feedback hack
        args = (args, jax.tree_map(jnp.zeros_like, (state[:2], state[:2])))
        
        states = self._init_zero_arrays(init_state, args)
        states = tree_set_idx(states, init_state, 0)
        
        if DEBUG: #! jax.debug doesn't work inside of lax loops?
            # this tqdm doesn't show except on an exception, which might be useful
            for i in tqdm(range(self.n_steps),
                          desc="steps"):
                states, args = self._body_func(i, (states, args))
                
            return states, args   
                 
        return lax.fori_loop(
            0, 
            self.n_steps, 
            self._body_func,
            (states, args),
        )
    
    def _init_zero_arrays(self, state, args):
        return jax.tree_util.tree_map(
            lambda x: jnp.zeros((self.n_steps, *x.shape), dtype=x.dtype),
            eqx.filter_eval_shape(self.step, state, args)
        )


# %%
def get_model_generator(
    dt, 
    n_hidden, 
    n_steps, 
    feedback_delay=0, 
    tau=0.01, 
    out_nonlinearity=jax.nn.sigmoid,
):
    """Returns a function that generates models with the given parameters.
    """
    def make_model(key):    
        system = TwoLinkMuscled(
            muscle_model=TodorovLiVirtualMuscle(), 
            activator=ActivationFilter(
                tau_act=tau,  
                tau_deact=tau,
            )
        )
        mechanics = Mechanics(system, dt)
        # target state + feedback: angular pos & vel of joints & cartesian EE 
        n_input = system.twolink.state_size * 2 + N_DIM * 2  
        net = RNN(n_input, system.control_size, n_hidden, key=key, out_nonlinearity=out_nonlinearity)
        body = SimpleFeedback(net, mechanics, delay=feedback_delay)

        return Recursion(body, n_steps)
    
    make_ensemble = eqx.filter_vmap(make_model)
    
    return make_ensemble


# %%
n_replicates = 3

make_ensemble = get_model_generator(0.1, 50, 50)
key = jrandom.PRNGKey(0)
keys = jrandom.split(key, n_replicates)
models = make_ensemble(keys)

optim = optax.adam(1e-3)

opt_states = eqx.filter_vmap(lambda model: optim.init(eqx.filter(model, eqx.is_array)))(
    models
)   

opt_states


# %% [markdown]
# Define the loss function and training loop.

# %%
def loss_fn(
    diff_model, 
    static_model, 
    init_state, 
    target_state, 
    term_weights=dict(
        position=1., 
        final_velocity=1., 
        control=1e-5, 
        hidden=1e-6, 
    ),
    weight_decay=1e-4,
    discount=1.,
):  
    """Quadratic in states, controls, and hidden activities.
    
    Assumes the `target_state` is fixed; i.e. this is not a tracking task.
    
    User can apply a temporal discount broadcastable by elementwise multiplication with `(n_batch, n_steps)`.
    """
    model = eqx.combine(diff_model, static_model)  
    batched_model = jax.vmap(model)

    # dataset gives init state in terms of effector position, but we need joint angles
    init_joints_pos = eqx.filter_vmap(twolink_effector_pos_to_angles)(
        model.step.mechanics.system.twolink, init_state
    )
    # #! assumes zero initial velocity; TODO convert initial velocity also
    # TODO: `System` should provide a method for this?
    init_state = (
        init_joints_pos, 
        jnp.zeros_like(init_joints_pos),  
        jnp.zeros((init_joints_pos.shape[0], 
                   model.step.mechanics.system.control_size)),  # per-muscle activation
    )
    
    (joints_states, ee_states, controls, activities, _), _ = batched_model(
        init_state, target_state
    )
    
    states = ee_states  # operational space loss
  
    # sum over xyz, apply temporal discount, sum over time
    position_loss = jnp.sum(discount * jnp.sum((states[0] - target_state[:, None, :2]) ** 2, axis=-1), axis=-1)
    
    loss_terms = dict(
        #final_position=jnp.sum((states[..., -1, :2] - target_state[..., :2]) ** 2, axis=-1).squeeze(),  # sum over xyz
        position=position_loss,  
        final_velocity=jnp.sum((states[1][:, -1] - target_state[..., 2:]) ** 2, axis=-1).squeeze(),  # over xyz
        control=jnp.sum(controls ** 2, axis=(-1, -2)),  # over control variables and time
        hidden=jnp.sum(activities ** 2, axis=(-1, -2)),  # over network units and time
    )
    
    # mean over batch
    loss_terms = jax.tree_map(lambda x: jnp.mean(x, axis=0), loss_terms)
    # term scaling
    loss_terms = jax.tree_map(lambda term, weight: term * weight, loss_terms, term_weights) 
    
    # NOTE: optax also gives optimizers that implement weight decay
    if weight_decay is not None:
        # this is separate because the tree map of `jnp.mean` doesn't like floats
        # and it doesn't make sense to batch-mean the model parameters anyway
        loss_terms['weight_decay'] = weight_decay * tree_sum_squares(diff_model)
        
    # sum over terms
    loss = jax.tree_util.tree_reduce(lambda x, y: x + y, loss_terms)
    
    return loss, loss_terms


# %%
def get_evaluate_func(
    model,
    workspace, 
    n_directions=8, 
    reach_length=0.05,
    discount=1.,
    term_weights=None,
):
    """Prepare the center-out task for evaluating the model.
    
    Returns a function that takes a model and returns losses and a figure.
    """
    centers = internal_grid_points(workspace, 2)
    state_endpoints = jnp.concatenate([
        centreout_endpoints(jnp.array(center), n_directions, 0, reach_length) 
        for center in centers
    ], axis=1)

    target_states = state_endpoints[1]
    init_joints_pos = eqx.filter_vmap(twolink_effector_pos_to_angles)(
        model.step.mechanics.system.twolink, state_endpoints[0, :, :2]
    )
    # #! assumes zero initial velocity; TODO convert initial velocity also
    init_states = (
        init_joints_pos, 
        jnp.zeros_like(init_joints_pos),
        jnp.zeros((init_joints_pos.shape[0], 
                model.step.mechanics.system.control_size)),
    )

    def loss_fn(states, controls, activities):
        # TODO: implementing `Loss` as a class would stop this from repeating the other cost function?
        position_loss = jnp.sum(discount * jnp.sum((states[0] - target_states[:, None, :2]) ** 2, axis=-1), axis=-1)
    
        loss_terms = dict(
            #final_position=jnp.sum((states[..., -1, :2] - target_state[..., :2]) ** 2, axis=-1).squeeze(),  # sum over xyz
            position=position_loss,  
            final_velocity=jnp.sum((states[1][:, -1] - target_states[..., 2:]) ** 2, axis=-1).squeeze(),  # over xyz
            control=jnp.sum(controls ** 2, axis=(-1, -2)),  # over control variables and time
            hidden=jnp.sum(activities ** 2, axis=(-1, -2)),  # over network units and time
        )
    
        # mean over batch
        loss_terms = jax.tree_map(lambda x: jnp.mean(x, axis=0), loss_terms)
        if term_weights is not None:
            # term scaling
            loss_terms = jax.tree_map(lambda term, weight: term * weight, loss_terms, term_weights)
        
        loss = jax.tree_util.tree_reduce(lambda x, y: x + y, loss_terms)
        
        return loss, loss_terms

    def evaluate(trained):
        states, _ = jax.vmap(trained)(
            init_states, target_states
        )
        
        loss, loss_terms = loss_fn(states[1], states[2], states[3])
        
        return loss, loss_terms, states

    return evaluate, state_endpoints


# %%
def train(
    n_replicates=1,
    n_steps=100,
    dt=0.05,
    feedback_delay_steps=5,
    workspace = jnp.array([[-0.2, 0.2], [0.10, 0.50]]),
    batch_size=500,
    n_batches=2500,
    epochs=1,
    learning_rate=1e-2,
    term_weights=dict(
        position=1., 
        final_velocity=1., 
        control=1e-5, 
        hidden=1e-6, 
    ),
    weight_decay=1e-4,
    hidden_size=50,
    seed=5566,
    log_step=100,
    restore_checkpoint=False,
):
    key = jrandom.PRNGKey(seed)  # for data generation
    
    keys = jrandom.split(key, n_replicates)  # for models initialization
    
    make_ensemble = get_model_generator(
        dt, hidden_size, n_steps, feedback_delay=feedback_delay_steps
    )

    models = make_ensemble(keys)

    def get_batch(batch_size, key):
        """Segment endpoints uniformly distributed in a rectangular workspace."""
        return uniform_endpoints(key, batch_size, N_DIM, workspace)
    
    # only train the RNN layer (input weights & hidden weights and biases)
    filter_spec = jax.tree_util.tree_map(lambda _: False, models)
    filter_spec = eqx.tree_at(
        lambda tree: (tree.step.net.cell.weight_hh, 
                      tree.step.net.cell.weight_ih, 
                      tree.step.net.cell.bias),
        filter_spec,
        replace=(True, True, True)
    )     
    
    position_error_discount = jnp.linspace(1./n_steps, 1., n_steps) ** 6
    evaluate, eval_endpoints = get_evaluate_func(
        models, 
        workspace, 
        discount=position_error_discount, 
        term_weights=term_weights
    )
    
    # prepare training machinery
    optim = optax.adam(learning_rate)
    
    # #! vmap the ensembles of models onto the same data
    @eqx.filter_vmap(in_axes=(eqx.if_array(0), None, None, eqx.if_array(0)))
    def train_step(model, init_state, target_state, opt_state):
        diff_model, static_model = eqx.partition(model, filter_spec)
        (loss, loss_terms), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
            diff_model, static_model, init_state, target_state, 
            discount=position_error_discount, term_weights=term_weights,
            weight_decay=weight_decay,
        )
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, loss_terms, model, opt_state
    
    if not DEBUG:
        train_step = eqx.filter_jit(train_step)
    
    # tensorboard setup    
    writer = SummaryWriter(tb_logdir)
    # display loss terms in the same figure under "Custom Scalars"
    layout = {
        "Loss terms": {
            "Training loss": ["Multiline", ["Loss/train"] + [f'Loss/train/{term}'
                                            for term in term_weights.keys()]],
            "Evaluation loss": ["Multiline", ["Loss/eval"] + [f'Loss/eval/{term}'
                                              for term in term_weights.keys()]],
        },
    }
    writer.add_custom_scalars(layout)    
    
    # def get_loss_arrays(n_batches, term_weights):
    #     pass 
    
    losses = jnp.empty((n_batches, n_replicates))
    losses_terms = dict(zip(
        term_weights.keys(), 
        [jnp.empty((n_batches, n_replicates)) for _ in term_weights]
    ))
    
    if restore_checkpoint:
        with open(chkpt_dir / "last_batch.txt", 'r') as f:
            last_batch = int(f.read()) 
            
        start_batch = last_batch + 1
        
        models = eqx.tree_deserialise_leaves(chkpt_dir / f'model{last_batch}.eqx', models)
        losses, losses_terms = eqx.tree_deserialise_leaves(
            chkpt_dir / f'losses{last_batch}.eqx', 
            (losses, losses_terms),
        )

        print(f"Restored checkpoint from training step {last_batch}")
        
    else:
        start_batch = 0
        delete_contents(chkpt_dir)  
        
    # TODO: should also restore this from checkpoint
    opt_states = eqx.filter_vmap(lambda model: optim.init(eqx.filter(model, eqx.is_array)))(
        models
    )    

    #for _ in range(epochs): #! assume 1 epoch (no fixed dataset)
    for batch in tqdm(range(start_batch, n_batches),
                      desc='batch', initial=start_batch, total=n_batches):
        key, _ = jrandom.split(key)
        # TODO: I think `init_state` isn't a tuple here but the old concatenated version...
        init_state, target_state = get_batch(batch_size, key)
        
        loss, loss_terms, models, opt_states = train_step(
            models, init_state, target_state, opt_states
        )
        
        losses = losses.at[batch].set(loss)
        losses_terms = tree_set_idx(losses_terms, loss_terms, batch)
        
        # tensorboard losses on every iteration
        # #! just report for one of the replicates
        writer.add_scalar('Loss/train', loss[0].item(), batch)
        for term, loss_term in loss_terms.items():
            writer.add_scalar(f'Loss/train/{term}', loss_term[0].item(), batch)
        
        if jnp.sum(jnp.isnan(loss)) > n_replicates // 2:
            raise ValueError(f"\nNaN loss on more than 50% of replicates at batch {batch}!")
        
        # checkpointing and evaluation occasionally
        if (batch + 1) % log_step == 0:
            # model checkpoint
            eqx.tree_serialise_leaves(chkpt_dir / f'models{batch}.eqx', models)
            eqx.tree_serialise_leaves(chkpt_dir / f'losses{batch}.eqx', 
                                      (losses, losses_terms))
            with open(chkpt_dir / "last_batch.txt", 'w') as f:
                f.write(str(batch)) 
            
            # tensorboard
            loss_eval, loss_eval_terms, states = eqx.filter_vmap(evaluate)(models)
            fig, _ = plot_states_forces_2d(
                    states[1][0][0], states[1][1][0], states[2][0], eval_endpoints[..., :2], 
                    cmap='plasma', workspace=workspace
            )
        
            writer.add_figure('Eval/centerout', fig, batch)
            writer.add_scalar('Loss/eval', loss_eval[0].item(), batch)
            for term, loss_term in loss_eval_terms.items():
                writer.add_scalar(f'Loss/eval/{term}', loss_term[0].item(), batch)
                
            tqdm.write(f"step: {batch}, training loss: {loss[0]:.4f}", file=sys.stderr)
            tqdm.write(f"step: {batch}, center out loss: {loss_eval[0]:.4f}", file=sys.stderr)
    
    # TODO: run logging: save evaluation figure, loss curve, commit ID, date, etc. along with model
    eqx.tree_serialise_leaves(model_dir / f'models_final.eqx', models)
    
    return models, losses, losses_terms


# %% [markdown]
# Train the model.

# %%
workspace = jnp.array([[-0.15, 0.15], 
                       [0.20, 0.50]])

term_weights = dict(
    position=1., 
    final_velocity=1., 
    control=1e-4, 
    hidden=0., 
)

n_replicates=10
n_steps = 50

# %%
models, losses, losses_terms = train(
    n_replicates=n_replicates,
    batch_size=500, 
    dt=0.05, 
    feedback_delay_steps=0,
    n_batches=7000, 
    n_steps=n_steps, 
    hidden_size=50, 
    seed=5566,
    learning_rate=0.05,
    log_step=500,
    workspace=workspace,
    term_weights=term_weights,
    weight_decay=None,
    restore_checkpoint=False,
)

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
