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
import os

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from datetime import datetime
from functools import cached_property
from pathlib import Path
import sys
from typing import Optional

import diffrax as dfx
import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp 
import jax.random as jrandom
import matplotlib.pyplot as plt
import numpy as np
import optax 
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
    plot_activity_heatmap,
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
tb_logdir = Path("runs")

model_dir = Path("../models/")

# %%
TB_PREFIX = "nb10"

# %%
aa = list(jnp.arange(100))
bb = jnp.array(101, dtype=jnp.int32)
def test(aa):
    aa.pop(0)
    aa.append(bb)
    return aa
    
print(test(test(aa)))


# %%
aa = jnp.arange(100)
# %timeit jnp.roll(aa, -1, axis=0)

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
    
    def __call__(self, input, state, key):
        system_state, solver_state = state
        # using (0, dt) for (tprev, tnext) seems fine if there's no t dependency in the system
        mechanics_state, _, _, solver_state, _ = self.solver.step(
            self.term, 0, self.dt, mechanics_state, input, solver_state, made_jump=False
        )
        return input, (system_state, solver_state)
    
    def init_state(self, input, state, key):
        # #! init from self.mechanics.init_state if none given?
        inputs_zero = jnp.zeros((self.system.control_size,))
        return (state, 
                self.solver.init(self.term, 0, self.dt, state, inputs_zero))


class Wire(eqx.Module):
    """Connection delay implemented as a queue, with added noise.
    
    A list implementation is faster than modifying a JAX array.
    
    TODO: 
    - Infer delay steps from time.
    """
    delay: int 
    noise_std: float 
    
    def __call__(self, input, state, key):
        # state = jax.tree_map(lambda x: jnp.roll(x, -1, axis=0), state)
        # state = tree_set_idx(state, self._add_noise(input, key), -1)
        _, queue = state
        queue.append(input)
        output = self._add_noise(queue.pop(0), key)
        return input, (output, queue), key 
    
    def init_state(self, input, state, key):
        # return jax.tree_map(
        #     lambda x: jnp.zeros((self.delay, *x.shape), dtype=x.dtype),
        #     input
        # )
        input_zeros = jax.tree_map(jnp.zeros_like, input)
        return (input_zeros, 
                (self.delay - 1) * [input_zeros] + [input])
        
    @cached_property
    def _add_noise(self):
        if self.noise_std is None:
            return lambda x: x
        else:
            return self.__add_noise 
    
    def __add_noise(self, x, key):
        return x + self.noise_std * jrandom.normal(key, x.shape) 


class SimpleFeedback(eqx.Module):
    """Simple feedback loop with a single RNN and single mechanical system."""
    net: eqx.Module  
    mechanics: Mechanics 
    afferent: Wire
    
    def __init__(self, net, mechanics, afferent):
        self.net = net
        self.mechanics = mechanics
        self.afferent = afferent
    
    def __call__(self, input, state, key):
        mechanics_state, _, _, net_state, afferent_state = state
        
        # #! if we split the key multiple times here, won't we end up with the 
        # #! same keys used in subsequent steps in `Recursion`?
        key1, key2 = jrandom.split(key)        
        
        # mechanics state feedback plus task inputs (e.g. target state)
        control, net_state = self.net((input, afferent_state), net_state, key1)
        
        _, mechanics_state = self.mechanics(control, mechanics_state)
        
        # #! wouldn't need to worry about referencing `self.mechanics...twolink` 
        # #! if `nlink_angular_to_cartesian` was a method of an `NLink` class
        ee_state = tuple(arr[:, -1] for arr in nlink_angular_to_cartesian(
            self.mechanics.system.twolink, mechanics_state[0], mechanics_state[1]
        ))
        
        afferent_state = self.afferent((mechanics_state[:2], ee_state), afferent_state, key2)
        
        state = (mechanics_state, ee_state, control, net_state, afferent_state)
        
        return input, state, key1
    
    def init_state(self, state, key): 
        mechanics_state, _, _, _, _, _ = state        
        system_state, _ = mechanics_state
        
        # #! how to avoid this here? "Vision" module?
        ee_state = tuple(arr[:, -1] for arr in nlink_angular_to_cartesian(
            self.mechanics.system.twolink, system_state[0], system_state[1]
        ))
        
        return (
            system_state, 
            ee_state, 
            jnp.zeros((self.net.out_size,)),
            jnp.zeros((self.net.hidden_size,)),
            self.afferent.init_state((system_state[:2], ee_state)),
        )
    

class Recursion(eqx.Module):
    """"""
    step: eqx.Module 
    n_steps: int = eqx.field(static=True)
    
    def __init__(self, step, n_steps):
        self.step = step
        self.n_steps = n_steps        
        
    def _body_func(self, i, x):
        inputs, states, key = x
        
        _, key = jrandom.split(key)
         
        state = tree_get_idx(states, i) 
        # #! todo: pytree input with indexing! e.g. for moving target
        input = inputs 
        input, state, key = self.step(input, state, key)
        states = tree_set_idx(states, state, i + 1)
        
        return inputs, states, key
    
    def __call__(self, input, state, key):
        # #! could give `Recursion.init_state` and initialize `states` there, including that for `step`
        init_state = self.step.init_state(state)
        
        # # #! part of the feedback hack
        # args = (jax.tree_map(jnp.zeros_like, (state[:2], state[:2])),)
        
        # #! todo: use an eqx filter to determine which states get memorized;
        # #! self.step should be the one to tell it so, probably;
        # #! for the others, just keep and pass the current state
        states = self._init_zero_arrays(input, init_state, key)
        states = tree_set_idx(states, init_state, 0)
        
        if DEBUG: 
            # this tqdm doesn't show except on an exception, which might be useful
            for i in tqdm(range(self.n_steps),
                          desc="steps"):
                input, states, key = self._body_func(i, (input, states, key))
                
            return input, states, key    
                 
        input, states, key = lax.fori_loop(
            0, 
            self.n_steps, 
            self._body_func,
            (input, states, key),
        )
        
        return states
    
    def init_state(self, input, state, key):
        # TODO: would it be faster to use placeholders in a list?
        return jax.tree_util.tree_map(
            lambda x: jnp.zeros((self.n_steps, *x.shape), dtype=x.dtype),
            eqx.filter_eval_shape(self.step, input, state, key)
        )


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
    
    system = TwoLinkMuscled(
        muscle_model=TodorovLiVirtualMuscle(), 
        activator=ActivationFilter(
            tau_act=tau,  
            tau_deact=tau,
        )
    )
    mechanics = Mechanics(system, dt)
    # target state + feedback: angular pos & vel of joints & cartesian EE 
    # #! this should be inferred by `Body` based on the desired wiring and knowledge of the task structure
    n_input = system.twolink.state_size * 2 + N_DIM * 2  
    net = RNN(n_input, system.control_size, n_hidden, key=key, out_nonlinearity=out_nonlinearity)
    body = SimpleFeedback(net, mechanics, delay=feedback_delay)

    return Recursion(body, n_steps)


# %% [markdown]
# Define the loss function and training loop.

# %%
def loss_fn(
    diff_model, 
    static_model, 
    init_state, 
    target_state, 
    key,
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
    
    # #! stuff after this point is largely model-specific
    batched_model = jax.vmap(model, in_axes=(0, 0, None))  #? `in_axes` are model-specific?
    
    # dataset gives init state in terms of effector position, but we need joint angles
    init_joints_pos = eqx.filter_vmap(twolink_effector_pos_to_angles)(
        model.step.mechanics.system.twolink, init_state
    )
    # #! assumes zero initial velocity; TODO convert initial velocity also
    # TODO: the model should provide a way to initialize this, given partial user input
    init_state = (
        init_joints_pos, 
        jnp.zeros_like(init_joints_pos),  
        jnp.zeros((init_joints_pos.shape[0], 
                   model.step.mechanics.system.control_size)),  # per-muscle activation
    )
    
    (joints_states, ee_states, controls, activities, _), _, _ = batched_model(
        init_state, target_state, key
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
    
    # #! stuff after this point isn't model-specific
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

    @eqx.filter_jit
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
    
    batched_model = eqx.filter_jit(jax.vmap(model, in_axes=(0, 0, None)))

    def evaluate(model, key):
        (states, ee_states, controls, activities, _), _, _ = batched_model(
            init_states, target_states, key
        )
        
        loss, loss_terms = loss_fn(ee_states, controls, activities)
        
        fig, _ = plot_states_forces_2d(
            ee_states[0], ee_states[1], controls[:, 2:, -2:], state_endpoints[..., :2], 
            force_labels=('Biarticular controls', 'Flexor', 'Extensor'), 
            cmap='plasma', workspace=workspace
        )
        
        return loss, loss_terms, states, controls, activities, fig

    return evaluate


# %%
def train(
    model=None,  # start from existing model
    n_steps=100,
    dt=0.05,
    feedback_delay_steps=5,
    workspace = jnp.array([[-0.2, 0.2], 
                           [0.10, 0.50]]),
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
    restore_checkpoint=False,  # should be exclusive with `model is not None`
):
    key = jrandom.PRNGKey(seed)
    
    if model is None:
        model = get_model(key, dt, hidden_size, n_steps, 
                          feedback_delay=feedback_delay_steps)

    def get_batch(batch_size, key):
        """Segment endpoints uniformly distributed in a rectangular workspace."""
        return uniform_endpoints(key, batch_size, N_DIM, workspace)
    
    # only train the RNN layer (input weights & hidden weights and biases)
    filter_spec = jax.tree_util.tree_map(lambda _: False, model)
    filter_spec = eqx.tree_at(
        lambda tree: (tree.step.net.cell.weight_hh, 
                      tree.step.net.cell.weight_ih, 
                      tree.step.net.cell.bias),
        filter_spec,
        replace=(True, True, True)
    )     
    
    position_error_discount = jnp.linspace(1./n_steps, 1., n_steps) ** 6
    evaluate = get_evaluate_func(
        model, 
        workspace, 
        discount=position_error_discount, 
        term_weights=term_weights
    )
    
    # prepare training machinery
    optim = optax.adam(learning_rate)
    
    def train_step(model, init_state, target_state, opt_state, key):
        diff_model, static_model = eqx.partition(model, filter_spec)
        (loss, loss_terms), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
            diff_model, static_model, init_state, target_state, key, 
            discount=position_error_discount, term_weights=term_weights,
            weight_decay=weight_decay,
        )
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, loss_terms, model, opt_state
    
    if not DEBUG:
        train_step = eqx.filter_jit(train_step)
    
    # tensorboard setup   
    timestr = datetime.today().strftime("%Y%m%d-%H%M%S") 
    writer = SummaryWriter(tb_logdir / f"{timestr}_{TB_PREFIX}")
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
    
    losses = jnp.empty((n_batches,))
    losses_terms = dict(zip(
        term_weights.keys(), 
        [jnp.empty((n_batches,)) for _ in term_weights]
    ))
    
    def get_last_checkpoint():
        with open(chkpt_dir / "last_batch.txt", 'r') as f:
            last_batch = int(f.read()) 
            
        model = eqx.tree_deserialise_leaves(chkpt_dir / f'model{last_batch}.eqx', model)
        losses, losses_terms = eqx.tree_deserialise_leaves(
            chkpt_dir / f'losses{last_batch}.eqx', 
            (losses, losses_terms),
        )
        return last_batch, model, losses, losses_terms
    
    if restore_checkpoint:
        last_batch, model, losses, losses_terms = get_last_checkpoint()
        start_batch = last_batch + 1
        print(f"Restored checkpoint from training step {last_batch}")
    else:
        start_batch = 1
        delete_contents(chkpt_dir)  
        
    # TODO: should also restore this from checkpoint
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    #for _ in range(epochs): #! assume 1 epoch (no fixed dataset)
    # batch is 1-indexed for printing and logging purposes (batch 100 is the 100th batch)
    for batch in tqdm(range(start_batch, n_batches + 1),
                      desc='batch', initial=start_batch, total=n_batches):
        keyb, keyt, keye = jrandom.split(key, 3)
        # TODO: I think `init_state` isn't a tuple here but the old concatenated version...
        init_state, target_state = get_batch(batch_size, keyb)
        
        loss, loss_terms, model, opt_state = train_step(
            model, init_state, target_state, opt_state, keyt
        )
        
        losses = losses.at[batch].set(loss)
        losses_terms = tree_set_idx(losses_terms, loss_terms, batch)
        
        # tensorboad
        writer.add_scalar('Loss/train', loss.item(), batch)
        for term, loss_term in loss_terms.items():
            writer.add_scalar(f'Loss/train/{term}', loss_term.item(), batch)
        
        if jnp.isnan(loss):
            last_batch, model, losses, losses_terms = get_last_checkpoint()
            print(f"\nNaN loss at batch {batch}!")
            print(f"Returning checkpoint from batch {last_batch}.")
            return model, losses, losses_terms
        
        if batch % log_step == 0:
            # model checkpoint
            eqx.tree_serialise_leaves(chkpt_dir / f'model{batch}.eqx', model)
            eqx.tree_serialise_leaves(chkpt_dir / f'losses{batch}.eqx', 
                                      (losses, losses_terms))
            with open(chkpt_dir / "last_batch.txt", 'w') as f:
                f.write(str(batch)) 
            
            # tensorboard
            loss_eval, loss_eval_terms, _, _, _, fig = evaluate(model, keye)
            writer.add_figure('Eval/centerout', fig, batch)
            writer.add_scalar('Loss/eval', loss_eval.item(), batch)
            for term, loss_term in loss_eval_terms.items():
                writer.add_scalar(f'Loss/eval/{term}', loss_term.item(), batch)
                
            tqdm.write(f"step: {batch}, training loss: {loss:.4f}", file=sys.stderr)
            tqdm.write(f"step: {batch}, center out loss: {loss_eval:.4f}", file=sys.stderr)
    
    # TODO: run logging: save evaluation figure, loss curve, commit ID, date, etc. along with model
    eqx.tree_serialise_leaves(model_dir / f'model_final.eqx', model)
    
    return model, losses, losses_terms


# %% [markdown]
# Train the model.

# %%
workspace = jnp.array([[-0.15, 0.15], 
                       [0.20, 0.50]])

term_weights = dict(
    position=1., 
    final_velocity=0.1, 
    control=1e-4, 
    hidden=0., 
)

# %%
model, losses, losses_terms = train(
    batch_size=500, 
    dt=0.1, 
    feedback_delay_steps=0,
    n_batches=10000, 
    n_steps=50, 
    hidden_size=50, 
    seed=5566,
    learning_rate=0.1,
    log_step=500,
    workspace=workspace,
    term_weights=term_weights,
    weight_decay=None,
    restore_checkpoint=False,
)

# %%
plot_loglog_losses(losses, losses_terms)
plt.show()

# %% [markdown]
# Optionally, load an existing model

# %%
model = get_model()
model = eqx.tree_deserialise_leaves(model_dir / f'model_final.eqx', model)

# %% [markdown]
# Evaluate on a centre-out task

# %%
evaluate = get_evaluate_func(model, workspace, term_weights=term_weights)
loss, loss_terms, states, controls, activities, fig = evaluate(model, key=jrandom.PRNGKey(0))

# %% [markdown]
# Plot entire arm trajectory for an example direction

# %%
idx = 1

# %%
# convert all joints to Cartesian since I only saved the EE state
xy_pos = eqx.filter_vmap(nlink_angular_to_cartesian)(
    model.step.mechanics.system.twolink, states[0].reshape(-1, 2), states[1].reshape(-1, 2)
)[0].reshape(states[0].shape[0], -1, 2, 2)

# %%
ax = plot_2D_joint_positions(xy_pos[idx], add_root=True)
plt.show()

# %% [markdown]
# Network hidden activities over time for the same example reach direction

# %%
# semilogx is interesting in this case without a GO cue
# ...helps to visualize the initial activity
fig, ax = plt.subplots(1, 1)
ax.semilogx(activities[idx])
ax.set_xlabel('Time step')
ax.set_ylabel('Hidden unit activity')
plt.show()

# %% [markdown]
# Heatmap of network activity over time for an example direction

# %%
plot_activity_heatmap(activities[2], cmap='viridis')

# %% [markdown]
# ## Debugging stuff

# %% [markdown]
# ### Get jaxpr for the loss function

# %%
key = jrandom.PRNGKey(5566)
batch_size = 10
init_state, target_state = uniform_endpoints(key, batch_size, N_DIM, workspace)

filter_spec = jax.tree_util.tree_map(lambda _: False, model)
filter_spec = eqx.tree_at(
    lambda tree: (tree.step.net.cell.weight_hh, 
                    tree.step.net.cell.weight_ih, 
                    tree.step.net.cell.bias),
    filter_spec,
    replace=(True, True, True)
)     

diff_model, static_model = eqx.partition(model, filter_spec)

grad_fn = eqx.filter_jit(eqx.filter_value_and_grad(loss_fn, has_aux=True))
jaxpr, _, _ = eqx.filter_make_jaxpr(grad_fn)(diff_model, static_model, init_state, target_state)

# %% [markdown]
# The result is absolutely huge. I guess I should try to repeat this for the un-vmapped model.

# %%
jaxpr
