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
# %load_ext autoreload
# %autoreload 2

# %%
import os
import logging
import sys

from IPython import get_ipython

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

# redirect stderr (e.g. warnings) to file
stderr_log = sys.stderr = open('log/stderr.log', 'w')
get_ipython().log.handlers[0].stream = stderr_log 
get_ipython().log.setLevel(logging.INFO)

# %%
import dataclasses
from dataclasses import dataclass
from datetime import datetime
import functools
import json
import logging
from pathlib import Path
import sys
from typing import Optional

import diffrax as dfx
import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp 
import jax.random as jrandom
from jaxtyping import Array, Float, PyTree
import matplotlib.pyplot as plt
import numpy as np
import optax 
from torch.utils.tensorboard import SummaryWriter
import tqdm
from tqdm.auto import tqdm

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
    catchtime,
    delete_contents,
    git_commit_id,
    internal_grid_points,
    tree_get_idx, 
    tree_set_idx, 
    tree_sum_squares,
)

# %% [markdown]
# Simple feedback model with a single-layer RNN controlling a two-link arm to reach from a starting position to a target position. 

# %%
DEBUG = False

logging.getLogger("jax").setLevel(logging.INFO)

jax.config.update("jax_debug_nans", DEBUG)
jax.config.update("jax_enable_x64", False)

# not sure if this will work or if I need to use the env variable version
#jax.config.update("jax_traceback_filtering", DEBUG)  

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
NB_PREFIX = "nb10"

# %% [markdown]
# Define the model components.

# %%
from abc import ABCMeta


class _StateMeta(ABCMeta): 
    """Based on `eqx._module._ModuleMeta`."""
    
    def __new__(mcs, name, bases, dict_, **kwargs):
        # #? I think this works to add slots for dataclass fields as long as no default values supplied
        # TODO: test performance difference
        dict_['__slots__'] = tuple(dict_['__annotations__'].keys())
        
        cls = super().__new__(mcs, name, bases, dict_, **kwargs)
        
        cls = dataclass(eq=False, repr=False)(
            cls
        )
        
        def datacls_flatten(datacls):
            field_names, field_values = zip(*[
                #(field_.name, datacls.__dict__[field_.name])
                (field_.name, getattr(datacls, field_.name))
                for field_ in dataclasses.fields(datacls)
            ])
            
            # for field_ in dataclasses.fields(datacls):
            #     field_names.append(field_.name)
            #     field_values.append(datacls.__dict__[field_.name])
                              
            children, aux = tuple(field_values), tuple(field_names)
            return children, aux
        
        def datacls_unflatten(cls, aux, children):
            field_names, field_values = aux, children
            datacls = object.__new__(cls)
            for name, value in zip(field_names, field_values):
                object.__setattr__(datacls, name, value)
            return datacls 
        
        jax.tree_util.register_pytree_node(
            cls, 
            flatten_func=datacls_flatten,
            unflatten_func=functools.partial(datacls_unflatten, cls),
        )
        
        return cls
        
class AbstractState(metaclass=_StateMeta):
    """Base class for state dataclasses.
    
    Automatically instantiated as a dataclass with slots.
    
    Based on 
    """
    __annotations__ = dict()
    
    def __hash__(self):
        return hash(tuple(jax.tree_util.tree_leaves(self)))



# %%
class MechanicsState(AbstractState):
    system: PyTree[Array]
    solver: PyTree 


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
    
    def __call__(self, input, state: MechanicsState):
        # using (0, dt) for (tprev, tnext) seems fine if there's no t dependency in the system
        system_state, _, _, solver_state, _ = self.solver.step(
            self.term, 
            0, 
            self.dt, 
            state.system, 
            input, 
            state.solver, 
            made_jump=False,
        )
        # # #! I don't even return solver state, so apparently it's not important
        return MechanicsState(system_state, solver_state)
    
    def init(self, system_state, input=None, key=None):
        args = inputs_empty = jnp.zeros((self.system.control_size,))
        return MechanicsState(
            system_state,  # self.system.init()
            self.solver.init(self.term, 0, self.dt, system_state, args),
        )


class SimpleFeedbackState(AbstractState):
    mechanics: MechanicsState
    ee: PyTree[Array]
    control: Array
    hidden: PyTree


class SimpleFeedback(eqx.Module):
    """Simple feedback loop with a single RNN and single mechanical system."""
    net: eqx.Module  
    mechanics: Mechanics 
    delay: int = eqx.field(static=True)
    #perturbation: Optional[eqx.Module]
    
    def __init__(self, net, mechanics, delay=0):
        self.net = net
        self.mechanics = mechanics
        self.delay = delay + 1  # indexing: delay=0 -> storage len=1, idx=-1
    
    def __call__(
        self, 
        input, 
        state: SimpleFeedbackState, 
        args, 
        key
    ) -> SimpleFeedbackState:
        
        feedback_state = args  #! part of feedback hack
        
        key1, key2 = jrandom.split(key)
        
        # mechanics state feedback plus task inputs (e.g. target state)
        control, hidden = self.net((input, feedback_state), state.hidden, key1)
        
        # TODO: construct pytree of controls + extra inputs
        # TODO: transform extra inputs to appropriate forces
        
        mechanics_state = self.mechanics(state.control, state.mechanics)        
        
        # TODO: could be replaced with a general call to a `Mechanics` method that knows about the operational space
        twolink = self.mechanics.system.twolink
        ee_state = tuple(arr[:, -1] for arr in twolink.forward_kinematics(
            mechanics_state.system[0], mechanics_state.system[1]
        ))
        
        return SimpleFeedbackState(mechanics_state, ee_state, control, hidden)
    
    def init(self, ee_state): 
              
        # dataset gives init state in terms of effector position, but we need joint angles
        init_joints_pos = self.mechanics.system.twolink.inverse_kinematics(
            ee_state[0]
        )
        # TODO the tuple structure of pos-vel should be introduced in data generation, and kept throughout
        # #! assumes zero initial velocity; TODO convert initial velocity also
        system_state = (
            init_joints_pos, 
            jnp.zeros_like(init_joints_pos),  
            jnp.zeros((self.mechanics.system.control_size,)),  # per-muscle activation
        )

        return SimpleFeedbackState(
            self.mechanics.init(system_state),
            ee_state,
            jnp.zeros((self.net.out_size,)),
            self.net.init(),
        )


class Recursion(eqx.Module):
    """"""
    step: eqx.Module 
    n_steps: int = eqx.field(static=True)
    
    def __init__(self, step, n_steps):
        self.step = step
        self.n_steps = n_steps        
        
    def _body_func(self, i, x):
        input, states, key = x
        
        key1, key2 = jrandom.split(key)
        
        # # #! this ultimately shouldn't be here, but costs less memory than a `SimpleFeedback`-based storage hack:
        feedback = (
            tree_get_idx(states.mechanics.system[:2], i - self.step.delay),  # omit muscle activation
            tree_get_idx(states.ee, i - self.step.delay),  # ee state
        )
        args = feedback
        
        state = tree_get_idx(states, i)
        state = self.step(input, state, args, key1)
        states = tree_set_idx(states, state, i + 1)
        
        return input, states, key2
    
    def __call__(self, input, system_state, key):
        key1, key2, key3 = jrandom.split(key, 3)
        
        state = self.step.init(system_state) #! maybe this should be outside
        
        #! `args` is vestigial. part of the feedback hack
        args = jax.tree_map(jnp.zeros_like, (state.mechanics.system[:2], state.ee))
        
        states = self.init(input, state, args, key2)
        
        if DEBUG: 
            # this tqdm doesn't show except on an exception, which might be useful
            for i in tqdm(range(self.n_steps),
                          desc="steps"):
                input, states, key3 = self._body_func(i, (input, states, key3))
                
            return states
                 
        _, states, _ = lax.fori_loop(
            0, 
            self.n_steps, 
            self._body_func,
            (input, states, key3),
        )
        
        return states
    
    def init(self, input, state, args, key):
        # 1. generate empty trajectories of states 
        outputs = eqx.filter_eval_shape(self.step, input, state, args, key)
        # `eqx.is_array_like` is False for jax.ShapeDtypeSty
        scalars, array_structs = eqx.partition(outputs, eqx.is_array_like)
        asarrays = eqx.combine(jax.tree_map(jnp.asarray, scalars), array_structs)
        states = jax.tree_map(
            lambda x: jnp.zeros((self.n_steps, *x.shape), dtype=x.dtype),
            asarrays,
        )
        # 2. initialize the first state
        states = tree_set_idx(states, state, 0)
        return states


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
    # target state + feedback: angular pos & vel of joints & cartesian EE 
    n_input = system.twolink.state_size * 2 + N_DIM * 2  
    cell = eqx.nn.GRUCell(n_input, n_hidden, key=key1)
    net = RNN(cell, system.control_size, out_nonlinearity=out_nonlinearity, key=key2)
    body = SimpleFeedback(
        net, 
        mechanics, 
        delay=feedback_delay,
        #perturbation=None,
    )

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
    
    batched_model = jax.vmap(model, in_axes=(0, 0, None)) 
    
    states = batched_model(target_state, init_state, key)
    
    # #! stuff after this point is largely model-specific
    # (system_states, _), ee_states, controls, activities = states
    # states = ee_states  # operational space loss
  
    # sum over xyz, apply temporal discount, sum over time
    position_loss = jnp.sum(discount * jnp.sum((states.ee[0] - target_state[0][:, None]) ** 2, axis=-1), axis=-1)
    
    loss_terms = dict(
        #final_position=jnp.sum((states[..., -1, :2] - target_state[..., :2]) ** 2, axis=-1).squeeze(),  # sum over xyz
        position=position_loss,  
        final_velocity=jnp.sum((states.ee[1][:, -1] - target_state[1]) ** 2, axis=-1).squeeze(),  # over xyz
        control=jnp.sum(states.control ** 2, axis=(-1, -2)),  # over control variables and time
        hidden=jnp.sum(states.hidden ** 2, axis=(-1, -2)),  # over network units and time
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
        # #! this is actually weight + bias decay. should be separate
        loss_terms['weight_decay'] = weight_decay * tree_sum_squares(diff_model)
        
    # sum over terms
    loss = jax.tree_util.tree_reduce(lambda x, y: x + y, loss_terms)
    
    return loss, loss_terms


# %%
def get_evaluate_func(
    model,
    workspace, 
    n_directions=8, 
    grid_points_per_dim=2,
    reach_length=0.05,
    discount=1.,
    term_weights=None,
):
    """Prepare the center-out task for evaluating the model.
    
    Returns a function that takes a model and returns losses and a figure.
    """
    centers = internal_grid_points(workspace, grid_points_per_dim)
    pos_endpoints = jnp.concatenate([
        centreout_endpoints(jnp.array(center), n_directions, 0, reach_length) 
        for center in centers
    ], axis=1)
    vel_endpoints = jnp.zeros_like(pos_endpoints)
    init_states, target_states = tuple(zip(pos_endpoints, vel_endpoints))
    
    @eqx.filter_jit
    def loss_fn(states):
        # TODO: implementing `Loss` as a class would stop this from repeating the other cost function?
        position_loss = jnp.sum(discount * jnp.sum((states.ee[0] - pos_endpoints[1][:, None]) ** 2, axis=-1), axis=-1)
    
        loss_terms = dict(
            #final_position=jnp.sum((states[..., -1, :2] - target_state[..., :2]) ** 2, axis=-1).squeeze(),  # sum over xyz
            position=position_loss,  
            final_velocity=jnp.sum((states.ee[1][:, -1] - vel_endpoints[1]) ** 2, axis=-1).squeeze(),  # over xyz
            control=jnp.sum(states.control ** 2, axis=(-1, -2)),  # over control variables and time
            hidden=jnp.sum(states.hidden ** 2, axis=(-1, -2)),  # over network units and time
        )
    
        # mean over batch
        loss_terms = jax.tree_map(lambda x: jnp.mean(x, axis=0), loss_terms)
        if term_weights is not None:
            # term scaling
            loss_terms = jax.tree_map(lambda term, weight: term * weight, loss_terms, term_weights)
        
        loss = jax.tree_util.tree_reduce(lambda x, y: x + y, loss_terms)
        
        return loss, loss_terms

    def evaluate_centerout(model, key):
        # TODO: arbitrary # of batch dims? i.e. to keep different center-out sets separate
        batched_model = jax.vmap(model, in_axes=(0, 0, None))
        with catchtime() as t:
            states = batched_model(target_states, init_states, key)
        
        loss, loss_terms = loss_fn(states)
        
        return loss, loss_terms, states, t.time

    def make_eval_fig(ee_states, controls, workspace):
        # #! maybe this shouldn't be in here, but I need to call it in multiple places
        fig, _ = plot_states_forces_2d(
            ee_states[0], ee_states[1], controls[:, 2:, -2:], pos_endpoints, 
            force_labels=('Biarticular controls', 'Flexor', 'Extensor'), 
            cmap='plasma', workspace=workspace
        )
        return fig

    return evaluate_centerout, make_eval_fig


# %%
def train(
    model=None,  # start from existing model
    n_steps=100,
    hidden_size=50,
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
        pos_endpoints = uniform_endpoints(key, batch_size, N_DIM, workspace)
        vel_endpoints = jnp.zeros_like(pos_endpoints)
        init_states, target_states = tuple(zip(pos_endpoints, vel_endpoints))
        return init_states, target_states
    
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
    evaluate, make_eval_fig = get_evaluate_func(
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
    writer = SummaryWriter(tb_logdir / f"{timestr}_{NB_PREFIX}")
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
    
    def get_last_checkpoint(model):
        try:
            with open(chkpt_dir / "last_batch.txt", 'r') as f:
                last_batch = int(f.read()) 
        except FileNotFoundError:
            return
            
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
                      desc='batch', initial=start_batch, total=n_batches, file=sys.stdout):
        key, key_train, key_eval = jrandom.split(key, 3)
        # TODO: I think `init_state` isn't a tuple here but the old concatenated version...
        init_state, target_state = get_batch(batch_size, key)
        
        loss, loss_terms, model, opt_state = train_step(
            model, init_state, target_state, opt_state, key_train
        )
        
        losses = losses.at[batch].set(loss)
        losses_terms = tree_set_idx(losses_terms, loss_terms, batch)
        
        # tensorboad
        writer.add_scalar('Loss/train', loss.item(), batch)
        for term, loss_term in loss_terms.items():
            writer.add_scalar(f'Loss/train/{term}', loss_term.item(), batch)
        
        if jnp.isnan(loss):
            print(f"\nNaN loss at batch {batch}!")
            if (checkpoint := get_last_checkpoint(model)) is not None:
                last_batch, model, losses, losses_terms = checkpoint
                print(f"Returning checkpoint from batch {last_batch}.")
            else:
                print("No checkpoint found, returning model from final iteration.")
            return model, losses, losses_terms
        
        if batch % log_step == 0:
            # model checkpoint
            eqx.tree_serialise_leaves(chkpt_dir / f'model{batch}.eqx', model)
            eqx.tree_serialise_leaves(chkpt_dir / f'losses{batch}.eqx', 
                                      (losses, losses_terms))
            with open(chkpt_dir / "last_batch.txt", 'w') as f:
                f.write(str(batch)) 
            
            # tensorboard
            loss_eval, loss_eval_terms, states, t = evaluate(model, key_eval)
            fig = make_eval_fig(states.ee, states.control, workspace)
            writer.add_figure('Eval/centerout', fig, batch)
            writer.add_scalar('Loss/eval', loss_eval.item(), batch)
            writer.add_scalar('Loss/eval/time', t, batch)
            for term, loss_term in loss_eval_terms.items():
                writer.add_scalar(f'Loss/eval/{term}', loss_term.item(), batch)
            
            # TODO: could format a single string
            tqdm.write(f"step: {batch}", file=sys.stdout)
            tqdm.write(f"\ttraining loss: {loss:.4f}", file=sys.stdout)
            tqdm.write(f"\tevaluation loss: {loss_eval:.4f}", file=sys.stdout)
            
    def save_model(
            model, 
            hyperparams: Optional[dict] = None, 
            fig_funcs: Optional[dict]=None
    ):
        
        timestr = datetime.today().strftime("%Y%m%d-%H%M%S") 
        name = f"model_{timestr}_{NB_PREFIX}"
        eqx.tree_serialise_leaves(model_dir / f'{name}.eqx', model)
        if hyperparams is not None:
            with open(model_dir / f'{name}.json', 'w') as f:
                hyperparams_str = json.dumps(hyperparams, indent=4)
                f.write(hyperparams_str)
        if fig_funcs is not None:
            for label, fig_func in fig_funcs.items():
                fig = fig_func()
                fig.savefig(model_dir / f'{name}_{label}.png')
                plt.close(fig)
    
    # save model along with hyperparameters and training/evaluation figures
    loss_eval, loss_eval_terms, states, _ = evaluate(model, key_eval)
    fig_funcs = dict(
        loss=lambda: plot_loglog_losses(losses, losses_terms)[0],
        eval=lambda: make_eval_fig(states.ee, states.control, workspace),
    )
    hyperparams = dict(
        workspace=workspace.tolist(),
        batch_size=batch_size,
        n_batches=n_batches,
        epochs=epochs,
        learning_rate=learning_rate,
        term_weights=term_weights,
        weight_decay=weight_decay,
        seed=seed,
        commit_hash=git_commit_id(),
    )
    save_model(model, hyperparams, fig_funcs)
    #writer.add_hparams(hyperparams, {"hparam/eval_loss": loss_eval.item()})
    
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
    dt=0.05, 
    feedback_delay_steps=0,
    n_batches=500, 
    n_steps=50, 
    hidden_size=50, 
    seed=5566,
    learning_rate=0.05,
    log_step=100,
    workspace=workspace,
    term_weights=term_weights,
    weight_decay=None,
    restore_checkpoint=False,
)

plot_loglog_losses(losses, losses_terms)
plt.show()

# %% [markdown]
# Optionally, load an existing model

# %%
model = get_model()
model = eqx.tree_deserialise_leaves("../models/model_20230926-093821_nb10.eqx", model)

# %% [markdown]
# Evaluate on a centre-out task

# %%
evaluate, make_eval_plot = get_evaluate_func(model, workspace, term_weights=term_weights, grid_points_per_dim=2)
loss, loss_terms, states, t = evaluate(model, key=jrandom.PRNGKey(0))

# %%
fig = make_eval_plot(states[1], states[2], workspace)

# %% [markdown]
# Plot entire arm trajectory for an example direction

# %%
idx = 0

# %%
# convert all joints to Cartesian since I only saved the EE state
xy_pos = eqx.filter_vmap(nlink_angular_to_cartesian)(
    model.step.mechanics.system.twolink, states[0][0].reshape(-1, 2), states[0][1].reshape(-1, 2)
)[0].reshape(states[0][0].shape[0], -1, 2, 2)

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
