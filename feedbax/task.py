"""Tasks on which models are trained and evaluated.

TODO:
- `RandomReaches` and `RandomReachesDelayed` are similar. 
   See what can be abstracted.
- Ditto `get_target_seq` and `get_scalar_epoch_seq`.
    - Also, the way `seq` and `seqs` are generated is similar to `states` in 
      `Iterator.init`...

:copyright: Copyright 2023 by MLL <mll@mll.bio>.
:license: Apache 2.0, see LICENSE for details.
"""

from abc import abstractmethod, abstractproperty
from functools import cached_property
import logging 
from typing import Callable, Dict, Optional, Tuple

import equinox as eqx
from equinox import AbstractVar, field
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, Int, PyTree, Shaped
import numpy as np

from feedbax.loss import AbstractLoss
from feedbax.types import CartesianState2D
from feedbax.utils import internal_grid_points


logger = logging.getLogger(__name__)


N_DIM = 2


class AbstractTaskInput(eqx.Module):
    intervenors: AbstractVar[Dict[str, jax.Array]]


class SimpleReachTaskInput:
    stim: Float[Array, "n_steps 1"]
    intervenors: Dict[str, jax.Array]
    

class DelayedReachTaskInput(AbstractTaskInput):
    stim: Float[Array, "n_steps 1"]
    hold: Int[Array, "n_steps 1"]
    stim_on: Int[Array, "n_steps 1"]
    intervenors: Dict[str, jax.Array]


class AbstractTaskTrialSpec(eqx.Module):
    init: PyTree
    input: AbstractTaskInput
    target: PyTree
    

# TODO: this same trial spec also applies to tracking and stabilization tasks...
#? does it apply to all kinds of movement tasks we're interested in?
# if so we can eliminate the `AbstractTaskTrialSpec`
class ReachTrialSpec(AbstractTaskTrialSpec):
    init: CartesianState2D 
    input: AbstractTaskInput
    target: CartesianState2D


class AbstractTask(eqx.Module):
    """Abstract base class for tasks.
    
    Associates a trial generator with a loss function and a set of . Also provides methods
    for evaluating a suitable model. 
    
    TODO: 
    - Could use `__call__` instead of `eval_trials`.
    - Should allow for both dynamically generating trials, and predefined 
      datasets of trials. Currently training is dynamic, and evaluation is 
      static.
    """
    loss_func: AbstractVar[AbstractLoss]
    
    @abstractmethod
    def get_train_trial(
        self, 
        key: jax.Array,
    ) -> Tuple[AbstractTaskTrialSpec, Optional[PyTree]]:
        """Return a single training trial for the task.
        
        May also return a pytree of auxiliary data, e.g. that would be 
        useful for plotting or analysis but not for model training or
        evaluation.
        """
        ...  
        
    @abstractproperty    
    def trials_validation(
        self
    ) -> Tuple[AbstractTaskTrialSpec, Optional[PyTree]]:
        """Return the batch of validation trials associated with the task.
        
        May also return a pytree of auxiliary data, e.g. that would be 
        useful for plotting or analysis but not for model evaluation.
        """
        ...

    def eval(
        self, 
        model: eqx.Module, 
        key: jax.Array,
    ) -> Tuple[Float[Array, ""], PyTree, PyTree]:
        """Evaluate a model on the task's validation set of trials."""
        
        return self.eval_trials(model, self.trials_validation[0], key)
    
    @eqx.filter_jit
    def eval_train_batch(
        self, 
        model: eqx.Module, 
        batch_size: int, 
        key: jax.Array,
    ) -> Tuple[Tuple[Float[Array, ""], PyTree, PyTree], 
               AbstractTaskTrialSpec, 
               PyTree]:
        """Evaluate a model on a single batch of training trials."""
        keys = jr.split(key, batch_size)
        trials, aux = jax.vmap(self.get_train_trial)(keys)
        
        return self.eval_trials(model, trials, key), trials, aux
    
    @eqx.filter_jit
    def eval_ensemble_train_batch(
        self,
        models,
        n_replicates,  # TODO: infer from `models`
        batch_size,
        key,
    ):
        """Evaluate an ensemble of models on training batches.
        
        TODO: 
        - Allow user to control whether they are evaluated on the same batch,
          or different ones (as is currently the case).
        - Similar functions for evaluating arbitrary trials, and validation set
        """
        models_arrays, models_other = eqx.partition(models, eqx.is_array)
        def evaluate_single(model_arrays, model_other, batch_size, key):
            model = eqx.combine(model_arrays, model_other)
            return self.eval_train_batch(model, batch_size, key)  
        keys_eval = jr.split(key, n_replicates)
        return eqx.filter_vmap(evaluate_single, in_axes=(0, None, None, 0))(
            models_arrays, models_other, batch_size, keys_eval
        )

    @eqx.filter_jit
    @jax.named_scope("fbx.AbstractTask.eval_trials")
    def eval_trials(
        self, 
        model: eqx.Module, 
        trial_specs: AbstractTaskTrialSpec, 
        key: jax.Array,
    ) -> Tuple[Float[Array, ""], PyTree, PyTree]:
        """Evaluate a model on a set of trials.
        """      
        
        states = jax.vmap(model, in_axes=(0, 0, None))(
            trial_specs.input, trial_specs.init, key
        ) 
        
        loss, loss_terms = self.loss_func(
            states, 
            trial_specs.target, 
            trial_specs.input
        )
        
        return loss, loss_terms, states


class RandomReaches(AbstractTask):
    """Train on random reach endpoints in rectangular workspace.
    
    Validation set is center-out reaches. 
    
    TODO:
    - Could assume a default loss function (e.g. `loss.simple_reach_loss`)
      and allow the user to pass just `loss_term_weights`.
    """
    loss_func: AbstractLoss
    workspace: Float[Array, "ndim 2"] = field(converter=jnp.asarray)
    n_steps: int
    eval_n_directions: int 
    eval_reach_length: float
    eval_grid_n: int  # e.g. 2 -> 2x2 grid of center-out reach sets
    
    N_DIM = 2
    
    @eqx.filter_jit
    @jax.named_scope("fbx.RandomReaches.get_train_trial")
    def get_train_trial(
        self, 
        key: jax.Array
    ) -> [ReachTrialSpec, None]:
        """Random reach endpoints in a 2D rectangular workspace.
        
        TODO:
        - Try JIT.
        """
        
        pos_endpoints = jr.uniform(
            key, 
            (2, self.N_DIM), 
            minval=self.workspace[:, 0], 
            maxval=self.workspace[:, 1]
        )
        vel_endpoints = jnp.zeros_like(pos_endpoints)
        forces = jnp.zeros_like(pos_endpoints)
        
        init_state, target_state = jax.tree_map(
            lambda x: CartesianState2D(*x),
            list(zip(pos_endpoints, vel_endpoints, forces)),
            is_leaf=lambda x: isinstance(x, tuple)
        )
        # make targets as sequences, because `Iterator` and `Loss` want that
        target_state = jax.tree_map(
            lambda x: jnp.broadcast_to(x, (self.n_steps, *x.shape)),
            target_state,
        )
        task_input = target_state
        
        return ReachTrialSpec(init_state, task_input, target_state), None
        
    @cached_property
    def trials_validation(self) -> [ReachTrialSpec, None]:
        """Center-out reaches across a regular workspace grid."""
        
        centers = internal_grid_points(self.workspace, self.eval_grid_n)
        pos_endpoints = jax.vmap(
            centreout_endpoints, 
            in_axes=(0, None, None),
            out_axes=1,
        )(
            centers, self.eval_n_directions, self.eval_reach_length
        ).reshape((2, -1, self.N_DIM))
        vel_endpoints = jnp.zeros_like(pos_endpoints)
        forces = jnp.zeros_like(pos_endpoints)
        
        init_states, target_states = jax.tree_map(
            lambda x: CartesianState2D(*x),
            list(zip(pos_endpoints, vel_endpoints, forces)),
            is_leaf=lambda x: isinstance(x, tuple)
        )
        # this is kind of annoying, but because the batch is explicit here, we
        # need to swap axes as well as broadcast
        target_states = jax.tree_map(
            lambda x: jnp.swapaxes(
                jnp.broadcast_to(x, (self.n_steps, *x.shape)),
                0, 1
            ),
            target_states,
        )
        task_inputs = target_states 
        
        return ReachTrialSpec(init_states, task_inputs, target_states), None


class DelayTaskInput(eqx.Module):
    stim: PyTree[Float[Array, "n_steps ..."]]
    hold: Int[Array, "n_steps 1"]
    stim_on: Int[Array, "n_steps 1"]


class RandomReachesDelayed(AbstractTask):
    """Uniform random endpoints in a rectangular workspace.
    
    e.g. allows for a stimulus epoch, followed by a delay period, then movement.
    
    TODO: 
    """
    loss_func: AbstractLoss 
    workspace: Float[Array, "ndim 2"] = field(converter=jnp.asarray)
    n_steps: int 
    epoch_len_ranges: Tuple[Tuple[int, int], ...]
    eval_n_directions: int
    eval_reach_length: float
    eval_grid_n: int  
    stim_epochs: Tuple[int, ...] = field(default=(1,), converter=jnp.asarray)
    hold_epochs: Tuple[int, ...] = field(default=(0, 1, 2), converter=jnp.asarray)
    key_eval: jax.Array = field(default_factory=lambda: jr.PRNGKey(0))

    @eqx.filter_jit
    @jax.named_scope("fbx.RandomReachesDelayed.get_train_trial")
    def get_train_trial(
        self, 
        key: jax.Array
    ) -> [ReachTrialSpec, Int[Array, "n_epochs"]]:
        """Random reach endpoints in a 2D rectangular workspace."""
        
        key1, key2 = jr.split(key)
        pos_endpoints = jr.uniform(
            key1, 
            (2, N_DIM), 
            minval=self.workspace[:, 0], 
            maxval=self.workspace[:, 1]
        )
        vel_endpoints = jnp.zeros_like(pos_endpoints)
        
        init_state, target_state = jax.tree_map(
            lambda x: CartesianState2D(*x),
            list(zip(pos_endpoints, vel_endpoints)),
            is_leaf=lambda x: isinstance(x, tuple)
        )
        task_inputs, target_states, epoch_start_idxs = self.get_sequences(
            init_state, target_state, key2
        )      
        
        return (
            ReachTrialSpec(init_state, task_inputs, target_states), 
            epoch_start_idxs,
        )
        
    @cached_property
    def trials_validation(self):
        """Center-out reaches across a regular workspace grid."""
        
        centers = internal_grid_points(self.workspace, self.eval_grid_n)
        pos_endpoints = jax.vmap(
            centreout_endpoints, 
            in_axes=(0, None, None),
            out_axes=1,
        )(
            centers, self.eval_n_directions, self.eval_reach_length
        ).reshape((2, -1, self.N_DIM))
        vel_endpoints = jnp.zeros_like(pos_endpoints)
        
        init_states, target_states = jax.tree_map(
            lambda x: CartesianState2D(*x),
            list(zip(pos_endpoints, vel_endpoints)),
            is_leaf=lambda x: isinstance(x, tuple)
        )        
        epochs_keys = jr.split(self.key_eval, init_states.pos.shape[0])
        task_inputs, target_states, epoch_start_idxs = jax.vmap(self.get_sequences)(
            init_states, target_states, epochs_keys
        )    
           
        return (
            ReachTrialSpec(init_states, task_inputs, target_states), 
            epoch_start_idxs,
        )
    
    def get_sequences(
        self,  
        init_states: CartesianState2D, 
        target_states: CartesianState2D, 
        key: jax.Array,
    ) -> Tuple[DelayTaskInput, CartesianState2D, Int[Array, "n_epochs"]]:
        """Convert static task inputs to sequences, and make hold signal.
        """        
        epoch_lengths = gen_epoch_lengths(key, self.epoch_len_ranges)
        epoch_start_idxs = jnp.pad(
            jnp.cumsum(epoch_lengths), 
            (1, 0), 
            constant_values=(0, -1)
        )
        epoch_masks = get_masks(self.n_steps, epoch_start_idxs)
        move_epoch_mask = jnp.logical_not(
            jnp.prod(epoch_masks, axis=0)
        )[None, :]
        
        stim_seqs = get_masked_seqs(
            target_states, 
            epoch_masks[self.stim_epochs]
        )
        target_seqs = jax.tree_map(
            lambda x, y: x + y, 
            get_masked_seqs(target_states, move_epoch_mask),
            get_masked_seqs(init_states, epoch_masks[self.hold_epochs]),
        )
        stim_on_seq = get_scalar_epoch_seq(
            epoch_start_idxs, self.n_steps, 1., self.stim_epochs
        )
        hold_seq = get_scalar_epoch_seq(
            epoch_start_idxs, self.n_steps, 1., self.hold_epochs
        )
        
        task_input = DelayTaskInput(stim_seqs, hold_seq, stim_on_seq)
        target_states = target_seqs
        
        return task_input, target_states, epoch_start_idxs  


def uniform_endpoints(
    key: jr.PRNGKey,
    ndim: int = 2, 
    workspace: Float[Array, "ndim 2"] = jnp.array([[-1., 1.], 
                                                   [-1., 1.]]),
) -> Float[Array, "2 ndim"]:
    """Segment endpoints uniformly distributed in a rectangular workspace."""
    return jr.uniform(
        key, 
        (2, ndim),   # (start/end, ...)
        minval=workspace[:, 0], 
        maxval=workspace[:, 1]
    )


def centreout_endpoints(
    center: Float[Array, "2"], 
    n_directions: int, 
    length: float,
    angle_offset: float = 0, 
) -> Float[Array, "2 n_directions 2"]: 
    """Segment endpoints starting in the centre and ending equally spaced on a circle."""
    angles = jnp.linspace(0, 2 * np.pi, n_directions + 1)[:-1]
    angles = angles + angle_offset

    starts = jnp.tile(center, (n_directions, 1))
    ends = center + length * jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)

    return jnp.stack([starts, ends], axis=0)  


def gen_epoch_lengths(
    key: jr.PRNGKey,
    ranges: Tuple[Tuple[int, int], ...]=((1, 3),  # (min, max) for first epoch
                                         (2, 5),  # second epoch
                                         (1, 3)),  
) -> Int[Array, "n_epochs"]:
    """Generate a random integer in each given ranges."""
    ranges = jnp.array(ranges, dtype=int)
    return jr.randint(key, (ranges.shape[0],), *ranges.T)


def get_masks(
    length: int, 
    idx_bounds: Int[Array, "_"],
):
    """Get a 1D mask of length `length` with `False` values at `idxs`."""
    idxs = jnp.arange(length)
    #? could also use `arange` to get ranges of idxs
    mask_fn = lambda e: (idxs < idx_bounds[e]) + (idxs > idx_bounds[e + 1] - 1)
    return jnp.stack([mask_fn(e) for e in range(len(idx_bounds) - 1)])


def get_masked_seqs(
    arrays: PyTree, 
    masks: Tuple[Int[Array, "n"], ...],
    init_fn: Callable[[Tuple[int, ...]], Shaped[Array, "..."]] = jnp.zeros,
) -> PyTree:
    """Expand arrays with an initial axis of length `n`, and fill with 
    original array values where the intersection of `masks` is `False`.
    
    That is, each expanded array will be filled with the values from `array`
    for all indices where *any* of the masks is `False`.
    
    Returns a PyTree with the same structure as `target`, but where each
    array has an additional sequence dimension, and the original `target` 
    values are assigned only during the target epoch, as bounded by
    `target_idxs`.
    
    TODO:
    - Find a better name.
    """
    
    seqs = jax.tree_map(
        lambda x: init_fn((masks.shape[1], *x.shape)),
        arrays
    )
    # seqs = tree_set_idx(seqs, targets, slice(*epoch_idxs[target_epoch:target_epoch + 2]))
    mask = jnp.prod(masks, axis=0)
    seqs = jax.tree_map(
        lambda x, y: jnp.where(
            jnp.expand_dims(mask, np.arange(y.ndim) + 1), 
            x, 
            y[None,:]
        ), 
        seqs, 
        arrays
    )
    return seqs


def get_scalar_epoch_seq(
    epoch_idxs: Int[Array, "n_epochs-1"], 
    n_steps: int, 
    hold_value: float, 
    hold_epochs: Tuple[int, ...],
):
    """A scalar sequence with a non-zero value held during `hold_epochs`.
    
    Similar to `get_target_steps`, but not for a PyTree.
    """
    seq = jnp.zeros((n_steps,))
    idxs = jnp.arange(n_steps)
    #fill_idxs = jnp.arange(epoch_idxs[hold_epoch], epoch_idxs[hold_epoch + 1])
    mask_fn = lambda e: (idxs < epoch_idxs[e]) + (idxs > epoch_idxs[e + 1] - 1)
    mask = jnp.prod(jnp.stack([mask_fn(e) for e in hold_epochs]), axis=0)
    seq = jnp.where(mask, seq, jnp.array(hold_value))
    return jnp.expand_dims(seq, -1)