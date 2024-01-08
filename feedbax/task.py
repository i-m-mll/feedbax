"""Tasks on which models are trained and evaluated.

TODO:
- `ReachTrialSpec.init_spec` values could be lambdas, to allow for 
  state-dependent state initialization. Though I'm not sure of the 
  use case for that, if any, since the current principle is to 
  use a standard/constant initial state except where explicitly
  overridden by these methods. So there would be no input variation.
- Maybe allow initial mods to model parameters, in addition to substates.
- Should `AbstractTask` be a generic of `StateT`?
- Some of the private functions could be public.
- Refactor `get_target_seq` and `get_scalar_epoch_seq` redundancy.
    - Also, the way `seq` and `seqs` are generated is similar to `states` in 
      `Iterator.init`...

:copyright: Copyright 2023 by MLL <mll@mll.bio>.
:license: Apache 2.0, see LICENSE for details.
"""

#! Can't do this because `AbstractVar` annotations can't be stringified.
# from __future__ import annotations


from abc import abstractmethod, abstractproperty
from collections import OrderedDict
from functools import cached_property
import logging 
from typing import (
    TYPE_CHECKING,
    Callable, 
    Dict, 
    Optional, 
    Tuple, 
    TypeVar,
)

import equinox as eqx
from equinox import AbstractVar, field
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jaxtyping import Array, Float, Int, PyTree, Shaped
import numpy as np

from feedbax.loss import AbstractLoss, LossDict
if TYPE_CHECKING:
    from feedbax.model import AbstractModel, AbstractModelState
from feedbax.state import AbstractState, CartesianState2D
from feedbax.utils import AbstractTransformedOrderedDict, internal_grid_points, get_where_str


logger = logging.getLogger(__name__)


N_DIM = 2


StateT = TypeVar("StateT", bound=AbstractState)

@jtu.register_pytree_node_class
class InitSpecDict(AbstractTransformedOrderedDict[
    Callable[["AbstractModelState"], PyTree[Array]],
    PyTree[Array]
]):
    """An `OrderedDict` that allows attribute-accessing lambdas as keys.
    
    The user can also access by the string keys. These are equivalent:
    
        `init_spec[lambda state: state.mechanics.effector]`
        
        `init_spec['mechanics.effector']`
        
    This assumes that keys will be limited to `where` lambdas: in which only 
    the first argument is used, and only used to access its attributes (i.e.
    to indicate a node/leaf of a PyTree argument).
    
    Tests of performance:
    - Construction is about 100x slower than `OrderedDict`. For dicts of 
      size relevant to our use case, this means ~100 us instead of ~1 us.
      A modest increase in dict size only changes this modestly.
    - Access is about 800x slower, and as expected this doesn't
      change with dict size because there's just a constant overhead 
      for doing the key transformation on each access. 
        - Each access is about 26 us, and this is also the duration of 
          a call to `get_where_str`.
    - `list(init_spec.items())` is about 100x slower (24 us versus 234 ns)
      for single entry, and about 260x slower (149 us versus 571 ns) for
      6 entries, which is about as many as I expect anyone to use in the 
      near future, when constructing a task.      

    In general this is pretty slow, but since we only need to do a single
    construction and a single access of `init_spec` per batch/evaluation, 
    I doubt this is an issue. 
    
    Optimizations should focus on `get_where_str`.
    """
    
    def _key_transform(self, key):
        if isinstance(key, Callable):
            return get_where_str(key)
        return key


class AbstractTaskInput(eqx.Module):
    intervenors: AbstractVar[Dict[str, jax.Array]]


class AbstractTaskTrialSpec(eqx.Module):
    init: InitSpecDict
    # init: OrderedDict[Callable[["AbstractModelState"], PyTree[Array]], 
    #                        PyTree[Array]]
    input: AbstractTaskInput
    target: PyTree


class SimpleReachTaskInput:
    stim: Float[Array, "time 1"]
    intervenors: Dict[str, jax.Array]
    

class DelayedReachTaskInput(AbstractTaskInput):
    stim: Float[Array, "time 1"]
    hold: Int[Array, "time 1"]
    stim_on: Int[Array, "time 1"]
    intervenors: Dict[str, jax.Array]


# TODO: this same trial spec also applies to tracking and stabilization tasks...
#? does it apply to all kinds of movement tasks we're interested in?
# if so we can eliminate the `AbstractTaskTrialSpec`
class ReachTrialSpec(AbstractTaskTrialSpec):
    init: InitSpecDict
    # init: OrderedDict[Callable[["AbstractModelState"], CartesianState2D], 
    #                        CartesianState2D] 
    input: AbstractTaskInput
    target: CartesianState2D
    
    @cached_property
    def goal(self):
        return jax.tree_map(lambda x: x[:, -1], self.target)


class AbstractTask(eqx.Module):
    """Abstract base class for tasks.
    
    Associates a training trial generator with a loss function and a set of 
    validation trials. Also provides methods for evaluating suitable models
    in trials. 
    
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
    
    @abstractproperty 
    def n_trials_validation(self) -> int:
        """Size of the validation set."""
        ...

    def eval(
        self, 
        model: "AbstractModel[StateT]", 
        key: jax.Array,
    ) -> Tuple[LossDict, StateT]:
        """Evaluate a model on the task's validation set of trials."""
        
        keys = jr.split(key, self.n_trials_validation)
        
        return self.eval_trials(model, self.trials_validation[0], keys)

    @eqx.filter_jit
    def eval_ensemble(
        self,
        models,
        n_replicates,
        key,
    ):
        """Evaluate an ensemble of models on the task's validation set."""
        models_arrays, models_other = eqx.partition(models, eqx.is_array)
        def evaluate_single(model_arrays, model_other, key):
            model = eqx.combine(model_arrays, model_other)
            return self.eval(model, key)
        keys_eval = jr.split(key, n_replicates)
        return eqx.filter_vmap(evaluate_single, in_axes=(0, None, 0))(
            models_arrays, models_other, keys_eval
        )
        
    @eqx.filter_jit
    def eval_train_batch(
        self, 
        model: "AbstractModel[StateT]", 
        batch_size: int, 
        key: jax.Array,
    ) -> Tuple[Tuple[LossDict, StateT], 
               AbstractTaskTrialSpec, 
               PyTree]:
        """Evaluate a model on a single batch of training trials."""
        key_batch, key_eval = jr.split(key)
        keys_batch = jr.split(key_batch, batch_size)
        keys_eval = jr.split(key_eval, batch_size)
        
        trials, aux = jax.vmap(self.get_train_trial)(keys_batch)
        
        return self.eval_trials(model, trials, keys_eval), trials, aux
    
    @eqx.filter_jit
    def eval_ensemble_train_batch(
        self,
        models: "AbstractModel[StateT]",
        n_replicates: int,  
        batch_size: int,
        key: jax.Array,
    ):
        """Evaluate an ensemble of models on training batches.
        
        TODO: 
        - Infer `n_replicates` from `models_arrays`
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
        model: "AbstractModel[StateT]", 
        trial_specs: AbstractTaskTrialSpec, 
        keys: jax.Array,
    ) -> Tuple[LossDict, StateT]:
        """Evaluate a model on a set of trials.
        """      
        init_states = jax.vmap(model.step.init)(key=keys)
        
        for where_substate, init_substates in trial_specs.init.items():
            init_states = eqx.tree_at(
                where_substate, 
                init_states,
                init_substates, 
            )
        
        init_states = jax.vmap(model.step.state_consistency_update)(
            init_states
        )
        
        states = jax.vmap(model, in_axes=(0, 0, 0))(
            trial_specs.input, init_states, keys
        ) 
        
        losses = self.loss_func(
            states, 
            trial_specs.target, 
            trial_specs.input
        )
        
        return losses, states


def _pos_only_states(
    pos_endpoints: Float[Array, "... ndim=2"]
):
    """Construct Cartesian init and target states with zero force and velocity.
    """
    vel_endpoints = jnp.zeros_like(pos_endpoints)
    forces = jnp.zeros_like(pos_endpoints)
    
    states = jax.tree_map(
        lambda x: CartesianState2D(*x),
        list(zip(pos_endpoints, vel_endpoints, forces)),
        is_leaf=lambda x: isinstance(x, tuple)
    )
    
    return states


def _centerout_endpoints_grid(
    workspace: Float[Array, "bounds=2 ndim=2"],
    eval_grid_n: int,
    eval_n_directions: int,
    eval_reach_length: float,
):
    """Sets of center-out reaches, their centers in a grid across a workspace.
    """
    centers = internal_grid_points(workspace, eval_grid_n)
    pos_endpoints = jax.vmap(
        centreout_endpoints, 
        in_axes=(0, None, None),
        out_axes=1,
    )(
        centers, eval_n_directions, eval_reach_length
    ).reshape((2, -1, N_DIM))
    return pos_endpoints


def _forceless_task_inputs(
    target_states: CartesianState2D,
) -> CartesianState2D:
    """Only position and velocity of targets are supplied to the model.
    """
    return CartesianState2D(
        pos=target_states.pos, 
        vel=target_states.vel,
        force=None,
    )


class RandomReaches(AbstractTask):
    """Reaches between random endpoints in a rectangular workspace.
    
    Validation set is center-out reaches. 
    
    NOTE:
    - This passes a sequence of target velocities equal to zero, assuming that 
      the user will only associate a cost function that penalizes the initial
      or final velocities. If the intervening velocities are penalized, this 
      no longer makes sense as a reaching task.
    
    TODO:
    - Could assume a default loss function (e.g. `loss.simple_reach_loss`)
      and allow the user to pass just `loss_term_weights`.
    """
    loss_func: AbstractLoss
    workspace: Float[Array, "bounds=2 ndim=2"] = field(converter=jnp.asarray)
    n_steps: int
    eval_n_directions: int = 7
    eval_reach_length: float = 0.5
    eval_grid_n: int = 1  # e.g. 2 -> 2x2 grid of center-out reach sets
    
    N_DIM = 2
    
    @eqx.filter_jit
    @jax.named_scope("fbx.RandomReaches.get_train_trial")
    def get_train_trial(
        self, 
        key: jax.Array
    ) -> [ReachTrialSpec, None]:
        """Random reach endpoints in a 2D rectangular workspace.
        """
        
        pos_endpoints = uniform_tuples(key, n=2, bounds=self.workspace)
                
        init_state, target_state = \
            _pos_only_states(pos_endpoints)
        
        # Broadcast the fixed targets to a sequence with the desired number of 
        # time steps, since that's what `Iterator` and `Loss` will expect.
        # Hopefully this should not use up any extra memory. 
        target_state = jax.tree_map(
            lambda x: jnp.broadcast_to(x, (self.n_steps, *x.shape)),
            target_state,
        )
        task_input = _forceless_task_inputs(target_state)

        
        return ReachTrialSpec(
            init=InitSpecDict({
               lambda state: state.mechanics.effector: init_state 
            }),
            input=task_input, 
            target=target_state,
        ), None
        
    @cached_property
    def trials_validation(self) -> [ReachTrialSpec, None]:
        """Center-out reaches across a regular workspace grid."""
        
        pos_endpoints = _centerout_endpoints_grid(
            self.workspace,
            self.eval_grid_n,
            self.eval_n_directions,
            self.eval_reach_length,
        )
        
        init_states, target_states = \
            _pos_only_states(pos_endpoints)
        
        # Broadcast to the desired number of time steps. Awkwardly, we also 
        # need to use `swapaxes` because the batch dimension is explicit, here.
        target_states = jax.tree_map(
            lambda x: jnp.swapaxes(
                jnp.broadcast_to(x, (self.n_steps, *x.shape)),
                0, 1
            ),
            target_states,
        )
        
        task_inputs = _forceless_task_inputs(target_states)
        
        return ReachTrialSpec(
            init=InitSpecDict({
               lambda state: state.mechanics.effector: init_states 
            }),
            input=task_inputs, 
            target=target_states,
        ), None
        
    @property
    def n_trials_validation(self) -> int:
        """Size of the validation set."""
        return self.eval_grid_n ** 2 * self.eval_n_directions


class DelayTaskInput(eqx.Module):
    stim: PyTree[Float[Array, "time ..."]]
    hold: Int[Array, "time 1"]
    stim_on: Int[Array, "time 1"]


class RandomReachesDelayed(AbstractTask):
    """Uniform random endpoints in a rectangular workspace.
    
    e.g. allows for a stimulus epoch, followed by a delay period, then movement.
    
    TODO: 
    - Add a default loss function.
    - Also allow epoch lengths to be specified in terms of fraction of total
    """
    loss_func: AbstractLoss 
    workspace: Float[Array, "bounds=2 ndim=2"] = field(converter=jnp.asarray)
    n_steps: int 
    epoch_len_ranges: Tuple[Tuple[int, int], ...] = field(
        default=(
            (5, 15),  # start
            (10, 20),  # stim
            (10, 25),  # delay
        )
    )
    eval_n_directions: int = 7
    eval_reach_length: float = 0.5
    eval_grid_n: int = 1
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
        pos_endpoints = uniform_tuples(key1, n=2, bounds=self.workspace)
                
        init_state, target_state = \
            _pos_only_states(pos_endpoints)
        
        task_inputs, target_states, epoch_start_idxs = self.get_sequences(
            init_state, target_state, key2
        )      
        
        return (
            ReachTrialSpec(
                init=InitSpecDict({
                    lambda state: state.mechanics.effector: init_state 
                }),
                input=task_inputs, 
                target=target_states,
            ), 
            epoch_start_idxs,
        )
        
    @cached_property
    def trials_validation(self):
        """Center-out reaches across a regular workspace grid."""
        
        pos_endpoints = _centerout_endpoints_grid(
            self.workspace,
            self.eval_grid_n,
            self.eval_n_directions,
            self.eval_reach_length,
        )
        
        init_states, target_states = \
            _pos_only_states(pos_endpoints)
        
        epochs_keys = jr.split(self.key_eval, init_states.pos.shape[0])
        task_inputs, target_states, epoch_start_idxs = jax.vmap(self.get_sequences)(
            init_states, target_states, epochs_keys
        )    
           
        return (
            ReachTrialSpec(
                init=InitSpecDict({
                    lambda state: state.mechanics.effector: init_states 
                }),
                input=task_inputs, 
                target=target_states,
            ), 
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
            _forceless_task_inputs(target_states), 
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
    
    def n_trials_validation(self) -> int:
        """Size of the validation set."""
        return self.eval_grid_n ** 2 * self.eval_n_directions

class Stabilization(AbstractTask):
    """Postural stabilization task at random points in workspace.
    
    Validation set is center-out reaches. 
    
    TODO:
    - This should involve some kind of perturbations, which might be different
      for the validation set.
    """
    loss_func: AbstractLoss
    workspace: Float[Array, "bounds=2 ndim=2"] = field(converter=jnp.asarray)
    n_steps: int
    eval_grid_n: int  # e.g. 2 -> 2x2 grid 
    eval_workspace: Optional[Float[Array, "bounds=2 ndim=2"]] = field(
        converter=jnp.asarray, default=None)
    
    N_DIM = 2
    
    @eqx.filter_jit
    @jax.named_scope("fbx.RandomReaches.get_train_trial")
    def get_train_trial(
        self, 
        key: jax.Array
    ) -> [ReachTrialSpec, None]:
        """Random reach endpoints in a 2D rectangular workspace.
        """
        
        points = uniform_tuples(key, n=1, bounds=self.workspace)
                
        target_state = \
            _pos_only_states(points)
        
        init_state = target_state
        
        # Broadcast the fixed targets to a sequence with the desired number of 
        # time steps, since that's what `Iterator` and `Loss` will expect.
        # Hopefully this should not use up any extra memory. 
        target_state = jax.tree_map(
            lambda x: jnp.broadcast_to(x, (self.n_steps, *x.shape)),
            target_state,
        )
        task_input = _forceless_task_inputs(target_state)
        
        return ReachTrialSpec(
            init=InitSpecDict({
                lambda state: state.mechanics.effector: init_state
            }),
            input=task_input, 
            target=target_state
        ), None
        
    @cached_property
    def trials_validation(self) -> [ReachTrialSpec, None]:
        """Center-out reaches across a regular workspace grid."""
        
        if self.eval_workspace is None:
            workspace = self.workspace
        else:
            workspace = self.eval_workspace
        
        pos_endpoints = points_grid(
            workspace,
            self.eval_grid_n,
        )
        
        target_states = \
            _pos_only_states(pos_endpoints)
        
        init_states = target_states
        
        # Broadcast to the desired number of time steps. Awkwardly, we also 
        # need to use `swapaxes` because the batch dimension is explicit, here.
        target_states = jax.tree_map(
            lambda x: jnp.swapaxes(
                jnp.broadcast_to(x, (self.n_steps, *x.shape)),
                0, 1
            ),
            target_states,
        )
        
        task_inputs = _forceless_task_inputs(target_states)
        
        return ReachTrialSpec(
            init=InitSpecDict({
                lambda state: state.mechanics.effector: init_states 
            }),
            input=task_inputs,
            target=target_states
        ), None
    
    def n_trials_validation(self) -> int:
        """Size of the validation set."""
        return self.eval_grid_n ** 2 


def points_grid(
    workspace: Float[Array, "bounds=2 ndim=2"],
    grid_n: int | Tuple[int, int],
):
    """A regular grid of points over a rectangular workspace.
    
    Args:
    
        grid_n: Number of grid points in each dimension.
    """
    if isinstance(grid_n, int):
        grid_n = (grid_n, grid_n)
    
    xy_1d = map(lambda x: jnp.linspace(*x[0], x[1]), 
                zip(workspace.T, grid_n))
    grid = jnp.stack(jnp.meshgrid(*xy_1d))
    grid_points = grid.reshape(2, -1).T
    return grid_points


def uniform_tuples(
    key: jax.Array,
    n: int,
    bounds: Float[Array, "bounds=2 ndim=2"],
):
    """Tuples of points uniformly distributed in some (2D) bounds.
    """
    return jr.uniform(
        key, 
        (n, N_DIM), 
        minval=bounds[0], 
        maxval=bounds[1]
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