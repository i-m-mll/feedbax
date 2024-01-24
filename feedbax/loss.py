"""Composable loss function modules operating on state PyTrees.

TODO:

- `LossDict` only computes the total loss once, but when we append a `LossDict`
  for a single timestep to `losses_history` in `TaskTrainer`, we lose the loss
  total for that time step. When it is needed later (e.g. on plotting the loss)
  it will be recomputed, once. It is also not serialized along with the 
  `losses_history`. I doubt this is a significant computational loss 
  (how many loss terms * training iterations could be involved? 1e6?)
  to have to compute from time to time, but perhaps it would be nice to 
  include the total as part of flatten/unflatten. It'd probably just require
  that we allow passing the total on instantiation, however that would be kind
  of weird.
    - Even if we have 6 loss terms with 1e6 iterations, it only takes ~130 ms 
    to compute `losses.total`. Given that we only need to compute this once
    per session or so, it shouldn't be a problem.
- The time aggregation could be done in `CompositeLoss`, if we unsqueeze
  terms that don't have a time dimension. This would allow time aggregation
  to be controlled in one place, if for some reason it makes sense to change 
  how this aggregation occurs across all loss terms.
- Protocols for all the different `state` types/fields?
    - Alternatively we could make `AbstractLoss` generic over an 
      `AbstractState` typevar, however that might not make sense for typing
      the compositions (e.g. `__sum__`) since the composite can support any
      state pytrees that have the right combination of fields, not just pytrees
      that have an identical structure.
- L2 by default, but should allow for other norms

:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

#! Can't do this because `AbstractVar` annotations can't be stringified.
# from __future__ import annotations

from abc import abstractmethod
from functools import cache, cached_property
import logging
from typing import (
    Callable,
    ClassVar, 
    Dict,
    List, 
    Optional, 
    Sequence,
    Tuple,
)

import equinox as eqx
from equinox import AbstractVar
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, Float, PyTree

from feedbax.state import CartesianState2D, HasEffectorState
from feedbax.utils import get_unique_label, unzip2


logger = logging.getLogger(__name__)

@jtu.register_pytree_node_class
class LossDict(dict[str, Array]):
    """Stores loss terms and automatically calculates the sum."""
    
    @cached_property
    def total(self):
        loss_term_values = list(self.values())
        return jax.tree_util.tree_reduce(lambda x, y: x + y, loss_term_values)
        # return jnp.sum(jtu.tree_map(
        #     lambda *args: sum(args), 
        #     *loss_term_values
        # ))
    
    def __setitem__(self, key, value):
        raise TypeError("LossDict does not support item assignment")
        
    def update(self, dict_):
        raise TypeError("LossDict does not support update")
    
    def __or__(self, other):
        return LossDict({**self, **other})
    
    def tree_flatten(self):
        """The same flatten function used by JAX for `dict`"""
        return unzip2(sorted(self.items()))[::-1]

    @classmethod
    def tree_unflatten(cls, keys, values):
        return LossDict(zip(keys, values))


class AbstractLoss(eqx.Module):
    """Abstract base class for loss functions.
    
    Instances can be composed by addition and scalar multiplication.
    """
    label: AbstractVar[str]
    
    @abstractmethod
    def __call__(
        self,
        states: PyTree, 
        targets: PyTree, 
        task_inputs: Optional[PyTree] = None,
    ) -> LossDict:
        ...        

    def __add__(self, other: "AbstractLoss") -> "CompositeLoss":
        return CompositeLoss(terms=(self, other), weights=(1., 1.))

    def __radd__(self, other: "AbstractLoss") -> "CompositeLoss":
        return self.__add__(other)
    
    def __sub__(self, other: "AbstractLoss") -> "CompositeLoss":
        #? I don't know if this even makes sense but it's easy to implement.
        return CompositeLoss(terms=(self, other), weights=(1., -1.))
    
    def __rsub__(self, other: "AbstractLoss") -> "CompositeLoss":
        return CompositeLoss(terms=(self, other), weights=(-1., 1.))
    
    def __neg__(self) -> "CompositeLoss":
        return CompositeLoss(terms=(self,), weights=(-1.,))

    def __mul__(self, other) -> "CompositeLoss":
        """Assume scalar multiplication."""        
        if eqx.is_array_like(other):
            if eqx.is_array(other) and not other.shape == ():
                raise ValueError("Can't multiply loss term by non-scalar array")
            return CompositeLoss(terms=(self,), weights=(other,))
        else:
            raise ValueError("Can't multiply loss term by non-numeric type")
        
    def __rmul__(self, other):
        return self.__mul__(other)


class CompositeLoss(AbstractLoss):
    """Composite of simpler loss functions.
    
    During construction the user may pass dictionaries and/or sequences of 
    `AbstractLoss` instances ("terms") and weights. Any `CompositeLoss` terms 
    are flattened, incorporating their simple terms into the new composite
    loss, and multiplying through their component weights by the single weight
    passed as an argument for that composite term. If a composite term has a
    user-specified label, that label will be prepended to the labels of its 
    component terms, on flattening. If the flattened terms still do not have
    unique labels, they will be suffixed with the lowest integer that makes 
    them unique. 
    
    TODO:
    - Different aggregation schemes.
    """
    terms: Dict[str, AbstractLoss]
    weights: Dict[str, float]
    label: str 
    
    def __init__(
        self,
        terms: Dict[str, AbstractLoss] | Sequence[AbstractLoss],
        weights: Optional[Dict[str, float] | Sequence[float]] = None,
        label: str = "",
    ):
        self.label = label
        
        if isinstance(terms, dict):
            labels, terms = list(zip(*terms.items()))
        else:
            labels = [term.label for term in terms]
        
        if weights is None:
            weights = jax.tree_map(lambda _: 1., terms)
        elif not len(terms) == len(weights):
            raise ValueError("Mismatch between number of loss terms " 
                             + "and number of term weights")
        
        if isinstance(weights, dict):
            weights = list(weights.values())
            
        # Split into lists of data for simple and composite terms.
        term_tuples_split: Tuple[List[Tuple[str, AbstractLoss, float]],
                                 List[Tuple[str, AbstractLoss, float]]] 
        term_tuples_split = eqx.partition(
            list(zip(labels, terms, weights)),
            lambda x: not isinstance(x[1], CompositeLoss),
            is_leaf=lambda x: isinstance(x, tuple),
        )
        
        # Removes the `None` values from the lists.
        term_tuples_leaves = jax.tree_map(
            lambda x: jtu.tree_leaves(x, 
                                      is_leaf=lambda x: isinstance(x, tuple)),
            term_tuples_split,
            is_leaf=lambda x: isinstance(x, list),
        )

        # Start with the simple terms, if there are any. 
        if term_tuples_leaves[0] == []:
            all_labels, all_terms, all_weights = (), (), ()
        else:
            all_labels, all_terms, all_weights = zip(*term_tuples_leaves[0])
            
        # Make sure the simple term labels are unique.
        for i, label in enumerate(all_labels):
            label = get_unique_label(label, all_labels[:i])
            all_labels = all_labels[:i] + (label,) + all_labels[i+1:]
            
        # Flatten the composite terms, assuming they have the usual dict 
        # attributes. We only need to flatten one level, because this `__init__`
        # (and the immutability of `eqx.Module`) ensures no deeper nestings 
        # are ever constructed except through extreme hacks.
        for group_label, composite_term, group_weight in term_tuples_leaves[1]:
            labels = composite_term.terms.keys()
            
            # If a unique label for the composite term is available, use it to 
            # format the labels of the flattened terms.
            if group_label != "":
                labels = [f"{group_label}_{label}" for label in labels]
            elif composite_term.label != "":
                labels = [f"{composite_term.label}_{label}" 
                          for label in labels]
            
            # Make sure the labels are unique.
            for label in labels:
                label = get_unique_label(label, all_labels)
                all_labels += (label,)
            
            all_terms += tuple(composite_term.terms.values())
            all_weights += tuple(
                [group_weight * weight 
                 for weight in composite_term.weights.values()]
            )
        
        self.terms = dict(zip(all_labels, all_terms))
        self.weights = dict(zip(all_labels, all_weights))
    
    def __or__(self, other: "CompositeLoss") -> "CompositeLoss":
        """Merge two composite losses, overriding terms with the same label."""
        return CompositeLoss(
            terms=self.terms | other.terms, 
            weights=self.weights | other.weights,
            label=other.label,
        )
 
    @jax.named_scope("fbx.CompositeLoss")
    def __call__(
        self, 
        states: PyTree, 
        targets: PyTree,
        task_inputs: Optional[PyTree] = None,
    ) -> LossDict:
        
        # evaluate loss terms and merge resulting `LossDict`s
        losses = jtu.tree_reduce(
            lambda x, y: x | y,
            jax.tree_map(
                lambda term: term(states, targets, task_inputs), 
                self.terms,
                is_leaf=lambda x: isinstance(x, AbstractLoss),
            ),
            is_leaf=lambda x: isinstance(x, LossDict),
        )

        # aggregate over batch 
        losses = jax.tree_map(lambda x: jnp.mean(x, axis=0), losses)
        
        if self.weights is not None:
            # term scaling
            losses = jax.tree_map(
                lambda term, weight: term * weight, 
                dict(losses), 
                self.weights
            )
            
        return LossDict(losses)


class EffectorPositionLoss(AbstractLoss):
    """
    
    Note that if discount is shaped such that it gives non-zero weight to the
    position error during the fixation period of (say) a delayed reach task,
    then typically the target will be specified as the fixation point during
    that period, and `EffectorPositionLoss` will also act as a fixation loss.
    However, when we are using certain kinds of goal error discounting (e.g.
    exponential, favouring errors near the end of the trial) then the fixation
    loss may not be weighed into `EffectorPositionLoss`, and it may be
    appropriate to add `EffectorFixationLoss` to the composite loss. However,
    in that case the same result could still be achieved using a single
    instance of `EffectorPositionLoss`, by passing a `discount` that's the sum
    of the goal error discount (say, non-zero only near the end of the trial)
    and the hold signal (non-zero only during the fixation period) scaled by
    the relative weights of the goal and fixation error losses.
    
    TODO: do we handle the temporal discount here? or return the sequence of losses
    """
    label: str = "effector_position"
    discount_func: Optional[Callable[[int], Float[Array, "time"]]] = \
        lambda n_steps: power_discount(n_steps, discount_exp=6)[None, :]

    def __call__(
        self, 
        states: HasEffectorState, 
        targets: PyTree,
        task_inputs: Optional[PyTree] = None,
    ) -> LossDict:
        
        # sum over xyz
        loss = jnp.sum(
            (states.mechanics.effector.pos - targets.pos) ** 2, 
            axis=-1
        )
        
        # temporal discount
        if self.discount_func is not None:
            loss = loss * self.discount(loss.shape[-1])
        
        # sum over time
        loss = jnp.sum(loss, axis=-1)
        
        return LossDict({self.label: loss})
    
    #@cache
    def discount(self, n_steps):
        # Can't use a cache because of JIT. 
        # But we only need to run this once per training iteration...
        return self.discount_func(n_steps)


class EffectorStraightPathLoss(AbstractLoss):
    """Penalize non-straight paths between initial and final state.
    
    Calculates the length of the paths followed, and normalizes by the
    Euclidean (straight-line) distance between the initial and final state.
    
    The parameter `normalize_by` controls whether to normalize by the 
    Euclidean distance between the actual initial & final states, or the 
    actual initial state & the task-specified goal state.
    """
    label: str = "effector_straight_path"
    normalize_by: str = "actual"

    def __call__(
        self, 
        states: HasEffectorState, 
        targets: PyTree,
        task_inputs: Optional[PyTree] = None,
    ) -> LossDict:
        
        effector_pos = states.mechanics.effector.pos
        pos_diff = jnp.diff(effector_pos, axis=1)
        piecewise_lengths = jnp.linalg.norm(pos_diff, axis=-1)
        path_length = jnp.sum(piecewise_lengths, axis=1)
        if self.normalize_by == "actual":
            final_pos = effector_pos[:, -1]
        elif self.normalize_by == "goal":
            final_pos = targets.pos
        init_final_diff = final_pos - effector_pos[:, 0]
        straight_length = jnp.linalg.norm(init_final_diff, axis=-1)
        
        loss = path_length / straight_length
        
        return LossDict({self.label: loss})


class EffectorFixationLoss(AbstractLoss):
    """"""
    label: str = "effector_fixation"
    
    def __call__(
        self, 
        states: PyTree, 
        targets: PyTree,
        task_inputs: PyTree,
    ) -> LossDict:
        
        loss = jnp.sum(
            (states.mechanics.effector.pos - targets.pos) ** 2, 
            axis=-1
        )
        
        loss = loss * jnp.squeeze(task_inputs.hold)
        
        # sum over time
        loss = jnp.sum(loss, axis=-1)
        
        return LossDict({self.label: loss})


class EffectorFinalVelocityLoss(AbstractLoss):
    """
    
    TODO:
    - For tracking, an ongoing (not just final) velocity loss might make sense
    """
    label: str = "effector_final_velocity"

    def __call__(
        self, 
        states: PyTree, 
        targets: PyTree,
        task_inputs: Optional[PyTree] = None,
    ) -> LossDict:
        
        loss = jnp.sum(
            (states.mechanics.effector.vel[:, -1] - targets.vel[:, -1]) ** 2, 
            axis=-1
        )
        
        return LossDict({self.label: loss})


class NetworkOutputLoss(AbstractLoss):
    """"""
    label: str = "nn_output"

    def __call__(
        self, 
        states: PyTree, 
        #! **kwargs?
        targets: PyTree,
        task_inputs: Optional[PyTree] = None,
    ) -> LossDict:
        
        #loss = jnp.sum(states.network.output ** 2, axis=-1)
        loss = jnp.sum(states.network.output ** 2, axis=-1)
        
        # sum over time
        loss = jnp.sum(loss, axis=-1)
        
        return LossDict({self.label: loss})


class NetworkActivityLoss(AbstractLoss):
    """"""
    label: str = "nn_hidden"

    def __call__(
        self, 
        states: PyTree, 
        targets: PyTree,
        task_inputs: Optional[PyTree] = None,
    ) -> LossDict:
        
        loss = jnp.sum(states.network.hidden ** 2, axis=-1)
        
        # sum over time
        loss = jnp.sum(loss, axis=-1)
        
        return LossDict({self.label: loss})

        
def power_discount(n_steps, discount_exp=6):
    """Discounting vector, a power law curve from 0 to 1, start to end.
    """
    if discount_exp == 0:
        return 1.
    else:
        return jnp.linspace(1. / n_steps, 1., n_steps) ** discount_exp
    

def mse(x, y):
    """Mean squared error."""
    return jax.tree_map(
        lambda x, y: jnp.mean((x - y) ** 2),
        x, y,
    )