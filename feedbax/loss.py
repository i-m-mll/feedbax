"""

TODO:
- The time aggregation could be done in `CompositeLoss`, if we unsqueeze
  terms that don't have a time dimension. This would allow time aggregation
  to be controlled in one place, if for some reason it makes sense to change 
  how this aggregation occurs across all loss terms.
- Protocols for all the different `state` types/fields
- Could use another dataclass instead of a dict for loss terms,
  though if they're not referenced in transformations I don't 
  know that it really matters.
- L2 by default, but should allow for other norms

:copyright: Copyright 2023 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from abc import abstractmethod
import logging
from typing import (
    ClassVar, 
    Dict, 
    Optional, 
    Sequence,
    Tuple,
)

import equinox as eqx
from equinox import AbstractVar
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree

from feedbax.state import CartesianState2D, HasEffectorState


logger = logging.getLogger(__name__)


class LossDict(eqx.Module):
    ...


class AbstractLoss(eqx.Module):
    """Abstract base class for loss functions.
    
    Enforces that concrete subclasses should have a string label.
    
    TODO: 
    - Time aggregation should happen in every concrete subclass.
      Could add a method/abstractvar to `AbstractLoss`.
    - Should probably allow the user to override with their own label.
    """
    labels: AbstractVar[Tuple[str, ...]]
    
    @abstractmethod
    def __call__(
        self,
        states: PyTree, 
        targets: PyTree, 
        task_inputs: Optional[PyTree] = None,
    ) -> Tuple[float, Dict[str, float]]:
        ...        


class CompositeLoss(AbstractLoss):
    """Composite of simpler loss functions.
    
    TODO:
    - Different aggregation schemes.
    - Perhaps, nesting of losses. As of now, if any of the component losses
      are themselves instances of `CompositeLoss`, their own component terms
      are not remembered. 
    - Perhaps change the labeling scheme; if we don't allow nesting then 
      it is inappropriate to just take the first label name from each component.
    """
    terms: Sequence[AbstractLoss]
    weights: Optional[Sequence[float]]
    labels: Tuple[str, ...]
    
    def __init__(
        self, 
        terms: Sequence[AbstractLoss], 
        weights: Optional[Sequence[float]] = None,
    ):
        if weights is None:
            weights = [1.] * len(terms)
        elif not len(terms) == len(weights):
            raise ValueError("Mismatch between number of loss terms and term weights")
        #! assumes all the components are simple losses, and `labels` is a one-tuple
        if isinstance(terms, dict):
            self.labels = tuple(terms.keys())
            self.terms = terms 
        else:
            self.labels = [term.labels[0] for term in terms]
            self.terms = dict(zip(self.labels, terms))
        if not isinstance(weights, dict):
            self.weights = dict(zip(self.labels, weights))
        else:
            self.weights = weights
    
    @jax.named_scope("fbx.CompositeLoss")
    def __call__(
        self, 
        states: PyTree, 
        targets: PyTree,
        task_inputs: Optional[PyTree] = None,
    ) -> Tuple[float, Dict[str, float]]:
        
        # evaluate loss terms
        loss_terms = jax.tree_map(
            lambda term: term(states, targets, task_inputs)[0], 
            self.terms,
            is_leaf=lambda x: isinstance(x, AbstractLoss),
        )
        # aggregate over batch 
        loss_terms = jax.tree_map(lambda x: jnp.mean(x, axis=0), loss_terms)
        if self.weights is not None:
            # term scaling
            loss_terms = jax.tree_map(
                lambda term, weight: term * weight, 
                loss_terms, 
                self.weights
            )
        
        loss = jax.tree_util.tree_reduce(lambda x, y: x + y, loss_terms)
        
        return loss, loss_terms


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
    labels: Tuple[str, ...]
    discount: Optional[Float[Array, "time"]] = None

    def __init__(
        self, 
        labels=("effector_position",), 
        discount=None,
    ):
        self.labels = labels
        if discount is not None:
            self.discount = discount[None, :]  # singleton batch dimension
        else:   
            self.discount = None

    def __call__(
        self, 
        states: HasEffectorState, 
        targets: PyTree,
        task_inputs: Optional[PyTree] = None,
    ) -> Tuple[float, Dict[str, float]]:
        
        # sum over xyz
        loss = jnp.sum(
            (states.mechanics.effector.pos - targets.pos) ** 2, 
            axis=-1
        )
        
        # temporal discount
        if self.discount is not None:
            loss = loss * self.discount
        
        # sum over time
        loss = jnp.sum(loss, axis=-1)
        
        return loss, {self.labels[0]: loss}


class EffectorStraightPathLoss(AbstractLoss):
    """Penalize non-straight paths between initial and final state.
    
    Calculates the length of the paths followed, and normalizes by the
    Euclidean distance between the initial and final state.
    
    The parameter `normalize_by` controls whether to normalize by the 
    Euclidean distance between the initial and final actual states, or the 
    initial actual state, but the goal state defined by the task.
    """
    labels: Tuple[str, ...] = ("effector_straight_path",)
    normalize_by: str = "actual"

    def __call__(
        self, 
        states: HasEffectorState, 
        targets: PyTree,
        task_inputs: Optional[PyTree] = None,
    ) -> Tuple[float, Dict[str, float]]:
        
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
        
        return loss, {self.labels[0]: loss}


class EffectorFixationLoss(AbstractLoss):
    """"""
    labels: Tuple[str, ...] = ("effector_fixation",)
    
    def __call__(
        self, 
        states: PyTree, 
        targets: PyTree,
        task_inputs: PyTree,
    ) -> Tuple[float, Dict[str, float]]:
        
        loss = jnp.sum(
            (states.mechanics.effector.pos - targets.pos) ** 2, 
            axis=-1
        )
        
        loss = loss * jnp.squeeze(task_inputs.hold)
        
        # sum over time
        loss = jnp.sum(loss, axis=-1)
        
        return loss, {self.labels[0]: loss}


class EffectorFinalVelocityLoss(AbstractLoss):
    """
    
    TODO: how do we handle calculating oss for a single timestep only?
    """
    labels: Tuple[str, ...] = ("effector_final_velocity",)

    def __call__(
        self, 
        states: PyTree, 
        targets: PyTree,
        task_inputs: Optional[PyTree] = None,
    ) -> Tuple[float, Dict[str, float]]:
        
        loss = jnp.sum(
            (states.mechanics.effector.vel[:, -1] - targets.vel[:, -1]) ** 2, 
            axis=-1
        )
        
        return loss, {self.labels[0]: loss}


class NetworkOutputLoss(AbstractLoss):
    """"""
    labels: Tuple[str, ...] = ("nn_output",)

    def __call__(
        self, 
        states: PyTree, 
        #! **kwargs?
        targets: PyTree,
        task_inputs: Optional[PyTree] = None,
    ) -> Tuple[float, Dict[str, float]]:
        
        loss = jnp.sum(states.network.output ** 2, axis=-1)
        
        # sum over time
        loss = jnp.sum(loss, axis=-1)
        
        return loss, {self.labels[0]: loss}


class NetworkActivityLoss(AbstractLoss):
    """"""
    labels: Tuple[str, ...] = ("nn_activity",)

    def __call__(
        self, 
        states: PyTree, 
        targets: PyTree,
        task_inputs: Optional[PyTree] = None,
    ) -> Tuple[float, Dict[str, float]]:
        
        loss = jnp.sum(states.network.activity ** 2, axis=-1)
        
        # sum over time
        loss = jnp.sum(loss, axis=-1)
        
        return loss, {self.labels[0]: loss}

        
def power_discount(n_steps, discount_exp=6):
    """Discounting vector, a power law curve from 0 to 1, start to end.
    """
    return jnp.linspace(1. / n_steps, 1., n_steps) ** discount_exp