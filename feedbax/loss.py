"""

TODO:
- Protocols for all the different `state` types/fields
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
    Protocol, 
    Tuple,
    runtime_checkable,
)

import equinox as eqx
from equinox import AbstractVar
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree


logger = logging.getLogger(__name__)


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
        self, states: PyTree, targets: PyTree
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
    terms: Tuple[AbstractLoss, ...]
    weights: Optional[Tuple[float, ...]]
    labels: Tuple[str, ...]
    
    def __init__(self, terms, weights):
        assert len(terms) == len(weights)
        #! assumes all the components are simple losses, and `labels` is a one-tuple
        self.labels = [term.labels[0] for term in terms]
        self.terms = dict(zip(self.labels, terms))
        self.weights = dict(zip(self.labels, weights))
        
        
    def __call__(
        self, states: PyTree, 
        targets: PyTree
    ) -> Tuple[float, Dict[str, float]]:
        
        # evaluate loss terms
        loss_terms = jax.tree_map(
            lambda term: term(states, targets)[0], 
            self.terms,
            is_leaf=lambda x: isinstance(x, AbstractLoss),
        )
        # aggregate over time 
        #! this should be done in all the other classes as well! they aren't returning scalars
        loss_terms = jax.tree_map(lambda x: jnp.sum(x, axis=-1), loss_terms)
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


@runtime_checkable
class HasEffectorState(Protocol):
    effector: Tuple[Float[Array, "2"], Float[Array, "2"]]
    # effector: CartesianState


class EffectorPositionLoss(AbstractLoss):
    """
    
    TODO: do we handle the temporal discount here? or return the sequence of losses
    """
    labels: Tuple[str, ...]
    discount: Optional[Float[Array, "time"]] = None

    def __init__(
        self, 
        labels=("ee_position",), 
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
    ) -> Tuple[float, Dict[str, float]]:
        
        loss = jnp.sum(
            (states.effector[0] - targets[0][:, None]) ** 2, 
            axis=-1
        )
        
        # temporal discount
        if self.discount is not None:
            loss = loss * self.discount
        
        return loss, {self.labels[0]: loss}


# class EffectorDiscountedPositionLoss(AbstractLoss):
#     labels: Tuple[str, ...] = ("ee_position",)
#     discount: 

#     def __call__(
#         self, 
#         states: PyTree, 
#         targets: PyTree,
#     ) -> Tuple[float, Dict[str, float]]:
        
#         loss = jnp.sum(
#             (states.effector[0] - targets[0][:, None]) ** 2, 
#             axis=-1
#         )
        
#         # temporal discount
#         loss = jnp.sum(
#             loss * jnp.exp(-states.time[:, None]), 
#             axis=0
#         )
        
#         return loss, {self.labels[0]: loss}

class EffectorFinalVelocityLoss(AbstractLoss):
    """
    
    TODO: how do we handle calculating oss for a single timestep only?
    """
    labels: Tuple[str, ...] = ("ee_final_velocity",)

    def __call__(
        self, states: PyTree, targets: PyTree
    ) -> Tuple[float, Dict[str, float]]:
        
        loss = jnp.sum(
            (states.effector[1][:, -1] - targets[1]) ** 2, 
            axis=-1
        )
        
        # so the result plays nicely with time aggregation
        loss = jnp.expand_dims(loss, axis=-1)
        
        return loss, {self.labels[0]: loss}


class ControlLoss(AbstractLoss):
    """"""
    labels: Tuple[str, ...] = ("control",)

    def __call__(
        self, states: PyTree, targets: PyTree
    ) -> Tuple[float, Dict[str, float]]:
        
        loss = jnp.sum(states.control ** 2, axis=-1)
        
        return loss, {self.labels[0]: loss}


class NetworkActivityLoss(AbstractLoss):
    """"""
    labels: Tuple[str, ...] = ("activity",)

    def __call__(
        self, states: PyTree, targets: PyTree
    ) -> Tuple[float, Dict[str, float]]:
        
        loss = jnp.sum(states.hidden ** 2, axis=-1)
        
        return loss, {self.labels[0]: loss}

        
    