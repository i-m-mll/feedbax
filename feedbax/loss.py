"""

TODO:
- Protocols for all the different `state` types/fields
- Interface of `CompositeLoss` should match `AbstractLoss`
  i.e. `AbstractLoss` concretes' `__call__` should also return a single-entry dict `loss_terms`

:copyright: Copyright 2023 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from abc import abstractmethod
import logging
from typing import ClassVar, Dict, Optional, Tuple

import equinox as eqx
from equinox import AbstractClassVar
import jax
import jax.numpy as jnp
from jaxtyping import PyTree


logger = logging.getLogger(__name__)


class AbstractLoss(eqx.Module):
    """Abstract base class for loss functions.
    
    Enforces that concrete subclasses should have a string label.
    
    TODO: 
    - Should probably allow the user to override with their own label.
      Actually, 
    """
    label: AbstractClassVar[str]
    
    @abstractmethod
    def __call__(
        self, states: PyTree, targets: PyTree
    ) -> Tuple[float, Dict[str, float]]:
        ...        


class CompositeLoss(AbstractLoss):
    terms: Tuple[AbstractLoss, ...]
    weights: Optional[Tuple[float, ...]]
    
    def __init__(self, terms, weights):
        assert len(terms) == len(weights)
        labels = [term.LABEL for term in terms]
        self.terms = dict(zip(labels, terms))
        self.weights = dict(zip(labels, weights))
        
    def __call__(
        self, states: PyTree, targets: PyTree
    ) -> Tuple[float, Dict[str, float]]:
        
        # evaluate loss terms
        loss_terms = jax.tree_map(lambda term: term(states, targets), self.terms)
        # aggregate over time 
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


class EffectorPositionLoss(AbstractLoss):
    """
    
    TODO: do we handle the temporal discount here? or return the sequence of losses
    """
    LABEL: ClassVar[str] = "ee_position"

    def __call__(
        self, states: PyTree, targets: PyTree
    ) -> Tuple[float, Dict[str, float]]:
        
        loss = jnp.sum(
            (states.ee.pos - targets.pos[:, None]) ** 2, 
            axis=-1
        )
        
        return loss, {self.LABEL: loss}


class EffectorVelocityLoss(AbstractLoss):
    """
    
    TODO: how do we handle calculating oss for a single timestep only?
    """
    LABEL: ClassVar[str] = "ee_final_velocity"

    def __call__(
        self, states: PyTree, targets: PyTree
    ) -> Tuple[float, Dict[str, float]]:
        
        loss = jnp.sum(
            (states.ee.vel - targets.vel) ** 2, 
            axis=-1
        )
        
        return loss, {self.LABEL: loss}


class ControlLoss(AbstractLoss):
    """"""
    LABEL: ClassVar[str] = "control"

    def __call__(
        self, states: PyTree, targets: PyTree
    ) -> Tuple[float, Dict[str, float]]:
        
        loss = jnp.sum(states.control ** 2, axis=-1)
        
        return loss, {self.LABEL: loss}


class NetworkActivityLoss(AbstractLoss):
    """"""
    LABEL: ClassVar[str] = "activity"

    def __call__(
        self, states: PyTree, targets: PyTree
    ) -> Tuple[float, Dict[str, float]]:
        
        loss = jnp.sum(states.hidden ** 2, axis=-1)
        
        return loss, {self.LABEL: loss}

        
    