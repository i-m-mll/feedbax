"""

TODO:
- Protocols for all the different `state` types/fields
- L2 by default, but should allow for other norms

:copyright: Copyright 2023 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from abc import abstractmethod
import logging
from typing import ClassVar, Dict, Optional, Tuple

import equinox as eqx
from equinox import AbstractVar
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
    labels: AbstractVar[Tuple[str, ...]]
    
    @abstractmethod
    def __call__(
        self, states: PyTree, targets: PyTree
    ) -> Tuple[float, Dict[str, float]]:
        ...        


class CompositeLoss(AbstractLoss):
    terms: Tuple[AbstractLoss, ...]
    weights: Optional[Tuple[float, ...]]
    labels: Tuple[str, ...]
    
    def __init__(self, terms, weights):
        assert len(terms) == len(weights)
        # TODO: flatten 
        self.labels = [term.labels for term in terms]
        self.terms = dict(zip(self.labels, terms))
        self.weights = dict(zip(self.labels, weights))
        
        
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
    labels: Tuple[str, ...] = ("ee_position",)

    def __call__(
        self, states: PyTree, targets: PyTree
    ) -> Tuple[float, Dict[str, float]]:
        
        loss = jnp.sum(
            (states.ee.pos - targets.pos[:, None]) ** 2, 
            axis=-1
        )
        
        return loss, {self.labels[0]: loss}


class EffectorVelocityLoss(AbstractLoss):
    """
    
    TODO: how do we handle calculating oss for a single timestep only?
    """
    labels: Tuple[str, ...] = ("ee_final_velocity",)

    def __call__(
        self, states: PyTree, targets: PyTree
    ) -> Tuple[float, Dict[str, float]]:
        
        loss = jnp.sum(
            (states.ee.vel - targets.vel) ** 2, 
            axis=-1
        )
        
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

        
    