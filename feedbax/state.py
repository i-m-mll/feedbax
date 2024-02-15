"""Common types used across the package.

:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from collections.abc import Callable
from copy import deepcopy
from functools import cached_property
import logging
from typing import (
    Optional, 
    Generic,
    Protocol, 
    TypeVar, 
    runtime_checkable,
)

import equinox as eqx
from equinox import field
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree


logger = logging.getLogger(__name__)


class AbstractState(eqx.Module):
    """Base class for model states."""
    
    def copy(self):
        return deepcopy(self)


StateT = TypeVar("StateT", bound=AbstractState)


class StateBounds(eqx.Module, Generic[StateT]):
    """Bounds on a state."""
    low: Optional[StateT]
    high: Optional[StateT] 
    
    @cached_property
    def filter_spec(self) -> PyTree[bool]:
        return jax.tree_map(
            lambda x: x is not None, 
            self, 
            is_leaf=lambda x: isinstance(x, jax.Array) or x is None,
        )


class CartesianState2D(AbstractState):
    """Cartesian state of a system."""
    pos: Float[Array, "... 2"] = field(default_factory=lambda: jnp.zeros(2))
    vel: Float[Array, "... 2"] = field(default_factory=lambda: jnp.zeros(2))
    force: Float[Array, "... 2"] = field(default_factory=lambda: jnp.zeros(2))
    

@runtime_checkable
class HasEffectorState(Protocol):
    effector: CartesianState2D


@runtime_checkable
class HasMechanicsState(Protocol):
    mechanics: HasEffectorState
  
        
def clip_state(
    bounds: StateBounds[StateT],
    state: StateT, 
) -> StateT:
    """Constrain a state to the given bounds.
    
    TODO: 
    - Maybe we can `tree_map` this, but I'm not sure it matters,
      especially since it might require we make a bizarre
      `StateBounds[Callable]` for the operations...
    """

    if bounds.low is not None:
        state = _clip_state_to_bound(
            state, bounds.low, bounds.filter_spec.low, jnp.greater
        )
    if bounds.high is not None:
        state = _clip_state_to_bound(
            state, bounds.high, bounds.filter_spec.high, jnp.less
        )
    return state


def _clip_state_to_bound(
    state: StateT, 
    bound: StateT, 
    filter_spec: PyTree[bool],
    op: Callable,
) -> StateT:
    """A single (one-sided) clipping operation."""
    states_to_clip, states_other = eqx.partition(
        state,
        filter_spec,
    )    
    
    states_clipped = jax.tree_map(
        lambda x, y: jnp.where(op(x, y), x, y),
        states_to_clip,
        bound,
    )
    
    return eqx.combine(states_other, states_clipped)