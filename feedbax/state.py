"""Common types used across the package.

:copyright: Copyright 2023 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

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
    ...


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