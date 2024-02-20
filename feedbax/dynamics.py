"""Base classes for continuous dynamical systems.

:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0, see LICENSE for details.
"""

from abc import abstractmethod, abstractproperty
import logging 
from typing import Optional

import equinox as eqx
from equinox import AbstractVar
import jax 
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

from feedbax.model import AbstractModel
from feedbax.state import CartesianState, StateBounds, StateT


logger = logging.getLogger(__name__)


class AbstractDynamicalSystem(AbstractModel[StateT]):
    """Base class for continuous dynamical systems.
    
    ??? dev-note "Development note"
        Vector fields for biomechanical models are generally not
        time-dependent. That is, the argument `t` to `vector_field` is
        typically unused. This is apparent in the way we alias `vector_field`
        to `__call__`, which is a method that `AbstractModel` requires.
        
        Perhaps it is unnecessary to inherit from `AbstractModel`, though.
    """
    
    def __call__(
        self,
        input: PyTree[Array],
        state: StateT, 
        key: PRNGKeyArray,
    ) -> StateT:
        """Alias for `vector_field`, with a modified signature."""
        return self.vector_field(None, state, input)
    
    @abstractmethod
    def vector_field(
        self, 
        t: float | None, 
        state: StateT, 
        input: PyTree[Array],  # controls
    ) -> StateT:
        """Returns scalar (e.g. time) derivatives of the system's states."""
        ...

    @abstractproperty
    def input_size(self) -> int:
        """Number of control inputs."""
        ...
    
    @abstractmethod
    def init(self, *, key: Optional[PRNGKeyArray] = None) -> StateT:
        """Returns the initial state of the system.
        """
        ...
    
    def step(self) -> "AbstractDynamicalSystem[StateT]":
        return self
        

class AbstractLTISystem(AbstractDynamicalSystem[StateT]):
    """A linear, continuous, time-invariant system.
    
    Inspired by [this Diffrax example](https://docs.kidger.site/diffrax/examples/kalman_filter/).
    
    Attributes:
        A: The state evolution matrix.
        B: The control matrix.
        C: The observation matrix.
    """
    A: AbstractVar[Float[Array, "state state"]] 
    B: AbstractVar[Float[Array, "state input"]]  
    C: AbstractVar[Float[Array, "obs state"]]  
    
    @jax.named_scope("fbx.AbstractLTISystem")
    def vector_field(
        self, 
        t: float, 
        state: Array,
        input: Float[Array, "input"]
    ) -> Array:
        """Returns time derivatives of the system's states.        
        """ 
        return self.A @ state + self.B @ input
    
    @property
    def input_size(self) -> int:
        """Number of control variables."""
        return self.B.shape[1]
    
    @property 
    def state_size(self) -> int:
        """Number of state variables."""
        return self.A.shape[1]
    
    @property
    def bounds(self) -> StateBounds[CartesianState]:
        return StateBounds(low=None, high=None)
    
    def init(
        self,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> CartesianState:
        """Return a default state for the linear system."""
        return CartesianState()