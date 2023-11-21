"""Base classes for continuous dynamical systems.

TODO: 
- Maybe `System` protocol should be `AbstractSystem(eqx.Module)`.

:copyright: Copyright 2023 by MLL <mll@mll.bio>.
:license: Apache 2.0, see LICENSE for details.
"""

import logging 
from typing import Protocol, TypeVar

import equinox as eqx
from jaxtyping import Array, Float, PyTree

from feedbax.types import AbstractState, StateBounds


logger = logging.getLogger(__name__)

    
StateT = TypeVar("StateT", bound=AbstractState)


class System(Protocol[StateT]):
    """Interface expected of dynamical systems classes.
    
    TODO:
    - Signature of `init`.
    """
    
    def vector_field(
        self, 
        t: float, 
        y: StateT, 
        args: PyTree,  # controls
    ) -> StateT:
        """Vector field of the system."""
        ...

    @property
    def control_size(self) -> int:
        """Number of control inputs."""
        ...
    
    def init(self, **kwargs) -> StateT:
        """Initial state of the system."""
        ...
    
    @property
    def bounds(self) -> StateBounds[StateT]:
        """Suggested bounds on the state.
        
        These will only be applied if...
        """
        ...