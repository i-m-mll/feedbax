"""Base class for models of skeletal dynamics.

:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from abc import abstractmethod
import logging
from typing import Optional, TypeVar

from jax import Array
from jaxtyping import PRNGKeyArray

from feedbax.dynamics import AbstractDynamicalSystem
from feedbax.state import AbstractState, CartesianState


logger = logging.getLogger(__name__)


class AbstractSkeletonState(AbstractState):
    ...
    # effector: CartesianState
    # config: PyTree


StateT = TypeVar("StateT", bound=AbstractSkeletonState)


class AbstractSkeleton(AbstractDynamicalSystem[StateT]):
    
    @abstractmethod
    def forward_kinematics(self, state: StateT) -> CartesianState:
        """Return the Cartesian state of the joints given the configuration
        of the skeleton."""
        ...
        
    @abstractmethod 
    def inverse_kinematics(self, state: CartesianState) -> StateT:
        """Return the configuration of the skeleton given the Cartesian state 
        of the final joint (usually, the effector)."""
        ...
        
    @abstractmethod
    def effector(self, state: StateT) -> CartesianState:
        """Return the Cartesian state of the effector."""
        ...
    
    @abstractmethod
    def update_state_given_effector_force(
        self, 
        effector_force: Array, 
        state: StateT,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> StateT:
        """Update the state of the skeleton given an effector force."""
        ...
        
    # @abstractmethod
    # def init(self, *, key: Optional[PRNGKeyArray] = None) -> StateT:
    #     """Returns the initial state of the system.
    #     """
    #     ...
    