"""Base class for models of skeletal dynamics.

:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from abc import abstractmethod
from typing import Optional, TypeVar

from equinox import Module
from jax import Array
from jaxtyping import PRNGKeyArray, Scalar

from feedbax.mechanics.dynamics import AbstractDynamicalSystem
from feedbax.runtime.state import CartesianState




class AbstractSkeletonState(Module):
    ...
    # effector: CartesianState
    # config: PyTree


StateT = TypeVar("StateT", bound=AbstractSkeletonState)


class AbstractSkeleton(AbstractDynamicalSystem[StateT]):
    """Base class for models of skeletal dynamics."""

    @abstractmethod
    def vector_field(
        self,
        t: Scalar,
        state: StateT,
        input: Array,
    ) -> StateT:
        """Return the time derivative of the state of the skeleton."""
        ...

    @abstractmethod
    def init(self, *, key: PRNGKeyArray) -> StateT:
        """Return a default state for the skeleton."""
        ...

    @abstractmethod
    def forward_kinematics(self, state: StateT) -> CartesianState:
        """Return the Cartesian state of the joints and end effector,
        given the configuration of the skeleton."""
        ...

    @abstractmethod
    def inverse_kinematics(self, effector_state: CartesianState) -> StateT:
        """Return the configuration of the skeleton given the Cartesian state
        of the end effector."""
        ...

    @abstractmethod
    def effector(self, state: StateT) -> CartesianState:
        """Return the Cartesian state of the end effector."""
        ...

    def project_state(self, state: StateT) -> StateT:
        """Project a configuration after an accepted integration step."""
        return state

    @abstractmethod
    def update_state_given_effector_force(
        self,
        effector_force: Array,
        state: StateT,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> StateT:
        """Replace the held endpoint force. Pass zero to release the load."""
        ...
