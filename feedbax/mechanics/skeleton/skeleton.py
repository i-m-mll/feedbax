"""Base class for models of skeletal dynamics.

:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from abc import abstractmethod
import logging
from typing import Optional, TypeVar

from equinox import Module
from jax import Array
from jaxtyping import PRNGKeyArray, Scalar

from feedbax.dynamics import AbstractDynamicalSystem, LinearSystem
from feedbax.state import CartesianState


logger = logging.getLogger(__name__)


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

    @abstractmethod
    def update_state_given_effector_force(
        self,
        effector_force: Array,
        state: StateT,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> StateT:
        """Update the state of the skeleton given a force on the end effector."""
        ...

    def linearize(self, state: StateT) -> LinearSystem:
        """Continuous-time ``(A, B, B_w)`` matrices for the skeleton about ``state``.

        Returns the bare-skeleton linearisation: the state vector for the
        returned ``LinearSystem`` is the skeleton configuration only, with
        no force-filter augmentation. ``Mechanics.linearize_with_force_filter``
        composes the result with a first-order force filter when one is in use.

        For exactly-linear skeletons (e.g. ``PointMass``) this returns the
        exact closed-form matrices and ``state`` is ignored. For nonlinear
        skeletons (``TwoLinkArm``, ``MJXSkeleton``), an autodiff-based
        Jacobian linearisation about ``state`` would be the natural extension
        but is not implemented in this base class.

        Subclasses that cannot meaningfully linearise (e.g. MJX-backed
        skeletons whose dynamics are opaque) should raise
        ``NotImplementedError`` rather than provide a misleading
        approximation.

        Args:
            state: The nominal skeleton state about which to linearise.
                Ignored by exactly-linear skeletons.

        Returns:
            A ``LinearSystem`` with continuous-time ``A``, ``B``, and
            (optionally) ``B_w`` matrices, plus ``state_indices`` slots,
            and ``dt=None``.

        Raises:
            NotImplementedError: If the subclass has not provided an
                implementation.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.linearize is not implemented. "
            f"Override on the subclass to provide a closed-form or "
            f"autodiff-based linearisation."
        )

