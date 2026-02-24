"""Biomechanical plant models.

:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Mapping, Sequence
import logging
from typing import Generic, Optional, Tuple

import equinox as eqx
from equinox import Module, field
import jax
import jax.numpy as jnp
import jax.tree as jt
from jaxtyping import Array, Float, PRNGKeyArray, PyTree, Scalar

from feedbax.dynamics import AbstractDynamicalSystem
from feedbax.mechanics.muscle import AbstractMuscle, MuscleState
from feedbax.mechanics.skeleton.arm import TwoLinkArm
from feedbax.mechanics.skeleton.skeleton import AbstractSkeleton, AbstractSkeletonState
from feedbax.state import StateBounds, StateT, clip_state


logger = logging.getLogger(__name__)


class PlantState(Module, Generic[StateT]):
    """State of a biomechanical plant.

    Attributes:
        skeleton: The skeletal state.
        muscles: The muscle state, if present.
    """

    skeleton: StateT
    muscles: Optional[MuscleState] = None


class DynamicsComponent(eqx.Module, Generic[StateT]):
    dynamics: AbstractDynamicalSystem
    where_input: Callable[[PyTree[Array], StateT], PyTree[Array]]
    where_state: Callable[[StateT], PyTree[Array]]


class AbstractPlant(AbstractDynamicalSystem[PlantState]):
    """Base class for plant models combining kinematics and dynamics."""

    skeleton: AbstractSkeleton
    clip_states: bool = True

    def vector_field(
        self, t: Scalar, state: PlantState, input: PyTree[Array]
    ) -> PlantState:
        d_state = jt.map(jnp.zeros_like, state)
        for component in self.dynamics_spec.values():
            d_state = eqx.tree_at(
                component.where_state,
                d_state,
                component.dynamics.vector_field(
                    t,
                    component.where_state(state),
                    component.where_input(input, state),
                ),
            )
        return d_state

    def kinematics_update(
        self,
        input: PyTree[Array],
        state: PlantState,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PlantState:
        """Apply instantaneous (non-ODE) updates to the plant state."""
        return state

    @property
    def bounds(self) -> PyTree[StateBounds]:
        return PlantState(
            skeleton=getattr(self.skeleton, "bounds", StateBounds(low=None, high=None)),
            muscles=None,
        )

    @property
    def input_size(self) -> int:
        return self.skeleton.input_size

    @property
    def dynamics_spec(self) -> Mapping[str, DynamicsComponent[PlantState]]:
        """Mapping of differential components. Subclasses may override."""
        raise NotImplementedError

    @abstractmethod
    def init(self, *, key: PRNGKeyArray) -> PlantState:
        """Return a default plant state."""
        ...

    def _clip_state(self, bounds: StateBounds, substate):
        return clip_state(bounds, substate)


class AbstractMuscledPlant(AbstractPlant):
    # Override clip_states to come before muscle_model (dataclass field ordering)
    clip_states: bool = True
    muscle_model: AbstractMuscle = field(default=None)  # type: ignore

    @property
    def bounds(self) -> PyTree[StateBounds]:
        return PlantState(
            skeleton=self.skeleton.bounds,
            muscles=self.muscle_model.bounds,
        )


class DirectForceInput(AbstractPlant):
    """Skeleton controlled directly by external forces/torques."""

    skeleton: AbstractSkeleton
    clip_states: bool

    def __init__(
        self,
        skeleton: AbstractSkeleton,
        clip_states: bool = True,
        *,
        key: Optional[PRNGKeyArray] = None,
    ):
        self.skeleton = skeleton
        self.clip_states = clip_states

    @property
    def dynamics_spec(self) -> Mapping[str, DynamicsComponent[PlantState]]:
        return dict(
            {
                "skeleton": DynamicsComponent[PlantState](
                    dynamics=self.skeleton,
                    where_input=lambda input, state: input,
                    where_state=lambda state: state.skeleton,
                )
            }
        )

    def kinematics_update(
        self,
        input: PyTree[Array],
        state: PlantState,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PlantState:
        if not self.clip_states:
            return state
        return eqx.tree_at(
            lambda s: s.skeleton,
            state,
            self._clip_state(self.bounds.skeleton, state.skeleton),
        )

    @property
    def memory_spec(self) -> PyTree[bool]:
        return PlantState(
            skeleton=True,
            muscles=False,
        )

    def init(self, *, key: PRNGKeyArray) -> PlantState:
        return PlantState(
            skeleton=self.skeleton.init(key=key),
            muscles=None,
        )


class MuscledArm(AbstractMuscledPlant):
    """Two-link arm actuated by muscles."""

    skeleton: AbstractSkeleton
    muscle_model: AbstractMuscle
    activator: AbstractDynamicalSystem
    clip_states: bool
    n_muscles: int
    moment_arms: Float[Array, "links=2 muscles"]
    theta0: Float[Array, "links=2 muscles"]
    l0: Float[Array, "muscles"]
    f0: Float[Array, "muscles"]

    def __init__(
        self,
        muscle_model: AbstractMuscle,
        activator: AbstractDynamicalSystem,
        skeleton: AbstractSkeleton = TwoLinkArm(),
        clip_states: bool = True,
        moment_arms: Float[Array, "links=2 muscles"] | Sequence[Sequence[float]] = (
            jnp.array(
                (
                    (2.0, -2.0, 0.0, 0.0, 1.50, -2.0),
                    (0.0, 0.0, 2.0, -2.0, 2.0, -1.50),
                )
            )
        ),
        theta0: Float[Array, "links=2 muscles"] | Sequence[Sequence[float]] = (
            2
            * jnp.pi
            * jnp.array(
                (
                    (15.0, 4.88, 0.0, 0.0, 4.5, 2.12),
                    (0.0, 0.0, 80.86, 109.32, 92.96, 91.52),
                )
            )
            / 360.0
        ),
        l0: Float[Array, "muscles"] | Sequence[float] = jnp.array(
            (7.32, 3.26, 6.4, 4.26, 5.95, 4.04)
        ),
        f0: Float[Array, "muscles"] | Sequence[float] = jnp.array(
            (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        ),
        *,
        key: Optional[PRNGKeyArray] = None,
    ):
        self.skeleton = skeleton
        self.activator = activator
        self.clip_states = clip_states

        self.moment_arms = jnp.array(moment_arms)
        self.theta0 = jnp.array(theta0)
        self.l0 = jnp.array(l0)
        self.f0 = jnp.array(f0)
        self.n_muscles = self.moment_arms.shape[1]

        if not self.theta0.shape[1] == self.l0.shape[0] == self.moment_arms.shape[1]:
            raise ValueError(
                "moment_arms, theta0, and l0 must have the same number of muscles"
            )
        self.muscle_model = muscle_model.change_n_muscles(self.n_muscles)

    @property
    def dynamics_spec(self) -> Mapping[str, DynamicsComponent[PlantState]]:
        return dict(
            {
                "muscle_activation": DynamicsComponent(
                    dynamics=self.activator,
                    where_input=lambda input, state: input,
                    where_state=lambda state: state.muscles.activation,
                ),
                "skeleton": DynamicsComponent(
                    dynamics=self.skeleton,
                    where_input=lambda input, state: state.skeleton.torque,
                    where_state=lambda state: state.skeleton,
                ),
            }
        )

    def kinematics_update(
        self,
        input: PyTree[Array],
        state: PlantState,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PlantState:
        if self.clip_states:
            state = eqx.tree_at(
                lambda s: s.skeleton,
                state,
                self._clip_state(self.bounds.skeleton, state.skeleton),
            )

        # Muscle geometry
        length = self._muscle_length(state.skeleton.angle)
        velocity = self._muscle_velocity(state.skeleton.d_angle)
        state = eqx.tree_at(
            lambda s: (s.muscles.length, s.muscles.velocity),
            state,
            (length, velocity),
        )

        # Clip muscle state
        state = eqx.tree_at(
            lambda s: s.muscles,
            state,
            self._clip_state(self.bounds.muscles, state.muscles),
        )

        # Muscle tension
        muscles = self.muscle_model(state.muscles.activation, state.muscles, key=key)
        state = eqx.tree_at(lambda s: s.muscles, state, muscles)

        # Muscle torques
        torque = self._muscle_torques(state.muscles)
        state = eqx.tree_at(lambda s: s.skeleton.torque, state, torque)

        return state

    def _muscle_length(self, angle: Array) -> Array:
        moment_arms, l0, theta0 = self.moment_arms, self.l0, self.theta0
        l = (
            1
            + (
                moment_arms[0] * (theta0[0] - angle[0])
                + moment_arms[1] * (theta0[1] - angle[1])
            )
            / l0
        )
        return l

    def _muscle_velocity(self, d_angle: Array) -> Array:
        moment_arms, l0 = self.moment_arms, self.l0
        v = (moment_arms[0] * d_angle[0] + moment_arms[1] * d_angle[1]) / l0
        return v

    def _muscle_torques(self, muscles: MuscleState) -> Array:
        return self.moment_arms @ (self.f0 * muscles.tension)

    @property
    def memory_spec(self) -> PyTree[bool]:
        return PlantState(
            skeleton=True,
            muscles=True,
        )

    def init(self, *, key: PRNGKeyArray) -> PlantState:
        key1, key2 = jax.random.split(key)
        return PlantState(
            skeleton=self.skeleton.init(key=key1),
            muscles=self.muscle_model.init(key=key2),
        )

    @property
    def input_size(self) -> int:
        return self.n_muscles
