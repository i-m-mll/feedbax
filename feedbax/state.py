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
    """Base class for model states.

    !!! NOTE ""
        Currently this is empty, and only used for collectively typing its subclasses.
    """

    ...


StateT = TypeVar("StateT", AbstractState, Array)


class StateBounds(eqx.Module, Generic[StateT]):
    """Specifies bounds on a state.

    Attributes:
        low: A state PyTree giving lower bounds.
        high: A state PyTree giving upper bounds.
    """

    low: Optional[StateT]
    high: Optional[StateT]

    @cached_property
    def filter_spec(self) -> PyTree[bool]:
        """A matching PyTree, indicated which parts of the state are bounded."""
        return jax.tree_map(
            lambda x: x is not None,
            self,
            is_leaf=lambda x: isinstance(x, Array) or x is None,
        )


class CartesianState(AbstractState):
    """Cartesian state of a mechanical system in two spatial dimensions.

    Attributes:
        pos: The position coordinates of the point(s) in the system.
        vel: The respective velocities.
        force: The respective forces.
    """

    pos: Float[Array, "... 2"] = field(default_factory=lambda: jnp.zeros(2))
    vel: Float[Array, "... 2"] = field(default_factory=lambda: jnp.zeros(2))
    force: Float[Array, "... 2"] = field(default_factory=lambda: jnp.zeros(2))


@runtime_checkable
class HasEffectorState(Protocol):
    effector: CartesianState


@runtime_checkable
class HasMechanicsEffectorState(Protocol):
    mechanics: HasEffectorState


def clip_state(
    bounds: StateBounds[StateT],
    state: StateT,
) -> StateT:
    """Returns a state clipped to the given bounds.

    Arguments:
        bounds: The lower and upper bounds to clip the state to.
        state: The state to clip.
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
