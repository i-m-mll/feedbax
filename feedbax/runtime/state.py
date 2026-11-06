"""Common types used across the package.

:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from collections.abc import Callable

import logging
from typing import (
    Optional,
    Generic,
    TypeAlias,
    TypeVar,
)

from equinox import Module, field
import jax.numpy as jnp
import jax.tree as jt
from jaxtyping import Array, Float, PyTree


logger = logging.getLogger(__name__)


State: TypeAlias = PyTree #[Array]
StateT = TypeVar("StateT", bound=State)


class StateBounds(Module, Generic[StateT]):
    """Specifies bounds on a state.

    Attributes:
        low: A state PyTree giving lower bounds.
        high: A state PyTree giving upper bounds.
    """

    low: Optional[StateT]
    high: Optional[StateT]

    @property
    def filter_spec(self) -> PyTree[bool]:
        """A matching PyTree, indicated which parts of the state are bounded."""
        return jt.map(
            lambda x: x is not None,
            self,
            is_leaf=lambda x: isinstance(x, Array) or x is None,
        )


class CartesianState(Module):
    """Cartesian state of a mechanical system in two spatial dimensions.

    Attributes:
        pos: The position coordinates of the point(s) in the system.
        vel: The respective velocities.
        force: The respective forces.
    """

    pos: Float[Array, "... 2"] = field(default_factory=lambda: jnp.zeros(2))
    vel: Float[Array, "... 2"] = field(default_factory=lambda: jnp.zeros(2))
    force: Float[Array, "... 2"] = field(default_factory=lambda: jnp.zeros(2))


def clip_state(
    bounds: StateBounds[StateT],
    state: StateT,
) -> StateT:
    """Clip values outside the bounds, preserving dtypes and NaNs.

    Arguments:
        bounds: The lower and upper bounds to clip the state to.
        state: The state to clip.
    """

    if bounds.low is not None:
        state = _clip_state_to_bound(state, bounds.low, jnp.less)
    if bounds.high is not None:
        state = _clip_state_to_bound(state, bounds.high, jnp.greater)
    return state


def _clip_state_to_bound(
    state: StateT,
    bound: StateT,
    outside: Callable[[Array, Array], Array],
) -> StateT:
    """A single (one-sided) clipping operation."""
    def clip_leaf(limit, value):
        if limit is None:
            return value
        limit = jnp.asarray(limit, dtype=jnp.result_type(value))
        return jnp.where(outside(value, limit), limit, value)

    return jt.map(
        clip_leaf,
        bound,
        state,
        is_leaf=lambda x: x is None,
    )
