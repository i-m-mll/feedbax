"""

:copyright: Copyright 2023 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

import logging
from typing import Optional, Protocol, runtime_checkable

import equinox as eqx
from equinox import field
import jax.numpy as jnp
from jaxtyping import Array, Float


logger = logging.getLogger(__name__)


class CartesianState2D(eqx.Module):
    """Cartesian state of a system."""
    pos: Float[Array, "... 2"]
    vel: Float[Array, "... 2"]
    force: Float[Array, "... 2"] = field(default_factory=lambda: jnp.zeros(2))
    
    
@runtime_checkable
class HasEffectorState(Protocol):
    effector: CartesianState2D


@runtime_checkable
class HasMechanicsState(Protocol):
    mechanics: HasEffectorState