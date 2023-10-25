"""

:copyright: Copyright 2023 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

import logging

import equinox as eqx
from jaxtyping import Array, Float


logger = logging.getLogger(__name__)


class CartesianState2D(eqx.Module):
    """Cartesian state of a system."""
    pos: Float[Array, "... 2"]
    vel: Float[Array, "... 2"]