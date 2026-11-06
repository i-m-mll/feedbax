"""Kinematics for a planar serial arm with relative joint angles."""

import jax
import jax.numpy as jnp
from jaxtyping import Array

from feedbax.runtime.state import CartesianState


def _link_positions_and_offsets(lengths: Array, angle: Array) -> tuple[Array, Array]:
    heading = jnp.cumsum(angle)
    offsets = lengths[:, None] * jnp.stack((jnp.cos(heading), jnp.sin(heading)), axis=-1)
    return jnp.cumsum(offsets, axis=0), offsets


def link_kinematics(lengths: Array, angle: Array, d_angle: Array) -> tuple[Array, Array]:
    """Return position and velocity of each link endpoint as (links, xy) arrays."""
    positions, offsets = _link_positions_and_offsets(lengths, angle)
    # A link's velocity is its offset rotated 90 degrees counterclockwise,
    # scaled by its angular velocity. Keeping the rate separate preserves
    # promotion when joint angles and rates have different dtypes.
    offset_tangents = jnp.stack((-offsets[:, 1], offsets[:, 0]), axis=-1)
    offset_velocities = offset_tangents * jnp.cumsum(d_angle)[:, None]
    return positions, jnp.cumsum(offset_velocities, axis=0)


def effector_jacobian(lengths: Array, angle: Array) -> Array:
    """Return the Jacobian of the distal link endpoint, with shape (xy, links)."""

    def effector_position(angle: Array) -> Array:
        positions, _ = _link_positions_and_offsets(lengths, angle)
        return positions[-1]

    return jax.jacfwd(effector_position)(angle)


def distal_state(links: CartesianState) -> CartesianState:
    """Select the Cartesian state of the last link endpoint."""
    return CartesianState(pos=links.pos[-1], vel=links.vel[-1], force=links.force[-1])
