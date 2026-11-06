"""Mechanics of planar arms with relative joint angles."""

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


def two_link_inertia(
    lengths: Array,
    masses: Array,
    inertias: Array,
    center_offsets: Array,
    elbow_angle: Array,
) -> tuple[Array, Array]:
    """Return the joint-space inertia matrix and inertial coupling coefficient."""
    distal_inertia = inertias[1]
    base_inertia = inertias[0] + distal_inertia + masses[1] * lengths[0] ** 2
    coupling = masses[1] * lengths[0] * center_offsets[1]
    cross_term = coupling * jnp.cos(elbow_angle)
    cross_inertia = distal_inertia + cross_term
    matrix = jnp.array(
        (
            (base_inertia + 2 * cross_term, cross_inertia),
            (cross_inertia, distal_inertia),
        )
    )
    return matrix, coupling


def two_link_angular_acceleration(
    lengths: Array,
    masses: Array,
    inertias: Array,
    center_offsets: Array,
    friction: Array,
    angle: Array,
    d_angle: Array,
    torque: Array,
) -> Array:
    """Solve for angular acceleration from the applied joint torques."""
    matrix, coupling = two_link_inertia(lengths, masses, inertias, center_offsets, angle[1])
    shoulder_rate, elbow_rate = d_angle
    coriolis = (
        jnp.array((-elbow_rate * (2 * shoulder_rate + elbow_rate), shoulder_rate ** 2))
        * coupling
        * jnp.sin(angle[1])
    )
    return jnp.linalg.solve(matrix, torque - coriolis - friction @ d_angle)
