"""Motor task specifications and generators for RL training.

Provides task types (reach, hold, track, swing) with both Python-level
and fully JAX-traceable samplers for use in vmapped/scanned environments.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


# Task type constants
TASK_REACH = 0
TASK_HOLD = 1
TASK_TRACK = 2
TASK_SWING = 3

TASK_NAMES = {
    TASK_REACH: "reach",
    TASK_HOLD: "hold",
    TASK_TRACK: "track",
    TASK_SWING: "swing",
}


class TaskSpec(NamedTuple):
    """Task specification for a single episode.

    Attributes:
        task_type: Integer task type (0=reach, 1=hold, 2=track, 3=swing).
        target_pos: Target end-effector positions, shape ``(T, 2)``.
        target_vel: Target end-effector velocities, shape ``(T, 2)``.
        perturbation: External perturbation forces, shape ``(T, 2)``.
    """

    task_type: int
    target_pos: Float[Array, "T 2"]
    target_vel: Float[Array, "T 2"]
    perturbation: Float[Array, "T 2"]


def _velocity_from_positions(
    positions: Float[Array, "T 2"], dt: float
) -> Float[Array, "T 2"]:
    """Finite-difference velocity from position trajectory."""
    vel = jnp.zeros_like(positions)
    vel = vel.at[1:].set((positions[1:] - positions[:-1]) / dt)
    return vel


def _vel_from_pos(
    positions: Float[Array, "T 2"], dt: Float[Array, ""],
) -> Float[Array, "T 2"]:
    """Finite-difference velocity. dt may be a JAX tracer (safe in JIT)."""
    vel = jnp.zeros_like(positions)
    return vel.at[1:].set((positions[1:] - positions[:-1]) / dt)


def reach_task(
    timestamps: Float[Array, " T"],
    start: Float[Array, " 2"],
    target: Float[Array, " 2"],
) -> TaskSpec:
    """Minimum-jerk reach from start to target.

    Args:
        timestamps: Time points, shape ``(T,)``.
        start: Starting position, shape ``(2,)``.
        target: Target position, shape ``(2,)``.

    Returns:
        TaskSpec with minimum-jerk position profile.
    """
    t0 = timestamps[0]
    tf = timestamps[-1]
    duration = tf - t0
    s = (timestamps - t0) / jnp.maximum(duration, 1e-6)
    s = jnp.clip(s, 0.0, 1.0)
    profile = 10 * s**3 - 15 * s**4 + 6 * s**5
    target_pos = start + (target - start) * profile[:, None]
    dt = float(timestamps[1] - timestamps[0])
    target_vel = _velocity_from_positions(target_pos, dt)
    perturb = jnp.zeros_like(target_pos)
    return TaskSpec(TASK_REACH, target_pos, target_vel, perturb)


def hold_task(
    timestamps: Float[Array, " T"],
    hold_pos: Float[Array, " 2"],
    *,
    perturb_time: float | None = None,
    perturb_force: Float[Array, " 2"] | None = None,
) -> TaskSpec:
    """Hold a fixed target with an optional external perturbation.

    Args:
        timestamps: Time points, shape ``(T,)``.
        hold_pos: Position to hold, shape ``(2,)``.
        perturb_time: Time at which to apply perturbation (default: midpoint).
        perturb_force: Perturbation force vector (default: ``[3.0, 0.0]``).

    Returns:
        TaskSpec with static target and impulse perturbation.
    """
    target_pos = jnp.tile(hold_pos[None, :], (timestamps.shape[0], 1))
    target_vel = jnp.zeros_like(target_pos)
    perturb = jnp.zeros_like(target_pos)
    if perturb_time is None:
        perturb_time = float(timestamps[len(timestamps) // 2])
    if perturb_force is None:
        perturb_force = jnp.array([3.0, 0.0])
    idx = int(jnp.argmin(jnp.abs(timestamps - perturb_time)))
    perturb = perturb.at[idx].set(perturb_force)
    return TaskSpec(TASK_HOLD, target_pos, target_vel, perturb)


def _catmull_rom(p0: Array, p1: Array, p2: Array, p3: Array, t: Array) -> Array:
    """Catmull-Rom spline interpolation."""
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        (2 * p1)
        + (-p0 + p2) * t
        + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
        + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
    )


def tracking_task(
    timestamps: Float[Array, " T"],
    key: PRNGKeyArray,
    *,
    radius: float = 0.35,
) -> TaskSpec:
    """Spline-like tracking task using Catmull-Rom interpolation.

    Args:
        timestamps: Time points, shape ``(T,)``.
        key: PRNG key for random control point generation.
        radius: Approximate workspace radius for control points.

    Returns:
        TaskSpec with smooth spline trajectory.
    """
    n_points = 6
    key_angles, key_radii = jax.random.split(key)
    angles = jax.random.uniform(key_angles, (n_points,), minval=0.0, maxval=2 * jnp.pi)
    radii = radius * (0.6 + 0.4 * jax.random.uniform(key_radii, (n_points,)))
    points = jnp.stack([radii * jnp.cos(angles), radii * jnp.sin(angles)], axis=-1)
    points = jnp.concatenate([points[:1], points, points[-1:]], axis=0)

    n_segments = points.shape[0] - 3
    segment_len = timestamps.shape[0] // n_segments

    targets = []
    for seg in range(n_segments):
        start = seg * segment_len
        end = (seg + 1) * segment_len if seg < n_segments - 1 else timestamps.shape[0]
        t = jnp.linspace(0.0, 1.0, end - start)
        targets.append(
            _catmull_rom(
                points[seg],
                points[seg + 1],
                points[seg + 2],
                points[seg + 3],
                t[:, None],
            )
        )
    target_pos = jnp.concatenate(targets, axis=0)
    dt = float(timestamps[1] - timestamps[0])
    target_vel = _velocity_from_positions(target_pos, dt)
    perturb = jnp.zeros_like(target_pos)
    return TaskSpec(TASK_TRACK, target_pos, target_vel, perturb)


def swing_task(
    timestamps: Float[Array, " T"],
    *,
    radius: float = 0.45,
    amplitude: float = 0.6,
    frequency: float = 0.5,
) -> TaskSpec:
    """Swing target along an arc with sinusoidal phase.

    Args:
        timestamps: Time points, shape ``(T,)``.
        radius: Distance from origin to arc center.
        amplitude: Angular amplitude of the swing.
        frequency: Oscillation frequency in Hz.

    Returns:
        TaskSpec with sinusoidal arc trajectory.
    """
    t = timestamps - timestamps[0]
    angle = amplitude * jnp.sin(2 * jnp.pi * frequency * t)
    target_pos = jnp.stack(
        [radius * jnp.cos(angle), radius * jnp.sin(angle)], axis=-1
    )
    dt = float(timestamps[1] - timestamps[0])
    target_vel = _velocity_from_positions(target_pos, dt)
    perturb = jnp.zeros_like(target_pos)
    return TaskSpec(TASK_SWING, target_pos, target_vel, perturb)


def sample_task(
    timestamps: Float[Array, " T"],
    key: PRNGKeyArray,
    *,
    task_type: int | None = None,
    reach_radius: float = 0.5,
) -> TaskSpec:
    """Sample one of the four task types uniformly.

    This version uses Python control flow and is NOT safe inside
    ``jax.jit`` / ``jax.vmap``. Use ``sample_task_jax`` for traced contexts.

    Args:
        timestamps: Time points, shape ``(T,)``.
        key: PRNG key.
        task_type: Force a specific task type (0-3), or None for random.
        reach_radius: Workspace radius for reach/hold tasks.

    Returns:
        Sampled TaskSpec.
    """
    if task_type is None:
        key, subkey = jax.random.split(key)
        task_type = int(jax.random.randint(subkey, (), 0, 4))

    if task_type == TASK_REACH:
        key_start, key_target = jax.random.split(key)
        start = jax.random.uniform(
            key_start, (2,), minval=-reach_radius * 0.5, maxval=reach_radius * 0.5,
        )
        target = jax.random.uniform(
            key_target, (2,), minval=-reach_radius, maxval=reach_radius,
        )
        return reach_task(timestamps, start, target)
    if task_type == TASK_HOLD:
        hold_pos = jax.random.uniform(
            key, (2,), minval=-reach_radius * 0.5, maxval=reach_radius * 0.5,
        )
        return hold_task(timestamps, hold_pos)
    if task_type == TASK_TRACK:
        return tracking_task(timestamps, key)
    return swing_task(timestamps)


def sample_task_jax(
    timestamps: Float[Array, " T"],
    key: PRNGKeyArray,
    reach_radius: float = 0.5,
) -> TaskSpec:
    """Fully JAX-traceable task sampler for vmapped/scanned environments.

    Unlike ``sample_task``, this never calls ``float()`` or ``int()`` on
    traced values, so it is safe inside ``jax.lax.scan`` and ``jax.vmap``.

    All four task types are computed unconditionally; the dynamic task_type
    index selects which one to return.

    Args:
        timestamps: Time points, shape ``(T,)``.
        key: PRNG key.
        reach_radius: Workspace radius for reach/hold tasks.

    Returns:
        Sampled TaskSpec with dynamically selected task type.
    """
    T = timestamps.shape[0]
    dt = timestamps[1] - timestamps[0]

    key, type_key, k1, k2, k3 = jax.random.split(key, 5)
    task_type = jax.random.randint(type_key, (), 0, 4)

    # --- Reach (minimum-jerk) ---
    rk1, rk2 = jax.random.split(k1)
    start = jax.random.uniform(
        rk1, (2,), minval=-reach_radius * 0.5, maxval=reach_radius * 0.5,
    )
    target = jax.random.uniform(
        rk2, (2,), minval=-reach_radius, maxval=reach_radius,
    )
    t0 = timestamps[0]
    tf = timestamps[-1]
    s = jnp.clip((timestamps - t0) / jnp.maximum(tf - t0, 1e-6), 0.0, 1.0)
    profile = 10 * s**3 - 15 * s**4 + 6 * s**5
    reach_pos = start + (target - start) * profile[:, None]
    reach_vel = _vel_from_pos(reach_pos, dt)
    reach_perturb = jnp.zeros((T, 2))

    # --- Hold (static target + perturbation pulse) ---
    hold_center = jax.random.uniform(
        k2, (2,), minval=-reach_radius * 0.5, maxval=reach_radius * 0.5,
    )
    hold_pos = jnp.tile(hold_center[None, :], (T, 1))
    hold_vel = jnp.zeros((T, 2))
    hold_perturb = jnp.zeros((T, 2))
    mid_idx = T // 2
    hold_perturb = hold_perturb.at[mid_idx].set(jnp.array([3.0, 0.0]))

    # --- Track (Catmull-Rom spline) ---
    n_pts = 6
    k3a, k3b = jax.random.split(k3)
    angles = jax.random.uniform(k3a, (n_pts,), minval=0.0, maxval=2 * jnp.pi)
    radii = 0.35 * (0.6 + 0.4 * jax.random.uniform(k3b, (n_pts,)))
    pts = jnp.stack(
        [radii * jnp.cos(angles), radii * jnp.sin(angles)], axis=-1,
    )
    pts = jnp.concatenate([pts[:1], pts, pts[-1:]], axis=0)
    n_seg = pts.shape[0] - 3
    seg_len = T // n_seg

    track_parts = []
    for seg in range(n_seg):
        s_start = seg * seg_len
        s_end = (seg + 1) * seg_len if seg < n_seg - 1 else T
        t_param = jnp.linspace(0.0, 1.0, s_end - s_start)
        p0, p1, p2, p3 = pts[seg], pts[seg + 1], pts[seg + 2], pts[seg + 3]
        t2 = t_param * t_param
        t3 = t2 * t_param
        interp = 0.5 * (
            (2 * p1)
            + (-p0 + p2) * t_param[:, None]
            + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2[:, None]
            + (-p0 + 3 * p1 - 3 * p2 + p3) * t3[:, None]
        )
        track_parts.append(interp)
    track_pos = jnp.concatenate(track_parts, axis=0)
    track_vel = _vel_from_pos(track_pos, dt)
    track_perturb = jnp.zeros((T, 2))

    # --- Swing (sinusoidal arc) ---
    t_sw = timestamps - timestamps[0]
    angle_sw = 0.6 * jnp.sin(2 * jnp.pi * 0.5 * t_sw)
    swing_pos = jnp.stack(
        [0.45 * jnp.cos(angle_sw), 0.45 * jnp.sin(angle_sw)], axis=-1,
    )
    swing_vel = _vel_from_pos(swing_pos, dt)
    swing_perturb = jnp.zeros((T, 2))

    # Stack all four and select with dynamic index
    all_pos = jnp.stack([reach_pos, hold_pos, track_pos, swing_pos])
    all_vel = jnp.stack([reach_vel, hold_vel, track_vel, swing_vel])
    all_pt = jnp.stack([reach_perturb, hold_perturb, track_perturb, swing_perturb])

    return TaskSpec(
        task_type=task_type,
        target_pos=all_pos[task_type],
        target_vel=all_vel[task_type],
        perturbation=all_pt[task_type],
    )
