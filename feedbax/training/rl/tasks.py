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

    Deprecated: use ``TaskParams`` with ``target_at_t`` for lazy evaluation.

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


@jax.tree_util.register_pytree_node_class
class TaskParams(NamedTuple):
    """Compact task parameters for lazy target generation.

    Attributes:
        task_type: Integer task type (0=reach, 1=hold, 2=track, 3=swing).
        start_pos: Starting position, shape ``(2,)``.
        end_pos: Target endpoint, shape ``(2,)``.
        perturb_time_idx: Timestep index for perturbation onset.
        perturb_force: Perturbation force vector, shape ``(2,)``.
        control_points: Catmull-Rom control points, shape ``(6, 2)``.
        t0: Start time.
        tf: End time.
        dt: Timestep.
        n_steps: Total number of timesteps.
    """

    task_type: int
    start_pos: Float[Array, " 2"]
    end_pos: Float[Array, " 2"]
    perturb_time_idx: int
    perturb_force: Float[Array, " 2"]
    control_points: Float[Array, " 6 2"]
    t0: Float[Array, ""]
    tf: Float[Array, ""]
    dt: Float[Array, ""]
    n_steps: int

    def tree_flatten(self):
        children = (
            self.task_type,
            self.start_pos,
            self.end_pos,
            self.perturb_time_idx,
            self.perturb_force,
            self.control_points,
            self.t0,
            self.tf,
            self.dt,
        )
        aux_data = {"n_steps": self.n_steps}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (
            task_type,
            start_pos,
            end_pos,
            perturb_time_idx,
            perturb_force,
            control_points,
            t0,
            tf,
            dt,
        ) = children
        return cls(
            task_type=task_type,
            start_pos=start_pos,
            end_pos=end_pos,
            perturb_time_idx=perturb_time_idx,
            perturb_force=perturb_force,
            control_points=control_points,
            t0=t0,
            tf=tf,
            dt=dt,
            n_steps=aux_data["n_steps"],
        )


def _fk_planar(
    joint_angles: Float[Array, " n_joints"],
    segment_lengths: Float[Array, " n_joints"],
) -> Float[Array, " 2"]:
    """Forward kinematics for a planar serial arm.

    Computes the end-effector position from joint angles and segment lengths.
    Works with any number of joints (not hardcoded to a specific topology).

    Args:
        joint_angles: Angle of each joint, shape ``(n_joints,)``.
        segment_lengths: Length of each segment, shape ``(n_joints,)``.

    Returns:
        End-effector position in Cartesian space, shape ``(2,)``.
    """
    cumulative_angles = jnp.cumsum(joint_angles)
    x = jnp.sum(segment_lengths * jnp.cos(cumulative_angles))
    y = jnp.sum(segment_lengths * jnp.sin(cumulative_angles))
    return jnp.array([x, y])


def _sample_reachable_pos(
    key: PRNGKeyArray,
    segment_lengths: Float[Array, " n_joints"],
) -> Float[Array, " 2"]:
    """Sample a reachable effector position via random joint angles + FK.

    Joint angles are sampled in ``[0.1*pi, 0.9*pi]`` to avoid workspace
    boundary extremes (fully extended or fully folded configurations).
    The resulting position is reachable by construction since it comes
    from valid joint angles through forward kinematics.

    Args:
        key: PRNG key.
        segment_lengths: Segment lengths, shape ``(n_joints,)``.

    Returns:
        Reachable Cartesian position, shape ``(2,)``.
    """
    n_joints = segment_lengths.shape[0]
    angles = jax.random.uniform(
        key, shape=(n_joints,), minval=0.1 * jnp.pi, maxval=0.9 * jnp.pi,
    )
    return _fk_planar(angles, segment_lengths)


def _finite_diff_velocity(
    positions: Float[Array, "T 2"], dt: Float[Array, ""],
) -> Float[Array, "T 2"]:
    """Finite-difference velocity with boundary clamping."""
    n_steps = positions.shape[0]
    idx = jnp.arange(n_steps)
    idx_minus = jnp.clip(idx - 1, 0, n_steps - 1)
    idx_plus = jnp.clip(idx + 1, 0, n_steps - 1)
    dt_safe = jnp.maximum(dt, 1e-6)
    return (positions[idx_plus] - positions[idx_minus]) / (2.0 * dt_safe)


def _velocity_from_positions(
    positions: Float[Array, "T 2"], dt: float
) -> Float[Array, "T 2"]:
    """Finite-difference velocity from position trajectory."""
    return _finite_diff_velocity(positions, jnp.asarray(dt))


def _vel_from_pos(
    positions: Float[Array, "T 2"], dt: Float[Array, ""],
) -> Float[Array, "T 2"]:
    """Finite-difference velocity. dt may be a JAX tracer (safe in JIT)."""
    return _finite_diff_velocity(positions, dt)


def _reach_pos_at_t(params: TaskParams, t_index: Array) -> Float[Array, " 2"]:
    t0 = jnp.asarray(params.t0)
    tf = jnp.asarray(params.tf)
    dt = jnp.asarray(params.dt)
    t = t0 + dt * t_index
    duration = jnp.maximum(tf - t0, 1e-6)
    s = jnp.clip((t - t0) / duration, 0.0, 1.0)
    profile = 10 * s**3 - 15 * s**4 + 6 * s**5
    return params.start_pos + (params.end_pos - params.start_pos) * profile


def _hold_pos_at_t(params: TaskParams, _: Array) -> Float[Array, " 2"]:
    return params.end_pos


def _track_pos_at_t(params: TaskParams, t_index: Array) -> Float[Array, " 2"]:
    points = jnp.concatenate(
        [params.control_points[:1], params.control_points, params.control_points[-1:]],
        axis=0,
    )
    n_seg = points.shape[0] - 3
    n_steps = jnp.asarray(params.n_steps)
    segment_len = jnp.maximum(n_steps // n_seg, 1)
    seg = jnp.minimum(t_index // segment_len, n_seg - 1)
    seg_start = seg * segment_len
    seg_end = jnp.where(seg < n_seg - 1, (seg + 1) * segment_len, n_steps)
    seg_len = jnp.maximum(seg_end - seg_start, 1)
    local_idx = t_index - seg_start
    t = local_idx / jnp.maximum(seg_len - 1, 1)
    return _catmull_rom(
        points[seg],
        points[seg + 1],
        points[seg + 2],
        points[seg + 3],
        t,
    )


def _swing_pos_at_t(params: TaskParams, t_index: Array) -> Float[Array, " 2"]:
    t0 = jnp.asarray(params.t0)
    dt = jnp.asarray(params.dt)
    t = t0 + dt * t_index
    # Derive swing radius from start_pos (set to [radius, 0] at sampling time)
    # so that FK-aware sampling propagates through to trajectory reconstruction.
    radius = jnp.sqrt(params.start_pos[0] ** 2 + params.start_pos[1] ** 2)
    radius = jnp.maximum(radius, 1e-6)  # Avoid degenerate zero radius
    angle = 0.6 * jnp.sin(2 * jnp.pi * 0.5 * t)
    return jnp.stack([radius * jnp.cos(angle), radius * jnp.sin(angle)], axis=-1)


def _target_pos_at_t(params: TaskParams, t_index: Array) -> Float[Array, " 2"]:
    def reach_fn(args):
        p, t = args
        return _reach_pos_at_t(p, t)

    def hold_fn(args):
        p, t = args
        return _hold_pos_at_t(p, t)

    def track_fn(args):
        p, t = args
        return _track_pos_at_t(p, t)

    def swing_fn(args):
        p, t = args
        return _swing_pos_at_t(p, t)

    task_type = jnp.asarray(params.task_type)
    return jax.lax.switch(
        task_type,
        (reach_fn, hold_fn, track_fn, swing_fn),
        (params, t_index),
    )


def target_at_t(
    params: TaskParams, t_index: Array
) -> tuple[Float[Array, " 2"], Float[Array, " 2"]]:
    """Return target position/velocity at a given timestep."""
    t_index = jnp.asarray(t_index)
    n_steps = jnp.asarray(params.n_steps)
    t_minus = jnp.clip(t_index - 1, 0, n_steps - 1)
    t_plus = jnp.clip(t_index + 1, 0, n_steps - 1)
    pos = _target_pos_at_t(params, t_index)
    pos_minus = _target_pos_at_t(params, t_minus)
    pos_plus = _target_pos_at_t(params, t_plus)
    dt_safe = jnp.maximum(jnp.asarray(params.dt), 1e-6)
    vel = (pos_plus - pos_minus) / (2.0 * dt_safe)
    return pos, vel


def sample_task_params_jax(
    key: PRNGKeyArray,
    task_type: int | Array | None,
    n_steps: int,
    dt: float,
    *,
    segment_lengths: Float[Array, " n_joints"] | None = None,
    reach_radius: float = 0.5,
    track_radius: float = 0.35,
) -> TaskParams:
    """Sample compact task parameters in a fully JAX-traceable way.

    When ``segment_lengths`` is provided, targets are sampled in joint space
    and mapped to Cartesian space via forward kinematics. This guarantees all
    targets are physically reachable by the arm. When ``segment_lengths`` is
    ``None``, falls back to uniform Cartesian sampling within ``reach_radius``
    (legacy behavior, no reachability guarantee).

    Args:
        key: PRNG key.
        task_type: Force a specific task type (0-3), or ``None`` to sample
            uniformly. May be a traced integer (safe inside vmap).
        n_steps: Number of timesteps per episode.
        dt: Physics timestep in seconds.
        segment_lengths: Segment lengths for FK-based reachable sampling.
            Shape ``(n_joints,)``. If ``None``, uses legacy Cartesian sampling.
        reach_radius: Workspace radius for legacy Cartesian sampling.
        track_radius: Workspace radius for legacy tracking control points.

    Returns:
        Sampled TaskParams.
    """
    use_fk = segment_lengths is not None

    key, type_key, k1, k2, k3 = jax.random.split(key, 5)
    if task_type is None:
        task_type = jax.random.randint(type_key, (), 0, 4)

    t0 = jnp.asarray(0.0)
    tf = jnp.asarray(dt) * (n_steps - 1)
    dt_arr = jnp.asarray(dt)

    # --- Reach ---
    rk1, rk2 = jax.random.split(k1)
    if use_fk:
        reach_start = _sample_reachable_pos(rk1, segment_lengths)
        reach_end = _sample_reachable_pos(rk2, segment_lengths)
    else:
        reach_start = jax.random.uniform(
            rk1, (2,), minval=-reach_radius * 0.5, maxval=reach_radius * 0.5,
        )
        reach_end = jax.random.uniform(
            rk2, (2,), minval=-reach_radius, maxval=reach_radius,
        )
    reach_cp = jnp.zeros((6, 2))
    reach_perturb_idx = jnp.array(0, dtype=jnp.int32)
    reach_perturb = jnp.zeros((2,))

    # --- Hold ---
    # Bug: 67e2e5e -- Randomize perturbation direction, magnitude, and timing
    key, hk1, hk2, hk3 = jax.random.split(key, 4)
    if use_fk:
        hold_pos = _sample_reachable_pos(k2, segment_lengths)
    else:
        hold_pos = jax.random.uniform(
            k2, (2,), minval=-reach_radius * 0.5, maxval=reach_radius * 0.5,
        )
    hold_cp = jnp.zeros((6, 2))
    # Random perturbation direction (uniform angle)
    perturb_angle = jax.random.uniform(hk1, shape=(), minval=0.0, maxval=2 * jnp.pi)
    # Random magnitude (normal, clipped to avoid degenerate cases)
    perturb_mag = jnp.clip(
        jax.random.normal(hk2, shape=()) * 1.0 + 3.0, 0.5, 6.0,
    )
    hold_perturb = perturb_mag * jnp.array(
        [jnp.cos(perturb_angle), jnp.sin(perturb_angle)],
    )
    # Jittered timing (+-10% of episode around midpoint)
    hold_perturb_idx = jax.random.randint(
        hk3, shape=(), minval=int(0.4 * n_steps), maxval=int(0.6 * n_steps) + 1,
    ).astype(jnp.int32)

    # --- Track ---
    n_pts = 6
    k3a, k3b = jax.random.split(k3)
    if use_fk:
        # Sample each control point as a reachable position via FK
        cp_keys = jax.random.split(k3a, n_pts)
        track_cp = jax.vmap(_sample_reachable_pos, in_axes=(0, None))(
            cp_keys, segment_lengths,
        )
    else:
        angles = jax.random.uniform(k3a, (n_pts,), minval=0.0, maxval=2 * jnp.pi)
        radii = track_radius * (0.6 + 0.4 * jax.random.uniform(k3b, (n_pts,)))
        track_cp = jnp.stack(
            [radii * jnp.cos(angles), radii * jnp.sin(angles)], axis=-1,
        )
    track_perturb_idx = jnp.array(0, dtype=jnp.int32)
    track_perturb = jnp.zeros((2,))

    # --- Swing ---
    # When FK is available, derive swing radius from arm reach instead of
    # using a hardcoded 0.45. Use ~60% of max reach for the oscillation
    # center, keeping the full swing arc within the reachable workspace.
    if use_fk:
        swing_radius = 0.6 * jnp.sum(segment_lengths)
    else:
        swing_radius = jnp.asarray(0.45)
    swing_start = jnp.array([swing_radius, 0.0])
    swing_t = tf
    swing_angle = 0.6 * jnp.sin(2 * jnp.pi * 0.5 * swing_t)
    swing_end = jnp.stack(
        [swing_radius * jnp.cos(swing_angle), swing_radius * jnp.sin(swing_angle)],
        axis=-1,
    )
    swing_cp = jnp.zeros((6, 2))
    swing_perturb_idx = jnp.array(0, dtype=jnp.int32)
    swing_perturb = jnp.zeros((2,))

    all_start = jnp.stack([reach_start, hold_pos, track_cp[0], swing_start])
    all_end = jnp.stack([reach_end, hold_pos, track_cp[-1], swing_end])
    all_cp = jnp.stack([reach_cp, hold_cp, track_cp, swing_cp])
    all_perturb_idx = jnp.stack(
        [reach_perturb_idx, hold_perturb_idx, track_perturb_idx, swing_perturb_idx]
    )
    all_perturb = jnp.stack([reach_perturb, hold_perturb, track_perturb, swing_perturb])

    return TaskParams(
        task_type=task_type,
        start_pos=all_start[task_type],
        end_pos=all_end[task_type],
        perturb_time_idx=all_perturb_idx[task_type],
        perturb_force=all_perturb[task_type],
        control_points=all_cp[task_type],
        t0=t0,
        tf=tf,
        dt=dt_arr,
        n_steps=n_steps,
    )


def reconstruct_trajectory(
    params: TaskParams,
) -> tuple[Float[Array, "T 2"], Float[Array, "T 2"]]:
    """Reconstruct full target position/velocity arrays from params."""
    n_steps = int(params.n_steps)
    t_idx = jnp.arange(n_steps)
    return jax.vmap(lambda t: target_at_t(params, t))(t_idx)


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
    task_type: int | Array | None = None,
) -> TaskSpec:
    """Fully JAX-traceable task sampler for vmapped/scanned environments.

    Deprecated: use ``sample_task_params_jax`` with ``target_at_t`` instead.

    Unlike ``sample_task``, this never calls ``float()`` or ``int()`` on
    traced values, so it is safe inside ``jax.lax.scan`` and ``jax.vmap``.

    All four task types are computed unconditionally; the dynamic task_type
    index selects which one to return.

    Args:
        timestamps: Time points, shape ``(T,)``.
        key: PRNG key.
        reach_radius: Workspace radius for reach/hold tasks.
        task_type: Force a specific task type (0-3), or None to sample
            uniformly. May be a traced integer (safe inside vmap).

    Returns:
        Sampled TaskSpec with dynamically selected task type.
    """
    T = timestamps.shape[0]
    dt = timestamps[1] - timestamps[0]

    key, type_key, k1, k2, k3 = jax.random.split(key, 5)
    if task_type is None:
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
