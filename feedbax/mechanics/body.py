"""Body parameterization (phi) for configurable N-DOF planar chains.

Provides BodyPreset for specifying physical parameters of articulated bodies,
with support for random sampling and flat vector conversion for neural conditioning.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


class BodyPreset(eqx.Module):
    """Complete body parameterization phi for an N-link planar chain with muscles.

    Arrays have shape ``(n_joints,)``, ``(n_muscles,)``, or
    ``(n_muscles, n_joints)`` for moment arm magnitudes.

    Attributes:
        segment_lengths: Length of each segment in meters.
        segment_masses: Mass of each segment in kg.
        joint_damping: Viscous damping at each joint in N*m*s/rad.
        joint_stiffness: Passive stiffness at each joint in N*m/rad.
        muscle_pcsa: Physiological cross-sectional area of each muscle in cm^2.
        muscle_optimal_fiber_length: Optimal fiber length of each muscle in meters.
        muscle_tendon_slack_length: Tendon slack length of each muscle in meters.
        muscle_moment_arm_magnitudes: Unsigned moment arm magnitudes per
            muscle-joint pair, shape ``(n_muscles, n_joints)``.  Combined
            with ``MuscleTopology.sign`` to produce the signed moment arm matrix.
        tau_act: Activation time constant in seconds.
        tau_deact: Deactivation time constant in seconds.
    """

    segment_lengths: Float[Array, " n_joints"]
    segment_masses: Float[Array, " n_joints"]
    joint_damping: Float[Array, " n_joints"]
    joint_stiffness: Float[Array, " n_joints"]
    muscle_pcsa: Float[Array, " n_muscles"]
    muscle_optimal_fiber_length: Float[Array, " n_muscles"]
    muscle_tendon_slack_length: Float[Array, " n_muscles"]
    muscle_moment_arm_magnitudes: Float[Array, "n_muscles n_joints"]
    tau_act: float = 0.01
    tau_deact: float = 0.04


class BodyPresetBounds(eqx.Module):
    """Min/max bounds for sampling BodyPreset parameters.

    Each field pair (e.g. ``segment_lengths_min``, ``segment_lengths_max``)
    defines the uniform sampling range for the corresponding BodyPreset field.
    """

    segment_lengths_min: Float[Array, " n_joints"]
    segment_lengths_max: Float[Array, " n_joints"]
    segment_masses_min: Float[Array, " n_joints"]
    segment_masses_max: Float[Array, " n_joints"]
    joint_damping_min: Float[Array, " n_joints"]
    joint_damping_max: Float[Array, " n_joints"]
    joint_stiffness_min: Float[Array, " n_joints"]
    joint_stiffness_max: Float[Array, " n_joints"]
    muscle_pcsa_min: Float[Array, " n_muscles"]
    muscle_pcsa_max: Float[Array, " n_muscles"]
    muscle_optimal_fiber_length_min: Float[Array, " n_muscles"]
    muscle_optimal_fiber_length_max: Float[Array, " n_muscles"]
    muscle_tendon_slack_length_min: Float[Array, " n_muscles"]
    muscle_tendon_slack_length_max: Float[Array, " n_muscles"]
    muscle_moment_arm_magnitudes_min: Float[Array, "n_muscles n_joints"]
    muscle_moment_arm_magnitudes_max: Float[Array, "n_muscles n_joints"]


def default_3link_bounds() -> BodyPresetBounds:
    """Sensible parameter ranges for a 3-link planar arm with 6 muscles.

    Uses monoarticular topology (2 muscles per joint), so the moment arm
    magnitude matrix is ``(6, 3)`` with non-zero entries only on the
    diagonal blocks.
    """
    n_joints = 3
    n_muscles = 6
    # Each muscle only spans its own joint; magnitude ~0.02m +/-30%.
    mag_min = jnp.zeros((n_muscles, n_joints))
    mag_max = jnp.zeros((n_muscles, n_joints))
    default_arm = 0.02
    for j in range(n_joints):
        for m in range(2):
            idx = j * 2 + m
            mag_min = mag_min.at[idx, j].set(default_arm * 0.7)
            mag_max = mag_max.at[idx, j].set(default_arm * 1.3)
    return BodyPresetBounds(
        segment_lengths_min=jnp.array([0.15, 0.12, 0.08]),
        segment_lengths_max=jnp.array([0.40, 0.35, 0.25]),
        segment_masses_min=jnp.array([1.0, 0.7, 0.3]),
        segment_masses_max=jnp.array([4.0, 3.0, 1.5]),
        joint_damping_min=jnp.full(3, 0.05),
        joint_damping_max=jnp.full(3, 0.5),
        joint_stiffness_min=jnp.full(3, 0.1),
        joint_stiffness_max=jnp.full(3, 2.0),
        muscle_pcsa_min=jnp.full(6, 2.0),
        muscle_pcsa_max=jnp.full(6, 15.0),
        muscle_optimal_fiber_length_min=jnp.full(6, 0.05),
        muscle_optimal_fiber_length_max=jnp.full(6, 0.20),
        muscle_tendon_slack_length_min=jnp.full(6, 0.02),
        muscle_tendon_slack_length_max=jnp.full(6, 0.10),
        muscle_moment_arm_magnitudes_min=mag_min,
        muscle_moment_arm_magnitudes_max=mag_max,
    )


def default_2link_bounds() -> BodyPresetBounds:
    """Sensible parameter ranges for a 2-link arm with 6 muscles.

    Layout: 4 monoarticular + 2 biarticular (from
    ``default_6muscle_2link_topology``).  Moment arm magnitudes are derived
    from ``TwoLinkArmMuscleGeometry.default_six_muscle()`` with ~30% variation.
    """
    # Reference values from geometry.py defaults:
    #   shoulder_moment_arm = 0.04
    #   elbow_moment_arm    = 0.025
    #   biarticular_shoulder_arm = 0.035
    #   biarticular_elbow_arm    = 0.022
    mag_min = jnp.array([
        [0.04 * 0.7, 0.0],         # shoulder flexor
        [0.04 * 0.7, 0.0],         # shoulder extensor
        [0.0, 0.025 * 0.7],        # elbow flexor
        [0.0, 0.025 * 0.7],        # elbow extensor
        [0.035 * 0.7, 0.022 * 0.7],  # biarticular flexor
        [0.035 * 0.7, 0.022 * 0.7],  # biarticular extensor
    ])
    mag_max = jnp.array([
        [0.04 * 1.3, 0.0],         # shoulder flexor
        [0.04 * 1.3, 0.0],         # shoulder extensor
        [0.0, 0.025 * 1.3],        # elbow flexor
        [0.0, 0.025 * 1.3],        # elbow extensor
        [0.035 * 1.3, 0.022 * 1.3],  # biarticular flexor
        [0.035 * 1.3, 0.022 * 1.3],  # biarticular extensor
    ])
    return BodyPresetBounds(
        segment_lengths_min=jnp.array([0.15, 0.12]),
        segment_lengths_max=jnp.array([0.40, 0.35]),
        segment_masses_min=jnp.array([1.0, 0.7]),
        segment_masses_max=jnp.array([4.0, 3.0]),
        joint_damping_min=jnp.full(2, 0.05),
        joint_damping_max=jnp.full(2, 0.5),
        joint_stiffness_min=jnp.full(2, 0.1),
        joint_stiffness_max=jnp.full(2, 2.0),
        muscle_pcsa_min=jnp.full(6, 2.0),
        muscle_pcsa_max=jnp.full(6, 15.0),
        muscle_optimal_fiber_length_min=jnp.full(6, 0.05),
        muscle_optimal_fiber_length_max=jnp.full(6, 0.20),
        muscle_tendon_slack_length_min=jnp.full(6, 0.02),
        muscle_tendon_slack_length_max=jnp.full(6, 0.10),
        muscle_moment_arm_magnitudes_min=mag_min,
        muscle_moment_arm_magnitudes_max=mag_max,
    )


def to_flat(preset: BodyPreset) -> Float[Array, " n_phi"]:
    """Concatenate all BodyPreset fields into a flat vector for neural conditioning.

    Args:
        preset: The body preset to flatten.

    Returns:
        A 1D array of shape ``(n_phi,)``.
    """
    return jnp.concatenate([
        preset.segment_lengths,
        preset.segment_masses,
        preset.joint_damping,
        preset.joint_stiffness,
        preset.muscle_pcsa,
        preset.muscle_optimal_fiber_length,
        preset.muscle_tendon_slack_length,
        preset.muscle_moment_arm_magnitudes.ravel(),
        jnp.array([preset.tau_act, preset.tau_deact]),
    ])


def from_flat(
    arr: Float[Array, " n_phi"],
    n_joints: int = 3,
    n_muscles: int = 6,
) -> BodyPreset:
    """Reconstruct a BodyPreset from a flat vector.

    Args:
        arr: Flat array of shape ``(n_phi,)``.
        n_joints: Number of joints (default 3).
        n_muscles: Number of muscles (default 6).

    Returns:
        Reconstructed BodyPreset.
    """
    idx = 0

    def take(n: int) -> Array:
        nonlocal idx
        result = arr[idx : idx + n]
        idx += n
        return result

    return BodyPreset(
        segment_lengths=take(n_joints),
        segment_masses=take(n_joints),
        joint_damping=take(n_joints),
        joint_stiffness=take(n_joints),
        muscle_pcsa=take(n_muscles),
        muscle_optimal_fiber_length=take(n_muscles),
        muscle_tendon_slack_length=take(n_muscles),
        muscle_moment_arm_magnitudes=take(n_muscles * n_joints).reshape(
            n_muscles, n_joints,
        ),
        tau_act=float(take(1)[0]),
        tau_deact=float(take(1)[0]),
    )


def flat_dim(n_joints: int = 3, n_muscles: int = 6) -> int:
    """Total number of parameters in the flat phi vector.

    Args:
        n_joints: Number of joints.
        n_muscles: Number of muscles.

    Returns:
        Dimension of the flat vector.
    """
    # 4 per-joint fields + 3 per-muscle fields + n_muscles*n_joints magnitudes + 2 tau
    return 4 * n_joints + 3 * n_muscles + n_muscles * n_joints + 2


def sample_preset(bounds: BodyPresetBounds, key: PRNGKeyArray) -> BodyPreset:
    """Sample a random BodyPreset uniformly within bounds.

    Args:
        bounds: Sampling ranges for each parameter.
        key: PRNG key.

    Returns:
        A randomly sampled BodyPreset.
    """
    keys = jax.random.split(key, 8)

    def uniform(k: PRNGKeyArray, lo: Array, hi: Array) -> Array:
        return jax.random.uniform(k, lo.shape, minval=lo, maxval=hi)

    return BodyPreset(
        segment_lengths=uniform(keys[0], bounds.segment_lengths_min, bounds.segment_lengths_max),
        segment_masses=uniform(keys[1], bounds.segment_masses_min, bounds.segment_masses_max),
        joint_damping=uniform(keys[2], bounds.joint_damping_min, bounds.joint_damping_max),
        joint_stiffness=uniform(keys[3], bounds.joint_stiffness_min, bounds.joint_stiffness_max),
        muscle_pcsa=uniform(keys[4], bounds.muscle_pcsa_min, bounds.muscle_pcsa_max),
        muscle_optimal_fiber_length=uniform(
            keys[5],
            bounds.muscle_optimal_fiber_length_min,
            bounds.muscle_optimal_fiber_length_max,
        ),
        muscle_tendon_slack_length=uniform(
            keys[6],
            bounds.muscle_tendon_slack_length_min,
            bounds.muscle_tendon_slack_length_max,
        ),
        muscle_moment_arm_magnitudes=uniform(
            keys[7],
            bounds.muscle_moment_arm_magnitudes_min,
            bounds.muscle_moment_arm_magnitudes_max,
        ),
    )
