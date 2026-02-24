"""Body parameterization (φ) for configurable N-DOF planar chains.

Provides BodyPreset for specifying physical parameters of articulated bodies,
with support for random sampling and flat vector conversion for neural conditioning.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


class BodyPreset(eqx.Module):
    """Complete body parameterization φ for an N-link planar chain with muscles.

    All arrays are 1D with shape ``(n_joints,)`` or ``(n_muscles,)``.

    Attributes:
        segment_lengths: Length of each segment in meters.
        segment_masses: Mass of each segment in kg.
        joint_damping: Viscous damping at each joint in N·m·s/rad.
        joint_stiffness: Passive stiffness at each joint in N·m/rad.
        muscle_pcsa: Physiological cross-sectional area of each muscle in cm².
        muscle_optimal_fiber_length: Optimal fiber length of each muscle in meters.
        muscle_tendon_slack_length: Tendon slack length of each muscle in meters.
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


def default_3link_bounds() -> BodyPresetBounds:
    """Sensible parameter ranges for a 3-link planar arm with 6 muscles."""
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
    )


def to_flat(preset: BodyPreset) -> Float[Array, " n_phi"]:
    """Concatenate all BodyPreset fields into a flat vector for neural conditioning.

    Args:
        preset: The body preset to flatten.

    Returns:
        A 1D array of shape ``(n_phi,)`` where ``n_phi = 4*n_joints + 3*n_muscles + 2``.
    """
    return jnp.concatenate([
        preset.segment_lengths,
        preset.segment_masses,
        preset.joint_damping,
        preset.joint_stiffness,
        preset.muscle_pcsa,
        preset.muscle_optimal_fiber_length,
        preset.muscle_tendon_slack_length,
        jnp.array([preset.tau_act, preset.tau_deact]),
    ])


def from_flat(
    arr: Float[Array, " n_phi"], n_joints: int = 3, n_muscles: int = 6
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
        tau_act=float(take(1)[0]),
        tau_deact=float(take(1)[0]),
    )


def flat_dim(n_joints: int = 3, n_muscles: int = 6) -> int:
    """Total number of parameters in the flat φ vector.

    Args:
        n_joints: Number of joints.
        n_muscles: Number of muscles.

    Returns:
        Dimension of the flat vector (e.g. 32 for 3-link/6-muscle).
    """
    return 4 * n_joints + 3 * n_muscles + 2


def sample_preset(bounds: BodyPresetBounds, key: PRNGKeyArray) -> BodyPreset:
    """Sample a random BodyPreset uniformly within bounds.

    Args:
        bounds: Sampling ranges for each parameter.
        key: PRNG key.

    Returns:
        A randomly sampled BodyPreset.
    """
    keys = jax.random.split(key, 7)

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
    )
