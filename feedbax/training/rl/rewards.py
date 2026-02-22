"""Reward functions for reinforcement learning training.

Provides effector-space reward computation with task-type-dependent
bonuses, fully compatible with JAX JIT compilation.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from feedbax.training.rl.tasks import TASK_HOLD, TASK_REACH, TASK_SWING, TASK_TRACK


class RewardConfig(eqx.Module):
    """Reward function weights.

    Attributes:
        effort_weight: Penalty weight for muscle effort (L2 excitation).
        velocity_weight: Penalty weight for velocity tracking error.
        hold_bonus: Bonus reward when holding within threshold.
        hold_threshold: Distance threshold for hold bonus in meters.
    """

    effort_weight: float = 0.005
    velocity_weight: float = 0.1
    hold_bonus: float = 1.0
    hold_threshold: float = 0.02


def compute_reward(
    *,
    task_type: Float[Array, ""],
    effector_pos: Float[Array, " 2"],
    target_pos: Float[Array, " 2"],
    effector_vel: Float[Array, " 2"],
    target_vel: Float[Array, " 2"],
    muscle_excitations: Float[Array, " n_muscles"],
    effort_weight: float,
    velocity_weight: float,
    hold_bonus: float,
    hold_threshold: float,
) -> Float[Array, ""]:
    """JIT-compatible reward using ``jnp.where`` for task-type branching.

    Computes a scalar reward based on distance to target, effort cost,
    velocity tracking, and task-specific bonuses.

    Args:
        task_type: Integer task type as a scalar array.
        effector_pos: Current effector position, shape ``(2,)``.
        target_pos: Target effector position, shape ``(2,)``.
        effector_vel: Current effector velocity, shape ``(2,)``.
        target_vel: Target effector velocity, shape ``(2,)``.
        muscle_excitations: Current muscle excitations, shape ``(n_muscles,)``.
        effort_weight: Effort penalty weight.
        velocity_weight: Velocity error penalty weight.
        hold_bonus: Bonus for holding within threshold.
        hold_threshold: Distance threshold for hold bonus.

    Returns:
        Scalar reward value.
    """
    distance = jnp.linalg.norm(effector_pos - target_pos)
    effort = jnp.mean(muscle_excitations**2)

    vel_error = jnp.where(
        task_type == TASK_TRACK,
        jnp.linalg.norm(effector_vel - target_vel),
        jnp.linalg.norm(effector_vel),
    )

    reward = -distance - effort_weight * effort - velocity_weight * vel_error

    reward += jnp.where(
        (task_type == TASK_HOLD) & (distance < hold_threshold),
        hold_bonus,
        0.0,
    )
    reward += jnp.where(
        task_type == TASK_SWING,
        0.1 * jnp.linalg.norm(effector_vel),
        0.0,
    )
    reward += jnp.where(
        task_type == TASK_REACH,
        -0.05 * distance,
        0.0,
    )

    return reward
