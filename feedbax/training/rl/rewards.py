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
    step: Float[Array, ""] | None = None,
    n_steps: int | None = None,
) -> Float[Array, ""]:
    """JIT-compatible reward using ``jnp.where`` for task-type branching.

    Computes a scalar reward based on distance to target, effort cost,
    velocity tracking, and task-specific bonuses.

    The velocity penalty is discounted by episode phase so the agent is
    free to move early in the episode and only penalized for residual
    velocity near the end.  When ``step`` and ``n_steps`` are provided
    the discount is ``(step / (n_steps - 1)) ** 2`` (quadratic ramp
    from 0 to 1).  When omitted the penalty applies at full strength
    every step (backward-compatible).

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
        step: Current timestep index (scalar int32 array). Optional for
            backward compatibility; when ``None`` the velocity discount
            is 1.0 (no discount).
        n_steps: Total timesteps per episode. Required when ``step`` is
            provided.

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

    # Bug: 3014308 -- Quadratic ramp: near-zero penalty early, full penalty
    # at episode end. Prevents the "freeze trap" where the agent learns to
    # never move because velocity is penalized uniformly.
    if step is not None and n_steps is not None:
        phase = step / jnp.maximum(n_steps - 1, 1)
        vel_discount = phase ** 2
    else:
        vel_discount = 1.0

    reward = (
        -distance
        - effort_weight * effort
        - velocity_weight * vel_discount * vel_error
    )

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
