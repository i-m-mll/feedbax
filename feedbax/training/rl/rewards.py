"""Reward functions for reinforcement learning training.

Provides unified effector-space reward computation with proximity-gated
velocity matching, fully compatible with JAX JIT compilation.

Design (Bug: 67e2e5e):
    The velocity penalty uses ``||v_effector - v_target||`` for ALL task
    types.  For REACH/HOLD ``v_target = 0``, so this reduces to
    ``||v_effector||``.  For TRACK/SWING ``v_target`` is the target
    velocity, so the agent matches the desired speed profile.

    The penalty is gated by two signals:
    - **Proximity gate** (Gaussian): penalty grows as the effector
      approaches the target.  Width = ``2 * hold_threshold``.
    - **Temporal gate** (``(t/T)^4``): near-zero until ~60 % of the
      episode, preventing the freeze-trap where early velocity penalty
      discourages any movement.

    A floor of ``0.3 * prox_gate`` ensures that if the agent reaches
    the target early (before the temporal gate opens) it still receives
    velocity shaping.

    Task-specific SWING (+0.1*||v||) and REACH (-0.05*dist) bonuses
    from the previous version are removed — they double-counted or
    rewarded arbitrary velocity.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


class RewardConfig(eqx.Module):
    """Reward function weights.

    Attributes:
        effort_weight: Penalty weight for muscle effort (L2 excitation).
        velocity_weight: Penalty weight for velocity tracking error.
        hold_bonus: Bonus reward when within ``hold_threshold`` of target.
        hold_threshold: Distance threshold for proximity bonus in meters.
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
    """Unified proximity-gated velocity-matching reward.

    Computes a scalar reward from three terms:
    ``-distance - effort_weight * effort - velocity_weight * vel_gate * vel_error``
    plus a proximity bonus when the effector is within ``hold_threshold``.

    The velocity error is ``||v_effector - v_target||`` regardless of
    task type, and is gated by proximity (Gaussian) and episode phase
    (quartic ramp).

    Args:
        task_type: Integer task type as a scalar array (unused in the
            current formulation but kept for API compatibility).
        effector_pos: Current effector position, shape ``(2,)``.
        target_pos: Target effector position, shape ``(2,)``.
        effector_vel: Current effector velocity, shape ``(2,)``.
        target_vel: Target effector velocity, shape ``(2,)``.
        muscle_excitations: Current muscle excitations, shape ``(n_muscles,)``.
        effort_weight: Effort penalty weight.
        velocity_weight: Velocity error penalty weight.
        hold_bonus: Bonus for being within ``hold_threshold`` of target.
        hold_threshold: Distance threshold for proximity bonus.
        step: Current timestep index (scalar int32 array).  Optional for
            backward compatibility; when ``None`` the temporal gate is 1.0.
        n_steps: Total timesteps per episode.  Required when ``step`` is
            provided.

    Returns:
        Scalar reward value.
    """
    distance = jnp.linalg.norm(effector_pos - target_pos)
    effort = jnp.mean(muscle_excitations**2)

    # Bug: 67e2e5e — Unified velocity matching: ||v_eff - v_target|| for ALL
    # task types.  REACH/HOLD have v_target=0, TRACK/SWING match target speed.
    vel_error = jnp.linalg.norm(effector_vel - target_vel)

    # Proximity gate: Gaussian centered on target, width = 2 * hold_threshold.
    prox_gate = jnp.exp(-(distance / (2 * hold_threshold)) ** 2)

    # Temporal gate: near-zero until ~60% of episode (quartic ramp).
    if step is not None and n_steps is not None:
        phase = step / jnp.maximum(n_steps - 1, 1)
        time_gate = phase ** 4
    else:
        time_gate = 1.0

    # Combined gate: active near target AND/OR later in episode.
    # Floor of 0.3 * prox_gate ensures velocity shaping if target reached early.
    vel_gate = jnp.maximum(time_gate * prox_gate, 0.3 * prox_gate)

    # Base reward: distance + effort + proximity-gated velocity matching.
    reward = (
        -distance
        - effort_weight * effort
        - velocity_weight * vel_gate * vel_error
    )

    # Proximity bonus for any task within hold_threshold (Klar et al.).
    reward += jnp.where(
        distance < hold_threshold,
        hold_bonus,
        0.0,
    )

    return reward
