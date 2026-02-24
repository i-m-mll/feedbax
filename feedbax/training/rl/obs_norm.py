"""Observation normalization with Welford's online algorithm.

Provides running mean/variance tracking and normalization for RL
observations. Statistics are maintained as an immutable
``ObsNormState`` PyTree that can be vmapped for per-body normalization.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


class ObsNormState(eqx.Module):
    """Running statistics for observation normalization (Welford's algorithm).

    Attributes:
        count: Number of observations seen so far.
        mean: Running mean per observation dimension.
        var: Running variance per observation dimension.
    """

    count: Float[Array, ""]
    mean: Float[Array, "obs_dim"]
    var: Float[Array, "obs_dim"]


def init_obs_norm(obs_dim: int) -> ObsNormState:
    """Initialize observation normalization state.

    Args:
        obs_dim: Dimensionality of the observation vector.

    Returns:
        Fresh ``ObsNormState`` with count=0, mean=zeros, var=ones.
    """
    return ObsNormState(
        count=jnp.array(0.0),
        mean=jnp.zeros(obs_dim),
        var=jnp.ones(obs_dim),
    )


def update_obs_norm(
    state: ObsNormState,
    batch: Float[Array, "batch obs_dim"],
) -> ObsNormState:
    """Update running statistics with a new batch of observations.

    Uses Welford's parallel/batch merge algorithm to combine the
    existing running statistics with batch-level statistics, giving
    numerically stable incremental mean and variance.

    Args:
        state: Current running statistics.
        batch: New observations, shape ``(batch_size, obs_dim)``.

    Returns:
        Updated ``ObsNormState`` with merged statistics.
    """
    batch_count = jnp.array(batch.shape[0], dtype=jnp.float32)
    batch_mean = jnp.mean(batch, axis=0)
    batch_var = jnp.var(batch, axis=0)

    # Welford's parallel merge formula
    new_count = state.count + batch_count
    delta = batch_mean - state.mean
    new_mean = state.mean + delta * batch_count / jnp.maximum(new_count, 1.0)
    m_a = state.var * state.count
    m_b = batch_var * batch_count
    m2 = m_a + m_b + delta**2 * state.count * batch_count / jnp.maximum(new_count, 1.0)
    new_var = m2 / jnp.maximum(new_count, 1.0)

    return ObsNormState(count=new_count, mean=new_mean, var=new_var)


def normalize_obs(
    state: ObsNormState,
    obs: Float[Array, "... obs_dim"],
    eps: float = 1e-8,
    clip: float = 10.0,
) -> Array:
    """Normalize observations using running statistics.

    Subtracts the running mean and divides by the running standard
    deviation, then clips to ``[-clip, clip]``. Broadcasting handles
    arbitrary leading batch dimensions.

    Args:
        state: Running statistics from ``update_obs_norm``.
        obs: Observations to normalize, with ``obs_dim`` as the last axis.
        eps: Small constant for numerical stability in the denominator.
        clip: Symmetric clipping bound for normalized values.

    Returns:
        Normalized (and clipped) observations, same shape as ``obs``.
    """
    normalized = (obs - state.mean) / jnp.sqrt(state.var + eps)
    return jnp.clip(normalized, -clip, clip)
