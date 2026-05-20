"""Observation normalization with Welford's online algorithm.

Provides running mean/variance tracking and normalization for RL observations.
Statistics are maintained as immutable PyTrees that can be vmapped for per-body
normalization.
"""

from __future__ import annotations

from typing import Literal, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float


ObsNormMode = Literal["standardize", "passthrough", "scale", "angle", "phase"]

_MODE_STANDARDIZE = 0
_MODE_PASSTHROUGH = 1
_MODE_SCALE = 2
_MODE_ANGLE = 3
_MODE_PHASE = 4
_MODE_TO_ID: dict[str, int] = {
    "standardize": _MODE_STANDARDIZE,
    "passthrough": _MODE_PASSTHROUGH,
    "scale": _MODE_SCALE,
    "angle": _MODE_ANGLE,
    "phase": _MODE_PHASE,
}


class RunningNormState(eqx.Module):
    """Running statistics for normalization via Welford parallel merge.

    Attributes:
        count: Number of observations seen so far.
        mean: Running mean per observation dimension.
        var: Running population variance per observation dimension.
    """

    count: Float[Array, ""]
    mean: Float[Array, "obs_dim"]  # noqa: F821
    var: Float[Array, "obs_dim"]  # noqa: F821


ObsNormState = RunningNormState


class ObsNormConfig(eqx.Module):
    """Per-dimension observation normalization configuration.

    ``mode`` uses the integer IDs produced by :func:`init_obs_norm_config`.
    ``scale_min`` and ``scale_max`` are only used for ``"scale"`` dimensions.
    """

    mode: Array
    scale_min: Float[Array, "obs_dim"]  # noqa: F821
    scale_max: Float[Array, "obs_dim"]  # noqa: F821


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


def init_obs_norm_config(
    modes: ObsNormMode | Sequence[ObsNormMode],
    *,
    obs_dim: int | None = None,
    scale_min: float | Sequence[float] | Array = 0.0,
    scale_max: float | Sequence[float] | Array = 1.0,
) -> ObsNormConfig:
    """Initialize per-dimension normalization configuration.

    Args:
        modes: Either one mode applied to every dimension or a mode per
            observation dimension.
        obs_dim: Required when ``modes`` is a single mode string.
        scale_min: Fixed lower bound for ``"scale"`` dimensions.
        scale_max: Fixed upper bound for ``"scale"`` dimensions.

    Returns:
        ``ObsNormConfig`` with array fields suitable for JIT/vmap use.
    """
    if isinstance(modes, str):
        if obs_dim is None:
            raise ValueError("obs_dim is required when modes is a single string")
        mode_ids = jnp.full((obs_dim,), _MODE_TO_ID[modes], dtype=jnp.int32)
    else:
        mode_ids = jnp.asarray([_MODE_TO_ID[mode] for mode in modes], dtype=jnp.int32)
        if obs_dim is not None and len(mode_ids) != obs_dim:
            raise ValueError("modes length must match obs_dim")
        obs_dim = len(mode_ids)

    scale_min_arr = jnp.broadcast_to(jnp.asarray(scale_min, dtype=jnp.float32), (obs_dim,))
    scale_max_arr = jnp.broadcast_to(jnp.asarray(scale_max, dtype=jnp.float32), (obs_dim,))
    return ObsNormConfig(mode=mode_ids, scale_min=scale_min_arr, scale_max=scale_max_arr)


def _batch_to_state(batch: Float[Array, "batch obs_dim"]) -> RunningNormState:
    batch_count = jnp.asarray(batch.shape[0], dtype=batch.dtype)
    return RunningNormState(
        count=batch_count,
        mean=jnp.mean(batch, axis=0),
        var=jnp.var(batch, axis=0),
    )


def merge_obs_norm(
    state: RunningNormState,
    other: RunningNormState,
) -> RunningNormState:
    """Merge two running-stat states with Welford's parallel formula."""
    new_count = state.count + other.count
    safe_count = jnp.maximum(new_count, 1.0)
    delta = other.mean - state.mean
    new_mean = state.mean + delta * other.count / safe_count
    m_a = state.var * state.count
    m_b = other.var * other.count
    m2 = m_a + m_b + delta**2 * state.count * other.count / safe_count
    new_var = m2 / safe_count
    return RunningNormState(count=new_count, mean=new_mean, var=new_var)


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
    if batch.shape[0] == 0:
        return state
    return merge_obs_norm(state, _batch_to_state(batch))


def maybe_update_obs_norm(
    state: ObsNormState,
    batch: Float[Array, "batch obs_dim"],
    training: Bool[Array, ""] | bool,
) -> ObsNormState:
    """Update running statistics only when ``training`` is true.

    This keeps eval paths JIT-compatible without requiring callers to branch in
    Python.
    """
    return jax.lax.cond(
        jnp.asarray(training),
        lambda st: update_obs_norm(st, batch),
        lambda st: st,
        state,
    )


def normalize_obs(
    state: ObsNormState,
    obs: Float[Array, "... obs_dim"],
    eps: float = 1e-8,
    clip: float = 10.0,
    config: ObsNormConfig | None = None,
) -> Array:
    """Normalize observations using running statistics.

    Subtracts the running mean and divides by the running standard
    deviation, then clips to ``[-clip, clip]``. If ``config`` is provided,
    per-dimension modes can leave values unchanged, apply fixed-range scaling,
    or use bounded angle/phase transforms. Broadcasting handles arbitrary
    leading batch dimensions.

    Args:
        state: Running statistics from ``update_obs_norm``.
        obs: Observations to normalize, with ``obs_dim`` as the last axis.
        eps: Small constant for numerical stability in the denominator.
        clip: Symmetric clipping bound for normalized values.
        config: Optional per-dimension normalization configuration.

    Returns:
        Normalized (and clipped) observations, same shape as ``obs``.
    """
    standardized = (obs - state.mean) / jnp.sqrt(state.var + eps)
    standardized = jnp.clip(standardized, -clip, clip)
    if config is None:
        return standardized

    denom = jnp.maximum(config.scale_max - config.scale_min, eps)
    scaled = (obs - config.scale_min) / denom
    angle = jnp.arctan2(jnp.sin(obs), jnp.cos(obs)) / jnp.pi
    phase = jnp.mod(obs, 1.0)

    out = jnp.where(config.mode == _MODE_STANDARDIZE, standardized, obs)
    out = jnp.where(config.mode == _MODE_SCALE, scaled, out)
    out = jnp.where(config.mode == _MODE_ANGLE, angle, out)
    out = jnp.where(config.mode == _MODE_PHASE, phase, out)
    return out


update_stats = update_obs_norm
normalize = normalize_obs

__all__ = [
    "ObsNormConfig",
    "ObsNormMode",
    "ObsNormState",
    "RunningNormState",
    "init_obs_norm",
    "init_obs_norm_config",
    "maybe_update_obs_norm",
    "merge_obs_norm",
    "normalize",
    "normalize_obs",
    "update_obs_norm",
    "update_stats",
]
