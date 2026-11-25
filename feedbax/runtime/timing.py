"""Pair times computed at execution with the Model clock."""

import jax.numpy as jnp


def runtime_time_steps(seconds, step_seconds):
    """Round a computed time to a whole step, with half ties away from zero."""
    count = seconds / step_seconds
    whole = jnp.floor(jnp.abs(count))
    rounded = whole + (jnp.abs(count) - whole >= 0.5)
    return jnp.copysign(rounded, count).astype(int)
