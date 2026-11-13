"""Half-open whole-trial and whole-epoch selection on the post-step axis."""

from typing import NamedTuple

import jax
import jax.numpy as jnp


class EpochRecord(NamedTuple):
    starts: jax.Array
    ends: jax.Array
    outcomes: jax.Array


def in_epoch(steps, start, end):
    """An unreached epoch has sentinel bounds and contributes no steps."""
    return (start >= 0) & (start <= steps) & (steps < end)


def whole_window(record: EpochRecord, n_steps: int, epochs: tuple[int, ...]) -> jax.Array:
    """An empty epoch selection denotes the whole trial."""
    steps = jnp.arange(n_steps)
    if not epochs:
        return jnp.broadcast_to(True, (*record.starts.shape[:-1], n_steps))
    indices = jnp.asarray(epochs)
    start = jnp.take(record.starts, indices, axis=-1)[..., None]
    end = jnp.take(record.ends, indices, axis=-1)[..., None]
    return jnp.any(in_epoch(steps, start, end), axis=-2)
