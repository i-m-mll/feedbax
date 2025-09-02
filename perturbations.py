from collections.abc import Callable
from typing import Optional

import jax
import jax.numpy as jnp
import jax.random as jr
from feedbax.intervene import ConstantInput
from feedbax.intervene.schedule import TimeSeriesParam
from jax_cookbook import is_type
from jaxtyping import Array, PRNGKeyArray


def random_unit_vector(key, dim):
    # Could do `jnp.zeros((dim,)).at[impulse_dim].set(1)` for vector toward one dimension
    v = jr.normal(key, (dim,))
    return v / jnp.linalg.norm(v)


def unmask_1d_at_idx(length, start_idx, unmask_length):
    """Return a 1D array of zeros, with ones only at `start_id : start_idx + unmask_length`."""
    mask = jnp.zeros(length, bool)
    return jax.lax.dynamic_update_slice(mask, jnp.ones(unmask_length, bool), (start_idx,))


def impulse_active(
    n_steps: int,
    impulse_duration: int,
    start_bounds: Optional[tuple[int, int]] = None,
    start_idx_fn: Callable[[PRNGKeyArray, tuple[int, int]], Array] = (
        lambda key, start_bounds: jr.randint(key, (1,), *start_bounds)[0]
    ),
):
    """Return a function that determines when a field is active on a given trial."""
    if start_bounds is None:
        start_bounds = (0, n_steps)

    def f(trial_spec, _, key):
        start_idx = start_idx_fn(key, start_bounds)
        return TimeSeriesParam(unmask_1d_at_idx(n_steps - 1, start_idx, impulse_duration))

    return f


def feedback_impulse(
    n_steps,
    amplitude,
    duration,  # in time steps
    feedback_var,  # 0 (pos) or 1 (vel)
    start_timestep,
    feedback_dim=None,  # x or y
):
    idxs_impulse = slice(start_timestep, start_timestep + duration)
    trial_mask = jnp.zeros((n_steps - 1,), bool).at[idxs_impulse].set(True)

    if feedback_dim is None:
        array = lambda trial_spec, batch_info, key: random_unit_vector(key, 2)
    else:
        array = jnp.zeros((2,)).at[feedback_dim].set(1)

    return ConstantInput.with_params(
        out_where=lambda channel_state: channel_state.output[feedback_var],
        scale=amplitude,
        arrays=array,
        active=TimeSeriesParam(trial_mask),
        # active=impulse_active(
        #     model_info.n_steps,
        #     impulse_duration,
        #     # Always apply the impulse 25% of the way through the trial
        #     start_idx_fn=lambda key, start_bounds: (
        #         int(0.66 * (start_bounds[1] - start_bounds[0]))
        #     ),
        # ),
    )
