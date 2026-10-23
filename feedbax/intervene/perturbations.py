import jax.numpy as jnp
import jax.random as jr
from feedbax.intervene import ConstantInput, ConstantInputParams


def random_unit_vector(key, dim):
    # Could do `jnp.zeros((dim,)).at[impulse_dim].set(1)` for vector toward one dimension
    v = jr.normal(key, (dim,))
    return v / jnp.linalg.norm(v)


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
        def array(trial_spec, batch_info, key):
            return random_unit_vector(key, 2)
    else:
        array = jnp.zeros((2,)).at[feedback_dim].set(1)

    return ConstantInput(
        params=ConstantInputParams(
            scale=amplitude,
            arrays=array,
            active=trial_mask,
        ),
    )
