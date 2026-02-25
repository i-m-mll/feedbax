"""Tests for lazy task generation utilities."""

import jax
import jax.numpy as jnp
import pytest

from feedbax.training.rl.tasks import (
    TASK_HOLD,
    TASK_REACH,
    TASK_SWING,
    TASK_TRACK,
    reconstruct_trajectory,
    sample_task_jax,
    sample_task_params_jax,
    target_at_t,
)


@pytest.fixture
def key():
    return jax.random.PRNGKey(0)


@pytest.fixture
def n_steps():
    return 128


@pytest.fixture
def dt():
    return 0.01


SEGMENT_LENGTHS = jnp.array([0.3, 0.25])

# Default keyword args for sample_task_params_jax (legacy Cartesian sampling,
# no curriculum, fixed task type).
DEFAULT_KW = dict(
    segment_lengths=SEGMENT_LENGTHS,
    use_fk=False,
    max_target_distance=10.0,
    use_curriculum=False,
    single_task=True,
)


@pytest.mark.parametrize("task_type", [TASK_REACH, TASK_HOLD, TASK_TRACK, TASK_SWING])
def test_target_at_t_matches_task_spec(key, n_steps, dt, task_type):
    timestamps = jnp.arange(n_steps) * dt
    task_spec = sample_task_jax(timestamps, key, task_type=task_type)
    params = sample_task_params_jax(key, task_type, n_steps, dt, **DEFAULT_KW)

    t_idx = jnp.arange(n_steps)
    pos, vel = jax.vmap(lambda t: target_at_t(params, t))(t_idx)

    assert jnp.allclose(pos, task_spec.target_pos, atol=1e-5)
    assert jnp.allclose(vel, task_spec.target_vel, atol=1e-4)


@pytest.mark.parametrize("task_type", [TASK_REACH, TASK_HOLD, TASK_TRACK, TASK_SWING])
def test_reconstruct_trajectory_matches_task_spec(key, n_steps, dt, task_type):
    timestamps = jnp.arange(n_steps) * dt
    task_spec = sample_task_jax(timestamps, key, task_type=task_type)
    params = sample_task_params_jax(key, task_type, n_steps, dt, **DEFAULT_KW)

    pos, vel = reconstruct_trajectory(params)

    assert jnp.allclose(pos, task_spec.target_pos, atol=1e-5)
    assert jnp.allclose(vel, task_spec.target_vel, atol=1e-4)


def test_sample_task_params_shapes(key, n_steps, dt):
    params = sample_task_params_jax(key, TASK_TRACK, n_steps, dt, **DEFAULT_KW)
    assert params.start_pos.shape == (2,)
    assert params.end_pos.shape == (2,)
    assert params.control_points.shape == (6, 2)
    assert params.perturb_force.shape == (2,)
    assert params.n_steps == n_steps


def test_jit_and_vmap_compatible(key, n_steps, dt):
    jitted_sampler = jax.jit(
        lambda k: sample_task_params_jax(k, TASK_REACH, n_steps, dt, **DEFAULT_KW)
    )
    params = jitted_sampler(key)

    jitted_target = jax.jit(lambda p, t: target_at_t(p, t))
    pos, vel = jitted_target(params, jnp.array(3))
    assert pos.shape == (2,)
    assert vel.shape == (2,)

    jitted_recon = jax.jit(reconstruct_trajectory)
    traj_pos, traj_vel = jitted_recon(params)
    assert traj_pos.shape == (n_steps, 2)
    assert traj_vel.shape == (n_steps, 2)

    keys = jax.random.split(key, 4)
    params_batched = jax.vmap(
        lambda k: sample_task_params_jax(k, TASK_HOLD, n_steps, dt, **DEFAULT_KW)
    )(keys)
    batched_pos, batched_vel = jax.vmap(reconstruct_trajectory)(params_batched)
    assert batched_pos.shape == (4, n_steps, 2)
    assert batched_vel.shape == (4, n_steps, 2)
