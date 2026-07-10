"""Tests for lazy task generation utilities."""

import jax
import jax.numpy as jnp
import pytest

from feedbax.training.rl.tasks import (
    TASK_HOLD,
    TASK_REACH,
    TASK_TRACK,
    reach_task,
    reach_task_params,
    reconstruct_trajectory,
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


def test_sample_task_params_shapes(key, n_steps, dt):
    params = sample_task_params_jax(key, TASK_TRACK, n_steps, dt, **DEFAULT_KW)
    assert params.start_pos.shape == (2,)
    assert params.end_pos.shape == (2,)
    assert params.control_points.shape == (6, 2)
    assert params.perturb_force.shape == (2,)
    assert params.n_steps == n_steps


def test_reach_task_params_materialize_without_stored_trajectory(n_steps, dt):
    start = jnp.array([0.0, 0.0])
    target = jnp.array([0.3, -0.2])
    params = reach_task_params(start, target, n_steps, dt)

    dense_leaves = [
        leaf
        for leaf in jax.tree_util.tree_leaves(params)
        if getattr(leaf, "shape", ()) == (n_steps, 2)
    ]
    assert dense_leaves == []

    pos, vel = reconstruct_trajectory(params)
    dense_spec = reach_task(jnp.arange(n_steps) * dt, start, target)

    assert pos.shape == (n_steps, 2)
    assert vel.shape == (n_steps, 2)
    assert jnp.allclose(pos, dense_spec.target_pos)
    assert jnp.allclose(vel, dense_spec.target_vel, atol=1e-4)
    assert dense_spec.target_pos.shape == (n_steps, 2)


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
