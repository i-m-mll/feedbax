"""Tests for feedbax.training.rl.tasks."""

import jax
import jax.numpy as jnp
import pytest

from feedbax.training.rl.tasks import (
    TASK_HOLD,
    TASK_REACH,
    TASK_SWING,
    TASK_TRACK,
    TaskSpec,
    hold_task,
    reach_task,
    sample_task,
    sample_task_jax,
    swing_task,
    tracking_task,
)


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


@pytest.fixture
def timestamps():
    return jnp.linspace(0.0, 2.0, 1000)


class TestTaskSpec:
    def test_reach(self, timestamps):
        start = jnp.array([0.0, 0.0])
        target = jnp.array([0.3, 0.2])
        task = reach_task(timestamps, start, target)
        assert task.task_type == TASK_REACH
        assert task.target_pos.shape == (1000, 2)
        assert task.target_vel.shape == (1000, 2)
        assert jnp.all(jnp.isfinite(task.target_pos))

    def test_hold(self, timestamps):
        task = hold_task(timestamps, jnp.array([0.2, 0.1]))
        assert task.task_type == TASK_HOLD
        assert task.target_pos.shape == (1000, 2)
        # All positions should be the same (static hold)
        assert jnp.allclose(task.target_pos[0], task.target_pos[-1])

    def test_tracking(self, timestamps, key):
        task = tracking_task(timestamps, key)
        assert task.task_type == TASK_TRACK
        assert task.target_pos.shape == (1000, 2)
        assert jnp.all(jnp.isfinite(task.target_pos))

    def test_swing(self, timestamps):
        task = swing_task(timestamps)
        assert task.task_type == TASK_SWING
        assert task.target_pos.shape == (1000, 2)

    def test_sample_task(self, timestamps, key):
        task = sample_task(timestamps, key)
        assert task.task_type in (TASK_REACH, TASK_HOLD, TASK_TRACK, TASK_SWING)
        assert task.target_pos.shape == (1000, 2)


class TestSampleTaskJax:
    def test_basic(self, timestamps, key):
        task = sample_task_jax(timestamps, key)
        assert task.target_pos.shape == (1000, 2)
        assert jnp.all(jnp.isfinite(task.target_pos))

    def test_jit_compatible(self, timestamps, key):
        jitted = jax.jit(sample_task_jax)
        task = jitted(timestamps, key)
        assert task.target_pos.shape == (1000, 2)
        assert jnp.all(jnp.isfinite(task.target_pos))

    def test_vmap_compatible(self, timestamps):
        keys = jax.random.split(jax.random.PRNGKey(0), 4)
        tasks = jax.vmap(sample_task_jax, in_axes=(None, 0))(timestamps, keys)
        assert tasks.target_pos.shape == (4, 1000, 2)

    def test_deterministic(self, timestamps, key):
        t1 = sample_task_jax(timestamps, key)
        t2 = sample_task_jax(timestamps, key)
        assert jnp.allclose(t1.target_pos, t2.target_pos)
