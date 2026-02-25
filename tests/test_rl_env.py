"""Tests for feedbax.training.rl.env."""

import jax
import jax.numpy as jnp
import pytest

from feedbax.mechanics.plant import DirectForceInput
from feedbax.mechanics.skeleton.arm import TwoLinkArm
from feedbax.training.rl.env import (
    RLEnvConfig,
    auto_reset,
    rl_env_get_obs,
    rl_env_reset,
    rl_env_step,
)
from feedbax.training.rl.tasks import sample_task_params_jax


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


@pytest.fixture
def plant():
    return DirectForceInput(TwoLinkArm(), clip_states=False)


@pytest.fixture
def config():
    return RLEnvConfig(
        n_steps=100,
        dt=0.01,
        frame_skip=1,
        n_joints=2,
        n_muscles=2,
        action_scale=10.0,
        action_offset=-5.0,
    )


SEGMENT_LENGTHS = jnp.array([0.3, 0.25])


@pytest.fixture
def task(key):
    return sample_task_params_jax(
        key, 0, 100, 0.01,
        segment_lengths=SEGMENT_LENGTHS,
        use_fk=False,
        max_target_distance=10.0,
        use_curriculum=False,
        single_task=False,
    )


@pytest.fixture
def state(plant, config, task, key):
    return rl_env_reset(plant, config, task, key)


class TestRLEnvReset:
    def test_shapes(self, state, config):
        assert state.muscle_activations.shape == (config.n_muscles,)
        assert state.t_index == 0

    def test_finite(self, state):
        assert jnp.all(jnp.isfinite(state.muscle_activations))


class TestRLEnvStep:
    def test_step(self, plant, config, state):
        action = jnp.ones(config.n_muscles) * 0.5
        new_state, obs, reward, done = rl_env_step(plant, config, state, action)
        assert new_state.t_index == 1
        assert jnp.isfinite(reward)
        assert obs.ndim == 1

    def test_done_at_end(self, plant, config, state):
        action = jnp.zeros(config.n_muscles)
        for _ in range(config.n_steps):
            state, obs, reward, done = rl_env_step(plant, config, state, action)
        assert float(done) == 1.0


class TestRLEnvObs:
    def test_shape(self, plant, config, state):
        obs = rl_env_get_obs(plant, config, state)
        expected_dim = config.n_joints * 2 + config.n_muscles + 2 + 2 + 2 + 1
        assert obs.shape == (expected_dim,)
        assert jnp.all(jnp.isfinite(obs))


class TestAutoReset:
    def test_reset_on_done(self, plant, config, state, key):
        done = jnp.array(1.0)
        new_state = auto_reset(plant, config, state, done, key)
        assert new_state.t_index == 0

    def test_no_reset_on_not_done(self, plant, config, state, key):
        # Step once to get t_index=1
        action = jnp.zeros(config.n_muscles)
        state, _, _, _ = rl_env_step(plant, config, state, action)
        done = jnp.array(0.0)
        new_state = auto_reset(plant, config, state, done, key)
        assert new_state.t_index == 1
