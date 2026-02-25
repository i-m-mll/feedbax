"""Tests for REACH-only mode and n_steps frame_skip helper.

Bug: 2055433
"""

import jax
import jax.numpy as jnp
import pytest

from feedbax.mechanics.plant import DirectForceInput
from feedbax.mechanics.skeleton.arm import TwoLinkArm
from feedbax.training.rl.env import (
    RLEnvConfig,
    auto_reset,
    compute_n_steps,
    rl_env_reset,
    rl_env_step,
)
from feedbax.training.rl.tasks import TASK_REACH, sample_task_params_jax


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


@pytest.fixture
def plant():
    return DirectForceInput(TwoLinkArm(), clip_states=False)


@pytest.fixture
def base_config():
    return RLEnvConfig(
        n_steps=100,
        dt=0.01,
        frame_skip=1,
        n_joints=2,
        n_muscles=2,
        action_scale=10.0,
        action_offset=-5.0,
    )


class TestDefaultTaskType:
    def test_default_task_type_none(self, base_config):
        """Default config has None — random task types work."""
        assert base_config.default_task_type is None

    def test_default_task_type_reach(self, plant, base_config, key):
        """With default_task_type=0, auto_reset always produces REACH tasks."""
        config = RLEnvConfig(
            n_steps=base_config.n_steps,
            dt=base_config.dt,
            frame_skip=base_config.frame_skip,
            n_joints=base_config.n_joints,
            n_muscles=base_config.n_muscles,
            action_scale=base_config.action_scale,
            action_offset=base_config.action_offset,
            default_task_type=TASK_REACH,
        )

        # Create an initial state, then trigger auto_reset many times.
        task = sample_task_params_jax(
            key, TASK_REACH, config.n_steps, config.dt,
            segment_lengths=jnp.array([0.3, 0.25]),
            use_fk=False,
            max_target_distance=10.0,
            use_curriculum=False,
            single_task=True,
        )
        state = rl_env_reset(plant, config, task, key)
        done = jnp.array(1.0)

        for i in range(20):
            key = jax.random.fold_in(key, i)
            state = auto_reset(plant, config, state, done, key)
            task_type = int(state.task.task_type)
            assert task_type == TASK_REACH, (
                f"Expected REACH (0) but got task_type={task_type} on iteration {i}"
            )


class TestComputeNSteps:
    def test_compute_n_steps(self):
        """2.0s episode, dt=0.002, frame_skip=5 -> 200 control steps."""
        assert compute_n_steps(2.0, 0.002, 5) == 200

    def test_compute_n_steps_no_frameskip(self):
        """2.0s episode, dt=0.002, no frame_skip -> 1000 control steps."""
        assert compute_n_steps(2.0, 0.002) == 1000
