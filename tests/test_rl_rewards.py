"""Tests for feedbax.training.rl.rewards."""

import jax
import jax.numpy as jnp
import pytest

from feedbax.training.rl.rewards import RewardConfig, compute_reward
from feedbax.training.rl.tasks import TASK_HOLD, TASK_REACH, TASK_SWING, TASK_TRACK


class TestRewardConfig:
    def test_defaults(self):
        cfg = RewardConfig()
        assert cfg.effort_weight == 0.005
        assert cfg.hold_threshold == 0.02


class TestComputeReward:
    @pytest.fixture
    def base_kwargs(self):
        return dict(
            effector_pos=jnp.array([0.1, 0.2]),
            target_pos=jnp.array([0.1, 0.2]),
            effector_vel=jnp.array([0.0, 0.0]),
            target_vel=jnp.array([0.0, 0.0]),
            muscle_excitations=jnp.zeros(6),
            effort_weight=0.005,
            velocity_weight=0.1,
            hold_bonus=1.0,
            hold_threshold=0.02,
        )

    def test_reach_reward_finite(self, base_kwargs):
        r = compute_reward(task_type=jnp.float32(TASK_REACH), **base_kwargs)
        assert jnp.isfinite(r)

    def test_hold_bonus(self, base_kwargs):
        r_hold = compute_reward(task_type=jnp.float32(TASK_HOLD), **base_kwargs)
        r_reach = compute_reward(task_type=jnp.float32(TASK_REACH), **base_kwargs)
        # Hold gets bonus when on target, reach doesn't
        assert float(r_hold) > float(r_reach)

    def test_effort_penalty(self, base_kwargs):
        r_zero = compute_reward(task_type=jnp.float32(TASK_REACH), **base_kwargs)
        base_kwargs["muscle_excitations"] = jnp.ones(6)
        r_high = compute_reward(task_type=jnp.float32(TASK_REACH), **base_kwargs)
        assert float(r_zero) > float(r_high)

    def test_distance_penalty(self, base_kwargs):
        r_on = compute_reward(task_type=jnp.float32(TASK_REACH), **base_kwargs)
        base_kwargs["effector_pos"] = jnp.array([1.0, 1.0])
        r_off = compute_reward(task_type=jnp.float32(TASK_REACH), **base_kwargs)
        assert float(r_on) > float(r_off)

    def test_jit_compatible(self, base_kwargs):
        jitted = jax.jit(lambda: compute_reward(
            task_type=jnp.float32(TASK_REACH), **base_kwargs
        ))
        r = jitted()
        assert jnp.isfinite(r)

    def test_all_task_types(self, base_kwargs):
        for tt in [TASK_REACH, TASK_HOLD, TASK_TRACK, TASK_SWING]:
            r = compute_reward(task_type=jnp.float32(tt), **base_kwargs)
            assert jnp.isfinite(r)
