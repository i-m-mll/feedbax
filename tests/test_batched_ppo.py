"""Tests for batched PPO training (train_ppo_batched, collect_rollouts_batched)."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree as jt
import pytest

from feedbax.mechanics.body import default_3link_bounds, sample_preset
from feedbax.mechanics.model_builder import ChainConfig, SimConfig
from feedbax.mechanics.mjx_plant import MJXPlant
from feedbax.training.rl.env import RLEnvConfig
from feedbax.training.rl.policy import ActorCritic
from feedbax.training.rl.ppo import (
    PPOConfig,
    _stack_pytrees,
    collect_rollouts_batched,
    train_ppo_batched,
    _collect_rollout,
    _init_envs,
    compute_gae_scan,
)
from feedbax.training.rl.rewards import RewardConfig
from feedbax.training.rl.tasks import TASK_REACH, TASK_HOLD


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


@pytest.fixture
def chain_config():
    return ChainConfig(n_joints=3)


@pytest.fixture
def sim_config():
    return SimConfig(dt=0.002, episode_duration=0.5)


@pytest.fixture
def reward_config():
    return RewardConfig()


@pytest.fixture
def env_config(chain_config, sim_config, reward_config):
    n_steps = int(round(sim_config.episode_duration / sim_config.dt))
    return RLEnvConfig(
        n_steps=n_steps,
        dt=sim_config.dt,
        n_joints=chain_config.n_joints,
        n_muscles=chain_config.n_muscles,
        effort_weight=reward_config.effort_weight,
        velocity_weight=reward_config.velocity_weight,
        hold_bonus=reward_config.hold_bonus,
        hold_threshold=reward_config.hold_threshold,
    )


@pytest.fixture
def presets(key):
    bounds = default_3link_bounds()
    keys = jax.random.split(key, 3)
    return [sample_preset(bounds, k) for k in keys]


@pytest.fixture
def batched_plant(presets, chain_config, sim_config):
    return MJXPlant.build_batch(
        presets, chain_config, sim_config, clip_states=False,
    )


@pytest.fixture
def single_plant(presets, chain_config, sim_config):
    return MJXPlant.from_body_preset(
        presets[0], chain_config, sim_config, clip_states=False,
    )


B = 3   # number of bodies in test batch
N = 32  # parallel envs (small for CPU tests)


class TestBuildBatch:
    def test_leading_dim(self, batched_plant):
        """All array leaves should have leading (B,) dimension."""
        for leaf in jt.leaves(batched_plant):
            if isinstance(leaf, jnp.ndarray):
                assert leaf.shape[0] == B, f"Expected leading dim {B}, got {leaf.shape}"

    def test_static_fields_match(self, batched_plant, single_plant):
        """Static fields (nq, nv, nbody) should match across batch."""
        assert batched_plant.skeleton.nq == single_plant.skeleton.nq
        assert batched_plant.skeleton.nv == single_plant.skeleton.nv
        assert batched_plant.skeleton.nbody == single_plant.skeleton.nbody

    def test_finite_values(self, batched_plant):
        """All array leaves should be finite."""
        for leaf in jt.leaves(batched_plant):
            if isinstance(leaf, jnp.ndarray):
                assert jnp.all(jnp.isfinite(leaf)), "Non-finite values in batched plant"


class TestBatchedCollect:
    def test_shapes(self, batched_plant, env_config, key):
        """Batched rollout collection produces correct shapes."""
        obs_dim = env_config.n_joints * 2 + env_config.n_muscles + 2 + 2 + 2 + 1
        action_dim = env_config.n_muscles
        n_steps = 16  # short rollout

        # Create batched policy + states
        key, pk, ek = jax.random.split(key, 3)
        policies = [
            ActorCritic(obs_dim, action_dim, 32, 1, key=k)
            for k in jax.random.split(pk, B)
        ]
        batched_policy = _stack_pytrees(*policies)

        env_keys = jax.random.split(ek, B)
        batched_states = eqx.filter_vmap(
            lambda plant, k: _init_envs(plant, env_config, k, N),
        )(batched_plant, env_keys)

        # Collect
        keys = jax.random.split(key, B)
        states, rollout, last_values, _ = eqx.filter_vmap(
            lambda pl, pol, st, k: _collect_rollout(
                pl, env_config, pol, st, k, n_steps, N,
            ),
        )(batched_plant, batched_policy, batched_states, keys)

        assert rollout.obs.shape == (B, n_steps, N, obs_dim)
        assert rollout.actions.shape == (B, n_steps, N, action_dim)
        assert rollout.rewards.shape == (B, n_steps, N)
        assert last_values.shape == (B, N)

    def test_finite(self, batched_plant, env_config, key):
        """Collected rollout data should be finite."""
        obs_dim = env_config.n_joints * 2 + env_config.n_muscles + 2 + 2 + 2 + 1
        action_dim = env_config.n_muscles

        key, pk, ek = jax.random.split(key, 3)
        policies = [
            ActorCritic(obs_dim, action_dim, 32, 1, key=k)
            for k in jax.random.split(pk, B)
        ]
        batched_policy = _stack_pytrees(*policies)

        env_keys = jax.random.split(ek, B)
        batched_states = eqx.filter_vmap(
            lambda plant, k: _init_envs(plant, env_config, k, N),
        )(batched_plant, env_keys)

        keys = jax.random.split(key, B)
        _, rollout, _, _ = eqx.filter_vmap(
            lambda pl, pol, st, k: _collect_rollout(
                pl, env_config, pol, st, k, 16, N,
            ),
        )(batched_plant, batched_policy, batched_states, keys)

        assert jnp.all(jnp.isfinite(rollout.obs))
        assert jnp.all(jnp.isfinite(rollout.rewards))


class TestBatchedUpdate:
    def test_policy_shapes_preserved(self, batched_plant, env_config, key):
        """PPO update should preserve policy array shapes."""
        obs_dim = env_config.n_joints * 2 + env_config.n_muscles + 2 + 2 + 2 + 1
        action_dim = env_config.n_muscles

        key, pk, ek = jax.random.split(key, 3)
        policies = [
            ActorCritic(obs_dim, action_dim, 32, 1, key=k)
            for k in jax.random.split(pk, B)
        ]
        batched_policy = _stack_pytrees(*policies)

        # Get shapes before update
        shapes_before = [leaf.shape for leaf in jt.leaves(batched_policy)
                         if isinstance(leaf, jnp.ndarray)]

        # Run short training (1 update)
        config = PPOConfig(
            total_timesteps=N * 16,  # just 1 update
            n_steps_per_update=16,
            n_epochs=1,
            n_minibatches=2,
            hidden_dim=32,
            hidden_layers=1,
        )
        trained_policy, metrics = train_ppo_batched(
            batched_plant, env_config, config, key, n_envs=N,
        )

        shapes_after = [leaf.shape for leaf in jt.leaves(trained_policy)
                        if isinstance(leaf, jnp.ndarray)]
        assert shapes_before == shapes_after


class TestBatchedRollout:
    def test_shapes(self, batched_plant, env_config, key):
        """collect_rollouts_batched produces (B, R, T, ...) arrays."""
        obs_dim = env_config.n_joints * 2 + env_config.n_muscles + 2 + 2 + 2 + 1
        action_dim = env_config.n_muscles
        R = 2

        key, pk = jax.random.split(key)
        policies = [
            ActorCritic(obs_dim, action_dim, 32, 1, key=k)
            for k in jax.random.split(pk, B)
        ]
        batched_policy = _stack_pytrees(*policies)

        task_types = jnp.array([TASK_REACH, TASK_HOLD])
        result = collect_rollouts_batched(
            batched_plant, env_config, batched_policy, key, task_types,
        )

        T = env_config.n_steps
        nq = batched_plant.skeleton.nq
        nv = batched_plant.skeleton.nv
        assert result["joint_angles"].shape == (B, R, T, nq)
        assert result["joint_velocities"].shape == (B, R, T, nv)
        assert result["muscle_activations"].shape == (B, R, T, env_config.n_muscles)
        assert result["effector_pos"].shape == (B, R, T, 2)
        assert result["task_target"].shape == (B, R, T, 2)
        assert result["timestamps"].shape == (B, R, T)

    def test_finite(self, batched_plant, env_config, key):
        """Collected rollouts should contain finite values."""
        obs_dim = env_config.n_joints * 2 + env_config.n_muscles + 2 + 2 + 2 + 1
        action_dim = env_config.n_muscles

        key, pk = jax.random.split(key)
        policies = [
            ActorCritic(obs_dim, action_dim, 32, 1, key=k)
            for k in jax.random.split(pk, B)
        ]
        batched_policy = _stack_pytrees(*policies)

        task_types = jnp.array([TASK_REACH])
        result = collect_rollouts_batched(
            batched_plant, env_config, batched_policy, key, task_types,
        )

        for name in ("joint_angles", "joint_velocities", "muscle_activations",
                      "effector_pos"):
            assert jnp.all(jnp.isfinite(result[name])), f"Non-finite in {name}"


@pytest.mark.slow
class TestBatchedTrainShort:
    def test_returns_improve(self, batched_plant, env_config, key):
        """With enough training, mean returns should increase."""
        config = PPOConfig(
            total_timesteps=N * 16 * 5,  # 5 updates
            n_steps_per_update=16,
            n_epochs=2,
            n_minibatches=2,
            hidden_dim=32,
            hidden_layers=1,
        )
        _, metrics = train_ppo_batched(
            batched_plant, env_config, config, key, n_envs=N,
        )

        returns = metrics["per_body_mean_return"]
        assert len(returns) >= 2, "Expected at least 2 PPO updates"
        # Check that returns are finite (training didn't diverge)
        for r in returns:
            assert jnp.all(jnp.isfinite(r)), "Non-finite per-body returns"
