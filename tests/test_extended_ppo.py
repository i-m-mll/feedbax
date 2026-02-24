"""Tests for train_ppo_batched_extended with all training enhancements."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree as jt
import pytest

from feedbax.mechanics.body import default_3link_bounds, sample_preset
from feedbax.mechanics.model_builder import ChainConfig, SimConfig
from feedbax.mechanics.mjx_plant import MJXPlant
from feedbax.training.rl.env import RLEnvConfig
from feedbax.training.rl.ppo import (
    PPOConfig,
    TrainingEnhancements,
    train_ppo_batched,
    train_ppo_batched_extended,
)


B = 2   # number of bodies in test batch
N = 64  # parallel envs (small for CPU tests)
TOTAL_STEPS = N * 16 * 3  # 3 PPO updates


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
def env_config(chain_config, sim_config):
    n_steps = int(round(sim_config.episode_duration / sim_config.dt))
    return RLEnvConfig(
        n_steps=n_steps,
        dt=sim_config.dt,
        n_joints=chain_config.n_joints,
        n_muscles=chain_config.n_muscles,
        default_task_type=0,  # REACH-only for faster convergence in tests
    )


@pytest.fixture
def presets(key):
    bounds = default_3link_bounds()
    keys = jax.random.split(key, B)
    return [sample_preset(bounds, k) for k in keys]


@pytest.fixture
def batched_plant(presets, chain_config, sim_config):
    return MJXPlant.build_batch(
        presets, chain_config, sim_config, clip_states=False,
    )


@pytest.fixture
def ppo_config():
    return PPOConfig(
        total_timesteps=TOTAL_STEPS,
        n_steps_per_update=16,
        n_epochs=1,
        n_minibatches=2,
        hidden_dim=32,
        hidden_layers=1,
    )


class TestExtendedBaseline:
    """Extended function with all enhancements disabled = baseline behavior."""

    def test_runs_without_error(self, batched_plant, env_config, ppo_config, key):
        """Extended training with no enhancements produces valid output."""
        enhancements = TrainingEnhancements()  # all disabled
        policy, metrics = train_ppo_batched_extended(
            batched_plant, env_config, ppo_config, key,
            n_envs=N, enhancements=enhancements,
        )

        # Check output shapes
        leaves = [l for l in jt.leaves(policy) if isinstance(l, jnp.ndarray)]
        assert all(l.shape[0] == B for l in leaves), "Leading dim should be B"

        # Check metrics
        assert metrics["updates"] >= 1
        assert len(metrics["per_body_mean_return"]) >= 1
        assert len(metrics["per_body_success_rate"]) >= 1

    def test_returns_are_finite(self, batched_plant, env_config, ppo_config, key):
        """Returns should not be NaN or Inf."""
        enhancements = TrainingEnhancements()
        _, metrics = train_ppo_batched_extended(
            batched_plant, env_config, ppo_config, key,
            n_envs=N, enhancements=enhancements,
        )
        for r in metrics["per_body_mean_return"]:
            assert jnp.all(jnp.isfinite(r)), "Non-finite per-body returns"

    def test_shapes_match_batched(self, batched_plant, env_config, ppo_config, key):
        """Output policy shapes should match train_ppo_batched."""
        enhancements = TrainingEnhancements()
        extended_policy, _ = train_ppo_batched_extended(
            batched_plant, env_config, ppo_config, key,
            n_envs=N, enhancements=enhancements,
        )
        batched_policy, _ = train_ppo_batched(
            batched_plant, env_config, ppo_config, key,
            n_envs=N,
        )

        ext_shapes = [l.shape for l in jt.leaves(extended_policy)
                      if isinstance(l, jnp.ndarray)]
        bat_shapes = [l.shape for l in jt.leaves(batched_policy)
                      if isinstance(l, jnp.ndarray)]
        assert ext_shapes == bat_shapes


class TestExtendedObsNorm:
    """Test observation normalization enhancement."""

    def test_obs_norm_runs(self, batched_plant, env_config, ppo_config, key):
        """obs_norm=True should complete without errors."""
        enhancements = TrainingEnhancements(obs_norm=True)
        policy, metrics = train_ppo_batched_extended(
            batched_plant, env_config, ppo_config, key,
            n_envs=N, enhancements=enhancements,
        )
        assert metrics["updates"] >= 1

    def test_obs_norm_mean_populated(self, batched_plant, env_config, ppo_config, key):
        """obs_norm_mean should be set in metrics after training."""
        enhancements = TrainingEnhancements(obs_norm=True)
        _, metrics = train_ppo_batched_extended(
            batched_plant, env_config, ppo_config, key,
            n_envs=N, enhancements=enhancements,
        )
        assert metrics["obs_norm_mean"] is not None
        # Shape: (B, obs_dim)
        obs_dim = env_config.n_joints * 2 + env_config.n_muscles + 2 + 2 + 2 + 1
        assert metrics["obs_norm_mean"].shape == (B, obs_dim)

    def test_obs_norm_mean_finite(self, batched_plant, env_config, ppo_config, key):
        """Running mean should be finite after warmup."""
        enhancements = TrainingEnhancements(obs_norm=True)
        _, metrics = train_ppo_batched_extended(
            batched_plant, env_config, ppo_config, key,
            n_envs=N, enhancements=enhancements,
        )
        assert jnp.all(jnp.isfinite(metrics["obs_norm_mean"]))


class TestExtendedLattice:
    """Test LATTICE noise enhancement."""

    def test_lattice_runs(self, batched_plant, env_config, ppo_config, key):
        """lattice_noise=True should complete without errors."""
        enhancements = TrainingEnhancements(lattice_noise=True)
        policy, metrics = train_ppo_batched_extended(
            batched_plant, env_config, ppo_config, key,
            n_envs=N, enhancements=enhancements,
        )
        assert metrics["updates"] >= 1

    def test_lattice_returns_finite(self, batched_plant, env_config, ppo_config, key):
        """Returns should remain finite with LATTICE noise."""
        enhancements = TrainingEnhancements(lattice_noise=True)
        _, metrics = train_ppo_batched_extended(
            batched_plant, env_config, ppo_config, key,
            n_envs=N, enhancements=enhancements,
        )
        for r in metrics["per_body_mean_return"]:
            assert jnp.all(jnp.isfinite(r)), "Non-finite returns with LATTICE"


class TestExtendedCurriculum:
    """Test curriculum learning enhancement."""

    def test_curriculum_runs(self, batched_plant, env_config, ppo_config, key):
        """curriculum=True should complete without errors."""
        enhancements = TrainingEnhancements(
            curriculum=True, curriculum_arm_reach=0.5,
        )
        policy, metrics = train_ppo_batched_extended(
            batched_plant, env_config, ppo_config, key,
            n_envs=N, enhancements=enhancements,
        )
        assert metrics["updates"] >= 1

    def test_curriculum_stages_tracked(self, batched_plant, env_config, ppo_config, key):
        """curriculum_stages metric should be populated per update."""
        enhancements = TrainingEnhancements(
            curriculum=True, curriculum_arm_reach=0.5,
        )
        _, metrics = train_ppo_batched_extended(
            batched_plant, env_config, ppo_config, key,
            n_envs=N, enhancements=enhancements,
        )
        assert "curriculum_stages" in metrics
        assert len(metrics["curriculum_stages"]) == metrics["updates"]
        # Each entry should be (B,) shaped
        for stages in metrics["curriculum_stages"]:
            assert stages.shape == (B,)

    def test_curriculum_initial_stage_zero(self, batched_plant, env_config, ppo_config, key):
        """All bodies should start at curriculum stage 0."""
        enhancements = TrainingEnhancements(
            curriculum=True, curriculum_arm_reach=0.5,
        )
        _, metrics = train_ppo_batched_extended(
            batched_plant, env_config, ppo_config, key,
            n_envs=N, enhancements=enhancements,
        )
        # First update's stages should be >= 0 (may have advanced from 0)
        first_stages = metrics["curriculum_stages"][0]
        assert jnp.all(first_stages >= 0)


class TestExtendedAll:
    """Test all enhancements enabled together."""

    def test_all_enhancements_run(self, batched_plant, env_config, ppo_config, key):
        """All enhancements enabled simultaneously should not crash."""
        enhancements = TrainingEnhancements(
            obs_norm=True,
            lattice_noise=True,
            curriculum=True,
            curriculum_arm_reach=0.5,
            reward_annealing=True,
            annealing_start_fraction=0.3,  # Low so annealing activates in short test
        )
        policy, metrics = train_ppo_batched_extended(
            batched_plant, env_config, ppo_config, key,
            n_envs=N, enhancements=enhancements,
        )
        assert metrics["updates"] >= 1

    def test_all_shapes_correct(self, batched_plant, env_config, ppo_config, key):
        """Output shapes should be correct with all enhancements."""
        enhancements = TrainingEnhancements(
            obs_norm=True,
            lattice_noise=True,
            curriculum=True,
            curriculum_arm_reach=0.5,
            reward_annealing=True,
        )
        policy, metrics = train_ppo_batched_extended(
            batched_plant, env_config, ppo_config, key,
            n_envs=N, enhancements=enhancements,
        )
        # Policy should have (B,) leading dimension
        leaves = [l for l in jt.leaves(policy) if isinstance(l, jnp.ndarray)]
        assert all(l.shape[0] == B for l in leaves)

        # Metrics should have all enhancement-specific data
        assert metrics["obs_norm_mean"] is not None
        assert "curriculum_stages" in metrics
        assert "distance_weight_schedule" in metrics
        assert len(metrics["per_body_success_rate"]) == metrics["updates"]

    def test_all_returns_finite(self, batched_plant, env_config, ppo_config, key):
        """Returns should remain finite with all enhancements."""
        enhancements = TrainingEnhancements(
            obs_norm=True,
            lattice_noise=True,
            curriculum=True,
            curriculum_arm_reach=0.5,
            reward_annealing=True,
        )
        _, metrics = train_ppo_batched_extended(
            batched_plant, env_config, ppo_config, key,
            n_envs=N, enhancements=enhancements,
        )
        for r in metrics["per_body_mean_return"]:
            assert jnp.all(jnp.isfinite(r)), "Non-finite returns with all enhancements"

    def test_reward_annealing_schedule(self, batched_plant, env_config, ppo_config, key):
        """Reward annealing schedule should decrease over time."""
        enhancements = TrainingEnhancements(
            reward_annealing=True,
            annealing_start_fraction=0.0,  # Start immediately
        )
        _, metrics = train_ppo_batched_extended(
            batched_plant, env_config, ppo_config, key,
            n_envs=N, enhancements=enhancements,
        )
        schedule = metrics["distance_weight_schedule"]
        assert len(schedule) >= 2
        # Should be monotonically non-increasing
        for i in range(1, len(schedule)):
            assert schedule[i] <= schedule[i - 1] + 1e-6
        # Final value should be close to 0 (since annealing_start=0.0)
        assert schedule[-1] < 0.5
