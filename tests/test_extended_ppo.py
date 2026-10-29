"""Tests for train_ppo_batched with all training enhancements."""

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
)


pytestmark = pytest.mark.optional_ppo


B = 2  # number of bodies in test batch
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
        presets,
        chain_config,
        sim_config,
        clip_states=False,
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


@pytest.mark.parametrize(
    "enhancements",
    [
        pytest.param(TrainingEnhancements(), id="baseline"),
        pytest.param(TrainingEnhancements(obs_norm=True), id="observation-normalization"),
        pytest.param(TrainingEnhancements(lattice_noise=True), id="lattice-noise"),
        pytest.param(
            TrainingEnhancements(curriculum=True, curriculum_arm_reach=0.5),
            id="curriculum",
        ),
        pytest.param(
            TrainingEnhancements(
                obs_norm=True,
                lattice_noise=True,
                curriculum=True,
                curriculum_arm_reach=0.5,
                reward_annealing=True,
                annealing_start_fraction=0.3,
            ),
            id="all-enhancements",
        ),
        pytest.param(
            TrainingEnhancements(reward_annealing=True, annealing_start_fraction=0.0),
            id="reward-annealing",
        ),
    ],
)
def test_training_contract(batched_plant, env_config, ppo_config, key, enhancements):
    """Each enhancement configuration produces valid policies and populated metrics."""
    policy, metrics = train_ppo_batched(
        batched_plant,
        env_config,
        ppo_config,
        key,
        n_envs=N,
        enhancements=enhancements,
    )

    leaves = [leaf for leaf in jt.leaves(policy) if isinstance(leaf, jnp.ndarray)]
    assert leaves, "Expected policy array leaves"
    for index, leaf in enumerate(leaves):
        assert leaf.shape[0] == B, f"Policy leaf {index} has shape {leaf.shape}"
        assert jnp.all(jnp.isfinite(leaf)), f"Non-finite policy leaf {index}"

    updates = metrics["updates"]
    assert updates == 3, "Expected all configured PPO updates to complete"
    for name in ("per_body_mean_return", "per_body_success_rate"):
        assert len(metrics[name]) == updates, f"Missing updates in {name}"
        for index, values in enumerate(metrics[name]):
            assert values.shape == (B,), f"{name} update {index} has shape {values.shape}"
            assert jnp.all(jnp.isfinite(values)), f"Non-finite {name} at update {index}"

    if enhancements.obs_norm:
        mean = metrics["obs_norm_mean"]
        assert mean is not None, "Observation normalization mean was not populated"
        obs_dim = env_config.n_joints * 2 + env_config.n_muscles + 2 + 2 + 2 + 1
        assert mean.shape == (B, obs_dim)
        assert jnp.all(jnp.isfinite(mean)), "Non-finite observation normalization mean"

    if enhancements.curriculum:
        stages = metrics["curriculum_stages"]
        assert len(stages) == updates, "Missing curriculum stages"
        for index, values in enumerate(stages):
            assert values.shape == (B,), f"Curriculum update {index} has shape {values.shape}"
            assert jnp.all(values >= 0), f"Negative curriculum stage at update {index}"

    if enhancements.reward_annealing:
        schedule = jnp.asarray(metrics["distance_weight_schedule"])
        assert schedule.shape == (updates,), "Missing reward annealing updates"
        assert jnp.all(jnp.isfinite(schedule)), "Non-finite reward annealing schedule"
        assert jnp.all(jnp.diff(schedule) <= 1e-6), "Reward weights must not increase"
        assert schedule[-1] < 0.5, "Reward annealing did not decrease the distance weight"

    if enhancements == TrainingEnhancements():
        default_policy, _ = train_ppo_batched(batched_plant, env_config, ppo_config, key, n_envs=N)
        default_leaves = [
            leaf for leaf in jt.leaves(default_policy) if isinstance(leaf, jnp.ndarray)
        ]
        assert [leaf.shape for leaf in leaves] == [leaf.shape for leaf in default_leaves]
