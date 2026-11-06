"""Tests for feedbax.training.rl.ppo."""

import ast
import inspect

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import feedbax.training.rl.ppo as ppo
from feedbax.training.rl.policy import ActorCritic, BetaParams, sample_action_with_noise
from feedbax.training.rl.ppo import compute_gae_scan


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


def test_gae_stops_at_episode_boundaries_and_bootstraps_collection_cutoff():
    # Both lanes reset after reward 2. Only the second lane ends at the
    # final transition, so only the first lane uses the final critic value.
    rewards = jnp.broadcast_to(jnp.array([[1.0], [2.0], [10.0], [20.0]]), (4, 2))
    values = jnp.broadcast_to(jnp.array([[5.0], [6.0], [7.0], [8.0]]), (4, 2))
    dones = jnp.array([[False, False], [True, True], [False, False], [False, True]])
    advantages, returns = compute_gae_scan(
        rewards, values, dones, jnp.array([9.0, 9.0]), gamma=0.5, gae_lambda=0.5,
    )
    assert jnp.allclose(advantages, jnp.array([
        [-2.0, -2.0], [-4.0, -4.0], [11.125, 10.0], [16.5, 12.0],
    ]))
    assert jnp.allclose(returns, jnp.array([
        [3.0, 3.0], [2.0, 2.0], [18.125, 17.0], [24.5, 20.0],
    ]))


class TestActorCritic:
    def test_init(self, key):
        policy = ActorCritic(obs_dim=10, action_dim=4, hidden_dim=32, hidden_layers=2, key=key)
        assert policy.obs_dim == 10
        assert policy.action_dim == 4

    @pytest.mark.parametrize("use_lattice", [False, True], ids=["ordinary", "lattice"])
    def test_public_sampling_preserves_seeded_actions_and_scores(self, key, use_lattice):
        policy = ActorCritic(obs_dim=3, action_dim=3, hidden_dim=4, hidden_layers=1, key=key)
        policy = eqx.tree_at(
            lambda model: model.actor.layers[-1].bias,
            policy,
            jnp.array([0.0, 200_000.0, 0.0, 200_000.0, 0.0, 0.0]),
        )
        obs = jnp.zeros(3)
        noise = jnp.array([0.7, -0.5, 0.3, 0.9]) if use_lattice else None
        dist, expected_value = policy.dist_and_value(obs, noise)
        draws = jax.random.beta(key, dist.alpha, dist.beta)
        assert draws[0] < 1e-4
        assert draws[1] > 1.0 - 1e-4
        assert 1e-4 < draws[2] < 1.0 - 1e-4
        expected_action = jnp.clip(draws, 1e-4, 1.0 - 1e-4)

        if use_lattice:
            action, log_prob, value = sample_action_with_noise(policy, obs, key, noise)
        else:
            action, log_prob, value = policy.sample_action(obs, key)
        assert jnp.array_equal(action, expected_action)
        assert jnp.allclose(log_prob, dist.joint_log_prob(expected_action))
        assert jnp.allclose(value, expected_value)

    def test_dist_and_value(self, key):
        policy = ActorCritic(obs_dim=10, action_dim=4, hidden_dim=32, hidden_layers=2, key=key)
        obs = jnp.zeros(10)
        dist, value = policy.dist_and_value(obs)
        assert isinstance(dist, BetaParams)
        assert dist.alpha.shape == (4,)
        assert jnp.all(dist.alpha > 0)
        assert jnp.isfinite(value)

    def test_batched(self, key):
        policy = ActorCritic(obs_dim=10, action_dim=4, hidden_dim=32, hidden_layers=2, key=key)
        obs = jnp.zeros((8, 10))
        dist, value = policy.dist_and_value(obs)
        assert dist.alpha.shape == (8, 4)
        assert value.shape == (8,)


class TestBetaParams:
    def test_log_prob(self):
        dist = BetaParams(alpha=jnp.array([2.0, 3.0]), beta=jnp.array([2.0, 3.0]))
        lp = dist.log_prob(jnp.array([0.5, 0.5]))
        assert lp.shape == (2,)
        assert jnp.all(jnp.isfinite(lp))

    def test_entropy(self):
        dist = BetaParams(alpha=jnp.array([2.0, 3.0]), beta=jnp.array([2.0, 3.0]))
        ent = dist.entropy()
        assert ent.shape == (2,)
        assert jnp.all(jnp.isfinite(ent))

    def test_sample(self):
        dist = BetaParams(alpha=jnp.array([2.0]), beta=jnp.array([2.0]))
        s = dist.sample(jax.random.PRNGKey(0))
        assert s.shape == (1,)
        assert float(s[0]) > 0 and float(s[0]) < 1


class TestPPOConsolidation:
    def test_public_training_api_is_batched_only(self):
        assert hasattr(ppo, "train_ppo_batched")
        assert not hasattr(ppo, "train_ppo")
        assert not hasattr(ppo, "train_ppo_batched_extended")

    def _assert_no_host_scalar_syncs(self, obj):
        source = inspect.getsource(obj)
        tree = ast.parse(source)
        called_names = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        called_attrs = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert "float" not in called_names
        assert "int" not in called_names
        assert "item" not in called_attrs
        assert "device_get" not in called_attrs
        return source, tree

    def test_outer_training_body_uses_scan_without_host_scalar_syncs(self):
        source, _ = self._assert_no_host_scalar_syncs(ppo._run_ppo_updates)
        assert "jax.lax.scan" in source

    def test_public_training_entrypoint_has_no_python_outer_update_loop(self):
        source = inspect.getsource(ppo.train_ppo_batched)
        tree = ast.parse(source)
        assert not any(isinstance(node, ast.While) for node in ast.walk(tree))
