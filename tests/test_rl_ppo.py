"""Tests for feedbax.training.rl.ppo."""

import ast
import inspect

import jax
import jax.numpy as jnp
import pytest

import feedbax.training.rl.ppo as ppo
from feedbax.training.rl.policy import ActorCritic, BetaParams
from feedbax.training.rl.ppo import compute_gae_scan


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


class TestGAE:
    def test_shapes(self):
        T, N = 16, 4
        rewards = jnp.ones((T, N))
        values = jnp.ones((T, N)) * 0.5
        dones = jnp.zeros((T, N))
        last_values = jnp.ones(N) * 0.5
        advantages, returns = compute_gae_scan(
            rewards, values, dones, last_values, gamma=0.99, gae_lambda=0.95
        )
        assert advantages.shape == (T, N)
        assert returns.shape == (T, N)

    def test_finite(self):
        T, N = 16, 4
        rewards = jnp.ones((T, N))
        values = jnp.ones((T, N)) * 0.5
        dones = jnp.zeros((T, N))
        last_values = jnp.ones(N) * 0.5
        advantages, returns = compute_gae_scan(
            rewards, values, dones, last_values, gamma=0.99, gae_lambda=0.95
        )
        assert jnp.all(jnp.isfinite(advantages))
        assert jnp.all(jnp.isfinite(returns))

    def test_reference_match(self):
        """Compare scan GAE against Python loop reference implementation."""
        T, N = 8, 2
        key = jax.random.PRNGKey(0)
        rewards = jax.random.normal(key, (T, N))
        values = jax.random.normal(jax.random.PRNGKey(1), (T, N))
        dones = jnp.zeros((T, N))
        last_values = jnp.ones(N)
        gamma, lam = 0.99, 0.95

        # Reference: Python loop
        ref_adv = jnp.zeros((T, N))
        gae = jnp.zeros(N)
        values_ext = jnp.concatenate([values, last_values[None]], axis=0)
        for t in reversed(range(T)):
            mask = 1.0 - dones[t]
            delta = rewards[t] + gamma * values_ext[t + 1] * mask - values[t]
            gae = delta + gamma * lam * mask * gae
            ref_adv = ref_adv.at[t].set(gae)

        advantages, _ = compute_gae_scan(rewards, values, dones, last_values, gamma, lam)
        assert jnp.allclose(advantages, ref_adv, atol=1e-5)


class TestActorCritic:
    def test_init(self, key):
        policy = ActorCritic(obs_dim=10, action_dim=4, hidden_dim=32, hidden_layers=2, key=key)
        assert policy.obs_dim == 10
        assert policy.action_dim == 4

    def test_sample_action(self, key):
        policy = ActorCritic(obs_dim=10, action_dim=4, hidden_dim=32, hidden_layers=2, key=key)
        obs = jnp.zeros(10)
        action, log_prob, value = policy.sample_action(obs, key)
        assert action.shape == (4,)
        assert jnp.all(action >= 0) and jnp.all(action <= 1)
        assert jnp.isfinite(log_prob)
        assert jnp.isfinite(value)

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

    def test_update_body_uses_lax_loop_without_host_scalar_syncs(self):
        source = inspect.getsource(ppo._update_one)
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
        assert "jax.lax.fori_loop" in source
        assert "float" not in called_names
        assert "int" not in called_names
        assert "item" not in called_attrs
        assert "device_get" not in called_attrs
