"""Tests for feedbax.training.rl.obs_norm."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from feedbax.training.rl.obs_norm import (
    ObsNormState,
    init_obs_norm,
    normalize_obs,
    update_obs_norm,
)


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


class TestObsNorm:
    def test_init_shapes(self):
        """Correct shapes for obs_dim=17."""
        state = init_obs_norm(17)
        assert state.count.shape == ()
        assert state.mean.shape == (17,)
        assert state.var.shape == (17,)
        assert float(state.count) == 0.0
        assert jnp.allclose(state.mean, jnp.zeros(17))
        assert jnp.allclose(state.var, jnp.ones(17))

    def test_single_update(self):
        """Mean/var updated correctly for a known batch."""
        state = init_obs_norm(3)
        batch = jnp.array([
            [1.0, 2.0, 3.0],
            [3.0, 4.0, 5.0],
            [5.0, 6.0, 7.0],
        ])
        state = update_obs_norm(state, batch)

        expected_mean = jnp.array([3.0, 4.0, 5.0])
        # Population variance: var([1,3,5]) = 8/3, var([2,4,6]) = 8/3, etc.
        expected_var = jnp.var(batch, axis=0)

        assert float(state.count) == 3.0
        assert jnp.allclose(state.mean, expected_mean, atol=1e-6)
        assert jnp.allclose(state.var, expected_var, atol=1e-6)

    def test_multiple_updates(self, key):
        """Converges to true mean/var for N(3, 2) samples."""
        true_mean = 3.0
        true_std = 2.0
        n_batches = 50
        batch_size = 200
        obs_dim = 5

        state = init_obs_norm(obs_dim)
        for i in range(n_batches):
            k = jax.random.fold_in(key, i)
            batch = true_mean + true_std * jax.random.normal(k, (batch_size, obs_dim))
            state = update_obs_norm(state, batch)

        assert float(state.count) == n_batches * batch_size
        assert jnp.allclose(state.mean, true_mean * jnp.ones(obs_dim), atol=0.15)
        assert jnp.allclose(state.var, true_std**2 * jnp.ones(obs_dim), atol=0.3)

    def test_normalize_centered(self, key):
        """After updating with data, normalized output has mean~0, std~1."""
        obs_dim = 10
        state = init_obs_norm(obs_dim)

        # Accumulate enough data for stable statistics
        for i in range(20):
            k = jax.random.fold_in(key, i)
            batch = 5.0 + 3.0 * jax.random.normal(k, (500, obs_dim))
            state = update_obs_norm(state, batch)

        # Normalize a fresh batch from the same distribution
        test_batch = 5.0 + 3.0 * jax.random.normal(
            jax.random.PRNGKey(999), (1000, obs_dim)
        )
        normed = normalize_obs(state, test_batch)

        assert jnp.allclose(jnp.mean(normed, axis=0), 0.0, atol=0.15)
        assert jnp.allclose(jnp.std(normed, axis=0), 1.0, atol=0.15)

    def test_normalize_clipping(self):
        """Extreme values are clipped to [-clip, clip]."""
        state = ObsNormState(
            count=jnp.array(100.0),
            mean=jnp.zeros(3),
            var=jnp.ones(3),
        )
        # Values far outside normal range
        obs = jnp.array([[100.0, -100.0, 0.5]])
        normed = normalize_obs(state, obs, clip=5.0)

        assert jnp.all(normed <= 5.0)
        assert jnp.all(normed >= -5.0)
        # The 0.5 value should not be clipped
        assert jnp.allclose(normed[0, 2], 0.5, atol=1e-5)

    def test_welford_merge_correct(self, key):
        """Compare incremental Welford against numpy for 1000 samples in 10 batches."""
        obs_dim = 8
        n_batches = 10
        batch_size = 100

        # Generate all data upfront
        all_data = jax.random.normal(key, (n_batches * batch_size, obs_dim))

        # Incremental Welford
        state = init_obs_norm(obs_dim)
        for i in range(n_batches):
            batch = all_data[i * batch_size : (i + 1) * batch_size]
            state = update_obs_norm(state, batch)

        # Numpy reference (full dataset)
        all_np = np.array(all_data)
        ref_mean = np.mean(all_np, axis=0)
        ref_var = np.var(all_np, axis=0)

        assert jnp.allclose(state.mean, jnp.array(ref_mean), atol=1e-4)
        assert jnp.allclose(state.var, jnp.array(ref_var), atol=1e-4)
        assert float(state.count) == n_batches * batch_size

    def test_vmap_compatible(self, key):
        """jax.vmap(update_obs_norm) works for per-body normalization."""
        n_bodies = 4
        obs_dim = 6
        batch_size = 32

        # Stack of per-body states
        states = jax.vmap(lambda _: init_obs_norm(obs_dim))(jnp.arange(n_bodies))

        # Per-body batches: (n_bodies, batch_size, obs_dim)
        batches = jax.random.normal(key, (n_bodies, batch_size, obs_dim))

        # Vmapped update
        new_states = jax.vmap(update_obs_norm)(states, batches)

        assert new_states.count.shape == (n_bodies,)
        assert new_states.mean.shape == (n_bodies, obs_dim)
        assert new_states.var.shape == (n_bodies, obs_dim)
        assert jnp.all(new_states.count == batch_size)

        # Verify per-body correctness for body 0
        expected_mean = jnp.mean(batches[0], axis=0)
        expected_var = jnp.var(batches[0], axis=0)
        assert jnp.allclose(new_states.mean[0], expected_mean, atol=1e-5)
        assert jnp.allclose(new_states.var[0], expected_var, atol=1e-5)
