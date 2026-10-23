"""Tests for LATTICE correlated latent-space exploration noise."""

import jax
import jax.numpy as jnp
import pytest

from feedbax.training.rl.policy import (
    ActorCritic,
    LatticeNoiseState,
    forward_with_latent_noise,
    init_lattice_noise,
    maybe_resample_noise,
    sample_action_with_noise,
)

OBS_DIM = 6
ACTION_DIM = 4
HIDDEN_DIM = 32
HIDDEN_LAYERS = 2


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


@pytest.fixture
def policy(key):
    return ActorCritic(
        OBS_DIM, ACTION_DIM, HIDDEN_DIM, HIDDEN_LAYERS, key=key
    )


@pytest.fixture
def obs(key):
    return jax.random.normal(key, (OBS_DIM,))


class TestInitNoise:
    def test_init_noise_shape(self, key):
        """Noise vector has correct shape and scale is dimension-appropriate."""
        state = init_lattice_noise(HIDDEN_DIM, key)
        assert state.noise.shape == (HIDDEN_DIM,)
        assert state.steps_since_resample == 0

        # Scale should be 0.1 * hidden_dim^(-0.5); check noise is in a
        # reasonable range (within ~4 sigma for 32 samples).
        expected_scale = 0.1 * HIDDEN_DIM ** (-0.5)
        assert jnp.std(state.noise) < 4 * expected_scale


class TestResample:
    def test_resample_after_interval(self, key):
        """Noise is resampled once steps_since_resample reaches the interval."""
        state = init_lattice_noise(HIDDEN_DIM, key)
        original_noise = state.noise

        # Advance to exactly the resample threshold.
        resample_interval = 8
        for i in range(resample_interval):
            k = jax.random.fold_in(key, i + 100)
            state = maybe_resample_noise(state, k, resample_interval)

        # After `resample_interval` increments the counter hits the threshold
        # on the next call, so the (interval+1)-th call triggers resample.
        # Actually: we start at 0, increment 8 times -> counter=8 >= 8 -> resample.
        # The 8th call sees counter=7, increments to... wait, let's trace carefully.
        #
        # init: steps=0
        # call 1: 0 < 8 -> increment -> steps=1
        # call 2: 1 < 8 -> increment -> steps=2
        # ...
        # call 8: 7 < 8 -> increment -> steps=8
        # call 9: 8 >= 8 -> RESAMPLE -> steps=0, new noise
        #
        # So after 8 calls, steps=8 but noise hasn't changed yet.
        assert state.steps_since_resample == 8
        assert jnp.array_equal(state.noise, original_noise)

        # The 9th call triggers the resample.
        k_resample = jax.random.fold_in(key, 999)
        state = maybe_resample_noise(state, k_resample, resample_interval)
        assert state.steps_since_resample == 0
        assert not jnp.array_equal(state.noise, original_noise)

    def test_no_resample_before_interval(self, key):
        """Noise is unchanged before the resample interval elapses."""
        state = init_lattice_noise(HIDDEN_DIM, key)
        original_noise = state.noise

        for i in range(5):
            k = jax.random.fold_in(key, i + 200)
            state = maybe_resample_noise(state, k, resample_interval=8)

        assert jnp.array_equal(state.noise, original_noise)
        assert state.steps_since_resample == 5

    def test_temporal_correlation(self, key):
        """Noise vector is identical across consecutive steps within an interval."""
        state = init_lattice_noise(HIDDEN_DIM, key)
        snapshots = []
        for i in range(4):
            k = jax.random.fold_in(key, i + 300)
            state = maybe_resample_noise(state, k, resample_interval=8)
            snapshots.append(state.noise)

        # All snapshots within the interval should be the same vector.
        for snap in snapshots:
            assert jnp.array_equal(snap, snapshots[0])


class TestForwardWithNoise:
    def test_forward_with_noise_shapes(self, policy, obs, key):
        """Output shapes match the standard forward pass."""
        noise = init_lattice_noise(HIDDEN_DIM, key).noise
        dist, value = forward_with_latent_noise(policy, obs, noise)

        assert dist.alpha.shape == (ACTION_DIM,)
        assert dist.beta.shape == (ACTION_DIM,)
        assert value.shape == ()

    def test_noise_affects_output(self, policy, obs, key):
        """Non-zero noise changes the actor output relative to zero noise."""
        zero_noise = jnp.zeros(HIDDEN_DIM)
        nonzero_noise = init_lattice_noise(HIDDEN_DIM, key).noise

        dist_clean, _ = forward_with_latent_noise(policy, obs, zero_noise)
        dist_noisy, _ = forward_with_latent_noise(policy, obs, nonzero_noise)

        assert not jnp.allclose(dist_clean.alpha, dist_noisy.alpha)

    def test_zero_noise_matches_standard(self, policy, obs):
        """With zero noise, output is identical to the standard forward pass."""
        zero_noise = jnp.zeros(HIDDEN_DIM)
        dist_lattice, value_lattice = forward_with_latent_noise(
            policy, obs, zero_noise
        )
        dist_standard, value_standard = policy.dist_and_value(obs)

        assert jnp.allclose(dist_lattice.alpha, dist_standard.alpha, atol=1e-6)
        assert jnp.allclose(dist_lattice.beta, dist_standard.beta, atol=1e-6)
        assert jnp.allclose(value_lattice, value_standard, atol=1e-6)

    def test_critic_unaffected(self, policy, obs, key):
        """Critic value is the same regardless of latent noise."""
        zero_noise = jnp.zeros(HIDDEN_DIM)
        large_noise = jnp.ones(HIDDEN_DIM) * 10.0

        _, value_clean = forward_with_latent_noise(policy, obs, zero_noise)
        _, value_noisy = forward_with_latent_noise(policy, obs, large_noise)

        assert jnp.allclose(value_clean, value_noisy, atol=1e-6)


class TestSampleActionWithNoise:
    def test_sample_shapes(self, policy, obs, key):
        """sample_action_with_noise returns correct shapes."""
        noise = init_lattice_noise(HIDDEN_DIM, key).noise
        action, log_prob, value = sample_action_with_noise(
            policy, obs, key, noise
        )

        assert action.shape == (ACTION_DIM,)
        assert log_prob.shape == ()
        assert value.shape == ()

    def test_action_bounded(self, policy, obs, key):
        """Sampled actions are in (0, 1) range."""
        noise = init_lattice_noise(HIDDEN_DIM, key).noise
        action, _, _ = sample_action_with_noise(policy, obs, key, noise)

        assert jnp.all(action > 0.0)
        assert jnp.all(action < 1.0)
