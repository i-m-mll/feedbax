"""Tests for feedbax.mechanics.body."""

import jax
import jax.numpy as jnp
import pytest

from feedbax.mechanics.body import (
    BodyPreset,
    BodyPresetBounds,
    default_2link_bounds,
    default_3link_bounds,
    flat_dim,
    from_flat,
    sample_preset,
    to_flat,
)


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


@pytest.fixture
def bounds():
    return default_3link_bounds()


@pytest.fixture
def preset(bounds, key):
    return sample_preset(bounds, key)


class TestBodyPreset:
    def test_create(self, preset):
        assert preset.segment_lengths.shape == (3,)
        assert preset.segment_masses.shape == (3,)
        assert preset.muscle_pcsa.shape == (6,)
        assert preset.muscle_moment_arm_magnitudes.shape == (6, 3)
        assert preset.tau_act == 0.01
        assert preset.tau_deact == 0.04

    def test_sample_within_bounds(self, bounds, key):
        preset = sample_preset(bounds, key)
        assert jnp.all(preset.segment_lengths >= bounds.segment_lengths_min)
        assert jnp.all(preset.segment_lengths <= bounds.segment_lengths_max)
        assert jnp.all(preset.muscle_pcsa >= bounds.muscle_pcsa_min)
        assert jnp.all(preset.muscle_pcsa <= bounds.muscle_pcsa_max)
        assert jnp.all(
            preset.muscle_moment_arm_magnitudes
            >= bounds.muscle_moment_arm_magnitudes_min
        )
        assert jnp.all(
            preset.muscle_moment_arm_magnitudes
            <= bounds.muscle_moment_arm_magnitudes_max
        )

    def test_different_keys_differ(self, bounds):
        p1 = sample_preset(bounds, jax.random.PRNGKey(0))
        p2 = sample_preset(bounds, jax.random.PRNGKey(1))
        assert not jnp.allclose(p1.segment_lengths, p2.segment_lengths)


class TestFlatConversion:
    def test_to_flat_shape(self, preset):
        flat = to_flat(preset)
        # 4*3 joints + 3*6 muscles + 6*3 magnitudes + 2 tau = 12+18+18+2 = 50
        assert flat.shape == (flat_dim(3, 6),)
        assert flat.shape == (50,)

    def test_round_trip(self, preset):
        flat = to_flat(preset)
        reconstructed = from_flat(flat, n_joints=3, n_muscles=6)
        assert jnp.allclose(preset.segment_lengths, reconstructed.segment_lengths)
        assert jnp.allclose(preset.muscle_pcsa, reconstructed.muscle_pcsa)
        assert jnp.allclose(
            preset.muscle_moment_arm_magnitudes,
            reconstructed.muscle_moment_arm_magnitudes,
        )
        assert reconstructed.tau_act == pytest.approx(preset.tau_act, abs=1e-6)
        assert reconstructed.tau_deact == pytest.approx(preset.tau_deact, abs=1e-6)

    def test_flat_dim(self):
        # 4*n_joints + 3*n_muscles + n_muscles*n_joints + 2
        assert flat_dim(3, 6) == 4 * 3 + 3 * 6 + 6 * 3 + 2  # 50
        assert flat_dim(2, 4) == 4 * 2 + 3 * 4 + 4 * 2 + 2   # 30
        assert flat_dim(2, 6) == 4 * 2 + 3 * 6 + 6 * 2 + 2   # 40

    def test_all_finite(self, preset):
        flat = to_flat(preset)
        assert jnp.all(jnp.isfinite(flat))


class TestBounds:
    def test_default_3link(self):
        bounds = default_3link_bounds()
        assert bounds.segment_lengths_min.shape == (3,)
        assert bounds.muscle_pcsa_max.shape == (6,)
        assert bounds.muscle_moment_arm_magnitudes_min.shape == (6, 3)
        assert bounds.muscle_moment_arm_magnitudes_max.shape == (6, 3)
        assert jnp.all(bounds.segment_lengths_min < bounds.segment_lengths_max)

    def test_default_2link(self):
        bounds = default_2link_bounds()
        assert bounds.segment_lengths_min.shape == (2,)
        assert bounds.muscle_pcsa_max.shape == (6,)
        assert bounds.muscle_moment_arm_magnitudes_min.shape == (6, 2)
        assert bounds.muscle_moment_arm_magnitudes_max.shape == (6, 2)
        assert jnp.all(bounds.segment_lengths_min < bounds.segment_lengths_max)
        # Biarticular muscles (rows 4,5) should have non-zero bounds at both joints.
        assert float(bounds.muscle_moment_arm_magnitudes_min[4, 0]) > 0
        assert float(bounds.muscle_moment_arm_magnitudes_min[4, 1]) > 0
        assert float(bounds.muscle_moment_arm_magnitudes_min[5, 0]) > 0
        assert float(bounds.muscle_moment_arm_magnitudes_min[5, 1]) > 0
