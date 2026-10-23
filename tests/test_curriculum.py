"""Tests for curriculum learning in feedbax.training.rl.tasks.

Bug: 2055433 -- Progressive target distance for REACH task.
"""

import jax
import jax.numpy as jnp
import pytest

from feedbax.training.rl.tasks import (
    CURRICULUM_STAGES,
    CurriculumState,
    init_curriculum,
    sample_task_params_jax,
    update_curriculum,
)


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


class TestCurriculumState:
    def test_init_state(self):
        """Stage 0 starts at fraction 0.2 with zeroed counters."""
        state = init_curriculum()
        assert int(state.stage) == 0
        assert int(state.success_count) == 0
        assert int(state.total_count) == 0
        assert float(state.max_target_fraction) == pytest.approx(0.2)

    def test_advance_after_window(self):
        """5 consecutive successes advances from stage 0 to stage 1."""
        state = init_curriculum()
        for _ in range(5):
            state = update_curriculum(state, jnp.array(0.90))
        assert int(state.stage) == 1
        assert float(state.max_target_fraction) == pytest.approx(0.35)
        # Counters reset after advancement
        assert int(state.success_count) == 0
        assert int(state.total_count) == 0

    def test_no_advance_without_consecutive(self):
        """A failure mid-window resets the consecutive counter."""
        state = init_curriculum()
        # 3 successes
        for _ in range(3):
            state = update_curriculum(state, jnp.array(0.90))
        assert int(state.success_count) == 3
        # 1 failure resets the streak
        state = update_curriculum(state, jnp.array(0.50))
        assert int(state.success_count) == 0
        assert int(state.stage) == 0
        # 4 more successes still not enough (need 5 consecutive)
        for _ in range(4):
            state = update_curriculum(state, jnp.array(0.90))
        assert int(state.stage) == 0
        assert int(state.success_count) == 4
        # One more pushes over the edge
        state = update_curriculum(state, jnp.array(0.90))
        assert int(state.stage) == 1

    def test_max_stage_clamp(self):
        """Cannot advance beyond the last curriculum stage."""
        n_stages = CURRICULUM_STAGES.shape[0]
        state = init_curriculum()
        # Advance through all stages
        for _ in range(n_stages):
            for _ in range(5):
                state = update_curriculum(state, jnp.array(0.95))
        # Should be clamped at last stage
        assert int(state.stage) == n_stages - 1
        assert float(state.max_target_fraction) == pytest.approx(
            float(CURRICULUM_STAGES[-1]),
        )
        # One more round of successes should not go further
        for _ in range(5):
            state = update_curriculum(state, jnp.array(0.95))
        assert int(state.stage) == n_stages - 1

    def test_exact_threshold(self):
        """Success rate exactly at 0.85 counts as a success."""
        state = init_curriculum()
        for _ in range(5):
            state = update_curriculum(state, jnp.array(0.85))
        assert int(state.stage) == 1

    def test_below_threshold(self):
        """Success rate just below 0.85 does not count."""
        state = init_curriculum()
        for _ in range(10):
            state = update_curriculum(state, jnp.array(0.849))
        assert int(state.stage) == 0


SEGMENT_LENGTHS = jnp.array([0.3, 0.25])


class TestTargetDistanceClamping:
    def test_target_distance_clamped(self, key):
        """With max_target_distance=0.1, sampled reach targets stay within 0.1m."""
        max_dist = 0.1
        keys = jax.random.split(key, 50)
        for k in keys:
            params = sample_task_params_jax(
                k, task_type=0, n_steps=100, dt=0.01,
                segment_lengths=SEGMENT_LENGTHS,
                use_fk=False,
                max_target_distance=max_dist,
                use_curriculum=True,
                single_task=True,
            )
            direction = params.end_pos - params.start_pos
            dist = float(jnp.sqrt(jnp.sum(direction ** 2)))
            assert dist <= max_dist + 1e-6, (
                f"Target distance {dist} exceeds max {max_dist}"
            )

    def test_target_distance_clamped_fk(self, key):
        """FK-based sampling also respects max_target_distance."""
        max_dist = 0.05
        segment_lengths = jnp.array([0.3, 0.25, 0.2])
        keys = jax.random.split(key, 50)
        for k in keys:
            params = sample_task_params_jax(
                k, task_type=0, n_steps=100, dt=0.01,
                segment_lengths=segment_lengths,
                use_fk=True,
                max_target_distance=max_dist,
                use_curriculum=True,
                single_task=True,
            )
            direction = params.end_pos - params.start_pos
            dist = float(jnp.sqrt(jnp.sum(direction ** 2)))
            assert dist <= max_dist + 1e-6, (
                f"Target distance {dist} exceeds max {max_dist}"
            )

    def test_none_max_distance_no_effect(self, key):
        """Without curriculum, max_target_distance has no effect."""
        params_a = sample_task_params_jax(
            key, task_type=0, n_steps=100, dt=0.01,
            segment_lengths=SEGMENT_LENGTHS,
            use_fk=False,
            max_target_distance=10.0,
            use_curriculum=False,
            single_task=True,
        )
        params_b = sample_task_params_jax(
            key, task_type=0, n_steps=100, dt=0.01,
            segment_lengths=SEGMENT_LENGTHS,
            use_fk=False,
            max_target_distance=0.01,
            use_curriculum=False,
            single_task=True,
        )
        assert jnp.allclose(params_a.start_pos, params_b.start_pos)
        assert jnp.allclose(params_a.end_pos, params_b.end_pos)

    def test_short_distance_preserves_direction(self, key):
        """Clamping preserves the direction from start to end."""
        max_dist = 0.05
        # First sample without clamping to get the original direction
        params_unclamped = sample_task_params_jax(
            key, task_type=0, n_steps=100, dt=0.01,
            segment_lengths=SEGMENT_LENGTHS,
            use_fk=False,
            max_target_distance=10.0,
            use_curriculum=False,
            single_task=True,
        )
        params_clamped = sample_task_params_jax(
            key, task_type=0, n_steps=100, dt=0.01,
            segment_lengths=SEGMENT_LENGTHS,
            use_fk=False,
            max_target_distance=max_dist,
            use_curriculum=True,
            single_task=True,
        )
        # Start positions should be identical (clamping only affects end)
        assert jnp.allclose(params_unclamped.start_pos, params_clamped.start_pos)
        # Direction should be preserved (unit vectors should match)
        dir_orig = params_unclamped.end_pos - params_unclamped.start_pos
        dir_clamp = params_clamped.end_pos - params_clamped.start_pos
        norm_orig = jnp.sqrt(jnp.sum(dir_orig ** 2))
        norm_clamp = jnp.sqrt(jnp.sum(dir_clamp ** 2))
        # Only check direction when both are nonzero
        if float(norm_orig) > 1e-6 and float(norm_clamp) > 1e-6:
            unit_orig = dir_orig / norm_orig
            unit_clamp = dir_clamp / norm_clamp
            assert jnp.allclose(unit_orig, unit_clamp, atol=1e-5)


class TestCurriculumVmap:
    def test_vmap_compatible(self):
        """Curriculum state vmaps across bodies (per-body advancement)."""
        n_bodies = 4
        # Create a batch of curriculum states at different stages
        states = jax.vmap(lambda _: init_curriculum())(jnp.arange(n_bodies))
        # Manually advance some bodies to different stages
        # Body 0: 5 successes -> stage 1
        # Body 1: 0 successes -> stage 0
        # Body 2: 5 successes -> stage 1
        # Body 3: 0 successes -> stage 0
        success_rates = jnp.array([0.90, 0.50, 0.90, 0.50])
        for _ in range(5):
            states = jax.vmap(update_curriculum)(states, success_rates)

        # Bodies 0 and 2 should have advanced, 1 and 3 should not
        stages = states.stage
        assert int(stages[0]) == 1
        assert int(stages[1]) == 0
        assert int(stages[2]) == 1
        assert int(stages[3]) == 0

    def test_jit_compatible(self):
        """Curriculum update is JIT-compatible."""
        state = init_curriculum()

        @jax.jit
        def step(s, rate):
            return update_curriculum(s, rate)

        for _ in range(5):
            state = step(state, jnp.array(0.90))
        assert int(state.stage) == 1
