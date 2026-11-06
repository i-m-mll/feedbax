import jax
import jax.numpy as jnp
import pytest

from feedbax.tasks import TrialTimeline


def test_timeline_queries_preserve_leading_trial_axes():
    timeline = TrialTimeline.from_epochs_events(
        n_steps=4,
        epoch_bounds=jnp.array([[[0, 1, 4], [0, 3, 4]], [[0, 0, 3], [0, 2, 4]]]),
        epoch_names=("hold", "move"),
        event_steps=jnp.array([[[1, 2], [3, 3]], [[0, 1], [2, 1]]]),
        event_names=("go", "cue"),
    )
    steps = jnp.arange(-1, 5)

    def queries(schedule, epoch, step):
        return (
            *schedule.epoch_bounds_by_idx(epoch),
            schedule.epoch_mask_by_idx(epoch),
            schedule.epoch_idx_at(step),
            schedule.epoch_idx_at(steps),
            schedule.event_time("go"),
            *schedule.window_for_event_centered("go", before=1, after=2),
        )

    mapped_queries = jax.vmap(jax.vmap(queries, in_axes=(0, None, None)), in_axes=(0, None, None))
    epoch = jnp.int32(1)
    step = jnp.int32(1)
    actual = jax.jit(queries)(timeline, epoch, step)
    mapped = jax.jit(mapped_queries)(timeline, epoch, step)
    for result, reference in zip(actual, mapped, strict=True):
        assert jnp.array_equal(result, reference)

    _, _, mask, index_at_step, indices, event_time, window_start, window_end = actual
    expected_indices = jnp.array(
        [
            [[-1, 0, 1, 1, 1, -1], [-1, 0, 0, 0, 1, -1]],
            [[-1, 1, 1, 1, -1, -1], [-1, 0, 0, 1, 1, -1]],
        ]
    )
    assert jnp.array_equal(index_at_step, jnp.array([[1, 0], [1, 0]]))
    assert jnp.array_equal(indices, expected_indices)
    assert jnp.array_equal(mask, expected_indices[..., 1:-1] == epoch)
    assert jnp.array_equal(event_time, jnp.array([[1, 3], [0, 2]]))
    assert jnp.array_equal(window_start, jnp.array([[0, 2], [0, 1]]))
    assert jnp.array_equal(window_end, jnp.array([[3, 4], [2, 4]]))

    single = jax.tree.map(lambda field: field[0, 1], timeline)
    assert single.epoch_name_at(3) == "move"
    assert single.events_at(3) == ["go", "cue"]
    assert type(single.event_time("go")) is int
    with pytest.raises(ValueError, match="unbatched epoch bounds"):
        timeline.epoch_name_at(3)
    with pytest.raises(ValueError, match="unbatched event steps"):
        timeline.events_at(3)
