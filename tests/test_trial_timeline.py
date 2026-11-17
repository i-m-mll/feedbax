import jax
import jax.numpy as jnp

from feedbax.runtime.epoch_windows import EpochRecord, whole_window


def test_epoch_windows_preserve_leading_trial_axes():
    starts = jnp.array([[[0, 1], [0, 3]], [[0, 0], [0, 2]]])
    ends = jnp.array([[[1, 4], [3, 4]], [[0, 3], [2, 4]]])
    epochs = EpochRecord(starts=starts, ends=ends, outcomes=jnp.zeros_like(starts))

    def movement(record):
        return whole_window(record, 4, (1,))

    expected = jnp.array([
        [[False, True, True, True], [False, False, False, True]],
        [[True, True, True, False], [False, False, True, True]],
    ])
    actual = jax.jit(movement)(epochs)
    assert jnp.array_equal(actual, expected)
    assert jnp.array_equal(actual, jax.vmap(jax.vmap(movement))(epochs))
