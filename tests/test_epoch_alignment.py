import jax.numpy as jnp
import numpy as np
import pytest

from feedbax.analysis.state_utils import get_align_epoch_start


@pytest.mark.parametrize(
    ("anchor", "expected"),
    [
        ("min", [[10, 11, -9, 13, 14], [22, 23, 24, np.nan, np.nan]]),
        (2, [[np.nan, 10, 11, -9, 13, 14], [21, 22, 23, 24, np.nan, np.nan]]),
        ("max", [[np.nan, np.nan, 10, 11, -9, 13, 14], [20, 21, 22, 23, 24, np.nan, np.nan]]),
        (-2, [[13, 14, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan, np.nan]]),
    ],
    ids=["earliest", "mixed", "latest", "cropped-away"],
)
def test_epoch_alignment_places_samples_and_masks_only_padding(anchor, expected):
    epoch_starts = jnp.array([1, 3])
    # Distinct channels make swapped axes visible. One real sample equals the pad value.
    values = jnp.array([[10, 11, -9, 13, 14], [20, 21, 22, 23, 24]], dtype=jnp.int16)
    states = jnp.stack((values, -values)).transpose(0, 2, 1)  # channel, time, trial

    aligned = get_align_epoch_start(
        epoch_starts, time_axis=1, trial_axis=-1, anchor=anchor, pad_value=-9
    )(states)

    expected = np.array(expected)
    expected = np.stack((expected, -expected)).transpose(0, 2, 1)
    np.testing.assert_array_equal(aligned.mask, ~np.isnan(expected))
    np.testing.assert_array_equal(aligned.data, np.nan_to_num(expected, nan=-9))
    assert aligned.data.dtype == states.dtype
    assert aligned.mask.dtype == jnp.bool_
