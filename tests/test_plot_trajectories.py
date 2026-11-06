import json

import jax.numpy as jnp
import numpy as np
import pytest

from feedbax._vendor.jax_cookbook import MaskedArray


def test_trajectories_preserves_masked_gaps():
    pytest.importorskip("polars")
    from feedbax.plot import trajectories

    masked = MaskedArray(
        jnp.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]),
        jnp.array([[[True, True], [False, False], [True, True]]]),
    )
    options = dict(show_mean=False, endpoints="none")
    figure = trajectories([masked], **options)
    unwrapped = trajectories([masked.unwrap()], **options)

    (trace,) = figure.data
    np.testing.assert_array_equal(trace.x, [1.0, np.nan, 5.0])
    np.testing.assert_array_equal(trace.y, [2.0, np.nan, 6.0])
    assert json.loads(figure.to_json())["data"] == json.loads(unwrapped.to_json())["data"]
