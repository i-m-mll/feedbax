import jax
import jax.numpy as jnp
import numpy as np
import pytest

from feedbax.analysis.eig import eig


@pytest.mark.parametrize(
    ("matrix", "spectrum"),
    [
        (
            [[0.0, -2.0, 1.0], [0.5, 0.0, -0.5], [0.0, 0.0, 0.75]],
            [-1j, 1j, 0.75],
        ),
        (
            [[[1 + 2j, 3j], [0, -0.5j]], [[-2 + 1j, 1 - 2j], [0, 0.25 - 1j]]],
            [[1 + 2j, -0.5j], [-2 + 1j, 0.25 - 1j]],
        ),
    ],
    ids=["real", "complex-batch"],
)
@pytest.mark.parametrize(
    ("flags", "want_left", "want_right"),
    [
        ({}, True, True),
        ({"compute_right_eigenvectors": False}, True, False),
        ({"compute_left_eigenvectors": False}, False, True),
        ({"compute_left_eigenvectors": False, "compute_right_eigenvectors": False}, False, False),
    ],
    ids=["full-default", "left", "right", "values"],
)
def test_eig_returns_spectrum_and_requested_bases(matrix, spectrum, flags, want_left, want_right):
    matrix = jnp.asarray(matrix)
    vals, left, right = jax.jit(lambda arr: eig(arr, **flags))(matrix)
    np.testing.assert_allclose(np.sort_complex(vals), np.sort_complex(spectrum), atol=1e-6)

    if want_left:
        np.testing.assert_allclose(
            matrix.conj().mT @ left, left * vals.conj()[..., None, :], atol=1e-6
        )
        np.testing.assert_allclose(jnp.linalg.norm(left, axis=-2), 1, atol=1e-6)
    else:
        assert left is None

    if want_right:
        np.testing.assert_allclose(matrix @ right, right * vals[..., None, :], atol=1e-6)
        np.testing.assert_allclose(jnp.linalg.norm(right, axis=-2), 1, atol=1e-6)
    else:
        assert right is None
