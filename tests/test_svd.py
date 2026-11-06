import jax
import jax.numpy as jnp
import numpy as np
import pytest

from feedbax.analysis.eig import SVD, DecompPorts
from feedbax.analysis.inputs import Data


@pytest.mark.parametrize("reduced", [False, True], ids=["full-real", "reduced-complex"])
def test_svd_returns_column_bases_for_batched_matrices(reduced):
    rng = np.random.default_rng(0)
    shape = (2, 4, 2, 3) if reduced else (5, 3, 2)
    matrix = rng.normal(size=shape)
    if reduced:
        matrix = matrix + 1j * rng.normal(size=shape)
    matrix = jnp.asarray(matrix)
    batch = shape[:-2]
    m, n = shape[-2:]
    k = min(m, n)

    analysis = SVD(inputs=DecompPorts(matrices=Data.states))
    options = {"full_matrices": False} if reduced else {}
    result = analysis.compute(None, matrices={"leaf": matrix}, metadata="unused", **options)
    vals, left, right = (field["leaf"] for field in result)

    assert vals.shape == (*batch, k)
    assert left.shape == (*batch, m, k if reduced else m)
    assert right.shape == (*batch, n, k if reduced else n)
    np.testing.assert_allclose(
        (left[..., :k] * vals[..., None, :]) @ right[..., :k].conj().mT,
        matrix,
        atol=1e-6,
    )
    for basis in (left, right):
        cols = basis.shape[-1]
        identity = jnp.broadcast_to(jnp.eye(cols), (*batch, cols, cols))
        np.testing.assert_allclose(basis.conj().mT @ basis, identity, atol=1e-6)


def test_svd_values_only_preserves_nested_matrix_positions():
    analysis = SVD(inputs=DecompPorts(matrices=Data.states))
    matrices = {
        "batch": [jnp.asarray([[[0.0, 3.0], [2.0, 0.0]], [[5.0, 0.0], [0.0, 1.0]]])],
        "row": jnp.asarray([[3.0, 4.0]]),
    }

    compute_values = jax.jit(lambda tree: analysis.compute(None, matrices=tree, compute_uv=False))
    vals, left, right = compute_values(matrices)

    np.testing.assert_allclose(vals["batch"][0], [[3, 2], [5, 1]])
    np.testing.assert_allclose(vals["row"], [5])
    assert left == {"batch": [None], "row": None}
    assert right == {"batch": [None], "row": None}
