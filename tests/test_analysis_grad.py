from __future__ import annotations

from collections import OrderedDict

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax.nn import softmax

from feedbax.analysis.grad import (
    _compute_grads,
    hessian_block_opnorm_matrix_u,
    hessian_opnorm_all_fn,
    hessian_opnorm_per_fn,
    jacobian_opnorm_all_fn,
    jacobian_opnorm_per_fn,
    reducer_frobenius_hutchinson,
    reducer_trace_hutchinson,
)


def test_compute_grads_preserves_all_argnum_columns() -> None:
    fns = OrderedDict(
        first=lambda x, y: x * y,
        second=lambda x, y: x + 2.0 * y,
    )
    fn_args = (
        OrderedDict(first=jnp.array(3.0), second=jnp.array(5.0)),
        OrderedDict(first=jnp.array(7.0), second=jnp.array(11.0)),
    )

    grads_x, grads_y = _compute_grads(
        lambda func, *args, argnums: tuple(jnp.asarray(value) for value in args),
        fns,
        fn_args,
        argnums=(0, 1),
    )

    assert grads_x["first"] == 3.0
    assert grads_x["second"] == 5.0
    assert grads_y["first"] == 7.0
    assert grads_y["second"] == 11.0


def test_compute_grads_rejects_mismatched_leaf_arity() -> None:
    fns = {"a": lambda x, y: x + y}
    fn_args = ({"a": jnp.array(1.0)}, {"a": jnp.array(2.0)})

    with pytest.raises(ValueError, match="gradient result arity"):
        _compute_grads(
            lambda func, *args, argnums: (jnp.array(1.0),),
            fns,
            fn_args,
            argnums=(0, 1),
        )


def test_hutchinson_reducers_require_explicit_keys() -> None:
    with pytest.raises(TypeError):
        reducer_frobenius_hutchinson()

    with pytest.raises(TypeError):
        reducer_trace_hutchinson()


def test_hutchinson_reducers_are_deterministic_for_same_key() -> None:
    like = jnp.zeros((4,), dtype=jnp.float32)

    def identity(x):
        return x

    frob = reducer_frobenius_hutchinson(key=jr.PRNGKey(0), samples=8)
    trace = reducer_trace_hutchinson(key=jr.PRNGKey(1), samples=8)

    assert frob(identity, like) == reducer_frobenius_hutchinson(key=jr.PRNGKey(0), samples=8)(
        identity, like
    )
    assert trace(identity, like) == reducer_trace_hutchinson(key=jr.PRNGKey(1), samples=8)(
        identity, like
    )


@pytest.mark.parametrize(
    ("func", "jacobian"),
    [
        pytest.param(softmax, [[0.25, -0.25], [-0.25, 0.25]], id="softmax"),
        pytest.param(lambda x: x - x[::-1], [[1.0, -1.0], [-1.0, 1.0]], id="difference"),
        pytest.param(lambda x: jnp.ones_like(x[:1]), [[0.0, 0.0]], id="constant"),
    ],
)
def test_jacobian_opnorm_for_nullspace_cases(func, jacobian) -> None:
    expected = np.linalg.svd(jacobian, compute_uv=False)[0]
    actual = jacobian_opnorm_per_fn(func, (jnp.zeros(2),), 0)

    assert float(actual) == pytest.approx(expected, rel=1e-5, abs=1e-7)


@pytest.mark.parametrize(
    ("scale", "nested_dtype", "rtol"),
    [
        (1.0, jnp.float32, 1e-5),
        (1e-25, jnp.float32, 1e-5),
        (1e25, jnp.float32, 1e-5),
        (1.0, jnp.float16, 5e-4),
    ],
    ids=("ordinary", "tiny", "large", "mixed-dtype"),
)
def test_joint_jacobian_opnorm_for_nested_pytrees(scale, nested_dtype, rtol) -> None:
    jacobian = scale * np.array([[1.5, -0.5], [-0.5, 1.5]])
    matrix = jnp.asarray(jacobian)

    def func(x, y):
        (a,) = x["inputs"]
        out = matrix @ jnp.stack([a, y])
        return {"outputs": (out[0], out[1])}

    expected = np.linalg.svd(jacobian, compute_uv=False)[0]
    args = ({"inputs": (jnp.array(0.0, dtype=nested_dtype),)}, jnp.array(0.0, dtype=jnp.float32))
    actual = jacobian_opnorm_all_fn(func, args, (0, 1))

    assert float(actual) == pytest.approx(expected, rel=rtol, abs=0.0)


def test_hessian_opnorm_for_joint_and_rectangular_blocks() -> None:
    def energy(x, y):
        difference = x[0] - x[1]
        return -0.5 * difference**2 + difference * (y[0] - 2 * y[1] + y[2])

    def vector_energy(x, y):
        value = energy(x, y)
        return jnp.stack([value, 2 * value])

    h_xx = np.array([[-1.0, 1.0], [1.0, -1.0]])
    h_xy = np.array([[1.0, -2.0, 1.0], [-1.0, 2.0, -1.0]])
    h_yy = np.zeros((3, 3))
    hessian = np.block([[h_xx, h_xy], [h_xy.T, h_yy]])
    args = (jnp.zeros(2), jnp.zeros(3))
    idxs = (0, 1)

    joint = hessian_opnorm_all_fn(energy, args, idxs)
    per_arg = hessian_opnorm_per_fn(energy, args, 0)
    blocks = hessian_block_opnorm_matrix_u(vector_energy, args, idxs, u=jnp.array([0.5, 0.25]))
    expected_blocks = [
        [np.linalg.svd(block, compute_uv=False)[0] for block in row]
        for row in ((h_xx, h_xy), (h_xy.T, h_yy))
    ]

    assert float(joint) == pytest.approx(np.linalg.svd(hessian, compute_uv=False)[0], rel=1e-5)
    assert float(per_arg) == pytest.approx(expected_blocks[0][0], rel=1e-5)
    np.testing.assert_allclose(blocks, expected_blocks, rtol=1e-5, atol=1e-7)
