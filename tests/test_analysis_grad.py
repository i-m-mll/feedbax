from __future__ import annotations

from collections import OrderedDict

import jax.numpy as jnp
import jax.random as jr
import pytest

from feedbax.analysis.grad import (
    _compute_grads,
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
