from __future__ import annotations

from collections import OrderedDict

import jax.numpy as jnp
import pytest

from feedbax.analysis.grad import _compute_grads


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
