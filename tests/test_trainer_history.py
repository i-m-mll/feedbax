"""Regression tests for TaskTrainer history-writing helpers."""

import jax.numpy as jnp
import numpy as np
from jax_cookbook.tree import array_set

from feedbax.objectives.loss import TermTree


def test_term_tree_history_insert_accepts_none_weight_skeleton() -> None:
    """TaskTrainer loss history may store ``None`` metadata before first write."""

    history = TermTree.branch(
        "loss",
        {
            "reach": TermTree.leaf(
                "reach",
                jnp.zeros((3, 2)),
                weight=None,
            )
        },
        weight=None,
    )
    value = TermTree.branch(
        "loss",
        {
            "reach": TermTree.leaf(
                "reach",
                jnp.asarray([4.0, 6.0]),
                weight=1.0,
            )
        },
        weight=1.0,
    )

    updated = array_set(history, value, 1)

    np.testing.assert_allclose(updated["reach"].value[1], jnp.asarray([4.0, 6.0]))
    assert updated.weight == 1.0
    assert updated["reach"].weight == 1.0
