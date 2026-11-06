"""Named weights on retained and streaming composite losses."""

import jax
import jax.numpy as jnp
import pytest

from feedbax.objectives import CompositeLoss, ModelLoss, make_streaming_loss_fn


def test_named_weights_survive_reordering_and_partial_update():
    a = ModelLoss(label="a", loss_fn=jnp.square)
    b = ModelLoss(label="b", loss_fn=lambda model: 10 * jnp.square(model))
    weights = {"b": 3.0, "a": 2.0}
    original = CompositeLoss({"a": a, "b": b}, weights)
    reordered = CompositeLoss({"b": b, "a": a}, weights)
    patched = original.with_weights({"a": 4.0})

    for loss, expected in ((original, (32, 64)), (reordered, (32, 64)), (patched, (34, 68))):
        def retained(model):
            return loss(None, None, model).total

        def streamed(model):
            step_loss = make_streaming_loss_fn(loss, None, model, n_steps=2)
            return step_loss(None, 0) + step_loss(None, 1)

        for evaluate in (retained, streamed):
            value, gradient = jax.value_and_grad(evaluate)(jnp.array(1.0))
            assert (float(value), float(gradient)) == expected


def test_weight_keys_are_bound_before_labeling_and_flattening():
    a = ModelLoss(label="a", loss_fn=jnp.square)
    b = ModelLoss(label="b", loss_fn=jnp.square)
    child = CompositeLoss((a, b), {"b": 3.0, "a": 2.0}, label="inner")
    loss = CompositeLoss(
        {"group": child, "single": a}, {"single": 5.0, "group": 7.0}, user_labels=False
    )

    assert list(loss.flatten_weights().items()) == [
        ("a", 5.0),
        ("inner_a", 14.0),
        ("inner_b", 21.0),
    ]


def test_invalid_named_weights_are_rejected():
    a = ModelLoss(label="a", loss_fn=jnp.square)
    with pytest.raises(ValueError, match=r"Missing: \['a'\]. Unknown: \['typo'\]"):
        CompositeLoss({"a": a}, {"typo": 2.0})
    with pytest.raises(ValueError, match="unique labels"):
        CompositeLoss((a, a), {"a": 2.0})

    positional = CompositeLoss((a, a), (2.0, 3.0))
    assert positional.flatten_weights() == {"a": 2.0, "a_0": 3.0}
    with pytest.raises(ValueError, match=r"Unknown loss weight keys: \['typo'\]"):
        positional.with_weights({"typo": 2.0})
