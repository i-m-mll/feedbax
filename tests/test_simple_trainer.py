from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest

from feedbax.training import SimpleTrainer


def test_default_sgd_trains_an_equinox_mlp() -> None:
    model = eqx.nn.MLP(1, 1, width_size=4, depth=1, key=jax.random.key(0))
    x = jnp.array([1.0])
    y = jnp.array([2.0])

    trained = SimpleTrainer()(model, x, y, n_iter=20)

    assert jnp.mean((trained(x) - y) ** 2) < jnp.mean((model(x) - y) ** 2)


class AffineModel(eqx.Module):
    weight: jax.Array
    offset: jax.Array
    activation: Callable[[jax.Array], jax.Array]
    scale: int = eqx.field(static=True)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.activation(self.scale * self.weight * x + self.offset)


def test_adamw_uses_current_weights_and_preserves_nontrainable_values() -> None:
    model = AffineModel(weight=jnp.array(2.0), offset=jnp.array(3), activation=jax.nn.relu, scale=2)
    x = jnp.array([1.0])
    y = jnp.array([10.0])

    def loss(model, x, y):
        return jnp.mean((2 * model(x) - y) ** 2)

    optimizer = optax.adamw(learning_rate=0.1, b1=0.0, b2=0.0, eps=0.0, weight_decay=0.5)
    trained = SimpleTrainer(loss_func=loss, optimizer=optimizer)(
        model, x, y, n_iter=2, progress_bar=False
    )

    # Each gradient is positive, so its normalized value is 1. The first update
    # gives 1.8 and the second gives 1.61 when decay uses the current weight.
    assert float(trained.weight) == pytest.approx(1.61)
    assert trained.offset == model.offset
    assert trained.offset.dtype == model.offset.dtype
    assert trained.activation is model.activation
    assert trained.scale == model.scale
