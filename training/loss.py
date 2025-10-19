import jax.numpy as jnp
from feedbax.loss import ModelLoss


def readout_norm_func(weights):
    return jnp.linalg.norm(weights, axis=(-2, -1), ord="fro")


def get_readout_norm_loss(value: float) -> ModelLoss:
    """Returns a loss term that penalizes deviation of the readout norm from `value`."""
    return ModelLoss(
        "fix_readout_norm",
        lambda model: (readout_norm_func(model.step.net.readout.weight) - value) ** 2,
    )
