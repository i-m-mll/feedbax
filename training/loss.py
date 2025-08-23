import jax.numpy as jnp


from feedbax.loss import ModelLoss


readout_norm_func = lambda weights: jnp.linalg.norm(weights, axis=(-2, -1), ord='fro')
get_readout_norm_loss = lambda value: ModelLoss(
    "readout_norm",
    lambda model: (readout_norm_func(model.step.net.readout.weight) - value) ** 2
)