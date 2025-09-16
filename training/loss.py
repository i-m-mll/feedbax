import equinox as eqx
import jax.numpy as jnp
from feedbax.loss import ModelLoss, StopAtGoalLoss
from feedbax.xabdeef.losses import simple_reach_loss

from feedbax_experiments.types import TreeNamespace


def readout_norm_func(weights):
    return jnp.linalg.norm(weights, axis=(-2, -1), ord="fro")


def get_readout_norm_loss(value: float) -> ModelLoss:
    """Returns a loss term that penalizes deviation of the readout norm from `value`."""
    return ModelLoss(
        "readout_norm",
        lambda model: (readout_norm_func(model.step.net.readout.weight) - value) ** 2,
    )


#! TODO: This should be project-specific!
#! This would be easy enough to do, by constructing the loss function only in the
#! `setup_task_model_pair` function of a project-specific module. However, we also
#! need to construct the loss function in `post_training` in `setup_train_histories`.
#! So, we need to use the same
def get_reach_loss(hps: TreeNamespace):
    """Get loss function for simple reaching with some optional extras."""
    loss_fn = simple_reach_loss()

    if not getattr(hps, "loss", False):
        return loss_fn

    if getattr(hps.loss, "stop_at_goal", False):
        loss_fn = loss_fn + StopAtGoalLoss(**hps.loss.stop_at_goal)

    if getattr(hps.loss, "fix_readout_norm", False):
        loss_fn = loss_fn + get_readout_norm_loss(**hps.loss.fix_readout_norm)

    if getattr(hps.loss, "weights", False):
        loss_fn = eqx.tree_at(
            lambda loss_fn: loss_fn.weights,
            loss_fn,
            {**loss_fn.weights, **hps.loss.weights},
        )

    return loss_fn
