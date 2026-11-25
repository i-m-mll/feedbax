"""Feedbax's supervised gradient training method."""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
from pydantic import Field

from feedbax.compiler.trainability import model_trainability
from feedbax.contracts.base import StrictModel
from feedbax.training.method_runtime import TrainingContext
from feedbax.training.methods import TrainingMetrics, TrainingState, training_method
from feedbax.training.optimizers import DEFAULT_OPTIMIZER, OptimizerSpec, build_step_optimizer, optimizer_configuration

if TYPE_CHECKING:
    from jax import Array
    from optax import OptState

    from feedbax.runtime.graph import Graph


class StandardTrainingSettings(StrictModel):
    optimizer: OptimizerSpec = Field(default=DEFAULT_OPTIMIZER)
    grad_clip: float | None = Field(default=1.0, gt=0, allow_inf_nan=False)


class _OptimizerState(eqx.Module):
    """Moments may carry across a rate change if their optimizer ABI still agrees."""

    configuration: tuple[str, bytes, float | None] = eqx.field(static=True)
    value: OptState


def _init_supervised(settings: StandardTrainingSettings, model: Graph, _key: Array) -> TrainingState:
    optimizer = build_step_optimizer(settings.optimizer, gradient_clip=settings.grad_clip)
    _, trainable = model_trainability(model)
    state = _OptimizerState(
        (*optimizer_configuration(settings.optimizer), settings.grad_clip),
        optimizer.init(eqx.filter(model, trainable)),
    )
    return TrainingState(model, state)


@training_method(
    settings=StandardTrainingSettings, init=_init_supervised, label="Supervised gradient",
    metrics=("train_loss", "grad_norm"),
)
def supervised_gradient(
    state: TrainingState, ctx: TrainingContext[StandardTrainingSettings],
) -> tuple[TrainingState, TrainingMetrics]:
    trials, trial_keys = ctx.sample_trials()

    def loss(model: Graph):
        result = ctx.score(model, trials, trial_keys)
        return result.loss, {name: term.value for name, term in result.terms.items()}

    optimizer = ctx.optimizer(ctx.settings.optimizer, gradient_clip=ctx.settings.grad_clip)
    result = ctx.optimizer_step(optimizer, state.model, state.extra.value, loss)
    return TrainingState(result.model, _OptimizerState(state.extra.configuration, result.state)), {
        "train_loss": result.loss,
        "loss_terms": result.aux,
        "grad_norm": result.grad_norm,
    }
