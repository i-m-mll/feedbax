"""Feedbax's supervised gradient training method."""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
from pydantic import Field

from feedbax.compiler.trainability import model_trainability
from feedbax.contracts.base import StrictModel
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.training import OptimizerSpec
from feedbax.training.method_runtime import TrainingContext
from feedbax.training.methods import TrainingMetrics, TrainingState, training_method
from feedbax.training.optimizers import build_step_optimizer

if TYPE_CHECKING:
    from jax import Array
    from optax import OptState

    from feedbax.runtime.graph import Graph


class StandardTrainingSettings(StrictModel):
    optimizer: OptimizerSpec = Field(
        default_factory=lambda: OptimizerSpec(type="adam", params={"learning_rate": 0.001}),
    )
    grad_clip: float | None = Field(default=1.0, gt=0, allow_inf_nan=False)


class _OptimizerState(eqx.Module):
    """Moments may carry across a rate change if their optimizer ABI still agrees."""

    configuration: tuple[str, bytes, float | None] = eqx.field(static=True)
    value: OptState


def _init_supervised(settings: StandardTrainingSettings, model: Graph, _key: Array) -> TrainingState:
    optimizer = build_step_optimizer(settings.optimizer, gradient_clip=settings.grad_clip)
    _, trainable = model_trainability(model)
    config = settings.optimizer
    params = {name: value for name, value in config.params.items() if name != "learning_rate"}
    state = _OptimizerState(
        (config.type.lower(), canonical_json_v2_bytes(params), settings.grad_clip),
        optimizer.init(eqx.filter(model, trainable)),
    )
    return TrainingState(model, state)


@training_method(settings=StandardTrainingSettings, init=_init_supervised, label="Supervised gradient")
def supervised_gradient(
    state: TrainingState, ctx: TrainingContext[StandardTrainingSettings],
) -> tuple[TrainingState, TrainingMetrics]:
    trials, trial_keys = ctx.sample_trials()

    def loss(model: Graph):
        result = ctx.objective(ctx.rollout(model, trials, trial_keys), trials, trial_keys)
        return result.loss, {name: term.value for name, term in result.terms.items()}

    optimizer = ctx.optimizer(ctx.settings.optimizer, gradient_clip=ctx.settings.grad_clip)
    result = ctx.optimizer_step(optimizer, state.model, state.extra.value, loss)
    return TrainingState(result.model, _OptimizerState(state.extra.configuration, result.state)), {
        "train_loss": result.loss,
        "loss_terms": result.aux,
        "grad_norm": result.grad_norm,
    }
