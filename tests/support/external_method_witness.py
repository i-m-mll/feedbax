"""Training methods installed in a separate distribution by the Host witness."""

import os
from pathlib import Path
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from pydantic import BaseModel, Field

from feedbax.compiler.trainability import model_trainability
from feedbax.training.methods import TrainingState, training_method


class AdversarialSettings(BaseModel):
    learning_rate: float = Field(gt=0)
    limit: float = Field(gt=0)


def _optimizer(settings):
    return optax.with_extra_args_support(optax.sgd(settings.learning_rate, momentum=0.4))


def _check_adversarial_allowed():
    if gate := os.environ.get("FEEDBAX_TEST_ADVERSARIAL_STOP"):
        if Path(gate).with_suffix(".model-only").exists():
            raise AssertionError("A model-only consumer invoked the producer's training method")


def init_adversarial(settings, model, key):
    _check_adversarial_allowed()
    _, mask = model_trainability(model)
    price = jax.random.uniform(key, (), minval=0.1, maxval=0.2)
    return TrainingState(model, {"price": price, "optimizer": _optimizer(settings).init(eqx.filter(model, mask))})


def _wait_for_host_stop(armed: str, step):
    gate = Path(armed)
    if int(step) != 1:
        return
    try:
        gate.unlink()
    except FileNotFoundError:
        return
    gate.with_suffix(".ready").touch()
    stop = Path(os.environ["FEEDBAX_HOST_STOP_FILE"])
    deadline = time.monotonic() + 15
    while not stop.exists():
        if time.monotonic() > deadline:
            raise TimeoutError("Host did not deliver the checkpoint stop")
        time.sleep(.01)


@training_method(settings=AdversarialSettings, init=init_adversarial)
def adversarial(state, ctx):
    _check_adversarial_allowed()
    if armed := os.environ.get("FEEDBAX_TEST_ADVERSARIAL_STOP"):
        jax.debug.callback(lambda step: _wait_for_host_stop(armed, step), ctx.step)

    trials, keys = ctx.sample_trials()
    target = trials.inputs["target_pos"]
    price = state.extra["price"]

    def score(model, replacement):
        outputs = ctx.rollout(model, trials, keys, input_overrides={"target_pos": replacement})
        return ctx.objective(outputs, trials, keys).loss

    def gain(delta):
        return score(state.model, target + delta) - price * jnp.sum(delta ** 2)

    def ascend(_, delta):
        return jnp.clip(delta + 0.5 * jax.grad(gain)(delta), -ctx.settings.limit, ctx.settings.limit)

    delta = jax.lax.fori_loop(0, 2, ascend, jnp.zeros_like(target[0, 0]))

    def loss(model):
        scores = jnp.stack((score(model, target), score(model, target + delta)))
        return jnp.mean(scores), scores

    update = ctx.optimizer_step(_optimizer(ctx.settings), state.model, state.extra["optimizer"], loss)
    price = price + jnp.sum(delta ** 2)
    return TrainingState(update.model, {"price": price, "optimizer": update.state}), {
        "price": price,
        "losses": update.aux,
    }


class ReceiverSettings(BaseModel):
    initial_count: int


def init_receiver(settings, model, key):
    return TrainingState(model, jnp.asarray(settings.initial_count, dtype=jnp.int32))


@training_method(settings=ReceiverSettings, init=init_receiver)
def receiver(state, ctx):
    count = state.extra + 1
    return TrainingState(state.model, count), {"count": count}
