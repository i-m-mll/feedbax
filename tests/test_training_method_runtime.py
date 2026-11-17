"""Behavior of compiled replica training blocks and the explicit schedule clock."""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from pydantic import BaseModel

from feedbax.contracts.training import LrScheduleSpec, OptimizerSpec
from feedbax.plugins.declarations import DistributionSource, callable_declaration
from feedbax.runtime.graph import Graph
from feedbax.training.method_runtime import TrainingLoop, initialize_method, replica_value, stack_replicas
from feedbax.training.methods import TrainingMethodRegistry, TrainingState, training_method
from feedbax.training.optimizers import build_step_optimizer


class _Settings(BaseModel):
    reject_step: int


def _init(settings: _Settings, model, key):
    return TrainingState(model, {"price": jnp.asarray(0.0), "memory": key, "reject": jnp.array(False)})


@training_method(settings=_Settings, init=_init)
def _update(state, ctx):
    extra = state.extra
    reject = (ctx.step == ctx.settings.reject_step) & extra["reject"]
    candidate = {
        **extra,
        "price": jnp.where(reject, jnp.nan, extra["price"] + 1),
        "memory": jr.fold_in(extra["memory"], ctx.step),
    }
    samples = jr.uniform(ctx.key, (2,))
    return TrainingState(state.model, candidate), {"sample": samples}


def test_replica_failure_keeps_the_same_committed_prefix_after_resume():
    registry = TrainingMethodRegistry()
    registry.register_declaration(
        _update, callable_declaration(_update), key="witness.update",
        source=DistributionSource("witness", "example-witness", "1.0"),
    )
    method = registry.resolve("witness.update")
    settings = _Settings(reject_step=2)
    replicas = ("west", "east")
    models = (Graph(), Graph())
    initial = initialize_method(method, settings, models=models, seed=17, replica_ids=replicas)
    rejected = replica_value(initial, 1)
    state = stack_replicas([
        replica_value(initial, 0),
        TrainingState(rejected.model, {**rejected.extra, "reject": jnp.array(True)}),
    ])

    def loop(ids):
        return TrainingLoop(method, settings, execution=None, objective=None, batch_size=1, seed=17, replica_ids=ids)

    executor = loop(replicas)
    full = executor.run_block(state, completed=0, count=5)
    assert full.rejected and full.completed == 2
    np.testing.assert_array_equal(full.state.extra["price"], [2.0, 2.0])
    assert full.metrics["sample"].shape == (2, 2, 2)
    assert not np.array_equal(full.metrics["sample"][:, 0], full.metrics["sample"][:, 1])

    prefix = executor.run_block(state, completed=0, count=1)
    resumed = loop(replicas).run_block(prefix.state, completed=prefix.completed, count=4)
    assert resumed.rejected and resumed.completed == full.completed
    np.testing.assert_array_equal(resumed.state.extra["price"], full.state.extra["price"])
    np.testing.assert_array_equal(jr.key_data(resumed.state.extra["memory"]), jr.key_data(full.state.extra["memory"]))
    np.testing.assert_array_equal(resumed.metrics["sample"], full.metrics["sample"][1:])

    reversed_ids = replicas[::-1]
    reordered = loop(reversed_ids).run_block(
        stack_replicas([replica_value(state, 1), replica_value(state, 0)]), completed=0, count=3,
    )
    np.testing.assert_array_equal(reordered.metrics["sample"][:, 0], full.metrics["sample"][:, 1])


def test_optimizer_rate_follows_explicit_step_after_clock_reset():
    schedule = LrScheduleSpec(
        kind="delayed_cosine", learning_rate_0=0.25,
        constant_lr_iterations=0, total_steps=4, cosine_annealing_alpha=0.0,
    )
    optimizer = build_step_optimizer(OptimizerSpec(type="sgd", lr_schedule=schedule))
    gradient = jnp.asarray(1.0, dtype=jnp.float32)
    restart = optimizer.init(jnp.zeros_like(gradient))

    @jax.jit
    def step(state, index):
        return optimizer.update(gradient, state, schedule_step=index)

    terminal_update, advanced = step(restart, 4)
    reset_update, _ = step(advanced, 0)
    assert float(terminal_update) == 0.0
    assert float(reset_update) == -0.25
