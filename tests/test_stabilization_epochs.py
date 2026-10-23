"""Focused contracts for ``Stabilization`` full-trial epoch naming."""

import jax
import jax.numpy as jnp

from feedbax.objectives.loss import CompositeLoss
from feedbax.tasks import Stabilization


def _task(epoch_name: str | None = None) -> Stabilization:
    return Stabilization(
        n_steps=5,
        loss_func=CompositeLoss(()),
        workspace=jnp.asarray([[-1.0, -0.5], [1.0, 0.75]], dtype=jnp.float32),
        eval_grid_n=2,
        epoch_name=epoch_name,
    )


def test_stabilization_epoch_name_populates_train_and_validation_timelines() -> None:
    task = _task("stabilization")

    train = task.get_train_trial(jax.random.PRNGKey(17))
    validation = task.get_validation_trials(jax.random.PRNGKey(23))

    assert train.timeline.epoch_names == ("stabilization",)
    assert train.timeline.n_steps == task.n_steps - 1
    assert jnp.array_equal(
        train.timeline.epoch_bounds,
        jnp.asarray([0, task.n_steps - 1], dtype=jnp.int32),
    )

    assert validation.timeline.epoch_names == ("stabilization",)
    assert validation.timeline.n_steps == task.n_steps - 1
    assert jnp.array_equal(
        validation.timeline.epoch_bounds,
        jnp.broadcast_to(
            jnp.asarray([0, task.n_steps - 1], dtype=jnp.int32),
            (task.n_validation_trials, 2),
        ),
    )


def test_stabilization_without_epoch_name_preserves_unnamed_timelines() -> None:
    task = _task()

    train = task.get_train_trial(jax.random.PRNGKey(17))
    validation = task.get_validation_trials(jax.random.PRNGKey(23))

    for timeline in (train.timeline, validation.timeline):
        assert timeline.n_steps == task.n_steps
        assert timeline.epoch_bounds is None
        assert timeline.epoch_names == ()
