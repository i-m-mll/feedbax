"""End-to-end rollout and position-loss checks for scalar SimpleReaches."""

from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from feedbax.objectives import make_streaming_loss_fn
from feedbax.runtime.batch import BatchInfo
from feedbax.runtime.iteration import run_component
from feedbax.tasks import TaskTrialSpec, WhereDict, prepare_trial
from feedbax.xabdeef import point_mass_nn_simple_reaches


_POSITION = "mechanics.effector.pos"
_POSITION_WEIGHTS = np.array([1 / 729, 64 / 729, 1.0])


@pytest.fixture(scope="module")
def context():
    return point_mass_nn_simple_reaches(n_steps=4, eval_n_directions=2, key=jr.key(0))


def _scalar_validation_trial(trials, i):
    timeline = trials.timeline
    if timeline.has_epochs:
        timeline = replace(timeline, epoch_bounds=timeline.epoch_bounds[i])
    return TaskTrialSpec(
        inits=jax.tree.map(lambda x: x[i], trials.inits),
        inputs=jax.tree.map(lambda x: x[i], trials.inputs),
        targets=WhereDict(),
        timeline=timeline,
    )


def _scalar_history(model, trial, key):
    prepared = prepare_trial(model, trial)
    assert prepared.n_steps == 3
    _, _, history = run_component(
        model,
        prepared.inputs,
        prepared.init_state,
        key=key,
        n_steps=prepared.n_steps,
    )
    return jax.tree.map(lambda x: x[1:], history)


def _assert_histories(states, histories):
    expected = jax.tree.map(lambda *xs: np.stack(xs), *histories)
    actual_leaves, actual_structure = jax.tree.flatten(states)
    expected_leaves, expected_structure = jax.tree.flatten(expected)
    assert actual_structure == expected_structure
    for actual, reference in zip(actual_leaves, expected_leaves, strict=True):
        np.testing.assert_allclose(actual, reference, atol=1e-6, rtol=1e-6)
    assert states.mechanics.effector.pos.shape == (len(histories), 3, 2)


def _assert_position_loss(losses, histories, target_positions):
    positions = np.stack([history.mechanics.effector.pos for history in histories])
    errors = positions - np.asarray(target_positions)
    expected = np.square(errors).sum(axis=-1) @ _POSITION_WEIGHTS
    np.testing.assert_allclose(
        losses["effector_position"].value, expected, atol=1e-6, rtol=1e-6
    )


@pytest.mark.parametrize("epoch_name", [None, "reach"])
def test_simple_reaches_validation_matches_scalar_rollouts(context, epoch_name):
    model = context.model
    task = replace(context.task, epoch_name=epoch_name)
    key = jr.key(1)
    trials = task.validation_trials

    states, losses = task.eval_with_loss(model, key)

    histories = [
        _scalar_history(model, _scalar_validation_trial(trials, i), jr.split(trial_key)[1])
        for i, trial_key in enumerate(jr.split(key, 2))
    ]
    _assert_histories(states, histories)
    _assert_position_loss(losses, histories, trials.inputs["effector_target"].pos)


@pytest.mark.parametrize("epoch_name", [None, "reach"])
def test_simple_reaches_generated_training_batch_matches_scalar_rollouts(context, epoch_name):
    model = context.model
    task = replace(context.task, epoch_name=epoch_name)
    batch = BatchInfo(size=2, current=0, total=1)
    key = jr.key(2)

    states, losses, _ = task.eval_train_batch(model, batch, key)

    generation_key, evaluation_key = jr.split(key)
    trials = [
        task.get_train_trial_with_intervenor_params(trial_key, batch)
        for trial_key in jr.split(generation_key, batch.size)
    ]
    histories = [
        _scalar_history(model, trial, jr.split(trial_key)[1])
        for trial, trial_key in zip(trials, jr.split(evaluation_key, batch.size), strict=True)
    ]
    _assert_histories(states, histories)
    _assert_position_loss(
        losses, histories, np.stack([trial.inputs["effector_target"].pos for trial in trials])
    )


def test_simple_reaches_shared_discount_weights_each_control_step(context):
    model, task = context.model, context.task
    trials = task.validation_trials
    trial_key = jr.key(3)
    history = _scalar_history(model, _scalar_validation_trial(trials, 0), jr.split(trial_key)[1])
    positions = history.mechanics.effector.pos

    # Repeat one physical trial three times. Give each scoring target distinct,
    # nonzero errors so both an axis swap and any missing time sample change the loss.
    offsets = jnp.array([[1.0, 2.0, 3.0], [4.0, 1.0, 2.0], [2.0, 3.0, 1.0]])
    targets = positions + offsets[..., None] * jnp.array([1.0, -1.0])
    trials = replace(
        trials,
        inits=jax.tree.map(lambda x: jnp.repeat(x[:1], 3, axis=0), trials.inits),
        inputs=jax.tree.map(lambda x: jnp.repeat(x[:1], 3, axis=0), trials.inputs),
        targets=eqx.tree_at(lambda x: x[_POSITION].value, trials.targets, targets),
    )

    states, losses = task.eval_trials_with_loss(model, trials, jnp.stack([trial_key] * 3))

    _assert_histories(states, [history] * 3)
    expected = 2 * np.square(offsets) @ _POSITION_WEIGHTS
    np.testing.assert_allclose(
        losses["effector_position"].value, expected, atol=1e-6, rtol=1e-6
    )

    position = task.loss_func.terms["effector_position"]
    scalar = replace(
        _scalar_validation_trial(trials, 0),
        targets=WhereDict({position.where: replace(trials.targets[_POSITION], value=targets[0])}),
    )
    prepared = prepare_trial(model, scalar)
    _, _, streamed_loss = run_component(
        model,
        prepared.inputs,
        prepared.init_state,
        key=jr.split(trial_key)[1],
        n_steps=prepared.n_steps,
        streaming_loss_fn=make_streaming_loss_fn(position, scalar, model, prepared.n_steps),
    )
    np.testing.assert_allclose(streamed_loss, expected[0], atol=1e-6, rtol=1e-6)
