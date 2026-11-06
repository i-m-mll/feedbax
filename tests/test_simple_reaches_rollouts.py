"""End-to-end rollout and position-loss checks for scalar SimpleReaches."""

from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from feedbax.objectives import CompositeLoss, TargetSpec, make_streaming_loss_fn
from feedbax.runtime.batch import BatchInfo
from feedbax.runtime.graph import Component, Graph
from feedbax.runtime.iteration import run_component
from feedbax.runtime.state import CartesianState
from feedbax.tasks import (
    SimpleReaches,
    TaskTrialSpec,
    TrialSpecDependency,
    WhereDict,
    prepare_trial,
)
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
        task.sample_trial(trial_key, batch)
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



class _DriveRecorder(Component):
    input_ports = ("drive",)
    output_ports = ()
    state_index: eqx.nn.StateIndex
    offset_index: eqx.nn.StateIndex

    def __init__(self):
        self.state_index = eqx.nn.StateIndex(CartesianState())
        self.offset_index = eqx.nn.StateIndex(jnp.zeros(2))

    def __call__(self, inputs, state, *, key):
        effector = state.get(self.state_index)
        pos = inputs["drive"] + state.get(self.offset_index)
        return {}, state.set(self.state_index, replace(effector, pos=pos))


def _select_last_step(trial):
    return jnp.array([trial.timeline.n_steps - 1])


class _TimeMajorTarget(TargetSpec):
    @property
    def batch_axes(self):
        return type(self)(value=1 if self.value is not None else None)


class _ReachesWithNativeDependency(SimpleReaches):
    def _construct_trial_spec(self, effector_init_state, effector_target_state):
        trial = super()._construct_trial_spec(effector_init_state, effector_target_state)
        target = trial.targets[_POSITION]
        value = target.value
        if effector_init_state.pos.ndim > 1:
            value = jnp.swapaxes(value, 0, 1)
        target = _TimeMajorTarget(value, discount=target.discount, time_idxs=_select_last_step)
        return replace(
            trial,
            targets=WhereDict({_POSITION: target}),
            inits=WhereDict({
                **trial.inits,
                "mechanics.effector.offset_index": TrialSpecDependency(
                    lambda trial, key: trial.inputs["effector_target"].pos[0]
                ),
            }),
            extra={
                "offset": TrialSpecDependency(lambda trial, key: trial.inputs["effector_target"].pos[0]),
            },
        )


def test_trial_dependency_follows_canonical_where_selector() -> None:
    class SelectorDependencyReaches(SimpleReaches):
        add_unrelated: bool = eqx.field(static=True, default=False)

        def get_train_trial(self, key, batch_info=None):
            trial = super().get_train_trial(key, batch_info)
            dependency = TrialSpecDependency(lambda _, key: jr.normal(key, (4,)))
            entries = {"state.z": dependency} if self.add_unrelated else {}
            entries[lambda state: state.state.a] = dependency
            return replace(trial, inits=WhereDict(entries))

    task = SelectorDependencyReaches(
        n_steps=5, loss_func=CompositeLoss(()), workspace=((-1.0, -1.0), (1.0, 1.0)),
    )
    key = jr.key(31)
    original = task.sample_trial(key)
    edited = replace(task, add_unrelated=True).sample_trial(key)
    np.testing.assert_array_equal(original.inits["state.a"], edited.inits["state.a"])
    assert not np.array_equal(edited.inits["state.a"], edited.inits["state.z"])


def test_simple_reaches_realizes_added_inputs_per_validation_trial():
    task = _ReachesWithNativeDependency(
        n_steps=5,
        loss_func=CompositeLoss(()),
        workspace=((-1.0, -1.0), (1.0, 1.0)),
        eval_n_directions=7,
        epoch_name="reach",
    )

    def noise(trial, key):
        return jr.normal(key, (trial.timeline.n_steps, 2))

    def drive(trial, key):
        return trial.inputs["z_noise"] + noise(trial, key) + trial.extra["offset"]

    edited_task = task.add_input("a_unrelated", noise)
    task = task.add_input("z_noise", noise).add_input("drive", drive)
    sampled = task.sample_trial(jr.key(0))
    assert sampled.inputs["drive"].shape == (4, 2)
    drive_noise = sampled.inputs["drive"] - sampled.inputs["z_noise"] - sampled.extra["offset"]
    assert not np.allclose(drive_noise, sampled.inputs["z_noise"])

    edited_sample = edited_task.add_input("z_noise", noise).add_input("drive", drive).sample_trial(
        jr.key(0)
    )
    for name in ("z_noise", "drive"):
        np.testing.assert_array_equal(edited_sample.inputs[name], sampled.inputs[name])

    trials = task.validation_trials
    drives = np.asarray(trials.inputs["drive"])
    assert drives.shape == (7, 4, 2)
    assert np.unique(trials.inputs["z_noise"], axis=0).shape[0] == 7
    np.testing.assert_array_equal(drives, replace(task).validation_trials.inputs["drive"])
    np.testing.assert_array_equal(
        trials.extra["offset"], trials.inputs["effector_target"].pos[:, 0]
    )
    drive_noise = drives - trials.inputs["z_noise"] - trials.extra["offset"][:, None, :]
    assert not np.allclose(drive_noise, trials.inputs["z_noise"])
    np.testing.assert_array_equal(
        trials.targets[_POSITION].value, jnp.swapaxes(trials.inputs["effector_target"].pos, 0, 1)
    )
    assert trials.targets[_POSITION].discount.shape == (4,)
    assert trials.targets[_POSITION].time_idxs is _select_last_step
    np.testing.assert_array_equal(trials.timeline.epoch_bounds, np.tile([0, 4], (7, 1)))

    mechanics = Graph(
        nodes={"effector": _DriveRecorder()},
        input_ports=("drive",),
        input_bindings={"drive": ("effector", "drive")},
    )
    model = Graph(
        nodes={"mechanics": mechanics},
        input_ports=("drive",),
        input_bindings={"drive": ("mechanics", "drive")},
    )
    states = task.eval(model, jr.key(1))
    np.testing.assert_allclose(
        states.nodes["mechanics"].nodes["effector"].pos,
        drives + trials.extra["offset"][:, None, :],
    )

    _, _, training_trials = task.eval_train_batch(model, BatchInfo(3, 0, 1), jr.key(2))
    np.testing.assert_array_equal(
        training_trials.targets[_POSITION].value,
        jnp.swapaxes(training_trials.inputs["effector_target"].pos, 0, 1),
    )
