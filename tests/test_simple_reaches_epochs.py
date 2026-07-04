"""Tests for ``SimpleReaches`` epoch and deterministic endpoint behavior."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from feedbax.objectives.loss import CompositeLoss, EpochMaskedLoss, StateDerivativeLoss
from feedbax.tasks import SimpleReaches, TaskTrialSpec
from feedbax.tasks.timeline_masks import build_task_timeline_mask


_WORKSPACE = jnp.asarray([[-1.0, -0.5], [1.0, 0.75]], dtype=jnp.float32)


_DEFAULT_GOLDEN = {
    "train": {
        "init_force": [0.0, 0.0],
        "init_pos": [-0.49553704261779785, 0.6044191122055054],
        "init_vel": [0.0, 0.0],
        "input_force": None,
        "input_pos": [
            [0.5312545299530029, -0.03461199998855591],
            [0.5312545299530029, -0.03461199998855591],
            [0.5312545299530029, -0.03461199998855591],
            [0.5312545299530029, -0.03461199998855591],
        ],
        "input_vel": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        "target_discount": [0.000244140625, 0.015625, 0.177978515625, 1.0],
        "target_value": [
            [0.5312545299530029, -0.03461199998855591],
            [0.5312545299530029, -0.03461199998855591],
            [0.5312545299530029, -0.03461199998855591],
            [0.5312545299530029, -0.03461199998855591],
        ],
        "timeline_epoch_bounds": None,
        "timeline_epoch_names": [],
        "timeline_event_names": [],
        "timeline_event_steps": None,
        "timeline_n_steps": 5,
    },
    "validation": {
        "init_force": [[0.0, 0.0], [0.0, 0.0]],
        "init_pos": [[0.0, 0.125], [0.0, 0.125]],
        "init_vel": [[0.0, 0.0], [0.0, 0.0]],
        "input_force": None,
        "input_pos": [
            [[0.25, 0.125], [0.25, 0.125], [0.25, 0.125], [0.25, 0.125]],
            [
                [-0.25, 0.12499997764825821],
                [-0.25, 0.12499997764825821],
                [-0.25, 0.12499997764825821],
                [-0.25, 0.12499997764825821],
            ],
        ],
        "input_vel": [
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        ],
        "target_discount": [0.000244140625, 0.015625, 0.177978515625, 1.0],
        "target_value": [
            [[0.25, 0.125], [0.25, 0.125], [0.25, 0.125], [0.25, 0.125]],
            [
                [-0.25, 0.12499997764825821],
                [-0.25, 0.12499997764825821],
                [-0.25, 0.12499997764825821],
                [-0.25, 0.12499997764825821],
            ],
        ],
        "timeline_epoch_bounds": None,
        "timeline_epoch_names": [],
        "timeline_event_names": [],
        "timeline_event_steps": None,
        "timeline_n_steps": 5,
    },
}


class _FakeEffector(eqx.Module):
    pos: jnp.ndarray


class _FakeNet(eqx.Module):
    hidden: jnp.ndarray


class _FakeMechanics(eqx.Module):
    effector: _FakeEffector


class _FakeStates(eqx.Module):
    net: _FakeNet
    mechanics: _FakeMechanics


def _make_default_task() -> SimpleReaches:
    return SimpleReaches(
        n_steps=5,
        loss_func=CompositeLoss(()),
        workspace=_WORKSPACE,
        eval_grid_n=1,
        eval_n_directions=2,
        eval_reach_length=0.25,
    )


def _make_epoch_task(**overrides: Any) -> SimpleReaches:
    params = {
        "n_steps": 5,
        "loss_func": CompositeLoss(()),
        "workspace": _WORKSPACE,
        "eval_grid_n": 1,
        "eval_n_directions": 2,
        "eval_reach_length": 0.25,
        "epoch_name": "movement",
    }
    params.update(overrides)
    return SimpleReaches(**params)


def _first_value(mapping: Mapping[Any, Any]) -> Any:
    return next(iter(mapping.values()))


def _snapshot(spec: TaskTrialSpec) -> dict[str, Any]:
    init_state = _first_value(spec.inits)
    target_spec = _first_value(spec.targets)
    inputs = spec.inputs["effector_target"]
    timeline = spec.timeline
    return {
        "init_pos": init_state.pos,
        "init_vel": init_state.vel,
        "init_force": init_state.force,
        "input_pos": inputs.pos,
        "input_vel": inputs.vel,
        "input_force": inputs.force,
        "target_value": target_spec.value,
        "target_discount": target_spec.discount,
        "timeline_n_steps": timeline.n_steps,
        "timeline_epoch_bounds": timeline.epoch_bounds,
        "timeline_epoch_names": list(timeline.epoch_names),
        "timeline_event_steps": timeline.event_steps,
        "timeline_event_names": list(timeline.event_names),
    }


def _timeline_mask_spec(spec: TaskTrialSpec) -> dict[str, Any]:
    timeline = spec.timeline
    bounds = jnp.asarray(timeline.epoch_bounds)
    if bounds.ndim == 2:
        bounds = bounds[0]
    bounds_list = [int(value) for value in bounds.tolist()]
    epochs = []
    for index, name in enumerate(timeline.epoch_names):
        epochs.append(
            {
                "id": f"epoch:{index}",
                "label": name,
                "index": index,
                "length": {
                    "schema_version": "feedbax.spec.studio.value.v1",
                    "mode": "constant",
                    "value": {"steps": bounds_list[index + 1] - bounds_list[index]},
                    "units": "steps",
                    "metadata": {},
                },
                "metadata": {},
            }
        )
    return {
        "timeline": {
            "schema_version": "feedbax.spec.studio.task_timeline.v1",
            "epochs": epochs,
            "segments": [],
            "metadata": {"source": "simple_reaches_trial_timeline"},
        }
    }


def _states_matching_targets(spec: TaskTrialSpec) -> _FakeStates:
    init_state = _first_value(spec.inits)
    target_spec = _first_value(spec.targets)
    target_value = target_spec.value
    if init_state.pos.ndim == 1:
        pos = jnp.concatenate([init_state.pos[None, :], target_value], axis=0)
        pos = pos[None, ...]
    else:
        pos = jnp.concatenate([init_state.pos[:, None, :], target_value], axis=1)
    hidden = jnp.arange(pos.shape[1], dtype=jnp.float32)[None, :, None]
    hidden = jnp.broadcast_to(hidden, (pos.shape[0], pos.shape[1], 2))
    return _FakeStates(
        net=_FakeNet(hidden=hidden),
        mechanics=_FakeMechanics(effector=_FakeEffector(pos=pos)),
    )


def _assert_snapshot_equal(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    assert actual.keys() == expected.keys()
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if isinstance(expected_value, list):
            assert jnp.array_equal(jnp.asarray(actual_value), jnp.asarray(expected_value))
        else:
            assert actual_value == expected_value


def test_simple_reaches_defaults_match_prechange_golden() -> None:
    """Default train and validation specs preserve the pre-change output."""
    task = _make_default_task()

    train = task.get_train_trial(jax.random.PRNGKey(17))
    validation = task.get_validation_trials(jax.random.PRNGKey(23))

    _assert_snapshot_equal(_snapshot(train), _DEFAULT_GOLDEN["train"])
    _assert_snapshot_equal(_snapshot(validation), _DEFAULT_GOLDEN["validation"])


def test_simple_reaches_epoch_name_populates_target_length_bounds() -> None:
    task = _make_epoch_task()

    train = task.get_train_trial(jax.random.PRNGKey(17))
    validation = task.get_validation_trials(jax.random.PRNGKey(23))
    train_target = _first_value(train.targets)
    validation_target = _first_value(validation.targets)

    assert train.timeline.epoch_names == ("movement",)
    assert train.timeline.n_steps == task.n_steps - 1
    assert jnp.array_equal(train.timeline.epoch_bounds, jnp.asarray([0, 4], dtype=jnp.int32))
    assert train_target.discount.shape == (4,)

    assert validation.timeline.epoch_names == ("movement",)
    assert validation.timeline.n_steps == task.n_steps - 1
    assert jnp.array_equal(
        validation.timeline.epoch_bounds,
        jnp.asarray([[0, 4], [0, 4]], dtype=jnp.int32),
    )
    assert validation_target.discount.shape == (2, 4)
    assert jnp.array_equal(
        validation_target.discount,
        jnp.broadcast_to(train_target.discount, (2, 4)),
    )


def test_simple_reaches_epoch_consumers_accept_target_length_epoch() -> None:
    task = _make_epoch_task()
    validation = task.get_validation_trials(jax.random.PRNGKey(23))
    states = _states_matching_targets(validation)

    base = StateDerivativeLoss(label="net_hidden_derivative")
    wrapped = EpochMaskedLoss(
        label="net_hidden_derivative_movement",
        base_loss=base,
        epoch_indices=(0,),
    )

    base_value = base(states, trial_specs=validation, model=None).value
    wrapped_value = wrapped(states, trial_specs=validation, model=None).value

    assert jnp.allclose(wrapped_value, base_value)

    timeline_mask = build_task_timeline_mask(
        _timeline_mask_spec(validation),
        segment_name="movement",
        n_steps=task.n_steps - 1,
    )
    assert jnp.array_equal(timeline_mask.mask, jnp.ones((task.n_steps - 1,), dtype=bool))


def test_simple_reaches_fixed_endpoints_drive_train_and_single_validation_trial() -> None:
    fixed_endpoints = jnp.asarray([[0.1, -0.2], [0.4, 0.6]], dtype=jnp.float32)
    task = _make_epoch_task(fixed_endpoints=fixed_endpoints)

    train_a = task.get_train_trial(jax.random.PRNGKey(1))
    train_b = task.get_train_trial(jax.random.PRNGKey(2))
    validation = task.get_validation_trials(jax.random.PRNGKey(3))
    train_init = _first_value(train_a.inits)
    train_target = _first_value(train_a.targets)
    validation_init = _first_value(validation.inits)
    validation_target = _first_value(validation.targets)

    assert task.n_validation_trials == 1
    assert jnp.array_equal(_snapshot(train_a)["init_pos"], _snapshot(train_b)["init_pos"])
    assert jnp.array_equal(_snapshot(train_a)["target_value"], _snapshot(train_b)["target_value"])

    assert jnp.array_equal(train_init.pos, fixed_endpoints[0])
    assert jnp.array_equal(
        train_target.value,
        jnp.broadcast_to(fixed_endpoints[1], (task.n_steps - 1, 2)),
    )
    assert train_a.timeline.n_steps == task.n_steps - 1
    assert jnp.array_equal(
        train_a.timeline.epoch_bounds,
        jnp.asarray([0, task.n_steps - 1], dtype=jnp.int32),
    )

    assert jnp.array_equal(validation_init.pos, fixed_endpoints[None, 0])
    assert jnp.array_equal(
        validation_target.value,
        jnp.broadcast_to(fixed_endpoints[None, 1], (1, task.n_steps - 1, 2)),
    )
    assert validation_target.discount.shape == (1, task.n_steps - 1)
    assert validation.timeline.n_steps == task.n_steps - 1
    assert jnp.array_equal(
        validation.timeline.epoch_bounds,
        jnp.asarray([[0, task.n_steps - 1]], dtype=jnp.int32),
    )
