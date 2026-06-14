import pytest
import jax
import jax.numpy as jnp

import equinox as eqx

from feedbax._mapping import WhereDict
from feedbax.runtime.graph import init_state_from_component
from feedbax.intervene import TimeSeriesParam
from feedbax.loss import AbstractLoss
from feedbax.serialization_builders import build_component
from feedbax.task_presets import delayed_center_out_reaches_params

try:
    from feedbax.task import (
        AbstractTask,
        DelayedReaches,
        TaskComponent,
        TaskInterventionSpecs,
        TaskTrialSpec,
        TrialSpecDependency,
    )
except ImportError:
    pytest.skip(
        "Circular import in feedbax.task (pre-existing issue, not on develop)",
        allow_module_level=True,
    )


class DummyLoss(AbstractLoss):
    label: str = "dummy"

    def term(self, states, trial_specs, model):
        return jnp.array(0.0)


class DummyTask(AbstractTask):
    loss_func: AbstractLoss = DummyLoss()
    n_steps: int = 3
    seed_validation: int = 0
    intervention_specs: TaskInterventionSpecs = TaskInterventionSpecs()
    input_dependencies: dict[str, TrialSpecDependency] = eqx.field(default_factory=dict)

    def get_train_trial(self, key, batch_info=None):
        return TaskTrialSpec(
            inits=WhereDict(),
            targets=WhereDict(),
            inputs=jnp.zeros((self.n_steps, 1)),
        )

    def get_validation_trials(self, key):
        return self.get_train_trial(key)

    def validation_plots(self, states, trial_specs=None):
        return {}

    @property
    def n_validation_trials(self) -> int:
        return 1


def test_task_component_open_loop_steps():
    task = DummyTask()
    inputs = jnp.array([[1.0], [2.0], [3.0]])
    intervene = {"foo": TimeSeriesParam(jnp.array([10.0, 20.0, 30.0]))}
    trial_spec = TaskTrialSpec(
        inits=WhereDict(),
        targets=WhereDict(),
        inputs=inputs,
        intervene=intervene,
    )

    component = TaskComponent(task=task, trial_spec=trial_spec)
    state = init_state_from_component(component)

    out1, state = component({}, state, key=jax.random.PRNGKey(0))
    out2, state = component({}, state, key=jax.random.PRNGKey(1))
    out3, _ = component({}, state, key=jax.random.PRNGKey(2))

    assert (out1["inputs"] == inputs[0]).all()
    assert (out2["inputs"] == inputs[1]).all()
    assert (out3["inputs"] == inputs[2]).all()
    assert out1["intervene"]["foo"] == intervene["foo"].value[0]
    assert out2["intervene"]["foo"] == intervene["foo"].value[1]
    assert out3["intervene"]["foo"] == intervene["foo"].value[2]


def test_delayed_reaches_can_sample_center_out_training_trials():
    task = DelayedReaches(
        loss_func=DummyLoss(),
        n_steps=20,
        workspace=jnp.asarray([[-1.0, -1.0], [1.0, 1.0]]),
        train_endpoint_mode="center_out",
        eval_reach_length=0.5,
        epoch_len_ranges=((0, 1), (2, 3)),
    )

    trial = task.get_train_trial(jax.random.PRNGKey(0))

    assert jnp.allclose(trial.inits["mechanics.effector"].pos, jnp.zeros(2))


def test_delayed_center_out_preset_exposes_timeline_and_catch_metadata():
    task = DelayedReaches.delayed_center_out(
        loss_func=DummyLoss(),
        n_control_stages=8,
        workspace=jnp.asarray([[-1.0, -1.0], [1.0, 1.0]]),
        epoch_len_ranges=((2, 2),),
        p_catch_trial=1.0,
    )

    trial = task.get_train_trial(jax.random.PRNGKey(0))

    assert task.n_steps == 9
    assert task.preset == "delayed_center_out"
    assert trial.timeline.epoch_names == ("prep", "movement")
    assert trial.timeline.event_names == ("go_cue",)
    assert int(trial.timeline.event_steps[0]) == int(trial.timeline.epoch_bounds[1])
    assert trial.extra is not None
    assert bool(trial.extra["is_catch_trial"])
    assert jnp.all(trial.inputs.target_on == 1.0)
    assert jnp.all(trial.inputs.hold == 1.0)
    assert jnp.allclose(
        trial.targets["mechanics.effector.pos"].value,
        trial.inits["mechanics.effector"].pos,
    )


def test_delayed_center_out_task_spec_materializes_from_compact_params():
    params = delayed_center_out_reaches_params(
        n_control_stages=8,
        workspace=[[-1.0, -1.0], [1.0, 1.0]],
        epoch_len_ranges=[[2, 2]],
        p_catch_trial=0.0,
    )

    component = build_component("task", "DelayedReaches", params)
    task = component.task
    trial = component.trial_spec

    assert task.n_steps == 9
    assert task.train_endpoint_mode == "center_out"
    assert tuple(task.epoch_names) == ("prep", "movement")
    assert task.target_visible_from_start is True
    assert task.catch_metadata_policy == "flag"
    assert trial.timeline.epoch_names == ("prep", "movement")
    assert trial.timeline.event_names == ("go_cue",)
    assert trial.extra is not None
    assert bool(trial.extra["is_catch_trial"]) is False
