import jax
import jax.numpy as jnp
import jax.tree as jt
import jax_cookbook.tree as jtree
import numpy as np

import equinox as eqx
from equinox.nn import StateIndex

from feedbax.runtime.mapping import WhereDict
from feedbax.runtime.graph import (
    Component,
    Graph,
    RolloutStepHookResult,
    Wire,
    init_state_from_component,
)
from feedbax.intervene import InterventionSpec, TimeSeriesParam
from feedbax.objectives.loss import AbstractLoss
from feedbax.compiler.builders import build_component
from feedbax.tasks.presets import delayed_center_out_reaches_params

from feedbax.tasks import (
    AbstractTask,
    DelayedReaches,
    TaskComponent,
    TaskInterventionSpecs,
    TaskTrialSpec,
    TrialSpecDependency,
    eval_ensemble_on_trials,
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


class ValidationRngTask(DummyTask):
    seed_validation: int = 123
    intervention_specs: TaskInterventionSpecs = TaskInterventionSpecs(
        validation={
            "rng": InterventionSpec(
                {
                    "sample": lambda trial_spec, batch_info, *, key: jax.random.uniform(key, ()),
                }
            )
        }
    )

    def get_validation_trials(self, key):
        return TaskTrialSpec(
            inits=WhereDict(),
            targets=WhereDict(),
            inputs=jax.random.uniform(key, (self.n_validation_trials,)),
        )


class _EvalCounter(Component):
    input_ports = ()
    output_ports = ("output",)

    state_index: StateIndex

    def __init__(self):
        self.state_index = StateIndex(jnp.array(0.0, dtype=jnp.float32))

    def __call__(self, inputs, state, *, key):
        count = state.get(self.state_index) + 1.0
        state = state.set(self.state_index, count)
        return {"output": count}, state


class _EvalAccumulator(Component):
    input_ports = ("delta", "base")
    output_ports = ("total",)

    state_index: StateIndex

    def __init__(self):
        self.state_index = StateIndex(jnp.array(0.0, dtype=jnp.float32))

    def __call__(self, inputs, state, *, key):
        total = state.get(self.state_index) + inputs["delta"]
        state = state.set(self.state_index, total)
        return {"total": total + 0.0 * inputs["base"]}, state

    def state_view(self, state):
        return state.get(self.state_index)


class DynamicInputTask(DummyTask):
    def get_train_trial(self, key, batch_info=None):
        return TaskTrialSpec(
            inits=WhereDict(),
            targets=WhereDict(),
            inputs={"delta": jnp.zeros((self.n_steps,), dtype=jnp.float32)},
        )


def _make_dynamic_eval_graph():
    return Graph(
        nodes={"counter": _EvalCounter(), "acc": _EvalAccumulator()},
        wires=(Wire("counter", "output", "acc", "base"),),
        input_ports=("delta",),
        output_ports=("total",),
        input_bindings={"delta": ("acc", "delta")},
        output_bindings={"total": ("acc", "total")},
    )


def _eval_hook(context):
    if context.node_name != "acc":
        return None
    updated = dict(context.port_values)
    updated[(context.node_name, "delta")] = context.port_values[("counter", "output")]
    return RolloutStepHookResult(port_values=updated)


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


def test_validation_trials_use_separate_rng_streams() -> None:
    task = ValidationRngTask()
    key_trials, _, _ = jax.random.split(jax.random.PRNGKey(task.seed_validation), 3)

    trial_specs = task.validation_trials

    assert jnp.allclose(trial_specs.inputs, jax.random.uniform(key_trials, (1,)))
    assert trial_specs.intervene["rng"]["sample"].shape == (1,)
    assert not jnp.allclose(trial_specs.intervene["rng"]["sample"], trial_specs.inputs)


def test_eval_trials_rollout_step_hook_runs_inside_rollout():
    task = DynamicInputTask()
    model = _make_dynamic_eval_graph()
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    trial_specs = eqx.filter_vmap(task.get_train_trial)(keys)

    states_no_hook = task.eval_trials(model, trial_specs, keys)
    states_hook = task.eval_trials(
        model,
        trial_specs,
        keys,
        rollout_step_hook=_eval_hook,
    )

    assert jnp.allclose(states_no_hook.nodes["acc"], jnp.zeros((2, task.n_steps)))
    assert jnp.allclose(
        states_hook.nodes["acc"],
        jnp.array([[1.0, 3.0, 6.0], [1.0, 3.0, 6.0]], dtype=jnp.float32),
    )


def test_public_eval_ensemble_on_trials_matches_private_partition_path():
    task = DynamicInputTask()
    n_replicates = 2
    model = jtree.get_ensemble(
        lambda *, key: _make_dynamic_eval_graph(),
        n=n_replicates,
        key=jax.random.PRNGKey(0),
    )
    trial_specs = TaskTrialSpec(
        inits=WhereDict(),
        targets=WhereDict(),
        inputs={
            "delta": jnp.array(
                [[1.0, 0.0, 2.0], [0.5, 0.5, 0.5]],
                dtype=jnp.float32,
            )
        },
    )
    key = jax.random.PRNGKey(123)
    n_trials = trial_specs.inputs["delta"].shape[0]

    def eval_explicit_trials(model, key, rollout_step_hook=None):
        return task.eval_trials(
            model,
            trial_specs,
            jax.random.split(key, n_trials),
            rollout_step_hook=rollout_step_hook,
        )

    expected = task._eval_ensemble(
        eval_explicit_trials,
        model,
        n_replicates,
        key,
    )
    actual = eval_ensemble_on_trials(
        task,
        model,
        trial_specs,
        key=key,
        n_replicates=n_replicates,
    )

    assert jt.structure(actual) == jt.structure(expected)
    jt.map(
        lambda actual_leaf, expected_leaf: (
            np.testing.assert_array_equal(
                np.asarray(actual_leaf),
                np.asarray(expected_leaf),
            )
            if eqx.is_array(actual_leaf)
            else actual_leaf == expected_leaf
        ),
        actual,
        expected,
    )


def test_trainer_loss_wrapper_rollout_step_hook_runs_inside_rollout():
    from feedbax.training.trainer import grad_wrap_abstract_loss

    task = DynamicInputTask()
    model = _make_dynamic_eval_graph()
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    trial_specs = eqx.filter_vmap(task.get_train_trial)(keys)
    init_states = eqx.filter_vmap(lambda _: init_state_from_component(model))(keys)
    diff_model, static_model = eqx.partition(model, eqx.is_array)

    _, (_, states_no_hook) = grad_wrap_abstract_loss(task.loss_func)(
        diff_model,
        static_model,
        trial_specs,
        init_states,
        keys,
    )
    _, (_, states_hook) = grad_wrap_abstract_loss(
        task.loss_func,
        rollout_step_hook=_eval_hook,
    )(
        diff_model,
        static_model,
        trial_specs,
        init_states,
        keys,
    )

    assert jnp.allclose(states_no_hook.nodes["acc"], jnp.zeros((2, task.n_steps)))
    assert jnp.allclose(
        states_hook.nodes["acc"],
        jnp.array([[1.0, 3.0, 6.0], [1.0, 3.0, 6.0]], dtype=jnp.float32),
    )


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


def test_delayed_center_out_task_spec_materializes_from_compact_params(
    application_registry_bundle,
):
    params = delayed_center_out_reaches_params(
        n_control_stages=8,
        workspace=[[-1.0, -1.0], [1.0, 1.0]],
        epoch_len_ranges=[[2, 2]],
        p_catch_trial=0.0,
    )

    component = build_component(
        "task",
        "DelayedReaches",
        params,
        component_registry=application_registry_bundle.components,
    )
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
