import jax
import jax.numpy as jnp
import jax.tree as jt
import feedbax._vendor.jax_cookbook.tree as jtree
import numpy as np
import pytest

import equinox as eqx
from equinox.nn import StateIndex

from feedbax.runtime.batch import BatchInfo
from feedbax.runtime.mapping import WhereDict
from feedbax.runtime.graph import (
    Component,
    Graph,
    RolloutStepHookResult,
    Wire,
    init_state_from_component,
)
from feedbax.objectives.loss import AbstractLoss
from feedbax.compiler.builders import build_component
from feedbax.component_registry.retained import retained_component_registry
from feedbax.tasks.presets import delayed_center_out_reaches_params

from feedbax.tasks import (
    AbstractTask,
    DelayedReaches,
    TaskComponent,
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
    input_dependencies: dict[str, TrialSpecDependency] = eqx.field(
        default_factory=lambda: {
            "rng": TrialSpecDependency(lambda trial, key: jax.random.uniform(key, ())),
        },
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


def _make_dynamic_eval_graph(accumulator=None):
    if accumulator is None:
        accumulator = _EvalAccumulator()
    return Graph(
        nodes={"counter": _EvalCounter(), "acc": accumulator},
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
    trial_spec = TaskTrialSpec(
        inits=WhereDict(),
        targets=WhereDict(),
        inputs={"observation": inputs, "force": jnp.array([10.0, 20.0, 30.0])},
    )

    component = TaskComponent(task=task, trial_spec=trial_spec)
    state = init_state_from_component(component)

    out1, state = component({}, state, key=jax.random.PRNGKey(0))
    out2, state = component({}, state, key=jax.random.PRNGKey(1))
    out3, _ = component({}, state, key=jax.random.PRNGKey(2))

    assert (out1["inputs"]["observation"] == inputs[0]).all()
    assert (out2["inputs"]["observation"] == inputs[1]).all()
    assert (out3["inputs"]["observation"] == inputs[2]).all()
    assert out1["inputs"]["force"] == 10.0
    assert out2["inputs"]["force"] == 20.0
    assert out3["inputs"]["force"] == 30.0


def test_validation_trials_use_separate_rng_streams() -> None:
    task = ValidationRngTask()
    key_trials, _, _ = jax.random.split(jax.random.PRNGKey(task.seed_validation), 3)

    trial_specs = task.validation_trials

    assert jnp.allclose(trial_specs.inputs["task"], jax.random.uniform(key_trials, (1,)))
    assert trial_specs.inputs["rng"].shape == (1,)
    assert not jnp.allclose(trial_specs.inputs["rng"], trial_specs.inputs["task"])


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


class _EnsembleAccumulator(Component):
    input_ports = ("delta", "base")
    output_ports = ("total",)

    gain: jax.Array
    state_index: StateIndex

    def __call__(self, inputs, state, *, key):
        update = jnp.stack(
            (
                self.gain * inputs["base"],
                inputs["delta"] + jax.random.uniform(key),
            )
        )
        total = state.get(self.state_index) + update
        return {"total": total}, state.set(self.state_index, total)

    def state_view(self, state):
        return state.get(self.state_index)


class _EnsembleLoss(AbstractLoss):
    label: str = "ensemble"

    def term(self, states, trial_specs, model):
        return states.nodes["acc"][..., 0]


class _EnsembleTask(DynamicInputTask):
    loss_func: AbstractLoss = _EnsembleLoss()

    def get_train_trial(self, key, batch_info=None):
        return TaskTrialSpec(
            inits=WhereDict(),
            targets=WhereDict(),
            inputs={"delta": jax.random.uniform(key, (self.n_steps,))},
        )


def _ensemble_graphs(n_replicates, *, map_initial_state):
    base = _make_dynamic_eval_graph(
        _EnsembleAccumulator(jnp.array(1.0), StateIndex(jnp.array([4.0, -2.0])))
    ).prepare()

    def gain(graph):
        return graph.nodes["acc"].gain

    def initializer(graph):
        return graph.nodes["acc"].state_index.init

    def build(*, key):
        gain_key, init_key = jax.random.split(key)
        graph = eqx.tree_at(gain, base, jax.random.uniform(gain_key))
        if map_initial_state:
            graph = eqx.tree_at(
                initializer, graph, initializer(base) + jax.random.uniform(init_key, (2,))
            )
        return graph

    axes = jt.map(lambda _: None, base)
    axes = eqx.tree_at(
        (lambda graph: (gain(graph), initializer(graph))) if map_initial_state else gain,
        axes,
        (0, 1) if map_initial_state else 0,
        is_leaf=lambda x: x is None,
    )
    key = jax.random.PRNGKey(0)
    models = jtree.get_ensemble(
        build,
        n=n_replicates,
        key=key,
        out_axes=axes,
    )
    scalar_models = [
        build(key=replicate_key) for replicate_key in jax.random.split(key, n_replicates)
    ]
    return models, axes, scalar_models


@pytest.mark.parametrize("n_replicates", [2, 3])
def test_public_eval_ensemble_on_trials_preserves_shared_state_initializer(n_replicates):
    task = _EnsembleTask()
    models, axes, scalar_models = _ensemble_graphs(n_replicates, map_initial_state=False)
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
    actual = eval_ensemble_on_trials(
        task,
        models,
        trial_specs,
        key=key,
        n_replicates=n_replicates,
        model_axes=axes,
    )

    expected = [
        task.eval_trials(model, trial_specs, jax.random.split(replicate_key, n_trials))
        for model, replicate_key in zip(scalar_models, jax.random.split(key, n_replicates))
    ]
    totals = actual.nodes["acc"]
    assert totals.shape == (n_replicates, n_trials, task.n_steps, 2)
    np.testing.assert_allclose(
        totals, jnp.stack([state.nodes["acc"] for state in expected]), rtol=1e-6, atol=1e-6
    )


@pytest.mark.parametrize("ensemble_random_trials", [False, True])
def test_eval_ensemble_train_batch_matches_scalar_with_mapped_initializers(ensemble_random_trials):
    task = _EnsembleTask()
    n_replicates = 2
    models, axes, scalar_models = _ensemble_graphs(n_replicates, map_initial_state=True)
    key = jax.random.PRNGKey(123)
    batch_info = BatchInfo(size=2, current=0, total=1)
    states, losses, trials = task.eval_ensemble_train_batch(
        models,
        n_replicates,
        batch_info,
        key,
        ensemble_random_trials=ensemble_random_trials,
        model_axes=axes,
    )

    keys = jax.random.split(key, n_replicates) if ensemble_random_trials else [key] * n_replicates
    for i, (model, replicate_key) in enumerate(zip(scalar_models, keys)):
        scalar_states, scalar_losses, scalar_trials = task.eval_train_batch(
            model, batch_info, replicate_key
        )
        np.testing.assert_allclose(
            states.nodes["acc"][i], scalar_states.nodes["acc"], rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(losses.value[i], scalar_losses.value, rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(trials.inputs["delta"][i], scalar_trials.inputs["delta"])


def test_eval_ensemble_train_batch_rejects_replicate_count_mismatch_with_shared_key():
    task = DynamicInputTask()
    key = jax.random.PRNGKey(123)
    models = jtree.get_ensemble(lambda *, key: _make_dynamic_eval_graph(), n=2, key=key)

    with pytest.raises(ValueError, match="inconsistent sizes"):
        task.eval_ensemble_train_batch(
            models,
            n_replicates=3,
            batch_info=BatchInfo(size=2, current=0, total=1),
            key=key,
            ensemble_random_trials=False,
            model_axes=eqx.if_array(0),
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
        epoch_len_ranges=((0, 2), (2, 4)),
    )

    trial = task.get_train_trial(jax.random.PRNGKey(0))

    assert jnp.allclose(trial.inits["mechanics.effector"].pos, jnp.zeros(2))
    assert 0 <= trial.timeline.epoch_bounds[1] < 2
    assert 2 <= trial.timeline.epoch_bounds[2] - trial.timeline.epoch_bounds[1] < 4
    assert trial.timeline.epoch_bounds[-1] == 19


@pytest.mark.parametrize(
    ("stages", "prep_range", "expected_go_cue"),
    [(1, (0, 0), 0), (8, (7, 7), 7), (8, (7, 8), 7)],
)
def test_delayed_center_out_can_use_last_control_stage_for_movement(
    stages, prep_range, expected_go_cue
):
    task = DelayedReaches.delayed_center_out(
        loss_func=DummyLoss(),
        n_control_stages=stages,
        workspace=[[-1.0, -1.0], [1.0, 1.0]],
        epoch_len_ranges=[list(prep_range)],
        p_catch_trial=0.0,
    )

    trial = task.get_train_trial(jax.random.PRNGKey(0))

    np.testing.assert_array_equal(trial.timeline.epoch_bounds, [0, expected_go_cue, stages])
    np.testing.assert_array_equal(trial.timeline.event_steps, [expected_go_cue])
    np.testing.assert_array_equal(trial.inputs.hold[:, 0], [1] * expected_go_cue + [0])


def test_delayed_center_out_rejects_short_rollout_with_default_prep():
    with pytest.raises(ValueError, match="only 8 are available"):
        DelayedReaches.delayed_center_out(
            loss_func=DummyLoss(),
            n_control_stages=8,
            workspace=[[-1.0, -1.0], [1.0, 1.0]],
        )


def test_delayed_reaches_accepts_movement_in_sampled_epoch_without_remainder():
    task = DelayedReaches(
        loss_func=DummyLoss(),
        n_steps=3,
        workspace=[[-1.0, -1.0], [1.0, 1.0]],
        epoch_len_ranges=((2, 2),),
        epoch_names=("movement", "after_movement"),
        hold_epochs=(1,),
        target_on_epochs=(0,),
        move_epochs=(1, 0),
        go_cue_event_name="go",
        p_catch_trial=0.0,
    )

    trial = task.get_train_trial(jax.random.PRNGKey(0))

    assert trial.timeline.n_steps == 2
    np.testing.assert_array_equal(trial.timeline.epoch_bounds, [0, 2, 2])
    np.testing.assert_array_equal(trial.timeline.event_steps, [0])
    np.testing.assert_array_equal(
        trial.targets["mechanics.effector.pos"].value, trial.inputs.effector_target.pos
    )


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
    registry = retained_component_registry()
    params = delayed_center_out_reaches_params(
        n_control_stages=8,
        workspace=[[-1.0, -1.0], [1.0, 1.0]],
        epoch_len_ranges=[[2, 2]],
        p_catch_trial=0.0,
    )

    component = build_component(
        "task",
        f"feedbax.{DelayedReaches.__name__}",
        params,
        component_registry=registry,
    )
    task = component.task
    trial = component.trial_spec

    with pytest.raises(ValueError, match="epoch lengths must be integers"):
        build_component(
            "task",
            f"feedbax.{DelayedReaches.__name__}",
            {**params, "epoch_len_ranges": [[2, 2.5]]},
            component_registry=registry,
        )

    assert task.n_steps == 9
    assert task.train_endpoint_mode == "center_out"
    assert tuple(task.epoch_names) == ("prep", "movement")
    assert task.target_visible_from_start is True
    assert task.catch_metadata_policy == "flag"
    assert trial.timeline.epoch_names == ("prep", "movement")
    assert trial.timeline.event_names == ("go_cue",)
    assert trial.extra is not None
    assert bool(trial.extra["is_catch_trial"]) is False
