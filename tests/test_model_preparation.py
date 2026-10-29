"""Reset, reconstruction and optimizer contracts for complete prepared models."""

import pickle
from types import SimpleNamespace

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree as jt
import optax
import pytest

from feedbax.runtime.components import GRU, RateLimiter
from feedbax.runtime.filters import FirstOrderFilter
from feedbax.runtime.graph import Graph
from feedbax.web.worker.execution import _trainable_filter


def test_prepared_gru_reconstruction_reuses_trace_and_retains_weight_updates():
    key = jax.random.PRNGKey(0)
    traces = []

    @eqx.filter_jit
    def step(model, state):
        traces.append(None)
        return model({"input": jnp.ones(2)}, state, key=key)

    original = GRU(2, 3, key=key).prepare()
    updated = eqx.tree_at(lambda m: m.cell.weight_ih, original, original.cell.weight_ih + 1)
    rebuilt = GRU(2, 3, key=key).prepare()
    outputs = []
    for model in (original, updated, rebuilt, pickle.loads(pickle.dumps(rebuilt))):
        out, _ = step(model, model.init_state(key=key))
        outputs.append(out["output"])
    assert len(traces) == 1
    assert not jnp.allclose(outputs[0], outputs[1])
    assert jnp.array_equal(outputs[0], outputs[2])
    assert jnp.array_equal(outputs[0], outputs[3])


def test_vector_rate_limiter_reset_changes_reuse_trace_and_survive_pickle():
    key = jax.random.PRNGKey(0)
    traces = []

    @eqx.filter_jit
    def step(model):
        traces.append(None)
        outputs, _ = model({"input": jnp.full(2, 10.0)}, model.init_state(key=key), key=key)
        return outputs["output"]

    original = RateLimiter(n_dims=2, max_rate=1, dt=1).prepare()
    changed = eqx.tree_at(
        lambda m: m.state_index.init, original, jnp.full(2, 2.0, dtype=jnp.float32)
    )
    rebuilt = RateLimiter(n_dims=2, max_rate=1, dt=1, initial_value=2).prepare()
    for model, expected in (
        (original, 1),
        (changed, 3),
        (rebuilt, 3),
        (pickle.loads(pickle.dumps(changed)), 3),
    ):
        assert jnp.array_equal(step(model), jnp.full(2, expected))
    assert len(traces) == 1


def test_preparing_nested_shared_children_assigns_independent_reset_slots():
    child = RateLimiter(n_dims=2, initial_value=2).prepare()
    inner = Graph(nodes={"same": child}, wires=())
    graph = Graph(nodes={"same": child, "inner": inner}, wires=()).prepare().prepare()
    left = graph.nodes["same"]
    right = graph.nodes["inner"].nodes["same"]
    state = graph.init_state(key=jax.random.PRNGKey(0))
    state = state.set(left.state_index, jnp.full(2, 9.0, dtype=jnp.float32))
    assert jnp.array_equal(state.get(right.state_index), jnp.full(2, 2.0, dtype=jnp.float32))
    reset = graph.init_state(key=jax.random.PRNGKey(0))
    assert jnp.array_equal(reset.get(left.state_index), jnp.full(2, 2.0, dtype=jnp.float32))
    assert jnp.array_equal(reset.get(right.state_index), jnp.full(2, 2.0, dtype=jnp.float32))


@pytest.mark.parametrize(
    "solver_type, first_dtype", [(dfx.Euler, jnp.float32), (dfx.Heun, jnp.float64)]
)
def test_filter_change_input_preserves_solver_configuration_and_mixed_float_reset(
    solver_type, first_dtype
):
    with jax.experimental.enable_x64():
        original = FirstOrderFilter(
            tau_rise=0.2, tau_decay=0.3, dt=0.01, solver_type=solver_type, init_value=2
        ).prepare()
        proto = {"x": jnp.zeros(2, dtype=first_dtype), "y": jnp.zeros(3, dtype=jnp.float64)}
        changed = original.change_input(proto).prepare()
        assert eqx.tree_equal(changed.solver, original.solver)
        state = changed.init_state(key=jax.random.PRNGKey(0))
        assert state.get(changed.state_index).output["y"].dtype == jnp.float64
        out, _ = changed({"input": proto}, state, key=jax.random.PRNGKey(0))
        assert out["output"]["x"].dtype == first_dtype
        assert out["output"]["y"].dtype == jnp.float64
        assert jnp.all(out["output"]["x"] < 2)
        assert original.state_index.init.output.shape == (1,)
        with pytest.raises(ValueError, match="real floating"):
            original.change_input(jnp.zeros(2, dtype=jnp.int32))


def test_studio_update_trains_weights_and_keeps_reset_template_fixed():
    from feedbax.training.studio_executor import StudioUpdateKernel

    graph = Graph(nodes={"cell": GRU(2, 3, key=jax.random.PRNGKey(0))}, wires=()).prepare()
    compiled = SimpleNamespace(graph=graph, trainable_filter=_trainable_filter(graph, ("cell",)))
    optimizer = optax.sgd(0.1)

    def rollout(model, plan, key):
        cell = model.nodes["cell"]
        outputs, _ = cell({"input": jnp.ones(2)}, model.init_state(key=key), key=key)
        return outputs

    def loss(plan, outputs):
        return jnp.sum(outputs["output"]), {}

    # The former mask assigns reset payloads a real, nonzero optimizer update.
    grads = eqx.filter_grad(
        lambda model: loss(None, rollout(model, None, jax.random.PRNGKey(0)))[0]
    )(graph)
    assert jnp.any(grads.nodes["cell"].state_index.init.hidden != 0)
    old_update = eqx.apply_updates(
        graph, jt.map(lambda x: -0.1 * x if eqx.is_array(x) else None, grads)
    )
    assert not jnp.array_equal(
        old_update.nodes["cell"].state_index.init.hidden,
        graph.nodes["cell"].state_index.init.hidden,
    )

    kernel = StudioUpdateKernel(
        compiled=compiled, optimizer=optimizer, rollout_fn=rollout, evaluate_loss_fn=loss
    )
    trainable, _ = eqx.partition(graph, compiled.trainable_filter)
    slots = kernel(
        {
            "model": graph,
            "optimizer": optimizer.init(trainable),
            "prng": jax.random.PRNGKey(0),
            "batch_counter": jnp.array(0),
        }
    )
    trained = slots["model"].nodes["cell"]
    assert not jnp.array_equal(trained.cell.weight_ih, graph.nodes["cell"].cell.weight_ih)
    assert all(
        jnp.array_equal(before, after)
        for before, after in zip(
            jt.leaves(graph.nodes["cell"].state_index.init), jt.leaves(trained.state_index.init)
        )
    )


def test_eager_point_mass_rebuild_consistency_and_custom_callbacks_use_current_nodes():
    from feedbax.runtime.iteration import run_component
    from feedbax.xabdeef.models import point_mass_nn

    key = jax.random.PRNGKey(0)
    task = SimpleNamespace(
        get_train_trial_with_intervenor_params=lambda **kwargs: SimpleNamespace(inputs=jnp.zeros(1))
    )

    def build():
        return point_mass_nn(
            task,
            hidden_size=3,
            feedback_delay_steps=2,
            feedback_noise_std=0,
            motor_noise_std=0,
            key=key,
        )

    traces = []

    @eqx.filter_jit
    def rollout(model):
        traces.append(None)
        state = model.init_state(key=key)
        mechanics = state.get(model.mechanics.state_index)
        mechanics = eqx.tree_at(
            lambda value: value.effector.pos, mechanics, jnp.asarray([2.0, 3.0], dtype=jnp.float32)
        )
        state = state.set(model.mechanics.state_index, mechanics)
        state = model.state_consistency_update(state)
        view = model.state_view(state)
        outputs, _, _ = run_component(
            model, {"input": jnp.zeros((2, 1))}, state, key=key, n_steps=2
        )
        return view, outputs

    model = build()
    initial, outputs = rollout(model)
    rebuilt_initial, rebuilt_outputs = rollout(build())
    assert len(traces) == 1
    assert jnp.array_equal(initial.mechanics.plant.skeleton.pos, jnp.asarray([2.0, 3.0]))
    for pos, vel in initial.feedback.queue:
        assert jnp.array_equal(pos, jnp.asarray([2.0, 3.0]))
    assert eqx.tree_equal(initial, rebuilt_initial)
    assert eqx.tree_equal(outputs, rebuilt_outputs)

    def consistency(current, state):
        return state.set(current.net.state_index, current.net.state_index.init)

    def view(node_states):
        return node_states["net"]

    customized = model.with_state_consistency(consistency).with_state_view(view).prepare()
    assert customized.state_consistency_fn is consistency
    assert customized.state_view_fn is view
    state = customized.state_consistency_update(customized.init_state(key=key))
    assert eqx.tree_equal(customized.state_view(state), customized.net.state_index.init)


def test_preparation_keeps_task_parameter_and_latch_slots_independent():
    from feedbax.intervene import StateSelector, ThresholdLatchedForce, ThresholdLatchedForceParams

    def build():
        return ThresholdLatchedForce(
            state_selector=StateSelector(("pos", 0)),
            direction="increasing",
            dt=0.1,
            params=ThresholdLatchedForceParams(active=True),
        ).prepare()

    graph = Graph(
        nodes={"left": build(), "nested": Graph(nodes={"left": build()}, wires=())}, wires=()
    ).prepare()
    left, right = graph.nodes["left"], graph.nodes["nested"].nodes["left"]
    state = graph.init_state(key=jax.random.PRNGKey(0))
    params = eqx.tree_at(lambda value: value.active, state.get(left.params_index), jnp.array(False))
    latch = eqx.tree_at(lambda value: value.latched, state.get(left.latch_index), jnp.array(True))
    state = state.set(left.params_index, params).set(left.latch_index, latch)
    assert not bool(state.get(left.params_index).active)
    assert bool(state.get(right.params_index).active)
    assert bool(state.get(left.latch_index).latched)
    assert not bool(state.get(right.latch_index).latched)
    assert left.task_parameter_state_indices()[left.label].marker == left.params_index.marker
    assert state.get(right.params_index).scale.dtype == jnp.float32
