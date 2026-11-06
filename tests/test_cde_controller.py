"""Native CDE graph execution and continuation."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from feedbax.compiler import GraphDocument, compile_graph
from feedbax.compiler.serialization import graph_to_spec
from feedbax.component_registry import ComponentRegistry
from feedbax.compiler.cde_templates import cde_controller_spec
from feedbax.models.cde import cde_controller
from feedbax.runtime.graph import Graph, GraphTraceRequest


@pytest.mark.parametrize("decay", [-0.1, 1.1, float("inf")])
def test_decay_must_be_dissipative(decay):
    with pytest.raises(ValueError, match=r"decay must be finite and in the range \[0, 1\]"):
        cde_controller_spec(1, 2, 1, decay=decay)


@pytest.mark.parametrize("use_anti_nf", [False, True])
def test_native_steps_match_equations_and_carry_observations(use_anti_nf):
    key = jr.key(1)
    graph = cde_controller(
        3,
        4,
        2,
        vf_width=5,
        vf_depth=2,
        decay=0.25,
        use_anti_nf=use_anti_nf,
        alpha=0.6,
        key=key,
    )
    state = graph.init_state(key=key)
    h = graph.nodes["h0"].value
    previous = jnp.zeros(3)
    carry = None
    vf_key, readout_key, h0_key, gate_key = jr.split(key, 4)
    vector_field = eqx.nn.MLP(4, 12, 5, 2, final_activation=jax.nn.tanh, key=vf_key)
    readout = eqx.nn.Linear(4, 2, key=readout_key)
    gate = eqx.nn.GRUCell(3, 4, key=gate_key)
    assert jnp.array_equal(h, 0.01 * jr.normal(h0_key, (4,)))
    assert jnp.array_equal(graph.nodes["linear"].layer.bias, jnp.full(2, -5.0))

    for t, inputs in enumerate(
        (
            {"input": {"task": jnp.array([0.2])}, "feedback": jnp.array([0.3, 0.1])},
            {"feedback": jnp.array([0.5, 0.6, 0.7])},
            {"input": jnp.array([0.5, 0.6, 0.7])},
        )
    ):
        obs = jnp.concatenate(
            [x for port in ("input", "feedback") for x in jax.tree.leaves(inputs.get(port))]
        )
        expected = h + vector_field(h).reshape(4, 3) @ (obs - previous) - 0.25 * h
        if use_anti_nf:
            expected = expected + 0.6 * gate(obs, -h)
        outputs, state, carry = graph.step(inputs, state, carry, key=key, t=t)
        assert jnp.allclose(outputs["hidden"], expected)
        assert jnp.allclose(outputs["output"], jax.nn.sigmoid(readout.weight @ expected - 5.0))
        assert jnp.array_equal(carry[("obs_prev_in", "input")], obs)
        h = expected
        previous = obs


def test_fresh_reset_reads_learned_h0_and_continuation_uses_carry():
    key = jr.key(2)
    graph = cde_controller(1, 2, 1, decay=0.25, use_anti_nf=False, key=key)
    state = graph.init_state(key=key)
    h0 = jnp.array([2.0, -4.0])
    graph = eqx.tree_at(lambda g: g.nodes["h0"].value, graph, h0)
    obs = {"input": jnp.zeros(1)}

    def loss(model, carry):
        outputs, state_next, carry_next = model.step(obs, state, carry, key=key)
        return jnp.sum(outputs["hidden"]), (outputs, state_next, carry_next)

    (_, (outputs, _, carry)), grads = eqx.filter_value_and_grad(loss, has_aux=True)(graph, None)
    assert jnp.allclose(outputs["hidden"], 0.75 * h0)
    assert jnp.allclose(grads.nodes["h0"].value, jnp.full_like(h0, 0.75))

    changed = eqx.tree_at(lambda g: g.nodes["h0"].value, graph, -h0)
    (_, (outputs, _, _)), grads = eqx.filter_value_and_grad(loss, has_aux=True)(changed, carry)
    assert jnp.allclose(outputs["hidden"], 0.75**2 * h0)
    assert jnp.array_equal(grads.nodes["h0"].value, jnp.zeros_like(h0))
    overrides = {("h_prev_in", "input"): jnp.zeros_like(h0)}
    explicit = graph.initial_cycle_port_values(state, overrides)
    outputs, _, _ = graph.step(obs, state, explicit, key=key)
    assert jnp.array_equal(outputs["hidden"], jnp.zeros_like(h0))


def test_graph_round_trip_preserves_dtype_topology_and_initializer():
    with jax.experimental.enable_x64():
        key = jr.key(3)
        graph = cde_controller(2, 3, 1, vf_width=5, vf_depth=0, dtype=jnp.float64, key=key)
        spec = graph_to_spec(graph)
        registry = ComponentRegistry(load_user_components=False)
        rebuilt = compile_graph(GraphDocument(graph=spec), registry).graph
        state = rebuilt.init_state(key=key)
        outputs, _, carry = rebuilt.step({"input": jnp.ones(2, dtype=jnp.float32)}, state, key=key)
        assert rebuilt.component_bindings["h0"].trainable_override is True
        assert rebuilt.nodes["vf"].linears[0].weight.dtype == jnp.float64
        assert rebuilt.nodes["linear"].layer.bias.dtype == jnp.float64
        assert jnp.all(rebuilt.nodes["linear"].layer.bias == -5.0)
        assert rebuilt.nodes["gru_gate"].cell.weight_ih.dtype == jnp.float64
        assert rebuilt.nodes["h0"].value.dtype == jnp.float64
        assert jnp.array_equal(rebuilt.nodes["h0"].value, graph.nodes["h0"].value)
        assert outputs["output"].dtype == carry[("obs_prev_in", "input")].dtype == jnp.float64
        assert graph_to_spec(rebuilt) == spec


def test_scan_matches_jitted_and_vmapped_step_and_differentiates():
    key = jr.key(4)
    graph = cde_controller(2, 3, 2, vf_width=8, vf_depth=1, key=key)
    ensemble = eqx.filter_vmap(
        lambda model_key: cde_controller(2, 3, 2, vf_width=8, vf_depth=1, key=model_key)
    )(jr.split(key, 2))
    assert ensemble.nodes["h0"].value.shape == (2, 3)
    assert not jnp.array_equal(*ensemble.nodes["h0"].value)
    state = graph.init_state(key=key)
    observations = jnp.array([[1.0, 0.5], [0.2, -0.6], [0.1, 0.9]])

    @eqx.filter_jit
    def step(model, obs, current_state, carry, step_key):
        return model.step({"input": obs}, current_state, carry, key=step_key)

    batched = jax.vmap(lambda obs: step(graph, obs, state, None, key)[0])(observations)
    assert batched["output"].shape == (3, 2)
    carry = None
    stepped = []
    for obs, step_key in zip(observations, jr.split(key, 3)):
        outputs, state, carry = step(graph, obs, state, carry, step_key)
        stepped.append(outputs["output"])

    def loss(model):
        outputs, _ = model({"input": observations}, model.init_state(key=key), key=key)
        return jnp.sum(outputs["output"]), outputs

    (_, outputs), grads = eqx.filter_value_and_grad(loss, has_aux=True)(graph)
    assert jnp.allclose(outputs["output"], jnp.stack(stepped))
    assert jnp.all((outputs["output"] > 0) & (outputs["output"] < 1))
    assert jnp.any(grads.nodes["linear"].layer.weight != 0)
    assert jnp.any(grads.nodes["vf"].linears[-1].weight != 0)
    assert jnp.any(grads.nodes["gru_gate"].cell.weight_ih != 0)


def test_full_model_builder():
    from feedbax.analysis.aligned import DEFAULT_VARSET, ResponseVar
    from feedbax.tasks import SimpleReaches, prepare_trial
    from feedbax.xabdeef.losses import simple_reach_loss
    from feedbax.xabdeef.models import cde_controller_nn

    model_key, trial_key, run_key = jr.split(jr.key(5), 3)
    task = SimpleReaches(
        loss_func=simple_reach_loss(),
        workspace=jnp.array([[-1.0, -1.0], [1.0, 1.0]]),
        n_steps=10,
        eval_n_directions=2,
    )
    model = cde_controller_nn(
        task,
        n_steps=10,
        dt=0.05,
        hidden_dim=8,
        vf_width=16,
        vf_depth=1,
        motor_noise_std=0.02,
        key=model_key,
    )
    assert isinstance(model, Graph)
    assert isinstance(model.nodes["net"], Graph)
    trial = task.get_train_trial(key=trial_key)
    prepared = prepare_trial(model, trial)
    outputs, _, trace = model.rollout_with_trace(
        prepared.inputs,
        prepared.init_state,
        n_steps=prepared.n_steps,
        key=run_key,
        trace=(
            GraphTraceRequest(kind="port", selector="hidden", node="net", port="hidden"),
            GraphTraceRequest(kind="port", selector="command", node="net", port="output"),
            GraphTraceRequest(kind="state_path", selector="hidden_history", path="net.hidden"),
        ),
    )
    assert all(jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(outputs))
    assert trace["hidden"].shape == (prepared.n_steps, 8)
    assert jnp.all(jnp.isfinite(trace["hidden"]))
    _, _, history = model(
        prepared.inputs,
        prepared.init_state,
        n_steps=prepared.n_steps,
        key=run_key,
        return_state_history=True,
    )
    assert jnp.array_equal(history.net.hidden[0], model.net.nodes["h0"].value)
    assert jnp.array_equal(history.net.hidden[1:], trace["hidden"])
    assert jnp.array_equal(history.net.hidden[1:], trace["hidden_history"])
    assert jnp.allclose(DEFAULT_VARSET[ResponseVar.COMMAND].where(history)[1:], trace["command"])
    states, losses = task.eval_with_loss(model, run_key)
    assert states.net.hidden.shape == (2, prepared.n_steps, 8)
    assert jnp.all(jnp.isfinite(losses["nn_hidden"].value))
