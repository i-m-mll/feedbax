import pytest
import equinox as eqx
from equinox.nn import StateIndex
import jax
import jax.numpy as jnp

from jax_cookbook.tree import filter_spec_leaves
from feedbax.runtime.graph import (
    Component,
    Graph,
    GraphTraceRequest,
    RolloutStepHookResult,
    Wire,
    init_state_from_component,
)
from feedbax.runtime.iteration import run_component
from feedbax.runtime.iteration import iterate_component
from feedbax.runtime.where_selectors import attr_str_tree_to_where_func


class Increment(Component):
    input_ports = ("x",)
    output_ports = ("x",)

    def __call__(self, inputs, state, *, key):
        x = inputs["x"]
        return {"x": x + 1}, state


class Counter(Component):
    input_ports = ("input",)
    output_ports = ("output",)

    state_index: StateIndex

    def __init__(self):
        self.state_index = StateIndex(jnp.array(0))

    def __call__(self, inputs, state, *, key):
        count = state.get(self.state_index)
        count = count + 1
        state = state.set(self.state_index, count)
        return {"output": count}, state


class Accumulator(Component):
    input_ports = ("delta", "base")
    output_ports = ("total",)

    state_index: StateIndex

    def __init__(self):
        self.state_index = StateIndex(jnp.array(0.0, dtype=jnp.float32))

    def __call__(self, inputs, state, *, key):
        total = state.get(self.state_index) + inputs["delta"]
        state = state.set(self.state_index, total)
        return {"total": total + 0.0 * inputs.get("base", 0.0)}, state

    def state_view(self, state):
        return state.get(self.state_index)

    def initial_outputs(self, state_value):
        if state_value is None:
            return {}
        return {"total": state_value}


def test_graph_cycle_iteration():
    node = Increment()
    graph = Graph(
        nodes={"inc": node},
        wires=(Wire("inc", "x", "inc", "x", temporality="recurrent"),),
        input_ports=(),
        output_ports=("x",),
        input_bindings={},
        output_bindings={"x": ("inc", "x")},
    )

    outputs, _ = graph(
        {},
        init_state_from_component(graph),
        key=jax.random.PRNGKey(0),
        n_steps=3,
        cycle_init={("inc", "x"): jnp.array(0)},
    )

    assert (outputs["x"] == jnp.array([1, 2, 3])).all()


def test_iterate_component_state_history():
    component = Counter()
    inputs = jnp.zeros((2,))
    outputs, final_state, history = iterate_component(
        component,
        inputs,
        init_state_from_component(component),
        n_steps=2,
        key=jax.random.PRNGKey(0),
    )

    assert history is not None
    assert (history == jnp.array([0, 1, 2])).all()


def _make_dynamic_input_graph():
    return Graph(
        nodes={"counter": Counter(), "acc": Accumulator()},
        wires=(Wire("counter", "output", "acc", "base"),),
        input_ports=("delta",),
        output_ports=("total",),
        input_bindings={"delta": ("acc", "delta")},
        output_bindings={"total": ("acc", "total")},
    )


def _counter_to_accumulator_delta_hook(context):
    if context.node_name != "acc":
        return None
    assert context.graph.nodes[context.node_name] is context.component
    assert "base" in context.node_inputs
    updated = dict(context.port_values)
    updated[(context.node_name, "delta")] = context.port_values[("counter", "output")]
    return RolloutStepHookResult(port_values=updated)


def test_rollout_step_hook_none_preserves_scan_behavior():
    graph = _make_dynamic_input_graph()
    state_a = init_state_from_component(graph)
    state_b = init_state_from_component(graph)
    inputs = {"delta": jnp.zeros((3,), dtype=jnp.float32)}

    outputs_a, final_state_a, _ = run_component(
        graph,
        inputs,
        state_a,
        key=jax.random.PRNGKey(0),
        n_steps=3,
    )
    outputs_b, final_state_b, _ = run_component(
        graph,
        inputs,
        state_b,
        key=jax.random.PRNGKey(0),
        n_steps=3,
        rollout_step_hook=None,
    )

    assert jnp.allclose(outputs_a["total"], outputs_b["total"])
    assert jnp.allclose(
        graph.state_view(final_state_a).nodes["acc"],
        graph.state_view(final_state_b).nodes["acc"],
    )


def test_rollout_step_hook_updates_node_input_from_live_port_values():
    graph = _make_dynamic_input_graph()
    inputs = {"delta": jnp.zeros((3,), dtype=jnp.float32)}

    outputs_no_hook, _, _ = run_component(
        graph,
        inputs,
        init_state_from_component(graph),
        key=jax.random.PRNGKey(0),
        n_steps=3,
    )
    outputs_hook, _, _ = run_component(
        graph,
        inputs,
        init_state_from_component(graph),
        key=jax.random.PRNGKey(0),
        n_steps=3,
        rollout_step_hook=_counter_to_accumulator_delta_hook,
    )

    assert jnp.allclose(outputs_no_hook["total"], jnp.array([0.0, 0.0, 0.0]))
    assert jnp.allclose(outputs_hook["total"], jnp.array([1.0, 3.0, 6.0]))


def test_rollout_step_hook_affects_streaming_loss_without_state_history():
    graph = _make_dynamic_input_graph()
    inputs = {"delta": jnp.zeros((3,), dtype=jnp.float32)}

    def streaming_loss_fn(state_view, t):
        return state_view.nodes["acc"]

    _, _, no_hook_loss = run_component(
        graph,
        inputs,
        init_state_from_component(graph),
        key=jax.random.PRNGKey(0),
        n_steps=3,
        streaming_loss_fn=streaming_loss_fn,
    )
    _, _, hook_loss = run_component(
        graph,
        inputs,
        init_state_from_component(graph),
        key=jax.random.PRNGKey(0),
        n_steps=3,
        streaming_loss_fn=streaming_loss_fn,
        rollout_step_hook=_counter_to_accumulator_delta_hook,
    )

    assert no_hook_loss == jnp.array(0.0)
    assert hook_loss == jnp.array(10.0)


def test_rollout_step_hook_affects_cyclic_graph_streaming_loss():
    graph = Graph(
        nodes={"acc": Accumulator()},
        wires=(Wire("acc", "total", "acc", "base", temporality="recurrent"),),
        input_ports=("delta",),
        output_ports=("total",),
        input_bindings={"delta": ("acc", "delta")},
        output_bindings={"total": ("acc", "total")},
    )
    inputs = {"delta": jnp.zeros((3,), dtype=jnp.float32)}

    def streaming_loss_fn(state_view, t):
        return state_view.nodes["acc"]

    def recurrent_base_to_delta_hook(context):
        if context.node_name != "acc":
            return None
        updated = dict(context.port_values)
        updated[(context.node_name, "delta")] = context.node_inputs["base"] + 1.0
        return RolloutStepHookResult(port_values=updated)

    _, _, no_hook_loss = run_component(
        graph,
        inputs,
        init_state_from_component(graph),
        key=jax.random.PRNGKey(0),
        n_steps=3,
        streaming_loss_fn=streaming_loss_fn,
    )
    _, _, hook_loss = run_component(
        graph,
        inputs,
        init_state_from_component(graph),
        key=jax.random.PRNGKey(0),
        n_steps=3,
        streaming_loss_fn=streaming_loss_fn,
        rollout_step_hook=recurrent_base_to_delta_hook,
    )

    assert no_hook_loss == jnp.array(0.0)
    assert hook_loss == jnp.array(11.0)


# =============================================================================
# Public Graph.step API tests
# Bug: 0ec8492 — public single-step API that threads cycle wires across calls.
# =============================================================================


class _Scaler(Component):
    """Scales its input by a fixed factor: out = factor * in."""

    input_ports = ("x",)
    output_ports = ("y",)

    factor: jnp.ndarray

    def __init__(self, factor):
        self.factor = jnp.asarray(factor)

    def __call__(self, inputs, state, *, key):
        return {"y": self.factor * inputs["x"]}, state


class _CountingScaler(Component):
    """Scales its input and records eager calls through a static side channel."""

    input_ports = ("x",)
    output_ports = ("y",)

    factor: jnp.ndarray
    calls: list[str] = eqx.field(static=True)

    def __init__(self, factor, calls):
        self.factor = jnp.asarray(factor)
        self.calls = calls

    def __call__(self, inputs, state, *, key):
        self.calls.append("call")
        return {"y": self.factor * inputs["x"]}, state


class _AddOne(Component):
    """Stateless add-one: out = in + 1."""

    input_ports = ("x",)
    output_ports = ("y",)

    def __call__(self, inputs, state, *, key):
        return {"y": inputs["x"] + 1.0}, state


def _make_cyclic_graph():
    """Build a 2-component cyclic graph.

    ``a``: y = factor * x
    ``b``: y = x + 1

    Wires:
      external "input" -> a.x
      a.y -> b.x
      b.y -> a.x   (recurrent one-step feedback)

    External output: b.y.
    """
    a = _Scaler(0.5)
    b = _AddOne()
    graph = Graph(
        nodes={"a": a, "b": b},
        wires=(
            Wire("a", "y", "b", "x"),
            Wire("b", "y", "a", "x", temporality="recurrent"),
        ),
        input_ports=("input",),
        output_ports=("out",),
        input_bindings={"input": ("a", "x")},
        output_bindings={"out": ("b", "y")},
    )
    return graph


def _make_graph_input_initialized_cycle():
    return Graph(
        nodes={"a": _Scaler(0.5), "b": _AddOne()},
        wires=(
            Wire("a", "y", "b", "x"),
            Wire(
                "b",
                "y",
                "a",
                "x",
                temporality="recurrent",
                recurrent_initializer={
                    "kind": "graph-input",
                    "scope": "trial",
                    "source": "seed",
                    "state_slot": "x",
                },
            ),
        ),
        input_ports=("seed",),
        output_ports=("out",),
        input_bindings={},
        output_bindings={"out": ("b", "y")},
    )


def _make_node_output_initialized_cycle(calls=None, *, encoder_factor=2.0):
    encoder = (
        _Scaler(encoder_factor)
        if calls is None
        else _CountingScaler(encoder_factor, calls)
    )
    return Graph(
        nodes={"encoder": encoder, "a": _Scaler(0.5), "b": _AddOne()},
        wires=(
            Wire("a", "y", "b", "x"),
            Wire(
                "b",
                "y",
                "a",
                "x",
                temporality="recurrent",
                recurrent_initializer={
                    "kind": "node-output",
                    "scope": "trial",
                    "source_node": "encoder",
                    "source_port": "y",
                    "state_slot": "x",
                },
            ),
        ),
        input_ports=("context",),
        output_ports=("out",),
        input_bindings={"context": ("encoder", "x")},
        output_bindings={"out": ("b", "y")},
    )


def test_graph_rejects_instant_cycles():
    with pytest.raises(ValueError, match="same-step cycle"):
        Graph(
            nodes={"a": _Scaler(0.5), "b": _AddOne()},
            wires=(
                Wire("a", "y", "b", "x"),
                Wire("b", "y", "a", "x"),
            ),
            input_ports=("input",),
            output_ports=("out",),
            input_bindings={"input": ("a", "x")},
            output_bindings={"out": ("b", "y")},
        )._execution_order


def test_graph_rejects_missing_wire_source_port() -> None:
    with pytest.raises(ValueError, match="Wire source port 'a.missing' does not exist"):
        Graph(
            nodes={"a": _Scaler(0.5), "b": _AddOne()},
            wires=(Wire("a", "missing", "b", "x"),),
            input_ports=("input",),
            output_ports=("out",),
            input_bindings={"input": ("a", "x")},
            output_bindings={"out": ("b", "y")},
        )


def test_graph_rejects_missing_wire_target_port() -> None:
    with pytest.raises(ValueError, match="Wire target port 'b.missing' does not exist"):
        Graph(
            nodes={"a": _Scaler(0.5), "b": _AddOne()},
            wires=(Wire("a", "y", "b", "missing"),),
            input_ports=("input",),
            output_ports=("out",),
            input_bindings={"input": ("a", "x")},
            output_bindings={"out": ("b", "y")},
        )


def test_graph_with_state_view_attaches_view_out_of_place() -> None:
    graph = Graph(
        nodes={"counter": Counter()},
        wires=(),
        input_ports=(),
        output_ports=("out",),
        input_bindings={},
        output_bindings={"out": ("counter", "output")},
    )

    def state_view(node_states):
        return {"counter_value": node_states["counter"]}

    updated = graph.with_state_view(state_view)

    assert graph.state_view_fn is None
    assert updated is not graph
    assert updated.state_view_fn is state_view

    state = init_state_from_component(updated)
    view = updated.state_view(state)
    assert set(view) == {"counter_value"}
    assert jnp.array_equal(view["counter_value"], jnp.array(0))


class _RuntimeCallbackGraph(Graph):
    pass


def test_graph_with_state_view_preserves_fields_and_subclass() -> None:
    constraint = object()

    def state_view(node_states):
        return node_states

    def state_consistency(state):
        return state

    graph = _RuntimeCallbackGraph(
        nodes={"counter": Counter()},
        wires=(),
        input_ports=("in",),
        output_ports=("out",),
        input_bindings={"in": ("counter", "input")},
        output_bindings={"out": ("counter", "output")},
        state_consistency_fn=state_consistency,
        checkpoint=True,
        parameter_constraints=(constraint,),
    )

    updated = graph.with_state_view(state_view)

    assert type(updated) is type(graph)
    assert updated is not graph
    assert updated.nodes is graph.nodes
    assert updated.wires is graph.wires
    assert updated.input_ports is graph.input_ports
    assert updated.output_ports is graph.output_ports
    assert updated.input_bindings is graph.input_bindings
    assert updated.output_bindings is graph.output_bindings
    assert updated.state_view_fn is state_view
    assert updated.state_consistency_fn is state_consistency
    assert updated.checkpoint is graph.checkpoint
    assert updated.parameter_constraints == (constraint,)


def test_graph_with_state_consistency_preserves_fields_and_subclass() -> None:
    constraint = object()

    def state_view(node_states):
        return node_states

    def state_consistency(state):
        return state

    graph = _RuntimeCallbackGraph(
        nodes={"counter": Counter()},
        wires=(),
        input_ports=("in",),
        output_ports=("out",),
        input_bindings={"in": ("counter", "input")},
        output_bindings={"out": ("counter", "output")},
        state_view_fn=state_view,
        checkpoint=True,
        parameter_constraints=(constraint,),
    )

    updated = graph.with_state_consistency(state_consistency)

    assert type(updated) is type(graph)
    assert updated is not graph
    assert updated.nodes is graph.nodes
    assert updated.wires is graph.wires
    assert updated.input_ports is graph.input_ports
    assert updated.output_ports is graph.output_ports
    assert updated.input_bindings is graph.input_bindings
    assert updated.output_bindings is graph.output_bindings
    assert updated.state_view_fn is state_view
    assert updated.state_consistency_fn is state_consistency
    assert updated.checkpoint is graph.checkpoint
    assert updated.parameter_constraints == (constraint,)


def test_graph_step_acyclic_returns_empty_cycle_values():
    """For an acyclic graph, ``step`` returns an empty cycle dict and works without one."""
    a = _Scaler(2.0)
    graph = Graph(
        nodes={"a": a},
        wires=(),
        input_ports=("input",),
        output_ports=("out",),
        input_bindings={"input": ("a", "x")},
        output_bindings={"out": ("a", "y")},
    )

    state = init_state_from_component(graph)
    outputs, _, cycle = graph.step(
        {"input": jnp.array(3.0)},
        state,
        key=jax.random.PRNGKey(0),
    )
    assert cycle == {}
    assert outputs["out"] == jnp.array(6.0)

    # And explicitly None should also work.
    outputs, _, cycle = graph.step(
        {"input": jnp.array(3.0)},
        state,
        None,
        key=jax.random.PRNGKey(0),
    )
    assert cycle == {}
    assert outputs["out"] == jnp.array(6.0)


class _GraphWithDuplicateNet(Graph):
    """Graph fixture whose field and executable node intentionally diverge."""

    net: Component

    def __init__(self):
        executable_net = _Scaler(2.0)
        stale_field_net = _Scaler(5.0)
        super().__init__(
            nodes={"net": executable_net},
            wires=(),
            input_ports=("input",),
            output_ports=("out",),
            input_bindings={"input": ("net", "x")},
            output_bindings={"out": ("net", "y")},
        )
        self.net = stale_field_net


def test_attr_string_where_prefers_graph_node_over_duplicate_field():
    graph = _GraphWithDuplicateNet()
    where = attr_str_tree_to_where_func("net")

    assert where(graph) is graph.nodes["net"]
    assert where(graph) is not graph.net


def test_attr_string_filter_spec_targets_executable_graph_node():
    graph = _GraphWithDuplicateNet()
    where = attr_str_tree_to_where_func("net")
    filter_spec = filter_spec_leaves(graph, where)

    assert filter_spec.nodes["net"].factor is True
    assert filter_spec.net.factor is False


def test_attr_string_where_supports_explicit_nodes_index_path():
    graph = _GraphWithDuplicateNet()
    where = attr_str_tree_to_where_func("nodes['net'].factor")

    assert where(graph) == graph.nodes["net"].factor


def test_graph_step_cyclic_threads_cycle_values():
    """Smoke: chain three ``step`` calls on a cyclic graph; cycle values propagate.

    The back-edge target is ``a.x`` (cycle wire b.y -> a.x). Because external
    "input" also binds to ``a.x``, the explicit input wins on any step that
    provides one. We chain calls and verify the contract: ``cycle_port_values``
    returned is keyed by ``("a", "x")`` and equals ``b.y`` from the
    just-completed step.
    """
    graph = _make_cyclic_graph()
    assert graph._needs_iteration  # sanity: cycle detected

    state = init_state_from_component(graph)

    # First step: seed cycle explicitly (the toy components are stateless and
    # don't expose initial_outputs, so None would raise — see
    # _get_initial_cycle_values).
    seed_cycle = {("a", "x"): jnp.array(0.0)}

    # Step 1: input = 4, a.y = 0.5*4 = 2, b.y = 2 + 1 = 3
    out1, state, cyc1 = graph.step(
        {"input": jnp.array(4.0)},
        state,
        seed_cycle,
        key=jax.random.PRNGKey(0),
    )
    assert out1["out"] == jnp.array(3.0)
    assert ("a", "x") in cyc1
    # cycle stores the source value (b.y) of the back-edge.
    assert cyc1[("a", "x")] == jnp.array(3.0)

    # Step 2: pass cycle from step 1; input dominates a.x again.
    out2, state, cyc2 = graph.step(
        {"input": jnp.array(2.0)},
        state,
        cyc1,
        key=jax.random.PRNGKey(1),
    )
    # a.y = 0.5*2 = 1, b.y = 2
    assert out2["out"] == jnp.array(2.0)
    assert cyc2[("a", "x")] == jnp.array(2.0)

    # Step 3: omit external input -> cycle value (b.y = 2.0) drives a.x.
    # a.y = 0.5*2 = 1, b.y = 2
    out3, _, cyc3 = graph.step(
        {},
        state,
        cyc2,
        key=jax.random.PRNGKey(2),
    )
    assert out3["out"] == jnp.array(2.0)
    assert cyc3[("a", "x")] == jnp.array(2.0)


def test_graph_input_recurrent_initializer_seeds_call_once_from_trial_input() -> None:
    graph = _make_graph_input_initialized_cycle()
    state = init_state_from_component(graph)

    initial_cycle = graph.initial_cycle_port_values(
        state,
        inputs={"seed": jnp.array([8.0, 10.0])},
    )
    assert jnp.allclose(initial_cycle[("a", "x")], jnp.array([8.0, 10.0]))

    outputs, _ = graph(
        {"seed": jnp.array([8.0, 10.0])},
        state,
        key=jax.random.PRNGKey(0),
        n_steps=3,
    )

    assert jnp.allclose(
        outputs["out"],
        jnp.array([[5.0, 6.0], [3.5, 4.0], [2.75, 3.0]]),
    )


def test_graph_input_recurrent_initializer_step_uses_input_only_for_missing_carry() -> None:
    graph = _make_graph_input_initialized_cycle()
    state = init_state_from_component(graph)

    out1, state, cycle = graph.step(
        {"seed": jnp.array([8.0, 10.0])},
        state,
        key=jax.random.PRNGKey(0),
    )
    out2, _, next_cycle = graph.step(
        {"seed": jnp.array([100.0, 200.0])},
        state,
        cycle,
        key=jax.random.PRNGKey(1),
    )

    assert jnp.allclose(out1["out"], jnp.array([5.0, 6.0]))
    assert jnp.allclose(cycle[("a", "x")], jnp.array([5.0, 6.0]))
    assert jnp.allclose(out2["out"], jnp.array([3.5, 4.0]))
    assert jnp.allclose(next_cycle[("a", "x")], jnp.array([3.5, 4.0]))


def test_graph_input_recurrent_initializer_saved_cycle_restart_ignores_new_seed() -> None:
    graph = _make_graph_input_initialized_cycle()
    full_state = init_state_from_component(graph)
    restart_state = init_state_from_component(graph)

    full_outputs, _ = graph(
        {"seed": jnp.array([8.0, 10.0])},
        full_state,
        key=jax.random.PRNGKey(0),
        n_steps=4,
    )
    _, restart_state, saved_cycle = graph.step(
        {"seed": jnp.array([8.0, 10.0])},
        restart_state,
        key=jax.random.PRNGKey(0),
    )
    resumed_outputs, _ = graph(
        {"seed": jnp.array([100.0, 200.0])},
        restart_state,
        key=jax.random.PRNGKey(1),
        n_steps=3,
        cycle_init=saved_cycle,
    )

    assert jnp.allclose(resumed_outputs["out"], full_outputs["out"][1:])


def test_graph_input_recurrent_initializer_flows_to_nested_graph_from_parent_input() -> None:
    inner = _make_graph_input_initialized_cycle()
    parent = Graph(
        nodes={"inner": inner},
        wires=(),
        input_ports=("seed",),
        output_ports=("out",),
        input_bindings={"seed": ("inner", "seed")},
        output_bindings={"out": ("inner", "out")},
    )
    state = init_state_from_component(parent)

    outputs, _ = parent(
        {"seed": jnp.array([8.0, 10.0])},
        state,
        key=jax.random.PRNGKey(0),
        n_steps=3,
    )

    assert jnp.allclose(
        outputs["out"],
        jnp.array([[5.0, 6.0], [3.5, 4.0], [2.75, 3.0]]),
    )


def test_node_output_recurrent_initializer_seeds_call_once_from_trial_node() -> None:
    calls: list[str] = []
    graph = _make_node_output_initialized_cycle(calls)
    state = init_state_from_component(graph)

    outputs, _ = graph(
        {"context": jnp.array(3.0)},
        state,
        key=jax.random.PRNGKey(0),
        n_steps=3,
    )

    assert calls == ["call"]
    assert jnp.allclose(outputs["out"], jnp.array([4.0, 3.0, 2.5]))


def test_node_output_recurrent_initializer_step_uses_source_only_for_missing_carry() -> None:
    calls: list[str] = []
    graph = _make_node_output_initialized_cycle(calls)
    state = init_state_from_component(graph)

    out1, state, cycle = graph.step(
        {"context": jnp.array(3.0)},
        state,
        key=jax.random.PRNGKey(0),
    )
    out2, _, next_cycle = graph.step(
        {"context": jnp.array(100.0)},
        state,
        cycle,
        key=jax.random.PRNGKey(1),
    )

    assert calls == ["call"]
    assert out1["out"] == jnp.array(4.0)
    assert cycle[("a", "x")] == jnp.array(4.0)
    assert out2["out"] == jnp.array(3.0)
    assert next_cycle[("a", "x")] == jnp.array(3.0)


def test_node_output_recurrent_initializer_saved_cycle_restart_does_not_rerun() -> None:
    calls: list[str] = []
    graph = _make_node_output_initialized_cycle(calls)
    full_state = init_state_from_component(graph)
    restart_state = init_state_from_component(graph)

    full_outputs, _ = graph(
        {"context": jnp.array(3.0)},
        full_state,
        key=jax.random.PRNGKey(0),
        n_steps=4,
    )
    _, restart_state, saved_cycle = graph.step(
        {"context": jnp.array(3.0)},
        restart_state,
        key=jax.random.PRNGKey(0),
    )
    calls_after_saved_cycle = list(calls)
    resumed_outputs, _ = graph(
        {"context": jnp.array(100.0)},
        restart_state,
        key=jax.random.PRNGKey(1),
        n_steps=3,
        cycle_init=saved_cycle,
    )

    assert calls == calls_after_saved_cycle
    assert jnp.allclose(resumed_outputs["out"], full_outputs["out"][1:])


def test_node_output_recurrent_initializer_rejects_recurrent_target_dependency() -> None:
    with pytest.raises(
        ValueError,
        match=r"node-output.*source seed\.y depends on recurrent target a\.x",
    ):
        Graph(
            nodes={"a": _Scaler(0.5), "b": _AddOne(), "seed": _Scaler(2.0)},
            wires=(
                Wire("a", "y", "seed", "x"),
                Wire("seed", "y", "b", "x"),
                Wire(
                    "b",
                    "y",
                    "a",
                    "x",
                    temporality="recurrent",
                    recurrent_initializer={
                        "kind": "node-output",
                        "scope": "trial",
                        "source_node": "seed",
                        "source_port": "y",
                        "state_slot": "x",
                    },
                ),
            ),
            input_ports=(),
            output_ports=("out",),
            input_bindings={},
            output_bindings={"out": ("b", "y")},
        )


def test_node_output_recurrent_initializer_gradient_flows_to_source_parameters() -> None:
    graph = _make_node_output_initialized_cycle(encoder_factor=2.0)

    def loss_fn(model):
        outputs, _ = model(
            {"context": jnp.array(3.0)},
            init_state_from_component(model),
            key=jax.random.PRNGKey(0),
            n_steps=2,
        )
        return jnp.sum(outputs["out"])

    loss, grads = eqx.filter_value_and_grad(loss_fn)(graph)

    assert loss == jnp.array(7.0)
    assert jnp.allclose(grads.nodes["encoder"].factor, jnp.array(2.25))


def test_zero_constant_and_graph_input_recurrent_initializer_behavior_is_unchanged() -> None:
    zero_graph = Graph(
        nodes={"a": _Scaler(0.5), "b": _AddOne()},
        wires=(
            Wire("a", "y", "b", "x"),
            Wire(
                "b",
                "y",
                "a",
                "x",
                temporality="recurrent",
                recurrent_initializer={"kind": "zeros", "shape": []},
            ),
        ),
        input_ports=(),
        output_ports=("out",),
        input_bindings={},
        output_bindings={"out": ("b", "y")},
    )
    constant_graph = Graph(
        nodes={"a": _Scaler(0.5), "b": _AddOne()},
        wires=(
            Wire("a", "y", "b", "x"),
            Wire(
                "b",
                "y",
                "a",
                "x",
                temporality="recurrent",
                recurrent_initializer={"kind": "constant", "value": 8.0},
            ),
        ),
        input_ports=(),
        output_ports=("out",),
        input_bindings={},
        output_bindings={"out": ("b", "y")},
    )

    zero_outputs, _ = zero_graph(
        {},
        init_state_from_component(zero_graph),
        key=jax.random.PRNGKey(0),
        n_steps=2,
    )
    constant_outputs, _ = constant_graph(
        {},
        init_state_from_component(constant_graph),
        key=jax.random.PRNGKey(0),
        n_steps=2,
    )
    graph_input = _make_graph_input_initialized_cycle()
    graph_input_outputs, _ = graph_input(
        {"seed": jnp.array([8.0, 10.0])},
        init_state_from_component(graph_input),
        key=jax.random.PRNGKey(0),
        n_steps=2,
    )

    assert jnp.allclose(zero_outputs["out"], jnp.array([1.0, 1.5]))
    assert jnp.allclose(constant_outputs["out"], jnp.array([5.0, 3.5]))
    assert jnp.allclose(graph_input_outputs["out"], jnp.array([[5.0, 6.0], [3.5, 4.0]]))


def test_constant_recurrent_initializer_honors_dtype_key_under_x64(enable_jax_x64) -> None:
    graph = _make_cyclic_graph()

    typed_wire = Wire(
        "b",
        "y",
        "a",
        "x",
        temporality="recurrent",
        recurrent_initializer={
            "kind": "constant",
            "value": [1.0, 2.0, 3.0],
            "dtype": "float32",
        },
    )
    typed_value = graph._initial_value_from_recurrent_initializer(typed_wire)
    assert typed_value.dtype == jnp.float32
    assert jnp.array_equal(typed_value, jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32))

    # Without the dtype key, the same JSON-native list promotes to float64 under
    # globally-enabled x64 — the exact mismatch the dtype key exists to prevent.
    untyped_wire = Wire(
        "b",
        "y",
        "a",
        "x",
        temporality="recurrent",
        recurrent_initializer={"kind": "constant", "value": [1.0, 2.0, 3.0]},
    )
    untyped_value = graph._initial_value_from_recurrent_initializer(untyped_wire)
    assert untyped_value.dtype == jnp.float64


def test_constant_recurrent_initializer_without_dtype_is_unchanged() -> None:
    graph = _make_cyclic_graph()
    value = [4.0, 5.0]

    wire = Wire(
        "b",
        "y",
        "a",
        "x",
        temporality="recurrent",
        recurrent_initializer={"kind": "constant", "value": value},
    )
    result = graph._initial_value_from_recurrent_initializer(wire)
    expected = jnp.asarray(value)

    assert result.dtype == expected.dtype
    assert jnp.array_equal(result, expected)


def test_constant_recurrent_initializer_rejects_invalid_dtype() -> None:
    graph = _make_cyclic_graph()

    wire = Wire(
        "b",
        "y",
        "a",
        "x",
        temporality="recurrent",
        recurrent_initializer={
            "kind": "constant",
            "value": [1.0],
            "dtype": "not_a_dtype",
        },
    )
    with pytest.raises(ValueError, match="invalid dtype"):
        graph._initial_value_from_recurrent_initializer(wire)


def test_recurrent_cycle_init_error_names_wire_and_reason() -> None:
    graph = _make_cyclic_graph()
    state = init_state_from_component(graph)

    with pytest.raises(
        ValueError,
        match=(
            r"Missing initial values for recurrent cycle wires: "
            r"b\.y -> a\.x: source node 'b' has no initial state"
        ),
    ):
        graph.initial_cycle_port_values(state)


def test_graph_step_equivalent_to_call_with_iteration():
    """Manual loop of ``step`` calls matches ``Graph.__call__(n_steps=N)``."""
    graph = _make_cyclic_graph()
    n_steps = 5

    # Use varying inputs so each timestep is distinguishable.
    inputs_seq = jnp.arange(1.0, 1.0 + n_steps, dtype=jnp.float32)
    base_key = jax.random.PRNGKey(42)
    keys = jax.random.split(base_key, n_steps)
    seed_cycle = {("a", "x"): jnp.array(0.0, dtype=jnp.float32)}

    # Reference: __call__ with n_steps. Match its key-splitting convention
    # (it does jax.random.split(key, n_steps) internally).
    state_ref = init_state_from_component(graph)
    outputs_ref, _ = graph(
        {"input": inputs_seq},
        state_ref,
        key=base_key,
        n_steps=n_steps,
        cycle_init=seed_cycle,
    )

    # Manual loop using step.
    state_manual = init_state_from_component(graph)
    cycle = seed_cycle
    out_history = []
    for t in range(n_steps):
        out_t, state_manual, cycle = graph.step(
            {"input": inputs_seq[t]},
            state_manual,
            cycle,
            key=keys[t],
        )
        out_history.append(out_t["out"])
    out_manual = jnp.stack(out_history, axis=0)

    assert jnp.allclose(outputs_ref["out"], out_manual, atol=1e-7), (outputs_ref["out"], out_manual)


def test_graph_step_with_trace_returns_selected_boundary_values():
    graph = _make_cyclic_graph()
    state = init_state_from_component(graph)
    seed_cycle = {("a", "x"): jnp.array(0.0)}

    outputs, state, cycle, trace = graph.step_with_trace(
        {"input": jnp.array(4.0)},
        state,
        seed_cycle,
        key=jax.random.PRNGKey(0),
        trace=(
            GraphTraceRequest(kind="port", selector="port:a.y", node="a", port="y"),
            GraphTraceRequest(
                kind="edge",
                selector="edge:a.y->b.x",
                source_node="a",
                source_port="y",
                target_node="b",
                target_port="x",
            ),
            GraphTraceRequest(kind="graph_output", selector="graph_output:out", port="out"),
            GraphTraceRequest(
                kind="recurrent_carry",
                selector="recurrent_carry:a.x",
                node="a",
                port="x",
            ),
        ),
    )

    assert outputs["out"] == jnp.array(3.0)
    assert cycle[("a", "x")] == jnp.array(3.0)
    assert trace["port:a.y"] == jnp.array(2.0)
    assert trace["edge:a.y->b.x"] == jnp.array(2.0)
    assert trace["graph_output:out"] == jnp.array(3.0)
    assert trace["recurrent_carry:a.x"] == jnp.array(3.0)


def test_parent_graph_threads_nested_recurrent_carry():
    inner = Graph(
        nodes={"inc": _AddOne()},
        wires=(
            Wire(
                "inc",
                "y",
                "inc",
                "x",
                temporality="recurrent",
                recurrent_initializer={"kind": "constant", "value": 0.0},
            ),
        ),
        input_ports=(),
        output_ports=("out",),
        input_bindings={},
        output_bindings={"out": ("inc", "y")},
    )
    parent = Graph(
        nodes={"inner": inner},
        wires=(),
        input_ports=(),
        output_ports=("out",),
        input_bindings={},
        output_bindings={"out": ("inner", "out")},
    )

    assert parent._needs_iteration
    outputs, _ = parent(
        {},
        init_state_from_component(parent),
        key=jax.random.PRNGKey(0),
        n_steps=3,
    )

    assert jnp.allclose(outputs["out"], jnp.array([1.0, 2.0, 3.0]))
