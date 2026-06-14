import pytest
from equinox.nn import StateIndex
import jax
import jax.numpy as jnp

from feedbax._tree import filter_spec_leaves
from feedbax.runtime.graph import Component, Graph, GraphTraceRequest, Wire, init_state_from_component
from feedbax.iterate import iterate_component
from feedbax.misc import attr_str_tree_to_where_func


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
