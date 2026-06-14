import pytest
from equinox.nn import StateIndex
import jax
import jax.numpy as jnp

from feedbax.analysis import feedbax_graph_controller, graph_controller
from feedbax.contracts.graph import ComponentSpec, GraphSpec, WireSpec
from feedbax.runtime.graph import Component, Graph, GraphTraceRequest, Wire


class _StatefulGain(Component):
    input_ports = ("input",)
    output_ports = ("output",)

    gain: jnp.ndarray
    state_index: StateIndex

    def __init__(self, gain):
        self.gain = jnp.asarray(gain)
        self.state_index = StateIndex(jnp.zeros((1,), dtype=jnp.float32))

    def __call__(self, inputs, state, *, key):
        count = state.get(self.state_index)
        state = state.set(self.state_index, count + 1.0)
        return {"output": self.gain @ inputs["input"]}, state


class _FeedbackNetwork(Component):
    input_ports = ("input", "feedback")
    output_ports = ("force",)

    def __call__(self, inputs, state, *, key):
        return {"force": inputs["input"] + inputs["feedback"]}, state


class _Mechanics(Component):
    input_ports = ("force",)
    output_ports = ("obs",)

    def __call__(self, inputs, state, *, key):
        return {"obs": inputs["force"] + 1.0}, state


class _Feedback(Component):
    input_ports = ("obs",)
    output_ports = ("feedback",)

    def __call__(self, inputs, state, *, key):
        return {"feedback": 2.0 * inputs["obs"]}, state


def test_feedbax_graph_controller_preserves_simple_controller_shape():
    graph = Graph(
        nodes={"net": _StatefulGain([[1.0, 0.5]])},
        wires=(),
        input_ports=("input",),
        output_ports=("output",),
        input_bindings={"input": ("net", "input")},
        output_bindings={"output": ("net", "output")},
    )

    controller = feedbax_graph_controller(graph, key=jax.random.PRNGKey(0), dtype=jnp.float32)
    h0 = controller.initial_state()

    assert h0.shape == (1,)
    assert float(h0[0]) == 0.0

    h1, output1 = controller.step(h0, jnp.array([2.0, 4.0], dtype=jnp.float32), 0)
    assert float(output1[0]) == pytest.approx(4.0)
    assert float(h1[0]) == 1.0

    h2, output2 = controller.step(h1, jnp.array([2.0, 4.0], dtype=jnp.float32), 1)
    assert float(output2[0]) == pytest.approx(4.0)
    assert float(h2[0]) == 2.0


def test_graph_controller_threads_recurrent_feedback_cycle_and_trace():
    graph = Graph(
        nodes={
            "net": _FeedbackNetwork(),
            "mechanics": _Mechanics(),
            "feedback": _Feedback(),
        },
        wires=(
            Wire("net", "force", "mechanics", "force"),
            Wire("mechanics", "obs", "feedback", "obs"),
            Wire(
                "feedback",
                "feedback",
                "net",
                "feedback",
                temporality="recurrent",
                recurrent_initializer={"kind": "zeros", "shape": (1,)},
            ),
        ),
        input_ports=("input",),
        output_ports=("force",),
        input_bindings={"input": ("net", "input")},
        output_bindings={"force": ("net", "force")},
    )

    with pytest.raises(KeyError, match="feedback"):
        graph._call_single_step(
            {"input": jnp.array([2.0], dtype=jnp.float32)},
            graph.init_state(key=jax.random.PRNGKey(0)),
            key=jax.random.PRNGKey(1),
        )

    controller = graph_controller(
        graph,
        key=jax.random.PRNGKey(0),
        output_port="force",
        dtype=jnp.float32,
        trace=(
            GraphTraceRequest(
                kind="recurrent_carry",
                selector="recurrent_carry:net.feedback",
                node="net",
                port="feedback",
            ),
        ),
    )
    h0 = controller.initial_state()

    # Only the recurrent carry is packed because this graph has no component state.
    assert h0.shape == (1,)
    assert float(h0[0]) == 0.0

    step1 = controller.step_with_trace(h0, jnp.array([2.0], dtype=jnp.float32), 0)
    assert float(step1.output[0]) == pytest.approx(2.0)
    assert float(step1.trace["recurrent_carry:net.feedback"][0]) == pytest.approx(6.0)

    step2 = controller.step_with_trace(step1.state, jnp.array([2.0], dtype=jnp.float32), 1)
    assert float(step2.output[0]) == pytest.approx(8.0)
    assert float(step2.trace["recurrent_carry:net.feedback"][0]) == pytest.approx(18.0)


def test_graph_controller_static_inputs_and_mapping_inputs():
    graph = Graph(
        nodes={"net": _FeedbackNetwork()},
        wires=(),
        input_ports=("input", "feedback"),
        output_ports=("force",),
        input_bindings={
            "input": ("net", "input"),
            "feedback": ("net", "feedback"),
        },
        output_bindings={"force": ("net", "force")},
    )

    controller = graph_controller(
        graph,
        key=jax.random.PRNGKey(0),
        output_port="force",
        dtype=jnp.float32,
        static_inputs={"feedback": jnp.array([3.0], dtype=jnp.float32)},
    )

    h1, output = controller.step(
        controller.initial_state(),
        {"input": jnp.array([2.0], dtype=jnp.float32)},
        0,
    )

    assert h1.shape == (0,)
    assert float(output[0]) == pytest.approx(5.0)


def test_graph_controller_accepts_recurrent_graph_spec():
    spec = GraphSpec(
        nodes={
            "sum": ComponentSpec(
                type="Sum",
                params={},
                input_ports=["a", "b"],
                output_ports=["output"],
            )
        },
        wires=[
            WireSpec(
                source_node="sum",
                source_port="output",
                target_node="sum",
                target_port="b",
                temporality="recurrent",
                recurrent_initializer={"kind": "zeros", "shape": [1]},
            )
        ],
        input_ports=["input"],
        output_ports=["output"],
        input_bindings={"input": ("sum", "a")},
        output_bindings={"output": ("sum", "output")},
    )

    controller = graph_controller(spec, key=jax.random.PRNGKey(0), dtype=jnp.float32)
    h1, output1 = controller.step(
        controller.initial_state(),
        jnp.array([3.0], dtype=jnp.float32),
        0,
    )
    h2, output2 = controller.step(h1, jnp.array([3.0], dtype=jnp.float32), 1)

    assert controller.initial_state().shape == (1,)
    assert float(output1[0]) == pytest.approx(3.0)
    assert float(output2[0]) == pytest.approx(6.0)
    assert float(h2[0]) == pytest.approx(6.0)
