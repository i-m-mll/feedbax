import pytest
from equinox.nn import StateIndex
import jax
import jax.numpy as jnp

from feedbax.analysis import feedbax_graph_controller, graph_controller
from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.graph import ComponentSpec, GraphSpec, WireSpec
from feedbax.intervene import (
    PlanarTargetRelativeSelector,
    ThresholdLatchedForce,
    ThresholdLatchedForceParams,
)
from feedbax.runtime.graph import Component, Graph, GraphTraceRequest, Wire
from feedbax.runtime.state import CartesianState


class _StatefulGain(Component):
    input_ports = ("input",)
    output_ports = ("output",)

    gain: jnp.ndarray
    state_index: StateIndex

    def __init__(self, gain):
        self.gain = jnp.asarray(gain, dtype=jnp.float32)
        self.state_index = StateIndex(jnp.zeros((1,), dtype=jnp.float32))

    def __call__(self, inputs, state, *, key):
        count = state.get(self.state_index)
        state = state.set(self.state_index, count + 1.0)
        return {"output": self.gain @ inputs["input"] + count}, state


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


class _Comparison(Component):
    input_ports = ("input", "previous")
    output_ports = ("output",)

    def __call__(self, inputs, state, *, key):
        return {"output": inputs["input"] > inputs["previous"]}, state


@jax.experimental.enable_x64()
def test_feedbax_graph_controller_uses_native_state_precision_with_flat_coordinates():
    graph = Graph(
        nodes={"net": _StatefulGain([[1.0, 0.5]])},
        wires=(),
        input_ports=("input",),
        output_ports=("output",),
        input_bindings={"input": ("net", "input")},
        output_bindings={"output": ("net", "output")},
    )

    controller = feedbax_graph_controller(graph, key=jax.random.PRNGKey(0), dtype=jnp.float64)
    h0 = controller.initial_state()

    assert h0.shape == (1,)
    assert h0.dtype == jnp.float64
    assert float(h0[0]) == 0.0

    input_value = jnp.array([2**24 - 2, 4.0], dtype=jnp.float32)
    h1, output1 = controller.step(h0, input_value, 0)
    assert float(output1[0]) == 2**24
    assert float(h1[0]) == 1.0

    h2, output2 = controller.step(h1, input_value, 1)
    # The native float32 addition rounds away the one added by count.
    assert float(output2[0]) == 2**24
    assert h2.dtype == jnp.float64
    assert float(h2[0]) == 2.0


def test_graph_controller_rejects_native_boolean_latch_state():
    latch = ThresholdLatchedForce(
        state_selector=PlanarTargetRelativeSelector(position_path=("pos",), target_path=("goal",)),
        direction="increasing",
        dt=0.1,
        params=ThresholdLatchedForceParams(threshold=0.5, lateral_force=2.0, active=True),
    )
    input_ports = ("state", "target", "force")
    graph = Graph(
        nodes={"load": latch},
        wires=(),
        input_ports=input_ports,
        output_ports=("force",),
        input_bindings={port: ("load", port) for port in input_ports},
        output_bindings={"force": ("load", "force")},
    )

    key = jax.random.PRNGKey(0)
    state = graph.init_state(key=key)
    carry = None
    forces = []
    for t, position in enumerate((-0.5, 0.0, -0.2)):
        outputs, state, carry = graph.step(
            {
                "state": CartesianState(pos=jnp.array([position, 0.0]), vel=jnp.zeros(2)),
                "target": {"goal": jnp.array([1.0, 0.0])},
                "force": jnp.array([0.5, 0.25]),
            },
            state,
            carry,
            key=jax.random.fold_in(key, t),
            t=t,
        )
        forces.append(outputs["force"])

    assert jnp.array_equal(jnp.stack(forces), jnp.array([[0.5, 0.25], [0.5, 2.25], [0.5, 2.25]]))
    with pytest.raises(ValueError, match="graph state.*real floating.*bool"):
        graph_controller(graph, key=key, input_port="state", output_port="force", dtype=jnp.float32)


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

    key = jax.random.PRNGKey(0)
    trace_request = GraphTraceRequest(
        kind="recurrent_carry",
        selector="recurrent_carry:net.feedback",
        node="net",
        port="feedback",
    )
    controller = graph_controller(
        graph,
        key=key,
        output_port="force",
        dtype=jnp.float32,
        trace=(trace_request,),
    )
    h0 = controller.initial_state()

    # Only the recurrent carry is packed because this graph has no component state.
    assert h0.shape == (1,)
    assert float(h0[0]) == 0.0

    native_state = graph.init_state(key=key)
    native_carry = None
    adapted_state = h0
    inputs = jnp.array([[2.0], [3.0]], dtype=jnp.float32)
    for t, input_value in enumerate(inputs):
        native_outputs, native_state, native_carry = graph.step(
            {"input": input_value},
            native_state,
            native_carry,
            key=jax.random.fold_in(key, t),
            t=t,
        )
        step = controller.step_with_trace(adapted_state, input_value, t)
        assert jnp.array_equal(step.output, native_outputs["force"])
        assert jnp.array_equal(step.trace[trace_request.selector], native_carry["net", "feedback"])
        assert jnp.array_equal(step.state, native_carry["net", "feedback"])
        adapted_state = step.state
    assert float(step.output[0]) == pytest.approx(9.0)
    assert float(adapted_state[0]) == pytest.approx(20.0)

    def final_output_and_carry(h, controller_inputs):
        h, _ = controller.step(h, controller_inputs[0], 0)
        h, output = controller.step(h, controller_inputs[1], 1)
        return jnp.concatenate([output, h])

    # force_t = input_t + carry_t, carry_(t+1) = 2 * (force_t + 1).
    state_jacobian, input_jacobian = jax.jacfwd(final_output_and_carry, argnums=(0, 1))(h0, inputs)
    assert jnp.array_equal(state_jacobian, jnp.array([[2.0], [4.0]]))
    assert jnp.array_equal(input_jacobian[..., 0], jnp.array([[2.0, 1.0], [4.0, 2.0]]))


def test_graph_controller_rejects_boolean_carry_produced_after_float_initialization():
    graph = Graph(
        nodes={"compare": _Comparison()},
        wires=(
            Wire(
                "compare",
                "output",
                "compare",
                "previous",
                temporality="recurrent",
                recurrent_initializer={"kind": "zeros", "shape": (1,)},
            ),
        ),
        input_ports=("input",),
        output_ports=("output",),
        input_bindings={"input": ("compare", "input")},
        output_bindings={"output": ("compare", "output")},
    )
    controller = graph_controller(graph, key=jax.random.PRNGKey(0), dtype=jnp.float32)

    with pytest.raises(ValueError, match="recurrent carry.*real floating.*bool"):
        controller.step(controller.initial_state(), jnp.ones(1, dtype=jnp.float32), 0)


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

    with pytest.raises(ValueError, match="component_registry is required"):
        graph_controller(spec, key=jax.random.PRNGKey(0), dtype=jnp.float32)

    controller = graph_controller(
        spec,
        key=jax.random.PRNGKey(0),
        component_registry=ComponentRegistry(load_user_components=False),
        dtype=jnp.float32,
    )
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
