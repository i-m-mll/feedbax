import equinox as eqx
import pytest
from equinox.nn import StateIndex
import jax
import jax.numpy as jnp

from feedbax.analysis import feedbax_graph_controller, graph_controller
from feedbax.component_registry import ComponentRegistry
from feedbax.compiler.templates import (
    recurrent_graph_input_initializer,
    recurrent_node_output_initializer,
)
from feedbax.contracts.graph import ComponentSpec, GraphSpec, WireSpec
from feedbax.mechanics.force_fields import (
    PlanarTargetRelativeSelector,
    ThresholdLatchedForce,
    ThresholdLatchedForceParams,
)
from feedbax.models.cde import cde_controller
from feedbax.runtime.components import Noise, Sum
from feedbax.runtime.graph import Component, Graph, GraphTraceRequest, Wire
from feedbax.runtime.state import CartesianState
from tests.support.graph_compiler_test_support import spec_to_graph


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


def test_graph_controller_analyzes_cde_from_live_initial_hidden_through_continuation():
    key = jax.random.key(0)
    graph = cde_controller(1, 2, 1, use_anti_nf=False, decay=0.25, key=key)
    learned_hidden = jnp.array([2.0, -4.0], dtype=jnp.float32)
    graph = eqx.tree_at(lambda g: g.nodes["h0"].value, graph, learned_hidden)
    native_state = graph.init_state(key=key)
    controller = graph_controller(graph, key=key, dtype=jnp.float32)
    initial = controller.initial_state()
    hidden_carry_key = ("h_prev_in", "input")
    unpacked_state, unpacked_carry = controller.unpack_state(initial)
    assert jnp.array_equal(unpacked_carry[hidden_carry_key], learned_hidden)
    assert jnp.array_equal(controller.pack_state(unpacked_state, unpacked_carry), initial)

    obs = jnp.array([0.5], dtype=jnp.float32)
    native_carry = None
    flat_state = initial
    for t in range(2):
        native_output, native_state, native_carry = graph.step(
            {"input": obs}, native_state, native_carry, key=jax.random.fold_in(key, t), t=t
        )
        flat_state, flat_output = controller.step(flat_state, obs, t)
        unpacked, unpacked_carry = controller.unpack_state(flat_state)
        assert jnp.array_equal(flat_output, native_output["output"])
        assert eqx.tree_equal((unpacked, unpacked_carry), (native_state, native_carry))

    def step(h):
        return controller.step(h, obs, 2)[0]

    jacobian = jax.jit(jax.jacfwd(step))(flat_state)
    assert jacobian.shape == (3, 3)
    assert jnp.all(jnp.isfinite(jacobian))
    hidden_direction = jnp.ones(2, dtype=jnp.float32)
    base_state, carry = controller.unpack_state(flat_state)
    perturbed = {**carry, hidden_carry_key: carry[hidden_carry_key] + hidden_direction}
    direction = controller.pack_state(base_state, perturbed) - flat_state
    _, tangent = controller.unpack_state(jacobian @ direction)
    assert jnp.allclose(tangent[hidden_carry_key], 0.75 * hidden_direction)


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


@pytest.mark.parametrize("entry", ["nested-spec", "nested-graph", "explicit"])
def test_graph_controller_resets_from_static_inputs(entry):
    spec = GraphSpec(
        nodes={
            "sum": ComponentSpec(type="Sum", input_ports=["a", "b"], output_ports=["output"]),
            "filter": ComponentSpec(
                type="FirstOrderFilter",
                params={"tau_rise": 0.05, "tau_decay": 0.05, "dt": 0.001},
                input_ports=["input"],
                output_ports=["output"],
            ),
        },
        wires=[
            WireSpec(
                source_node="sum",
                source_port="output",
                target_node="sum",
                target_port="b",
                temporality="recurrent",
                recurrent_initializer=recurrent_graph_input_initializer("seed", state_slot="b"),
            ),
            WireSpec(
                source_node="sum", source_port="output", target_node="filter", target_port="input"
            ),
        ],
        input_ports=["input", "seed"],
        output_ports=["output", "filtered"],
        input_bindings={"input": ("sum", "a")},
        output_bindings={"output": ("sum", "output"), "filtered": ("filter", "output")},
    )
    registry = ComponentRegistry(load_user_components=False)
    key = jax.random.key(2)
    seed = jnp.array([3.0, 4.0], dtype=jnp.float32)
    static_inputs = {"seed": seed}
    native = spec_to_graph(spec, registry, {("sum", "b"): seed})
    if entry.startswith("nested"):
        input_bindings = {name: ("child", name) for name in spec.input_ports}
        output_bindings = {name: ("child", name) for name in spec.output_ports}
        native = Graph(
            nodes={"child": native},
            wires=(),
            input_ports=tuple(spec.input_ports),
            output_ports=tuple(spec.output_ports),
            input_bindings=input_bindings,
            output_bindings=output_bindings,
        )
        spec = GraphSpec(
            nodes={
                "child": ComponentSpec(
                    type="Subgraph", input_ports=spec.input_ports, output_ports=spec.output_ports
                )
            },
            subgraphs={"child": spec},
            input_ports=spec.input_ports,
            output_ports=spec.output_ports,
            input_bindings=input_bindings,
            output_bindings=output_bindings,
        )

    cycle_init = {("sum", "b"): -seed} if entry == "explicit" else None
    controller = graph_controller(
        spec if entry == "nested-spec" else native,
        key=key,
        component_registry=registry if entry == "nested-spec" else None,
        static_inputs=static_inputs,
        cycle_init=cycle_init,
        dtype=jnp.float32,
    )
    h0 = controller.initial_state()
    _, carry = controller.unpack_state(h0)
    (reset_value,) = jax.tree.leaves(carry)
    assert jnp.array_equal(reset_value, -seed if entry == "explicit" else seed)

    value = jnp.array([1.0, 2.0], dtype=jnp.float32)
    inputs = {**static_inputs, "input": value}
    native_state = native.init_state(key=key)
    native_carry = (
        None
        if cycle_init is None
        else native.initial_cycle_port_values(native_state, cycle_init, inputs=inputs)
    )
    native_outputs, _, native_next_carry = native.step(
        inputs, native_state, native_carry, key=jax.random.fold_in(key, 0)
    )
    result = controller.step_with_trace(h0, value, 0)
    _, adapted_next_carry = controller.unpack_state(result.state)
    assert jax.tree.all(jax.tree.map(jnp.array_equal, result.graph_outputs, native_outputs))
    assert jax.tree.all(jax.tree.map(jnp.array_equal, adapted_next_carry, native_next_carry))


def test_graph_controller_seeds_noise_reset_from_first_transition_key():
    graph = Graph(
        nodes={"noise": Noise(shape=(4,)), "sum": Sum()},
        wires=(
            Wire(
                "sum",
                "output",
                "sum",
                "b",
                temporality="recurrent",
                recurrent_initializer=recurrent_node_output_initializer(
                    "noise", "output", state_slot="b"
                ),
            ),
        ),
        input_ports=("input",),
        output_ports=("output", "live_noise"),
        input_bindings={"input": ("sum", "a")},
        output_bindings={"output": ("sum", "output"), "live_noise": ("noise", "output")},
    )
    value = jnp.zeros(4, dtype=jnp.float32)
    first_key = jax.random.key(21)
    second_key = jax.random.key(34)

    def custom_policy(key, t):
        return jax.random.fold_in(key, t + 17)

    def check_reset(key, *, custom=False):
        controller = graph_controller(
            graph,
            key=key,
            dtype=jnp.float32,
            **({"key_policy": custom_policy} if custom else {}),
        )
        h0 = controller.initial_state()
        _, carry = controller.unpack_state(h0)
        key_t = custom_policy(key, 0) if custom else key
        native_outputs, _, _ = graph.step({"input": value}, graph.init_state(key=key), key=key_t)
        result = controller.step_with_trace(h0, value, 0)
        assert jnp.array_equal(carry["sum", "b"], native_outputs["output"])
        assert jax.tree.all(jax.tree.map(jnp.array_equal, result.graph_outputs, native_outputs))
        return carry["sum", "b"]

    first = check_reset(first_key)
    assert jnp.array_equal(first, check_reset(first_key))
    assert not jnp.array_equal(first, check_reset(second_key))
    assert not jnp.array_equal(first, check_reset(first_key, custom=True))
