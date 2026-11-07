from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from feedbax.component_registry import ComponentRegistry
from feedbax.compiler.serialization import graph_to_spec
from feedbax.contracts.graph import ComponentSpec, GraphSpec
from feedbax.runtime.components import Constant, Pulse, Ramp, Sine
from feedbax.runtime.graph import init_state_from_component
from feedbax.runtime.iteration import run_component
from feedbax.web.worker.execution import _compile_trainability
from tests.support.graph_compiler_test_support import spec_to_graph


@pytest.mark.parametrize(
    ("source", "fields", "timing", "expected"),
    [
        (Constant, ("value",), {}, [[1, 2], [1, 2], [1, 2]]),
        (Ramp, ("slope", "intercept"), {"dt": 0.5}, [[1, 2], [1.5, 3], [2, 4]]),
        (Sine, ("amplitude", "offset"), {"dt": 0.25}, [[1, 2], [2, 4], [1, 2]]),
        (Pulse, ("amplitude", "offset"), {"dt": 0.25}, [[2, 4], [2, 4], [1, 2]]),
    ],
)
def test_vector_signal_values_change_without_retracing(source, fields, timing, expected) -> None:
    @eqx.filter_jit
    @eqx.debug.assert_max_traces(max_traces=1)
    def rollout(component, state, key):
        outputs, _, _ = run_component(component, {}, state, key=key, n_steps=3)
        return outputs["output"]

    key = jax.random.key(0)
    for scale in (1, 3):
        value = jnp.array([scale, 2 * scale], dtype=jnp.float32)
        component = source(**dict.fromkeys(fields, value), **timing).prepare()
        output = rollout(component, component.init_state(key=key), key)
        np.testing.assert_allclose(output, np.asarray(expected) * scale, atol=1e-6)
        assert output.dtype == value.dtype


@pytest.mark.parametrize("x64", [False, True])
def test_waveforms_keep_their_own_clocks_across_scanned_rollouts(x64: bool) -> None:
    with jax.experimental.enable_x64(x64):
        nodes = {
            "constant": ComponentSpec(
                type="feedbax.Constant",
                params={"value": 0.5},
                output_ports=["output"],
            ),
            "ramp": ComponentSpec(
                type="feedbax.Ramp",
                params={"slope": 4.0, "intercept": -1.0, "dt": 0.125},
                output_ports=["output"],
            ),
            "sine": ComponentSpec(
                type="feedbax.Sine",
                params={"amplitude": 2.0, "frequency": 1.0, "offset": 3.0, "dt": 0.25},
                output_ports=["output"],
            ),
            "pulse": ComponentSpec(
                type="feedbax.Pulse",
                params={
                    "amplitude": 3.0,
                    "period": 0.1,
                    "duty_cycle": 0.5,
                    "offset": -1.0,
                    "dt": 0.01,
                },
                output_ports=["output"],
            ),
        }
        spec = GraphSpec(
            nodes=nodes,
            output_ports=list(nodes),
            output_bindings={name: (name, "output") for name in nodes},
        )
        registry = ComponentRegistry()
        graph = spec_to_graph(spec, registry)
        serialized = graph_to_spec(graph, registry)
        for name, node in nodes.items():
            actual = serialized.nodes[name]
            assert actual.type == node.type
            assert {field: actual.params[field] for field in node.params} == node.params
        graph = spec_to_graph(serialized, registry)
        _, trainable_filter = _compile_trainability(serialized, graph, registry)
        assert not jax.tree.leaves(eqx.filter(graph, trainable_filter))

        key = jax.random.key(0)
        expected = {
            "constant": [0.5] * 11,
            "ramp": [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
            "sine": [3.0, 5.0, 3.0, 1.0, 3.0, 5.0, 3.0, 1.0, 3.0, 5.0, 3.0],
            "pulse": [2.0] * 5 + [-1.0] * 5 + [2.0],
        }

        eager_state = init_state_from_component(graph)
        for step in range(11):
            eager, eager_state, _ = graph.step({}, eager_state, key=key)
            for name, sample in eager.items():
                np.testing.assert_allclose(sample, expected[name][step], atol=2e-6)

        first, state, _ = graph.rollout_with_trace(
            {}, init_state_from_component(graph), key=key, n_steps=2, trace=()
        )
        resumed, state, _ = graph.rollout_with_trace({}, state, key=key, n_steps=9, trace=())
        for name, samples in expected.items():
            np.testing.assert_allclose(first[name], samples[:2], atol=2e-6)
            np.testing.assert_allclose(resumed[name], samples[2:], atol=2e-6)
            assert first[name].dtype == resumed[name].dtype == jnp.asarray(0.0).dtype
        for name in ("ramp", "sine", "pulse"):
            step = state.get(graph.nodes[name].step_index)
            assert step.shape == ()
            assert jnp.issubdtype(step.dtype, jnp.integer)
            assert step == 11


def test_waveform_clock_preserves_explicit_output_precision() -> None:
    value = jnp.asarray(1.0, dtype=jnp.float16)
    components = (
        Ramp(slope=value, intercept=value),
        Sine(amplitude=value, offset=value),
        Pulse(amplitude=value, offset=value),
    )
    key = jax.random.key(0)
    for node in components:
        step = partial(node, {}, key=key)
        initial_state = init_state_from_component(node)
        outputs, state = step(initial_state)
        assert outputs["output"].dtype == value.dtype
        outputs, _ = jax.jit(step)(state)
        assert outputs["output"].dtype == value.dtype


@pytest.mark.parametrize("policy", ["hold", "error"])
def test_affine_schedule_policy_applies_when_resuming_a_scanned_rollout(policy: str) -> None:
    spec = GraphSpec(
        nodes={
            "controller": ComponentSpec(
                type="feedbax.AffineFeedbackController",
                params={
                    "gain": [[[1.0]], [[2.0]], [[3.0]]],
                    "bias": [[1.0], [2.0]],
                    "feedforward": [[10.0], [20.0], [30.0], [40.0]],
                    "schedule_policy": policy,
                },
                input_ports=["feedback", "reference", "feedforward"],
                output_ports=["command"],
            )
        },
        input_ports=["feedback"],
        output_ports=["command"],
        input_bindings={"feedback": ("controller", "feedback")},
        output_bindings={"command": ("controller", "command")},
    )
    graph = spec_to_graph(spec, ComponentRegistry())
    key = jax.random.key(0)

    first, state, _ = graph.rollout_with_trace(
        {"feedback": jnp.full((2, 1), 2.0)},
        init_state_from_component(graph),
        key=key,
        n_steps=2,
        trace=(),
    )
    np.testing.assert_array_equal(first["command"], [[13.0], [26.0]])

    if policy == "hold":
        resumed, _, _ = graph.rollout_with_trace(
            {"feedback": jnp.full((3, 1), 2.0)}, state, key=key, n_steps=3, trace=()
        )
        np.testing.assert_array_equal(resumed["command"], [[38.0], [48.0], [48.0]])
    else:
        with pytest.raises(
            (RuntimeError, ValueError),
            match="AffineFeedbackController bias schedule length 2 was exhausted",
        ):
            resumed, _, _ = graph.rollout_with_trace(
                {"feedback": jnp.full((1, 1), 2.0)}, state, key=key, n_steps=1, trace=()
            )
            jax.block_until_ready(resumed)
        with pytest.raises(ValueError, match="bias schedule length 2 was exhausted at step 2"):
            graph.step({"feedback": jnp.array([2.0])}, state, key=key)
