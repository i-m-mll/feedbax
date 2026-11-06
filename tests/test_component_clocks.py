from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.graph import ComponentSpec, GraphSpec
from feedbax.runtime.components import Pulse, Ramp, Sine
from feedbax.runtime.graph import init_state_from_component
from tests.support.graph_compiler_test_support import spec_to_graph


@pytest.mark.parametrize("x64", [False, True])
def test_waveforms_keep_their_own_clocks_across_scanned_rollouts(x64: bool) -> None:
    with jax.experimental.enable_x64(x64):
        nodes = {
            "ramp": ComponentSpec(
                type="Ramp",
                params={"slope": 4.0, "intercept": -1.0, "dt": 0.125},
                output_ports=["output"],
            ),
            "sine": ComponentSpec(
                type="Sine",
                params={"amplitude": 2.0, "frequency": 1.0, "offset": 3.0, "dt": 0.25},
                output_ports=["output"],
            ),
            "pulse": ComponentSpec(
                type="Pulse",
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
        graph = spec_to_graph(spec, ComponentRegistry(load_user_components=False))
        key = jax.random.key(0)
        expected = {
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
                type="AffineFeedbackController",
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
    graph = spec_to_graph(spec, ComponentRegistry(load_user_components=False))
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
