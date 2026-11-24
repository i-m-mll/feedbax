"""Exercise published Model layouts for source and delayed Channel owners."""

import jax
import jax.numpy as jnp
import numpy as np

from feedbax.compiler import GraphDocument, compile_graph
from feedbax.compiler.model_document import lower_model_document
from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.model_document import ModelDocument
from feedbax.contracts.model_values import model_owner_problem
from feedbax.contracts.semantic_values import semantic_key
from feedbax.runtime.raw_state import bind_raw_state


def test_noise_source_can_drive_an_authored_model():
    model = ModelDocument.model_validate({
        "step_seconds": 0.01,
        "ports": {"outputs": [{"id": "sample", "name": "sample", "type": {"shape": [2]}}]},
        "interior": {
            "components": {"noise": {
                "kind": "leaf", "name": "Noise", "type": "feedbax.Noise",
                "fields": {"mean": 2.0, "std": 0.5, "shape": [2]},
            }},
            "wires": {"sample": {
                "source": {"owner": "component", "component": "noise", "port": "output"},
                "target": {"owner": "outputs", "port": "sample"},
            }},
        },
    })
    components = ComponentRegistry()
    lowered = lower_model_document(model, components)
    assert model_owner_problem(lowered.graph, "checkpoint", components) is None
    compiled = compile_graph(GraphDocument(graph=lowered.graph), components)
    key = jax.random.key(51)
    graph = compiled.graph
    initial = graph.init_state(key=key)
    first, _ = graph({}, initial, key=key)
    repeated, _ = graph({}, graph.init_state(key=key), key=key)
    changed, _ = graph({}, graph.init_state(key=key), key=jax.random.key(52))
    assert first["sample"].shape == (2,)
    np.testing.assert_array_equal(first["sample"], repeated["sample"])
    assert not np.array_equal(first["sample"], changed["sample"])


def test_trial_channel_reset_sets_every_delay_slot_and_records_noise():
    model = ModelDocument.model_validate({
        "step_seconds": 0.01,
        "ports": {
            "inputs": [{
                "id": "signal", "name": "signal", "type": {"shape": [2]},
                "default": {"value": [2.0, 3.0]},
            }],
            "outputs": [{"id": "sample", "name": "sample", "type": {"shape": [2]}}],
        },
        "interior": {
            "components": {"delay": {
                "kind": "leaf", "name": "Channel", "type": "feedbax.Channel",
                "fields": {"delay": 2, "noise_std": 1.0},
            }},
            "wires": {
                "incoming": {
                    "source": {"owner": "inputs", "port": "signal"},
                    "target": {"owner": "component", "component": "delay", "port": "input"},
                },
                "sample": {
                    "source": {"owner": "component", "component": "delay", "port": "output"},
                    "target": {"owner": "outputs", "port": "sample"},
                },
            },
        },
    })
    components = ComponentRegistry()
    lowered = lower_model_document(model, components)
    compiled = compile_graph(GraphDocument(graph=lowered.graph), components)
    graph = compiled.graph
    key = jax.random.key(43)
    marker = jnp.array([4.0, -5.0])
    reset = lowered.interface(model, components).reset.prepare(
        graph, {ModelAddress.at_state((), "delay", "output").to_text(): marker}, key=key,
    )
    layout = bind_raw_state(graph, compiled.resolved.graph)

    def recorded(symbol, state):
        return layout.capture(state, None)[semantic_key("model", "state", ("delay",), symbol)]

    np.testing.assert_array_equal(reset.outputs["sample"], marker)
    for slot in range(2):
        np.testing.assert_array_equal(recorded(f"queue/[{slot}]", reset.state), marker)
    np.testing.assert_array_equal(recorded("noise", reset.state), [0.0, 0.0])
    outputs, state = graph({}, reset.state, key=key)
    noise = recorded("noise", state)
    assert np.any(noise != 0)
    np.testing.assert_array_equal(outputs["sample"], marker + noise)
    np.testing.assert_array_equal(recorded("queue/[1]", state), [2.0, 3.0])
