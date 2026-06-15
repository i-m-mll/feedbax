from __future__ import annotations

import jax
import jax.numpy as jnp

from feedbax.contracts.graph import (
    AdditiveGraphChannelAdapterSpec,
    AdditiveGraphChannelTargetSpec,
    ComponentSpec,
    GraphSpec,
    WireSpec,
)
from feedbax.runtime.graph import init_state_from_component
from feedbax.runtime.graph_channel_adapters import materialize_additive_channel_adapters
from feedbax.integrations.provider import validate_graph_spec
from feedbax.contracts.graphs.serialization import spec_to_graph


def test_additive_channel_adapter_spec_round_trips_named_channel_metadata() -> None:
    spec = GraphSpec(
        additive_channel_adapters=[
            AdditiveGraphChannelAdapterSpec(
                label="command_input",
                input_key="perturbation.command",
                target=AdditiveGraphChannelTargetSpec(
                    kind="edge",
                    source_node="controller",
                    source_port="output",
                    target_node="mechanics",
                    target_port="force",
                ),
                payload_shape=[2],
                payload_dtype="float32",
                provenance_role="perturbation_input",
                metadata={"retained_observable": "edge:controller.output->mechanics.force"},
            )
        ]
    )

    round_tripped = GraphSpec.model_validate(spec.model_dump(mode="json"))
    adapter = round_tripped.additive_channel_adapters[0]

    assert adapter.label == "command_input"
    assert adapter.input_key == "perturbation.command"
    assert adapter.payload_shape == [2]
    assert adapter.payload_dtype == "float32"
    assert adapter.provenance_role == "perturbation_input"
    assert adapter.target.source_node == "controller"
    assert adapter.target.target_port == "force"


def test_edge_additive_channel_adapter_materializes_sum_and_runs() -> None:
    spec = GraphSpec(
        nodes={
            "source": ComponentSpec(
                type="Constant",
                params={"value": 1.0},
                input_ports=[],
                output_ports=["output"],
            ),
            "readout": ComponentSpec(
                type="Gain",
                params={"gain": 2.0},
                input_ports=["input"],
                output_ports=["output"],
            ),
        },
        wires=[
            WireSpec(
                source_node="source",
                source_port="output",
                target_node="readout",
                target_port="input",
            )
        ],
        additive_channel_adapters=[
            AdditiveGraphChannelAdapterSpec(
                label="sensory_feedback",
                input_key="perturbation.feedback",
                target=AdditiveGraphChannelTargetSpec(
                    kind="edge",
                    source_node="source",
                    source_port="output",
                    target_node="readout",
                    target_port="input",
                ),
                payload_shape=[2],
                provenance_role="perturbation_input",
            )
        ],
        output_ports=["output"],
        output_bindings={"output": ("readout", "output")},
    )

    materialized = materialize_additive_channel_adapters(spec)
    adapter_node = materialized.nodes["sensory_feedback_additive"]

    assert materialized.additive_channel_adapters == []
    assert materialized.input_bindings["perturbation.feedback"] == (
        "sensory_feedback_additive",
        "b",
    )
    assert adapter_node.type == "Sum"
    assert adapter_node.params["channel_adapter"]["payload_shape"] == [2]
    assert {
        (wire.source_node, wire.source_port, wire.target_node, wire.target_port)
        for wire in materialized.wires
    } == {
        ("source", "output", "sensory_feedback_additive", "a"),
        ("sensory_feedback_additive", "output", "readout", "input"),
    }

    graph = spec_to_graph(spec, {})
    state = init_state_from_component(graph)
    outputs, _ = graph(
        {"perturbation.feedback": jnp.asarray(3.0)},
        state,
        key=jax.random.PRNGKey(0),
    )

    assert jnp.array_equal(outputs["output"], jnp.asarray(8.0))
    assert validate_graph_spec(spec).valid


def test_input_additive_channel_adapter_wraps_existing_input_binding() -> None:
    spec = GraphSpec(
        nodes={
            "readout": ComponentSpec(
                type="Gain",
                params={"gain": 1.0},
                input_ports=["input"],
                output_ports=["output"],
            )
        },
        additive_channel_adapters=[
            AdditiveGraphChannelAdapterSpec(
                label="command_input",
                input_key="perturbation.command",
                target=AdditiveGraphChannelTargetSpec(
                    kind="input",
                    target_node="readout",
                    target_port="input",
                ),
                payload_shape=[2],
            )
        ],
        input_ports=["command"],
        output_ports=["output"],
        input_bindings={"command": ("readout", "input")},
        output_bindings={"output": ("readout", "output")},
    )

    graph = spec_to_graph(spec, {})
    state = init_state_from_component(graph)
    outputs, _ = graph(
        {
            "command": jnp.asarray([1.0, 2.0]),
            "perturbation.command": jnp.asarray([3.0, 5.0]),
        },
        state,
        key=jax.random.PRNGKey(0),
    )

    assert graph.input_bindings["command"] == ("command_input_additive", "a")
    assert graph.input_bindings["perturbation.command"] == ("command_input_additive", "b")
    assert jnp.array_equal(outputs["output"], jnp.asarray([4.0, 7.0]))
    assert validate_graph_spec(spec).valid


def test_input_additive_channel_adapter_binds_direct_model_input() -> None:
    spec = GraphSpec(
        nodes={
            "mechanics": ComponentSpec(
                type="LinearStateSpace",
                params={
                    "A": [[1.0, 0.0], [0.0, 1.0]],
                    "B": [[1.0], [0.0]],
                    "B_w": [[0.0], [2.0]],
                    "initial_state": [0.0, 0.0],
                    "pos_slice": [0, 1],
                    "vel_slice": [1, 2],
                },
                input_ports=["force", "epsilon"],
                output_ports=["effector", "state"],
            )
        },
        additive_channel_adapters=[
            AdditiveGraphChannelAdapterSpec(
                label="process_epsilon",
                input_key="perturbation.epsilon",
                target=AdditiveGraphChannelTargetSpec(
                    kind="input",
                    target_node="mechanics",
                    target_port="epsilon",
                ),
                payload_shape=[1],
                provenance_role="process_disturbance",
            )
        ],
        input_ports=["force"],
        output_ports=["state"],
        input_bindings={"force": ("mechanics", "force")},
        output_bindings={"state": ("mechanics", "state")},
    )

    graph = spec_to_graph(spec, {})
    state = init_state_from_component(graph)
    outputs, _ = graph(
        {
            "force": jnp.asarray([3.0]),
            "perturbation.epsilon": jnp.asarray([4.0]),
        },
        state,
        key=jax.random.PRNGKey(0),
    )

    assert graph.input_bindings["perturbation.epsilon"] == ("mechanics", "epsilon")
    assert jnp.array_equal(outputs["state"], jnp.asarray([3.0, 8.0]))
    assert validate_graph_spec(spec).valid
