from __future__ import annotations

from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.graph import ComponentSpec, GraphSpec
from feedbax.contracts.graphs.normalization import normalize_graph_for_studio_authoring


def test_normalization_preserves_param_schema_version() -> None:
    graph = GraphSpec(
        nodes={
            "channel": ComponentSpec(
                type="FeedbackChannel",
                params={"input_key": "feedback"},
                param_schema_version="feedbax.component.feedback_channel.v1",
                input_ports=["input"],
                output_ports=["output"],
            )
        }
    )

    normalized = normalize_graph_for_studio_authoring(graph)

    channel = normalized.nodes["channel"]
    assert channel.type == "Channel"
    assert channel.param_schema_version == "feedbax.component.feedback_channel.v1"


def test_normalization_preserves_nested_param_schema_version() -> None:
    graph = GraphSpec(
        nodes={
            "outer": ComponentSpec(
                type="PenzaiSubgraph",
                params={},
                param_schema_version="feedbax.component.penzai_subgraph.v1",
            )
        },
        subgraphs={
            "outer": GraphSpec(
                nodes={
                    "channel": ComponentSpec(
                        type="FeedbackChannel",
                        params={"input_key": "feedback"},
                        param_schema_version="feedbax.component.feedback_channel.v1",
                    )
                }
            )
        },
    )

    normalized = normalize_graph_for_studio_authoring(graph)

    assert normalized.nodes["outer"].type == "PenzaiAdapter"
    assert normalized.nodes["outer"].param_schema_version == "feedbax.component.penzai_subgraph.v1"
    nested_channel = normalized.subgraphs["outer"].nodes["channel"]
    assert nested_channel.type == "Channel"
    assert nested_channel.param_schema_version == "feedbax.component.feedback_channel.v1"


def test_normalization_materializes_external_dynamic_ports_from_explicit_registry() -> None:
    registry = ComponentRegistry(load_user_components=False)
    registry.register_component_type(
        "example.VariableOutputs",
        lambda params: params,
        param_schema=[{"name": "sizes", "type": "array", "default": [1, 1]}],
        input_ports=["input"],
        output_ports=["result_0", "result_1"],
        dynamic_port_policy={
            "count_param": "sizes",
            "count_mode": "sequence_length",
            "direction": "output",
            "fixed_input_ports": ["input"],
            "generated_name_template": "result_{index}",
            "dynamic_port_type": {"dtype": "vector"},
        },
    )
    graph = GraphSpec(
        nodes={
            "external": ComponentSpec(
                type="example.VariableOutputs",
                params={"sizes": [2, 3, 1]},
            )
        }
    )

    without_registry = normalize_graph_for_studio_authoring(graph)
    normalized = normalize_graph_for_studio_authoring(
        graph,
        component_registry=registry,
    )

    assert without_registry.nodes["external"].input_ports == []
    assert without_registry.nodes["external"].output_ports == []
    assert normalized.nodes["external"].input_ports == ["input"]
    assert normalized.nodes["external"].output_ports == [
        "result_0",
        "result_1",
        "result_2",
    ]
