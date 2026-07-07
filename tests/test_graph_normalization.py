from __future__ import annotations

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
    assert (
        normalized.nodes["outer"].param_schema_version
        == "feedbax.component.penzai_subgraph.v1"
    )
    nested_channel = normalized.subgraphs["outer"].nodes["channel"]
    assert nested_channel.type == "Channel"
    assert nested_channel.param_schema_version == "feedbax.component.feedback_channel.v1"
