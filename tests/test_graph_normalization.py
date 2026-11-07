from __future__ import annotations

from typing import ClassVar

from feedbax.component_registry import ComponentPack
from feedbax.contracts.component import DynamicPortPolicy

from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.graph import ComponentSpec, GraphSpec
from feedbax.compiler.normalization import normalize_graph_for_studio_authoring


def test_normalization_preserves_revision() -> None:
    graph = GraphSpec(
        nodes={
            "channel": ComponentSpec(
                type="FeedbackChannel",
                params={"input_key": "feedback"},
                revision=1,
                input_ports=["input"],
                output_ports=["output"],
            )
        }
    )

    normalized = normalize_graph_for_studio_authoring(graph)

    channel = normalized.nodes["channel"]
    assert channel.type == "Channel"
    assert channel.revision == 1


def test_normalization_preserves_nested_revision() -> None:
    graph = GraphSpec(
        nodes={
            "outer": ComponentSpec(
                type="PenzaiSubgraph",
                params={},
                revision=1,
            )
        },
        subgraphs={
            "outer": GraphSpec(
                nodes={
                    "channel": ComponentSpec(
                        type="FeedbackChannel",
                        params={"input_key": "feedback"},
                        revision=1,
                    )
                }
            )
        },
    )

    normalized = normalize_graph_for_studio_authoring(graph)

    assert normalized.nodes["outer"].type == "PenzaiAdapter"
    assert normalized.nodes["outer"].revision == 1
    nested_channel = normalized.subgraphs["outer"].nodes["channel"]
    assert nested_channel.type == "Channel"
    assert nested_channel.revision == 1


def test_normalization_materializes_external_dynamic_ports_from_explicit_registry() -> None:
    from types import ModuleType

    from feedbax.contracts.component import PortType
    from feedbax.contracts.portable import Ports
    from feedbax.runtime.graph import Component

    module = ModuleType("test_normalization_owners")

    class VariableOutputs(Component):
        sizes: list[int]
        ports: ClassVar = Ports(
            inputs={"input": PortType(dtype="vector")},
            outputs={},
            dynamic=DynamicPortPolicy.model_validate({
                "count_param": "sizes",
                "count_mode": "sequence_length",
                "direction": "output",
                "fixed_input_ports": ["input"],
                "generated_name_template": "result_{index}",
                "dynamic_port_type": {"dtype": "vector"},
            }),
        )

        def __call__(self, inputs, state, *, key):
            return {}, state

    VariableOutputs.__module__ = module.__name__
    module.VariableOutputs = VariableOutputs
    registry = ComponentRegistry()
    registry.ingest(ComponentPack(namespace="example", modules=(module,)))
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
