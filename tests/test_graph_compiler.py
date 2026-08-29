from __future__ import annotations

import pytest
from pydantic import ValidationError

from feedbax.compiler import (
    CompilationRecord,
    GraphDocument,
    ResolvedGraph,
    compile_graph,
)
from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.graph import ComponentSpec, GraphSpec, WireSpec
from feedbax.contracts.graph import (
    AdditiveGraphChannelAdapterSpec,
    AdditiveGraphChannelTargetSpec,
)


def _document() -> GraphDocument:
    return GraphDocument(
        graph=GraphSpec(
            nodes={
                "source": ComponentSpec(
                    type="Constant",
                    params={"value": 2.0},
                    output_ports=["output"],
                ),
                "gain": ComponentSpec(
                    type="Gain",
                    params={"gain": 3.0},
                    input_ports=["input"],
                    output_ports=["output"],
                ),
            },
            wires=[
                WireSpec(
                    source_node="source",
                    source_port="output",
                    target_node="gain",
                    target_port="input",
                )
            ],
            output_ports=["output"],
            output_bindings={"output": ("gain", "output")},
        )
    )


def test_compile_graph_is_deterministic_and_records_runtime_key_order() -> None:
    registry = ComponentRegistry(load_user_components=False)

    first = compile_graph(_document(), registry)
    second = compile_graph(_document(), registry)

    assert first.resolved == second.resolved
    assert first.record == second.record
    assert first.resolved.key_schedule.node_order == first.graph._execution_order
    assert first.resolved.source_map.entries[1].authored_path == "/graph/nodes/gain"
    assert ResolvedGraph.model_validate_json(first.resolved.model_dump_json()) == first.resolved
    assert CompilationRecord.model_validate_json(first.record.model_dump_json()) == first.record


def test_graph_document_rejects_unknown_version_without_restamping() -> None:
    payload = _document().model_dump(mode="json")
    payload["schema_version"] = "0"

    with pytest.raises(ValidationError, match="migration_intentionally_absent=yes"):
        GraphDocument.model_validate(payload)


def test_compile_graph_migrates_an_explicit_older_graph_schema() -> None:
    payload = _document().graph.model_dump(mode="json", exclude_none=True)
    payload["schema_version"] = "feedbax.spec.graph.v4"

    compiled = compile_graph(
        GraphDocument(graph=payload),
        ComponentRegistry(load_user_components=False),
    )

    assert compiled.record.graph_schema_source_version == "feedbax.spec.graph.v4"
    assert compiled.record.graph_schema_target_version == "feedbax.spec.graph.v5"
    assert compiled.record.migration_record_count == 1


def test_compiler_generated_adapter_has_truthful_source_map_origin() -> None:
    document = _document()
    graph = document.graph.model_copy(
        update={
            "additive_channel_adapters": [
                AdditiveGraphChannelAdapterSpec(
                    label="disturbance",
                    input_key="disturbance",
                    target=AdditiveGraphChannelTargetSpec(
                        kind="edge",
                        source_node="source",
                        source_port="output",
                        target_node="gain",
                        target_port="input",
                    ),
                    payload_shape=[],
                )
            ]
        }
    )

    compiled = compile_graph(
        GraphDocument(graph=graph),
        ComponentRegistry(load_user_components=False),
    )

    entry = next(
        item
        for item in compiled.record.source_map.entries
        if item.resolved_path == "/graph/nodes/disturbance_additive"
    )
    assert entry.origin == "compiler-generated"
    assert entry.authored_path == "/graph"
