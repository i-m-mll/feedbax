from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp
from pydantic import ValidationError

from feedbax.compiler import (
    CompilationRecord,
    GraphDocument,
    ResolvedGraph,
    compile_graph,
)
from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.graph import (
    ComponentSpec,
    GraphSpec,
    GraphUIState,
    NodeUIState,
    SemanticAnchor,
    WorkspaceDocument,
    WireSpec,
)
from feedbax.contracts.graph import (
    AdditiveGraphChannelAdapterSpec,
    AdditiveGraphChannelTargetSpec,
)
from feedbax.runtime.graph import init_state_from_component


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


def test_source_map_anchor_schema_rejects_pre_anchor_compiler_records() -> None:
    compiled = compile_graph(_document(), ComponentRegistry(load_user_components=False))
    resolved_payload = compiled.resolved.model_dump(mode="json")
    resolved_payload["schema_version"] = "1"
    record_payload = compiled.record.model_dump(mode="json")
    record_payload["schema_version"] = "1"

    with pytest.raises(ValidationError, match="migration_intentionally_absent=yes"):
        ResolvedGraph.model_validate(resolved_payload)
    with pytest.raises(ValidationError, match="migration_intentionally_absent=yes"):
        CompilationRecord.model_validate(record_payload)


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


def test_workspace_view_edits_cannot_change_semantic_or_runtime_identity() -> None:
    registry = ComponentRegistry(load_user_components=False)
    first = compile_graph(_document(), registry)
    root = SemanticAnchor(
        semantic_document_sha256=first.record.document_sha256,
        authored_path="/graph",
    )
    initial_workspace = WorkspaceDocument(semantic_root=root)
    moved_workspace = initial_workspace.model_copy(
        update={
            "graph_ui_state": GraphUIState(
                node_states={
                    "gain": NodeUIState(position={"x": 640, "y": 360})
                }
            )
        }
    )

    second = compile_graph(_document(), registry)
    first_output, _ = first.graph(
        {}, init_state_from_component(first.graph), key=jax.random.PRNGKey(0)
    )
    second_output, _ = second.graph(
        {}, init_state_from_component(second.graph), key=jax.random.PRNGKey(0)
    )

    assert moved_workspace != initial_workspace
    assert first.record.document_sha256 == second.record.document_sha256
    assert first.record.resolved_sha256 == second.record.resolved_sha256
    assert first.record == second.record
    assert first.record.key_schedule == second.record.key_schedule
    assert jnp.array_equal(first_output["output"], second_output["output"])


def test_graph_document_and_compiler_reject_presentation_state() -> None:
    payload = _document().model_dump(mode="json")
    payload["workspace_document"] = {
        "schema_id": "feedbax.workspace_document",
        "schema_version": "1",
    }

    with pytest.raises(ValidationError, match="extra_forbidden"):
        GraphDocument.model_validate(payload)
