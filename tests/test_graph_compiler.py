from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp
from pydantic import BaseModel, ValidationError, field_validator

from feedbax.compiler import (
    COMPILATION_RECORD_SCHEMA_VERSION,
    CompilationRecord,
    CompilerPhase,
    GraphCompilationError,
    GraphDocument,
    ResolvedGraph,
    compile_graph,
)
from feedbax.component_registry import ComponentMigration, ComponentRegistry
from feedbax.runtime.components import Constant, Gain
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


def test_compile_failure_is_a_stable_source_mapped_phase_diagnostic() -> None:
    document = _document()
    graph = document.graph.model_copy(
        update={
            "nodes": {
                **document.graph.nodes,
                "missing": ComponentSpec(
                    type="tests.missing_component",
                    input_ports=[],
                    output_ports=["output"],
                ),
            }
        }
    )

    with pytest.raises(GraphCompilationError) as caught:
        compile_graph(
            GraphDocument(graph=graph),
            ComponentRegistry(load_user_components=False),
        )

    record = caught.value.record
    diagnostic = record.diagnostics[0]
    assert record.document_sha256 == diagnostic.source_anchor.semantic_document_sha256
    assert diagnostic.code == "compiler.type_resolution.unresolved_component_type"
    assert diagnostic.phase is CompilerPhase.TYPE_RESOLUTION
    assert diagnostic.severity == "error"
    assert diagnostic.source_anchor.authored_path == "/graph/nodes/missing"
    assert diagnostic.expected_condition
    assert "tests.missing_component" in diagnostic.observed_condition
    assert diagnostic.actionable_context["action"]
    assert tuple(CompilerPhase) == (
        CompilerPhase.STRUCTURAL_PARSING,
        CompilerPhase.TYPE_RESOLUTION,
        CompilerPhase.COMPOSITE_AND_ACAUSAL_LOWERING,
        CompilerPhase.CONSTRAINT_SOLVING,
        CompilerPhase.SEMANTIC_VALIDATION,
        CompilerPhase.SCHEDULING,
        CompilerPhase.RESOLVED_IR_EMISSION,
    )


def test_compilation_record_v2_rejects_after_diagnostics_schema_change() -> None:
    compiled = compile_graph(_document(), ComponentRegistry(load_user_components=False))
    payload = compiled.record.model_dump(mode="json")
    payload["schema_version"] = "2"

    assert COMPILATION_RECORD_SCHEMA_VERSION == "5"
    assert compiled.record.diagnostics == ()
    with pytest.raises(ValidationError, match="migration_intentionally_absent=yes"):
        CompilationRecord.model_validate(payload)


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
    adapter = compiled.resolved.graph.nodes["disturbance_additive"]
    binding = compiled.graph.component_bindings["disturbance_additive"]
    assert adapter.param_schema_version == binding.param_schema_version == "1"


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
                node_states={"gain": NodeUIState(position={"x": 640, "y": 360})}
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


@pytest.mark.parametrize("nested", [False, True])
def test_migrated_component_semantics_are_the_executed_graph(nested: bool) -> None:
    registry = ComponentRegistry(load_user_components=False)
    registry.register_migration(
        ComponentMigration(
            source_type="LegacyGain",
            target_type="Gain",
            owner="feedbax",
            source_param_schema_version="legacy-scale",
            target_param_schema_version="1",
            migration_id="tests.legacy-gain-to-current",
            migrate_params=lambda params: {"gain": params["scale"]},
        )
    )
    child = GraphSpec(
        nodes={
            "gain": ComponentSpec(
                type="LegacyGain",
                params={"scale": 8.0},
                param_schema_version="legacy-scale",
                input_ports=["input"],
                output_ports=["output"],
            )
        },
        input_ports=["input"],
        input_bindings={"input": ("gain", "input")},
        output_ports=["output"],
        output_bindings={"output": ("gain", "output")},
    )
    spec = (
        GraphSpec(
            nodes={
                "nested": ComponentSpec(
                    type="Subgraph",
                    input_ports=["input"],
                    output_ports=["output"],
                )
            },
            subgraphs={"nested": child},
            input_ports=["input"],
            input_bindings={"input": ("nested", "input")},
            output_ports=["output"],
            output_bindings={"output": ("nested", "output")},
        )
        if nested
        else child
    )
    compiled = compile_graph(GraphDocument(graph=spec), registry)
    semantic = compiled.resolved.graph.subgraphs["nested"] if nested else compiled.resolved.graph
    runtime = compiled.graph.nodes["nested"] if nested else compiled.graph
    if nested:
        entry = next(
            item
            for item in compiled.resolved.source_map.entries
            if item.resolved_path == "/graph/subgraphs/nested/nodes/gain"
        )
        assert entry.authored_path == "/graph/subgraphs/nested/nodes/gain"
        assert entry.origin == "authored"
    node = semantic.nodes["gain"]
    binding = runtime.component_bindings["gain"]
    assert (node.type, node.param_schema_version, node.params) == ("Gain", "1", {"gain": 8.0})
    assert (binding.type_id, binding.param_schema_version) == (node.type, node.param_schema_version)
    outputs, _ = compiled.graph(
        {"input": jnp.asarray(3.0)},
        init_state_from_component(compiled.graph),
        key=jax.random.PRNGKey(0),
    )
    assert float(outputs["output"]) == 24.0
    assert child.nodes["gain"].type == "LegacyGain"


def test_migrations_precede_inference_and_defaults_follow_derived_dimensions() -> None:
    registry = ComponentRegistry(load_user_components=False)
    registry.register_component_type(
        "VectorSource",
        builder=lambda params: Constant(jnp.ones(params["width"])),
        param_schema=[{"name": "width", "type": "int", "default": 3}],
        output_ports=["output"],
        output_prototype_fn=lambda params, inputs: {"output": jnp.ones(params["width"])},
    )
    registry.register_migration(
        ComponentMigration(
            source_type="LegacyVectorSource",
            target_type="VectorSource",
            owner="feedbax",
            target_param_schema_version="1",
            migration_id="tests.legacy-vector-to-current",
            migrate_params=lambda params: {"width": params["n"]},
        )
    )
    graph = GraphSpec(
        nodes={
            "source": ComponentSpec(
                type="LegacyVectorSource", params={"n": 5}, output_ports=["output"]
            ),
            "linear": ComponentSpec(
                type="Linear",
                params={"output_size": 2, "initialization": "zeros"},
                input_ports=["input"],
                output_ports=["output"],
            ),
        },
        wires=[
            WireSpec(
                source_node="source",
                source_port="output",
                target_node="linear",
                target_port="input",
            )
        ],
        output_ports=["output"],
        output_bindings={"output": ("linear", "output")},
        derived_dimensions=[{"node": "linear", "param": "input_size", "port": "input"}],
    )
    compiled = compile_graph(GraphDocument(graph=graph), registry)
    params = compiled.resolved.graph.nodes["linear"].params
    runtime = compiled.graph.nodes["linear"]
    assert params["input_size"] == runtime.input_size == 5
    assert params["use_bias"] == runtime.use_bias is True
    outputs, _ = compiled.graph(
        {}, init_state_from_component(compiled.graph), key=jax.random.PRNGKey(0)
    )
    assert jnp.array_equal(outputs["output"], jnp.zeros(2))
    assert "input_size" not in graph.nodes["linear"].params


def test_unknown_component_version_points_to_nested_authored_node() -> None:
    child = _document().graph
    child = child.model_copy(
        update={
            "nodes": {
                **child.nodes,
                "gain": child.nodes["gain"].model_copy(update={"param_schema_version": "unknown"}),
            }
        }
    )
    graph = GraphSpec(
        nodes={"nested": ComponentSpec(type="Subgraph", output_ports=["output"])},
        subgraphs={"nested": child},
        output_ports=["output"],
        output_bindings={"output": ("nested", "output")},
    )
    with pytest.raises(GraphCompilationError) as caught:
        compile_graph(GraphDocument(graph=graph), ComponentRegistry(load_user_components=False))
    diagnostic = caught.value.record.diagnostics[0]
    assert diagnostic.source_anchor.authored_path == "/graph/subgraphs/nested/nodes/gain"
    assert "source_param_schema_version='unknown'" in diagnostic.observed_condition


@pytest.mark.parametrize("family,version", [(ResolvedGraph, "2"), (CompilationRecord, "3")])
def test_pre_canonical_evidence_requires_recompilation(family, version: str) -> None:
    compiled = compile_graph(_document(), ComponentRegistry(load_user_components=False))
    evidence = compiled.resolved if family is ResolvedGraph else compiled.record
    payload = evidence.model_dump(mode="json")
    payload["schema_version"] = version
    with pytest.raises(ValidationError, match="recompile the authored GraphDocument"):
        family.model_validate(payload)


def test_parameter_normalization_is_applied_once_before_runtime_construction() -> None:
    class GainParams(BaseModel):
        gain: float

        @field_validator("gain")
        @classmethod
        def convert_percent(cls, value: float) -> float:
            return value / 100

    registry = ComponentRegistry(load_user_components=False)
    registry.register_component_type(
        "PercentGain",
        builder=lambda params: Gain(gain=params["gain"]),
        param_model=GainParams,
        input_ports=["input"],
        output_ports=["output"],
    )
    graph = GraphSpec(
        nodes={
            "gain": ComponentSpec(
                type="PercentGain",
                params={"gain": 50.0},
                input_ports=["input"],
                output_ports=["output"],
            )
        },
        input_ports=["input"],
        input_bindings={"input": ("gain", "input")},
        output_ports=["output"],
        output_bindings={"output": ("gain", "output")},
    )
    compiled = compile_graph(GraphDocument(graph=graph), registry)
    assert compiled.resolved.graph.nodes["gain"].params["gain"] == 0.5
    outputs, _ = compiled.graph(
        {"input": jnp.asarray(8.0)},
        init_state_from_component(compiled.graph),
        key=jax.random.PRNGKey(0),
    )
    assert float(outputs["output"]) == 4.0
