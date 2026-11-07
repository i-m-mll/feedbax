from __future__ import annotations

from feedbax.component_registry import ComponentPack

import pytest
import jax
import jax.numpy as jnp
from pydantic import BaseModel, ValidationError, field_validator

from feedbax.compiler import (
    CompilationRecord,
    CompilerPhase,
    GraphCompilationError,
    GraphDocument,
    ResolvedGraph,
    compile_graph,
)
from feedbax.compiler.serialization import graph_to_spec
from feedbax.component_registry import ComponentRegistry
from feedbax.component_registry.history import Change, History, LegacyEndpoint
from feedbax.contracts.component import PortType
from feedbax.contracts.portable import ComponentOptions, Ports
from feedbax.runtime.graph import Component
from types import ModuleType
from typing import ClassVar
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


def _owner_registry(owner, history=None):
    module = ModuleType("test_compiler_owners")
    owner.__module__ = module.__name__
    setattr(module, owner.__name__, owner)
    registry = ComponentRegistry()
    registry.ingest(ComponentPack(
        namespace="example", modules=(module,),
        histories={owner: history} if history is not None else {},
    ))
    return registry


def test_replacing_a_removed_node_does_not_reuse_its_trainability() -> None:
    registry = ComponentRegistry()

    def compile_linear(**params):
        spec = GraphSpec(
            nodes={
                "linear": ComponentSpec(
                    type="feedbax.Linear",
                    params={"input_size": 1, "output_size": 1, **params},
                    input_ports=["input"],
                    output_ports=["output"],
                )
            }
        )
        return compile_graph(GraphDocument(graph=spec), registry).graph

    frozen = compile_linear(trainable=False)
    fresh = compile_linear().nodes["linear"]
    replacement = frozen.remove_node("linear").add_node("linear", fresh)

    assert graph_to_spec(frozen, registry).nodes["linear"].params["trainable"] is False
    assert "trainable" not in graph_to_spec(replacement, registry).nodes["linear"].params


def _document() -> GraphDocument:
    return GraphDocument(
        graph=GraphSpec(
            nodes={
                "source": ComponentSpec(
                    type="feedbax.Constant",
                    params={"value": 2.0},
                    output_ports=["output"],
                ),
                "gain": ComponentSpec(
                    type="feedbax.Gain",
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
    registry = ComponentRegistry()

    first = compile_graph(_document(), registry)
    second = compile_graph(_document(), registry)

    assert first.resolved == second.resolved
    assert first.record == second.record
    assert first.resolved.key_schedule.node_order == first.graph._execution_order
    assert first.resolved.source_map.entries[1].authored_path == "/graph/nodes/gain"
    assert ResolvedGraph.model_validate_json(first.resolved.model_dump_json()) == first.resolved
    assert CompilationRecord.model_validate_json(first.record.model_dump_json()) == first.record


@pytest.mark.parametrize(
    ("driver", "competing_owners", "path", "occupied_by"),
    [
        ("wire", ("source.output", "second.output"), "/graph/wires/1", "/graph/wires/0"),
        ("repeated_wire", ("source.output",), "/graph/wires/1", "/graph/wires/0"),
        (
            "binding",
            ("external input 'command'", "source.output"),
            "/graph/wires/0",
            "/graph/input_bindings/command",
        ),
        (
            "two_bindings",
            ("external input 'command'", "external input 'override'"),
            "/graph/input_bindings/override",
            "/graph/input_bindings/command",
        ),
    ],
)
def test_compile_rejects_competing_drivers_before_realization(
    driver, competing_owners, path, occupied_by
):
    graph = _document().graph
    if driver == "wire":
        graph.nodes["second"] = graph.nodes["source"].model_copy(update={"params": {"value": 5.0}})
        graph.wires.append(graph.wires[0].model_copy(update={"source_node": "second"}))
    elif driver == "repeated_wire":
        graph.wires.append(graph.wires[0])
    else:
        graph.input_ports.append("command")
        graph.input_bindings["command"] = ("gain", "input")
        if driver == "two_bindings":
            graph.wires.clear()
            graph.input_ports.append("override")
            graph.input_bindings["override"] = ("gain", "input")

    with pytest.raises(GraphCompilationError) as caught:
        compile_graph(GraphDocument(graph=graph), ComponentRegistry())

    (diagnostic,) = caught.value.record.diagnostics
    assert diagnostic.phase is CompilerPhase.STRUCTURAL_PARSING
    assert diagnostic.source_anchor.authored_path == path
    assert diagnostic.actionable_context["occupied_by"] == occupied_by
    assert "gain.input" in diagnostic.observed_condition
    assert all(owner in diagnostic.observed_condition for owner in competing_owners)


def test_graph_document_rejects_unknown_version_without_restamping() -> None:
    payload = _document().model_dump(mode="json")
    payload["schema_version"] = "0"

    with pytest.raises(ValidationError, match="migration_intentionally_absent=yes"):
        GraphDocument.model_validate(payload)


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
            ComponentRegistry(),
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


def test_compile_graph_migrates_an_explicit_older_graph_schema() -> None:
    payload = _document().graph.model_dump(mode="json", exclude_none=True)
    payload["schema_version"] = "feedbax.spec.graph.v4"

    compiled = compile_graph(
        GraphDocument(graph=payload),
        ComponentRegistry(),
    )

    assert compiled.record.graph_schema_source_version == "feedbax.spec.graph.v4"
    assert compiled.record.graph_schema_target_version == "feedbax.spec.graph.v6"


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
        ComponentRegistry(),
    )

    adapter_id, _ = compiled.resolved.graph.input_bindings["disturbance"]
    entry = next(
        item
        for item in compiled.record.source_map.entries
        if item.resolved_path == f"/graph/nodes/{adapter_id}"
    )
    assert entry.origin == "compiler-generated"
    assert entry.authored_path == "/graph"
    adapter = compiled.resolved.graph.nodes[adapter_id]
    binding = compiled.graph.component_bindings[adapter_id]
    assert adapter.revision == binding.revision == 1


def test_workspace_view_edits_cannot_change_semantic_or_runtime_identity() -> None:
    registry = ComponentRegistry()
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
    class ScaledGain(Component):
        gain: float
        ports: ClassVar = Ports(
            {"input": PortType(dtype="any")}, {"output": PortType(dtype="any")}
        )
        component_options: ClassVar = ComponentOptions(
            output_prototype_fn=lambda params, inputs: {"output": inputs.get("input", jnp.zeros(()))}
        )

        def __call__(self, inputs, state, *, key):
            return {"output": self.gain * inputs["input"]}, state

    registry = _owner_registry(ScaledGain, History(
        Change(2, renamed_from="LegacyGain", migrate=lambda params: {"gain": params["scale"]}),
        legacy=(LegacyEndpoint("LegacyGain", "legacy-scale", 1),),
    ))
    child = GraphSpec(
        nodes={
            "gain": ComponentSpec(
                type="example.LegacyGain",
                params={"scale": 8.0},
                revision=1,
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
                    type="feedbax.Subgraph",
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
    payload = spec.model_dump(mode="json")
    payload["schema_version"] = "feedbax.spec.graph.v5"
    persisted_child = payload["subgraphs"]["nested"] if nested else payload
    persisted_child["schema_version"] = "feedbax.spec.graph.v5"
    persisted_gain = persisted_child["nodes"]["gain"]
    persisted_gain.pop("revision")
    persisted_gain.update(type="LegacyGain", param_schema_version="legacy-scale")
    compiled = compile_graph(GraphDocument(graph=payload), registry)
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
    assert (node.type, node.revision, node.params) == ("example.ScaledGain", 2, {"gain": 8.0})
    assert (binding.type_id, binding.revision) == (node.type, node.revision)
    outputs, _ = compiled.graph(
        {"input": jnp.asarray(3.0)},
        init_state_from_component(compiled.graph),
        key=jax.random.PRNGKey(0),
    )
    assert float(outputs["output"]) == 24.0
    assert child.nodes["gain"].type == "example.LegacyGain"
    exported = graph_to_spec(compiled.graph, registry)
    exported_child = exported.subgraphs["nested"] if nested else exported
    saved = exported_child.nodes["gain"]
    assert (saved.type, saved.revision, saved.params) == ("example.ScaledGain", 2, {"gain": 8.0})


def test_migrations_precede_inference_and_defaults_follow_derived_dimensions() -> None:
    class VectorSource(Component):
        width: int = 3
        ports: ClassVar = Ports({}, {"output": PortType(dtype="vector")})
        component_options: ClassVar = ComponentOptions(
            output_prototype_fn=lambda params, inputs: {"output": jnp.ones(params["width"])}
        )

        def __call__(self, inputs, state, *, key):
            return {"output": jnp.ones(self.width)}, state

    registry = _owner_registry(VectorSource, History(Change(
        2, renamed_from="LegacyVectorSource", migrate=lambda params: {"width": params["n"]}
    )))
    graph = GraphSpec(
        nodes={
            "source": ComponentSpec(
                type="example.LegacyVectorSource", revision=1, params={"n": 5}, output_ports=["output"]
            ),
            "linear": ComponentSpec(
                type="feedbax.Linear",
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
                "gain": child.nodes["gain"].model_copy(update={"revision": 999}),
            }
        }
    )
    graph = GraphSpec(
        nodes={"nested": ComponentSpec(type="feedbax.Subgraph", output_ports=["output"])},
        subgraphs={"nested": child},
        output_ports=["output"],
        output_bindings={"output": ("nested", "output")},
    )
    with pytest.raises(GraphCompilationError) as caught:
        compile_graph(GraphDocument(graph=graph), ComponentRegistry())
    diagnostic = caught.value.record.diagnostics[0]
    assert diagnostic.source_anchor.authored_path == "/graph/subgraphs/nested/nodes/gain"
    assert "999" in diagnostic.observed_condition


def test_parameter_normalization_is_applied_once_before_runtime_construction() -> None:
    class GainParams(BaseModel):
        gain: float

        @field_validator("gain")
        @classmethod
        def convert_percent(cls, value: float) -> float:
            return value / 100

    class PercentGain(Component):
        gain: float
        config: ClassVar = GainParams
        ports: ClassVar = Ports(
            {"input": PortType(dtype="any")}, {"output": PortType(dtype="any")}
        )
        component_options: ClassVar = ComponentOptions(
            output_prototype_fn=lambda params, inputs: {"output": inputs.get("input", jnp.zeros(()))}
        )

        def __call__(self, inputs, state, *, key):
            return {"output": self.gain * inputs["input"]}, state

    registry = _owner_registry(PercentGain)
    graph = GraphSpec(
        nodes={
            "gain": ComponentSpec(
                type="example.PercentGain",
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
