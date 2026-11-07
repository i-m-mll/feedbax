from __future__ import annotations

from feedbax.component_registry import ComponentPack

import jax
import jax.numpy as jnp
import pytest

from feedbax.component_registry import ComponentRegistry
from feedbax.compiler import GraphDocument, compile_graph
from feedbax.contracts.graph import ComponentSpec, GraphSpec, build_default_studio_workspace
from feedbax.runtime.graph import Component
from feedbax.contracts.portable import Ports, ComponentOptions
from feedbax.contracts.component import PortType
from feedbax.studio.schema import enumerate_studio_schema_registry


@pytest.mark.parametrize("declared_size", [None, 4])
def test_custom_nested_prototype_drives_compiler_and_studio_dimensions(declared_size) -> None:
    components = ComponentRegistry()

    class VectorSource(Component):
        ports = Ports({}, {"output": PortType(dtype="any")})
        component_options = ComponentOptions(
            output_prototype_fn=lambda params, inputs: {"output": jnp.zeros(params["width"])}
        )
        width: int

        def __init__(self, width: int = 3):
            self.width = width

        def __call__(self, inputs, state, *, key):
            return {"output": jnp.zeros(self.width)}, state

    components.ingest(ComponentPack(namespace="test", components=(VectorSource,)))
    child = GraphSpec(
        nodes={"source": ComponentSpec(type="test.VectorSource", output_ports=["output"])},
        output_ports=["output"],
        output_bindings={"output": ("source", "output")},
    )
    params = {"output_size": 2}
    if declared_size is not None:
        params["input_size"] = declared_size
    graph = GraphSpec(
        nodes={
            "nested": ComponentSpec(type="feedbax.Subgraph", output_ports=["output"]),
            "linear": ComponentSpec(
                type="feedbax.Linear", params=params, input_ports=["input"], output_ports=["output"]
            ),
        },
        subgraphs={"nested": child},
        wires=[
            {
                "source_node": "nested",
                "source_port": "output",
                "target_node": "linear",
                "target_port": "input",
            }
        ],
        output_ports=["output"],
        output_bindings={"output": ("linear", "output")},
        derived_dimensions=[{"node": "linear", "param": "input_size", "port": "input"}],
    )
    schema = enumerate_studio_schema_registry(
        build_default_studio_workspace(label="Custom prototype"),
        graph=graph,
        component_registry=components,
    )
    dimension_issues = [
        issue for issue in schema.issues if issue.type.startswith("derived_dimension")
    ]
    if declared_size is not None:
        assert [issue.type for issue in dimension_issues] == ["derived_dimension_conflict"]
        assert "derived 3" in dimension_issues[0].message
    else:
        assert not dimension_issues
        port = next(
            port
            for port in schema.ports
            if port.node_id == "linear" and port.port == "input" and port.direction == "input"
        )
        assert port.value_schema.shape == [3]
        result = compile_graph(GraphDocument(graph=graph), components)
        assert result.resolved.graph.nodes["linear"].params["input_size"] == 3


@pytest.mark.parametrize(
    ("component_type", "nested"),
    [
        ("feedbax.DelayLine", False),
        ("feedbax.Channel", False),
        ("feedbax.FirstOrderFilter", False),
        ("feedbax.FirstOrderFilter", True),
    ],
)
def test_stateful_dtype_survives_scan_and_serialization(component_type, nested) -> None:
    from feedbax.compiler.serialization import graph_to_spec
    from feedbax.runtime.graph import init_state_from_component

    components = ComponentRegistry()
    leaf = GraphSpec(
        nodes={
            "sum": ComponentSpec(
                type="feedbax.Sum", input_ports=["a", "b"], output_ports=["output"]
            ),
            "reshape": ComponentSpec(
                type="feedbax.Reshape",
                params={"shape": [1, 3]},
                input_ports=["input"],
                output_ports=["output"],
            ),
            "stateful": ComponentSpec(
                type=component_type,
                params={"delay": 1, "add_noise": False}
                if component_type == "feedbax.Channel"
                else {"delay": 1}
                if component_type == "feedbax.DelayLine"
                else {"tau_rise": 0.05, "tau_decay": 0.05, "dt": 0.001},
                input_ports=["input"],
                output_ports=["output"],
            ),
        },
        wires=[
            dict(
                source_node="sum", source_port="output", target_node="reshape", target_port="input"
            ),
            dict(
                source_node="reshape",
                source_port="output",
                target_node="stateful",
                target_port="input",
            ),
        ],
        input_ports=["input"],
        input_bindings={"input": ("sum", "a")},
        output_ports=["output"],
        output_bindings={"output": ("stateful", "output")},
    )
    spec = (
        GraphSpec(
            nodes={
                "nested": ComponentSpec(
                    type="feedbax.Subgraph", input_ports=["input"], output_ports=["output"]
                )
            },
            subgraphs={"nested": leaf},
            input_ports=["input"],
            input_bindings={"input": ("nested", "input")},
            output_ports=["output"],
            output_bindings={"output": ("nested", "output")},
        )
        if nested
        else leaf
    )
    prototypes = {("__graph__", "input"): jnp.zeros(3, dtype=jnp.float16)}
    for _ in range(2):
        result = compile_graph(GraphDocument(graph=spec), components, input_prototypes=prototypes)
        graph = result.graph
        runtime_leaf = graph.nodes["nested"] if nested else graph
        component = runtime_leaf.nodes["stateful"]
        assert component.input_proto.shape == (1, 3)
        assert component.input_proto.dtype == jnp.float16
        state = init_state_from_component(graph)
        stateful_state = state.get(component.state_index)
        assert all(
            x.dtype == jnp.float16 for x in jax.tree.leaves(stateful_state) if hasattr(x, "dtype")
        )

        def step(carry, value):
            outputs, carry, _ = graph.step({"input": value}, carry, key=jax.random.PRNGKey(0))
            return carry, outputs["output"]

        _, output = jax.lax.scan(step, state, jnp.ones((3, 3), dtype=jnp.float16))
        assert output.shape == (3, 1, 3)
        assert output.dtype == jnp.float16
        assert bool(jnp.all(jnp.isfinite(output)))
        spec = graph_to_spec(graph, components)
        serialized_leaf = spec.subgraphs["nested"] if nested else spec
        node = serialized_leaf.nodes["stateful"]
        assert node.params["input_dtype"] == "float16"
        assert node.revision == 2


def test_stateful_prototype_declarations_and_schema_migration() -> None:
    from feedbax.compiler.prototypes import infer_node_input_prototypes, validate_or_add_input_shape
    from feedbax.runtime.components import DelayLine

    components = ComponentRegistry()
    for kind in ("feedbax.DelayLine", "feedbax.Channel", "feedbax.FirstOrderFilter"):
        old = {**components.get(kind).default_params, "input_shape": [3]}
        old.pop("input_dtype", None)
        migrated = components.resolve_component_spec(
            kind.removeprefix("feedbax."), old, legacy_token="1"
        )
        assert {name: migrated.params[name] for name in old} == old
        assert migrated.params.get("input_dtype") is None
        with pytest.raises(ValueError, match="source_legacy_token"):
            components.resolve_component_spec(
                kind.removeprefix("feedbax."), old, legacy_token="unknown"
            )
        meta = components.get(kind)
        assert meta.params.build(old).input_proto.dtype == DelayLine().input_proto.dtype
        normalized = validate_or_add_input_shape(
            {"input_shape": None}, jnp.zeros(3, dtype=jnp.float16), node_name="test", node_type=kind
        )
        assert normalized == {"input_shape": [3], "input_dtype": "float16"}
        for params in ({"input_shape": [4]}, {"input_dtype": "float32"}):
            with pytest.raises(ValueError, match="does not match inferred"):
                validate_or_add_input_shape(
                    params, jnp.zeros(3, dtype=jnp.float16), node_name="test", node_type=kind
                )
        for tree in ([jnp.zeros(3)], {"x": jnp.zeros(3)}):
            if kind == "feedbax.Channel":
                assert validate_or_add_input_shape({}, tree, node_name="test", node_type=kind) == {}
                with pytest.raises(ValueError, match="single-array"):
                    validate_or_add_input_shape(
                        {"input_shape": [3]}, tree, node_name="test", node_type=kind
                    )
                continue
            with pytest.raises(ValueError, match="single-array"):
                validate_or_add_input_shape({}, tree, node_name="test", node_type=kind)

    spec = GraphSpec(
        nodes={
            "source": ComponentSpec(type="feedbax.Sum", input_ports=["a"], output_ports=["output"]),
            "target": ComponentSpec(
                type="feedbax.DelayLine",
                params={"delay": 1},
                input_ports=["input"],
                output_ports=["output"],
            ),
        },
        wires=[
            dict(
                source_node="source",
                source_port="output",
                target_node="target",
                target_port="input",
            )
        ],
    )
    with pytest.raises(ValueError, match="dtype mismatch"):
        infer_node_input_prototypes(
            spec,
            {("source", "a"): jnp.zeros(3, dtype=jnp.float16), ("target", "input"): jnp.zeros(3)},
            {},
            components,
        )


def test_direct_filter_preserves_structured_signal_dtypes() -> None:
    from feedbax.runtime.filters import LowPassFilter
    from feedbax.runtime.graph import init_state_from_component

    with jax.experimental.enable_x64():
        prototype = {
            "position": jnp.zeros(2, dtype=jnp.float16),
            "velocity": (jnp.zeros(1, dtype=jnp.float32),),
            "acceleration": jnp.zeros(1, dtype=jnp.float64),
        }
        component = LowPassFilter(input_proto=prototype)
        state = init_state_from_component(component)
        values = jax.tree.map(
            lambda value: jnp.ones((3, *value.shape), dtype=value.dtype), prototype
        )

        def step(carry, value):
            outputs, carry = component({"input": value}, carry, key=jax.random.PRNGKey(0))
            return carry, outputs["output"]

        _, outputs = jax.lax.scan(step, state, values)
        assert jax.tree.structure(outputs) == jax.tree.structure(prototype)
        assert outputs["position"].dtype == jnp.float16
        assert outputs["velocity"][0].dtype == jnp.float32
        assert jnp.allclose(
            outputs["position"][:, 0], jnp.array([0.02, 0.0396, 0.058808]), atol=1e-4
        )
        assert jnp.allclose(outputs["velocity"][0][:, 0], jnp.array([0.02, 0.0396, 0.058808]))
        assert outputs["acceleration"].dtype == jnp.float64
        assert jnp.allclose(outputs["acceleration"][:, 0], jnp.array([0.02, 0.0396, 0.058808]))
        assert all(bool(jnp.all(jnp.isfinite(value))) for value in jax.tree.leaves(outputs))


@pytest.mark.parametrize("dtype", [jnp.int32, jnp.bool_, jnp.complex64])
def test_filter_rejects_non_real_floating_state_leaves(dtype) -> None:
    from feedbax.runtime.filters import LowPassFilter

    registry = ComponentRegistry()
    invalid = jnp.zeros(2, dtype=dtype)
    for prototype in (invalid, {"valid": jnp.zeros(2), "invalid": (invalid,)}):
        with pytest.raises(
            ValueError, match="FirstOrderFilter requires real floating state leaves"
        ):
            LowPassFilter(input_proto=prototype)
        with pytest.raises(
            ValueError, match="FirstOrderFilter requires real floating state leaves"
        ):
            registry.get("feedbax.FirstOrderFilter").output_prototype_fn({}, {"input": prototype})


def test_integer_source_filter_is_rejected_during_compilation() -> None:
    from feedbax.compiler import GraphCompilationError

    registry = ComponentRegistry()
    spec = GraphSpec(
        nodes={
            "source": ComponentSpec(
                type="feedbax.Constant", params={"value": 1}, output_ports=["output"]
            ),
            "filter": ComponentSpec(
                type="feedbax.FirstOrderFilter",
                params=registry.get("feedbax.FirstOrderFilter").default_params,
                input_ports=["input"],
                output_ports=["output"],
            ),
        },
        wires=[
            dict(
                source_node="source",
                source_port="output",
                target_node="filter",
                target_port="input",
            )
        ],
        output_ports=["output"],
        output_bindings={"output": ("filter", "output")},
    )
    with pytest.raises(
        GraphCompilationError, match="FirstOrderFilter requires real floating state leaves"
    ) as caught:
        compile_graph(GraphDocument(graph=spec), registry)
    assert caught.value.record.diagnostics[0].source_anchor.authored_path == "/graph/nodes/filter"
    assert registry.get("feedbax.FirstOrderFilter").revision == 2


@pytest.mark.parametrize("nested", [False, True])
def test_final_parameter_meaning_drives_derived_state_and_studio(nested) -> None:
    from pydantic import BaseModel, field_validator
    from feedbax.runtime.graph import init_state_from_component

    calls = {"source": 0, "relay": 0}

    class SourceParams(BaseModel):
        width: int

        @field_validator("width")
        @classmethod
        def normalize_width(cls, value):
            calls["source"] += 1
            return value + 1

    class RelayParams(BaseModel):
        width: int

        @field_validator("width")
        @classmethod
        def record_completion(cls, value):
            calls["relay"] += 1
            return value

    registry = ComponentRegistry()

    class NormalizedSource(Component):
        config = SourceParams
        ports = Ports({}, {"output": PortType(dtype="any")})
        component_options = ComponentOptions(
            output_prototype_fn=lambda params, inputs: {"output": jnp.zeros(params["width"])}
        )
        width: int

        def __init__(self, width: int):
            self.width = width

        def __call__(self, inputs, state, *, key):
            return {"output": jnp.ones(self.width)}, state

    registry.ingest(ComponentPack(namespace="test", components=(NormalizedSource,)))

    class DerivedRelay(Component):
        config = RelayParams
        ports = Ports({"input": PortType(dtype="any")}, {"output": PortType(dtype="any")})
        component_options = ComponentOptions(
            output_prototype_fn=lambda params, inputs: {"output": jnp.zeros(params["width"])}
        )
        width: int

        def __init__(self, width: int):
            self.width = width

        def __call__(self, inputs, state, *, key):
            return {"output": inputs["input"]}, state

    registry.ingest(ComponentPack(namespace="test", components=(DerivedRelay,)))
    typed = SourceParams(width=2)
    direct = NormalizedSource.from_config(typed)
    assert typed.width == direct.width == direct.configuration().width == 3
    assert calls == {"source": 1, "relay": 0}
    leaf = GraphSpec(
        nodes={
            "source": ComponentSpec(
                type="test.NormalizedSource", params={"width": 2}, output_ports=["output"]
            ),
            "relay": ComponentSpec(
                type="test.DerivedRelay", input_ports=["input"], output_ports=["output"]
            ),
            "delay": ComponentSpec(
                type="feedbax.DelayLine",
                params={"delay": 1},
                input_ports=["input"],
                output_ports=["output"],
            ),
        },
        wires=[
            dict(
                source_node="source", source_port="output", target_node="relay", target_port="input"
            ),
            dict(
                source_node="relay", source_port="output", target_node="delay", target_port="input"
            ),
        ],
        derived_dimensions=[dict(node="relay", param="width", port="input")],
        output_ports=["output"],
        output_bindings={"output": ("delay", "output")},
    )
    spec = (
        GraphSpec(
            nodes={"nested": ComponentSpec(type="feedbax.Subgraph", output_ports=["output"])},
            subgraphs={"nested": leaf},
            output_ports=["output"],
            output_bindings={"output": ("nested", "output")},
        )
        if nested
        else leaf
    )
    calls.update(source=0, relay=0)
    compiled = compile_graph(GraphDocument(graph=spec), registry)
    assert calls == {"source": 1, "relay": 1}
    canonical = compiled.resolved.graph.subgraphs["nested"] if nested else compiled.resolved.graph
    assert (
        canonical.nodes["source"].params["width"] == canonical.nodes["relay"].params["width"] == 3
    )
    assert canonical.nodes["delay"].params["input_shape"] == [3]
    graph = compiled.graph

    def step(state, _):
        outputs, state, _ = graph.step({}, state, key=jax.random.PRNGKey(0))
        return state, outputs["output"]

    _, outputs = jax.lax.scan(step, init_state_from_component(graph), None, length=3)
    assert outputs.shape == (3, 3)
    assert jnp.array_equal(outputs, jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]))
    from feedbax.compiler.serialization import graph_to_spec

    saved = graph_to_spec(graph, registry)
    saved_leaf = saved.subgraphs["nested"] if nested else saved
    assert saved_leaf.nodes["source"].parameter_state == "effective"
    assert saved_leaf.nodes["source"].params["width"] == 3
    rebuilt = compile_graph(
        GraphDocument(graph=GraphSpec.model_validate_json(saved.model_dump_json())), registry
    )
    rebuilt_graph = rebuilt.graph.nodes["nested"].graph if nested else rebuilt.graph
    assert rebuilt_graph.nodes["source"].width == 3
    assert calls == {"source": 1, "relay": 1}
    calls.update(source=0, relay=0)
    studio = enumerate_studio_schema_registry(
        build_default_studio_workspace(label="Normalized parameters"),
        graph=spec,
        component_registry=registry,
    )
    assert calls == {"source": 1, "relay": 1}
    assert not [issue for issue in studio.issues if issue.type.startswith("derived_dimension")]
    output = next(port for port in studio.ports if port.port == "output" and port.node_id is None)
    assert output.value_schema.shape == [3]


def test_parameter_normalizer_cannot_contradict_derived_dimension() -> None:
    from pydantic import BaseModel, field_validator
    from feedbax.compiler import GraphCompilationError

    class ContradictoryParams(BaseModel):
        width: int

        @field_validator("width")
        @classmethod
        def normalize_width(cls, value):
            return value + 1

    registry = ComponentRegistry()

    class ContradictoryRelay(Component):
        config = ContradictoryParams
        ports = Ports({"input": PortType(dtype="any")}, {"output": PortType(dtype="any")})
        component_options = ComponentOptions(
            output_prototype_fn=lambda params, inputs: {"output": jnp.zeros(params["width"])}
        )
        width: int

        def __init__(self, width: int):
            self.width = width

        def __call__(self, inputs, state, *, key):
            return {"output": inputs["input"]}, state

    registry.ingest(ComponentPack(namespace="test", components=(ContradictoryRelay,)))
    spec = GraphSpec(
        nodes={
            "relay": ComponentSpec(
                type="test.ContradictoryRelay", input_ports=["input"], output_ports=["output"]
            )
        },
        input_ports=["input"],
        input_bindings={"input": ("relay", "input")},
        derived_dimensions=[dict(node="relay", param="width", port="input")],
    )
    with pytest.raises(GraphCompilationError) as caught:
        compile_graph(GraphDocument(graph=spec), registry, {("__graph__", "input"): jnp.zeros(3)})
    diagnostic = caught.value.record.diagnostics[0]
    assert diagnostic.source_anchor.authored_path == "/graph/nodes/relay"
    assert "finalized 4 but derived 3" in diagnostic.observed_condition


def test_recurrent_constant_initializer_seeds_final_state_dtype_and_shape() -> None:
    graph = GraphSpec(
        nodes={
            "delay": ComponentSpec(
                type="feedbax.DelayLine",
                params={"delay": 1},
                input_ports=["input"],
                output_ports=["output"],
            )
        },
        wires=[
            dict(
                source_node="delay",
                source_port="output",
                target_node="delay",
                target_port="input",
                temporality="recurrent",
                recurrent_initializer={
                    "kind": "constant",
                    "value": [1.0, 2.0, 3.0],
                    "dtype": "float16",
                },
            )
        ],
    )
    compiled = compile_graph(GraphDocument(graph=graph), ComponentRegistry())
    params = compiled.resolved.graph.nodes["delay"].params
    assert params["input_shape"] == [3]
    assert params["input_dtype"] == "float16"
    assert compiled.graph.nodes["delay"].input_proto.dtype == jnp.float16


def test_prototype_dependencies_defer_without_reinterpreting_parameters() -> None:
    from feedbax.contracts.array_values import (
        ARRAY_VALUE_SCHEMA_ID,
        ARRAY_VALUE_SCHEMA_VERSION,
        ConstantArrayValueSpec,
    )

    registry = ComponentRegistry()
    value = ConstantArrayValueSpec(
        schema_id=ARRAY_VALUE_SCHEMA_ID,
        schema_version=ARRAY_VALUE_SCHEMA_VERSION,
        encoding="constant",
        shape=[1],
        dtype="float32",
        nonfinite="forbid",
        value=2.0,
    ).model_dump(mode="json")
    graph = GraphSpec(
        nodes={
            "base": ComponentSpec(
                type="feedbax.Constant", params={"value": value}, output_ports=["output"]
            ),
            "state": ComponentSpec(
                type="feedbax.Constant", params={"value": value}, output_ports=["output"]
            ),
            "affine": ComponentSpec(
                type="feedbax.AffineValueComposer",
                params={
                    "output_block_size": 1,
                    "feature_rules": [{"kind": "identity", "state_slice": [0, 1]}],
                },
                input_ports=["base", "state"],
                output_ports=["value"],
            ),
            "delay": ComponentSpec(
                type="feedbax.DelayLine",
                params={"delay": 1},
                input_ports=["input"],
                output_ports=["output"],
            ),
        },
        wires=[
            dict(
                source_node="base", source_port="output", target_node="affine", target_port="base"
            ),
            dict(
                source_node="state", source_port="output", target_node="affine", target_port="state"
            ),
            dict(
                source_node="affine", source_port="value", target_node="delay", target_port="input"
            ),
        ],
    )
    compiled = compile_graph(GraphDocument(graph=graph), registry)
    assert compiled.resolved.graph.nodes["delay"].params["input_shape"] == [1]
    assert compiled.resolved.graph.nodes["base"].params["value"] == value


def test_parameter_determined_prototypes_are_equivalent_across_subgraph_boundaries() -> None:
    from feedbax.runtime.graph import init_state_from_component

    linear = ComponentSpec(
        type="feedbax.Linear",
        params={"input_size": 2, "output_size": 3, "initialization": "zeros", "bias_init": 1.25},
        input_ports=["input"],
        output_ports=["output"],
    )
    body = GraphSpec(
        nodes={"linear": linear},
        input_ports=["input"],
        input_bindings={"input": ("linear", "input")},
        output_ports=["output"],
        output_bindings={"output": ("linear", "output")},
    )
    for nested in (False, True):
        source = "body" if nested else "linear"
        spec = GraphSpec(
            nodes={
                source: ComponentSpec(
                    type="feedbax.Subgraph", input_ports=["input"], output_ports=["output"]
                )
                if nested
                else linear,
                "delay": ComponentSpec(
                    type="feedbax.DelayLine",
                    params={"delay": 1},
                    input_ports=["input"],
                    output_ports=["output"],
                ),
            },
            subgraphs={"body": body} if nested else None,
            wires=[
                dict(
                    source_node=source,
                    source_port="output",
                    target_node="delay",
                    target_port="input",
                )
            ],
            input_ports=["input"],
            input_bindings={"input": (source, "input")},
            output_ports=["output"],
            output_bindings={"output": ("delay", "output")},
        )
        compiled = compile_graph(GraphDocument(graph=spec), ComponentRegistry())
        assert compiled.resolved.graph.nodes["delay"].params["input_shape"] == [3]
        graph = compiled.graph

        def step(state, value):
            outputs, state, _ = graph.step({"input": value}, state, key=jax.random.PRNGKey(0))
            return state, outputs["output"]

        _, output = jax.lax.scan(step, init_state_from_component(graph), jnp.ones((2, 2)))
        assert jnp.array_equal(output, jnp.asarray([[0.0] * 3, [1.25] * 3]))


@pytest.mark.parametrize(
    ("rule_patch", "shape", "reason"),
    [
        ({"node": "ghost"}, (2,), "targets missing node 'ghost'"),
        ({"port": "ghost"}, None, "targets missing input port linear.ghost"),
        ({"axis": 2}, (2,), "cannot select a positive dimension"),
        ({}, (0,), "cannot select a positive dimension"),
        ({}, None, "could not resolve a prototype"),
    ],
)
def test_invalid_derived_rules_reject_in_compilation_and_studio(rule_patch, shape, reason) -> None:
    from feedbax.compiler import GraphCompilationError
    from feedbax.studio.schema import validate_graph_connection_schema
    from feedbax.contracts.graph import StudioTaskBindingSpec

    rule = {"node": "linear", "param": "input_size", "port": "input", **rule_patch}
    rules = [rule]
    if "port" in rule_patch:
        rules.insert(0, {"node": "linear", "param": "input_size", "port": "input"})
    graph = GraphSpec(
        nodes={
            "linear": ComponentSpec(
                type="feedbax.Linear",
                params={"input_size": 2, "output_size": 3},
                input_ports=["input"],
                output_ports=["output"],
            )
        },
        derived_dimensions=rules,
        input_ports=["input"],
        input_bindings={"input": ("linear", "input")},
    )
    prototypes = {} if shape is None else {("__graph__", "input"): jnp.zeros(shape)}
    registry = ComponentRegistry()
    with pytest.raises(GraphCompilationError) as caught:
        compile_graph(GraphDocument(graph=graph), registry, prototypes)
    assert reason in caught.value.record.diagnostics[0].observed_condition
    if rule_patch.get("node") == "ghost":
        assert (
            caught.value.record.diagnostics[0].source_anchor.authored_path
            == "/graph/derived_dimensions/0"
        )
    bindings = (
        StudioTaskBindingSpec()
        if shape is None
        else StudioTaskBindingSpec.model_validate(
            {
                "exposed_data": [
                    {
                        "id": "signal",
                        "label": "Signal",
                        "kind": "signal",
                        "role": "model_input",
                        "path": "inputs.signal",
                        "bindable": True,
                        "expected_shape": list(shape),
                    }
                ],
                "bindings": [
                    {
                        "id": "signal",
                        "source_data_id": "signal",
                        "target_node_id": "linear",
                        "target_port": "input",
                        "role": "model_input",
                    }
                ],
            }
        )
    )
    issues = validate_graph_connection_schema(
        graph, "/graph", bindings, component_registry=registry
    )
    issue = next(issue for issue in issues if issue.type == "derived_dimension_unresolved")
    assert reason in issue.message
    assert issue.location["path"] == f"/graph/derived_dimensions/{len(rules) - 1}"
