from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from feedbax.component_registry import ComponentRegistry
from feedbax.compiler import GraphDocument, compile_graph
from feedbax.contracts.graph import ComponentSpec, GraphSpec, build_default_studio_workspace
from feedbax.runtime.components import Constant
from feedbax.studio.schema import enumerate_studio_schema_registry


@pytest.mark.parametrize("declared_size", [None, 4])
def test_custom_nested_prototype_drives_compiler_and_studio_dimensions(declared_size) -> None:
    components = ComponentRegistry(load_user_components=False)
    components.register_component_type(
        "VectorSource",
        builder=lambda params: Constant(jnp.zeros(params["width"])),
        param_schema=[{"name": "width", "type": "int", "default": 3}],
        output_ports=["output"],
        output_prototype_fn=lambda params, inputs: {"output": jnp.zeros(params["width"])},
    )
    child = GraphSpec(
        nodes={"source": ComponentSpec(type="VectorSource", output_ports=["output"])},
        output_ports=["output"],
        output_bindings={"output": ("source", "output")},
    )
    params = {"output_size": 2}
    if declared_size is not None:
        params["input_size"] = declared_size
    graph = GraphSpec(
        nodes={
            "nested": ComponentSpec(type="Subgraph", output_ports=["output"]),
            "linear": ComponentSpec(
                type="Linear", params=params, input_ports=["input"], output_ports=["output"]
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
        ("DelayLine", False),
        ("Channel", False),
        ("FirstOrderFilter", False),
        ("FirstOrderFilter", True),
    ],
)
def test_stateful_dtype_survives_scan_and_serialization(component_type, nested) -> None:
    from feedbax.compiler.serialization import graph_to_spec
    from feedbax.runtime.graph import init_state_from_component

    components = ComponentRegistry(load_user_components=False)
    leaf = GraphSpec(
        nodes={
            "sum": ComponentSpec(type="Sum", input_ports=["a", "b"], output_ports=["output"]),
            "reshape": ComponentSpec(
                type="Reshape",
                params={"shape": [1, 3]},
                input_ports=["input"],
                output_ports=["output"],
            ),
            "stateful": ComponentSpec(
                type=component_type,
                params={"delay": 1, "add_noise": False}
                if component_type == "Channel"
                else {"delay": 1}
                if component_type == "DelayLine"
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
                    type="Subgraph", input_ports=["input"], output_ports=["output"]
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
        assert node.param_schema_version == "2"


def test_stateful_prototype_declarations_and_schema_migration() -> None:
    from feedbax.compiler.prototypes import infer_node_input_prototypes, validate_or_add_input_shape
    from feedbax.runtime.components import DelayLine

    components = ComponentRegistry(load_user_components=False)
    for kind in ("DelayLine", "Channel", "FirstOrderFilter"):
        old = {**components.get(kind).default_params, "input_shape": [3]}
        old.pop("input_dtype", None)
        migrated = components.resolve_component_spec(kind, old, param_schema_version="1")
        assert {name: migrated.params[name] for name in old} == old
        assert migrated.params.get("input_dtype") is None
        with pytest.raises(ValueError, match="version"):
            components.resolve_component_spec(kind, old, param_schema_version="unknown")
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
            with pytest.raises(ValueError, match="single-array"):
                validate_or_add_input_shape({}, tree, node_name="test", node_type=kind)

    spec = GraphSpec(
        nodes={
            "source": ComponentSpec(type="Sum", input_ports=["a"], output_ports=["output"]),
            "target": ComponentSpec(
                type="DelayLine", input_ports=["input"], output_ports=["output"]
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
    from feedbax.runtime.filters import FirstOrderFilter
    from feedbax.runtime.graph import init_state_from_component

    prototype = {
        "position": jnp.zeros(2, dtype=jnp.float16),
        "velocity": (jnp.zeros(1, dtype=jnp.float32),),
    }
    component = FirstOrderFilter(input_proto=prototype)
    state = init_state_from_component(component)
    values = jax.tree.map(lambda value: jnp.ones((3, *value.shape), dtype=value.dtype), prototype)

    def step(carry, value):
        outputs, carry = component({"input": value}, carry, key=jax.random.PRNGKey(0))
        return carry, outputs["output"]

    _, outputs = jax.lax.scan(step, state, values)
    assert jax.tree.structure(outputs) == jax.tree.structure(prototype)
    assert outputs["position"].dtype == jnp.float16
    assert outputs["velocity"][0].dtype == jnp.float32
    assert jnp.allclose(outputs["position"][:, 0], jnp.array([0.02, 0.0396, 0.058808]), atol=1e-4)
    assert jnp.allclose(outputs["velocity"][0][:, 0], jnp.array([0.02, 0.0396, 0.058808]))
