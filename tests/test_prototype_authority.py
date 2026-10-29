from __future__ import annotations

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
