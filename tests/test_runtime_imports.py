import importlib

import jax
import jax.numpy as jnp


def test_runtime_namespace_exposes_graph_execution_entry_points() -> None:
    runtime = importlib.import_module("feedbax.runtime")
    modules = {
        name: importlib.import_module(f"feedbax.runtime.{name}")
        for name in (
            "graph",
            "channel",
            "state",
            "components",
            "selectors",
            "retained_observables",
            "state_feedback",
            "graph_channel_adapters",
            "parameter_constraints",
        )
    }

    assert modules["graph"].Graph is runtime.Graph
    assert modules["graph"].Component is runtime.Component
    assert modules["graph"].Wire is runtime.Wire
    assert modules["graph"].init_state_from_component is runtime.init_state_from_component
    assert hasattr(modules["channel"], "Channel")
    assert hasattr(modules["components"], "Gain")


def test_runtime_graph_primitives_build_and_execute_minimal_graph() -> None:
    graph_module = importlib.import_module("feedbax.runtime.graph")

    class Increment(graph_module.Component):
        input_ports = ("x",)
        output_ports = ("y",)

        def __call__(self, inputs, state, *, key):
            return {"y": inputs["x"] + 1}, state

    graph = graph_module.Graph(
        nodes={"inc": Increment()},
        wires=(),
        input_ports=("input",),
        output_ports=("output",),
        input_bindings={"input": ("inc", "x")},
        output_bindings={"output": ("inc", "y")},
    )

    outputs, _ = graph(
        {"input": jnp.array([1.0, 2.0])},
        graph_module.init_state_from_component(graph),
        key=jax.random.PRNGKey(0),
    )

    assert jnp.array_equal(outputs["output"], jnp.array([2.0, 3.0]))
