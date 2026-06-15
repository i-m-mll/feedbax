import importlib
import importlib.util

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


def test_domain_modules_use_canonical_package_homes() -> None:
    canonical_modules = [
        "feedbax.components.equinox",
        "feedbax.components.penzai",
        "feedbax.contracts.graphs.builders",
        "feedbax.contracts.graphs.materialization",
        "feedbax.contracts.graphs.normalization",
        "feedbax.contracts.graphs.prototypes",
        "feedbax.contracts.graphs.serialization",
        "feedbax.contracts.graphs.templates",
        "feedbax.contracts.schema_namespace",
        "feedbax.mechanics.dynamics",
        "feedbax.models.cde",
        "feedbax.models.feedback",
        "feedbax.models.networks",
        "feedbax.runtime.filters",
        "feedbax.runtime.noise",
    ]
    old_root_modules = [
        "feedbax.artifact_materialize",
        "feedbax.bodies",
        "feedbax.dynamics",
        "feedbax.eqx_components",
        "feedbax.filters",
        "feedbax.graph_normalization",
        "feedbax.graph_templates",
        "feedbax.nn",
        "feedbax.nn_cde",
        "feedbax.noise",
        "feedbax.penzai_component",
        "feedbax.schema_namespace",
        "feedbax.serialization",
        "feedbax.serialization_builders",
        "feedbax.serialization_prototypes",
    ]

    loaded = [importlib.import_module(module_name).__name__ for module_name in canonical_modules]
    rejected = [
        module_name
        for module_name in old_root_modules
        if importlib.util.find_spec(module_name) is None
    ]

    assert loaded == canonical_modules
    assert rejected == old_root_modules


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
