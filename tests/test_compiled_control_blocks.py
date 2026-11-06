"""Behavior of scalar-parameter control blocks compiled from graph documents."""

import jax
import jax.numpy as jnp
import pytest

from feedbax.compiler import CompilerPhase, GraphCompilationError, GraphDocument, compile_graph
from feedbax.component_registry import ComponentRegistry
from feedbax.compiler.serialization import graph_to_spec
from feedbax.contracts.graph import ComponentSpec, GraphSpec, WireSpec
from feedbax.control import PID, PIDDiscrete
from feedbax.runtime.graph import Graph, init_state_from_component


def test_compiled_control_blocks_run_on_float16_vectors() -> None:
    blocks = {
        "Integrator": ("input", {"dt": 0.25}),
        "Derivative": ("input", {"dt": 0.25}),
        "IntegratorDiscrete": ("input", {"dt": 0.5}),
        "UnitDelay": ("input", {"initial_value": -1.5}),
        "ZeroOrderHold": ("input", {"hold_steps": 2}),
        "PID": ("error", {"Kp": 2.5, "dt": 0.25}),
        "PIDDiscrete": ("error", {"Kp": 2.0, "dt": 0.25}),
    }
    controls = GraphSpec(
        nodes={
            name: ComponentSpec(
                type=name,
                params={"n_dims": 2, **params},
                input_ports=[port],
                output_ports=["output"],
            )
            for name, (port, params) in blocks.items()
        },
        input_ports=list(blocks),
        input_bindings={name: (name, port) for name, (port, _) in blocks.items()},
        output_ports=list(blocks),
        output_bindings={name: (name, "output") for name in blocks},
    )
    spec = GraphSpec(
        nodes={
            "a_controls": ComponentSpec(
                type="Subgraph", input_ports=list(blocks), output_ports=list(blocks)
            ),
            "z_source": ComponentSpec(
                type="Gain", params={"gain": 1.0}, input_ports=["input"], output_ports=["output"]
            ),
        },
        subgraphs={"a_controls": controls},
        wires=[
            WireSpec(
                source_node="z_source",
                source_port="output",
                target_node="a_controls",
                target_port=name,
            )
            for name in blocks
        ],
        input_ports=["command"],
        input_bindings={"command": ("z_source", "input")},
        output_ports=list(blocks),
        output_bindings={name: ("a_controls", name) for name in blocks},
    )
    samples = jnp.array([[1.0, -2.0], [3.0, 0.5], [-0.5, 4.0]], dtype=jnp.float16)
    compiled = compile_graph(
        GraphDocument(graph=spec),
        ComponentRegistry(load_user_components=False),
        input_prototypes={("__graph__", "command"): samples[0]},
    )
    graph = compiled.graph
    key = jax.random.key(0)

    def step(state, sample):
        outputs, state, _ = graph.step({"command": sample}, state, key=key)
        return state, outputs

    def rollout(state, sequence):
        return jax.lax.scan(step, state, sequence)

    run = jax.jit(rollout, donate_argnums=0)
    _, outputs = run(init_state_from_component(graph), samples)

    for name in blocks:
        assert outputs[name].shape == samples.shape
        assert outputs[name].dtype == jnp.float16
    assert jnp.array_equal(outputs["UnitDelay"], jnp.array([[-1.5, -1.5], [1, -2], [3, 0.5]]))
    assert jnp.array_equal(outputs["PID"], jnp.array([[2.5, -5], [7.5, 1.25], [-1.25, 10]]))


@pytest.mark.parametrize(
    ("shape", "dtype", "observed"),
    [
        ((3,), jnp.float16, "got shape (3,) and dtype float16"),
        ((2,), jnp.float32, "got shape (2,) and dtype float32"),
    ],
)
def test_compiled_control_rejects_late_incompatible_prototype(shape, dtype, observed) -> None:
    spec = GraphSpec(
        nodes={
            "a_delay": ComponentSpec(
                type="UnitDelay",
                params={"n_dims": 2, "input_dtype": "float16"},
                input_ports=["input"],
                output_ports=["output"],
            ),
            "z_source": ComponentSpec(
                type="Gain",
                params={"gain": 2.0},
                input_ports=["input"],
                output_ports=["output"],
            ),
        },
        wires=[
            WireSpec(
                source_node="z_source",
                source_port="output",
                target_node="a_delay",
                target_port="input",
            )
        ],
        input_ports=["command"],
        input_bindings={"command": ("z_source", "input")},
        output_ports=["output"],
        output_bindings={"output": ("a_delay", "output")},
    )

    with pytest.raises(GraphCompilationError) as caught:
        compile_graph(
            GraphDocument(graph=spec),
            ComponentRegistry(load_user_components=False),
            input_prototypes={("__graph__", "command"): jnp.zeros(shape, dtype=dtype)},
        )

    (diagnostic,) = caught.value.record.diagnostics
    assert diagnostic.phase is CompilerPhase.CONSTRAINT_SOLVING
    assert diagnostic.source_anchor.authored_path == "/graph/nodes/a_delay"
    assert "input requires shape (2,) and dtype float16" in diagnostic.observed_condition
    assert observed in diagnostic.observed_condition


def test_programmatic_pid_unbounded_limits_survive_serialization() -> None:
    registry = ComponentRegistry(load_user_components=False)
    graph = Graph(nodes={"continuous": PID(), "discrete": PIDDiscrete()})
    document = GraphDocument(graph=graph_to_spec(graph, registry))

    restored = compile_graph(
        GraphDocument.model_validate_json(document.model_dump_json()), registry
    ).graph
    assert restored.nodes["continuous"].integral_limit == float("inf")
    assert restored.nodes["discrete"].output_limit == float("inf")
