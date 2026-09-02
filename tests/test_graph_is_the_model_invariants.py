"""Component-level coverage for the "graph is the model" core principle.

``AGENTS.md``/``CLAUDE.md`` state the principle and three corollaries:

1. The graph is the model — the built artifact contains exactly the node types,
   params, and topology the spec describes; nothing is added, dropped, or
   defaulted silently.
2. No background construction — when a composite node has a subgraph, that
   subgraph is the source of truth and the outer/stale params on the node are
   never used to construct anything.
3. Absence of a subgraph is an error — never a fallback to outer params and
   never a synthesised default subgraph.

Assertions are made on the *built artifact* (the constructed ``Graph`` and its
component instances) rather than on intermediate plumbing, so they keep their
meaning as the build internals change.

Tests marked ``xfail(strict=True)`` pin invariants the current code violates.
They are real tests: they run, and they fail loudly (XPASS) the moment the
underlying bug is fixed. See Mandible issue ``8378254`` for the known silent
default substitutions.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import jax
import pytest

from feedbax.component_registry import ComponentRegistry, required_interior_domain
from feedbax.contracts.graph import ComponentSpec, GraphSpec, WireSpec
from feedbax.compiler.serialization import graph_to_spec
from tests.graph_compiler_test_support import spec_to_graph
from feedbax.compiler.templates import network_template_graph
from feedbax.runtime.graph import Graph
from feedbax.web.worker.diagnostics import GraphCompilationError
from feedbax.web.worker.execution import compile_training_run

# Outcomes that count as "the build refused to guess". A fix for any of the
# xfailed invariants may legitimately take the form of a loud rejection, so the
# xfail tests treat these as conformant.
REFUSALS = (ValueError, TypeError, KeyError, GraphCompilationError)


def _registry() -> ComponentRegistry:
    """A fresh registry per test; nothing process-global is mutated."""
    return ComponentRegistry(load_user_components=False)


def _gain_interior(gain: float, *, in_port: str, out_port: str) -> GraphSpec:
    """A minimal causal interior whose single observable parameter is ``gain``."""
    return GraphSpec(
        nodes={
            "scale": ComponentSpec(
                type="Gain",
                params={"gain": gain},
                input_ports=["input"],
                output_ports=["output"],
            )
        },
        input_ports=[in_port],
        output_ports=[out_port],
        input_bindings={in_port: ("scale", "input")},
        output_bindings={out_port: ("scale", "output")},
    )


def _composite_spec(
    node_type: str,
    outer_params: dict[str, Any],
    interior: GraphSpec | None,
    *,
    in_port: str,
    out_port: str,
) -> GraphSpec:
    return GraphSpec(
        nodes={
            "plant": ComponentSpec(
                type=node_type,
                params=dict(outer_params),
                input_ports=[in_port],
                output_ports=[out_port],
            )
        },
        input_ports=[in_port],
        output_ports=[out_port],
        input_bindings={in_port: ("plant", in_port)},
        output_bindings={out_port: ("plant", out_port)},
        subgraphs=None if interior is None else {"plant": interior},
    )


# ---------------------------------------------------------------------------
# Corollary 2: a composite node's subgraph is authoritative
# ---------------------------------------------------------------------------


def test_causal_composite_builds_from_subgraph_not_outer_params() -> None:
    """Outer params on a composite must not reach the built artifact."""
    interior = _gain_interior(7.0, in_port="excitation", out_port="force_2d")
    spec = _composite_spec(
        "PointMass8MuscleRelu",
        # Stale outer params describing a completely different plant.
        {"n_pairs": 4, "max_isometric_force": 500.0, "dt": 0.01},
        interior,
        in_port="excitation",
        out_port="force_2d",
    )

    registry = _registry()
    graph = spec_to_graph(spec, registry)
    built = graph.nodes["plant"]

    assert isinstance(built, Graph)
    assert list(built.nodes) == ["scale"]
    assert built.nodes["scale"].gain == 7.0
    assert built.input_ports == ("excitation",)
    assert built.output_ports == ("force_2d",)
    # No muscle machinery was constructed from the outer params.
    assert not any("Muscle" in type(node).__name__ for node in built.nodes.values())

    serialized = graph_to_spec(graph, registry)
    assert serialized.nodes["plant"].params == {}
    assert serialized.subgraphs is not None
    assert serialized.subgraphs["plant"].nodes["scale"].params == {"gain": 7.0}


def test_editing_the_subgraph_changes_the_built_artifact() -> None:
    """With outer params held fixed, the built model tracks the subgraph."""
    outer_params = {"n_pairs": 4, "max_isometric_force": 500.0, "dt": 0.01}
    registry = _registry()

    built_gains = []
    for gain in (2.0, 11.0):
        spec = _composite_spec(
            "PointMass8MuscleRelu",
            outer_params,
            _gain_interior(gain, in_port="excitation", out_port="force_2d"),
            in_port="excitation",
            out_port="force_2d",
        )
        built_gains.append(spec_to_graph(spec, registry).nodes["plant"].nodes["scale"].gain)

    assert built_gains == [2.0, 11.0]


def test_legacy_network_node_builds_from_subgraph_not_outer_params() -> None:
    """The legacy ``Network`` wrapper's own params describe nothing that is built."""
    interior = network_template_graph(
        {"input_size": 2, "hidden_size": 4, "out_size": 1, "hidden_type": "GRUCell"}
    )
    spec = GraphSpec(
        nodes={
            "net": ComponentSpec(
                type="Network",
                # Every value here contradicts the interior.
                params={
                    "input_size": 999,
                    "hidden_size": 100,
                    "out_size": 77,
                    "hidden_type": "LSTMCell",
                },
                input_ports=list(interior.input_ports),
                output_ports=list(interior.output_ports),
            )
        },
        input_ports=list(interior.input_ports),
        output_ports=list(interior.output_ports),
        input_bindings={port: ("net", port) for port in interior.input_ports},
        output_bindings={port: ("net", port) for port in interior.output_ports},
        subgraphs={"net": interior},
    )

    built = spec_to_graph(spec, _registry()).nodes["net"]

    assert isinstance(built, Graph)
    assert sorted(built.nodes) == ["cell", "input_mux", "readout"]
    cell = built.nodes["cell"]
    assert type(cell).__name__ == "GRU"  # not the LSTM the outer params name
    assert cell.hidden_size == 4  # not 100
    assert cell.input_size == 2  # not 999
    assert built.nodes["readout"].output_size == 1  # not 77


def test_acausal_composite_uses_interior_solver_not_outer_dt() -> None:
    """An acausal composite's timestep comes from its interior, not the node."""
    from feedbax.contracts.acausal import AcausalGraphSpec

    interior = AcausalGraphSpec(
        physical_domain="translational",
        nodes={
            "wall": ComponentSpec(type="Ground"),
            "mass": ComponentSpec(type="Mass", params={"mass": 1.0}),
            "spring": ComponentSpec(type="LinearSpring", params={"stiffness": 10.0}),
            "act": ComponentSpec(
                type="ActuationInput",
                params={"port_name": "u", "source_kind": "force"},
            ),
            "sense": ComponentSpec(
                type="SensorOutput",
                params={"port_name": "output", "quantity": "position"},
            ),
        },
        connections=[
            {"a": ("wall", "flange"), "b": ("spring", "flange_a")},
            {"a": ("spring", "flange_b"), "b": ("mass", "flange")},
            {"a": ("act", "flange"), "b": ("mass", "flange")},
            {"a": ("sense", "flange"), "b": ("mass", "flange")},
        ],
        solver={"solver_type": "euler", "dt": 0.001},
    )
    spec = GraphSpec(
        nodes={
            "plant": ComponentSpec(
                type="AcausalSystem",
                params={"dt": 0.5},  # stale outer value
                input_ports=["u"],
                output_ports=["output"],
            )
        },
        output_ports=["output"],
        output_bindings={"output": ("plant", "output")},
        subgraphs={"plant": interior},
    )

    built = spec_to_graph(spec, _registry()).nodes["plant"]

    assert float(built.dt) == pytest.approx(0.001)


# ---------------------------------------------------------------------------
# Corollary 3: absence of a subgraph is an error
# ---------------------------------------------------------------------------


def _composite_type_names() -> list[str]:
    registry = ComponentRegistry(load_user_components=False)
    return sorted(
        definition.name
        for definition in registry.list_all()
        if required_interior_domain(definition.name, registry) is not None
    )


@pytest.mark.parametrize("node_type", _composite_type_names())
def test_missing_subgraph_raises_for_every_composite_type(node_type: str) -> None:
    """No composite type may build, default, or synthesise an absent interior."""
    registry = _registry()
    meta = registry.get(node_type)
    assert meta is not None
    # Supply declared-required outer params so the failure is unambiguously
    # about the missing interior rather than about parameter validation.
    params = {schema.name: schema.default for schema in meta.param_schema if schema.required}
    spec = GraphSpec(
        nodes={
            "plant": ComponentSpec(
                type=node_type,
                params=params,
                input_ports=list(meta.input_ports),
                output_ports=list(meta.output_ports),
            )
        }
    )

    with pytest.raises(ValueError) as exc_info:
        spec_to_graph(spec, registry)

    message = str(exc_info.value)
    assert "plant" in message, message
    assert "subgraph" in message.lower() or "interior" in message.lower(), message


def test_missing_subgraph_raises_for_legacy_network_node() -> None:
    """``Network`` is not in the component registry but is still spec-reachable."""
    spec = GraphSpec(
        nodes={"net": ComponentSpec(type="Network", params={"hidden_size": 100})}
    )

    with pytest.raises(ValueError) as exc_info:
        spec_to_graph(spec, _registry())

    message = str(exc_info.value)
    assert "net" in message
    assert "no subgraph" in message


@pytest.mark.parametrize("node_type", ["Recurrent Controller", "Simple Feedback Loop"])
def test_missing_subgraph_is_not_filled_from_the_registered_template(node_type: str) -> None:
    """A registered ``template_graph`` is authoring scaffolding, not a fallback."""
    registry = _registry()
    meta = registry.get(node_type)
    assert meta is not None
    assert meta.template_graph is not None, "test presumes this type carries a template"

    spec = GraphSpec(
        nodes={
            "node": ComponentSpec(
                type=node_type,
                params={},
                input_ports=list(meta.input_ports),
                output_ports=list(meta.output_ports),
            )
        }
    )

    with pytest.raises(ValueError, match="requires a subgraph"):
        spec_to_graph(spec, registry)


# ---------------------------------------------------------------------------
# Corollary 1: node types, params, and topology correspond exactly to the spec
# ---------------------------------------------------------------------------


def _two_node_spec() -> GraphSpec:
    return GraphSpec(
        nodes={
            "gain": ComponentSpec(
                type="Gain",
                params={"gain": 2.5},
                input_ports=["input"],
                output_ports=["output"],
            ),
            "lin": ComponentSpec(
                type="Linear",
                params={
                    "input_size": 3,
                    "output_size": 2,
                    "use_bias": False,
                    "activation": "tanh",
                },
                input_ports=["input"],
                output_ports=["output"],
            ),
        },
        wires=[
            WireSpec(
                source_node="gain",
                source_port="output",
                target_node="lin",
                target_port="input",
            )
        ],
        input_ports=["u"],
        output_ports=["y"],
        input_bindings={"u": ("gain", "input")},
        output_bindings={"y": ("lin", "output")},
    )


def test_built_graph_node_set_matches_spec_exactly() -> None:
    graph = spec_to_graph(_two_node_spec(), _registry())

    assert sorted(graph.nodes) == ["gain", "lin"]
    assert type(graph.nodes["gain"]).__name__ == "Gain"
    assert type(graph.nodes["lin"]).__name__ == "Linear"


def test_built_graph_topology_matches_spec_exactly() -> None:
    graph = spec_to_graph(_two_node_spec(), _registry())

    assert [
        (wire.source_node, wire.source_port, wire.target_node, wire.target_port)
        for wire in graph.wires
    ] == [("gain", "output", "lin", "input")]
    assert graph.input_ports == ("u",)
    assert graph.output_ports == ("y",)
    assert graph.input_bindings == {"u": ("gain", "input")}
    assert graph.output_bindings == {"y": ("lin", "output")}


def test_built_component_params_match_spec_exactly() -> None:
    graph = spec_to_graph(_two_node_spec(), _registry())

    assert graph.nodes["gain"].gain == 2.5
    linear = graph.nodes["lin"]
    assert linear.input_size == 3
    assert linear.output_size == 2
    assert linear.use_bias is False
    assert linear.layer.bias is None
    assert linear.activation_name == "tanh"
    assert linear.activation is jax.nn.tanh


def test_round_trip_preserves_node_types_and_topology() -> None:
    graph = spec_to_graph(_two_node_spec(), _registry())

    round_tripped = graph_to_spec(graph)

    assert {name: node.type for name, node in round_tripped.nodes.items()} == {
        "gain": "Gain",
        "lin": "Linear",
    }
    assert [
        (wire.source_node, wire.source_port, wire.target_node, wire.target_port)
        for wire in round_tripped.wires
    ] == [("gain", "output", "lin", "input")]


def test_unknown_component_type_is_rejected_not_defaulted() -> None:
    spec = GraphSpec(nodes={"x": ComponentSpec(type="NoSuchComponent", params={})})

    with pytest.raises(ValueError) as exc_info:
        spec_to_graph(spec, _registry())

    message = str(exc_info.value)
    assert "NoSuchComponent" in message
    assert "Known component types" in message


def test_training_spec_without_a_loss_is_rejected_not_fabricated() -> None:
    """A loss-free run may not be compiled and trained against a fabricated MSE."""
    training_spec = _training_spec()
    training_spec["loss"] = {
        "type": "Composite",
        "label": "empty",
        "children": {},
    }

    with pytest.raises(GraphCompilationError) as exc_info:
        compile_training_run(
            component_registry=_registry(),
            graph_spec=_worker_linear_graph_spec(),
            training_spec=training_spec,
            task_spec={"type": "Generic", "params": {}},
            task_binding_spec=_constant_task_binding_spec(),
            cfg=_worker_cfg(),
        )

    assert exc_info.value.diagnostics[0].code == "worker.missing_loss_terms"


# ---------------------------------------------------------------------------
# Worker-path fixtures (kept local so this module has no cross-test coupling)
# ---------------------------------------------------------------------------


def _worker_linear_graph_spec(input_size: int = 1) -> dict:
    return GraphSpec(
        nodes={
            "readout": ComponentSpec(
                type="Linear",
                params={
                    "input_size": input_size,
                    "output_size": 1,
                    "activation": "identity",
                    "trainable": True,
                },
                input_ports=["input"],
                output_ports=["output"],
            )
        },
        output_ports=["output"],
        output_bindings={"output": ("readout", "output")},
    ).model_dump(mode="json", exclude_none=True)


def _worker_task_binding_spec(
    *,
    value_spec: dict | None,
    expected_shape: list | None = None,
) -> dict:
    exposed: dict[str, Any] = {
        "id": "model_input",
        "label": "Model input",
        "kind": "signal",
        "role": "model_input",
        "path": "inputs.model",
        "bindable": True,
        "expected_shape": expected_shape or ["time", 1],
        "metadata": {},
    }
    if value_spec is not None:
        exposed["value_spec"] = value_spec
    return {
        "schema_version": "feedbax.spec.studio.task_bindings.v2",
        "exposed_data": [exposed],
        "bindings": [
            {
                "id": "task:model_input->readout:input",
                "source_data_id": "model_input",
                "target_node_id": "readout",
                "target_port": "input",
                "role": "model_input",
                "metadata": {},
            }
        ],
        "metadata": {},
    }


def _constant_task_binding_spec() -> dict:
    return _worker_task_binding_spec(
        value_spec={
            "mode": "constant",
            "value": [1.0],
            "dtype": "float32",
            "shape": ["time", 1],
        }
    )


def _training_spec(**overrides) -> dict:
    spec: dict[str, Any] = {
        "optimizer": {"type": "adam", "params": {"learning_rate": 0.1}},
        "loss": {
            "type": "TargetStateLoss",
            "label": "output_zero",
            "selector": "graph_output:output",
            "target_value": [0.0],
            "weight": 1.0,
            "norm": "squared_l2",
        },
        "n_batches": 2,
        "batch_size": 1,
    }
    spec.update(overrides)
    return spec


def _worker_cfg(**overrides) -> SimpleNamespace:
    values: dict[str, Any] = {
        "n_reach_steps": 4,
        "learning_rate": 0.1,
        "grad_clip": 1.0,
        "snapshot_interval": 10,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


# ---------------------------------------------------------------------------
# Invariants the current code violates (issue 8378254 and two new findings).
#
# Each is xfail(strict=True): it fails today, and it will fail loudly as XPASS
# the moment the underlying substitution is removed.
# ---------------------------------------------------------------------------


def test_unknown_activation_name_is_not_silently_relu() -> None:
    """``activation='gelu'`` must be rejected or honoured, never turned into ReLU."""
    spec = GraphSpec(
        nodes={
            "lin": ComponentSpec(
                type="Linear",
                params={"input_size": 2, "output_size": 2, "activation": "gelu"},
                input_ports=["input"],
                output_ports=["output"],
            )
        },
        input_ports=["u"],
        output_ports=["y"],
        input_bindings={"u": ("lin", "input")},
        output_bindings={"y": ("lin", "output")},
    )

    with pytest.raises(ValueError, match="Unknown activation 'gelu'.*Supported values"):
        spec_to_graph(spec, _registry())


def test_leaky_rnn_cell_vocabulary_builds_a_leaky_cell() -> None:
    """The leaky vocabulary carries its dynamics through a GraphSpec round trip."""
    spec = network_template_graph(
        {
            "input_size": 3,
            "hidden_size": 4,
            "out_size": 2,
            "hidden_type": "LeakyRNNCell",
            "activation": "relu",
            "use_bias": False,
            "dt": 0.05,
            "tau": 0.5,
            "use_noise": True,
            "noise_strength": 0.03,
        }
    )
    first = spec_to_graph(spec, _registry())
    second = spec_to_graph(graph_to_spec(first), _registry())

    for graph in (first, second):
        cell = graph.nodes["cell"].cell
        assert float(cell.dt) == pytest.approx(0.05)
        assert float(cell.tau) == pytest.approx(0.5)
        assert cell.use_bias is False
        assert cell.nonlinearity is jax.nn.relu
        assert cell.use_noise is True
        assert float(cell.noise_strength) == pytest.approx(0.03)
        assert float(cell.alpha) == pytest.approx(0.1)


def test_rollout_length_is_not_silently_the_batch_count() -> None:
    """Rollout length is an architectural fact; it must be explicit."""
    cfg = SimpleNamespace(learning_rate=0.1, grad_clip=1.0, snapshot_interval=10)

    with pytest.raises(GraphCompilationError) as exc_info:
        compile_training_run(
            component_registry=_registry(),
            graph_spec=_worker_linear_graph_spec(),
            training_spec=_training_spec(n_batches=7),
            task_spec={"type": "Generic", "params": {}},
            task_binding_spec=_constant_task_binding_spec(),
            cfg=cfg,
        )
    assert exc_info.value.diagnostics[0].code == "worker.missing_rollout_length"


def test_task_data_without_a_value_spec_is_not_silently_zeros() -> None:
    """Unspecified task data is an incomplete model, not a zero signal."""
    with pytest.raises(GraphCompilationError) as exc_info:
        compile_training_run(
            component_registry=_registry(),
            graph_spec=_worker_linear_graph_spec(),
            training_spec=_training_spec(),
            task_spec={"type": "Generic", "params": {}},
            task_binding_spec=_worker_task_binding_spec(value_spec=None),
            cfg=_worker_cfg(),
        )
    assert exc_info.value.diagnostics[0].code == "worker.missing_task_data_value_spec"


def test_missing_task_workspace_is_not_silently_hardcoded() -> None:
    """The task's spatial extent must come from the task spec, not from source."""
    value_spec = {
        "mode": "function",
        "function_id": "delayed_reach_target_position",
        "dtype": "float32",
        "shape": ["time", 4],
    }

    with pytest.raises(GraphCompilationError) as exc_info:
        compile_training_run(
            component_registry=_registry(),
            graph_spec=_worker_linear_graph_spec(input_size=4),
            training_spec=_training_spec(),
            task_spec={"type": "DelayedReaches", "params": {}},
            task_binding_spec=_worker_task_binding_spec(
                value_spec=value_spec,
                expected_shape=["time", 4],
            ),
            cfg=_worker_cfg(n_reach_steps=3),
        )
    assert exc_info.value.diagnostics[0].code == "worker.missing_task_workspace"


def test_composite_with_a_subgraph_does_not_require_unused_outer_params() -> None:
    """If the interior is authoritative, the outer params cannot gate the build."""
    interior = _gain_interior(3.0, in_port="excitation", out_port="torques")
    spec = _composite_spec(
        "Arm6MuscleRigidTendon",
        {},  # no outer params at all; nothing here is used to build
        interior,
        in_port="excitation",
        out_port="torques",
    )

    # Observed today: ValueError "Arm6MuscleRigidTendon node 'plant' is missing
    # required parameter(s): 'dt'" — a parameter that is never read, because the
    # causal interior is compiled instead.
    graph = spec_to_graph(spec, _registry())

    assert graph.nodes["plant"].nodes["scale"].gain == 3.0


def test_unrecognized_component_param_is_not_silently_dropped() -> None:
    """A spec param that reaches nothing in the built model must be rejected."""
    spec = GraphSpec(
        nodes={
            "lin": ComponentSpec(
                type="Linear",
                params={
                    "input_size": 2,
                    "output_size": 2,
                    "nonexistent_param": 99,
                },
                input_ports=["input"],
                output_ports=["output"],
            )
        },
        input_ports=["u"],
        output_ports=["y"],
        input_bindings={"u": ("lin", "input")},
        output_bindings={"y": ("lin", "output")},
    )

    # The strict declaration owns the accepted vocabulary, so an unknown field
    # cannot disappear between the authored spec and the runtime model.
    with pytest.raises(REFUSALS):
        spec_to_graph(spec, _registry())
