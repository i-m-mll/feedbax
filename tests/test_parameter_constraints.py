from __future__ import annotations

from types import SimpleNamespace

from feedbax.component_registry import ComponentRegistry

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest

from feedbax.contracts.graph import ComponentSpec, GraphSpec, ParameterConstraintSpec
from feedbax.contracts.migrations import UnsupportedSpecVersion
from feedbax.compiler.templates import (
    network_template_graph,
    recurrent_controller_template_graph,
)
from feedbax.models.networks import (
    LeakyRNNCell,
    POPULATION_STRUCTURE_SCHEMA_ID,
    POPULATION_STRUCTURE_SCHEMA_VERSION,
    PopulationStructure,
    recurrent_controller,
    lower_population_constraints,
    population_input_kernel_mask,
    population_readout_kernel_mask,
    population_structure_from_spec,
)
from feedbax.objectives.loss import TermTree
from feedbax.runtime.components import Linear
from feedbax.runtime.graph import Graph
from feedbax.runtime.parameter_constraints import apply_parameter_constraints
from feedbax.compiler.serialization import graph_to_spec
from feedbax.tasks import TaskTrialSpec
from feedbax.training.studio_executor import StudioUpdateKernel
from feedbax.training.trainer import training_step_for_abstract_loss
from tests.support.graph_compiler_test_support import spec_to_graph


def _linear_constraint_spec(mask) -> GraphSpec:
    return GraphSpec(
        nodes={
            "readout": ComponentSpec(
                type="Linear",
                params={"input_size": 2, "output_size": 2, "activation": "identity"},
                input_ports=["input"],
                output_ports=["output"],
            )
        },
        input_ports=["input"],
        output_ports=["output"],
        input_bindings={"input": ("readout", "input")},
        output_bindings={"output": ("readout", "output")},
        parameter_constraints=[
            ParameterConstraintSpec(node="readout", role="weight", mask=mask, value=0.0)
        ],
    )


def test_graphspec_parameter_constraints_materialize_and_round_trip() -> None:
    spec = _linear_constraint_spec([[1, 0], [0, 1]])

    graph = spec_to_graph(spec, ComponentRegistry())

    weight = graph.nodes["readout"].layer.weight
    assert weight[0, 1] == 0.0
    assert weight[1, 0] == 0.0

    roundtrip = graph_to_spec(graph)
    assert roundtrip.parameter_constraints == spec.parameter_constraints


def test_remove_node_preserves_projection_for_surviving_nodes_across_round_trip() -> None:
    registry = ComponentRegistry()
    linear = ComponentSpec(
        type="Linear",
        params={"input_size": 1, "output_size": 1, "initialization": "zeros"},
        input_ports=["input"],
        output_ports=["output"],
    )
    child = GraphSpec(
        nodes={"readout": linear},
        parameter_constraints=[
            ParameterConstraintSpec(node="readout", role="weight", mask=False, value=3.0)
        ],
    )
    original = spec_to_graph(
        GraphSpec(
            nodes={"readout": linear, "readout_aux": linear, "child": ComponentSpec(type="Subgraph")},
            subgraphs={"child": child},
            parameter_constraints=[
                ParameterConstraintSpec(node="readout", role="weight", mask=False, value=7.0),
                ParameterConstraintSpec(node="readout_aux", role="weight", mask=False, value=5.0),
            ],
        ),
        registry,
    )
    edited = original.remove_node("readout")
    assert "readout" not in edited.nodes
    edited = eqx.tree_at(
        lambda graph: graph.get_node_attrs("readout_aux", "layer.weight")
        + graph.get_node_attrs("child.readout", "layer.weight"),
        edited,
        (jnp.zeros((1, 1)), jnp.zeros((1, 1))),
    )

    projected = apply_parameter_constraints(edited)
    serialized = graph_to_spec(edited, registry).model_dump_json()
    restored = spec_to_graph(GraphSpec.model_validate_json(serialized), registry)
    for graph in (projected, restored):
        assert graph.get_node("readout_aux").layer.weight.item() == 5.0
        assert graph.get_node("child.readout").layer.weight.item() == 3.0


def test_same_name_replacement_keeps_its_initialized_weight() -> None:
    original = Graph(
        nodes={"readout": Linear(1, 1, key=jax.random.key(0))},
        parameter_constraints=({"node": "readout", "role": "weight", "mask": False, "value": 7.0},),
    )
    fresh = Linear(1, 1, initialization="zeros", key=jax.random.key(1))
    edited = original.remove_node("readout").add_node("readout", fresh)

    assert apply_parameter_constraints(edited).get_node("readout").layer.weight.item() == 0.0
    assert apply_parameter_constraints(original).get_node("readout").layer.weight.item() == 7.0


@pytest.mark.parametrize("entry_point", ["supervised", "studio"])
def test_nested_parameter_constraints_survive_gradient_updates(entry_point: str) -> None:
    spec = GraphSpec(
        nodes={
            "readout": ComponentSpec(
                type="Linear",
                params={
                    "input_size": 2,
                    "output_size": 1,
                    "use_bias": False,
                    "initialization": "zeros",
                },
                input_ports=["input"],
                output_ports=["output"],
            ),
        },
        input_ports=["input"],
        output_ports=["output"],
        input_bindings={"input": ("readout", "input")},
        output_bindings={"output": ("readout", "output")},
        parameter_constraints=[
            ParameterConstraintSpec(node="readout", role="weight", mask=[[0, 1]], value=0.25)
        ],
    )
    for name in ("inner", "outer"):
        spec = GraphSpec(
            nodes={
                name: ComponentSpec(type="Subgraph", input_ports=["input"], output_ports=["output"])
            },
            subgraphs={name: spec},
            input_ports=["input"],
            output_ports=["output"],
            input_bindings={"input": (name, "input")},
            output_bindings={"output": (name, "output")},
        )
    graph = spec_to_graph(spec, ComponentRegistry())
    key = jax.random.PRNGKey(0)
    inputs = jnp.ones(2)
    optimizer = optax.sgd(0.1)
    opt_state = optimizer.init(eqx.filter(graph, eqx.is_inexact_array))

    def weight(model):
        return model.nodes["outer"].nodes["inner"].nodes["readout"].layer.weight

    def predict(model, inputs, key):
        outputs, _ = model({"input": inputs}, model.init_state(key=key), key=key)
        return outputs

    def squared_error(outputs):
        return jnp.mean((outputs["output"] - 1.0) ** 2)

    assert jnp.array_equal(weight(graph), jnp.array([[0.25, 0.0]]))
    # With all-ones inputs and initial output 0.25, both weights have gradient -1.5.

    if entry_point == "supervised":
        trial_specs = TaskTrialSpec(inits={}, targets={}, inputs={"input": inputs[None, None]})
        keys = jax.random.split(key, 1)
        init_states = eqx.filter_vmap(lambda key: graph.init_state(key=key))(keys)

        def loss(_states, trials, model):
            outputs = jax.vmap(lambda trial_inputs: predict(model, trial_inputs, key))(
                trials.inputs["input"][:, 0]
            )
            return TermTree.leaf("squared_error", squared_error(outputs))

        trained, _, _, _ = training_step_for_abstract_loss(
            optimizer, graph, opt_state, loss, eqx.is_inexact_array, trial_specs, init_states, keys
        )
    else:
        kernel = StudioUpdateKernel(
            compiled=SimpleNamespace(graph=graph, trainable_filter=eqx.is_inexact_array),
            optimizer=optimizer,
            rollout_fn=lambda model, _plan, key: predict(model, inputs, key),
            evaluate_loss_fn=lambda _plan, outputs: (squared_error(outputs), {}),
        )
        trained = kernel(
            {"model": graph, "optimizer": opt_state, "prng": key, "batch_counter": jnp.int32(0)}
        )["model"]

    assert weight(trained)[0, 0] == 0.25
    assert float(weight(trained)[0, 1]) == pytest.approx(0.15)


@pytest.mark.parametrize(
    ("component_type", "role", "hidden_size", "expected_rows"),
    [
        ("GRU", "input_kernel", 3, 9),
        ("LSTM", "input_kernel", 3, 12),
    ],
)
def test_recurrent_input_kernel_constraints_use_stable_roles(
    component_type: str,
    role: str,
    hidden_size: int,
    expected_rows: int,
) -> None:
    mask = [[1, 0] for _ in range(expected_rows)]
    graph = spec_to_graph(
        GraphSpec(
            nodes={
                "cell": ComponentSpec(
                    type=component_type,
                    params={"input_size": 2, "hidden_size": hidden_size},
                    input_ports=["input"],
                    output_ports=["output"],
                )
            },
            parameter_constraints=[
                ParameterConstraintSpec(node="cell", role=role, mask=mask, value=0.0)
            ],
        ),
        ComponentRegistry(),
    )

    assert jnp.all(graph.nodes["cell"].cell.weight_ih[:, 1] == 0.0)


def _fixed_population_structure() -> PopulationStructure:
    return PopulationStructure.from_indices(
        input_only_indices=[0],
        readout_only_indices=[1],
        recurrent_only_indices=[2],
        input_readout_indices=[3],
    )


def test_population_structure_to_spec_uses_governed_nested_schema_identity() -> None:
    spec = _fixed_population_structure().to_spec()

    assert spec["schema_id"] == POPULATION_STRUCTURE_SCHEMA_ID
    assert spec["schema_version"] == POPULATION_STRUCTURE_SCHEMA_VERSION
    assert spec["assignment"] == "explicit"
    assert spec["input_only_indices"] == [0]
    assert spec["readout_only_indices"] == [1]
    assert spec["recurrent_only_indices"] == [2]
    assert spec["input_readout_indices"] == [3]


def test_population_structure_from_spec_accepts_current_explicit_indices() -> None:
    restored = population_structure_from_spec(4, _fixed_population_structure().to_spec())

    assert restored.n_input_only == 1
    assert restored.n_readout_only == 1
    assert restored.n_recurrent_only == 1
    assert restored.n_input_readout == 1
    assert jnp.array_equal(restored.input_only_indices, jnp.array([0]))
    assert jnp.array_equal(restored.readout_only_indices, jnp.array([1]))
    assert jnp.array_equal(restored.recurrent_only_indices, jnp.array([2]))
    assert jnp.array_equal(restored.input_readout_indices, jnp.array([3]))


@pytest.mark.parametrize(
    "schema_version",
    ["feedbax.population_structure.v1", "feedbax.spec.population_structure.v99"],
)
def test_population_structure_from_spec_rejects_old_or_unknown_schema_versions(
    schema_version: str,
) -> None:
    spec = _fixed_population_structure().to_spec()
    spec["schema_version"] = schema_version

    with pytest.raises(ValueError, match="PopulationStructureSpec"):
        population_structure_from_spec(4, spec)


def test_population_structure_from_spec_rejects_future_schema_version_explicitly() -> None:
    spec = _fixed_population_structure().to_spec()
    spec["schema_version"] = f"{POPULATION_STRUCTURE_SCHEMA_ID}.v2"

    with pytest.raises(UnsupportedSpecVersion) as excinfo:
        population_structure_from_spec(4, spec)

    message = str(excinfo.value)
    assert "future population-structure versions" in message
    assert f"current_version='{POPULATION_STRUCTURE_SCHEMA_VERSION}'" in message


def test_population_structure_from_spec_rejects_wrong_schema_id() -> None:
    spec = _fixed_population_structure().to_spec()
    spec["schema_id"] = "feedbax.spec.other"

    with pytest.raises(ValueError, match="schema_id"):
        population_structure_from_spec(4, spec)


def test_population_lowering_repeats_gate_rows_and_selects_readout_columns() -> None:
    population = _fixed_population_structure()

    gru_constraints = lower_population_constraints(
        population,
        hidden_size=4,
        input_size=2,
        out_size=2,
        cell_type="GRU",
    )
    lstm_constraints = lower_population_constraints(
        population,
        hidden_size=4,
        input_size=2,
        out_size=2,
        cell_type="LSTM",
    )

    gru_input_mask = jnp.asarray(gru_constraints[0].mask)
    lstm_input_mask = jnp.asarray(lstm_constraints[0].mask)
    readout_mask = jnp.asarray(gru_constraints[1].mask)
    expected_gate = jnp.array(
        [
            [1, 1],
            [0, 0],
            [0, 0],
            [1, 1],
        ],
        dtype=bool,
    )

    assert jnp.array_equal(gru_input_mask, jnp.tile(expected_gate, (3, 1)))
    assert jnp.array_equal(lstm_input_mask, jnp.tile(expected_gate, (4, 1)))
    assert jnp.array_equal(
        readout_mask,
        jnp.array(
            [
                [0, 1, 0, 1],
                [0, 1, 0, 1],
            ],
            dtype=bool,
        ),
    )


def test_network_template_population_constraints_materialize_without_recurrent_masks() -> None:
    population = _fixed_population_structure()
    spec = network_template_graph(
        {
            "input_size": 2,
            "hidden_size": 4,
            "out_size": 2,
            "population_structure": population.to_spec(),
        }
    )
    subgraph = spec

    assert [
        (constraint.node, constraint.role) for constraint in subgraph.parameter_constraints
    ] == [
        ("cell", "input_kernel"),
        ("readout", "weight"),
    ]

    graph = spec_to_graph(spec, ComponentRegistry())
    input_mask = population_input_kernel_mask(population, 2, gate_count=3)
    readout_mask = population_readout_kernel_mask(population, 2)

    assert jnp.all(graph.nodes["cell"].cell.weight_ih[~input_mask.astype(bool)] == 0.0)
    assert jnp.all(graph.nodes["readout"].layer.weight[~readout_mask.astype(bool)] == 0.0)


@pytest.mark.parametrize(
    ("hidden_type", "gate_count"),
    [(eqx.nn.GRUCell, 3), (eqx.nn.LSTMCell, 4), (LeakyRNNCell, 1)],
)
def test_recurrent_controller_enforces_population_masks_with_encoder(
    hidden_type, gate_count: int
) -> None:
    population = _fixed_population_structure()
    network = recurrent_controller(
        2,
        4,
        out_size=2,
        encoding_size=3,
        hidden_type=hidden_type,
        population_structure=population,
        key=jax.random.key(1),
    )

    input_mask = population_input_kernel_mask(population, 3, gate_count=gate_count).astype(bool)
    readout_mask = population_readout_kernel_mask(population, 2).astype(bool)

    assert isinstance(network.nodes["encoder"], Linear)
    assert isinstance(network.nodes["readout"], Linear)
    assert jnp.array_equal(network.nodes["cell"].cell.weight_ih == 0, ~input_mask)
    assert jnp.array_equal(network.nodes["readout"].layer.weight == 0, ~readout_mask)


def test_recurrent_controller_without_readout_only_constrains_cell():
    network = recurrent_controller(
        2, 4, population_structure=_fixed_population_structure(), key=jax.random.key(0)
    )

    assert set(network.nodes) == {"input_mux", "cell"}
    assert [(item.node, item.role) for item in network.parameter_constraints] == [
        ("cell", "input_kernel")
    ]


def test_population_constraints_project_after_synthetic_update() -> None:
    population = _fixed_population_structure()
    graph = spec_to_graph(
        recurrent_controller_template_graph(
            input_size=2,
            hidden_size=4,
            out_size=2,
            cell_type="LSTM",
            population_structure=population,
        ),
        ComponentRegistry(),
    )
    graph = eqx.tree_at(
        lambda g: g.nodes["cell"].cell.weight_ih,
        graph,
        jnp.ones_like(graph.nodes["cell"].cell.weight_ih),
    )
    graph = eqx.tree_at(
        lambda g: g.nodes["readout"].layer.weight,
        graph,
        jnp.ones_like(graph.nodes["readout"].layer.weight),
    )
    graph = eqx.tree_at(
        lambda g: g.nodes["cell"].cell.weight_hh,
        graph,
        jnp.ones_like(graph.nodes["cell"].cell.weight_hh),
    )

    projected = apply_parameter_constraints(graph)
    input_mask = population_input_kernel_mask(population, 2, gate_count=4).astype(bool)
    readout_mask = population_readout_kernel_mask(population, 2).astype(bool)

    assert jnp.all(projected.nodes["cell"].cell.weight_ih[~input_mask] == 0.0)
    assert jnp.all(projected.nodes["cell"].cell.weight_hh == 1.0)
    assert jnp.all(projected.nodes["readout"].layer.weight[~readout_mask] == 0.0)


def test_parameter_constraints_reject_incompatible_mask_shape() -> None:
    with pytest.raises(ValueError, match="mask shape"):
        spec_to_graph(
            _linear_constraint_spec([[1, 0, 1]]), ComponentRegistry()
        )


def test_apply_parameter_constraints_rejects_unsupported_role() -> None:
    graph = spec_to_graph(
        _linear_constraint_spec([[1, 1], [1, 1]]), ComponentRegistry()
    )
    bad = ParameterConstraintSpec(node="readout", role="input_kernel", mask=[[1, 1], [1, 1]])

    with pytest.raises(ValueError, match="Unsupported Linear parameter role"):
        apply_parameter_constraints(graph, [bad])
