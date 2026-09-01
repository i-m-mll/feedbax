from __future__ import annotations

from feedbax.component_registry import ComponentRegistry

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from feedbax.contracts.graph import ComponentSpec, GraphSpec, ParameterConstraintSpec
from feedbax.contracts.migrations import UnsupportedSpecVersion
from feedbax.runtime.graph import Graph
from feedbax.contracts.graphs.templates import (
    network_template_graph,
    recurrent_controller_template_graph,
)
from feedbax.models.networks import (
    LeakyRNNCell,
    MaskedLinear,
    POPULATION_STRUCTURE_SCHEMA_ID,
    POPULATION_STRUCTURE_SCHEMA_VERSION,
    PopulationStructure,
    SimpleStagedNetwork,
    lower_population_constraints,
    population_input_kernel_mask,
    population_readout_kernel_mask,
    population_structure_from_spec,
)
from feedbax.runtime.parameter_constraints import apply_parameter_constraints
from feedbax.contracts.graphs import builders as graph_builders
from feedbax.contracts.graphs.serialization import graph_to_spec
from tests.graph_compiler_test_support import spec_to_graph


def _floating_leaf_dtypes(tree) -> set[jnp.dtype]:
    return {
        leaf.dtype
        for leaf in jax.tree.leaves(tree)
        if eqx.is_array(leaf) and jnp.issubdtype(leaf.dtype, jnp.floating)
    }


def test_compat_builders_preserve_legacy_default_dtype_under_x64(tmp_path) -> None:
    builders = [
        lambda params: graph_builders._build_linear({"input_size": 2, "output_size": 3, **params}),
        lambda params: graph_builders._build_mlp(
            {"input_size": 2, "output_size": 3, "hidden_sizes": [4], **params}
        ),
        lambda params: graph_builders._build_gru({"input_size": 2, "hidden_size": 3, **params}),
        lambda params: graph_builders._build_lstm({"input_size": 2, "hidden_size": 3, **params}),
        lambda params: graph_builders._build_network(
            {"input_size": 2, "hidden_size": 3, "out_size": 1, **params}
        ),
    ]
    with jax.experimental.enable_x64():
        for index, build in enumerate(builders):
            legacy = build({})
            explicit = build({"dtype": "float32"})
            path = tmp_path / f"legacy_dtype_{index}.eqx"
            eqx.tree_serialise_leaves(path, legacy)
            loaded = eqx.tree_deserialise_leaves(path, build({}))

            assert _floating_leaf_dtypes(legacy) == {jnp.dtype(jnp.float64)}
            assert _floating_leaf_dtypes(loaded) == {jnp.dtype(jnp.float64)}
            assert _floating_leaf_dtypes(explicit) == {jnp.dtype(jnp.float32)}


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

    graph = spec_to_graph(spec, ComponentRegistry(load_user_components=False))

    weight = graph.nodes["readout"].layer.weight
    assert weight[0, 1] == 0.0
    assert weight[1, 0] == 0.0

    roundtrip = graph_to_spec(graph)
    assert roundtrip.parameter_constraints == spec.parameter_constraints


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
        ComponentRegistry(load_user_components=False),
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

    graph = spec_to_graph(spec, ComponentRegistry(load_user_components=False))
    input_mask = population_input_kernel_mask(population, 2, gate_count=3)
    readout_mask = population_readout_kernel_mask(population, 2)

    assert jnp.all(graph.nodes["cell"].cell.weight_ih[~input_mask.astype(bool)] == 0.0)
    assert jnp.all(graph.nodes["readout"].layer.weight[~readout_mask.astype(bool)] == 0.0)


@pytest.mark.parametrize(
    ("hidden_type", "cell_type", "gate_count"),
    [
        (eqx.nn.GRUCell, "GRU", 3),
        (eqx.nn.LSTMCell, "LSTM", 4),
        (LeakyRNNCell, "VanillaRNN", 1),
    ],
)
def test_population_constraints_match_simplestagednetwork_fixed_assignment(
    hidden_type,
    cell_type: str,
    gate_count: int,
) -> None:
    population = _fixed_population_structure()
    legacy = SimpleStagedNetwork(
        input_size=2,
        hidden_size=4,
        out_size=2,
        hidden_type=hidden_type,
        population_structure=population,
        key=jax.random.PRNGKey(1),
    )
    graph = spec_to_graph(
        recurrent_controller_template_graph(
            input_size=2,
            hidden_size=4,
            out_size=2,
            cell_type=cell_type,
            population_structure=population,
        ),
        ComponentRegistry(load_user_components=False),
    )

    input_mask = population_input_kernel_mask(population, 2, gate_count=gate_count).astype(bool)
    legacy_readout_weight = legacy.readout.weight
    if isinstance(legacy.readout, MaskedLinear):
        legacy_readout_weight = legacy_readout_weight * legacy.readout.mask

    assert jnp.array_equal(legacy.hidden.weight_ih == 0.0, ~input_mask)
    assert jnp.array_equal(graph.nodes["cell"].cell.weight_ih == 0.0, ~input_mask)
    assert jnp.array_equal(
        legacy_readout_weight == 0.0,
        graph.nodes["readout"].layer.weight == 0.0,
    )


def test_simplestagednetwork_defaults_to_legacy_maskedlinear_for_all_ones_masks() -> None:
    population = PopulationStructure.create(
        hidden_size=4,
        key=jax.random.PRNGKey(0),
    )

    network = SimpleStagedNetwork(
        input_size=2,
        hidden_size=4,
        out_size=2,
        encoding_size=3,
        population_structure=population,
        key=jax.random.PRNGKey(1),
    )

    assert isinstance(network.encoder, MaskedLinear)
    assert isinstance(network.readout, MaskedLinear)
    assert jnp.all(network.encoder.mask)
    assert jnp.all(network.readout.mask)
    assert jnp.issubdtype(network.encoder.mask.dtype, jnp.floating)
    assert jnp.issubdtype(network.readout.mask.dtype, jnp.floating)


def test_simplestagednetwork_uses_plain_linear_for_explicit_all_ones_mask_opt_in() -> None:
    population = PopulationStructure.create(
        hidden_size=4,
        key=jax.random.PRNGKey(0),
    )

    network = SimpleStagedNetwork(
        input_size=2,
        hidden_size=4,
        out_size=2,
        encoding_size=3,
        population_structure=population,
        population_mask_mode="plain_all_ones",
        key=jax.random.PRNGKey(1),
    )

    assert isinstance(network.encoder, eqx.nn.Linear)
    assert isinstance(network.readout, eqx.nn.Linear)
    assert not isinstance(network.encoder, MaskedLinear)
    assert not isinstance(network.readout, MaskedLinear)


def test_simplestagednetwork_keeps_maskedlinear_for_structured_population_masks() -> None:
    network = SimpleStagedNetwork(
        input_size=2,
        hidden_size=4,
        out_size=2,
        encoding_size=3,
        population_structure=_fixed_population_structure(),
        population_mask_mode="plain_all_ones",
        key=jax.random.PRNGKey(1),
    )

    assert isinstance(network.readout, MaskedLinear)
    assert not jnp.all(network.readout.mask == 1)
    assert network.readout.mask.dtype == jnp.bool_


def test_legacy_all_ones_maskedlinear_serialization_loads_with_default_template(tmp_path) -> None:
    population = PopulationStructure.create(
        hidden_size=4,
        key=jax.random.PRNGKey(0),
    )
    legacy = SimpleStagedNetwork(
        input_size=2,
        hidden_size=4,
        out_size=2,
        encoding_size=3,
        population_structure=population,
        key=jax.random.PRNGKey(1),
    )
    optimized_template = SimpleStagedNetwork(
        input_size=2,
        hidden_size=4,
        out_size=2,
        encoding_size=3,
        population_structure=population,
        population_mask_mode="plain_all_ones",
        key=jax.random.PRNGKey(1),
    )
    default_template = SimpleStagedNetwork(
        input_size=2,
        hidden_size=4,
        out_size=2,
        encoding_size=3,
        population_structure=population,
        key=jax.random.PRNGKey(1),
    )
    path = tmp_path / "legacy.eqx"
    eqx.tree_serialise_leaves(path, legacy)

    with pytest.raises(Exception):
        eqx.tree_deserialise_leaves(path, optimized_template)

    loaded = eqx.tree_deserialise_leaves(path, default_template)

    assert isinstance(loaded.encoder, MaskedLinear)
    assert isinstance(loaded.readout, MaskedLinear)
    assert jnp.array_equal(loaded.encoder.linear.weight, legacy.encoder.linear.weight)
    assert jnp.array_equal(loaded.readout.linear.weight, legacy.readout.linear.weight)
    assert len(jax.tree.leaves(optimized_template)) != len(jax.tree.leaves(default_template))


def test_legacy_structured_float_mask_serialization_loads_with_default_template(tmp_path) -> None:
    population = _fixed_population_structure()
    legacy = SimpleStagedNetwork(
        input_size=2,
        hidden_size=4,
        out_size=2,
        encoding_size=3,
        population_structure=population,
        key=jax.random.PRNGKey(1),
    )
    default_template = SimpleStagedNetwork(
        input_size=2,
        hidden_size=4,
        out_size=2,
        encoding_size=3,
        population_structure=population,
        key=jax.random.PRNGKey(1),
    )
    explicit_new_template = SimpleStagedNetwork(
        input_size=2,
        hidden_size=4,
        out_size=2,
        encoding_size=3,
        population_structure=population,
        population_mask_mode="plain_all_ones",
        key=jax.random.PRNGKey(1),
    )
    path = tmp_path / "legacy_structured.eqx"
    eqx.tree_serialise_leaves(path, legacy)

    loaded = eqx.tree_deserialise_leaves(path, default_template)

    assert isinstance(loaded.readout, MaskedLinear)
    assert jnp.issubdtype(loaded.readout.mask.dtype, jnp.floating)
    assert jnp.array_equal(loaded.readout.mask, legacy.readout.mask)
    assert explicit_new_template.readout.mask.dtype == jnp.bool_


def test_network_compat_builder_requires_explicit_plain_all_ones_mask_opt_in() -> None:
    population = PopulationStructure.create(
        hidden_size=4,
        key=jax.random.PRNGKey(0),
    )
    base_params = {
        "input_size": 2,
        "hidden_size": 4,
        "out_size": 2,
        "encoding_size": 3,
        "population_structure": {
            "assignment": "explicit",
            "input_only_indices": population.input_only_indices.tolist(),
            "readout_only_indices": population.readout_only_indices.tolist(),
            "recurrent_only_indices": population.recurrent_only_indices.tolist(),
            "input_readout_indices": population.input_readout_indices.tolist(),
        },
    }

    legacy = graph_builders._build_network(base_params)
    optimized = graph_builders._build_network(
        {**base_params, "population_mask_mode": "plain_all_ones"},
    )

    assert isinstance(legacy.encoder, MaskedLinear)
    assert isinstance(legacy.readout, MaskedLinear)
    assert isinstance(optimized.encoder, eqx.nn.Linear)
    assert isinstance(optimized.readout, eqx.nn.Linear)


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
        ComponentRegistry(load_user_components=False),
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


def test_population_constraints_round_trip_from_legacy_network_serialization() -> None:
    population = _fixed_population_structure()
    legacy = SimpleStagedNetwork(
        input_size=2,
        hidden_size=4,
        out_size=2,
        population_structure=population,
        key=jax.random.PRNGKey(2),
    )
    spec = graph_to_spec(
        Graph(
            nodes={"network": legacy},
            input_ports=("input", "feedback"),
            output_ports=("output",),
            input_bindings={"input": ("network", "input"), "feedback": ("network", "feedback")},
            output_bindings={"output": ("network", "output")},
        )
    )

    assert spec.subgraphs is None
    assert "network" not in spec.nodes
    assert spec.parameter_constraints == list(
        lower_population_constraints(
            population,
            hidden_size=4,
            input_size=2,
            out_size=2,
            cell_type="GRU",
            cell_node="network_cell",
            readout_node="network_readout",
        )
    )

    restored = graph_to_spec(spec_to_graph(spec, ComponentRegistry(load_user_components=False)))
    assert restored.parameter_constraints == spec.parameter_constraints


def test_parameter_constraints_reject_incompatible_mask_shape() -> None:
    with pytest.raises(ValueError, match="mask shape"):
        spec_to_graph(
            _linear_constraint_spec([[1, 0, 1]]), ComponentRegistry(load_user_components=False)
        )


def test_apply_parameter_constraints_rejects_unsupported_role() -> None:
    graph = spec_to_graph(
        _linear_constraint_spec([[1, 1], [1, 1]]), ComponentRegistry(load_user_components=False)
    )
    bad = ParameterConstraintSpec(node="readout", role="input_kernel", mask=[[1, 1], [1, 1]])

    with pytest.raises(ValueError, match="Unsupported Linear parameter role"):
        apply_parameter_constraints(graph, [bad])
