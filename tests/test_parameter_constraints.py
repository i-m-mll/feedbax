from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest

from feedbax._mapping import WhereDict
from feedbax.contracts.graph import ComponentSpec, GraphSpec, ParameterConstraintSpec
from feedbax.runtime.graph import Graph
from feedbax.contracts.graphs.templates import network_template_graph, recurrent_controller_template_graph
from feedbax.objectives.loss import AbstractLoss
from feedbax.models.networks import (
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
from feedbax.contracts.graphs.serialization import graph_to_spec, spec_to_graph
from feedbax.tasks import AbstractTask, TaskInterventionSpecs, TaskTrialSpec, TrialSpecDependency
from feedbax.training.trainer import TaskTrainer


class _WeightSumLoss(AbstractLoss):
    label: str = "weight_sum"

    def term(self, states, trial_specs, model):
        return jnp.ones((1,)) * jnp.sum(model.nodes["readout"].layer.weight)


class _TinyTask(AbstractTask):
    loss_func: AbstractLoss = _WeightSumLoss()
    n_steps: int = 2
    seed_validation: int = 0
    intervention_specs: TaskInterventionSpecs = TaskInterventionSpecs()
    input_dependencies: dict[str, TrialSpecDependency] = eqx.field(default_factory=dict)

    def get_train_trial(self, key, batch_info=None):
        return TaskTrialSpec(
            inits=WhereDict(),
            targets=WhereDict(),
            inputs=jnp.ones((self.n_steps, 1)),
        )

    def get_validation_trials(self, key):
        return TaskTrialSpec(
            inits=WhereDict(),
            targets=WhereDict(),
            inputs=jnp.ones((self.n_validation_trials, self.n_steps, 1)),
        )

    def validation_plots(self, states, trial_specs=None):
        return {}

    @property
    def n_validation_trials(self) -> int:
        return 1


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

    graph = spec_to_graph(spec)

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
        )
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

    assert [(constraint.node, constraint.role) for constraint in subgraph.parameter_constraints] == [
        ("cell", "input_kernel"),
        ("readout", "weight"),
    ]

    graph = spec_to_graph(spec)
    input_mask = population_input_kernel_mask(population, 2, gate_count=3)
    readout_mask = population_readout_kernel_mask(population, 2)

    assert jnp.all(graph.nodes["cell"].cell.weight_ih[~input_mask.astype(bool)] == 0.0)
    assert jnp.all(graph.nodes["readout"].layer.weight[~readout_mask.astype(bool)] == 0.0)


@pytest.mark.parametrize(
    ("hidden_type", "cell_type", "gate_count"),
    [
        (eqx.nn.GRUCell, "GRU", 3),
        (eqx.nn.LSTMCell, "LSTM", 4),
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
        )
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


def test_population_constraints_project_after_synthetic_update() -> None:
    population = _fixed_population_structure()
    graph = spec_to_graph(
        recurrent_controller_template_graph(
            input_size=2,
            hidden_size=4,
            out_size=2,
            cell_type="LSTM",
            population_structure=population,
        )
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

    restored = graph_to_spec(spec_to_graph(spec))
    assert restored.parameter_constraints == spec.parameter_constraints


def test_parameter_constraints_reject_incompatible_mask_shape() -> None:
    with pytest.raises(ValueError, match="mask shape"):
        spec_to_graph(_linear_constraint_spec([[1, 0, 1]]))


def test_task_trainer_projects_constraints_after_optimizer_update() -> None:
    spec = GraphSpec(
        nodes={
            "readout": ComponentSpec(
                type="Linear",
                params={"input_size": 1, "output_size": 1, "activation": "identity"},
                input_ports=["input"],
                output_ports=["output"],
            )
        },
        input_ports=["input"],
        output_ports=["output"],
        input_bindings={"input": ("readout", "input")},
        output_bindings={"output": ("readout", "output")},
        parameter_constraints=[
            ParameterConstraintSpec(node="readout", role="weight", mask=[[0]], value=0.0)
        ],
    )
    model = spec_to_graph(spec)
    trainer = TaskTrainer(optimizer=optax.sgd(1.0), checkpointing=False)

    trained, _, _ = trainer(
        _TinyTask(),
        model,
        n_batches=1,
        batch_size=1,
        where_train=lambda graph: graph.nodes["readout"],
        disable_progress=True,
        key=jax.random.PRNGKey(0),
    )

    assert trained.nodes["readout"].layer.weight[0, 0] == 0.0


def test_apply_parameter_constraints_rejects_unsupported_role() -> None:
    graph = spec_to_graph(_linear_constraint_spec([[1, 1], [1, 1]]))
    bad = ParameterConstraintSpec(node="readout", role="input_kernel", mask=[[1, 1], [1, 1]])

    with pytest.raises(ValueError, match="Unsupported Linear parameter role"):
        apply_parameter_constraints(graph, [bad])
