from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest

from feedbax._mapping import WhereDict
from feedbax.contracts.graph import ComponentSpec, GraphSpec, ParameterConstraintSpec
from feedbax.loss import AbstractLoss
from feedbax.parameter_constraints import apply_parameter_constraints
from feedbax.serialization import graph_to_spec, spec_to_graph
from feedbax.task import AbstractTask, TaskInterventionSpecs, TaskTrialSpec, TrialSpecDependency
from feedbax.train import TaskTrainer


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
