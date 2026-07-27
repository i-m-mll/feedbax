from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import pytest

from feedbax.config.mapping import WhereDict
from feedbax.contracts.graph import ComponentSpec, GraphSpec
from feedbax.contracts.graphs.serialization import graph_to_spec, spec_to_graph
from feedbax.contracts.migrations import UnsupportedComponentMigration
from feedbax.mechanics import (
    STRUCTURAL_LINEAR_STATE_SPACE_PARAM_SCHEMA_VERSION,
    StructuralLinearDynamicsPerturbation,
    StructuralLinearStateSpace,
    structural_linear_transition,
)
from feedbax.runtime.graph import init_state_from_component
from feedbax.tasks import TaskTrialSpec, TrialTimeline, prepare_trial


def _spec(*, active: bool = False) -> GraphSpec:
    return GraphSpec(
        nodes={
            "plant": ComponentSpec(
                type="StructuralLinearStateSpace",
                params={
                    "A": [[1.0, 0.25], [0.0, 1.0]],
                    "B": [[0.0], [1.0]],
                    "B_w": [[0.0], [1.0]],
                    "delta_A": [[0.0, 0.5], [0.0, 0.0]],
                    "scale": 1.0,
                    "active": active,
                    "label": "structural_field",
                    "dt": 0.1,
                    "initial_state": [0.0, 2.0],
                    "pos_slice": [0, 1],
                    "vel_slice": [1, 2],
                },
                param_schema_version=STRUCTURAL_LINEAR_STATE_SPACE_PARAM_SCHEMA_VERSION,
                input_ports=["force", "epsilon"],
                output_ports=["effector", "state"],
            )
        },
        input_ports=["force", "epsilon"],
        output_ports=["effector", "state"],
        input_bindings={
            "force": ("plant", "force"),
            "epsilon": ("plant", "epsilon"),
        },
        output_bindings={
            "effector": ("plant", "effector"),
            "state": ("plant", "state"),
        },
    )


def test_structural_transition_is_state_dependent_not_additive_force() -> None:
    transition = jnp.eye(2)
    input_matrix = jnp.asarray([[0.0], [1.0]])
    delta_A = jnp.asarray([[0.0, 2.0], [0.0, 0.0]])
    perturbation = StructuralLinearDynamicsPerturbation(delta_A)
    state_a = jnp.asarray([0.0, 1.0])
    state_b = jnp.asarray([0.0, 3.0])

    structural_a = structural_linear_transition(transition, state_a, perturbation)
    structural_b = structural_linear_transition(transition, state_b, perturbation)
    additive = input_matrix @ jnp.asarray([2.0])

    assert jnp.allclose(structural_a, jnp.asarray([2.0, 1.0]))
    assert jnp.allclose(structural_b, jnp.asarray([6.0, 3.0]))
    assert additive[0] == 0.0
    assert structural_a[0] != transition[0] @ state_a
    assert structural_b[0] - (transition[0] @ state_b) != structural_a[0]


@pytest.mark.parametrize(
    ("active", "scale", "expected"),
    [
        (False, 1.0, [0.5, 2.0]),
        (True, 1.0, [1.5, 2.0]),
        (True, -1.0, [-0.5, 2.0]),
    ],
)
def test_signed_and_nominal_structural_variants(
    active: bool,
    scale: float,
    expected: list[float],
) -> None:
    component = StructuralLinearStateSpace(
        A=jnp.asarray([[1.0, 0.25], [0.0, 1.0]]),
        B=jnp.asarray([[0.0], [1.0]]),
        delta_A=jnp.asarray([[0.0, 0.5], [0.0, 0.0]]),
        initial_state=jnp.asarray([0.0, 2.0]),
        pos_slice=(0, 1),
        vel_slice=(1, 2),
        active=active,
        scale=scale,
    )
    outputs, _ = component(
        {"force": jnp.zeros((1,)), "epsilon": jnp.zeros((0,))},
        init_state_from_component(component),
        key=jr.PRNGKey(0),
    )

    assert jnp.allclose(outputs["state"], jnp.asarray(expected))


def test_task_trial_selects_one_constant_structural_variant() -> None:
    graph = spec_to_graph(_spec())
    component = graph.nodes["plant"]
    trial = TaskTrialSpec(
        inits=WhereDict(),
        targets=WhereDict(),
        inputs={
            "force": jnp.zeros((3, 1)),
            "epsilon": jnp.zeros((3, 1)),
        },
        intervene={
            "structural_field": StructuralLinearDynamicsPerturbation(
                delta_A=jnp.asarray([[0.0, -0.5], [0.0, 0.0]]),
                active=True,
            )
        },
        timeline=TrialTimeline(n_steps=3),
    )

    prepared = prepare_trial(graph, trial)
    selected = prepared.init_state.get(component.structural_params_index)

    assert prepared.n_steps == 3
    assert jnp.array_equal(
        selected.delta_A,
        jnp.asarray([[0.0, -0.5], [0.0, 0.0]]),
    )
    assert bool(selected.active)
    assert "intervene:structural_field" not in prepared.inputs


def test_graphspec_round_trip_preserves_structural_identity() -> None:
    graph = spec_to_graph(_spec(active=True))
    round_tripped = graph_to_spec(graph)
    node = round_tripped.nodes["plant"]

    assert node.type == "StructuralLinearStateSpace"
    assert node.param_schema_version == STRUCTURAL_LINEAR_STATE_SPACE_PARAM_SCHEMA_VERSION
    assert node.params["delta_A"] == [[0.0, 0.5], [0.0, 0.0]]
    assert node.params["label"] == "structural_field"
    assert node.params["active"] is True
    assert spec_to_graph(
        GraphSpec.model_validate_json(round_tripped.model_dump_json())
    ).nodes["plant"].label == "structural_field"


def test_structural_component_rejects_unknown_parameter_schema_version() -> None:
    spec = _spec()
    node = spec.nodes["plant"].model_copy(
        update={"param_schema_version": "feedbax.component.structural_linear_state_space.v0"}
    )
    incompatible = spec.model_copy(update={"nodes": {"plant": node}})

    with pytest.raises(
        UnsupportedComponentMigration,
        match="No component migration registered",
    ):
        spec_to_graph(incompatible)
