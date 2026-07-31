from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from feedbax.component_registry import ComponentRegistry
from feedbax.config.mapping import WhereDict
from feedbax.contracts.graph import (
    GRAPH_SPEC_SCHEMA_ID,
    GRAPH_SPEC_SCHEMA_VERSION_V4,
    ComponentSpec,
    GraphSpec,
)
from feedbax.contracts.array_values import (
    ARRAY_VALUE_SCHEMA_ID,
    ARRAY_VALUE_SCHEMA_VERSION,
)
from feedbax.contracts.graphs.serialization import graph_to_spec, spec_to_graph
from feedbax.contracts.migrations import UnsupportedComponentMigration, migrate_graph_spec
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


def _sparse_spec() -> GraphSpec:
    return GraphSpec(
        nodes={
            "plant": ComponentSpec(
                type="StructuralLinearStateSpace",
                params={
                    "A": jnp.eye(4).tolist(),
                    "B": [[0.0], [0.0], [0.0], [0.0]],
                    "B_w": [[0.0], [0.0], [0.0], [0.0]],
                    "delta_A": {
                        "schema_id": ARRAY_VALUE_SCHEMA_ID,
                        "schema_version": ARRAY_VALUE_SCHEMA_VERSION,
                        "encoding": "sparse_coo",
                        "shape": [4, 4],
                        "dtype": "float32",
                        "nonfinite": "forbid",
                        "fill": 0.0,
                        "entries": [{"coordinate": [3, 2], "value": -0.25}],
                    },
                    "scale": 2.0,
                    "active": True,
                    "label": "structural_field",
                    "dt": 0.1,
                    "initial_state": [0.0, 0.0, 1.0, 0.0],
                    "pos_slice": [0, 2],
                    "vel_slice": [2, 4],
                },
                param_schema_version=STRUCTURAL_LINEAR_STATE_SPACE_PARAM_SCHEMA_VERSION,
                input_ports=["force", "epsilon"],
                output_ports=["effector", "state"],
            )
        },
        input_ports=["force", "epsilon"],
        output_ports=["effector", "state"],
        input_bindings={"force": ("plant", "force"), "epsilon": ("plant", "epsilon")},
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


def test_sparse_entries_construct_the_same_jittable_dense_transition() -> None:
    perturbation = StructuralLinearDynamicsPerturbation.from_entries(
        (4, 4),
        [(3, 2, -0.25)],
        scale=2.0,
    )
    transition = jnp.eye(4)
    state = jnp.asarray([0.0, 0.0, 1.0, 0.0])

    result = jax.jit(structural_linear_transition)(transition, state, perturbation)

    assert perturbation.delta_A.shape == (4, 4)
    assert perturbation.delta_A[3, 2] == pytest.approx(-0.25)
    assert jnp.allclose(result, jnp.asarray([0.0, 0.0, 1.0, -0.5]))


def test_structural_registry_advertises_sparse_delta_a_authoring() -> None:
    meta = ComponentRegistry(
        load_user_components=False,
        discover_plugins=False,
    ).get("StructuralLinearStateSpace")
    assert meta is not None
    delta_A_schema = next(
        schema for schema in meta.param_schema if schema.name == "delta_A"
    )

    assert delta_A_schema.type == "object"
    assert delta_A_schema.default == {
        "schema_id": ARRAY_VALUE_SCHEMA_ID,
        "schema_version": ARRAY_VALUE_SCHEMA_VERSION,
        "shape": [4, 4],
        "dtype": "float32",
        "nonfinite": "forbid",
        "encoding": "sparse_coo",
        "fill": 0.0,
        "entries": [],
    }


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


def test_sparse_graphspec_round_trip_preserves_canonical_entry_identity() -> None:
    graph = spec_to_graph(_sparse_spec())
    component = graph.nodes["plant"]
    outputs, _ = component(
        {"force": jnp.zeros((1,)), "epsilon": jnp.zeros((1,))},
        init_state_from_component(component),
        key=jr.PRNGKey(0),
    )

    round_tripped = graph_to_spec(graph)
    node = round_tripped.nodes["plant"]

    assert jnp.allclose(outputs["state"], jnp.asarray([0.0, 0.0, 1.0, -0.5]))
    assert node.params["delta_A"] == {
        "schema_id": ARRAY_VALUE_SCHEMA_ID,
        "schema_version": ARRAY_VALUE_SCHEMA_VERSION,
        "encoding": "sparse_coo",
        "shape": [4, 4],
        "dtype": "float32",
        "nonfinite": "forbid",
        "fill": 0.0,
        "entries": [{"coordinate": [3, 2], "value": -0.25}],
    }
    assert (
        graph_to_spec(
            spec_to_graph(
                GraphSpec.model_validate_json(round_tripped.model_dump_json())
            )
        )
        .nodes["plant"]
        .params["delta_A"]
        == node.params["delta_A"]
    )


def test_nested_v4_sparse_migration_materializes_losslessly_and_preserves_envelope() -> None:
    high_precision = 1.0000000000000002
    legacy = {
        "schema_id": GRAPH_SPEC_SCHEMA_ID,
        "schema_version": GRAPH_SPEC_SCHEMA_VERSION_V4,
        "nodes": {
            "wrapper": {
                "type": "Subgraph",
                "params": {},
                "input_ports": [],
                "output_ports": [],
            }
        },
        "wires": [],
        "subgraphs": {
            "wrapper": {
                "schema_id": GRAPH_SPEC_SCHEMA_ID,
                "schema_version": GRAPH_SPEC_SCHEMA_VERSION_V4,
                "nodes": {
                    "plant": {
                        "type": "StructuralLinearStateSpace",
                        "params": {
                            "A": [[1.0, 0.0], [0.0, 1.0]],
                            "B": [[0.0], [0.0]],
                            "B_w": [[0.0], [0.0]],
                            "delta_A": {
                                "shape": [2, 2],
                                "entries": [{"row": 0, "column": 1, "value": high_precision}],
                            },
                            "initial_state": [0.0, 0.0],
                            "pos_slice": [0, 1],
                            "vel_slice": [1, 2],
                        },
                        "param_schema_version": (
                            STRUCTURAL_LINEAR_STATE_SPACE_PARAM_SCHEMA_VERSION
                        ),
                        "input_ports": ["force", "epsilon"],
                        "output_ports": ["effector", "state"],
                    }
                },
                "wires": [],
            }
        },
    }

    migration = migrate_graph_spec(legacy)
    nested_delta = migration.payload["subgraphs"]["wrapper"]["nodes"]["plant"]["params"]["delta_A"]
    assert nested_delta["dtype"] == "float64"
    with jax.experimental.enable_x64():
        graph = spec_to_graph(GraphSpec.model_validate(migration.payload))
        component = graph.nodes["wrapper"].nodes["plant"]

    assert component.initial_delta_A[0][1] == high_precision
    round_tripped = graph_to_spec(graph)
    round_trip_delta = round_tripped.subgraphs["wrapper"].nodes["plant"].params["delta_A"]
    assert round_trip_delta == nested_delta


@pytest.mark.parametrize(
    ("delta_A", "match"),
    [
        (
            {
                "schema_id": ARRAY_VALUE_SCHEMA_ID,
                "schema_version": ARRAY_VALUE_SCHEMA_VERSION,
                "encoding": "sparse_coo",
                "shape": [4, 4],
                "dtype": "float32",
                "nonfinite": "forbid",
                "fill": 0.0,
                "entries": [
                    {"coordinate": [3, 2], "value": -0.25},
                    {"coordinate": [3, 2], "value": 0.5},
                ],
            },
            "duplicated",
        ),
        (
            {
                "schema_id": ARRAY_VALUE_SCHEMA_ID,
                "schema_version": ARRAY_VALUE_SCHEMA_VERSION,
                "encoding": "sparse_coo",
                "shape": [4, 4],
                "dtype": "float32",
                "nonfinite": "forbid",
                "fill": 0.0,
                "entries": [{"coordinate": [4, 2], "value": -0.25}],
            },
            "outside shape",
        ),
        (
            {
                "schema_id": ARRAY_VALUE_SCHEMA_ID,
                "schema_version": ARRAY_VALUE_SCHEMA_VERSION,
                "encoding": "sparse_coo",
                "shape": [4, 3],
                "dtype": "float32",
                "nonfinite": "forbid",
                "fill": 0.0,
                "entries": [],
            },
            "square matrix",
        ),
    ],
)
def test_sparse_graphspec_rejects_invalid_entries(
    delta_A: dict[str, object],
    match: str,
) -> None:
    spec = _sparse_spec()
    node = spec.nodes["plant"]
    invalid = spec.model_copy(
        update={
            "nodes": {
                "plant": node.model_copy(
                    update={"params": {**node.params, "delta_A": delta_A}}
                )
            }
        }
    )

    with pytest.raises(ValueError, match=match):
        spec_to_graph(invalid)


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
