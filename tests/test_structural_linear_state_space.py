from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from feedbax.component_registry import ComponentRegistry
from feedbax.compiler import GraphCompilationError
from feedbax.runtime.mapping import WhereDict
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
from feedbax.compiler.serialization import graph_to_spec
from tests.support.graph_compiler_test_support import spec_to_graph
from feedbax.contracts.migrations import UnsupportedComponentMigration, migrate_graph_spec
from feedbax.mechanics import (
    StructuralLinearDynamicsPerturbation,
    StructuralLinearStateSpace,
    structural_linear_transition,
)
from feedbax.component_migrations.mechanics import (
    STRUCTURAL_LINEAR_STATE_SPACE_PARAM_SCHEMA_VERSION,
)
from feedbax.runtime.graph import init_state_from_component
from feedbax.tasks import TaskTrialSpec, TrialTimeline, prepare_trial


def _spec(*, active: bool = False) -> GraphSpec:
    return GraphSpec(
        nodes={
            "plant": ComponentSpec(
                type="feedbax.StructuralLinearStateSpace",
                params={
                    "A": [[1.0, 0.25], [0.0, 1.0]],
                    "B": [[0.0], [1.0]],
                    "B_w": [[0.0], [1.0]],
                    "delta_A": [[0.0, 0.5], [0.0, 0.0]],
                    "scale": 1.0,
                    "active": active,
                    "dt": 0.1,
                    "initial_state": [0.0, 2.0],
                    "pos_slice": [0, 1],
                    "vel_slice": [1, 2],
                },
                revision=3,
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
                type="feedbax.StructuralLinearStateSpace",
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
                    "dt": 0.1,
                    "initial_state": [0.0, 0.0, 1.0, 0.0],
                    "pos_slice": [0, 2],
                    "vel_slice": [2, 4],
                },
                revision=3,
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


def test_sparse_transition_uses_nominal_precision_under_jit() -> None:
    # The product has a bit beyond float32 precision as well as A itself.
    nominal = 1.0 + 2**-30
    delta = 0.25 + 2**-25
    scale = -(1.0 + 2**-23)
    effective = nominal - 0.25 - 2**-24 - 2**-48

    @jax.jit
    def matrix(transition, perturbation):
        return perturbation.effective_transition(transition)

    def step(transition, state, perturbation):
        return structural_linear_transition(transition, state, perturbation)[0]

    value_and_grad = jax.jit(jax.value_and_grad(step, argnums=(1, 2), allow_int=True))
    with jax.experimental.enable_x64():
        transition = jnp.asarray([[nominal]], dtype=jnp.float64)
        state = jnp.asarray([2.0], dtype=jnp.float64)
        for active in (False, True):
            perturbation = StructuralLinearDynamicsPerturbation.from_entries(
                (1, 1), [(0, 0, delta)], scale=scale, active=active
            )
            expected = effective if active else nominal
            applied = matrix(transition, perturbation)
            result, (state_grad, params_grad) = value_and_grad(transition, state, perturbation)

            assert applied.dtype == result.dtype == state_grad.dtype == jnp.float64
            assert applied.item() == state_grad.item() == expected
            assert result.item() == 2.0 * expected
            assert params_grad.delta_A.dtype == params_grad.scale.dtype == jnp.float32
            assert params_grad.delta_A.item() == (2.0 * scale if active else 0.0)
            assert params_grad.scale.item() == (2.0 * delta if active else 0.0)
            if not active:
                negative_zero = jnp.asarray([[-0.0]], dtype=jnp.float64)
                assert jnp.signbit(matrix(negative_zero, perturbation)).item()

        disabled_nonfinite = StructuralLinearDynamicsPerturbation(
            jnp.full((1, 1), jnp.nan, dtype=jnp.float32), scale=jnp.nan, active=False
        )
        result, (state_grad, params_grad) = value_and_grad(transition, state, disabled_nonfinite)
        assert result.item() == 2.0 * nominal
        assert state_grad.item() == nominal
        assert params_grad.delta_A.item() == params_grad.scale.item() == 0.0


@pytest.mark.parametrize("dtype", [jnp.int32, jnp.bool_, jnp.complex64])
def test_structural_transition_requires_floating_point_nominal_matrix(dtype) -> None:
    perturbation = StructuralLinearDynamicsPerturbation.from_entries((1, 1), [(0, 0, 0.25)])

    with pytest.raises(TypeError, match="floating-point dtype"):
        perturbation.effective_transition(jnp.ones((1, 1), dtype=dtype))


def test_structural_registry_advertises_sparse_delta_a_authoring() -> None:
    meta = ComponentRegistry().get("feedbax.StructuralLinearStateSpace")
    assert meta is not None
    delta_A_schema = next(schema for schema in meta.param_schema if schema.name == "delta_A")

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
        (False, 1.0, 1.5),
        (True, 1.0, 2.5),
        (True, -1.0, 0.5),
    ],
)
def test_signed_and_nominal_structural_variants(
    active: bool,
    scale: float,
    expected: float,
) -> None:
    with jax.experimental.enable_x64():
        tail = 2**-30
        component = StructuralLinearStateSpace(
            A=jnp.asarray([[1.0 + tail, 0.25], [0.0, 1.0]]),
            B=jnp.asarray([[0.0], [1.0]]),
            delta_A=jnp.zeros((2, 2)),
            initial_state=jnp.asarray([1.0, 2.0]),
            pos_slice=(0, 1),
            vel_slice=(1, 2),
        )
        trial = TaskTrialSpec(
            inits=WhereDict(
                {
                    "structural_params_index": StructuralLinearDynamicsPerturbation.from_entries(
                        (2, 2), [(0, 1, 0.5)], scale=scale, active=active
                    )
                }
            ),
            targets=WhereDict(),
            inputs={"force": jnp.zeros((1, 1)), "epsilon": jnp.zeros((1, 0))},
        )
        prepared = prepare_trial(component, trial)
        outputs, _ = eqx.filter_jit(component)(
            {port: value[0] for port, value in prepared.inputs.items()},
            prepared.init_state,
            key=jr.PRNGKey(0),
        )

        assert outputs["state"].dtype == jnp.float64
        assert outputs["state"].tolist() == [expected + tail, 2.0]


def test_task_trial_selects_one_constant_structural_variant() -> None:
    graph = spec_to_graph(_spec(), ComponentRegistry())
    component = graph.nodes["plant"]
    trial = TaskTrialSpec(
        inits=WhereDict(
            {
                "plant.structural_params_index": StructuralLinearDynamicsPerturbation(
                    delta_A=jnp.asarray([[0.0, -0.5], [0.0, 0.0]]),
                    active=True,
                )
            }
        ),
        targets=WhereDict(),
        inputs={
            "force": jnp.zeros((3, 1)),
            "epsilon": jnp.zeros((3, 1)),
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


def test_graphspec_round_trip_preserves_structural_identity() -> None:
    registry = ComponentRegistry()
    graph = spec_to_graph(_spec(active=True), registry)
    graph = eqx.tree_at(lambda model: model.nodes["plant"].state_index.init.vector, graph, jnp.array([3.0, 4.0]))
    round_tripped = graph_to_spec(graph)
    node = round_tripped.nodes["plant"]

    assert node.type == "feedbax.StructuralLinearStateSpace"
    assert node.revision == 3
    assert node.params["delta_A"] == [[0.0, 0.5], [0.0, 0.0]]
    assert node.params["active"] is True
    assert node.params["initial_state"] == [3.0, 4.0]


def test_sparse_graphspec_round_trip_preserves_canonical_entry_identity() -> None:
    registry = ComponentRegistry()
    graph = spec_to_graph(_sparse_spec(), registry)
    component = graph.nodes["plant"]
    assert isinstance(component, StructuralLinearStateSpace)
    edited_delta = jnp.zeros((4, 4)).at[3, 2].set(0.75)
    trial_state = init_state_from_component(component).set(
        component.structural_params_index,
        StructuralLinearDynamicsPerturbation(edited_delta, active=True),
    )
    outputs, _ = component(
        {"force": jnp.zeros((1,)), "epsilon": jnp.zeros((1,))},
        trial_state,
        key=jr.PRNGKey(0),
    )

    round_tripped = graph_to_spec(graph)
    node = round_tripped.nodes["plant"]

    assert jnp.allclose(outputs["state"], jnp.asarray([0.0, 0.0, 1.0, 0.75]))
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
            spec_to_graph(GraphSpec.model_validate_json(round_tripped.model_dump_json()), registry)
        )
        .nodes["plant"]
        .params["delta_A"]
        == node.params["delta_A"]
    )

    edited_A = component.A.at[0, 0].set(2.0)
    edited_model = eqx.tree_at(lambda model: model.nodes["plant"].A, graph, edited_A)
    edited_model_spec = graph_to_spec(edited_model).nodes["plant"]
    assert edited_model_spec.params["A"] == edited_A.tolist()
    assert edited_model_spec.params["delta_A"] == node.params["delta_A"]

    edited = eqx.tree_at(
        lambda model: model.nodes["plant"].structural_params_index.init.delta_A,
        graph,
        edited_delta,
    )
    edited_spec = graph_to_spec(edited)
    assert edited_spec.nodes["plant"].params["delta_A"] == edited_delta.tolist()
    reloaded = spec_to_graph(edited_spec, registry).nodes["plant"]
    assert jnp.array_equal(
        init_state_from_component(reloaded).get(reloaded.structural_params_index).delta_A,
        edited_delta,
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
        graph = spec_to_graph(
            GraphSpec.model_validate(migration.payload),
            ComponentRegistry(),
        )
        component = graph.nodes["wrapper"].nodes["plant"]

    assert (
        float(
            init_state_from_component(component)
            .get(component.structural_params_index)
            .delta_A[0, 1]
        )
        == high_precision
    )
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
                "plant": node.model_copy(update={"params": {**node.params, "delta_A": delta_A}})
            }
        }
    )

    with pytest.raises(ValueError, match=match):
        spec_to_graph(invalid, ComponentRegistry())


def test_structural_component_rejects_unknown_parameter_schema_version() -> None:
    spec = _spec()
    node = spec.nodes["plant"].model_copy(update={"revision": 999})
    incompatible = spec.model_copy(update={"nodes": {"plant": node}})

    with pytest.raises(GraphCompilationError, match="No component migration registered") as exc:
        spec_to_graph(incompatible, ComponentRegistry())

    assert isinstance(exc.value.__cause__, UnsupportedComponentMigration)
