"""Tests for declarative state-threshold-latched additive interventions."""

from __future__ import annotations

from typing import Any

import equinox as eqx
from equinox.nn import State
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.graph import ComponentSpec, GraphSpec, StudioTaskBindingSpec
from feedbax.contracts.graphs.serialization import graph_to_spec, spec_to_graph
from feedbax.contracts.migrations import UnsupportedComponentMigration
from feedbax.intervene import (
    PlanarTargetRelativeSelector,
    StateSelector,
    THRESHOLD_LATCHED_FORCE_SCHEMA_VERSION,
    THRESHOLD_LATCHED_FORCE_SCHEMA_VERSION_V1,
    ThresholdLatchedForce,
    ThresholdLatchedForceParams,
)
from feedbax.runtime.graph import init_state_from_component
from feedbax.runtime.state import CartesianState
from feedbax.runtime.task_bindings import (
    apply_task_parameter_state_inits,
    expose_task_bindings,
)


SCHEMA_VERSION = THRESHOLD_LATCHED_FORCE_SCHEMA_VERSION


def _component(
    *,
    direction: str = "increasing",
    threshold: float = 0.0,
    ramp_duration: float = 0.0,
    scale: float = 1.0,
    active: bool = True,
) -> ThresholdLatchedForce:
    return ThresholdLatchedForce(
        state_selector=StateSelector(("pos", 0)),
        direction=direction,
        dt=0.1,
        params=ThresholdLatchedForceParams(
            threshold=threshold,
            force=jnp.array([2.0, -1.0]),
            ramp_duration=ramp_duration,
            scale=scale,
            active=active,
        ),
        label="step_load",
    )


def _run(
    component: ThresholdLatchedForce,
    positions: jax.Array,
    *,
    params_override: ThresholdLatchedForceParams | None = None,
) -> jax.Array:
    def step(state: State, position: jax.Array) -> tuple[State, jax.Array]:
        inputs: dict[str, Any] = {
            "state": CartesianState(
                pos=jnp.array([position, 0.0]),
                vel=jnp.zeros((2,)),
            ),
            "force": jnp.array([0.5, 0.25]),
        }
        if params_override is not None:
            inputs["params_override"] = params_override
        outputs, state = component(inputs, state, key=jr.PRNGKey(0))
        return state, outputs["force"]

    _, forces = jax.lax.scan(step, State(component), positions)
    return forces


def _graph_spec(*, param_schema_version: str = SCHEMA_VERSION) -> GraphSpec:
    input_ports = ["state", "target", "force", "params_override"]
    return GraphSpec(
        nodes={
            "load": ComponentSpec(
                type="ThresholdLatchedForce",
                params={
                    "state_selector": {"kind": "fixed", "path": ["pos", 0]},
                    "direction": "increasing",
                    "threshold": 0.0,
                    "force": [2.0, -1.0],
                    "lateral_force": 0.0,
                    "ramp_duration": 0.2,
                    "scale": 1.0,
                    "active": True,
                    "dt": 0.1,
                    "label": "step_load",
                },
                param_schema_version=param_schema_version,
                input_ports=input_ports,
                output_ports=["force"],
            )
        },
        input_ports=input_ports,
        output_ports=["force"],
        input_bindings={port: ("load", port) for port in input_ports},
        output_bindings={"force": ("load", "force")},
    )


def _target_relative_component(
    *,
    threshold: float = 0.5,
    lateral_force: float = 2.0,
    scale: float = 1.0,
    active: bool = True,
) -> ThresholdLatchedForce:
    return ThresholdLatchedForce(
        state_selector=PlanarTargetRelativeSelector(
            position_path=("pos",),
            target_path=("goal",),
        ),
        direction="increasing",
        dt=0.1,
        params=ThresholdLatchedForceParams(
            threshold=threshold,
            lateral_force=lateral_force,
            ramp_duration=0.0,
            scale=scale,
            active=active,
        ),
        label="lateral_step",
    )


def _run_target_relative(
    component: ThresholdLatchedForce,
    positions: jax.Array,
    target: jax.Array,
    *,
    params_override: ThresholdLatchedForceParams | None = None,
) -> jax.Array:
    def step(state: State, position: jax.Array) -> tuple[State, jax.Array]:
        inputs: dict[str, Any] = {
            "state": CartesianState(pos=position, vel=jnp.zeros((2,))),
            "target": {"goal": target},
            "force": jnp.zeros((2,)),
        }
        if params_override is not None:
            inputs["params_override"] = params_override
        outputs, state = component(inputs, state, key=jr.PRNGKey(0))
        return state, outputs["force"]

    _, forces = jax.lax.scan(step, State(component), positions)
    return forces


def _target_relative_graph_spec() -> GraphSpec:
    input_ports = ["state", "target", "force", "params_override"]
    return GraphSpec(
        nodes={
            "load": ComponentSpec(
                type="ThresholdLatchedForce",
                params={
                    "state_selector": {
                        "kind": "planar_target_relative",
                        "position_path": ["pos"],
                        "target_path": ["goal"],
                    },
                    "direction": "increasing",
                    "threshold": 0.5,
                    "force": [0.0, 0.0],
                    "lateral_force": 2.0,
                    "ramp_duration": 0.0,
                    "scale": 1.0,
                    "active": True,
                    "dt": 0.1,
                    "label": "lateral_step",
                },
                param_schema_version=SCHEMA_VERSION,
                input_ports=input_ports,
                output_ports=["force"],
            )
        },
        input_ports=input_ports,
        output_ports=["force"],
        input_bindings={port: ("load", port) for port in input_ports},
        output_bindings={"force": ("load", "force")},
    )


def test_increasing_crossing_is_exact_and_latches_for_trial_remainder() -> None:
    forces = _run(_component(), jnp.array([-0.5, 0.0, 0.4, -0.2]))

    assert jnp.allclose(
        forces,
        jnp.array(
            [
                [0.5, 0.25],
                [2.5, -0.75],
                [2.5, -0.75],
                [2.5, -0.75],
            ]
        ),
    )


def test_decreasing_crossing_and_linear_ramp_complete() -> None:
    forces = _run(
        _component(direction="decreasing", threshold=0.0, ramp_duration=0.2),
        jnp.array([0.5, 0.0, -0.1, -0.2]),
    )

    assert jnp.allclose(
        forces,
        jnp.array(
            [
                [0.5, 0.25],
                [0.5, 0.25],
                [1.5, -0.25],
                [2.5, -0.75],
            ]
        ),
    )


def test_signed_and_nominal_trial_variants_are_batched_and_jitted() -> None:
    component = _component()
    positions = jnp.array([-0.5, 0.0, 0.2])

    def run_variant(scale: jax.Array, active: jax.Array) -> jax.Array:
        override = ThresholdLatchedForceParams(
            threshold=0.0,
            force=jnp.array([2.0, -1.0]),
            ramp_duration=0.0,
            scale=scale,
            active=active,
        )
        return _run(component, positions, params_override=override)

    batched = eqx.filter_jit(eqx.filter_vmap(run_variant))(
        jnp.array([1.0, -1.0, 1.0]),
        jnp.array([True, True, False]),
    )

    assert jnp.allclose(batched[0, -1], jnp.array([2.5, -0.75]))
    assert jnp.allclose(batched[1, -1], jnp.array([-1.5, 1.25]))
    assert jnp.allclose(batched[2], jnp.broadcast_to(jnp.array([0.5, 0.25]), (3, 2)))


def test_target_relative_selector_rotates_lateral_force_with_reach_direction() -> None:
    component = _target_relative_component()
    fractions = jnp.array([0.0, 0.5, 0.75])

    x_target = jnp.array([1.0, 0.0])
    y_start = jnp.array([1.0, -2.0])
    y_target = jnp.array([1.0, -1.0])
    x_forces = _run_target_relative(component, fractions[:, None] * x_target, x_target)
    y_forces = _run_target_relative(
        component,
        y_start + fractions[:, None] * (y_target - y_start),
        y_target,
    )

    assert jnp.allclose(x_forces, jnp.array([[0.0, 0.0], [0.0, 2.0], [0.0, 2.0]]))
    assert jnp.allclose(y_forces, jnp.array([[0.0, 0.0], [-2.0, 0.0], [-2.0, 0.0]]))


def test_target_relative_variants_are_batched_and_jitted() -> None:
    component = _target_relative_component()
    targets = jnp.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    scales = jnp.array([1.0, -1.0, 0.0])
    fractions = jnp.array([0.0, 0.5, 0.75])

    def run_variant(target: jax.Array, scale: jax.Array) -> jax.Array:
        override = ThresholdLatchedForceParams(
            threshold=0.5,
            lateral_force=2.0,
            ramp_duration=0.0,
            scale=scale,
            active=scale != 0,
        )
        return _run_target_relative(
            component,
            fractions[:, None] * target,
            target,
            params_override=override,
        )

    batched = eqx.filter_jit(eqx.filter_vmap(run_variant))(targets, scales)

    assert jnp.allclose(batched[0, -1], jnp.array([0.0, 2.0]))
    assert jnp.allclose(batched[1, -1], jnp.array([2.0, 0.0]))
    assert jnp.allclose(batched[2], jnp.zeros((3, 2)))


def test_graphspec_round_trip_preserves_trigger_latch_ramp_and_selector_identity() -> None:
    graph = spec_to_graph(_graph_spec(), ComponentRegistry(load_user_components=False))
    component = graph.nodes["load"]

    assert isinstance(component, ThresholdLatchedForce)
    assert component.state_selector == StateSelector(("pos", 0))
    assert component.direction == "increasing"
    assert component.dt == pytest.approx(0.1)
    assert set(component.task_parameter_state_indices()) == {"step_load"}

    roundtrip = graph_to_spec(graph)
    node = roundtrip.nodes["load"]
    assert node.type == "ThresholdLatchedForce"
    assert node.param_schema_version == SCHEMA_VERSION
    assert node.params == {
        "state_selector": {"kind": "fixed", "path": ["pos", 0]},
        "direction": "increasing",
        "threshold": 0.0,
        "force": [2.0, -1.0],
        "lateral_force": 0.0,
        "ramp_duration": pytest.approx(0.2),
        "scale": 1.0,
        "active": True,
        "dt": pytest.approx(0.1),
        "label": "step_load",
    }


def test_target_relative_graphspec_round_trip_preserves_frame_selectors() -> None:
    graph = spec_to_graph(
        _target_relative_graph_spec(), ComponentRegistry(load_user_components=False)
    )
    component = graph.nodes["load"]

    assert component.state_selector == PlanarTargetRelativeSelector(
        position_path=("pos",),
        target_path=("goal",),
    )
    node = graph_to_spec(graph).nodes["load"]
    assert node.param_schema_version == SCHEMA_VERSION
    assert node.params["state_selector"] == {
        "kind": "planar_target_relative",
        "position_path": ["pos"],
        "target_path": ["goal"],
    }
    assert node.params["lateral_force"] == pytest.approx(2.0)


def test_native_graph_materialization_runs_target_relative_frame() -> None:
    graph = spec_to_graph(
        _target_relative_graph_spec(), ComponentRegistry(load_user_components=False)
    )
    target = jnp.array([0.0, 1.0])
    positions = jnp.array([[0.0, 0.0], [0.0, 0.5], [0.0, 0.75]])

    def step(state: State, position: jax.Array) -> tuple[State, jax.Array]:
        outputs, state = graph(
            {
                "state": CartesianState(pos=position, vel=jnp.zeros((2,))),
                "target": {"goal": target},
                "force": jnp.zeros((2,)),
            },
            state,
            key=jr.PRNGKey(0),
        )
        return state, outputs["force"]

    _, forces = eqx.filter_jit(lambda: jax.lax.scan(step, State(graph), positions))()

    assert jnp.allclose(forces, jnp.array([[0.0, 0.0], [-2.0, 0.0], [-2.0, 0.0]]))


def test_native_graph_materialization_runs_runtime_state_trigger() -> None:
    graph = spec_to_graph(_graph_spec(), ComponentRegistry(load_user_components=False))
    component = graph.nodes["load"]
    forces = eqx.filter_jit(_run)(component, jnp.array([-0.2, 0.0, 0.2, 0.4]))

    assert jnp.allclose(
        forces,
        jnp.array(
            [
                [0.5, 0.25],
                [0.5, 0.25],
                [1.5, -0.25],
                [2.5, -0.75],
            ]
        ),
    )


def test_task_authored_trial_variant_initializes_native_component_state() -> None:
    graph = spec_to_graph(_graph_spec(), ComponentRegistry(load_user_components=False))
    binding_spec = StudioTaskBindingSpec.model_validate(
        {
            "exposed_data": [
                {
                    "id": "step_variant",
                    "label": "Step variant",
                    "kind": "intervention",
                    "role": "component_parameter",
                    "path": "intervene.step_load",
                    "bindable": True,
                    "metadata": {
                        "temporality": "per_trial",
                        "task_parameter_label": "step_load",
                    },
                }
            ],
            "bindings": [
                {
                    "id": "task:step_variant->load:params_override",
                    "source_data_id": "step_variant",
                    "target_node_id": "load",
                    "target_port": "params_override",
                    "role": "component_parameter",
                    "metadata": {"task_parameter_label": "step_load"},
                }
            ],
        }
    )
    exposure = expose_task_bindings(graph, binding_spec)
    state = apply_task_parameter_state_inits(
        init_state_from_component(exposure.graph),
        exposure.graph,
        exposure.state_init_plans,
        {
            "step_variant": ThresholdLatchedForceParams(
                threshold=0.0,
                force=jnp.array([2.0, -1.0]),
                lateral_force=0.0,
                ramp_duration=0.0,
                scale=-1.0,
                active=True,
            )
        },
    )
    component = exposure.graph.nodes["load"]
    initialized = state.get(component.params_index)

    assert initialized.active
    assert initialized.scale == pytest.approx(-1.0)
    assert jnp.allclose(initialized.force, jnp.array([2.0, -1.0]))


def test_v1_fixed_selector_migrates_to_v2() -> None:
    spec = _graph_spec(param_schema_version=THRESHOLD_LATCHED_FORCE_SCHEMA_VERSION_V1)
    node = spec.nodes["load"]
    node.params.pop("lateral_force")
    node.params["state_selector"] = {"path": ["pos", 0]}
    node.input_ports.remove("target")
    spec.input_ports.remove("target")
    del spec.input_bindings["target"]

    graph = spec_to_graph(spec, ComponentRegistry(load_user_components=False))

    assert graph.nodes["load"].state_selector == StateSelector(("pos", 0))
    migrated = graph_to_spec(graph).nodes["load"]
    assert migrated.param_schema_version == SCHEMA_VERSION
    assert migrated.params["state_selector"] == {"kind": "fixed", "path": ["pos", 0]}
    assert migrated.params["lateral_force"] == pytest.approx(0.0)


def test_unknown_parameter_schema_version_is_rejected_without_a_migration() -> None:
    registry = ComponentRegistry(load_user_components=False)

    with pytest.raises(UnsupportedComponentMigration, match="No component migration registered"):
        spec_to_graph(
            _graph_spec(param_schema_version="feedbax.component.threshold_latched_force.v0"),
            component_registry=registry,
        )
