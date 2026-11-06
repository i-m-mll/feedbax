"""Numerical and executable topology checks for the muscle force templates."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from feedbax.component_registry import ComponentRegistry
from feedbax.compiler.serialization import graph_to_spec
from feedbax.contracts.graph import ComponentSpec, GraphSpec
from feedbax.mechanics.geometry import PointMassRadialGeometry, TwoLinkArmMuscleGeometry
from feedbax.mechanics.muscles.relu_muscle import ReluMuscle
from feedbax.mechanics.muscles.thelen_muscle import RigidTendonHillMuscleThelen
from feedbax.mechanics.templates import Arm6MuscleRigidTendon, PointMass8MuscleRelu
from feedbax.runtime.graph import Graph, init_state_from_component
from tests.support.graph_compiler_test_support import spec_to_graph


@pytest.fixture(scope="module")
def registry():
    return ComponentRegistry()


@pytest.mark.parametrize("custom", [False, True])
@pytest.mark.parametrize("arm", [False, True])
def test_native_muscles_match_independent_actuator_steps(registry, custom, arm):
    key = jax.random.key(12)
    params = {"dt": 0.0006, "max_isometric_force": 167.0} if custom else {}
    if arm:
        geometry = TwoLinkArmMuscleGeometry.default_six_muscle()
        if custom:
            geometry = TwoLinkArmMuscleGeometry.from_moment_arm_matrix(
                [[0.04, 0.01], [-0.05, 0.01], [-0.02, -0.04]],
                [0.23, 0.22, 0.25],
            )
            params.update(optimal_muscle_length=0.11, tendon_slack_length=0.0)
        graph = Arm6MuscleRigidTendon(**params, geometry=geometry)
        reference = RigidTendonHillMuscleThelen(**params)
        output = "torques"
    else:
        if custom:
            params.update(tau_activation=0.024, tau_deactivation=0.039)
        n_pairs = 3 if custom else 4
        geometry = PointMassRadialGeometry(n_pairs)
        graph = PointMass8MuscleRelu(**params, n_pairs=n_pairs)
        reference = ReluMuscle(**params)
        output = "force_2d"

    assert isinstance(graph, Graph)
    spec = GraphSpec.model_validate_json(graph_to_spec(graph, registry).model_dump_json())
    graph = spec_to_graph(spec, registry)
    graph_state = init_state_from_component(graph)
    count = geometry.n_muscles
    reference_states = [init_state_from_component(reference) for _ in range(count)]
    for step, scale in enumerate((0.95, 0.5, 0.0, 0.7, 0.2)):
        inputs = {"excitation": scale * jnp.linspace(0.2, 1.0, count)}
        if arm:
            angles = jnp.array([0.2 + 0.03 * step, -0.1 + 0.02 * step])
            velocity = jnp.array([0.3 - 0.06 * step, 0.2])
            lengths = geometry.musculotendon_lengths(angles)
            velocities = geometry.musculotendon_velocities(angles, velocity)
            inputs.update(angles=angles, angular_velocities=velocity)
        forces = []
        activations = []
        for index, state in enumerate(reference_states):
            signals = {"excitation": inputs["excitation"][index]}
            if arm:
                signals.update(
                    musculotendon_length=lengths[index],
                    musculotendon_velocity=velocities[index],
                )
            result, reference_states[index] = reference(signals, state, key=key)
            forces.append(result["force"])
            activations.append(result["activation"])
        forces = jnp.stack(forces)
        expected_force = (
            geometry.forces_to_torques(angles, forces)
            if arm
            else geometry.forces_to_force_2d(forces)
        )

        actual, graph_state = graph(inputs, graph_state, key=key)
        assert tuple(actual) == (output, "forces", "activations")
        np.testing.assert_allclose(actual[output], expected_force, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(actual["forces"], forces, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(actual["activations"], activations, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    ("name", "count", "force_port"),
    [("Arm6MuscleRigidTendon", 6, "torques"), ("PointMass8MuscleRelu", 8, "force_2d")],
)
def test_registry_muscle_template_executes_rewired_interior(registry, name, count, force_port):
    meta = registry.get(name)
    template = meta.template_graph
    assert meta.template_kind == "executable"
    spec = GraphSpec(
        nodes={
            "generator": ComponentSpec(
                type=name,
                input_ports=template.input_ports,
                output_ports=template.output_ports,
            )
        },
        input_ports=template.input_ports,
        output_ports=template.output_ports,
        input_bindings={port: ("generator", port) for port in template.input_ports},
        output_bindings={port: ("generator", port) for port in template.output_ports},
        subgraphs={"generator": template},
    )
    original = spec_to_graph(spec, registry)
    edited = spec.model_copy(deep=True)
    for wire in edited.subgraphs["generator"].wires:
        if wire.source_node == "excitation" and wire.source_port in {"out_0", "out_1"}:
            wire.source_port = "out_1" if wire.source_port == "out_0" else "out_0"
    rewired = spec_to_graph(edited, registry)
    restored_spec = GraphSpec.model_validate_json(
        graph_to_spec(rewired, registry).model_dump_json()
    )
    restored = spec_to_graph(restored_spec, registry)

    key = jax.random.key(12)
    inputs = {"excitation": jnp.zeros(count).at[0].set(0.8)}
    before, _ = original(inputs, init_state_from_component(original), key=key)
    after, _ = rewired(inputs, init_state_from_component(rewired), key=key)
    roundtripped, _ = restored(inputs, init_state_from_component(restored), key=key)

    assert before["activations"][0] > before["activations"][1]
    assert after["activations"][0] < after["activations"][1]
    assert not np.allclose(before[force_port], after[force_port])
    assert restored_spec.nodes["generator"].params == {}
    for port in original.output_ports:
        np.testing.assert_array_equal(after[port], roundtripped[port])
