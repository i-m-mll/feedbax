"""Tests for standalone muscle models and effector templates.

Tests cover:
- ReluMuscle: activation dynamics, force computation, state management
- RigidTendonHillMuscleThelen: isometric force, FV curve, activation dynamics
- Arm6MuscleRigidTendon: multi-step execution, no NaN
- PointMass8MuscleRelu: multi-step execution, no NaN
- All: JIT compilation, gradient flow

:copyright: Copyright 2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from feedbax.compiler import GraphDocument, compile_graph
from feedbax.compiler.serialization import graph_to_spec
from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.graph import ComponentSpec, GraphSpec, WireSpec
from feedbax.runtime.graph import init_state_from_component
from feedbax.mechanics.muscles import ReluMuscle, RigidTendonHillMuscleThelen
from feedbax.mechanics.templates import Arm6MuscleRigidTendon, PointMass8MuscleRelu
from feedbax.mechanics.geometry import (
    PointMassRadialGeometry,
    TwoLinkArmMuscleGeometry,
    WrappingGeometry,
)

pytestmark = pytest.mark.usefixtures("enable_jax_x64")


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


@pytest.fixture
def relu_muscle():
    return ReluMuscle(max_isometric_force=500.0, dt=0.01)


@pytest.fixture
def thelen_muscle():
    return RigidTendonHillMuscleThelen(
        max_isometric_force=500.0,
        optimal_muscle_length=0.1,
        tendon_slack_length=0.1,
        dt=0.01,
    )


@pytest.fixture
def arm_template():
    return Arm6MuscleRigidTendon(dt=0.01)


@pytest.fixture
def pointmass_template():
    return PointMass8MuscleRelu(dt=0.01)


# ============================================================================
# ReluMuscle Tests
# ============================================================================


class TestReluMuscle:
    def test_activation_converges_to_excitation(self, relu_muscle, key):
        """Activation should converge toward constant excitation."""
        state = init_state_from_component(relu_muscle)
        excitation = jnp.array(0.8)

        # Run 200 steps
        for _ in range(200):
            inputs = {"excitation": excitation}
            outputs, state = relu_muscle(inputs, state, key=key)

        activation = outputs["activation"]
        assert jnp.allclose(activation, excitation, atol=0.01), (
            f"Expected activation ~{excitation}, got {activation}"
        )

    def test_force_equals_activation_times_fmax(self, relu_muscle, key):
        """Force should be activation * max_isometric_force."""
        state = init_state_from_component(relu_muscle)
        excitation = jnp.array(0.5)

        # Run enough steps for convergence
        for _ in range(200):
            inputs = {"excitation": excitation}
            outputs, state = relu_muscle(inputs, state, key=key)

        force = outputs["force"]
        activation = outputs["activation"]
        expected_force = activation * relu_muscle.max_isometric_force
        assert jnp.allclose(force, expected_force, atol=1e-5)

    def test_zero_excitation_gives_zero_force(self, relu_muscle, key):
        """Zero excitation should eventually give zero force."""
        state = init_state_from_component(relu_muscle)

        for _ in range(200):
            inputs = {"excitation": jnp.array(0.0)}
            outputs, state = relu_muscle(inputs, state, key=key)

        assert jnp.allclose(outputs["force"], 0.0, atol=1e-5)

    def test_full_excitation_gives_max_force(self, relu_muscle, key):
        """Full excitation should give max isometric force."""
        state = init_state_from_component(relu_muscle)

        for _ in range(500):
            inputs = {"excitation": jnp.array(1.0)}
            outputs, state = relu_muscle(inputs, state, key=key)

        expected = relu_muscle.max_isometric_force
        assert jnp.allclose(outputs["force"], expected, atol=5.0)

    def test_activation_dynamics_asymmetry(self, key):
        """Activation should rise faster than it decays."""
        muscle = ReluMuscle(
            max_isometric_force=500.0,
            tau_activation=0.01,
            tau_deactivation=0.1,
            dt=0.01,
        )
        state = init_state_from_component(muscle)

        # Rise: 50 steps at full excitation
        for _ in range(50):
            outputs, state = muscle({"excitation": jnp.array(1.0)}, state, key=key)
        rise_activation = outputs["activation"]

        # Reset
        state = init_state_from_component(muscle)
        # First get to full activation
        for _ in range(500):
            outputs, state = muscle({"excitation": jnp.array(1.0)}, state, key=key)

        # Decay: 50 steps at zero excitation
        for _ in range(50):
            outputs, state = muscle({"excitation": jnp.array(0.0)}, state, key=key)
        decay_activation = outputs["activation"]

        # After 50 steps: rise should be closer to 1.0 than decay is to 0.0
        rise_progress = rise_activation  # fraction of [0, 1]
        decay_progress = 1.0 - decay_activation  # fraction of [1, 0]
        assert rise_progress > decay_progress, (
            f"Rise progress {rise_progress} should exceed decay progress {decay_progress}"
        )


# ============================================================================
# RigidTendonHillMuscleThelen Tests
# ============================================================================


def test_thelen_graph_rollout_and_roundtrip_match_independent_muscles(key):
    inputs = {
        "excitation": jnp.array([[0.8, 0.1], [0.2, 0.9], [0.5, 0.3]]),
        "musculotendon_length": jnp.array([[0.22, 0.18], [0.19, 0.23], [0.21, 0.20]]),
        "musculotendon_velocity": jnp.array([[-0.15, 0.1], [0.12, -0.08], [0.0, 0.18]]),
    }
    n_steps = inputs["excitation"].shape[0]
    output_ports = ["force", "activation", "fiber_length", "fiber_velocity"]
    params = {"initial_activation": 0.35, "tendon_slack_length": 0.1, "dt": 0.01}
    spec = GraphSpec(
        nodes={
            "muscle": ComponentSpec(
                type="feedbax.RigidTendonHillMuscleThelen",
                params={**params, "n_muscles": 2},
                input_ports=list(inputs),
                output_ports=output_ports,
            ),
            "delay": ComponentSpec(
                type="feedbax.Channel",
                params={"delay": 1, "add_noise": False},
                input_ports=["input"],
                output_ports=["output"],
            ),
        },
        wires=[
            WireSpec(
                source_node="muscle",
                source_port="force",
                target_node="delay",
                target_port="input",
            )
        ],
        input_ports=list(inputs),
        input_bindings={port: ("muscle", port) for port in inputs},
        output_ports=[*output_ports, "delayed_force"],
        output_bindings={
            **{port: ("muscle", port) for port in output_ports},
            "delayed_force": ("delay", "output"),
        },
    )
    registry = ComponentRegistry()
    input_prototypes = {
        ("muscle", port): jnp.zeros_like(values[0]) for port, values in inputs.items()
    }
    graph = compile_graph(GraphDocument(graph=spec), registry, input_prototypes).graph

    expected = {port: [] for port in output_ports}
    for lane in range(2):
        muscle = RigidTendonHillMuscleThelen(**params)
        state = muscle.init_state(key=key)
        steps = []
        for t in range(n_steps):
            step_inputs = {port: values[t, lane] for port, values in inputs.items()}
            outputs, state = muscle(step_inputs, state, key=key)
            steps.append(outputs)
        for port in output_ports:
            expected[port].append(jnp.stack([outputs[port] for outputs in steps]))

    expected_outputs = {port: jnp.stack(lanes, axis=1) for port, lanes in expected.items()}
    force = expected_outputs["force"]
    expected_outputs["delayed_force"] = jnp.concatenate((jnp.zeros_like(force[:1]), force[:-1]))

    def assert_rollout_matches(model):
        outputs, _ = model(inputs, model.init_state(key=key), key=key, n_steps=n_steps)
        for port, expected_values in expected_outputs.items():
            np.testing.assert_allclose(
                outputs[port],
                expected_values,
                rtol=1e-6,
                equal_nan=False,
                err_msg=port,
            )

    assert_rollout_matches(graph)
    exported = GraphSpec.model_validate_json(graph_to_spec(graph, registry).model_dump_json())
    exported_params = exported.nodes["muscle"].params
    assert exported_params["n_muscles"] == 2
    assert exported_params["initial_activation"] == pytest.approx(params["initial_activation"])
    reloaded = compile_graph(GraphDocument(graph=exported), registry, input_prototypes).graph
    assert_rollout_matches(reloaded)


class TestThelenMuscle:
    def test_isometric_force_at_optimal_length(self, thelen_muscle, key):
        """At optimal length with full activation, force ~ max isometric."""
        state = init_state_from_component(thelen_muscle)
        mt_length = jnp.array(
            thelen_muscle.optimal_muscle_length + thelen_muscle.tendon_slack_length
        )

        # Drive activation to ~1.0
        for _ in range(500):
            inputs = {
                "excitation": jnp.array(1.0),
                "musculotendon_length": mt_length,
                "musculotendon_velocity": jnp.array(0.0),
            }
            outputs, state = thelen_muscle(inputs, state, key=key)

        force = outputs["force"]
        fmax = thelen_muscle.max_isometric_force
        # At optimal length and zero velocity, force should be close to F_max
        # Allow tolerance because activation may not reach exactly 1.0
        assert force > 0.5 * fmax, f"Isometric force {force} too low (expected near {fmax})"
        assert force < 1.5 * fmax, f"Isometric force {force} too high"

    def test_fv_concentric_reduces_force(self, thelen_muscle, key):
        """Shortening velocity should reduce force vs isometric."""
        state_iso = init_state_from_component(thelen_muscle)
        state_con = init_state_from_component(thelen_muscle)
        mt_length = jnp.array(
            thelen_muscle.optimal_muscle_length + thelen_muscle.tendon_slack_length
        )

        # Converge both to full activation
        for _ in range(500):
            inputs_iso = {
                "excitation": jnp.array(1.0),
                "musculotendon_length": mt_length,
                "musculotendon_velocity": jnp.array(0.0),
            }
            inputs_con = {
                "excitation": jnp.array(1.0),
                "musculotendon_length": mt_length,
                "musculotendon_velocity": jnp.array(-0.5),  # shortening
            }
            out_iso, state_iso = thelen_muscle(inputs_iso, state_iso, key=key)
            out_con, state_con = thelen_muscle(inputs_con, state_con, key=key)

        assert out_con["force"] < out_iso["force"], (
            f"Concentric force {out_con['force']} should be less than isometric {out_iso['force']}"
        )

    def test_fv_eccentric_increases_force(self, thelen_muscle, key):
        """Lengthening velocity should increase force vs isometric."""
        state_iso = init_state_from_component(thelen_muscle)
        state_ecc = init_state_from_component(thelen_muscle)
        mt_length = jnp.array(
            thelen_muscle.optimal_muscle_length + thelen_muscle.tendon_slack_length
        )

        for _ in range(500):
            inputs_iso = {
                "excitation": jnp.array(1.0),
                "musculotendon_length": mt_length,
                "musculotendon_velocity": jnp.array(0.0),
            }
            inputs_ecc = {
                "excitation": jnp.array(1.0),
                "musculotendon_length": mt_length,
                "musculotendon_velocity": jnp.array(0.3),  # lengthening
            }
            out_iso, state_iso = thelen_muscle(inputs_iso, state_iso, key=key)
            out_ecc, state_ecc = thelen_muscle(inputs_ecc, state_ecc, key=key)

        assert out_ecc["force"] > out_iso["force"], (
            f"Eccentric force {out_ecc['force']} should exceed isometric {out_iso['force']}"
        )

    def test_force_positive(self, thelen_muscle, key):
        """Force should never be negative."""
        state = init_state_from_component(thelen_muscle)
        mt_length = jnp.array(
            thelen_muscle.optimal_muscle_length + thelen_muscle.tendon_slack_length
        )

        for _ in range(50):
            inputs = {
                "excitation": jnp.array(0.5),
                "musculotendon_length": mt_length,
                "musculotendon_velocity": jnp.array(-0.2),
            }
            outputs, state = thelen_muscle(inputs, state, key=key)
            assert outputs["force"] >= 0.0, f"Negative force: {outputs['force']}"

    def test_activation_dynamics(self, thelen_muscle, key):
        """Activation should track excitation."""
        state = init_state_from_component(thelen_muscle)
        mt_length = jnp.array(
            thelen_muscle.optimal_muscle_length + thelen_muscle.tendon_slack_length
        )

        for _ in range(500):
            inputs = {
                "excitation": jnp.array(0.7),
                "musculotendon_length": mt_length,
                "musculotendon_velocity": jnp.array(0.0),
            }
            outputs, state = thelen_muscle(inputs, state, key=key)

        assert jnp.allclose(outputs["activation"], 0.7, atol=0.02)


# ============================================================================
# PointMassRadialGeometry Tests
# ============================================================================


class TestPointMassRadialGeometry:
    def test_n_muscles(self):
        geom = PointMassRadialGeometry(n_pairs=4)
        assert geom.n_muscles == 8

    def test_directions_shape(self):
        geom = PointMassRadialGeometry(n_pairs=4)
        assert geom.directions.shape == (8, 2)

    def test_antagonist_directions_opposite(self):
        """Each pair should have opposite directions."""
        geom = PointMassRadialGeometry(n_pairs=4)
        for i in range(4):
            d_pos = geom.directions[2 * i]
            d_neg = geom.directions[2 * i + 1]
            assert jnp.allclose(d_pos + d_neg, 0.0, atol=1e-5), f"Pair {i}: directions not opposite"

    def test_equal_forces_cancel(self):
        """Equal forces in both directions of a pair should cancel."""
        geom = PointMassRadialGeometry(n_pairs=4)
        forces = jnp.ones(8)
        force_2d = geom.forces_to_force_2d(forces)
        assert jnp.allclose(force_2d, 0.0, atol=1e-5)

    def test_single_muscle_direction(self):
        """A force in muscle 0 should push along direction 0."""
        geom = PointMassRadialGeometry(n_pairs=4)
        forces = jnp.zeros(8).at[0].set(10.0)
        force_2d = geom.forces_to_force_2d(forces)
        expected = 10.0 * geom.directions[0]
        assert jnp.allclose(force_2d, expected, atol=1e-5)


# ============================================================================
# Template Tests
# ============================================================================


class TestArm6MuscleRigidTendon:
    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_wrapping_virtual_work(self, key, dtype):
        geometry = TwoLinkArmMuscleGeometry(
            (
                WrappingGeometry(
                    wrap_radius=0.02,
                    attachment_distances=jnp.array([0.12, 0.13], dtype=dtype),
                    reference_length=0.2,
                    joint_index=1,
                ),
            )
        )
        arm = Arm6MuscleRigidTendon(geometry=geometry)
        state = init_state_from_component(arm)
        angular_velocities = jnp.array([-0.2, 0.4], dtype=dtype)
        poses = jnp.array([[0.1, q] for q in (0.3, -0.3, 0.005, -0.005, 0.0)], dtype=dtype)

        @jax.jit
        @jax.vmap
        def sample(angles):
            outputs, _ = arm(
                {
                    "excitation": jnp.ones(1, dtype=dtype),
                    "angles": angles,
                    "angular_velocities": angular_velocities,
                },
                state,
                key=key,
            )
            velocity = geometry.musculotendon_velocities(angles, angular_velocities)
            # Centered length differences stay on one side except at the cusp,
            # where the symmetric convention gives zero.
            step = 1e-3 * angular_velocities
            length_rate = (
                geometry.musculotendon_lengths(angles + step)
                - geometry.musculotendon_lengths(angles - step)
            ) / 2e-3
            return outputs["torques"], outputs["forces"], velocity, length_rate

        torques, forces, velocities, length_rates = sample(poses)
        assert velocities.dtype == dtype
        assert jnp.all(forces > 0)
        assert jnp.allclose(velocities, length_rates, atol=1e-5)
        assert jnp.allclose(
            torques @ angular_velocities,
            -jnp.sum(forces * length_rates, axis=-1),
            rtol=2e-3,
            atol=1e-6,
        )

    def test_100_steps_no_nan(self, arm_template, key):
        """100 steps should produce no NaN values."""
        state = init_state_from_component(arm_template)
        excitation = 0.3 * jnp.ones(6)
        angles = jnp.array([0.5, 0.8])
        angular_velocities = jnp.zeros(2)

        for _ in range(100):
            inputs = {
                "excitation": excitation,
                "angles": angles,
                "angular_velocities": angular_velocities,
            }
            outputs, state = arm_template(inputs, state, key=key)

        for name, val in outputs.items():
            assert not jnp.any(jnp.isnan(val)), f"NaN in output '{name}'"

    def test_output_shapes(self, arm_template, key):
        """Output shapes should match expectations."""
        state = init_state_from_component(arm_template)
        inputs = {
            "excitation": 0.5 * jnp.ones(6),
            "angles": jnp.zeros(2),
            "angular_velocities": jnp.zeros(2),
        }
        outputs, _ = arm_template(inputs, state, key=key)

        assert outputs["torques"].shape == (2,)
        assert outputs["forces"].shape == (6,)
        assert outputs["activations"].shape == (6,)

    def test_jit_works(self, arm_template, key):
        """JIT compilation should work."""
        state = init_state_from_component(arm_template)
        inputs = {
            "excitation": 0.5 * jnp.ones(6),
            "angles": jnp.zeros(2),
            "angular_velocities": jnp.zeros(2),
        }

        @jax.jit
        def step(inputs, state, key):
            return arm_template(inputs, state, key=key)

        outputs, new_state = step(inputs, state, key)
        assert not jnp.any(jnp.isnan(outputs["torques"]))

    def test_nonzero_excitation_produces_torques(self, arm_template, key):
        """Nonzero excitation should eventually produce nonzero torques."""
        state = init_state_from_component(arm_template)
        # Asymmetric excitation so flexor/extensor torques don't cancel
        excitation = jnp.array([0.8, 0.1, 0.8, 0.1, 0.5, 0.1])

        for _ in range(100):
            inputs = {
                "excitation": excitation,
                "angles": jnp.array([0.5, 0.8]),
                "angular_velocities": jnp.zeros(2),
            }
            outputs, state = arm_template(inputs, state, key=key)

        torques = outputs["torques"]
        assert jnp.any(jnp.abs(torques) > 1e-3), f"Expected nonzero torques, got {torques}"


class TestPointMass8MuscleRelu:
    def test_100_steps_no_nan(self, pointmass_template, key):
        """100 steps should produce no NaN values."""
        state = init_state_from_component(pointmass_template)
        excitation = 0.5 * jnp.ones(8)

        for _ in range(100):
            inputs = {"excitation": excitation}
            outputs, state = pointmass_template(inputs, state, key=key)

        for name, val in outputs.items():
            assert not jnp.any(jnp.isnan(val)), f"NaN in output '{name}'"

    def test_output_shapes(self, pointmass_template, key):
        """Output shapes should match expectations."""
        state = init_state_from_component(pointmass_template)
        inputs = {"excitation": 0.5 * jnp.ones(8)}
        outputs, _ = pointmass_template(inputs, state, key=key)

        assert outputs["force_2d"].shape == (2,)
        assert outputs["forces"].shape == (8,)
        assert outputs["activations"].shape == (8,)

    def test_jit_works(self, pointmass_template, key):
        """JIT compilation should work."""
        state = init_state_from_component(pointmass_template)
        inputs = {"excitation": 0.5 * jnp.ones(8)}

        @jax.jit
        def step(inputs, state, key):
            return pointmass_template(inputs, state, key=key)

        outputs, new_state = step(inputs, state, key)
        assert not jnp.any(jnp.isnan(outputs["force_2d"]))

    def test_equal_excitation_cancels(self, pointmass_template, key):
        """Equal excitation to all muscles should produce near-zero net force."""
        state = init_state_from_component(pointmass_template)

        for _ in range(200):
            inputs = {"excitation": 0.5 * jnp.ones(8)}
            outputs, state = pointmass_template(inputs, state, key=key)

        force_2d = outputs["force_2d"]
        assert jnp.allclose(force_2d, 0.0, atol=1.0), f"Expected near-zero force, got {force_2d}"

    def test_asymmetric_excitation_produces_force(self, pointmass_template, key):
        """Asymmetric excitation should produce nonzero net force."""
        state = init_state_from_component(pointmass_template)
        # Only activate positive-direction muscles
        excitation = jnp.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])

        for _ in range(200):
            inputs = {"excitation": excitation}
            outputs, state = pointmass_template(inputs, state, key=key)

        force_2d = outputs["force_2d"]
        assert jnp.linalg.norm(force_2d) > 1.0, f"Expected nonzero force, got {force_2d}"


# ============================================================================
# Gradient Tests
# ============================================================================


class TestGradients:
    def test_relu_muscle_gradient(self, relu_muscle, key):
        """Gradients should flow through ReluMuscle."""
        state = init_state_from_component(relu_muscle)

        def loss_fn(excitation):
            inputs = {"excitation": excitation}
            outputs, _ = relu_muscle(inputs, state, key=key)
            return outputs["force"]

        grad = jax.grad(loss_fn)(jnp.array(0.5))
        assert jnp.isfinite(grad), f"Non-finite gradient: {grad}"
        assert grad > 0.0, "Gradient should be positive for increasing excitation"

    def test_thelen_muscle_gradient(self, thelen_muscle, key):
        """Gradients should flow through Thelen muscle."""
        state = init_state_from_component(thelen_muscle)
        mt_length = jnp.array(
            thelen_muscle.optimal_muscle_length + thelen_muscle.tendon_slack_length
        )

        def loss_fn(excitation):
            inputs = {
                "excitation": excitation,
                "musculotendon_length": mt_length,
                "musculotendon_velocity": jnp.array(0.0),
            }
            outputs, _ = thelen_muscle(inputs, state, key=key)
            return outputs["force"]

        grad = jax.grad(loss_fn)(jnp.array(0.5))
        assert jnp.isfinite(grad), f"Non-finite gradient: {grad}"

    def test_arm_template_gradient(self, arm_template, key):
        """Gradients should flow through the arm template."""
        state = init_state_from_component(arm_template)

        def loss_fn(excitation):
            inputs = {
                "excitation": excitation,
                "angles": jnp.array([0.5, 0.8]),
                "angular_velocities": jnp.zeros(2),
            }
            outputs, _ = arm_template(inputs, state, key=key)
            return jnp.sum(outputs["torques"] ** 2)

        grad = jax.grad(loss_fn)(0.5 * jnp.ones(6))
        assert jnp.all(jnp.isfinite(grad)), f"Non-finite gradients: {grad}"

    def test_pointmass_template_gradient(self, pointmass_template, key):
        """Gradients should flow through the point mass template."""
        state = init_state_from_component(pointmass_template)

        def loss_fn(excitation):
            inputs = {"excitation": excitation}
            outputs, _ = pointmass_template(inputs, state, key=key)
            return jnp.sum(outputs["force_2d"] ** 2)

        grad = jax.grad(loss_fn)(0.5 * jnp.ones(8))
        assert jnp.all(jnp.isfinite(grad)), f"Non-finite gradients: {grad}"
