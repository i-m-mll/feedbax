"""Tests for Hill-type muscle models.

Tests force-length-velocity curves, rigid and compliant tendon muscles,
and musculoskeletal arm integration.

:copyright: Copyright 2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from dataclasses import replace
import math

import pytest
import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from feedbax.mechanics.hill_muscles import (
    ForceLengthCurve,
    PassiveForceLengthCurve,
    ForceVelocityCurve,
    TendonForceLengthCurve,
    HillMuscleParams,
    RigidTendonHillMuscle,
    CompliantTendonHillMuscle,
    ActivationDynamics,
)
from feedbax.mechanics.geometry import (
    ConstantMomentArmGeometry,
    TwoLinkArmMuscleGeometry,
)
from feedbax.mechanics.musculoskeletal import (
    RigidTendonMusculoskeletalArm,
    CompliantTendonMusculoskeletalArm,
)
from feedbax.mechanics.units import DEFAULT_MUSCLE_VMAX, ECCENTRIC_VELOCITY_LIMIT_FRACTION
from feedbax.runtime.graph import init_state_from_component


pytestmark = pytest.mark.usefixtures("enable_jax_x64")


class TestForceLengthCurve:
    """Tests for the active force-length relationship."""

    @pytest.fixture
    def fl_curve(self):
        return ForceLengthCurve()

    def test_optimal_length_force(self, fl_curve):
        """Force at optimal length (1.0) should be maximal (~1.0)."""
        force = fl_curve(jnp.array(1.0))
        assert jnp.isclose(force, 1.0, atol=0.01)

    def test_force_decreases_away_from_optimal(self, fl_curve):
        """Force should decrease as length deviates from optimal."""
        optimal_force = fl_curve(jnp.array(1.0))
        short_force = fl_curve(jnp.array(0.7))
        long_force = fl_curve(jnp.array(1.4))

        assert short_force < optimal_force
        assert long_force < optimal_force

    def test_symmetric_shape(self, fl_curve):
        """Curve should be roughly symmetric around optimal length."""
        short = fl_curve(jnp.array(0.8))
        long = fl_curve(jnp.array(1.2))
        # Not exactly symmetric due to Gaussian shape, but close
        assert jnp.abs(short - long) < 0.1

    def test_force_positive(self, fl_curve):
        """Force should always be positive."""
        lengths = jnp.linspace(0.5, 1.8, 50)
        forces = jax.vmap(fl_curve)(lengths)
        assert jnp.all(forces >= 0)

    def test_gradient_exists(self, fl_curve):
        """Gradient should exist and be finite."""
        grad_fn = jax.grad(lambda x: fl_curve(x))
        grad = grad_fn(jnp.array(1.0))
        assert jnp.isfinite(grad)


class TestPassiveForceLengthCurve:
    """Tests for the passive force-length relationship."""

    @pytest.fixture
    def passive_curve(self):
        return PassiveForceLengthCurve()

    def test_zero_below_optimal(self, passive_curve):
        """Passive force should be zero below optimal length."""
        force = passive_curve(jnp.array(0.9))
        assert jnp.isclose(force, 0.0, atol=1e-6)

    def test_increases_above_optimal(self, passive_curve):
        """Passive force should increase exponentially above optimal."""
        force_1 = passive_curve(jnp.array(1.2))
        force_2 = passive_curve(jnp.array(1.4))
        force_3 = passive_curve(jnp.array(1.6))

        assert force_2 > force_1
        assert force_3 > force_2

    def test_reaches_one_at_strain_threshold(self, passive_curve):
        """Passive force should reach ~1.0 at specified strain."""
        strain = passive_curve.strain_at_one_norm_force
        length = 1.0 + strain
        force = passive_curve(jnp.array(length))
        assert jnp.isclose(force, 1.0, atol=0.1)


class TestForceVelocityCurve:
    """Tests for the force-velocity relationship."""

    @pytest.fixture
    def fv_curve(self):
        return ForceVelocityCurve()

    def test_bounded_monotone_curve(self, fv_curve):
        velocities = jnp.linspace(-1.0, 0.1, 221)
        forces = jax.vmap(fv_curve)(velocities)

        assert fv_curve(jnp.array(-1.0)) == 0.0
        assert fv_curve(jnp.array(0.0)) == 1.0
        assert jnp.all((forces >= 0.0) & (forces <= 1.4))
        assert jnp.all(jnp.diff(forces) > 0.0)
        saturated_velocities = jnp.array([-2.0, 0.1, 0.25, 1.0])
        assert jnp.array_equal(fv_curve(saturated_velocities), jnp.array([0.0, 1.4, 1.4, 1.4]))

    def test_finite_reverse_gradients_at_branch_boundaries_and_former_poles(self, fv_curve):
        velocities = jnp.array([-1.0, -0.25, -1e-6, 0.0, 1e-6, 0.1, 0.25])
        gradients = jax.jit(jax.vmap(jax.grad(lambda v: fv_curve(v))))(velocities)

        assert jnp.all(jnp.isfinite(gradients))

    def test_inverse_recovers_velocity_and_saturates(self, fv_curve):
        forces = jnp.array([-0.1, 0.0, 1.0, 1.4, 1.5])
        assert jnp.array_equal(fv_curve.inverse(forces), jnp.array([-1.0, -1.0, 0.0, 0.1, 0.1]))

        velocities = jnp.array([-0.9, -0.5, -0.01, 0.01, 0.05, 0.09])
        assert jnp.allclose(fv_curve.inverse(fv_curve(velocities)), velocities)


class TestTendonForceLengthCurve:
    """Tests for tendon force-length relationship."""

    @pytest.fixture
    def tendon_curve(self):
        return TendonForceLengthCurve()

    def test_zero_below_slack(self, tendon_curve):
        """Tendon force should be zero below slack length."""
        force = tendon_curve(jnp.array(0.95))
        assert jnp.isclose(force, 0.0, atol=1e-6)

    def test_force_at_slack(self, tendon_curve):
        """Force at slack length should be zero."""
        force = tendon_curve(jnp.array(1.0))
        assert jnp.isclose(force, 0.0, atol=1e-6)

    def test_increasing_above_slack(self, tendon_curve):
        """Force should increase above slack length."""
        force_1 = tendon_curve(jnp.array(1.01))
        force_2 = tendon_curve(jnp.array(1.02))
        force_3 = tendon_curve(jnp.array(1.03))

        assert force_2 > force_1
        assert force_3 > force_2

    def test_inverse_roundtrip(self, tendon_curve):
        """Forward-inverse should round-trip."""
        length = jnp.array(1.02)
        force = tendon_curve(length)
        recovered_length = tendon_curve.inverse(force)
        assert jnp.isclose(length, recovered_length, atol=1e-6)


class TestRigidTendonHillMuscle:
    """Tests for rigid tendon Hill muscle."""

    @pytest.fixture
    def muscle_params(self):
        return HillMuscleParams(
            max_isometric_force=500.0,
            optimal_fiber_length=0.1,
            tendon_slack_length=0.15,
            pennation_angle=0.0,
            tau_activation=0.01,
            tau_deactivation=0.04,
            vmax=DEFAULT_MUSCLE_VMAX,
        )

    @pytest.fixture
    def muscle(self, muscle_params):
        return RigidTendonHillMuscle(params=muscle_params)

    def test_isometric_force_at_optimal(self, muscle):
        """Isometric force at optimal length should equal max isometric."""
        # MT length = tendon slack + optimal fiber
        mt_length = (
            muscle.params.tendon_slack_length +
            muscle.params.optimal_fiber_length
        )

        force = muscle.compute_force(
            activation=jnp.array(1.0),
            fiber_length=jnp.array(muscle.params.optimal_fiber_length),
            fiber_velocity=jnp.array(0.0),
            musculotendon_length=jnp.array(mt_length),
        )

        # Should be close to max isometric (slight passive component)
        assert jnp.isclose(force, muscle.params.max_isometric_force, rtol=0.05)

    def test_zero_activation_gives_passive_force(self, muscle):
        """Zero activation should give only passive force."""
        mt_length = (
            muscle.params.tendon_slack_length +
            muscle.params.optimal_fiber_length * 1.3  # stretched
        )

        force = muscle.compute_force(
            activation=jnp.array(0.0),
            fiber_length=jnp.array(muscle.params.optimal_fiber_length * 1.3),
            fiber_velocity=jnp.array(0.0),
            musculotendon_length=jnp.array(mt_length),
        )

        # Should have some passive force due to stretching
        assert force > 0
        assert force < muscle.params.max_isometric_force

    def test_fiber_length_from_mt_length(self, muscle):
        """Test rigid tendon fiber length calculation."""
        mt_length = jnp.array(0.2)  # arbitrary
        fiber_length = muscle.compute_fiber_length_from_mt_length(mt_length)

        expected = mt_length - muscle.params.tendon_slack_length
        assert jnp.isclose(fiber_length, expected, atol=1e-6)

    @pytest.mark.parametrize("angle", [0.005, 0.3])
    def test_fiber_velocity_differentiates_the_authored_length(self, muscle_params, angle):
        muscle = RigidTendonHillMuscle(params=replace(muscle_params, pennation_angle=angle))
        fiber_length, expected = jax.jvp(
            muscle.compute_fiber_length_from_mt_length,
            (jnp.asarray(0.24),), (jnp.asarray(-0.15),),
        )
        actual = muscle.compute_fiber_velocity_from_mt_velocity(fiber_length, jnp.asarray(-0.15))
        assert float(actual) == pytest.approx(float(expected), rel=1e-6)


class TestCompliantTendonMuscle:
    """Tests for compliant tendon Hill muscle."""

    @pytest.fixture
    def muscle_params(self):
        return HillMuscleParams(
            max_isometric_force=500.0,
            optimal_fiber_length=0.1,
            tendon_slack_length=0.15,
            pennation_angle=0.0,
            tau_activation=0.01,
            tau_deactivation=0.04,
            vmax=DEFAULT_MUSCLE_VMAX,
        )

    @pytest.fixture
    def muscle(self, muscle_params):
        return CompliantTendonHillMuscle(
            muscle_params=muscle_params,
            dt=0.001,
            key=jr.PRNGKey(0),
        )

    def test_initialization(self, muscle):
        """Test compliant muscle initializes correctly."""
        assert muscle.muscle_params.max_isometric_force == 500.0
        assert muscle.dt == 0.001

    def test_pennation_tracks_replaced_parameter(self, muscle_params):
        original = CompliantTendonHillMuscle(replace(muscle_params, pennation_angle=0.1))
        fresh = CompliantTendonHillMuscle(replace(muscle_params, pennation_angle=0.3))
        optimal_length = jnp.asarray(muscle_params.optimal_fiber_length)

        def pennation(angle):
            updated = eqx.tree_at(lambda model: model.muscle_params.pennation_angle, original, angle)
            return updated.compute_pennation_angle(optimal_length)

        value, derivative = jax.jit(jax.value_and_grad(pennation))(jnp.asarray(0.3))
        assert float(value) == pytest.approx(float(fresh.compute_pennation_angle(optimal_length)))
        assert float(derivative) == pytest.approx(1.0)

    def test_single_step(self, muscle):
        """Test single integration step works."""
        state = init_state_from_component(muscle)
        key = jr.PRNGKey(1)

        inputs = {
            "excitation": jnp.array(0.5),
            "musculotendon_length": jnp.array(0.25),
            "musculotendon_velocity": jnp.array(0.0),
        }

        outputs, new_state = muscle(inputs, state, key=key)

        assert "force" in outputs
        assert "state" in outputs
        assert jnp.isfinite(outputs["force"])

    def test_extract_outputs_rejects_state_only_force_contract(self, muscle):
        dae_state = init_state_from_component(muscle).get(muscle.state_index)

        with pytest.raises(NotImplementedError, match="musculotendon_length"):
            muscle.extract_outputs(dae_state.system)

    def test_single_step_force_uses_tendon_length_not_zero_placeholder(self, muscle):
        state = init_state_from_component(muscle)
        outputs, _ = muscle(
            {
                "excitation": jnp.array(0.5),
                "musculotendon_length": jnp.array(0.30),
                "musculotendon_velocity": jnp.array(0.0),
            },
            state,
            key=jr.PRNGKey(11),
        )

        assert outputs["force"] > 0.0

    @pytest.mark.parametrize(
        "activation, expected_velocity", [(0.6, -0.5), (1e-7, -0.5), (0.0, 0.0)],
    )
    def test_fiber_velocity_balances_fiber_and_tendon_force(
        self, muscle_params, activation, expected_velocity,
    ):
        params = replace(muscle_params, pennation_angle=0.3)
        muscle = CompliantTendonHillMuscle(params)
        projection = math.cos(params.pennation_angle)
        activation = jnp.asarray(activation)
        state = eqx.tree_at(
            lambda s: s.activation,
            muscle.init_system_state(key=jr.PRNGKey(2)),
            activation,
        )
        normalized_tendon_force = activation * projection / 6.0
        tendon_length = params.tendon_slack_length * muscle.tendon_force_length.inverse(
            normalized_tendon_force
        )
        if activation == 0.0:
            tendon_length -= 0.001  # A slack tendon remains unloaded after rounding.
        mt_length = state.fiber_length * projection + tendon_length

        derivative = muscle.vector_field(0.0, state, (activation, mt_length, jnp.array(0.0)))
        fiber_velocity = derivative.fiber_length
        fiber_force = muscle.compute_fiber_force(activation, state.fiber_length, fiber_velocity)
        tendon_force = muscle.compute_tendon_force(tendon_length)

        assert jnp.isclose(
            fiber_velocity, expected_velocity * params.vmax * params.optimal_fiber_length,
        )
        assert jnp.isclose(fiber_force * projection, tendon_force, rtol=1e-6, atol=0.0)


class TestMuscleGeometry:
    """Tests for muscle geometry computations."""

    def test_constant_moment_arm_length(self):
        """Test constant moment arm geometry."""
        geometry = ConstantMomentArmGeometry(
            moment_arms=jnp.array([0.04, 0.025]),
            reference_length=0.2,
        )

        angles = jnp.array([0.0, 0.0])
        length = geometry.musculotendon_length(angles)
        assert jnp.isclose(length, 0.2)

        # Flexion should shorten
        angles_flexed = jnp.array([0.5, 0.3])
        length_flexed = geometry.musculotendon_length(angles_flexed)
        assert length_flexed < 0.2

    def test_six_muscle_geometry(self):
        """Test standard 6-muscle geometry."""
        geometry = TwoLinkArmMuscleGeometry.default_six_muscle()

        assert geometry.n_muscles == 6

        angles = jnp.array([0.5, 0.3])
        lengths = geometry.musculotendon_lengths(angles)
        moment_arms = geometry.moment_arms(angles)

        assert lengths.shape == (6,)
        assert moment_arms.shape == (6, 2)
        assert jnp.all(jnp.isfinite(lengths))
        assert jnp.all(jnp.isfinite(moment_arms))

    def test_forces_to_torques(self):
        """Test muscle force to joint torque conversion."""
        geometry = TwoLinkArmMuscleGeometry.default_six_muscle()

        angles = jnp.array([0.5, 0.3])
        forces = jnp.ones(6) * 100.0  # 100N each

        torques = geometry.forces_to_torques(angles, forces)
        assert torques.shape == (2,)
        assert jnp.all(jnp.isfinite(torques))


class TestActivationDynamics:
    """Tests for activation dynamics."""

    @pytest.fixture
    def dynamics(self):
        return ActivationDynamics(tau_activation=0.01, tau_deactivation=0.04)

    def test_activation_increase(self, dynamics):
        """Activation should increase when excitation > activation."""
        d_act = dynamics(
            excitation=jnp.array(1.0),
            activation=jnp.array(0.5),
        )
        assert d_act > 0

    def test_activation_decrease(self, dynamics):
        """Activation should decrease when excitation < activation."""
        d_act = dynamics(
            excitation=jnp.array(0.0),
            activation=jnp.array(0.5),
        )
        assert d_act < 0

    def test_equilibrium(self, dynamics):
        """Derivative should be zero at equilibrium."""
        d_act = dynamics(
            excitation=jnp.array(0.5),
            activation=jnp.array(0.5),
        )
        assert jnp.isclose(d_act, 0.0, atol=1e-10)


class TestRigidTendonMusculoskeletalArm:
    """Tests for the integrated musculoskeletal arm model."""

    @pytest.fixture
    def arm(self):
        return RigidTendonMusculoskeletalArm(
            dt=0.01,
            key=jr.PRNGKey(0),
        )

    def test_initialization(self, arm):
        """Test arm initializes correctly."""
        assert arm.n_muscles == 6
        assert arm.dt == 0.01

    def test_non_euler_solver_rejected(self):
        with pytest.raises(ValueError, match="solver_type=Kvaerno3 is not supported"):
            RigidTendonMusculoskeletalArm(
                dt=0.01,
                solver_type=dfx.Kvaerno3,
                key=jr.PRNGKey(0),
            )

    def test_invalid_arm_parameter_shape_rejected(self):
        with pytest.raises(ValueError, match="arm_l must have shape"):
            RigidTendonMusculoskeletalArm(
                arm_l=(0.30,),
                dt=0.01,
                key=jr.PRNGKey(0),
            )

    def test_invalid_timestep_rejected(self):
        with pytest.raises(ValueError, match="dt must be finite and positive"):
            RigidTendonMusculoskeletalArm(dt=0.0, key=jr.PRNGKey(0))

    def test_muscle_params_must_match_geometry(self):
        params = tuple(
            HillMuscleParams(
                max_isometric_force=500.0,
                optimal_fiber_length=0.08,
                tendon_slack_length=0.12,
            )
            for _ in range(5)
        )
        with pytest.raises(ValueError, match="muscle_params length must match"):
            RigidTendonMusculoskeletalArm(
                muscle_params=params,
                dt=0.01,
                key=jr.PRNGKey(0),
            )

    def test_single_step(self, arm):
        """Test single integration step."""
        state = init_state_from_component(arm)
        key = jr.PRNGKey(1)

        excitations = jnp.ones(6) * 0.5
        inputs = {"excitations": excitations}

        outputs, new_state = arm(inputs, state, key=key)

        assert "effector" in outputs
        assert "state" in outputs
        assert "forces" in outputs
        assert "torques" in outputs

        assert outputs["effector"].pos.shape == (2,)
        assert outputs["forces"].shape == (6,)
        assert outputs["torques"].shape == (2,)

    def test_gradient_flow(self, arm):
        """Test gradients flow through musculoskeletal model."""
        state = init_state_from_component(arm)

        def loss_fn(excitations, state, key):
            inputs = {"excitations": excitations}
            outputs, _ = arm(inputs, state, key=key)
            return jnp.sum(outputs["effector"].pos ** 2)

        excitations = jnp.ones(6) * 0.5
        key = jr.PRNGKey(2)

        grad_exc = jax.grad(loss_fn)(excitations, state, key)

        assert grad_exc.shape == (6,)
        assert not jnp.any(jnp.isnan(grad_exc))

    def test_no_nan_in_simulation(self, arm):
        """Test no NaN in extended simulation.

        Scanned under one compiled rollout: same 1000 steps and the same per-step
        NaN check on effector position and muscle forces, without paying a host
        round trip and a device sync for each of them.
        """
        state = init_state_from_component(arm)

        n_steps = 1000
        excitations = jnp.array([0.5, 0.3, 0.4, 0.2, 0.6, 0.1])
        inputs = {"excitations": excitations}

        @eqx.filter_jit
        def rollout(initial_state, keys):
            def step(step_state, step_key):
                outputs, next_state = arm(inputs, step_state, key=step_key)
                nan_seen = jnp.any(jnp.isnan(outputs["effector"].pos)) | jnp.any(
                    jnp.isnan(outputs["forces"])
                )
                return next_state, nan_seen

            return jax.lax.scan(step, initial_state, keys)

        _final_state, nan_at_step = rollout(state, jr.split(jr.PRNGKey(3), n_steps))

        assert not jnp.any(nan_at_step), f"NaN at step {int(jnp.argmax(nan_at_step))}"

    def test_muscle_activation_bounds(self, arm):
        """Test that muscle activations stay bounded."""
        state = init_state_from_component(arm)
        key = jr.PRNGKey(4)

        # Oscillating excitations
        n_steps = 100
        for i in range(n_steps):
            key, subkey = jr.split(key)
            exc = 0.5 + 0.5 * jnp.sin(i * 0.1) * jnp.ones(6)
            outputs, state = arm({"excitations": exc}, state, key=subkey)

            # Check activation bounds
            activations = outputs["state"].activations
            assert jnp.all(activations >= 0)
            assert jnp.all(activations <= 1)


class TestCompliantTendonMusculoskeletalArm:
    """Tests for compliant tendon musculoskeletal arm."""

    @pytest.fixture
    def arm(self):
        return CompliantTendonMusculoskeletalArm(
            dt=0.001,
            key=jr.PRNGKey(0),
        )

    def test_initialization(self, arm):
        """Test initialization."""
        assert arm.n_muscles == 6
        assert arm.dt == 0.001

    def test_invalid_timestep_rejected(self):
        with pytest.raises(ValueError, match="dt must be finite and positive"):
            CompliantTendonMusculoskeletalArm(dt=0.0, key=jr.PRNGKey(0))

    def test_single_step(self, arm):
        """Test single step works."""
        state = init_state_from_component(arm)
        key = jr.PRNGKey(1)

        excitations = jnp.ones(6) * 0.3
        inputs = {"excitations": excitations}

        outputs, new_state = arm(inputs, state, key=key)

        assert "effector" in outputs
        assert "forces" in outputs
        assert jnp.all(jnp.isfinite(outputs["forces"]))

    @pytest.mark.parametrize(
        ("lengthening_force_multiplier", "lengthening_velocity"),
        [(1.0 + 7.0 / 30.0, 0.05), (2.0, ECCENTRIC_VELOCITY_LIMIT_FRACTION)],
        ids=["equilibrium", "velocity-cap"],
    )
    def test_tendon_load_drives_motion_and_outputs(
        self, lengthening_force_multiplier, lengthening_velocity
    ):
        params = HillMuscleParams(
            max_isometric_force=500.0,
            optimal_fiber_length=0.08,
            tendon_slack_length=0.12,
            pennation_angle=0.3,
        )
        activation = jnp.array([0.6, 0.6])
        projection = math.cos(params.pennation_angle)
        normalized_tendon_force = activation * projection * jnp.array([
            1.0 / 6.0, lengthening_force_multiplier,
        ])
        tendon_lengths = params.tendon_slack_length * TendonForceLengthCurve().inverse(
            normalized_tendon_force
        )
        geometry = TwoLinkArmMuscleGeometry(
            geometries=tuple(
                ConstantMomentArmGeometry(
                    moment_arms=jnp.array([moment_arm, 0.0]),
                    reference_length=params.optimal_fiber_length * projection + float(tendon_length),
                )
                for moment_arm, tendon_length in zip((0.04, -0.04), tendon_lengths, strict=True)
            )
        )
        key = jr.PRNGKey(0)
        arm = CompliantTendonMusculoskeletalArm(
            muscle_params=(params, params),
            geometry=geometry,
            arm_l=(0.30, 0.33),
            arm_m=(1.4, 1.0),
            arm_I=(0.025, 0.045),
            arm_s=(0.11, 0.16),
            arm_B=((0.0, 0.0), (0.0, 0.0)),
            dt=1e-5,
            key=key,
        )
        state = init_state_from_component(arm)
        dae_state = state.get(arm.state_index)
        dae_state = eqx.tree_at(lambda s: s.system.activations, dae_state, activation)
        state = state.set(arm.state_index, dae_state)

        outputs, stepped_state = arm({"excitations": activation}, state, key=key)
        stepped_system = stepped_state.get(arm.state_index).system
        normalized_velocities = (stepped_system.fiber_lengths - params.optimal_fiber_length) / (
            arm.dt * params.vmax * params.optimal_fiber_length
        )
        assert jnp.allclose(normalized_velocities, jnp.array([-0.5, lengthening_velocity]))

        # Euler leaves both joint angles at zero on the first step from rest.
        # The inertia there is fixed by the parameters above, and sin(0) makes
        # the Coriolis torque zero even after the step.
        inertia = jnp.array([[0.256, 0.093], [0.093, 0.045]])
        start_torque = 0.04 * params.max_isometric_force * (
            normalized_tendon_force[0] - normalized_tendon_force[1]
        )
        assert jnp.allclose(
            inertia @ (stepped_system.arm.d_angle / arm.dt), jnp.array([start_torque, 0.0])
        )

        post_acceleration = arm.vector_field(arm.dt, stepped_system, activation).arm.d_angle
        post_force_torque = 0.04 * (outputs["forces"][0] - outputs["forces"][1])
        assert jnp.allclose(outputs["torques"], jnp.array([post_force_torque, 0.0]))
        assert jnp.allclose(inertia @ post_acceleration, outputs["torques"])

    def test_gradient_flow_compliant(self, arm):
        """Test gradients through compliant tendon model."""
        state = init_state_from_component(arm)

        def loss_fn(excitations, state, key):
            inputs = {"excitations": excitations}
            outputs, _ = arm(inputs, state, key=key)
            return jnp.sum(outputs["effector"].pos ** 2)

        excitations = jnp.ones(6) * 0.3
        key = jr.PRNGKey(2)

        grad_exc = jax.grad(loss_fn)(excitations, state, key)

        assert not jnp.any(jnp.isnan(grad_exc))


class TestJitAndVmap:
    """Test JIT and vmap compatibility."""

    def test_jit_rigid_arm(self):
        """Test JIT compilation of rigid tendon arm."""
        arm = RigidTendonMusculoskeletalArm(dt=0.01, key=jr.PRNGKey(0))
        state = init_state_from_component(arm)

        @jax.jit
        def step(excitations, state, key):
            return arm({"excitations": excitations}, state, key=key)

        outputs, new_state = step(jnp.ones(6) * 0.5, state, jr.PRNGKey(1))
        assert outputs["effector"].pos.shape == (2,)

    def test_vmap_over_excitations(self):
        """Test vmapping over different excitation patterns."""
        arm = RigidTendonMusculoskeletalArm(dt=0.01, key=jr.PRNGKey(0))

        def run_with_excitation(excitations):
            state = init_state_from_component(arm)
            outputs, _ = arm({"excitations": excitations}, state, key=jr.PRNGKey(1))
            return outputs["effector"].pos

        batch_size = 8
        excitations = jr.uniform(jr.PRNGKey(2), (batch_size, 6))

        positions = jax.vmap(run_with_excitation)(excitations)
        assert positions.shape == (batch_size, 2)
        assert not jnp.any(jnp.isnan(positions))
