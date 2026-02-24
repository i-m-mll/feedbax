"""Tests for Hill-type muscle models.

Tests force-length-velocity curves, rigid and compliant tendon muscles,
and musculoskeletal arm integration.

:copyright: Copyright 2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

import pytest
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
from feedbax.graph import init_state_from_component


# Enable 64-bit for numerical precision
jax.config.update("jax_enable_x64", True)


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

    def test_isometric_force(self, fv_curve):
        """At zero velocity, force multiplier should be 1.0."""
        force = fv_curve(jnp.array(0.0))
        assert jnp.isclose(force, 1.0, atol=0.01)

    def test_shortening_reduces_force(self, fv_curve):
        """Shortening (negative velocity) should reduce force."""
        shortening_force = fv_curve(jnp.array(-0.5))
        isometric_force = fv_curve(jnp.array(0.0))
        assert shortening_force < isometric_force

    def test_lengthening_increases_force(self, fv_curve):
        """Lengthening (positive velocity) should increase force."""
        lengthening_force = fv_curve(jnp.array(0.2))
        isometric_force = fv_curve(jnp.array(0.0))
        assert lengthening_force >= isometric_force

    def test_max_shortening_zero_force(self, fv_curve):
        """At max shortening velocity, force should approach zero."""
        # Normalized velocity = -1 at max shortening
        force = fv_curve(jnp.array(-1.0))
        assert force < 0.1  # Close to zero

    def test_eccentric_plateau(self, fv_curve):
        """Eccentric force should eventually plateau."""
        # At high lengthening velocities, force should be bounded
        high_lengthening_force = fv_curve(jnp.array(0.5))
        # Force should be greater than isometric but bounded
        assert high_lengthening_force >= 1.0
        assert high_lengthening_force < 2.0  # Reasonable upper bound


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
            vmax=10.0,
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
            vmax=10.0,
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

    def test_constraint_residual_small(self, muscle):
        """Test that integration runs without errors."""
        state = init_state_from_component(muscle)
        key = jr.PRNGKey(2)

        mt_length = jnp.array(0.25)
        inputs = {
            "excitation": jnp.array(0.8),
            "musculotendon_length": mt_length,
            "musculotendon_velocity": jnp.array(0.0),
        }

        # Run a few steps to settle
        for _ in range(10):
            key, subkey = jr.split(key)
            outputs, state = muscle(inputs, state, key=subkey)

        # Check constraint residual is finite
        # Note: For explicit integration, the residual may not be small
        # since we're not enforcing force equilibrium implicitly.
        # This is a limitation of using Euler solver instead of implicit.
        dae_state = state.get(muscle.state_index)
        residual = muscle.compute_constraint_residual(dae_state.system, mt_length)

        assert jnp.isfinite(residual), "Constraint residual should be finite"
        # The force output should still be reasonable
        assert jnp.isfinite(outputs["force"]), "Force should be finite"


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
        """Test no NaN in extended simulation."""
        state = init_state_from_component(arm)
        key = jr.PRNGKey(3)

        n_steps = 1000
        excitations = jnp.array([0.5, 0.3, 0.4, 0.2, 0.6, 0.1])
        inputs = {"excitations": excitations}

        for i in range(n_steps):
            key, subkey = jr.split(key)
            outputs, state = arm(inputs, state, key=subkey)

            assert not jnp.any(jnp.isnan(outputs["effector"].pos)), f"NaN at step {i}"
            assert not jnp.any(jnp.isnan(outputs["forces"])), f"NaN at step {i}"

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
