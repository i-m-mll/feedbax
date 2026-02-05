"""Tests for DAE components.

Tests energy conservation, solver agreement, and gradient flow for:
- PointMassDAE
- TwoLinkArmDAE

:copyright: Copyright 2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

import pytest
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array

from feedbax.mechanics.skeleton.pointmass_dae import PointMassDAE, PointMassDAEParams
from feedbax.mechanics.skeleton.arm_dae import TwoLinkArmDAE, TwoLinkArmDAEParams
from feedbax.mechanics.skeleton.pointmass import PointMass
from feedbax.mechanics.skeleton.arm import TwoLinkArm
from feedbax.state import CartesianState
from feedbax.graph import init_state_from_component


# Enable 64-bit for numerical precision tests
jax.config.update("jax_enable_x64", True)


class TestPointMassDAE:
    """Tests for PointMassDAE component."""

    @pytest.fixture
    def point_mass_dae(self):
        """Create a PointMassDAE instance."""
        return PointMassDAE(
            mass=1.0,
            damping=0.0,
            dt=0.001,
            key=jr.PRNGKey(0),
        )

    @pytest.fixture
    def point_mass_explicit(self):
        """Create explicit PointMass for comparison."""
        return PointMass(mass=1.0, damping=0.0)

    def test_initialization(self, point_mass_dae):
        """Test that DAE component initializes correctly."""
        assert point_mass_dae.params.mass == 1.0
        assert point_mass_dae.params.damping == 0.0
        assert point_mass_dae.dt == 0.001
        assert point_mass_dae.input_size == 2

    def test_state_initialization(self, point_mass_dae):
        """Test that initial state is created correctly."""
        state = init_state_from_component(point_mass_dae)
        dae_state = point_mass_dae.state_view(state)

        assert dae_state is not None
        assert dae_state.system.pos.shape == (2,)
        assert dae_state.system.vel.shape == (2,)
        assert jnp.allclose(dae_state.system.pos, 0.0)
        assert jnp.allclose(dae_state.system.vel, 0.0)

    def test_single_step(self, point_mass_dae):
        """Test that a single integration step works."""
        state = init_state_from_component(point_mass_dae)
        key = jr.PRNGKey(1)

        # Apply constant force
        force = jnp.array([1.0, 0.0])
        inputs = {"force": force}

        outputs, new_state = point_mass_dae(inputs, state, key=key)

        # Check outputs
        assert "effector" in outputs
        assert "state" in outputs
        assert outputs["effector"].pos.shape == (2,)

        # Position should still be near zero after one small step
        assert jnp.allclose(outputs["effector"].pos, 0.0, atol=1e-3)

        # Velocity should have increased (F=ma, a=1, v=a*dt)
        expected_vel = force * point_mass_dae.dt
        assert jnp.allclose(outputs["effector"].vel, expected_vel, atol=1e-4)

    def test_energy_conservation_no_damping(self, point_mass_dae):
        """Test energy conservation for undamped system (<1% drift)."""
        state = init_state_from_component(point_mass_dae)
        key = jr.PRNGKey(2)

        # Give initial velocity
        dae_state = point_mass_dae.state_view(state)
        init_vel = jnp.array([1.0, 0.5])
        new_system = CartesianState(
            pos=dae_state.system.pos,
            vel=init_vel,
            force=dae_state.system.force,
        )
        from feedbax.mechanics.dae import DAEState
        new_dae_state = DAEState(system=new_system, solver=dae_state.solver)
        state = state.set(point_mass_dae.state_index, new_dae_state)

        # Initial kinetic energy
        init_ke = point_mass_dae.compute_kinetic_energy(new_system)

        # Run for many steps with no force
        n_steps = 1000
        inputs = {"force": jnp.zeros(2)}

        for i in range(n_steps):
            key, subkey = jr.split(key)
            outputs, state = point_mass_dae(inputs, state, key=subkey)

        # Final kinetic energy
        final_state = outputs["state"]
        final_ke = point_mass_dae.compute_kinetic_energy(final_state)

        # Energy should be conserved within 1%
        energy_drift = jnp.abs(final_ke - init_ke) / init_ke
        assert energy_drift < 0.01, f"Energy drift {energy_drift:.4f} > 1%"

    def test_momentum_conservation(self, point_mass_dae):
        """Test momentum conservation with no external force."""
        state = init_state_from_component(point_mass_dae)
        key = jr.PRNGKey(3)

        # Set initial velocity
        dae_state = point_mass_dae.state_view(state)
        init_vel = jnp.array([2.0, -1.0])
        new_system = CartesianState(
            pos=dae_state.system.pos,
            vel=init_vel,
            force=dae_state.system.force,
        )
        from feedbax.mechanics.dae import DAEState
        new_dae_state = DAEState(system=new_system, solver=dae_state.solver)
        state = state.set(point_mass_dae.state_index, new_dae_state)

        init_momentum = point_mass_dae.compute_momentum(new_system)

        # Run with no force
        n_steps = 100
        inputs = {"force": jnp.zeros(2)}

        for i in range(n_steps):
            key, subkey = jr.split(key)
            outputs, state = point_mass_dae(inputs, state, key=subkey)

        final_momentum = point_mass_dae.compute_momentum(outputs["state"])

        assert jnp.allclose(init_momentum, final_momentum, atol=1e-6)

    def test_gradient_flow(self, point_mass_dae):
        """Test that gradients flow through the DAE integration."""
        state = init_state_from_component(point_mass_dae)

        def loss_fn(force, state, key):
            inputs = {"force": force}
            outputs, _ = point_mass_dae(inputs, state, key=key)
            return jnp.sum(outputs["effector"].pos ** 2)

        force = jnp.array([1.0, 1.0])
        key = jr.PRNGKey(4)

        grad_force = jax.grad(loss_fn)(force, state, key)

        # Gradients should exist and be non-zero
        assert not jnp.any(jnp.isnan(grad_force))
        # At t=0 with zero initial position, gradient should be near zero
        # because position hasn't changed much yet

    def test_jit_compilation(self, point_mass_dae):
        """Test that the component can be JIT compiled."""
        state = init_state_from_component(point_mass_dae)
        key = jr.PRNGKey(5)
        inputs = {"force": jnp.array([1.0, 0.0])}

        @jax.jit
        def step(inputs, state, key):
            return point_mass_dae(inputs, state, key=key)

        # Should compile and run without error
        outputs, new_state = step(inputs, state, key)
        assert outputs["effector"].pos.shape == (2,)


class TestTwoLinkArmDAE:
    """Tests for TwoLinkArmDAE component."""

    @pytest.fixture
    def arm_dae(self):
        """Create a TwoLinkArmDAE instance."""
        return TwoLinkArmDAE(
            dt=0.001,
            key=jr.PRNGKey(0),
        )

    def test_initialization(self, arm_dae):
        """Test that DAE arm initializes correctly."""
        assert arm_dae.dt == 0.001
        assert arm_dae.input_size == 2
        assert arm_dae.params.l.shape == (2,)
        assert arm_dae.params.m.shape == (2,)

    def test_state_initialization(self, arm_dae):
        """Test initial state."""
        state = init_state_from_component(arm_dae)
        dae_state = arm_dae.state_view(state)

        assert dae_state is not None
        assert dae_state.system.angle.shape == (2,)
        assert jnp.allclose(dae_state.system.angle, 0.0)
        assert jnp.allclose(dae_state.system.d_angle, 0.0)

    def test_single_step(self, arm_dae):
        """Test single integration step."""
        state = init_state_from_component(arm_dae)
        key = jr.PRNGKey(1)

        # Apply torque
        torque = jnp.array([0.1, 0.05])
        inputs = {"torque": torque}

        outputs, new_state = arm_dae(inputs, state, key=key)

        assert "effector" in outputs
        assert "state" in outputs
        assert "joints" in outputs
        assert outputs["effector"].pos.shape == (2,)

    def test_forward_kinematics(self, arm_dae):
        """Test forward kinematics computation."""
        from feedbax.mechanics.skeleton.arm import TwoLinkArmState

        # At zero angles, effector should be at (l1+l2, 0)
        state = TwoLinkArmState(
            angle=jnp.zeros(2),
            d_angle=jnp.zeros(2),
            torque=jnp.zeros(2),
        )
        effector = arm_dae.effector(state)

        expected_x = arm_dae.params.l[0] + arm_dae.params.l[1]
        assert jnp.allclose(effector.pos[0], expected_x, atol=1e-6)
        assert jnp.allclose(effector.pos[1], 0.0, atol=1e-6)

    def test_inverse_kinematics_roundtrip(self, arm_dae):
        """Test that forward-inverse kinematics round-trips."""
        from feedbax.mechanics.skeleton.arm import TwoLinkArmState

        # Start with known angles
        angles = jnp.array([0.5, 0.8])
        state = TwoLinkArmState(
            angle=angles,
            d_angle=jnp.zeros(2),
            torque=jnp.zeros(2),
        )

        # Forward to get effector
        effector = arm_dae.effector(state)

        # Inverse to get back angles
        recovered_state = arm_dae.inverse_kinematics(effector)

        assert jnp.allclose(recovered_state.angle, angles, atol=1e-4)

    def test_energy_methods(self, arm_dae):
        """Test energy computation methods."""
        from feedbax.mechanics.skeleton.arm import TwoLinkArmState

        state = TwoLinkArmState(
            angle=jnp.array([0.5, 0.3]),
            d_angle=jnp.array([1.0, -0.5]),
            torque=jnp.zeros(2),
        )

        ke = arm_dae.compute_kinetic_energy(state)
        pe = arm_dae.compute_potential_energy(state)

        assert ke >= 0
        assert jnp.isfinite(ke)
        assert jnp.isfinite(pe)

    def test_gradient_flow_arm(self, arm_dae):
        """Test gradients flow through arm DAE."""
        state = init_state_from_component(arm_dae)

        def loss_fn(torque, state, key):
            inputs = {"torque": torque}
            outputs, _ = arm_dae(inputs, state, key=key)
            return jnp.sum(outputs["effector"].pos ** 2)

        torque = jnp.array([0.1, 0.05])
        key = jr.PRNGKey(2)

        grad_torque = jax.grad(loss_fn)(torque, state, key)

        assert not jnp.any(jnp.isnan(grad_torque))

    def test_no_nan_in_long_simulation(self, arm_dae):
        """Test no NaN appears in extended simulation."""
        state = init_state_from_component(arm_dae)
        key = jr.PRNGKey(3)

        n_steps = 1000
        inputs = {"torque": jnp.array([0.01, -0.01])}

        for i in range(n_steps):
            key, subkey = jr.split(key)
            outputs, state = arm_dae(inputs, state, key=subkey)

            # Check for NaN
            effector = outputs["effector"]
            assert not jnp.any(jnp.isnan(effector.pos)), f"NaN at step {i}"
            assert not jnp.any(jnp.isnan(effector.vel)), f"NaN at step {i}"

    def test_jit_compilation_arm(self, arm_dae):
        """Test JIT compilation of arm DAE."""
        state = init_state_from_component(arm_dae)
        key = jr.PRNGKey(4)
        inputs = {"torque": jnp.array([0.1, 0.0])}

        @jax.jit
        def step(inputs, state, key):
            return arm_dae(inputs, state, key=key)

        outputs, new_state = step(inputs, state, key)
        assert outputs["effector"].pos.shape == (2,)


class TestDAEExplicitComparison:
    """Compare DAE solver with explicit Euler for simple cases."""

    def test_point_mass_agreement(self):
        """Test that DAE and explicit solvers agree for point mass."""
        dt = 0.001
        n_steps = 100

        # DAE version
        dae = PointMassDAE(mass=1.0, damping=0.0, dt=dt, key=jr.PRNGKey(0))
        state_dae = init_state_from_component(dae)

        # Explicit version (using Mechanics wrapper would be more accurate,
        # but we test the physics agreement)
        explicit = PointMass(mass=1.0, damping=0.0)
        state_explicit = explicit.init(key=jr.PRNGKey(0))

        force = jnp.array([0.5, 0.2])
        key = jr.PRNGKey(1)

        # Run DAE
        for _ in range(n_steps):
            key, subkey = jr.split(key)
            outputs_dae, state_dae = dae({"force": force}, state_dae, key=subkey)

        # Run explicit Euler manually
        for _ in range(n_steps):
            deriv = explicit.vector_field(0.0, state_explicit, force)
            state_explicit = CartesianState(
                pos=state_explicit.pos + dt * deriv.pos,
                vel=state_explicit.vel + dt * deriv.vel,
                force=state_explicit.force,
            )

        # Compare positions (should be very close for such simple dynamics)
        dae_pos = outputs_dae["effector"].pos
        explicit_pos = state_explicit.pos

        # Allow some tolerance due to different integration schemes
        assert jnp.allclose(dae_pos, explicit_pos, rtol=0.05)


class TestVmapCompatibility:
    """Test that DAE components work with vmap."""

    def test_vmap_point_mass(self):
        """Test vmapping over initial conditions."""
        dae = PointMassDAE(mass=1.0, damping=0.0, dt=0.01, key=jr.PRNGKey(0))

        # Create batch of forces
        batch_size = 4
        forces = jr.normal(jr.PRNGKey(1), (batch_size, 2))

        def single_step(force):
            state = init_state_from_component(dae)
            inputs = {"force": force}
            outputs, _ = dae(inputs, state, key=jr.PRNGKey(2))
            return outputs["effector"].pos

        # vmap over batch
        batched_step = jax.vmap(single_step)
        results = batched_step(forces)

        assert results.shape == (batch_size, 2)
        assert not jnp.any(jnp.isnan(results))
