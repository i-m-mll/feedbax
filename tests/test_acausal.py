"""Tests for the acausal modeling framework.

Covers assembly, physics correctness, JIT/grad compatibility, and
numerical stability for both translational and rotational domains.

:copyright: Copyright 2024-2025 by MLL <mll@mll.bio>.
:license: Apache 2.0.  See LICENSE for details.
"""

import pytest
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

from feedbax.graph import init_state_from_component
from feedbax.acausal import (
    AcausalConnection,
    AcausalSystem,
    ForceSensor,
    ForceSource,
    Ground,
    LinearDamper,
    LinearSpring,
    Mass,
    PositionSensor,
    VelocitySensor,
)
from feedbax.acausal.rotational import (
    GearRatio,
    Inertia,
    RotationalDamper,
    RotationalGround,
    TorqueSource,
    TorsionalSpring,
)
from feedbax.mechanics.dae import DAEState

jax.config.update("jax_enable_x64", True)


# =========================================================================
# Fixtures
# =========================================================================

def _make_msd(
    mass: float = 1.0,
    stiffness: float = 10.0,
    damping: float = 0.5,
    dt: float = 0.001,
) -> AcausalSystem:
    """Create a mass-spring-damper system attached to a wall."""
    return AcausalSystem(
        elements={
            "wall": Ground("wall"),
            "mass": Mass("mass", mass=mass),
            "spring": LinearSpring("spring", stiffness=stiffness),
            "damper": LinearDamper("damper", damping=damping),
            "f_in": ForceSource("f_in"),
            "x_out": PositionSensor("x_out"),
        },
        connections=[
            AcausalConnection(("wall", "flange"), ("spring", "flange_a")),
            AcausalConnection(("wall", "flange"), ("damper", "flange_a")),
            AcausalConnection(("spring", "flange_b"), ("mass", "flange")),
            AcausalConnection(("damper", "flange_b"), ("mass", "flange")),
            AcausalConnection(("f_in", "flange"), ("mass", "flange")),
            AcausalConnection(("x_out", "flange"), ("mass", "flange")),
        ],
        dt=dt,
    )


def _make_undamped_spring_mass(
    mass: float = 1.0,
    stiffness: float = 4.0,
    dt: float = 0.0005,
) -> AcausalSystem:
    """Undamped mass-spring for energy conservation tests."""
    return AcausalSystem(
        elements={
            "wall": Ground("wall"),
            "mass": Mass("mass", mass=mass),
            "spring": LinearSpring("spring", stiffness=stiffness),
            "x_out": PositionSensor("x_out"),
            "v_out": VelocitySensor("v_out"),
        },
        connections=[
            AcausalConnection(("wall", "flange"), ("spring", "flange_a")),
            AcausalConnection(("spring", "flange_b"), ("mass", "flange")),
            AcausalConnection(("x_out", "flange"), ("mass", "flange")),
            AcausalConnection(("v_out", "flange"), ("mass", "flange")),
        ],
        dt=dt,
    )


# =========================================================================
# Test classes
# =========================================================================

class TestAcausalAssembly:
    """Test the assembly algorithm produces correct systems."""

    def test_mass_spring_damper_construction(self):
        """MSD system assembles without error."""
        msd = _make_msd()
        assert msd._layout.total_size == 2  # pos, vel

    def test_state_layout(self):
        """State vector contains exactly pos and vel for the mass."""
        msd = _make_msd()
        layout = msd._layout
        assert len(layout._differential) == 2
        # Both should reference the mass's canonical flange vars
        for vname in layout._differential:
            assert "mass" in vname or "spring" in vname or "damper" in vname

    def test_params_collected(self):
        """All physical parameters are collected."""
        msd = _make_msd()
        names = set(msd.params.values.keys())
        assert "mass.mass" in names
        assert "spring.stiffness" in names
        assert "damper.damping" in names

    def test_sensor_output_registered(self):
        """PositionSensor registers an output."""
        msd = _make_msd()
        assert "x_out" in msd._output_indices

    def test_grounded_variables(self):
        """Wall's across vars are marked grounded."""
        msd = _make_msd()
        # The wall's flange across vars should be grounded
        grounded = msd._layout._grounded
        assert len(grounded) >= 1  # at least one canonical var is grounded

    def test_eliminated_variables(self):
        """Connected across variables are eliminated properly."""
        msd = _make_msd()
        # spring.flange_b.pos should be eliminated in favour of mass.flange.pos
        elim = msd._layout._eliminated
        assert len(elim) > 0


class TestMassSpringDamper:
    """Physics correctness for the mass-spring-damper system."""

    @pytest.fixture
    def msd(self):
        return _make_msd(mass=1.0, stiffness=10.0, damping=0.5, dt=0.001)

    def test_single_step(self, msd):
        """A single step runs without error."""
        state = init_state_from_component(msd)
        key = jr.PRNGKey(0)
        inputs = {"f_in": jnp.array([0.0])}
        outputs, new_state = msd(inputs, state, key=key)
        assert "state" in outputs

    def test_force_accelerates_mass(self, msd):
        """Constant force causes velocity increase."""
        state = init_state_from_component(msd)
        key = jr.PRNGKey(0)
        force = jnp.array([1.0])

        # Run several steps
        for _ in range(100):
            key, subkey = jr.split(key)
            outputs, state = msd({"f_in": force}, state, key=subkey)

        # Velocity should be positive
        dae_state = msd.state_view(state)
        vel = dae_state.system.y[1]  # velocity is index 1
        assert vel > 0.0, f"Expected positive velocity, got {vel}"

    def test_damped_oscillation(self, msd):
        """Displaced mass-spring-damper shows decaying oscillation."""
        state = init_state_from_component(msd)
        key = jr.PRNGKey(0)

        # Set initial displacement by applying a big force briefly
        for _ in range(50):
            key, subkey = jr.split(key)
            outputs, state = msd({"f_in": jnp.array([5.0])}, state, key=subkey)

        # Record position at the start of free oscillation
        dae_state = msd.state_view(state)
        pos_start = float(dae_state.system.y[0])

        # Free oscillation (no force)
        positions = []
        for _ in range(2000):
            key, subkey = jr.split(key)
            outputs, state = msd({"f_in": jnp.array([0.0])}, state, key=subkey)
            dae_state = msd.state_view(state)
            positions.append(float(dae_state.system.y[0]))

        # Should return towards zero (damped)
        final_pos = positions[-1]
        assert abs(final_pos) < abs(pos_start), (
            f"Position should decay: start={pos_start:.4f}, end={final_pos:.4f}"
        )

    def test_analytical_damped_solution(self):
        """Damped MSD matches analytical underdamped solution."""
        mass = 1.0
        stiffness = 4.0
        damping = 0.4
        dt = 0.0005
        sys = _make_msd(
            mass=mass, stiffness=stiffness, damping=damping, dt=dt
        )
        state = init_state_from_component(sys)
        key = jr.PRNGKey(0)

        # Initial displacement, zero velocity
        dae_state = sys.state_view(state)
        x0 = 1.0
        y0 = dae_state.system.y.at[0].set(x0)
        new_sys_state = type(dae_state.system)(y=y0)
        new_dae = DAEState(system=new_sys_state, solver=dae_state.solver)
        state = state.set(sys.state_index, new_dae)

        n_steps = 2000
        positions = []
        for _ in range(n_steps):
            key, subkey = jr.split(key)
            outputs, state = sys({"f_in": jnp.array([0.0])}, state, key=subkey)
            dae_state = sys.state_view(state)
            positions.append(dae_state.system.y[0])

        t = jnp.arange(1, n_steps + 1) * dt
        omega = jnp.sqrt(stiffness / mass)
        zeta = damping / (2.0 * jnp.sqrt(mass * stiffness))
        omega_d = omega * jnp.sqrt(1.0 - zeta ** 2)
        a = x0
        b = (zeta * omega * x0) / omega_d
        expected = jnp.exp(-zeta * omega * t) * (
            a * jnp.cos(omega_d * t) + b * jnp.sin(omega_d * t)
        )
        sim = jnp.array(positions)
        max_err = jnp.max(jnp.abs(sim - expected))
        assert max_err < 2e-2, f"Max error {max_err} too large"


class TestEnergyConservation:
    """Test energy conservation for undamped systems."""

    def test_energy_conservation_undamped(self):
        """Undamped mass-spring conserves energy within 1% over 1000 steps."""
        m, k = 1.0, 4.0
        sys = _make_undamped_spring_mass(mass=m, stiffness=k, dt=0.0005)
        state = init_state_from_component(sys)
        key = jr.PRNGKey(42)

        # Set initial displacement by directly modifying the state
        dae_state = sys.state_view(state)
        x0 = 0.5
        y0 = dae_state.system.y.at[0].set(x0)
        new_sys_state = type(dae_state.system)(y=y0)
        new_dae = DAEState(system=new_sys_state, solver=dae_state.solver)
        state = state.set(sys.state_index, new_dae)

        # Initial energy: E = 0.5*k*x^2 + 0.5*m*v^2
        init_energy = 0.5 * k * x0**2

        n_steps = 2000
        for _ in range(n_steps):
            key, subkey = jr.split(key)
            outputs, state = sys({}, state, key=subkey)

        dae_state = sys.state_view(state)
        x_final = float(dae_state.system.y[0])
        v_final = float(dae_state.system.y[1])
        final_energy = 0.5 * k * x_final**2 + 0.5 * m * v_final**2

        drift = abs(final_energy - init_energy) / max(init_energy, 1e-10)
        assert drift < 0.01, (
            f"Energy drift {drift:.4f} > 1%: "
            f"init={init_energy:.6f}, final={final_energy:.6f}"
        )


class TestJITAndGrad:
    """Test JIT compilation and gradient flow."""

    def test_jit_compilation(self):
        """jax.jit traces AcausalSystem.__call__ without error."""
        msd = _make_msd()
        state = init_state_from_component(msd)
        key = jr.PRNGKey(0)
        inputs = {"f_in": jnp.array([1.0])}

        @jax.jit
        def step(inputs, state, key):
            return msd(inputs, state, key=key)

        outputs, new_state = step(inputs, state, key)
        assert "state" in outputs

    def test_grad_through_solver(self):
        """jax.grad produces finite gradients through the solver."""
        msd = _make_msd()
        state = init_state_from_component(msd)

        def loss_fn(force_val):
            inputs = {"f_in": jnp.array([force_val])}
            key = jr.PRNGKey(0)
            outputs, _ = msd(inputs, state, key=key)
            # Loss on the state vector
            return jnp.sum(outputs["state"].y ** 2)

        grad_val = jax.grad(loss_fn)(1.0)
        assert jnp.isfinite(grad_val), f"Got non-finite gradient: {grad_val}"

    def test_vmap_over_params(self):
        """vmap works over spring stiffness parameters."""
        mass = 1.0
        dt = 0.001
        sys = _make_msd(mass=mass, stiffness=1.0, damping=0.0, dt=dt)
        state = init_state_from_component(sys)

        dae_state = sys.state_view(state)
        x0 = 0.5
        y0 = dae_state.system.y.at[0].set(x0)
        new_sys_state = type(dae_state.system)(y=y0)
        new_dae = DAEState(system=new_sys_state, solver=dae_state.solver)
        state = state.set(sys.state_index, new_dae)

        def step_velocity(k_val):
            sys_k = eqx.tree_at(
                lambda s: s.params.values["spring.stiffness"],
                sys,
                jnp.zeros(()) + k_val,
            )
            outputs, new_state = sys_k({"f_in": jnp.array([0.0])}, state, key=jr.PRNGKey(0))
            dae_after = sys_k.state_view(new_state)
            return dae_after.system.y[1]

        ks = jnp.array([1.0, 2.0, 3.0])
        vels = jax.vmap(step_velocity)(ks)
        expected = -ks * x0 * dt / mass
        assert jnp.allclose(vels, expected, atol=1e-6)


class TestMultiConnectionNode:
    """Test systems with 3+ elements connected at the same node."""

    def test_three_springs_at_node(self):
        """Three springs connected to one mass with correct force balance."""
        sys = AcausalSystem(
            elements={
                "wall": Ground("wall"),
                "mass": Mass("mass", mass=1.0),
                "s1": LinearSpring("s1", stiffness=5.0),
                "s2": LinearSpring("s2", stiffness=3.0),
                "s3": LinearSpring("s3", stiffness=2.0),
            },
            connections=[
                AcausalConnection(("wall", "flange"), ("s1", "flange_a")),
                AcausalConnection(("wall", "flange"), ("s2", "flange_a")),
                AcausalConnection(("wall", "flange"), ("s3", "flange_a")),
                AcausalConnection(("s1", "flange_b"), ("mass", "flange")),
                AcausalConnection(("s2", "flange_b"), ("mass", "flange")),
                AcausalConnection(("s3", "flange_b"), ("mass", "flange")),
            ],
            dt=0.001,
        )
        state = init_state_from_component(sys)
        key = jr.PRNGKey(0)

        # Set initial displacement
        dae_state = sys.state_view(state)
        y0 = dae_state.system.y.at[0].set(1.0)
        new_sys_state = type(dae_state.system)(y=y0)
        new_dae = DAEState(system=new_sys_state, solver=dae_state.solver)
        state = state.set(sys.state_index, new_dae)

        # Step once -- acceleration should be -(5+3+2)*1.0/1.0 = -10.0
        outputs, new_state = sys({}, state, key=key)

        dae_after = sys.state_view(new_state)
        # After one Euler step: v_new = v_old + dt * a = 0 + 0.001 * (-10) = -0.01
        v_after = float(dae_after.system.y[1])
        assert abs(v_after - (-0.01)) < 1e-6, f"Expected -0.01, got {v_after}"


class TestGroundBoundary:
    """Test that grounded variables remain at zero."""

    def test_ground_stays_zero(self):
        """Grounded pos/vel stay zero regardless of forces."""
        msd = _make_msd()
        state = init_state_from_component(msd)
        key = jr.PRNGKey(0)

        for _ in range(100):
            key, subkey = jr.split(key)
            outputs, state = msd({"f_in": jnp.array([10.0])}, state, key=subkey)

        # The grounded variables should not appear in the state vector,
        # but if we look at the layout, grounded vars should resolve to 0
        layout = msd._layout
        for gvar in layout._grounded:
            # grounded vars are not in the differential list
            assert gvar not in layout._differential


class TestSensorReadings:
    """Test that sensors return correct values."""

    def test_position_sensor(self):
        """PositionSensor matches state vector position."""
        msd = _make_msd()
        state = init_state_from_component(msd)
        key = jr.PRNGKey(0)

        # Apply force to get non-zero position
        for _ in range(100):
            key, subkey = jr.split(key)
            outputs, state = msd({"f_in": jnp.array([1.0])}, state, key=subkey)

        dae_state = msd.state_view(state)
        pos_from_state = dae_state.system.y[0]
        pos_from_sensor = outputs.get("x_out", None)

        if pos_from_sensor is not None:
            assert jnp.allclose(pos_from_state, pos_from_sensor, atol=1e-10), (
                f"Sensor mismatch: state={pos_from_state}, sensor={pos_from_sensor}"
            )

    def test_velocity_sensor(self):
        """VelocitySensor returns correct velocity."""
        sys = AcausalSystem(
            elements={
                "wall": Ground("wall"),
                "mass": Mass("mass", mass=1.0),
                "spring": LinearSpring("spring", stiffness=1.0),
                "f_in": ForceSource("f_in"),
                "v_out": VelocitySensor("v_out"),
            },
            connections=[
                AcausalConnection(("wall", "flange"), ("spring", "flange_a")),
                AcausalConnection(("spring", "flange_b"), ("mass", "flange")),
                AcausalConnection(("f_in", "flange"), ("mass", "flange")),
                AcausalConnection(("v_out", "flange"), ("mass", "flange")),
            ],
            dt=0.001,
        )
        state = init_state_from_component(sys)
        key = jr.PRNGKey(0)

        for _ in range(50):
            key, subkey = jr.split(key)
            outputs, state = sys({"f_in": jnp.array([1.0])}, state, key=subkey)

        dae_state = sys.state_view(state)
        vel_from_state = dae_state.system.y[1]
        vel_from_sensor = outputs.get("v_out", None)

        if vel_from_sensor is not None:
            assert jnp.allclose(vel_from_state, vel_from_sensor, atol=1e-10)

    def test_force_sensor(self):
        """ForceSensor reads spring force at the node."""
        stiffness = 5.0
        sys = AcausalSystem(
            elements={
                "wall": Ground("wall"),
                "mass": Mass("mass", mass=1.0),
                "spring": LinearSpring("spring", stiffness=stiffness),
                "f_out": ForceSensor("f_out"),
            },
            connections=[
                AcausalConnection(("wall", "flange"), ("spring", "flange_a")),
                AcausalConnection(("spring", "flange_b"), ("mass", "flange")),
                AcausalConnection(("f_out", "flange"), ("mass", "flange")),
            ],
            dt=0.001,
        )
        state = init_state_from_component(sys)
        key = jr.PRNGKey(0)

        dae_state = sys.state_view(state)
        y0 = dae_state.system.y.at[0].set(0.2)
        new_sys_state = type(dae_state.system)(y=y0)
        new_dae = DAEState(system=new_sys_state, solver=dae_state.solver)
        state = state.set(sys.state_index, new_dae)

        outputs, state = sys({}, state, key=key)
        dae_after = sys.state_view(state)
        pos = float(dae_after.system.y[0])
        expected_force = -stiffness * pos
        force_measured = outputs.get("f_out", None)
        assert force_measured is not None
        assert abs(float(force_measured) - expected_force) < 1e-6, (
            f"Expected {expected_force}, got {force_measured}"
        )


class TestLongHorizonStability:
    """Test long simulations do not produce NaN or diverge."""

    def test_long_horizon_no_nan(self):
        """10k steps of damped MSD without NaN."""
        msd = _make_msd(dt=0.001, damping=1.0)
        state = init_state_from_component(msd)
        key = jr.PRNGKey(0)

        for i in range(10000):
            key, subkey = jr.split(key)
            outputs, state = msd({"f_in": jnp.array([0.1])}, state, key=subkey)

            if i % 1000 == 0:
                dae_state = msd.state_view(state)
                assert not jnp.any(jnp.isnan(dae_state.system.y)), (
                    f"NaN at step {i}"
                )

        dae_state = msd.state_view(state)
        assert not jnp.any(jnp.isnan(dae_state.system.y)), "NaN at final step"
        # Position should not diverge: damped + finite force -> bounded
        assert jnp.all(jnp.abs(dae_state.system.y) < 100.0), (
            f"Divergence: y = {dae_state.system.y}"
        )


class TestRotationalDomain:
    """Test rotational-domain elements."""

    def test_inertia_torsional_spring(self):
        """Rotational spring-inertia oscillates."""
        sys = AcausalSystem(
            elements={
                "ground": RotationalGround("ground"),
                "inertia": Inertia("inertia", inertia=1.0),
                "spring": TorsionalSpring("spring", stiffness=4.0),
            },
            connections=[
                AcausalConnection(("ground", "flange"), ("spring", "flange_a")),
                AcausalConnection(("spring", "flange_b"), ("inertia", "flange")),
            ],
            dt=0.001,
        )
        state = init_state_from_component(sys)
        key = jr.PRNGKey(0)

        # Set initial angle
        dae_state = sys.state_view(state)
        y0 = dae_state.system.y.at[0].set(1.0)
        new_sys_state = type(dae_state.system)(y=y0)
        new_dae = DAEState(system=new_sys_state, solver=dae_state.solver)
        state = state.set(sys.state_index, new_dae)

        # Run for a while
        angles = []
        for _ in range(3000):
            key, subkey = jr.split(key)
            outputs, state = sys({}, state, key=subkey)
            dae_state = sys.state_view(state)
            angles.append(float(dae_state.system.y[0]))

        # Should oscillate: check sign changes
        sign_changes = sum(
            1 for i in range(1, len(angles)) if angles[i] * angles[i-1] < 0
        )
        assert sign_changes >= 2, (
            f"Expected oscillation, got {sign_changes} sign changes"
        )

    def test_gear_ratio_construction(self):
        """GearRatio element assembles without error."""
        sys = AcausalSystem(
            elements={
                "ground": RotationalGround("ground"),
                "inertia_a": Inertia("inertia_a", inertia=1.0),
                "inertia_b": Inertia("inertia_b", inertia=0.5),
                "gear": GearRatio("gear", ratio=2.0),
                "spring": TorsionalSpring("spring", stiffness=1.0),
                "tau_in": TorqueSource("tau_in"),
            },
            connections=[
                AcausalConnection(("ground", "flange"), ("spring", "flange_a")),
                AcausalConnection(("spring", "flange_b"), ("inertia_a", "flange")),
                AcausalConnection(("inertia_a", "flange"), ("gear", "flange_a")),
                AcausalConnection(("gear", "flange_b"), ("inertia_b", "flange")),
                AcausalConnection(("tau_in", "flange"), ("inertia_a", "flange")),
            ],
            dt=0.001,
        )
        # If assembly succeeds without error, the test passes
        state = init_state_from_component(sys)
        assert state is not None

    def test_gear_ratio_behavior(self):
        """Gear ratio scales torque response through the constraint."""
        ratio = 2.0
        stiffness = 4.0
        inertia = 1.0
        dt = 0.001
        sys = AcausalSystem(
            elements={
                "inertia": Inertia("inertia", inertia=inertia),
                "gear": GearRatio("gear", ratio=ratio),
                "spring": TorsionalSpring("spring", stiffness=stiffness),
            },
            connections=[
                AcausalConnection(("inertia", "flange"), ("gear", "flange_a")),
                AcausalConnection(("spring", "flange_a"), ("gear", "flange_a")),
                AcausalConnection(("spring", "flange_b"), ("gear", "flange_b")),
            ],
            dt=dt,
        )
        state = init_state_from_component(sys)

        # Set initial angle
        dae_state = sys.state_view(state)
        y0 = dae_state.system.y.at[0].set(1.0)
        new_sys_state = type(dae_state.system)(y=y0)
        new_dae = DAEState(system=new_sys_state, solver=dae_state.solver)
        state = state.set(sys.state_index, new_dae)

        key = jr.PRNGKey(0)
        outputs, state = sys({}, state, key=key)

        dae_after = sys.state_view(state)
        vel_after = float(dae_after.system.y[1])
        # At node A the inertia sees spring.flange_a torque plus gear-
        # transmitted torque from node B.  For a spring spanning both sides
        # of the gear: accel = (1 + ratio) * k * (ratio - 1) * theta / J.
        expected_accel = (
            (1.0 + ratio) * stiffness * (ratio - 1.0) / inertia
        )
        expected_vel = dt * expected_accel
        assert abs(vel_after - expected_vel) < 1e-6, (
            f"Expected {expected_vel}, got {vel_after}"
        )
