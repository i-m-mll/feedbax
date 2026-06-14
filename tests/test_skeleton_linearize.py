"""Tests for ``AbstractSkeleton.linearize`` and ``Mechanics.linearize_with_force_filter``.

Verifies that the closed-form ``PointMass.linearize`` matches the bare-state
matrices used in ``rlrmp.analysis.hinf_riccati.linearize_pointmass`` and that
``Mechanics.linearize_with_force_filter`` augments correctly with a first-order
force filter. Bug: 488042f.
"""

import jax.numpy as jnp
import pytest

from feedbax.mechanics.dynamics import LinearSystem
from feedbax.mechanics.mechanics import Mechanics
from feedbax.mechanics.plant import DirectForceInput
from feedbax.mechanics.skeleton import PointMass


def _ref_pointmass_continuous(mass: float, damping: float):
    """Reference closed-form continuous matrices for the bare 4-state pointmass."""
    I2 = jnp.eye(2, dtype=jnp.float32)
    Z2 = jnp.zeros((2, 2), dtype=jnp.float32)
    A_c = jnp.block([[Z2, I2], [Z2, -(damping / mass) * I2]])
    B_c = jnp.concatenate([Z2, I2 / mass], axis=0)
    Bw_c = B_c
    return A_c, B_c, Bw_c


def _ref_pointmass_with_filter(mass: float, damping: float, tau: float):
    """Reference closed-form continuous matrices for the 6-state augmented plant."""
    I2 = jnp.eye(2, dtype=jnp.float32)
    Z2 = jnp.zeros((2, 2), dtype=jnp.float32)
    A_c = jnp.block(
        [
            [Z2, I2, Z2],
            [Z2, -(damping / mass) * I2, I2 / mass],
            [Z2, Z2, -(1.0 / tau) * I2],
        ]
    )
    B_c = jnp.concatenate([Z2, Z2, I2 / tau], axis=0)
    Bw_c = jnp.concatenate([Z2, I2 / mass, Z2], axis=0)
    return A_c, B_c, Bw_c


class TestPointMassLinearize:
    @pytest.mark.parametrize("mass,damping", [(1.0, 0.0), (1.0, 10.0), (0.5, 4.0)])
    def test_bare_skeleton_matches_closed_form(self, mass, damping):
        pm = PointMass(mass=mass, damping=damping)
        sys = pm.linearize(state=None)
        assert isinstance(sys, LinearSystem)
        A_ref, B_ref, Bw_ref = _ref_pointmass_continuous(mass, damping)

        assert jnp.allclose(sys.A, A_ref)
        assert jnp.allclose(sys.B, B_ref)
        assert jnp.allclose(sys.B_w, Bw_ref)
        assert sys.dt is None
        assert sys.state_indices == {"pos": (0, 2), "vel": (2, 4)}

    def test_linearize_ignores_state(self):
        pm = PointMass(mass=1.0, damping=10.0)
        sys_a = pm.linearize(state=None)
        # Pass an arbitrary CartesianState; result must be identical
        from feedbax.runtime.state import CartesianState
        nominal = CartesianState(pos=jnp.array([1.0, 2.0]), vel=jnp.array([0.5, -0.3]))
        sys_b = pm.linearize(state=nominal)
        assert jnp.allclose(sys_a.A, sys_b.A)
        assert jnp.allclose(sys_a.B, sys_b.B)


class TestMechanicsLinearizeWithForceFilter:
    def _make_mechanics(self, mass=1.0, damping=10.0, dt=0.01) -> Mechanics:
        pm = PointMass(mass=mass, damping=damping)
        plant = DirectForceInput(skeleton=pm)
        return Mechanics(plant=plant, dt=dt)

    @pytest.mark.parametrize("mass,damping,dt", [(1.0, 10.0, 0.01), (0.5, 4.0, 0.02)])
    def test_no_filter_preserves_bare_skeleton(self, mass, damping, dt):
        mech = self._make_mechanics(mass=mass, damping=damping, dt=dt)
        sys = mech.linearize_with_force_filter(skeleton_state=None, tau=None)
        A_ref, B_ref, Bw_ref = _ref_pointmass_continuous(mass, damping)
        assert jnp.allclose(sys.A, A_ref)
        assert jnp.allclose(sys.B, B_ref)
        assert jnp.allclose(sys.B_w, Bw_ref)
        assert sys.dt == pytest.approx(dt)

    @pytest.mark.parametrize(
        "mass,damping,tau,dt",
        [(1.0, 10.0, 0.05, 0.01), (0.5, 4.0, 0.1, 0.02)],
    )
    def test_with_filter_matches_hinf_riccati_form(self, mass, damping, tau, dt):
        mech = self._make_mechanics(mass=mass, damping=damping, dt=dt)
        sys = mech.linearize_with_force_filter(skeleton_state=None, tau=tau)
        A_ref, B_ref, Bw_ref = _ref_pointmass_with_filter(mass, damping, tau)
        assert sys.A.shape == (6, 6)
        assert sys.B.shape == (6, 2)
        assert sys.B_w.shape == (6, 2)
        assert jnp.allclose(sys.A, A_ref)
        assert jnp.allclose(sys.B, B_ref)
        assert jnp.allclose(sys.B_w, Bw_ref)
        assert sys.state_indices == {
            "pos": (0, 2),
            "vel": (2, 4),
            "force": (4, 6),
        }
        assert sys.dt == pytest.approx(dt)

    def test_tau_zero_equivalent_to_none(self):
        mech = self._make_mechanics()
        sys_none = mech.linearize_with_force_filter(skeleton_state=None, tau=None)
        sys_zero = mech.linearize_with_force_filter(skeleton_state=None, tau=0.0)
        assert jnp.allclose(sys_none.A, sys_zero.A)
        assert jnp.allclose(sys_none.B, sys_zero.B)
        assert jnp.allclose(sys_none.B_w, sys_zero.B_w)
