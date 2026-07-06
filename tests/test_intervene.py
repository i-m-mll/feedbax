"""Tests for ``feedbax.intervene`` components.

Currently focused on ``DynamicsMatrixPerturb`` (Bug: c723082) — the
force-channel embedding of a model-class ``ΔA`` perturbation. The
``CurlField`` / ``FixedField`` paths are exercised via integration tests
in the rlrmp tree.
"""

import equinox as eqx
from equinox.nn import State
import jax.numpy as jnp
import jax.random as jr
import pytest

from feedbax.intervene import (
    AddNoise,
    AddNoiseParams,
    DynamicsMatrixPerturb,
    DynamicsMatrixPerturbParams,
)
from feedbax.runtime.state import CartesianState


def _make_state(component: DynamicsMatrixPerturb, params: DynamicsMatrixPerturbParams) -> State:
    """Construct an Equinox ``State`` container with the component's params set."""
    state = State(component)
    return state.set(component.params_index, params)


def _call(
    component: DynamicsMatrixPerturb,
    *,
    effector: CartesianState,
    force_in: jnp.ndarray,
    params: DynamicsMatrixPerturbParams,
):
    state = _make_state(component, params)
    inputs = {"effector": effector, "force": force_in}
    out, _ = component(inputs, state, key=jr.PRNGKey(0))
    return out["force"]


class TestDynamicsMatrixPerturb:
    def test_inactive_passes_force_through(self):
        comp = DynamicsMatrixPerturb(mass=1.0)
        eff = CartesianState(pos=jnp.array([0.5, -0.3]), vel=jnp.array([1.0, 0.7]))
        f_in = jnp.array([0.1, 0.2])
        # delta_A is non-trivial but active=False ⇒ no perturbation
        params = DynamicsMatrixPerturbParams(
            active=False,
            delta_A=jnp.array(
                [[0.5, 0.0, 0.0, 0.0], [0.0, 0.5, 0.0, 0.0]],
                dtype=jnp.float32,
            ),
        )
        f_out = _call(comp, effector=eff, force_in=f_in, params=params)
        assert jnp.allclose(f_out, f_in)

    def test_zero_delta_A_zero_perturbation(self):
        comp = DynamicsMatrixPerturb(mass=1.0)
        eff = CartesianState(pos=jnp.array([1.0, 2.0]), vel=jnp.array([3.0, 4.0]))
        f_in = jnp.array([0.5, -0.5])
        params = DynamicsMatrixPerturbParams(
            active=True,
            delta_A=jnp.zeros((2, 4), dtype=jnp.float32),
        )
        f_out = _call(comp, effector=eff, force_in=f_in, params=params)
        assert jnp.allclose(f_out, f_in)

    def test_velocity_proportional_perturbation(self):
        # delta_A maps [pos, vel] → derivative perturbation; pick a known
        # diagonal so the analytical answer is trivial.
        mass = 2.0
        comp = DynamicsMatrixPerturb(mass=mass)
        # delta_A = [[0,0,k,0],[0,0,0,k]] ⇒ Δ(dot v) = k * vel
        # ⇒ Δf = mass * k * vel
        k = 0.7
        delta_A = jnp.array([[0.0, 0.0, k, 0.0], [0.0, 0.0, 0.0, k]], dtype=jnp.float32)
        eff = CartesianState(pos=jnp.array([0.0, 0.0]), vel=jnp.array([1.0, -2.0]))
        f_in = jnp.zeros(2)
        params = DynamicsMatrixPerturbParams(
            active=True,
            scale=1.0,
            delta_A=delta_A,
        )
        f_out = _call(comp, effector=eff, force_in=f_in, params=params)
        expected = mass * k * jnp.array([1.0, -2.0])
        assert jnp.allclose(f_out, expected)

    def test_scale_multiplies_perturbation(self):
        comp = DynamicsMatrixPerturb(mass=1.0)
        delta_A = jnp.array([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=jnp.float32)
        eff = CartesianState(pos=jnp.zeros(2), vel=jnp.array([2.0, 3.0]))
        f_in = jnp.zeros(2)
        # scale = 0.5 → half the perturbation
        p_full = DynamicsMatrixPerturbParams(active=True, scale=1.0, delta_A=delta_A)
        p_half = DynamicsMatrixPerturbParams(active=True, scale=0.5, delta_A=delta_A)
        f_full = _call(comp, effector=eff, force_in=f_in, params=p_full)
        f_half = _call(comp, effector=eff, force_in=f_in, params=p_half)
        assert jnp.allclose(f_half, 0.5 * f_full)

    def test_jit_traceable(self):
        # Smoke test: the component must be JIT-able.
        comp = DynamicsMatrixPerturb(mass=1.0)
        delta_A = jnp.eye(2, 4, dtype=jnp.float32)
        eff = CartesianState(pos=jnp.array([0.0, 0.0]), vel=jnp.array([1.0, 0.5]))
        f_in = jnp.zeros(2)
        params = DynamicsMatrixPerturbParams(active=True, delta_A=delta_A)

        @eqx.filter_jit
        def run(eff_, f_, p_):
            return _call(comp, effector=eff_, force_in=f_, params=p_)

        f_out = run(eff, f_in, params)
        assert f_out.shape == (2,)

    def test_mass_must_be_explicit_and_positive(self):
        with pytest.raises(ValueError, match="requires an explicit positive mass"):
            DynamicsMatrixPerturb()
        with pytest.raises(ValueError, match="must be finite and positive"):
            DynamicsMatrixPerturb(mass=0.0)


def test_add_noise_splits_key_per_signal_leaf() -> None:
    component = AddNoise(params=AddNoiseParams(active=True))
    state = State(component)
    signal = {
        "left": jnp.zeros((4,), dtype=jnp.float32),
        "right": jnp.zeros((4,), dtype=jnp.float32),
    }

    outputs, _ = component({"input": signal}, state, key=jr.PRNGKey(0))

    assert not jnp.allclose(outputs["output"]["left"], outputs["output"]["right"])
