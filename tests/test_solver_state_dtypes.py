"""Sampled ODE integration preserves physical storage and responds to changed inputs."""

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree as jt
import numpy as np
import pytest

from feedbax.mechanics import DiffraxBackend, Mechanics
from feedbax.mechanics.dae import DAEComponent
from feedbax.mechanics.plant import DirectForceInput
from feedbax.mechanics.skeleton.pointmass import PointMass
from feedbax.mechanics.skeleton.pointmass_dae import PointMassDAE
from feedbax.runtime.filters import FirstOrderFilter
from feedbax.runtime.state import CartesianState
from feedbax.training.rl.env import RLEnvConfig, _physics_substep


pytestmark = pytest.mark.usefixtures("enable_jax_x64")


@pytest.mark.parametrize("solver_type", [dfx.Euler, dfx.Heun, dfx.Tsit5])
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_filter_sampled_response_and_reset(solver_type, mixed, compiled):
    prototype = {
        "x": jnp.zeros(2, dtype=jnp.float32 if mixed else jnp.float64),
        "y": jnp.zeros(3, dtype=jnp.float64),
    }
    model = (
        FirstOrderFilter(
            tau_rise=0.2, tau_decay=0.3, dt=0.001, solver_type=solver_type, init_value=2
        )
        .change_input(prototype)
        .prepare()
    )
    state = model.init_state(key=jax.random.PRNGKey(0))
    step = eqx.filter_jit(model) if compiled else model
    for target, tau in [(0.0, 0.3), (4.0, 0.2)]:
        previous = state.get(model.state_index).output
        inputs = jt.map(lambda x: jnp.full_like(x, target), prototype)
        output, state = step({"input": inputs}, state, key=jax.random.PRNGKey(0))
        for name, value in output["output"].items():
            expected = target + (previous[name] - target) * np.exp(-model.dt / tau)
            np.testing.assert_allclose(
                value, expected, atol=3e-5 if solver_type is dfx.Euler else 1e-7
            )
            assert value.dtype == prototype[name].dtype
    reset = model.init_state(key=jax.random.PRNGKey(1)).get(model.state_index).output
    for name, value in reset.items():
        assert value.dtype == prototype[name].dtype
        np.testing.assert_array_equal(value, jnp.full_like(prototype[name], 2))


class MixedPointMassDAE(PointMassDAE):
    def init_system_state(self, *, key):
        return CartesianState(
            pos=jnp.zeros(2, dtype=jnp.float32),
            vel=jnp.zeros(2, dtype=jnp.float64),
            force=jnp.zeros(2, dtype=jnp.float32),
        )


class MixedPointMass(PointMass):
    def init(self, *, key):
        return CartesianState(
            pos=jnp.zeros(2, dtype=jnp.float32),
            vel=jnp.zeros(2, dtype=jnp.float64),
            force=jnp.zeros(2, dtype=jnp.float32),
        )


@pytest.mark.parametrize("solver_type", [dfx.Euler, dfx.Heun, dfx.Tsit5])
def test_dae_mixed_storage_and_changed_force(solver_type):
    model = MixedPointMassDAE(dt=0.1, solver_type=solver_type).prepare()
    state = model.init_state(key=jax.random.PRNGKey(0))
    step = eqx.filter_jit(model)
    for force in [0.0, 1.0, -1.0]:
        output, state = step({"force": jnp.full(2, force)}, state, key=jax.random.PRNGKey(0))
        physical = output["state"]
        assert physical.pos.dtype == jnp.float32
        assert physical.vel.dtype == jnp.float64
        assert physical.force.dtype == jnp.float32
        np.testing.assert_allclose(physical.vel, 0.1 if force == 1 else 0, atol=1e-12)
    if solver_type is not dfx.Euler:
        np.testing.assert_allclose(physical.pos, 0.01, atol=1e-8)


@pytest.mark.parametrize("backend", ["legacy", "single", "substeps", "remat"])
def test_fresh_mechanics_stateful_solver_jit_and_gradient(backend):
    def response(force):
        plant = DirectForceInput(MixedPointMass(mass=1.0), clip_states=False)
        model = Mechanics(
            plant,
            dt=0.1,
            solver_type=dfx.Tsit5,
            backend=(
                None
                if backend == "legacy"
                else DiffraxBackend(
                    0.1,
                    sub_dt=0.05 if backend in ("substeps", "remat") else 0.1,
                    solver=dfx.Tsit5(),
                )
            ),
            remat_substep=backend == "remat",
        ).prepare()
        state = model.init_state(key=jax.random.PRNGKey(0))
        _, state = model({"force": jnp.zeros(2)}, state, key=jax.random.PRNGKey(0))
        output, state = model({"force": jnp.full(2, force)}, state, key=jax.random.PRNGKey(0))
        physical = output["state"].plant.skeleton
        assert physical.pos.dtype == jnp.float32
        assert physical.vel.dtype == jnp.float64
        return physical.vel[0], physical.pos[0]

    (value, position), grad = jax.jit(jax.value_and_grad(response, has_aux=True))(jnp.array(1.0))
    np.testing.assert_allclose(value, 0.1, atol=1e-12)
    np.testing.assert_allclose(grad, 0.1, atol=1e-12)
    np.testing.assert_allclose(position, 0.005, atol=1e-8)


def test_rl_euler_helper_keeps_mixed_storage_in_scan():
    plant = DirectForceInput(MixedPointMass(mass=1.0), clip_states=False)
    config = RLEnvConfig(n_steps=2, dt=0.01, n_joints=2, n_muscles=2)
    initial = (plant.init(key=jax.random.PRNGKey(0)), jnp.ones(2))

    def step(carry, _):
        return _physics_substep(plant, config, jnp.ones(2), carry), None

    (physical, _), _ = jax.jit(lambda: jax.lax.scan(step, initial, None, length=2))()
    assert physical.skeleton.pos.dtype == jnp.float32
    assert physical.skeleton.vel.dtype == jnp.float64
    np.testing.assert_allclose(physical.skeleton.vel, 0.02, atol=1e-12)


class DecayPair(eqx.Module):
    x: jax.Array
    y: jax.Array


class DecayDAE(DAEComponent):
    def init_system_state(self, *, key):
        return DecayPair(jnp.ones(2, dtype=jnp.float32), jnp.full(3, 1 + 1e-10, dtype=jnp.float64))

    def vector_field(self, t, state, rate):
        return jt.map(lambda value: -rate[0] * value, state)

    def extract_outputs(self, state):
        return {"output": state}

    @property
    def input_size(self):
        return 1


@pytest.mark.parametrize("solver_type", [dfx.Euler, dfx.Heun, dfx.ImplicitEuler])
def test_decay_dae_scan_training_gradient_and_precision(solver_type):
    model = DecayDAE(dt=0.01, solver_type=solver_type).prepare()

    def rollout(rate):
        def step(state, _):
            outputs, state = model({"input": rate[None]}, state, key=jax.random.PRNGKey(0))
            return state, outputs["state"]

        state = model.init_state(key=jax.random.PRNGKey(0))
        return jax.lax.scan(step, state, None, length=10)[1]

    trajectory = eqx.filter_jit(rollout)(jnp.array(1.0))
    assert trajectory.x.dtype == jnp.float32
    assert trajectory.y.dtype == jnp.float64
    np.testing.assert_allclose(trajectory.y[-1], np.exp(-0.1) * (1 + 1e-10), atol=5e-4)
    assert abs(float(trajectory.y[0, 0]) - float(trajectory.x[0, 0])) > 1e-10

    def loss(rate):
        return jnp.sum((rollout(rate).y[-1] - 0.5) ** 2)

    value_and_grad = jax.jit(jax.value_and_grad(loss))
    value, grad = value_and_grad(jnp.array(1.0))
    assert jnp.isfinite(grad) and grad < 0
    assert value_and_grad(1.0 - 0.1 * grad)[0] < value


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("solver_type", [dfx.Euler, dfx.Heun, dfx.Tsit5])
def test_homogeneous_f32_filter_scan(x64, solver_type):
    with jax.experimental.enable_x64(x64):
        model = FirstOrderFilter(
            dt=0.001, solver_type=solver_type, input_proto=jnp.zeros(2, dtype=jnp.float32)
        ).prepare()

        def step(state, _):
            outputs, state = model(
                {"input": jnp.ones(2, dtype=jnp.float32)}, state, key=jax.random.PRNGKey(0)
            )
            return state, outputs["output"]

        state = model.init_state(key=jax.random.PRNGKey(0))
        _, outputs = eqx.filter_jit(lambda state: jax.lax.scan(step, state, None, length=3))(state)
        assert outputs.dtype == jnp.float32
        np.testing.assert_allclose(outputs[-1], 1 - np.exp(-0.003 / 0.05), atol=6e-4)


def test_stateful_solver_observes_external_state_edit():
    model = PointMassDAE(dt=0.1, damping=1.0, solver_type=dfx.Tsit5).prepare()
    state = model.init_state(key=jax.random.PRNGKey(0))
    _, state = model({"force": jnp.zeros(2)}, state, key=jax.random.PRNGKey(0))
    physical = state.get(model.state_index)
    edited = eqx.tree_at(lambda s: s.system.vel, physical, jnp.ones(2))
    state = state.set(model.state_index, edited)
    outputs, _ = eqx.filter_jit(model)({"force": jnp.zeros(2)}, state, key=jax.random.PRNGKey(0))
    np.testing.assert_allclose(outputs["state"].vel, np.exp(-0.1), atol=1e-9)
    np.testing.assert_allclose(outputs["state"].pos, 1 - np.exp(-0.1), atol=1e-9)


@pytest.mark.parametrize("solver_type", [dfx.Euler, dfx.Heun])
def test_compliant_tendon_override_uses_mixed_boundary(solver_type):
    from feedbax.mechanics.hill_muscles import (
        CompliantTendonHillMuscle,
        CompliantTendonState,
        HillMuscleParams,
    )

    class MixedMuscle(CompliantTendonHillMuscle):
        def init_system_state(self, *, key):
            return CompliantTendonState(
                activation=jnp.array(0.25, dtype=jnp.float32),
                fiber_length=jnp.array(0.1, dtype=jnp.float64),
            )

    model = MixedMuscle(
        HillMuscleParams(500.0, 0.1, 0.15), dt=0.0001, solver_type=solver_type
    ).prepare()
    inputs = {
        "excitation": jnp.array(0.5),
        "musculotendon_length": jnp.array(0.25),
        "musculotendon_velocity": jnp.array(0.0),
    }
    outputs, _ = eqx.filter_jit(model)(
        inputs, model.init_state(key=jax.random.PRNGKey(0)), key=jax.random.PRNGKey(0)
    )
    physical = outputs["state"]
    assert physical.activation.dtype == jnp.float32
    assert physical.fiber_length.dtype == jnp.float64
    expected = 0.5 - 0.25 * np.exp(-model.dt / model.muscle_params.tau_activation)
    np.testing.assert_allclose(physical.activation, expected, atol=2e-5)
    assert jnp.isfinite(outputs["force"])
