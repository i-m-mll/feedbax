"""Behavioral tests for intervention components and feedback impulses."""

import equinox as eqx
from equinox.nn import State
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from feedbax.runtime.signal_transforms import (
    AddNoise,
    AddNoiseParams,
    ConstantInput,
    ConstantInputParams,
)
from feedbax.models.feedback_impulse import feedback_impulse
from feedbax.mechanics.force_fields import DynamicsMatrixPerturb, DynamicsMatrixPerturbParams
from feedbax.runtime.graph import Graph, Wire, init_state_from_component
from feedbax.runtime.iteration import run_component
from feedbax.runtime.state import CartesianState
from feedbax.tasks.task import TaskTrialSpec, prepare_trial


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


def test_add_noise_keeps_each_signal_leaf_dtype_and_random_stream() -> None:
    component = AddNoise(params=AddNoiseParams(active=True))
    state = State(component)
    signal = (jnp.zeros((4,), dtype=jnp.float16), jnp.zeros((3,), dtype=jnp.bfloat16))
    key = jr.key(0)
    outputs, _ = component({"input": signal}, state, key=key)

    for result, leaf, leaf_key in zip(outputs["output"], signal, jr.split(key, 2), strict=True):
        assert result.dtype == leaf.dtype
        assert jnp.array_equal(result, jr.normal(leaf_key, leaf.shape, leaf.dtype))


@pytest.fixture
def feedback():
    pos = jnp.arange(12, dtype=jnp.float32).reshape(6, 2)
    vel = -pos - 1
    return pos, vel


def _run_feedback_impulse(feedback, *, seed=0, **kwargs):
    n_transitions = feedback[0].shape[0]
    component = feedback_impulse(n_steps=n_transitions + 1, **kwargs)
    outputs, _, _ = run_component(
        component,
        {"input": feedback},
        init_state_from_component(component),
        n_steps=n_transitions,
        key=jr.key(seed),
        state_filter=False,
    )
    result = outputs["output"]
    assert isinstance(result, tuple)
    return result


@pytest.mark.parametrize(
    ("feedback_var", "feedback_dim", "start", "duration", "active"),
    [
        (0, 1, 1, 2, slice(1, 3)),
        (1, 0, -2, 4, slice(0, 2)),
    ],
    ids=["position", "velocity-clipped-at-start"],
)
def test_feedback_impulse_adds_signed_offset_in_window(
    feedback, feedback_var, feedback_dim, start, duration, active
):
    actual = _run_feedback_impulse(
        feedback,
        amplitude=-0.25,
        feedback_var=feedback_var,
        feedback_dim=feedback_dim,
        start_timestep=start,
        duration=duration,
    )
    expected = list(feedback)
    expected[feedback_var] = expected[feedback_var].at[active, feedback_dim].add(-0.25)
    np.testing.assert_array_equal(actual, expected)


def test_nested_feedback_impulse_and_task_constant_input(feedback):
    pulse = feedback_impulse(
        n_steps=5, amplitude=-0.25, duration=4, feedback_var=1, start_timestep=3, feedback_dim=1
    )
    offsets = (jnp.array([0.0, 2.0]), jnp.array([3.0, 0.0]))
    graph = Graph(
        nodes={"pulse": pulse, "constant": ConstantInput(ConstantInputParams(arrays=offsets))},
        wires=(Wire("pulse", "output", "constant", "input"),),
        input_ports=("input",),
        output_ports=("output",),
        input_bindings={"input": ("pulse", "input")},
        output_bindings={"output": ("constant", "output")},
    )
    trial = prepare_trial(
        graph,
        TaskTrialSpec(
            inits={"constant.params_index.scale": jnp.asarray(2.0)},
            targets={},
            inputs={"input": feedback},
        ),
    )
    outputs, _, _ = run_component(
        graph,
        trial.inputs,
        trial.init_state,
        n_steps=trial.n_steps,
        key=jr.key(0),
        state_filter=False,
    )

    expected_pos = feedback[0] + jnp.array([0.0, 4.0])
    expected_vel = (feedback[1] + jnp.array([6.0, 0.0])).at[3, 1].add(-0.25)
    np.testing.assert_array_equal(outputs["output"], (expected_pos, expected_vel))


@pytest.mark.parametrize(
    ("start", "duration"),
    [(2, 0), (6, 3), (-3, 2)],
    ids=["zero-duration", "after-rollout", "before-rollout"],
)
def test_feedback_impulse_is_identity_without_active_transitions(feedback, start, duration):
    actual = _run_feedback_impulse(
        feedback,
        amplitude=0.25,
        feedback_var=0,
        start_timestep=start,
        duration=duration,
    )
    np.testing.assert_array_equal(actual, feedback)


def test_feedback_impulse_uses_one_seeded_direction_per_trial(feedback):
    kwargs = dict(amplitude=0.75, feedback_var=1, start_timestep=2, duration=3)
    pos, vel = _run_feedback_impulse(feedback, seed=7, **kwargs)
    np.testing.assert_array_equal(pos, feedback[0])
    np.testing.assert_array_equal(vel[:2], feedback[1][:2])
    np.testing.assert_array_equal(vel[5:], feedback[1][5:])

    offsets = vel[2:5] - feedback[1][2:5]
    np.testing.assert_allclose(jnp.linalg.norm(offsets, axis=1), 0.75, atol=1e-6)
    np.testing.assert_allclose(offsets, jnp.broadcast_to(offsets[0], offsets.shape), atol=1e-6)

    np.testing.assert_array_equal(_run_feedback_impulse(feedback, seed=7, **kwargs)[1], vel)
    other_vel = _run_feedback_impulse(feedback, seed=8, **kwargs)[1]
    assert not jnp.allclose(other_vel[2:5], vel[2:5])
