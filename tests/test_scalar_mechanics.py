"""Independent scalar constitutive laws and non-invasive measurement acceptance."""

import jax
import jax.numpy as jnp
import pytest

from feedbax.acausal import translational as linear, rotational as angular
from feedbax.acausal import AcausalSystem
from feedbax.acausal.assembly import assemble_system
from feedbax.acausal.base import AcausalConnection
from feedbax.acausal.system import AcausalSystemState
from feedbax.mechanics.dae import DAEState

pytestmark = [pytest.mark.usefixtures("enable_jax_x64"), pytest.mark.feedbax_contract]

DOMAINS = [
    (linear.Mass, linear.LinearSpring, linear.LinearDamper, linear.ForceSource,
     (linear.PositionSensor, linear.VelocitySensor, linear.ForceSensor), "pos", "vel"),
    (angular.Inertia, angular.TorsionalSpring, angular.RotationalDamper, angular.TorqueSource,
     (angular.AngleSensor, angular.AngularVelocitySensor, angular.TorqueSensor), "angle", "angular_vel"),
]


def _elements(domain, stiffness, damping, sensor_indices=()):
    inertia, spring, damper, source, sensors, coordinate, velocity = domain
    elements = {
        "a": inertia("a", 2.0), "b": inertia("b", 4.0),
        "spring": spring("spring", stiffness), "damper": damper("damper", damping),
        "source": source("source"),
    }
    connections = [AcausalConnection(("source", "flange"), ("a", "flange"))]
    for name in ("spring", "damper"):
        for terminal in ("a", "b"):
            connections.append(AcausalConnection((name, f"flange_{terminal}"), (terminal, "flange")))
    for index, kind in enumerate(sensor_indices):
        name = f"sensor{index}"
        elements[name] = sensors[kind](name)
        connections.append(AcausalConnection((name, "flange"), ("a", "flange")))
    return elements, connections


def _nonzero_state(layout, domain):
    coordinate, velocity = domain[-2:]
    state = jnp.zeros(layout.total_size)
    for body, position, speed in (("a", 1.25, -0.75), ("b", -0.5, 0.25)):
        for slot, value in ((coordinate, position), (velocity, speed)):
            state = state.at[layout.var_index(layout.resolve(f"{body}.flange.{slot}"))].set(value)
    return state


def _system(domain, stiffness, damping, sensor_indices=()):
    layout, vf, params, read = assemble_system(*_elements(domain, stiffness, damping, sensor_indices))
    return layout, jax.jit(vf), params, jax.jit(read), _nonzero_state(layout, domain)


@pytest.mark.parametrize("domain", DOMAINS, ids=["translational", "rotational"])
@pytest.mark.parametrize("stiffness,damping,accelerations", [
    (4.0, 0.0, (-1.0, 1.75)),
    (0.0, 3.0, (4.0, -0.75)),
    (4.0, 3.0, (0.5, 1.0)),
])
def test_two_body_laws(domain, stiffness, damping, accelerations):
    layout, vf, params, _, state = _system(domain, stiffness, damping)
    derivative = vf(0.0, state, (jnp.array([5.0]), params))
    coordinate, velocity = domain[-2:]
    for body, speed, acceleration in zip(("a", "b"), (-0.75, 0.25), accelerations):
        for slot, expected in ((coordinate, speed), (velocity, acceleration)):
            index = layout.var_index(layout.resolve(f"{body}.flange.{slot}"))
            assert float(derivative[index]) == pytest.approx(expected)


@pytest.mark.parametrize("domain", DOMAINS, ids=["translational", "rotational"])
@pytest.mark.parametrize("sensor_indices", [(), (0,), (1,), (2,), (2, 2), (0, 1, 2, 2)])
def test_sensors_preserve_vector_field_and_measure_net_applied_effort(domain, sensor_indices):
    layout, vf, params, read, state = _system(domain, 4.0, 3.0, sensor_indices)
    inputs = jnp.array([5.0])
    derivative = vf(0.0, state, (inputs, params))
    coordinate, velocity = domain[-2:]
    for body, speed, acceleration in (("a", -0.75, 0.5), ("b", 0.25, 1.0)):
        for slot, expected in ((coordinate, speed), (velocity, acceleration)):
            index = layout.var_index(layout.resolve(f"{body}.flange.{slot}"))
            assert float(derivative[index]) == pytest.approx(expected)
    outputs = read(state, inputs, params)
    for index, kind in enumerate(sensor_indices):
        label = f"sensor{index}"
        if kind < 2:
            value = state[layout.var_index(layout.resolve(layout._outputs[label]))]
        else:
            value = outputs[label]
        assert float(value) == pytest.approx((1.25, -0.75, 1.0)[kind])


@pytest.mark.parametrize("domain", DOMAINS, ids=["translational", "rotational"])
def test_public_simultaneous_sensor_outputs_are_transparent(domain):
    systems = [
        AcausalSystem(
            *_elements(domain, 4.0, 3.0, sensor_indices),
            dt=0.01,
            input_bindings={"drive": "source"},
            output_bindings=dict(zip(
                ("coordinate", "velocity", "effort", "effort_copy"),
                (f"sensor{index}" for index in range(len(sensor_indices))),
            )),
        )
        for sensor_indices in ((), (0, 1, 2, 2))
    ]
    final_states = []
    for system in systems:
        state = system.init_state(key=jax.random.PRNGKey(0))
        initial = system.state_view(state)
        state = state.set(system.state_index, DAEState(
            system=AcausalSystemState(y=_nonzero_state(system._layout, domain)),
            solver=initial.solver,
        ))
        outputs, state = system({"drive": jnp.asarray(5.0)}, state, key=jax.random.PRNGKey(1))
        final_states.append(system.state_view(state).system.y)
    assert jnp.allclose(final_states[0], final_states[1], atol=1e-12, rtol=0)
    assert set(outputs) == {"coordinate", "velocity", "effort", "effort_copy"}
    for label, expected in {
        "coordinate": 1.2425, "velocity": -0.745,
        "effort": 1.055, "effort_copy": 1.055,
    }.items():
        assert float(outputs[label]) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("domain", DOMAINS, ids=["translational", "rotational"])
def test_two_body_jacobian_and_power_balance(domain):
    layout, vf, params, _, state = _system(domain, 4.0, 3.0, (0, 1, 2, 2))
    indices = jnp.array([
        layout.var_index(layout.resolve(f"{body}.flange.{slot}"))
        for body in ("a", "b") for slot in domain[-2:]
    ])
    field = lambda y: vf(0.0, y, (jnp.array([5.0]), params))
    jacobian = jax.jacfwd(field)(state)[indices[:, None], indices[None, :]]
    assert jnp.allclose(jacobian, jnp.array([
        [0.0, 1.0, 0.0, 0.0],
        [-2.0, -1.5, 2.0, 1.5],
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.75, -1.0, -0.75],
    ]), atol=1e-12, rtol=0)

    def energy(y):
        qa, va, qb, vb = y[indices]
        return va**2 + 2.0 * vb**2 + 2.0 * (qa - qb)**2

    energy_rate = jnp.vdot(jax.grad(energy)(state), field(state))
    # Applied power is -3.75; viscous dissipation is 3 at this state.
    assert float(energy_rate) == pytest.approx(-6.75, abs=1e-12)
