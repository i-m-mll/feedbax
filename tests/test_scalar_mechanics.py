"""Independent scalar constitutive laws and non-invasive measurement acceptance."""

import jax
import jax.numpy as jnp
import pytest

from feedbax.acausal import translational as linear, rotational as angular
from feedbax.acausal.assembly import assemble_system
from feedbax.acausal.base import AcausalConnection

pytestmark = [pytest.mark.usefixtures("enable_jax_x64"), pytest.mark.feedbax_contract]

DOMAINS = [
    (linear.Mass, linear.LinearSpring, linear.LinearDamper, linear.ForceSource,
     (linear.PositionSensor, linear.VelocitySensor, linear.ForceSensor), "pos", "vel"),
    (angular.Inertia, angular.TorsionalSpring, angular.RotationalDamper, angular.TorqueSource,
     (angular.AngleSensor, angular.AngularVelocitySensor, angular.TorqueSensor), "angle", "angular_vel"),
]


def _system(domain, stiffness, damping, sensor_indices=()):
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
    layout, vf, params, read = assemble_system(elements, connections)
    state = jnp.zeros(layout.total_size)
    for body, position, speed in (("a", 1.25, -0.75), ("b", -0.5, 0.25)):
        for slot, value in ((coordinate, position), (velocity, speed)):
            state = state.at[layout.var_index(layout.resolve(f"{body}.flange.{slot}"))].set(value)
    return layout, jax.jit(vf), params, jax.jit(read), state


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
