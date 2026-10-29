"""Physical invariants for ideal lossless shaft constraints."""

import itertools

import jax
import jax.numpy as jnp
import pytest

from feedbax.acausal import AcausalConnection, AcausalSystem
from feedbax.acausal.assembly import assemble_system
from feedbax.runtime.graph import init_state_from_component
from feedbax.acausal.rotational import (
    AngleSensor,
    AngularVelocitySensor,
    GearRatio,
    Inertia,
    RotationalDamper,
    TorqueSensor,
    TorqueSource,
    TorsionalSpring,
    RotationalGround,
)

pytestmark = [pytest.mark.feedbax_contract, pytest.mark.usefixtures("enable_jax_x64")]


def _connection(a, b):
    return AcausalConnection(a, b)


def _shafts(ratio=2.0):
    elements = {
        "a": Inertia("a", inertia=2.0),
        "b": Inertia("b", inertia=3.0),
        "gear": GearRatio("gear", ratio=ratio),
        "drive_a": TorqueSource("drive_a"),
        "drive_b": TorqueSource("drive_b"),
    }
    connections = [
        _connection(("a", "flange"), ("gear", "flange_a")),
        _connection(("b", "flange"), ("gear", "flange_b")),
        _connection(("drive_a", "flange"), ("a", "flange")),
        _connection(("drive_b", "flange"), ("b", "flange")),
    ]
    for side, sensor_type in itertools.product(
        ("a", "b"), (AngleSensor, AngularVelocitySensor, TorqueSensor)
    ):
        name = f"{side}_{sensor_type.__name__}"
        elements[name] = sensor_type(name)
        connections.append(_connection((name, "flange"), (side, "flange")))
    return elements, connections


@pytest.mark.parametrize(
    "reverse_elements,reverse_connections", itertools.product((False, True), repeat=2)
)
@pytest.mark.parametrize("drive_side", ["a", "b"])
def test_reflected_inertia_and_physical_shaft_sensors(
    reverse_elements, reverse_connections, drive_side
):
    elements, connections = _shafts()
    if reverse_elements:
        elements = dict(reversed(list(elements.items())))
    if reverse_connections:
        connections = [
            AcausalConnection((c.element_b, c.port_b), (c.element_a, c.port_a))
            for c in reversed(connections)
        ]
    layout, vf, params, sensors = assemble_system(elements, connections)
    assert layout.total_size == 2
    y = jnp.array([0.4, 0.3])
    inputs = jnp.zeros(len(layout._inputs))
    inputs = inputs.at[layout._inputs[f"drive_{drive_side}.flange.torque"]].set(
        14.0 if drive_side == "a" else 7.0
    )
    dy = vf(0.0, y, (inputs, params))
    assert dy == pytest.approx([0.3, 1.0])
    output = sensors(y, inputs, params)
    assert output["a_AngleSensor"] == pytest.approx(0.4)
    assert output["b_AngleSensor"] == pytest.approx(0.8)
    assert output["b_AngularVelocitySensor"] == pytest.approx(0.6)
    assert output["a_TorqueSensor"] == pytest.approx(2.0)
    assert output["b_TorqueSensor"] == pytest.approx(6.0)
    applied_a, applied_b = (14.0, 0.0) if drive_side == "a" else (0.0, 7.0)
    reaction_power = (output["a_TorqueSensor"] - applied_a) * 0.3 + (
        output["b_TorqueSensor"] - applied_b
    ) * 0.6
    assert reaction_power == pytest.approx(0.0)
    assert jax.grad(
        lambda drive: vf(
            0.0, y, (inputs.at[layout._inputs["drive_a.flange.torque"]].set(drive), params)
        )[1]
    )(14.0) == pytest.approx(1 / 14)


@pytest.mark.parametrize("ratio", [2.0, -2.0, 0.5])
def test_external_power_equals_kinetic_energy_rate_and_ratio_is_differentiable(ratio):
    elements, connections = _shafts(ratio)
    layout, vf, params, sensors = assemble_system(elements, connections)
    inputs = jnp.zeros(len(layout._inputs))
    inputs = inputs.at[layout._inputs["drive_a.flange.torque"]].set(5.0)
    inputs = inputs.at[layout._inputs["drive_b.flange.torque"]].set(7.0)
    y = jnp.array([0.4, 0.3])

    def kinetic(state, parameters):
        readings = sensors(state, inputs, parameters)
        return (
            2 * readings["a_AngularVelocitySensor"] ** 2
            + 3 * readings["b_AngularVelocitySensor"] ** 2
        ) / 2

    dy = jax.jit(vf)(0.0, y, (inputs, params))
    output = sensors(y, inputs, params)
    energy_rate = jnp.vdot(jax.grad(kinetic)(y, params), dy)
    power = 5 * output["a_AngularVelocitySensor"] + 7 * output["b_AngularVelocitySensor"]
    assert energy_rate == pytest.approx(float(power))

    def acceleration(r):
        return vf(0.0, y, (inputs, {**params, "gear.ratio": r}))[1]

    derivative = jax.jit(jax.grad(acceleration))(jnp.array(ratio))
    delta = 1e-5
    finite_difference = (acceleration(ratio + delta) - acceleration(ratio - delta)) / (2 * delta)
    assert derivative == pytest.approx(float(finite_difference), rel=1e-7, abs=1e-8)


@pytest.mark.parametrize("damping", [0.0, 0.7])
def test_spring_energy_and_damper_dissipation(damping):
    elements = {
        "inertia": Inertia("inertia"),
        "gear": GearRatio("gear", ratio=2),
        "spring": TorsionalSpring("spring", stiffness=4),
        "damper": RotationalDamper("damper", damping=damping),
    }
    connections = [_connection(("inertia", "flange"), ("gear", "flange_a"))]
    for name in ("spring", "damper"):
        for side in ("a", "b"):
            connections.append(_connection((name, f"flange_{side}"), ("gear", f"flange_{side}")))
    _, vf, params, _ = assemble_system(elements, connections)
    y = jnp.array([1.0, 0.3])
    dy = vf(0.0, y, (jnp.zeros(0), params))

    def energy(state):
        return (state[1] ** 2 + 4 * (2 * state[0] - state[0]) ** 2) / 2

    energy_rate = jnp.vdot(jax.grad(energy)(y), dy)
    assert energy_rate == pytest.approx(-damping * (0.6 - 0.3) ** 2, abs=1e-12)
    if not damping:
        assert dy[1] == pytest.approx(-4.0)


def test_chain_ratios_and_runtime_sensor_outputs():
    elements, connections = _shafts()
    elements.update(
        {
            "second": GearRatio("second", ratio=-3),
            "c": Inertia("c", inertia=1),
            "c_angle": AngleSensor("c_angle"),
        }
    )
    connections.extend(
        [
            _connection(("b", "flange"), ("second", "flange_a")),
            _connection(("c", "flange"), ("second", "flange_b")),
            _connection(("c_angle", "flange"), ("c", "flange")),
        ]
    )
    system = AcausalSystem(elements, connections, dt=0.01)
    state = init_state_from_component(system)
    outputs, state = system(
        {"drive_a": jnp.array([50.0]), "drive_b": jnp.array([0.0])},
        state,
        key=jax.random.PRNGKey(0),
    )
    assert outputs["b_AngularVelocitySensor"] == pytest.approx(0.02)
    assert outputs["b_TorqueSensor"] == pytest.approx(6.0)
    y = jnp.array([0.4, 0.3])
    readings = system._compiled_sensor_fn(y, jnp.array([50.0, 0.0]), system.params.values)
    assert readings["c_angle"] == pytest.approx(-2.4)


@pytest.mark.parametrize("ratio", [0.0, float("nan"), float("inf")])
def test_singular_gear_ratio_is_rejected(ratio):
    elements, connections = _shafts(ratio)
    with pytest.raises(ValueError, match="finite and nonzero"):
        assemble_system(elements, connections)


def test_gear_cycles_are_rejected():
    elements, connections = _shafts()
    elements["extra"] = GearRatio("extra", ratio=0.5)
    connections.extend(
        [
            _connection(("extra", "flange_a"), ("b", "flange")),
            _connection(("extra", "flange_b"), ("a", "flange")),
        ]
    )
    with pytest.raises(ValueError, match="Gear constraint cycle"):
        assemble_system(elements, connections)


def test_b_side_ground_propagates_and_massless_shaft_is_balanced():
    elements, connections = _shafts()
    elements["wall"] = RotationalGround("wall")
    connections.append(_connection(("wall", "flange"), ("b", "flange")))
    layout, vf, params, sensors = assemble_system(elements, connections)
    assert layout.total_size == 0
    values = sensors(jnp.zeros(0), jnp.array([14.0, 0.0]), params)
    assert all(float(value) == 0 for value in values.values())
    elements, connections = _shafts()
    del elements["a"]
    connections = [
        AcausalConnection(
            ("gear", "flange_a") if c.element_a == "a" else (c.element_a, c.port_a),
            ("gear", "flange_a") if c.element_b == "a" else (c.element_b, c.port_b),
        )
        for c in connections
        if c.element_a != "a" or c.element_b != "gear"
    ]
    layout, vf, params, sensors = assemble_system(elements, connections)
    dy = vf(0.0, jnp.array([0.0, 0.3]), (jnp.array([12.0, 0.0]), params))
    assert dy[1] == pytest.approx(1.0)
    values = sensors(jnp.array([0.0, 0.3]), jnp.array([12.0, 0.0]), params)
    assert values["a_TorqueSensor"] == pytest.approx(0.0)
    assert values["b_TorqueSensor"] == pytest.approx(6.0)


def test_ratio_batching_observation_gradients_and_duplicate_sensor_transparency():
    elements, connections = _shafts()
    layout, vf, params, sensors = assemble_system(elements, connections)
    y = jnp.array([0.4, 0.3])
    inputs = jnp.array([14.0, 0.0])

    def observe(ratio):
        parameters = {**params, "gear.ratio": ratio}
        return jnp.stack(
            (vf(0.0, y, (inputs, parameters))[1], sensors(y, inputs, parameters)["b_AngleSensor"])
        )

    batched = jax.jit(jax.vmap(observe))(jnp.array([2.0, -2.0]))
    assert batched == pytest.approx(jnp.array([[1.0, 0.8], [1.0, -0.8]]))
    assert jax.jacfwd(observe)(2.0) == pytest.approx(jnp.array([-6 / 7, 0.4]))
    before = vf(0.0, y, (inputs, params))
    elements["000_duplicate"] = TorqueSensor("000_duplicate")
    connections.append(_connection(("000_duplicate", "flange"), ("b", "flange")))
    new_layout, new_vf, new_params, new_sensors = assemble_system(elements, connections)
    assert new_layout.total_size == 2
    assert new_vf(0.0, y, (inputs, new_params)) == pytest.approx(before)
    assert new_sensors(y, inputs, new_params)["000_duplicate"] == pytest.approx(6.0)
    assert jax.jacfwd(lambda r: new_vf(0.0, y, (inputs, {**new_params, "gear.ratio": r})))(
        2.0
    ) == pytest.approx(jnp.array([0.0, -6 / 7]))


def test_branch_with_inverse_path_and_colocated_inertias():
    elements, connections = _shafts()
    elements.update(
        {
            "extra_b": Inertia("extra_b", inertia=1.0),
            "c": Inertia("c", inertia=9.0),
            "reverse": GearRatio("reverse", ratio=3),
            "c_velocity": AngularVelocitySensor("c_velocity"),
        }
    )
    connections.extend(
        [
            _connection(("extra_b", "flange"), ("b", "flange")),
            _connection(("c", "flange"), ("reverse", "flange_a")),
            _connection(("b", "flange"), ("reverse", "flange_b")),
            _connection(("c_velocity", "flange"), ("c", "flange")),
        ]
    )
    layout, vf, params, sensors = assemble_system(elements, connections)
    assert layout.total_size == 2
    y, inputs = jnp.array([0.0, 0.3]), jnp.array([22.0, 0.0])
    assert vf(0.0, y, (inputs, params))[1] == pytest.approx(1.0)
    readings = sensors(y, inputs, params)
    assert readings["c_velocity"] == pytest.approx(0.2)
    assert readings["b_TorqueSensor"] == pytest.approx(8.0)
    # Kinetic energy grows only by the drive power across all three shafts.
    assert 2 * 0.3 * 1 + 4 * 0.6 * 2 + 9 * 0.2 * (2 / 3) == pytest.approx(22 * 0.3)


def test_grounded_net_torque_has_known_zero_acceleration_under_load():
    elements = {
        "wall": RotationalGround("wall"),
        "inertia": Inertia("inertia", inertia=2.0),
        "drive": TorqueSource("drive"),
        "net": TorqueSensor("net"),
    }
    connections = [
        _connection((name, "flange"), ("wall", "flange")) for name in ("inertia", "drive", "net")
    ]
    layout, _, params, sensors = assemble_system(elements, connections)
    assert layout.total_size == 0
    assert sensors(jnp.zeros(0), jnp.array([14.0]), params)["net"] == pytest.approx(0.0)


def test_free_net_torque_requires_solved_acceleration():
    elements = {"drive": TorqueSource("drive"), "net": TorqueSensor("net")}
    with pytest.raises(ValueError, match="no solved acceleration.*requires inertia"):
        assemble_system(elements, [_connection(("drive", "flange"), ("net", "flange"))])


def test_prescribed_net_effort_requires_acceleration_authority():
    from feedbax.acausal import ForceSensor, Mass, PrescribedMotion

    elements = {
        "driver": PrescribedMotion("driver"),
        "mass": Mass("mass"),
        "net": ForceSensor("net"),
    }
    connections = [_connection((name, "flange"), ("driver", "flange")) for name in ("mass", "net")]
    with pytest.raises(ValueError, match="prescribed velocity inputs do not supply acceleration"):
        assemble_system(elements, connections)


def test_prescribed_coordinate_and_velocity_observations_use_supplied_values():
    from feedbax.acausal import PositionSensor, PrescribedMotion, VelocitySensor

    elements = {
        "driver": PrescribedMotion("driver"),
        "position": PositionSensor("position"),
        "velocity": VelocitySensor("velocity"),
    }
    connections = [
        _connection((name, "flange"), ("driver", "flange")) for name in ("position", "velocity")
    ]
    layout, _, params, sensors = assemble_system(elements, connections)
    assert layout.total_size == 0
    outputs = sensors(jnp.zeros(0), jnp.array([0.4, -0.3]), params)
    assert outputs["position"] == pytest.approx(0.4)
    assert outputs["velocity"] == pytest.approx(-0.3)
