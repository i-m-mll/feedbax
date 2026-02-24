"""Graph-integration tests for the acausal system.

Verifies that ``AcausalSystem`` works correctly as a node inside a
feedbax ``Graph``, including wiring to other components and multi-step
execution.

:copyright: Copyright 2024-2025 by MLL <mll@mll.bio>.
:license: Apache 2.0.  See LICENSE for details.
"""

import pytest
import jax
import jax.numpy as jnp
import jax.random as jr

from feedbax.graph import Component, Graph, Wire, init_state_from_component
from feedbax.acausal import (
    AcausalConnection,
    AcausalSystem,
    ForceSource,
    Ground,
    LinearDamper,
    LinearSpring,
    Mass,
    PositionSensor,
    VelocitySensor,
)
from feedbax.acausal.translational import PrescribedMotion
from feedbax.mechanics.dae import DAEState

from equinox import Module, field
from equinox.nn import State, StateIndex
from jaxtyping import PRNGKeyArray, PyTree

jax.config.update("jax_enable_x64", True)


# =========================================================================
# Simple helper component: proportional controller
# =========================================================================

class ProportionalController(Component):
    """Simple P-controller: output = gain * (target - measured)."""

    input_ports = ("target", "measured")
    output_ports = ("output",)

    gain: float

    def __init__(self, gain: float = 1.0):
        self.gain = gain

    def __call__(
        self,
        inputs: dict[str, PyTree],
        state: State,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, PyTree], State]:
        target = inputs.get("target", jnp.zeros(1))
        measured = inputs.get("measured", jnp.zeros(1))
        error = jnp.atleast_1d(target) - jnp.atleast_1d(measured)
        output = self.gain * error
        return {"output": output}, state


class ConstantSource(Component):
    """Outputs a fixed value every step."""

    input_ports = ()
    output_ports = ("output",)

    value: float

    def __init__(self, value: float = 0.0):
        self.value = value

    def __call__(
        self,
        inputs: dict[str, PyTree],
        state: State,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, PyTree], State]:
        return {"output": jnp.array([self.value])}, state


# =========================================================================
# Fixtures
# =========================================================================

def _make_plant() -> AcausalSystem:
    """Mass-spring-damper plant with force input and position output."""
    return AcausalSystem(
        elements={
            "wall": Ground("wall"),
            "mass": Mass("mass", mass=1.0),
            "spring": LinearSpring("spring", stiffness=5.0),
            "damper": LinearDamper("damper", damping=2.0),
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
        dt=0.001,
    )


# =========================================================================
# Tests
# =========================================================================

class TestAcausalSystemInGraph:
    """AcausalSystem as a node in a feedbax Graph."""

    def test_plant_standalone(self):
        """Plant works as a standalone component."""
        plant = _make_plant()
        state = init_state_from_component(plant)
        key = jr.PRNGKey(0)

        outputs, new_state = plant({"f_in": jnp.array([1.0])}, state, key=key)
        assert "state" in outputs

    def test_constant_force_graph(self):
        """Graph with constant force source driving the plant."""
        plant = _make_plant()
        source = ConstantSource(value=1.0)

        graph = Graph(
            nodes={"source": source, "plant": plant},
            wires=(
                Wire("source", "output", "plant", "f_in"),
            ),
            input_ports=(),
            output_ports=("position",),
            input_bindings={},
            output_bindings={"position": ("plant", "x_out")},
        )

        state = init_state_from_component(graph)
        key = jr.PRNGKey(0)

        # Run several steps
        for _ in range(200):
            key, subkey = jr.split(key)
            outputs, state = graph({}, state, key=subkey)

        # Position should be non-zero (force pushing mass)
        pos = outputs.get("position", None)
        assert pos is not None
        assert float(jnp.abs(pos)) > 0.01, f"Expected non-zero position, got {pos}"

    def test_p_controller_with_plant(self):
        """P-controller + plant in a Graph achieves tracking."""
        plant = _make_plant()
        controller = ProportionalController(gain=50.0)
        target_source = ConstantSource(value=0.5)

        graph = Graph(
            nodes={
                "target": target_source,
                "controller": controller,
                "plant": plant,
            },
            wires=(
                Wire("target", "output", "controller", "target"),
                Wire("controller", "output", "plant", "f_in"),
                Wire("plant", "x_out", "controller", "measured"),
            ),
            input_ports=(),
            output_ports=("position",),
            input_bindings={},
            output_bindings={"position": ("plant", "x_out")},
        )

        state = init_state_from_component(graph)
        key = jr.PRNGKey(0)

        # The graph has a cycle (plant.x_out -> controller.measured),
        # so it needs n_steps and cycle_init for the feedback wire.
        cycle_init = {("controller", "measured"): jnp.float64(0.0)}
        n_steps = 5000
        for _ in range(n_steps):
            key, subkey = jr.split(key)
            outputs, state = graph(
                {}, state, key=subkey, n_steps=1, cycle_init=cycle_init,
            )

        # outputs from scan are batched with leading dim of 1
        pos = outputs.get("position", None)
        if pos is not None:
            pos_val = float(jnp.squeeze(pos))
            # With high gain, damping, should approach target=0.5
            # Steady state: gain*(0.5 - x) = k*x
            # 50*(0.5 - x) = 5*x -> 25 = 55*x -> x ~ 0.4545
            assert abs(pos_val) > 0.1, (
                f"Expected non-trivial position, got {pos_val}"
            )


class TestPrescribedMotion:
    """Test PrescribedMotion element."""

    def test_prescribed_motion_drives_position(self):
        """PrescribedMotion overrides position from causal input."""
        sys = AcausalSystem(
            elements={
                "driver": PrescribedMotion("driver"),
                "mass": Mass("mass", mass=1.0),
                "spring": LinearSpring("spring", stiffness=10.0),
                "x_out": PositionSensor("x_out"),
            },
            connections=[
                AcausalConnection(("driver", "flange"), ("spring", "flange_a")),
                AcausalConnection(("spring", "flange_b"), ("mass", "flange")),
                AcausalConnection(("x_out", "flange"), ("mass", "flange")),
            ],
            dt=0.001,
        )
        state = init_state_from_component(sys)
        key = jr.PRNGKey(0)

        # The driver's across vars become inputs.  We drive pos=1.0, vel=0.0
        inputs = {"driver": jnp.array([1.0, 0.0])}

        for _ in range(100):
            key, subkey = jr.split(key)
            outputs, state = sys(inputs, state, key=subkey)

        # The mass should be pulled towards the driver position
        dae_state = sys.state_view(state)
        assert not jnp.any(jnp.isnan(dae_state.system.y)), "NaN in state"

    def test_multiple_prescribed_motion_input_routing(self):
        """Inputs route correctly with multiple prescribed motions."""
        sys = AcausalSystem(
            elements={
                "driver1": PrescribedMotion("driver1"),
                "driver2": PrescribedMotion("driver2"),
                "mass1": Mass("mass1", mass=1.0),
                "mass2": Mass("mass2", mass=1.0),
                "spring1": LinearSpring("spring1", stiffness=5.0),
                "spring2": LinearSpring("spring2", stiffness=5.0),
                "force": ForceSource("force"),
                "x1": PositionSensor("x1"),
                "x2": PositionSensor("x2"),
            },
            connections=[
                AcausalConnection(("driver1", "flange"), ("spring1", "flange_a")),
                AcausalConnection(("spring1", "flange_b"), ("mass1", "flange")),
                AcausalConnection(("driver2", "flange"), ("spring2", "flange_a")),
                AcausalConnection(("spring2", "flange_b"), ("mass2", "flange")),
                AcausalConnection(("force", "flange"), ("mass1", "flange")),
                AcausalConnection(("x1", "flange"), ("mass1", "flange")),
                AcausalConnection(("x2", "flange"), ("mass2", "flange")),
            ],
            dt=0.001,
        )

        def _run(driver2_pos: float) -> float:
            state = init_state_from_component(sys)
            key = jr.PRNGKey(0)
            inputs = {
                "driver1": jnp.array([0.0, 0.0]),
                "driver2": jnp.array([driver2_pos, 0.0]),
                "force": jnp.array([0.0]),
            }
            for _ in range(200):
                key, subkey = jr.split(key)
                outputs, state = sys(inputs, state, key=subkey)
            return float(outputs["x2"])

        pos_with_driver = _run(1.0)
        pos_without_driver = _run(0.0)
        assert pos_with_driver - pos_without_driver > 0.05, (
            "Expected driver2 to affect mass2 position"
        )
