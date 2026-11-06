"""Simple ReLU muscle model as a feedbax Component.

Implements a minimal muscle: force = clip(activation, min_act, 1) * max_isometric_force,
with first-order activation dynamics integrated via Euler step.

:copyright: Copyright 2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
from equinox.nn import State, StateIndex
from jaxtyping import PRNGKeyArray, PyTree

from feedbax.component_registry.declarations import ParameterField
from feedbax.component_registry.discovery import component
from feedbax.contracts.component import PortType, PortTypeSpec
from feedbax.runtime.graph import Component

logger = logging.getLogger(__name__)


def scalar_muscle_output_prototype(
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Return scalar muscle force and activation prototypes."""

    proto = jnp.zeros(())
    return {"force": proto, "activation": proto}


@component(
    "ReluMuscle",
    attribute_paths={"initial_activation": "state_index.init"},
    category="Muscles",
    description="Simple muscle: force = activation * F_max.",
    parameter_fields=[
        ParameterField(
            name="max_isometric_force",
            type="float",
            default=500.0,
            min=0.0,
            required=True,
        ),
        ParameterField(
            name="tau_activation",
            type="float",
            default=0.015,
            min=0.0,
            required=False,
        ),
        ParameterField(
            name="tau_deactivation",
            type="float",
            default=0.05,
            min=0.0,
            required=False,
        ),
        ParameterField(
            name="min_activation",
            type="float",
            default=0.0,
            min=0.0,
            required=False,
        ),
        ParameterField(
            name="dt",
            type="float",
            default=0.01,
            min=0.0,
            required=True,
        ),
        ParameterField(
            name="initial_activation",
            type="float",
            default=0.0,
            min=0.0,
            required=False,
        ),
    ],
    icon="Zap",
    port_types=PortTypeSpec(
        inputs={"excitation": PortType(dtype="scalar")},
        outputs={
            "force": PortType(dtype="scalar"),
            "activation": PortType(dtype="scalar"),
        },
    ),
    output_prototype_fn=scalar_muscle_output_prototype,
)
class ReluMuscle(Component):
    """Simple muscle where force = clip(activation) * max_isometric_force.

    Activation follows first-order ODE: da/dt = (u - a) / tau, where tau
    depends on whether excitation exceeds current activation (activation vs
    deactivation). Uses Euler integration internally.

    Attributes:
        max_isometric_force: Peak force at full activation [N].
        tau_activation: Time constant for increasing activation [s].
        tau_deactivation: Time constant for decreasing activation [s].
        min_activation: Floor for clipped activation.
        dt: Euler integration timestep [s].
    """

    input_ports = ("excitation",)
    output_ports = ("force", "activation")

    max_isometric_force: float
    tau_activation: float = 0.015
    tau_deactivation: float = 0.05
    min_activation: float = 0.0
    dt: float = 0.01
    state_index: StateIndex

    def __init__(
        self,
        max_isometric_force: float = 500.0,
        tau_activation: float = 0.015,
        tau_deactivation: float = 0.05,
        min_activation: float = 0.0,
        dt: float = 0.01,
        initial_activation: float = 0.0,
    ):
        """Initialize ReluMuscle.

        Args:
            max_isometric_force: Peak isometric force [N].
            tau_activation: Activation time constant [s].
            tau_deactivation: Deactivation time constant [s].
            min_activation: Minimum activation level.
            dt: Integration timestep [s].
            initial_activation: Starting activation value.
        """
        self.max_isometric_force = max_isometric_force
        self.tau_activation = tau_activation
        self.tau_deactivation = tau_deactivation
        self.min_activation = min_activation
        self.dt = dt
        initial_state = jnp.zeros(()) + initial_activation
        self.state_index = StateIndex(initial_state)

    def __call__(
        self,
        inputs: dict[str, PyTree],
        state: State,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, PyTree], State]:
        """Execute one Euler step of activation dynamics and compute force.

        Args:
            inputs: Dict with 'excitation' scalar in [0, 1].
            state: Current State container.
            key: PRNG key (unused).

        Returns:
            Outputs dict with 'force' and 'activation', and updated state.
        """
        activation = state.get(self.state_index)
        excitation = inputs.get("excitation", jnp.array(0.0))

        # Activation ODE: da/dt = (u - a) / tau
        tau = jnp.where(
            excitation > activation,
            self.tau_activation,
            self.tau_deactivation,
        )
        da_dt = (excitation - activation) / tau
        new_activation = activation + da_dt * self.dt
        new_activation = jnp.clip(new_activation, self.min_activation, 1.0)

        # Force = activation * max_isometric_force
        force = new_activation * self.max_isometric_force

        state = state.set(self.state_index, new_activation)

        outputs = {
            "force": force,
            "activation": new_activation,
        }
        return outputs, state
