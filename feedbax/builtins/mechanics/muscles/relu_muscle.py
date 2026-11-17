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

import equinox as eqx
import jax.numpy as jnp
from equinox.nn import State, StateIndex
from jaxtyping import Array, PRNGKeyArray, PyTree

from feedbax.contracts.component import PortType
from feedbax.mechanics.hill_muscles import ActivationDynamics
from feedbax.mechanics.muscle_law import MuscleStage
from feedbax.runtime.graph import Component
from feedbax.contracts.portable import Ports, ConfigBinding, ComponentOptions
from typing import Annotated
from pydantic import Field

logger = logging.getLogger(__name__)


def scalar_muscle_output_prototype(
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Return scalar muscle force and activation prototypes."""

    proto = jnp.zeros(())
    return {"force": proto, "activation": proto}


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

    ports = Ports(
        inputs={"excitation": PortType(dtype="scalar")},
        outputs={"force": PortType(dtype="scalar"), "activation": PortType(dtype="scalar")},
    )
    config_bindings = (ConfigBinding(path="state_index.init", name="initial_activation"),)
    component_options = ComponentOptions(output_prototype_fn=scalar_muscle_output_prototype)

    max_isometric_force: float
    tau_activation: float = 0.015
    tau_deactivation: float = 0.05
    min_activation: float = 0.0
    dt: float = 0.01
    state_index: StateIndex

    def __init__(
        self,
        max_isometric_force: Annotated[float, Field(ge=0)] = 500.0,
        tau_activation: Annotated[float, Field(ge=0)] = 0.015,
        tau_deactivation: Annotated[float, Field(ge=0)] = 0.05,
        min_activation: Annotated[float, Field(ge=0)] = 0.0,
        dt: Annotated[float, Field(ge=0)] = 0.01,
        initial_activation: Annotated[float, Field(ge=0)] = 0.0,
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

    def as_muscle_law(self) -> ActivationTensionLaw:
        """Expose the same constitutive law for stage-local integration."""
        return ActivationTensionLaw(
            max_isometric_force=self.max_isometric_force,
            tau_activation=self.tau_activation,
            tau_deactivation=self.tau_deactivation,
            min_activation=self.min_activation,
            initial_activation=self.state_index.init,
        )

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
        law = self.as_muscle_law()
        da_dt = law.activation_derivative(activation, excitation)
        new_activation = jnp.clip(activation + da_dt * self.dt, self.min_activation, 1.0)
        force = law.tension(new_activation)

        state = state.set(self.state_index, new_activation)

        outputs = {
            "force": force,
            "activation": new_activation,
        }
        return outputs, state



class ActivationTensionLaw(eqx.Module):
    """First-order activation with tension linear in bounded activation."""

    max_isometric_force: float
    tau_activation: float
    tau_deactivation: float
    min_activation: float
    initial_activation: Array

    @property
    def state_size(self) -> int:
        return 1

    def init_state(self, length: Array, velocity: Array) -> Array:
        return self.initial_activation.astype(length.dtype).reshape(1)

    def activation_derivative(self, activation: Array, excitation: Array) -> Array:
        return ActivationDynamics(self.tau_activation, self.tau_deactivation)(excitation, activation)

    def tension(self, bounded_activation: Array) -> Array:
        return self.max_isometric_force * bounded_activation

    def evaluate(
        self, state: Array, excitation: Array, length: Array, velocity: Array,
    ) -> MuscleStage:
        activation = state[0]
        derivative = self.activation_derivative(activation, excitation)
        bounded = jnp.clip(activation, self.min_activation, 1.0)
        return MuscleStage(jnp.expand_dims(derivative, 0), self.tension(bounded))
