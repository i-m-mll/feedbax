"""Thelen 2003 rigid tendon Hill-type muscle model as a feedbax Component.

Implements the rigid tendon variant from Thelen (2003), with pre-computed
force-velocity constants following the MotorNet approach. Activation dynamics
use Euler integration; fiber length is determined algebraically from the
musculotendon length minus tendon slack length.

References:
    Thelen (2003): Adjustment of muscle mechanics model parameters to
        simulate dynamic contractions in older adults.

:copyright: Copyright 2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from __future__ import annotations

import logging
import math

import equinox as eqx
from equinox import Module, field
from equinox.nn import State, StateIndex
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

from feedbax.graph import Component


logger = logging.getLogger(__name__)


class RigidTendonHillMuscleThelen(Component):
    """Thelen 2003 rigid tendon model with pre-computed FV constants.

    Takes excitation, musculotendon length and velocity as inputs.
    Manages activation state via StateIndex. Fiber length is computed
    algebraically from musculotendon length and tendon slack length
    (rigid tendon assumption).

    Attributes:
        max_isometric_force: Peak isometric force [N].
        optimal_muscle_length: Fiber length at peak force [m].
        tendon_slack_length: Unstretched tendon length [m].
        vmax_factor: Maximum shortening velocity in optimal lengths/s.
        min_activation: Floor for activation clipping.
        tau_activation: Activation time constant [s].
        tau_deactivation: Deactivation time constant [s].
        dt: Euler integration timestep [s].
    """

    input_ports = ("excitation", "musculotendon_length", "musculotendon_velocity")
    output_ports = ("force", "activation", "fiber_length", "fiber_velocity")

    max_isometric_force: float
    optimal_muscle_length: float
    tendon_slack_length: float
    vmax_factor: float = 10.0
    min_activation: float = 0.001
    tau_activation: float = 0.015
    tau_deactivation: float = 0.05
    dt: float = 0.01

    # Pre-computed passive element constants
    pe_k: float = field(static=True)
    pe_1: float = field(static=True)
    pe_den: float = field(static=True)

    # Pre-computed contractile element constants
    ce_gamma: float = field(static=True)
    ce_Af: float = field(static=True)
    ce_fmlen: float = field(static=True)

    # Velocity-dependent pre-computed constants
    vmax: float = field(static=True)
    ce_0: float = field(static=True)
    ce_1: float = field(static=True)
    ce_2: float = field(static=True)
    ce_3: float = field(static=True)
    ce_4: float = field(static=True)
    ce_5: float = field(static=True)

    state_index: StateIndex
    _initial_state: Float[Array, ""] = field(static=True)

    def __init__(
        self,
        max_isometric_force: float = 500.0,
        optimal_muscle_length: float = 0.1,
        tendon_slack_length: float = 0.1,
        vmax_factor: float = 10.0,
        min_activation: float = 0.001,
        tau_activation: float = 0.015,
        tau_deactivation: float = 0.05,
        dt: float = 0.01,
        initial_activation: float = 0.001,
    ):
        """Initialize Thelen rigid tendon muscle.

        Args:
            max_isometric_force: Peak isometric force [N].
            optimal_muscle_length: Optimal fiber length [m].
            tendon_slack_length: Tendon slack length [m].
            vmax_factor: Max shortening velocity [optimal lengths/s].
            min_activation: Minimum activation level.
            tau_activation: Activation time constant [s].
            tau_deactivation: Deactivation time constant [s].
            dt: Integration timestep [s].
            initial_activation: Starting activation value.
        """
        self.max_isometric_force = max_isometric_force
        self.optimal_muscle_length = optimal_muscle_length
        self.tendon_slack_length = tendon_slack_length
        self.vmax_factor = vmax_factor
        self.min_activation = min_activation
        self.tau_activation = tau_activation
        self.tau_deactivation = tau_deactivation
        self.dt = dt

        # Pre-compute passive element constants (from MotorNet)
        self.pe_k = 5.0
        self.pe_1 = self.pe_k / 0.6
        self.pe_den = math.exp(self.pe_k) - 1.0

        # Pre-compute contractile element constants
        self.ce_gamma = 0.45
        self.ce_Af = 0.25
        self.ce_fmlen = 1.4

        # Velocity-dependent constants
        self.vmax = vmax_factor * optimal_muscle_length
        self.ce_0 = 3.0 * self.vmax
        self.ce_1 = self.ce_Af * self.vmax
        self.ce_2 = (
            3.0 * self.ce_Af * self.vmax * self.ce_fmlen
            - 3.0 * self.ce_Af * self.vmax
        )
        self.ce_3 = (
            8.0 * self.ce_Af * self.ce_fmlen + 8.0 * self.ce_fmlen
        )
        self.ce_4 = (
            self.ce_Af * self.ce_fmlen * self.vmax - self.ce_1
        )
        self.ce_5 = 8.0 * (self.ce_Af + 1.0)

        # State: activation scalar
        self._initial_state = jnp.array(initial_activation, dtype=jnp.float32)
        self.state_index = StateIndex(self._initial_state)

    def __call__(
        self,
        inputs: dict[str, PyTree],
        state: State,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, PyTree], State]:
        """Execute one step: update activation, compute force from FLV curves.

        Args:
            inputs: Dict with 'excitation', 'musculotendon_length',
                'musculotendon_velocity'.
            state: Current State container.
            key: PRNG key (unused).

        Returns:
            Outputs dict and updated state.
        """
        activation = state.get(self.state_index)
        excitation = inputs.get("excitation", jnp.array(0.0))
        mt_length = inputs.get("musculotendon_length", jnp.array(self.optimal_muscle_length + self.tendon_slack_length))
        mt_velocity = inputs.get("musculotendon_velocity", jnp.array(0.0))

        # --- Activation dynamics (Euler step) ---
        tau = jnp.where(
            excitation > activation,
            self.tau_activation,
            self.tau_deactivation,
        )
        da_dt = (excitation - activation) / tau
        new_activation = activation + da_dt * self.dt
        new_activation = jnp.clip(new_activation, self.min_activation, 1.0)

        # --- Rigid tendon: fiber length from geometry ---
        muscle_len = jnp.clip(mt_length - self.tendon_slack_length, 0.001)
        muscle_vel = mt_velocity

        # --- Force-length (active): Gaussian ---
        flce = jnp.exp(
            -((muscle_len / self.optimal_muscle_length) - 1.0) ** 2
            / self.ce_gamma
        )

        # --- Force-length (passive): exponential ---
        # l0_pe = optimal_muscle_length (normalized slack muscle length = 1.0)
        l0_pe = self.optimal_muscle_length
        flpe = jnp.clip(
            (jnp.exp(self.pe_1 * (muscle_len - l0_pe) / self.optimal_muscle_length) - 1.0)
            / self.pe_den,
            0.0,
        )

        # --- Force-velocity (Thelen with pre-computed constants) ---
        condition = muscle_vel <= 0.0
        nom = jnp.where(
            condition,
            self.ce_Af * (new_activation * self.ce_0 + 4.0 * muscle_vel + self.vmax),
            self.ce_2 * new_activation + self.ce_3 * muscle_vel + self.ce_4,
        )
        den = jnp.where(
            condition,
            new_activation * 3.0 * self.ce_1 + self.ce_1 - 4.0 * muscle_vel,
            self.ce_4 * new_activation * 3.0 + self.ce_5 * muscle_vel + self.ce_4,
        )
        # Prevent division by zero
        den = jnp.where(jnp.abs(den) < 1e-10, 1e-10, den)
        fvce = jnp.clip(nom / den, 0.0)

        # --- Total force ---
        force = (new_activation * flce * fvce + flpe) * self.max_isometric_force

        state = state.set(self.state_index, new_activation)

        outputs = {
            "force": force,
            "activation": new_activation,
            "fiber_length": muscle_len,
            "fiber_velocity": muscle_vel,
        }
        return outputs, state
