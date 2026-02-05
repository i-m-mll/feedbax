"""Eight-muscle point mass effector template with ReluMuscle actuators.

Wraps eight ReluMuscle instances and a PointMassRadialGeometry into a
single Component that converts an excitation vector to a 2D force.

:copyright: Copyright 2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from __future__ import annotations

import logging

import equinox as eqx
from equinox import Module, field
from equinox.nn import State
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

from feedbax.graph import Component
from feedbax.mechanics.geometry import PointMassRadialGeometry
from feedbax.mechanics.muscles.relu_muscle import ReluMuscle


logger = logging.getLogger(__name__)


class PointMass8MuscleRelu(Component):
    """Eight-muscle point mass with ReluMuscle actuators.

    Four antagonist pairs arranged radially at 0, 45, 90, 135 degrees.
    Each pair has a positive and negative direction muscle. The combined
    2D force is the sum of individual muscle forces projected along
    their direction vectors.

    Attributes:
        muscles: Tuple of 8 ReluMuscle components.
        geometry: Radial geometry mapping forces to 2D.
    """

    input_ports = ("excitation",)
    output_ports = ("force_2d", "forces", "activations")

    muscles: tuple[ReluMuscle, ...]
    geometry: PointMassRadialGeometry

    def __init__(
        self,
        n_pairs: int = 4,
        max_isometric_force: float = 500.0,
        dt: float = 0.01,
        tau_activation: float = 0.015,
        tau_deactivation: float = 0.05,
    ):
        """Initialize 8-muscle point mass template.

        Args:
            n_pairs: Number of antagonist pairs (default 4 -> 8 muscles).
            max_isometric_force: Peak force for each muscle [N].
            dt: Integration timestep [s].
            tau_activation: Activation time constant [s].
            tau_deactivation: Deactivation time constant [s].
        """
        self.geometry = PointMassRadialGeometry(n_pairs=n_pairs)
        n_muscles = 2 * n_pairs

        self.muscles = tuple(
            ReluMuscle(
                max_isometric_force=max_isometric_force,
                tau_activation=tau_activation,
                tau_deactivation=tau_deactivation,
                dt=dt,
            )
            for _ in range(n_muscles)
        )

    def __call__(
        self,
        inputs: dict[str, PyTree],
        state: State,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, PyTree], State]:
        """Execute one step for all muscles, then compute 2D force.

        Args:
            inputs: Dict with 'excitation' [n_muscles].
            state: Current State container.
            key: PRNG key.

        Returns:
            Outputs dict with 'force_2d' [2], 'forces' [n_muscles],
            'activations' [n_muscles], and updated state.
        """
        excitation = inputs.get("excitation", jnp.zeros(len(self.muscles)))

        forces_list = []
        activations_list = []
        for i, muscle in enumerate(self.muscles):
            muscle_inputs = {"excitation": excitation[i]}
            muscle_outputs, state = muscle(muscle_inputs, state, key=key)
            forces_list.append(muscle_outputs["force"])
            activations_list.append(muscle_outputs["activation"])

        forces = jnp.stack(forces_list)
        activations = jnp.stack(activations_list)

        # Convert to 2D force via geometry
        force_2d = self.geometry.forces_to_force_2d(forces)

        outputs = {
            "force_2d": force_2d,
            "forces": forces,
            "activations": activations,
        }
        return outputs, state
