"""Six-muscle arm effector template with Thelen rigid tendon muscles.

Wraps six RigidTendonHillMuscleThelen instances and a TwoLinkArmMuscleGeometry
into a single Component that converts excitation vectors to joint torques.

:copyright: Copyright 2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from __future__ import annotations

import logging
from typing import Optional

import equinox as eqx
from equinox import Module, field
from equinox.nn import State
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

from feedbax.graph import Component
from feedbax.mechanics.geometry import TwoLinkArmMuscleGeometry
from feedbax.mechanics.muscles.thelen_muscle import RigidTendonHillMuscleThelen


logger = logging.getLogger(__name__)


class Arm6MuscleRigidTendon(Component):
    """Six-muscle arm with Thelen rigid tendon muscles.

    Combines six RigidTendonHillMuscleThelen muscles with a
    TwoLinkArmMuscleGeometry to produce joint torques from
    a 6-element excitation vector.

    Muscle arrangement (default):
        0: Shoulder flexor
        1: Shoulder extensor
        2: Elbow flexor
        3: Elbow extensor
        4: Biarticular flexor
        5: Biarticular extensor

    Attributes:
        muscles: Tuple of 6 Thelen muscle components.
        geometry: Two-link arm muscle geometry.
    """

    input_ports = ("excitation", "angles", "angular_velocities")
    output_ports = ("torques", "forces", "activations")

    muscles: tuple[RigidTendonHillMuscleThelen, ...]
    geometry: TwoLinkArmMuscleGeometry

    def __init__(
        self,
        dt: float = 0.01,
        max_isometric_force: float = 500.0,
        optimal_muscle_length: float = 0.1,
        tendon_slack_length: float = 0.1,
        geometry: Optional[TwoLinkArmMuscleGeometry] = None,
    ):
        """Initialize 6-muscle arm template.

        Args:
            dt: Integration timestep [s].
            max_isometric_force: Peak isometric force for each muscle [N].
            optimal_muscle_length: Optimal fiber length for each muscle [m].
            tendon_slack_length: Tendon slack length for each muscle [m].
            geometry: Muscle geometry. If None, uses default 6-muscle layout.
        """
        if geometry is None:
            geometry = TwoLinkArmMuscleGeometry.default_six_muscle()
        self.geometry = geometry

        n_muscles = geometry.n_muscles
        self.muscles = tuple(
            RigidTendonHillMuscleThelen(
                max_isometric_force=max_isometric_force,
                optimal_muscle_length=optimal_muscle_length,
                tendon_slack_length=tendon_slack_length,
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
        """Execute one step for all muscles, then compute torques.

        Args:
            inputs: Dict with 'excitation' [6], 'angles' [2],
                'angular_velocities' [2].
            state: Current State container.
            key: PRNG key.

        Returns:
            Outputs dict with 'torques' [2], 'forces' [6],
            'activations' [6], and updated state.
        """
        excitation = inputs.get("excitation", jnp.zeros(len(self.muscles)))
        angles = inputs.get("angles", jnp.zeros(2))
        angular_velocities = inputs.get("angular_velocities", jnp.zeros(2))

        # Compute geometry
        mt_lengths = self.geometry.musculotendon_lengths(angles)
        mt_velocities = self.geometry.musculotendon_velocities(
            angles, angular_velocities
        )

        # Run each muscle
        forces_list = []
        activations_list = []
        for i, muscle in enumerate(self.muscles):
            muscle_inputs = {
                "excitation": excitation[i],
                "musculotendon_length": mt_lengths[i],
                "musculotendon_velocity": mt_velocities[i],
            }
            muscle_outputs, state = muscle(muscle_inputs, state, key=key)
            forces_list.append(muscle_outputs["force"])
            activations_list.append(muscle_outputs["activation"])

        forces = jnp.stack(forces_list)
        activations = jnp.stack(activations_list)

        # Convert forces to joint torques via moment arm matrix
        torques = self.geometry.forces_to_torques(angles, forces)

        outputs = {
            "torques": torques,
            "forces": forces,
            "activations": activations,
        }
        return outputs, state
