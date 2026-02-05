"""Muscle geometry and routing for musculoskeletal models.

This module provides geometric computations for muscle paths across joints,
including:
- Moment arm calculations
- Muscle-tendon length from joint angles
- Muscle velocity from joint velocities
- Point-to-point and wrapping geometries

:copyright: Copyright 2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from __future__ import annotations

from abc import abstractmethod
import logging
from typing import Optional

import equinox as eqx
from equinox import Module, field
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


logger = logging.getLogger(__name__)


# ============================================================================
# Muscle Path Geometry
# ============================================================================


class AbstractMuscleGeometry(Module):
    """Base class for muscle path geometry.

    Defines how muscle-tendon unit length and moment arms change
    with joint configuration.
    """

    @abstractmethod
    def musculotendon_length(self, angles: Array) -> Array:
        """Compute muscle-tendon length from joint angles.

        Args:
            angles: Joint angles [n_joints].

        Returns:
            Muscle-tendon length [m].
        """
        ...

    @abstractmethod
    def moment_arm(self, angles: Array) -> Array:
        """Compute moment arm(s) for each spanned joint.

        The moment arm relates muscle force to joint torque:
            torque = moment_arm * force

        Args:
            angles: Joint angles [n_joints].

        Returns:
            Moment arm(s) [m] for each relevant joint.
        """
        ...

    def musculotendon_velocity(
        self,
        angles: Array,
        angular_velocities: Array,
    ) -> Array:
        """Compute muscle-tendon velocity from joint velocities.

        Uses: v_mt = -r(theta) * d_theta (moment arm relationship)

        Args:
            angles: Joint angles [n_joints].
            angular_velocities: Joint angular velocities [n_joints].

        Returns:
            Muscle-tendon velocity [m/s]. Negative = shortening.
        """
        r = self.moment_arm(angles)
        # Moment arm sign convention: positive moment arm means
        # positive angular velocity causes muscle shortening
        return -jnp.sum(r * angular_velocities)


class ConstantMomentArmGeometry(AbstractMuscleGeometry):
    """Simple geometry with constant moment arm.

    Useful for muscles with approximately constant moment arms
    or for simplified models.

    Attributes:
        moment_arms: Constant moment arm for each spanned joint [m].
        reference_length: MT length at zero joint angles [m].
    """

    moment_arms: Float[Array, "n_joints"] = field(converter=jnp.asarray)
    reference_length: float

    def musculotendon_length(self, angles: Array) -> Array:
        """Compute MT length assuming linear relationship with angles.

        L_mt = L_ref - sum(r * theta)

        Args:
            angles: Joint angles.

        Returns:
            Muscle-tendon length.
        """
        return self.reference_length - jnp.sum(self.moment_arms * angles)

    def moment_arm(self, angles: Array) -> Array:
        """Return constant moment arms (independent of angle)."""
        return self.moment_arms


class PolynomialMomentArmGeometry(AbstractMuscleGeometry):
    """Geometry with polynomial moment arm functions.

    Moment arms are computed as polynomials of joint angle:
        r(theta) = sum_i(c_i * theta^i)

    Muscle-tendon length is the integral of the moment arm curve.

    Attributes:
        coefficients: Polynomial coefficients [n_joints, degree+1].
            coefficients[j, i] is the coefficient for theta^i for joint j.
        reference_length: MT length at zero angles [m].
    """

    coefficients: Float[Array, "n_joints degree"] = field(converter=jnp.asarray)
    reference_length: float

    def moment_arm(self, angles: Array) -> Array:
        """Compute moment arms as polynomial of joint angles.

        Args:
            angles: Joint angles [n_joints].

        Returns:
            Moment arms [n_joints].
        """
        n_joints = self.coefficients.shape[0]
        degree = self.coefficients.shape[1]

        # Build polynomial powers: [1, theta, theta^2, ...]
        def poly_eval(angle, coeffs):
            powers = angle ** jnp.arange(degree)
            return jnp.sum(coeffs * powers)

        return jax.vmap(poly_eval)(angles, self.coefficients)

    def musculotendon_length(self, angles: Array) -> Array:
        """Compute MT length by integrating moment arm.

        For polynomial moment arm, the integrated length change is:
            delta_L = -sum_j sum_i (c_{j,i} * theta_j^(i+1) / (i+1))

        Args:
            angles: Joint angles [n_joints].

        Returns:
            Muscle-tendon length [m].
        """
        n_joints = self.coefficients.shape[0]
        degree = self.coefficients.shape[1]

        def integrated_term(angle, coeffs):
            # Integrate: c_i * theta^i -> c_i * theta^(i+1) / (i+1)
            powers = jnp.arange(degree) + 1
            integrated = coeffs * (angle ** powers) / powers
            return jnp.sum(integrated)

        length_changes = jax.vmap(integrated_term)(angles, self.coefficients)
        return self.reference_length - jnp.sum(length_changes)


class WrappingGeometry(AbstractMuscleGeometry):
    """Geometry for muscles that wrap around cylindrical surfaces.

    Models muscle paths that wrap around joint centers or bony
    prominences, creating angle-dependent moment arms.

    Attributes:
        wrap_radius: Radius of wrapping surface [m].
        attachment_distance: Distance from wrap center to attachments [m].
        reference_length: MT length at zero wrap angle [m].
    """

    wrap_radius: float
    attachment_distances: Float[Array, "2"] = field(converter=jnp.asarray)
    reference_length: float
    joint_index: int = 0  # Which joint this wrapping applies to

    def _wrap_angle(self, joint_angle: Array) -> Array:
        """Compute the arc length of muscle contact with wrap surface.

        Args:
            joint_angle: Joint angle [rad].

        Returns:
            Wrap arc angle [rad].
        """
        # Simplified: wrap angle is proportional to joint angle
        # More complex models would compute actual tangent points
        return jnp.abs(joint_angle)

    def musculotendon_length(self, angles: Array) -> Array:
        """Compute MT length including wrap arc.

        L_mt = L_straight + r * wrap_angle

        Args:
            angles: Joint angles.

        Returns:
            Muscle-tendon length.
        """
        wrap_angle = self._wrap_angle(angles[self.joint_index])
        wrap_length = self.wrap_radius * wrap_angle
        return self.reference_length + wrap_length

    def moment_arm(self, angles: Array) -> Array:
        """Compute moment arm for wrapping geometry.

        For pure wrapping, moment arm equals wrap radius when in contact.

        Args:
            angles: Joint angles.

        Returns:
            Moment arms (only non-zero for wrapped joint).
        """
        n_joints = angles.shape[0]
        moment_arms = jnp.zeros(n_joints)
        # Moment arm = wrap_radius when wrapping
        is_wrapping = jnp.abs(angles[self.joint_index]) > 0.01
        moment_arms = moment_arms.at[self.joint_index].set(
            jnp.where(is_wrapping, self.wrap_radius, 0.0)
        )
        return moment_arms


# ============================================================================
# Two-Link Arm Muscle Geometry
# ============================================================================


class TwoLinkArmMuscleGeometry(Module):
    """Collection of muscle geometries for a two-link arm.

    Provides a standardized set of 6 muscles (3 per joint) commonly
    used in planar arm models:
    - Shoulder flexor/extensor
    - Elbow flexor/extensor
    - Biarticular flexor/extensor

    Attributes:
        geometries: List of AbstractMuscleGeometry for each muscle.
    """

    geometries: tuple[AbstractMuscleGeometry, ...]

    @classmethod
    def default_six_muscle(
        cls,
        shoulder_moment_arm: float = 0.04,
        elbow_moment_arm: float = 0.025,
        biarticular_shoulder_arm: float = 0.035,
        biarticular_elbow_arm: float = 0.022,
        reference_length: float = 0.2,
    ) -> "TwoLinkArmMuscleGeometry":
        """Create standard 6-muscle geometry for two-link arm.

        Muscle arrangement:
            0: Shoulder flexor (positive moment arm at shoulder)
            1: Shoulder extensor (negative moment arm at shoulder)
            2: Elbow flexor (positive moment arm at elbow)
            3: Elbow extensor (negative moment arm at elbow)
            4: Biarticular flexor (positive at both joints)
            5: Biarticular extensor (negative at both joints)

        Args:
            shoulder_moment_arm: Moment arm for uniarticular shoulder muscles.
            elbow_moment_arm: Moment arm for uniarticular elbow muscles.
            biarticular_shoulder_arm: Shoulder moment arm for biarticular muscles.
            biarticular_elbow_arm: Elbow moment arm for biarticular muscles.
            reference_length: Reference MT length at neutral position.

        Returns:
            TwoLinkArmMuscleGeometry instance.
        """
        geometries = (
            # Shoulder flexor
            ConstantMomentArmGeometry(
                moment_arms=jnp.array([shoulder_moment_arm, 0.0]),
                reference_length=reference_length,
            ),
            # Shoulder extensor
            ConstantMomentArmGeometry(
                moment_arms=jnp.array([-shoulder_moment_arm, 0.0]),
                reference_length=reference_length,
            ),
            # Elbow flexor
            ConstantMomentArmGeometry(
                moment_arms=jnp.array([0.0, elbow_moment_arm]),
                reference_length=reference_length,
            ),
            # Elbow extensor
            ConstantMomentArmGeometry(
                moment_arms=jnp.array([0.0, -elbow_moment_arm]),
                reference_length=reference_length,
            ),
            # Biarticular flexor
            ConstantMomentArmGeometry(
                moment_arms=jnp.array([biarticular_shoulder_arm, biarticular_elbow_arm]),
                reference_length=reference_length * 1.5,  # Longer biarticular
            ),
            # Biarticular extensor
            ConstantMomentArmGeometry(
                moment_arms=jnp.array([-biarticular_shoulder_arm, -biarticular_elbow_arm]),
                reference_length=reference_length * 1.5,
            ),
        )
        return cls(geometries=geometries)

    def musculotendon_lengths(self, angles: Array) -> Array:
        """Compute all muscle-tendon lengths.

        Args:
            angles: Joint angles [2].

        Returns:
            MT lengths [n_muscles].
        """
        return jnp.array([g.musculotendon_length(angles) for g in self.geometries])

    def moment_arms(self, angles: Array) -> Array:
        """Compute moment arm matrix.

        Args:
            angles: Joint angles [2].

        Returns:
            Moment arm matrix [n_muscles, n_joints].
        """
        return jnp.stack([g.moment_arm(angles) for g in self.geometries])

    def musculotendon_velocities(
        self,
        angles: Array,
        angular_velocities: Array,
    ) -> Array:
        """Compute all muscle-tendon velocities.

        Args:
            angles: Joint angles [2].
            angular_velocities: Joint angular velocities [2].

        Returns:
            MT velocities [n_muscles]. Negative = shortening.
        """
        return jnp.array([
            g.musculotendon_velocity(angles, angular_velocities)
            for g in self.geometries
        ])

    @property
    def n_muscles(self) -> int:
        """Number of muscles."""
        return len(self.geometries)

    def forces_to_torques(self, angles: Array, forces: Array) -> Array:
        """Convert muscle forces to joint torques.

        torque = R^T @ forces (moment arm matrix transpose)

        Args:
            angles: Joint angles [2].
            forces: Muscle forces [n_muscles].

        Returns:
            Joint torques [2].
        """
        R = self.moment_arms(angles)  # [n_muscles, n_joints]
        return R.T @ forces  # [n_joints]


# ============================================================================
# Point Mass Radial Geometry
# ============================================================================


class PointMassRadialGeometry(Module):
    """Radial geometry for muscles around a 2D point mass.

    Arranges antagonist pairs of muscles at evenly-spaced angles.
    Default configuration: 4 pairs (8 muscles) at 0, 45, 90, 135 degrees.
    Each pair has a positive and negative direction muscle.
    Directions are interleaved: [pos0, neg0, pos1, neg1, ...].

    Attributes:
        n_muscles: Total number of muscles (2 * n_pairs).
        directions: Unit direction vectors for each muscle [n_muscles, 2].
    """

    n_muscles: int = field(static=True)
    directions: Float[Array, "n_muscles 2"]

    def __init__(self, n_pairs: int = 4):
        """Initialize radial geometry.

        Args:
            n_pairs: Number of antagonist pairs.
        """
        self.n_muscles = 2 * n_pairs
        angles = jnp.linspace(0, jnp.pi, n_pairs, endpoint=False)
        pos_dirs = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)
        neg_dirs = -pos_dirs

        # Interleave: [pos0, neg0, pos1, neg1, ...]
        interleaved = jnp.empty((2 * n_pairs, 2))
        for i in range(n_pairs):
            interleaved = interleaved.at[2 * i].set(pos_dirs[i])
            interleaved = interleaved.at[2 * i + 1].set(neg_dirs[i])
        self.directions = interleaved

    def forces_to_force_2d(self, forces: Array) -> Array:
        """Convert individual muscle forces to a 2D net force vector.

        F_2d = sum(force_i * direction_i)

        Args:
            forces: Muscle forces [n_muscles].

        Returns:
            Net 2D force vector [2].
        """
        return jnp.sum(forces[:, None] * self.directions, axis=0)
