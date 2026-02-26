"""Analytical musculoskeletal plant using pure JAX Lagrangian dynamics.

Implements a musculoskeletal plant model with:
- Two-link arm Lagrangian rigid-body dynamics (no MuJoCo dependency)
- Hill-type rigid-tendon muscle model with force-length-velocity curves
- First-order activation dynamics
- Constant moment arm muscle geometry

The ODE state has 10 variables: 2 angles + 2 angular velocities + 6 activations.
Fiber lengths and velocities are determined algebraically (rigid tendon assumption).

:copyright: Copyright 2024-2025 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import logging
from typing import Optional

import equinox as eqx
from equinox import Module, field
import jax
import jax.numpy as jnp
import jax.tree as jt
from jaxtyping import Array, Float, PRNGKeyArray, PyTree, Scalar

from feedbax.mechanics.body import BodyPreset
from feedbax.mechanics.hill_muscles import (
    ForceLengthCurve,
    ForceVelocityCurve,
    PassiveForceLengthCurve,
)
from feedbax.mechanics.muscle_config import MuscleTopology
from feedbax.mechanics.plant import AbstractPlant, DynamicsComponent, PlantState
from feedbax.mechanics.skeleton.arm import TwoLinkArm, TwoLinkArmState
from feedbax.state import CartesianState, StateBounds


logger = logging.getLogger(__name__)


# Specific tension: force per unit PCSA.  30 N/cm^2 matches MJXPlant.
SPECIFIC_TENSION = 30.0  # N/cm^2


class AnalyticalMuscleState(Module):
    """Muscle state for the analytical plant (rigid tendon).

    With rigid tendons, fiber lengths are algebraically determined from
    joint angles and moment arm geometry -- they are not ODE state variables.
    Only activations are integrated as part of the ODE.

    Attributes:
        activations: Muscle activation levels, shape ``(n_muscles,)``.
            Values in [0, 1].
    """

    activations: Float[Array, " n_muscles"]


class AnalyticalMusculoskeletalPlant(AbstractPlant):
    """Musculoskeletal plant with pure JAX analytical Lagrangian dynamics.

    Combines two-link arm rigid-body dynamics with Hill-type muscle models
    using rigid (inextensible) tendons.  The entire vector field --
    activation dynamics, muscle force computation, and skeleton dynamics --
    is computed monolithically in a single ``vector_field()`` call, making
    it fully differentiable through JAX without any MuJoCo dependency.

    The moment arm matrix uses the convention ``torques = R^T @ forces``
    where ``R`` has shape ``(n_muscles, n_joints)``.

    Attributes:
        skeleton: TwoLinkArm providing kinematics (FK, IK, effector, bounds).
        clip_states: Whether to clip states to bounds after integration.
        segment_lengths: Segment lengths from the body preset, shape
            ``(n_joints,)``.  Used for FK-based reachable target sampling.
        moment_arms: Signed moment arm matrix, shape ``(n_muscles, n_joints)``.
        muscle_gear: Maximum isometric force per muscle, shape ``(n_muscles,)``.
            Computed as ``PCSA * specific_tension``.
        optimal_fiber_length: Optimal fiber length per muscle, shape
            ``(n_muscles,)``.
        tendon_slack_length: Tendon slack length per muscle, shape
            ``(n_muscles,)``.
        mt_reference_length: Musculotendon reference length at zero joint
            angles, shape ``(n_muscles,)``.
        vmax: Maximum shortening velocity in optimal fiber lengths per
            second, shape ``(n_muscles,)``.
        tau_act: Activation time constant in seconds.
        tau_deact: Deactivation time constant in seconds.
    """

    skeleton: TwoLinkArm
    clip_states: bool = eqx.field(static=True)
    segment_lengths: Float[Array, " n_joints"]
    moment_arms: Float[Array, "n_muscles n_joints"]
    muscle_gear: Float[Array, " n_muscles"]
    optimal_fiber_length: Float[Array, " n_muscles"]
    tendon_slack_length: Float[Array, " n_muscles"]
    mt_reference_length: Float[Array, " n_muscles"]
    vmax: Float[Array, " n_muscles"]
    tau_act: float
    tau_deact: float
    force_length: ForceLengthCurve = field(static=True)
    passive_force_length: PassiveForceLengthCurve = field(static=True)
    force_velocity: ForceVelocityCurve = field(static=True)
    max_acceleration: float = 500.0  # rad/s^2 — safety clamp for ODE stability
    max_velocity: float = 50.0  # rad/s — safety clamp for ODE stability

    def __init__(
        self,
        skeleton: TwoLinkArm,
        segment_lengths: Array,
        moment_arms: Array,
        muscle_gear: Array,
        optimal_fiber_length: Array,
        tendon_slack_length: Array,
        mt_reference_length: Array,
        vmax: Array,
        tau_act: float = 0.01,
        tau_deact: float = 0.04,
        clip_states: bool = True,
        *,
        key: Optional[PRNGKeyArray] = None,
    ):
        """Initialize the analytical musculoskeletal plant.

        Args:
            skeleton: TwoLinkArm instance providing kinematics.
            segment_lengths: Segment lengths, shape ``(n_joints,)``.
            moment_arms: Signed moment arm matrix, shape ``(n_muscles, n_joints)``.
            muscle_gear: Max isometric force per muscle, shape ``(n_muscles,)``.
            optimal_fiber_length: Optimal fiber length per muscle, shape
                ``(n_muscles,)``.
            tendon_slack_length: Tendon slack length per muscle, shape
                ``(n_muscles,)``.
            mt_reference_length: MT reference length at zero angles, shape
                ``(n_muscles,)``.
            vmax: Max shortening velocity in opt. fiber lengths/s, shape
                ``(n_muscles,)``.
            tau_act: Activation time constant in seconds.
            tau_deact: Deactivation time constant in seconds.
            clip_states: Whether to clip states to bounds.
            key: Unused PRNG key (for interface compatibility).
        """
        self.skeleton = skeleton
        self.clip_states = clip_states
        self.segment_lengths = jnp.asarray(segment_lengths)
        self.moment_arms = jnp.asarray(moment_arms)
        self.muscle_gear = jnp.asarray(muscle_gear)
        self.optimal_fiber_length = jnp.asarray(optimal_fiber_length)
        self.tendon_slack_length = jnp.asarray(tendon_slack_length)
        self.mt_reference_length = jnp.asarray(mt_reference_length)
        self.vmax = jnp.asarray(vmax)
        self.tau_act = tau_act
        self.tau_deact = tau_deact
        self.force_length = ForceLengthCurve()
        self.passive_force_length = PassiveForceLengthCurve()
        self.force_velocity = ForceVelocityCurve()

    # ------------------------------------------------------------------
    # Muscle geometry (constant moment arms)
    # ------------------------------------------------------------------

    def _musculotendon_lengths(
        self, angles: Float[Array, " n_joints"],
    ) -> Float[Array, " n_muscles"]:
        """Compute musculotendon lengths from joint angles.

        Uses the linear model: ``L_mt = L_ref - R @ angles``.

        Args:
            angles: Joint angles, shape ``(n_joints,)``.

        Returns:
            MT lengths, shape ``(n_muscles,)``.
        """
        return self.mt_reference_length - self.moment_arms @ angles

    def _musculotendon_velocities(
        self,
        d_angles: Float[Array, " n_joints"],
    ) -> Float[Array, " n_muscles"]:
        """Compute musculotendon velocities from angular velocities.

        ``v_mt = -R @ d_angles`` (negative because positive moment arm
        and positive angular velocity means shortening).

        Args:
            d_angles: Angular velocities, shape ``(n_joints,)``.

        Returns:
            MT velocities, shape ``(n_muscles,)``.  Negative = shortening.
        """
        return -self.moment_arms @ d_angles

    def _fiber_lengths(
        self, mt_lengths: Float[Array, " n_muscles"],
    ) -> Float[Array, " n_muscles"]:
        """Compute fiber lengths from MT lengths (rigid tendon).

        ``L_fiber = L_mt - L_tendon_slack``

        Args:
            mt_lengths: Musculotendon lengths, shape ``(n_muscles,)``.

        Returns:
            Fiber lengths, shape ``(n_muscles,)``.
        """
        return mt_lengths - self.tendon_slack_length

    def _fiber_velocities(
        self, mt_velocities: Float[Array, " n_muscles"],
    ) -> Float[Array, " n_muscles"]:
        """Compute fiber velocities from MT velocities (rigid tendon).

        With zero pennation angle, ``v_fiber = v_mt``.

        Args:
            mt_velocities: MT velocities, shape ``(n_muscles,)``.

        Returns:
            Fiber velocities, shape ``(n_muscles,)``.
        """
        return mt_velocities

    # ------------------------------------------------------------------
    # Muscle force computation
    # ------------------------------------------------------------------

    def _compute_muscle_forces(
        self,
        activations: Float[Array, " n_muscles"],
        fiber_lengths: Float[Array, " n_muscles"],
        fiber_velocities: Float[Array, " n_muscles"],
    ) -> Float[Array, " n_muscles"]:
        """Compute muscle forces via Hill-type rigid tendon model.

        ``F = F0 * (activation * fl(l_norm) * fv(v_norm) + passive_fl(l_norm))``

        Args:
            activations: Muscle activations in [0, 1], shape ``(n_muscles,)``.
            fiber_lengths: Current fiber lengths, shape ``(n_muscles,)``.
            fiber_velocities: Current fiber velocities, shape ``(n_muscles,)``.

        Returns:
            Muscle forces (non-negative), shape ``(n_muscles,)``.
        """
        # Clamp fiber lengths to prevent numerical issues.
        min_length = 0.3 * self.optimal_fiber_length
        max_length = 2.0 * self.optimal_fiber_length
        clamped_fiber = jnp.clip(fiber_lengths, min_length, max_length)

        norm_length = clamped_fiber / self.optimal_fiber_length

        # Clamp velocity to reasonable range.
        max_vel = self.vmax * self.optimal_fiber_length
        clamped_velocity = jnp.clip(
            fiber_velocities, -max_vel, max_vel * 0.1,
        )
        norm_velocity = clamped_velocity / (
            self.vmax * self.optimal_fiber_length
        )

        # Force-length-velocity curves (vmapped over muscles).
        fl = jax.vmap(self.force_length)(norm_length)
        fv = jax.vmap(self.force_velocity)(norm_velocity)
        passive = jax.vmap(self.passive_force_length)(norm_length)

        force = self.muscle_gear * (activations * fl * fv + passive)
        return jnp.maximum(force, 0.0)

    # ------------------------------------------------------------------
    # Lagrangian dynamics
    # ------------------------------------------------------------------

    def _lagrangian_dynamics(
        self,
        angles: Float[Array, "2"],
        d_angles: Float[Array, "2"],
        torques: Float[Array, "2"],
    ) -> Float[Array, "2"]:
        """Compute angular accelerations via Lagrangian dynamics.

        Implements ``M(q) * ddq = tau - C(q, dq) - B * dq`` where M is the
        configuration-dependent mass matrix, C encodes Coriolis and centripetal
        effects, and B is the joint friction matrix.

        Args:
            angles: Joint angles, shape ``(2,)``.
            d_angles: Angular velocities, shape ``(2,)``.
            torques: Net joint torques, shape ``(2,)``.

        Returns:
            Angular accelerations, shape ``(2,)``.
        """
        # Reuse the precomputed inertia constants from the TwoLinkArm skeleton.
        # TwoLinkArm eagerly initializes _a in __init__, so this is safe.
        a0, a1, a2 = self.skeleton._a

        # Coriolis and centripetal torques.
        c_vec = (
            jnp.array((
                -d_angles[1] * (2 * d_angles[0] + d_angles[1]),
                d_angles[0] ** 2,
            ))
            * a1
            * jnp.sin(angles[1])
        )

        # Configuration-dependent mass matrix.
        cs1 = jnp.cos(angles[1])
        tmp = a2 + a1 * cs1
        M = jnp.array([
            [a0 + 2 * a1 * cs1, tmp],
            [tmp, a2 * jnp.ones_like(cs1)],
        ])

        # Net torque after subtracting Coriolis and friction.
        net_torque = torques - c_vec - self.skeleton.B @ d_angles

        # Angular acceleration via linear solve (more stable than inversion).
        return jnp.linalg.solve(M, net_torque)

    # ------------------------------------------------------------------
    # AbstractPlant interface
    # ------------------------------------------------------------------

    @jax.named_scope("fbx.AnalyticalMusculoskeletalPlant.vector_field")
    def vector_field(
        self,
        t: Scalar,
        state: PlantState,
        input: Float[Array, " n_muscles"],
    ) -> PlantState:
        """Compute time derivatives of the full plant state.

        Monolithic vector field computing activation dynamics, muscle forces
        (via rigid-tendon Hill model), and Lagrangian skeleton dynamics in
        a single pass.

        Args:
            t: Time (unused).
            state: Current plant state with ``skeleton`` (TwoLinkArmState)
                and ``muscles`` (AnalyticalMuscleState).
            input: Muscle excitations in [0, 1], shape ``(n_muscles,)``.

        Returns:
            PlantState containing time derivatives:
            - ``skeleton.angle`` = angular velocities
            - ``skeleton.d_angle`` = angular accelerations
            - ``muscles.activations`` = activation derivatives
        """
        excitations = jnp.clip(input, 0.0, 1.0)
        angles = state.skeleton.angle
        d_angles = jnp.clip(
            state.skeleton.d_angle, -self.max_velocity, self.max_velocity,
        )
        activations = state.muscles.activations

        # --- Activation dynamics ---
        # First-order filter with separate time constants for
        # activation (excitation > current) and deactivation.
        tau = jnp.where(
            excitations > activations, self.tau_act, self.tau_deact,
        )
        d_activations = (excitations - activations) / tau

        # --- Muscle geometry (rigid tendon) ---
        mt_lengths = self._musculotendon_lengths(angles)
        mt_velocities = self._musculotendon_velocities(d_angles)
        fiber_lengths = self._fiber_lengths(mt_lengths)
        fiber_velocities = self._fiber_velocities(mt_velocities)

        # --- Muscle forces ---
        forces = self._compute_muscle_forces(
            activations, fiber_lengths, fiber_velocities,
        )

        # --- Joint torques via moment arms ---
        joint_torques = self.moment_arms.T @ forces

        # --- Lagrangian dynamics ---
        dd_angles = self._lagrangian_dynamics(angles, d_angles, joint_torques)
        # Safety clamp: prevent extreme accelerations that blow up ODE integration.
        dd_angles = jnp.clip(
            dd_angles, -self.max_acceleration, self.max_acceleration,
        )

        return PlantState(
            skeleton=TwoLinkArmState(
                angle=d_angles,
                d_angle=dd_angles,
                torque=jnp.zeros(2),
            ),
            muscles=AnalyticalMuscleState(activations=d_activations),
        )

    @property
    def dynamics_spec(self) -> Mapping[str, DynamicsComponent[PlantState]]:
        """Empty -- vector_field is overridden directly (monolithic)."""
        return {}

    def kinematics_update(
        self,
        input: PyTree[Array],
        state: PlantState,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PlantState:
        """Apply instantaneous updates: clip angles and activations.

        Args:
            input: Muscle excitations (unused here).
            state: Current plant state.
            key: Unused.

        Returns:
            Plant state with clipped skeleton angles and muscle activations.
        """
        if not self.clip_states:
            return state

        # Clip skeleton state (joint angles to bounds).
        state = eqx.tree_at(
            lambda s: s.skeleton,
            state,
            self._clip_state(self.bounds.skeleton, state.skeleton),
        )

        # Clip activations to [0, 1].
        state = eqx.tree_at(
            lambda s: s.muscles.activations,
            state,
            jnp.clip(state.muscles.activations, 0.0, 1.0),
        )

        return state

    @property
    def bounds(self) -> PlantState:
        """State bounds for clipping.

        Returns skeleton bounds from TwoLinkArm and activation bounds [0, 1].
        """
        return PlantState(
            skeleton=self.skeleton.bounds,
            muscles=StateBounds(
                low=AnalyticalMuscleState(
                    activations=jnp.zeros(self.n_muscles),
                ),
                high=AnalyticalMuscleState(
                    activations=jnp.ones(self.n_muscles),
                ),
            ),
        )

    def init(self, *, key: PRNGKeyArray) -> PlantState:
        """Initialize the plant state at rest (zeros).

        Args:
            key: PRNG key (used by skeleton init).

        Returns:
            PlantState with zero angles, velocities, and activations.
        """
        return PlantState(
            skeleton=self.skeleton.init(key=key),
            muscles=AnalyticalMuscleState(
                activations=jnp.zeros(self.n_muscles),
            ),
        )

    @property
    def n_muscles(self) -> int:
        """Number of muscles."""
        return self.moment_arms.shape[0]

    @property
    def input_size(self) -> int:
        """Number of muscle inputs (policy output dimension).

        Returns ``n_muscles`` (not ``n_joints``), because the policy
        outputs per-muscle excitations.
        """
        return self.n_muscles

    # ------------------------------------------------------------------
    # Construction from BodyPreset
    # ------------------------------------------------------------------

    @classmethod
    def from_body_preset(
        cls,
        preset: BodyPreset,
        chain_config,
        *,
        clip_states: bool = True,
        key: Optional[PRNGKeyArray] = None,
    ) -> AnalyticalMusculoskeletalPlant:
        """Construct from a BodyPreset and chain configuration.

        Derives all physical parameters from the preset:
        - Segment inertia ``I = (1/3) * m * l^2`` (uniform rod approximation)
        - Center of mass ``s = l / 2``
        - Joint friction from ``joint_damping`` (diagonal B matrix)
        - Signed moment arms from preset magnitudes and topology signs
        - Muscle gear (max isometric force) from PCSA * specific tension
        - MT reference lengths from optimal fiber length + tendon slack length

        Args:
            preset: BodyPreset with physical parameters.
            chain_config: ChainConfig with topology (provides
                ``muscle_topology``).
            clip_states: Whether to clip states to joint limits.
            key: Optional PRNG key.

        Returns:
            Configured AnalyticalMusculoskeletalPlant.
        """
        n_joints = len(preset.segment_lengths)
        if n_joints != 2:
            raise ValueError(
                f"AnalyticalMusculoskeletalPlant requires exactly 2 joints, "
                f"got {n_joints}. For other topologies, use MJXPlant."
            )

        seg_l = jnp.asarray(preset.segment_lengths)
        seg_m = jnp.asarray(preset.segment_masses)

        # Uniform rod inertia: I = (1/3) * m * l^2
        seg_I = (1.0 / 3.0) * seg_m * seg_l ** 2
        # Center of mass at midpoint.
        seg_s = seg_l / 2.0

        # Joint friction: diagonal matrix from per-joint damping.
        damping = jnp.asarray(preset.joint_damping)
        B = jnp.diag(damping)

        skeleton = TwoLinkArm(
            l=seg_l,
            m=seg_m,
            I=seg_I,
            s=seg_s,
            B=B,
        )

        # Signed moment arms from preset magnitudes and topology.
        topology = chain_config.muscle_topology
        sign_arr = topology.sign_array
        routing_arr = topology.routing_array
        moment_arms = preset.muscle_moment_arm_magnitudes * sign_arr
        moment_arms = jnp.where(routing_arr, moment_arms, 0.0)

        # Muscle gear: max isometric force = PCSA * specific tension.
        muscle_gear = preset.muscle_pcsa * SPECIFIC_TENSION

        # MT reference length at zero angles.
        # For constant moment arms: L_mt(0) = L_ref, and at rest we want
        # the fiber to be near optimal length, so:
        # L_ref = optimal_fiber_length + tendon_slack_length
        ofl = jnp.asarray(preset.muscle_optimal_fiber_length)
        tsl = jnp.asarray(preset.muscle_tendon_slack_length)
        mt_ref = ofl + tsl

        # Default vmax: 10 optimal lengths per second (standard value).
        vmax = jnp.full_like(ofl, 10.0)

        return cls(
            skeleton=skeleton,
            segment_lengths=seg_l,
            moment_arms=moment_arms,
            muscle_gear=muscle_gear,
            optimal_fiber_length=ofl,
            tendon_slack_length=tsl,
            mt_reference_length=mt_ref,
            vmax=vmax,
            tau_act=float(preset.tau_act),
            tau_deact=float(preset.tau_deact),
            clip_states=clip_states,
            key=key,
        )

    @classmethod
    def build_batch(
        cls,
        presets: Sequence[BodyPreset],
        chain_config,
        *,
        clip_states: bool = True,
    ) -> AnalyticalMusculoskeletalPlant:
        """Build N plants from presets, stack into a vmappable batch.

        All presets must produce the same topology (same n_joints, n_muscles).
        Only numeric parameters (masses, lengths, damping, etc.) may differ.

        Args:
            presets: Sequence of BodyPreset instances.
            chain_config: ChainConfig with topology (shared).
            clip_states: Whether to clip states to joint limits.

        Returns:
            AnalyticalMusculoskeletalPlant with leading ``(B,)`` dimension
            on all array leaves.
        """
        plants = [
            cls.from_body_preset(p, chain_config, clip_states=clip_states)
            for p in presets
        ]

        # Validate topology consistency.
        ref = plants[0]
        for i, p in enumerate(plants[1:], 1):
            assert p.n_muscles == ref.n_muscles, (
                f"Body {i}: n_muscles={p.n_muscles} != {ref.n_muscles}"
            )

        # Stack all array leaves.
        all_flat = [jt.flatten(p) for p in plants]
        template_treedef = all_flat[0][1]
        stacked_leaves = [
            jnp.stack(leaves)
            for leaves in zip(*[f[0] for f in all_flat])
        ]
        return jt.unflatten(template_treedef, stacked_leaves)
