"""Musculoskeletal arm model combining skeleton and Hill muscles.

This module integrates:
- Two-link arm rigid body dynamics
- Hill-type muscle models (rigid or compliant tendon)
- Muscle geometry (moment arms)

The result is a realistic neuromechanical arm model suitable for
motor control studies and neural network training.

:copyright: Copyright 2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from __future__ import annotations

from functools import cached_property
import logging
from typing import Optional, Type

import diffrax as dfx
import equinox as eqx
from equinox import Module, field
from equinox.nn import State, StateIndex
import jax
import jax.numpy as jnp
import jax.tree as jt
from jaxtyping import Array, Float, PRNGKeyArray, PyTree, Scalar
import optimistix as optx

from feedbax.graph import Component, init_state_from_component
from feedbax.mechanics.dae import DAEComponent, DAEState
from feedbax.mechanics.geometry import TwoLinkArmMuscleGeometry, AbstractMuscleGeometry
from feedbax.mechanics.hill_muscles import (
    HillMuscleParams,
    HillMuscleState,
    RigidTendonHillMuscle,
    CompliantTendonHillMuscle,
    ActivationDynamics,
    ForceLengthCurve,
    PassiveForceLengthCurve,
    ForceVelocityCurve,
    TendonForceLengthCurve,
)
from feedbax.mechanics.skeleton.arm import TwoLinkArmState
from feedbax.state import CartesianState


logger = logging.getLogger(__name__)


# ============================================================================
# Musculoskeletal State
# ============================================================================


class MusculoskeletalState(Module):
    """Combined state for musculoskeletal arm model.

    Attributes:
        arm: Two-link arm configuration state.
        activations: Muscle activation levels [n_muscles].
        fiber_lengths: Muscle fiber lengths [n_muscles] (for compliant tendon).
        fiber_velocities: Muscle fiber velocities [n_muscles].
        forces: Muscle forces [n_muscles].
    """

    arm: TwoLinkArmState
    activations: Float[Array, "n_muscles"]
    fiber_lengths: Float[Array, "n_muscles"]
    fiber_velocities: Float[Array, "n_muscles"]
    forces: Float[Array, "n_muscles"]


# ============================================================================
# Musculoskeletal Arm with Rigid Tendons
# ============================================================================


class RigidTendonMusculoskeletalArm(Component):
    """Musculoskeletal arm with rigid tendon muscles.

    Combines two-link arm dynamics with Hill muscle models using
    rigid (inextensible) tendons. This is simpler than compliant
    tendon models and avoids stiff dynamics.

    The model uses:
    - 6 muscles: 2 uniarticular per joint + 2 biarticular
    - Constant moment arms (simplified geometry)
    - First-order activation dynamics

    Inputs:
        excitations: Neural excitation signals [n_muscles].

    Outputs:
        effector: End effector Cartesian state.
        state: Full musculoskeletal state.
        forces: Muscle forces [n_muscles].
        torques: Joint torques [n_joints].
    """

    input_ports = ("excitations",)
    output_ports = ("effector", "state", "forces", "torques")

    # Arm parameters
    arm_l: Float[Array, "2"]
    arm_m: Float[Array, "2"]
    arm_I: Float[Array, "2"]
    arm_s: Float[Array, "2"]
    arm_B: Float[Array, "2 2"]

    # Muscle parameters
    muscle_params: tuple[HillMuscleParams, ...]
    geometry: TwoLinkArmMuscleGeometry
    activation_dynamics: ActivationDynamics

    # Integration
    dt: float
    solver: dfx.AbstractSolver = field(static=True)
    state_index: StateIndex
    _initial_state: MusculoskeletalState = field(static=True)

    # Curve instances (shared across muscles)
    force_length: ForceLengthCurve = field(static=True)
    passive_force_length: PassiveForceLengthCurve = field(static=True)
    force_velocity: ForceVelocityCurve = field(static=True)

    def __init__(
        self,
        n_muscles: int = 6,
        muscle_params: Optional[tuple[HillMuscleParams, ...]] = None,
        geometry: Optional[TwoLinkArmMuscleGeometry] = None,
        arm_l=(0.30, 0.33),
        arm_m=(1.4, 1.0),
        arm_I=(0.025, 0.045),
        arm_s=(0.11, 0.16),
        arm_B=((0.05, 0.025), (0.025, 0.05)),
        dt: float = 0.01,
        solver_type: Type[dfx.AbstractSolver] = dfx.Kvaerno3,
        *,
        key: Optional[PRNGKeyArray] = None,
    ):
        """Initialize musculoskeletal arm.

        Args:
            n_muscles: Number of muscles (default 6 for standard arrangement).
            muscle_params: Tuple of HillMuscleParams for each muscle.
                If None, uses default parameters.
            geometry: Muscle geometry. If None, uses default 6-muscle geometry.
            arm_l, arm_m, arm_I, arm_s, arm_B: Arm physical parameters.
            dt: Integration timestep.
            solver_type: Diffrax solver type.
            key: PRNG key.
        """
        # Store arm parameters
        self.arm_l = jnp.asarray(arm_l)
        self.arm_m = jnp.asarray(arm_m)
        self.arm_I = jnp.asarray(arm_I)
        self.arm_s = jnp.asarray(arm_s)
        self.arm_B = jnp.asarray(arm_B)

        # Default geometry
        if geometry is None:
            geometry = TwoLinkArmMuscleGeometry.default_six_muscle()
        self.geometry = geometry
        actual_n_muscles = geometry.n_muscles

        # Default muscle parameters
        if muscle_params is None:
            muscle_params = tuple(
                HillMuscleParams(
                    max_isometric_force=500.0,  # N
                    optimal_fiber_length=0.08,  # m
                    tendon_slack_length=0.12,  # m (rigid, but needed for reference)
                    pennation_angle=0.0,
                    tau_activation=0.01,
                    tau_deactivation=0.04,
                    vmax=10.0,
                )
                for _ in range(actual_n_muscles)
            )
        self.muscle_params = muscle_params

        # Activation dynamics
        self.activation_dynamics = ActivationDynamics(
            tau_activation=0.01,
            tau_deactivation=0.04,
        )

        # Shared curves
        self.force_length = ForceLengthCurve()
        self.passive_force_length = PassiveForceLengthCurve()
        self.force_velocity = ForceVelocityCurve()

        # Integration
        self.dt = dt
        self.solver = solver_type()

        if key is None:
            key = jax.random.PRNGKey(0)

        # Initialize state
        init_arm = TwoLinkArmState()
        init_activations = jnp.zeros(actual_n_muscles)
        init_mt_lengths = self.geometry.musculotendon_lengths(init_arm.angle)
        init_fiber_lengths = jnp.array([
            init_mt_lengths[i] - mp.tendon_slack_length
            for i, mp in enumerate(self.muscle_params)
        ])

        self._initial_state = MusculoskeletalState(
            arm=init_arm,
            activations=init_activations,
            fiber_lengths=init_fiber_lengths,
            fiber_velocities=jnp.zeros(actual_n_muscles),
            forces=jnp.zeros(actual_n_muscles),
        )
        self.state_index = StateIndex(self._initial_state)

    @cached_property
    def _arm_a(self) -> tuple[Array, Array, Array]:
        """Precomputed arm inertia constants."""
        return (
            self.arm_I[0] + self.arm_I[1] + self.arm_m[1] * self.arm_l[0] ** 2,
            self.arm_m[1] * self.arm_l[0] * self.arm_s[1],
            self.arm_I[1],
        )

    @cached_property
    def _term(self) -> dfx.ODETerm:
        """ODE term for integration."""
        # Type mismatch: diffrax uses RealScalarLike, we use jaxtyping.Scalar
        return dfx.ODETerm(self._vector_field)  # type: ignore[arg-type]

    @property
    def n_muscles(self) -> int:
        """Number of muscles."""
        return self.geometry.n_muscles

    def _compute_muscle_forces(
        self,
        activations: Array,
        fiber_lengths: Array,
        fiber_velocities: Array,
        mt_lengths: Array,
    ) -> Array:
        """Compute forces for all muscles.

        Args:
            activations: Muscle activations [n_muscles].
            fiber_lengths: Fiber lengths [n_muscles].
            fiber_velocities: Fiber velocities [n_muscles].
            mt_lengths: Musculotendon lengths [n_muscles].

        Returns:
            Muscle forces [n_muscles].
        """
        forces = []
        for i, mp in enumerate(self.muscle_params):
            # Clamp fiber length to prevent numerical issues
            min_length = 0.3 * mp.optimal_fiber_length
            max_length = 2.0 * mp.optimal_fiber_length
            clamped_fiber_length = jnp.clip(fiber_lengths[i], min_length, max_length)

            # Normalized quantities
            norm_length = clamped_fiber_length / mp.optimal_fiber_length

            # Clamp velocity to reasonable range
            max_vel = mp.vmax * mp.optimal_fiber_length
            clamped_velocity = jnp.clip(fiber_velocities[i], -max_vel, max_vel * 0.1)
            norm_velocity = clamped_velocity / (mp.vmax * mp.optimal_fiber_length)

            # Force-length-velocity
            fl = self.force_length(norm_length)
            fv = self.force_velocity(norm_velocity)
            passive = self.passive_force_length(norm_length)

            # Total force (ensure non-negative)
            force = mp.max_isometric_force * (activations[i] * fl * fv + passive)
            force = jnp.maximum(force, 0.0)
            forces.append(force)

        return jnp.array(forces)

    def _vector_field(
        self,
        t: Scalar,
        state: MusculoskeletalState,
        excitations: Array,
    ) -> MusculoskeletalState:
        """Compute time derivatives of musculoskeletal state.

        Args:
            t: Current time.
            state: Current state.
            excitations: Neural excitations [n_muscles].

        Returns:
            State derivatives.
        """
        arm = state.arm
        angle, d_angle = arm.angle, arm.d_angle

        # Activation dynamics
        d_activations = jax.vmap(self.activation_dynamics)(
            excitations, state.activations
        )

        # Muscle geometry
        mt_lengths = self.geometry.musculotendon_lengths(angle)
        mt_velocities = self.geometry.musculotendon_velocities(angle, d_angle)
        moment_arms = self.geometry.moment_arms(angle)

        # For rigid tendon, fiber length tracks MT length
        # d_fiber_length = d_mt_length (approximately)
        d_fiber_lengths = mt_velocities

        # Compute muscle forces
        forces = self._compute_muscle_forces(
            state.activations,
            state.fiber_lengths,
            state.fiber_velocities,
            mt_lengths,
        )

        # Muscle forces to joint torques
        torques = moment_arms.T @ forces  # [n_joints]

        # Arm dynamics
        c_vec = (
            jnp.array((-d_angle[1] * (2 * d_angle[0] + d_angle[1]), d_angle[0] ** 2))
            * self._arm_a[1]
            * jnp.sin(angle[1])
        )

        cs1 = jnp.cos(angle[1])
        tmp = self._arm_a[2] + self._arm_a[1] * cs1
        inertia_mat = jnp.array([
            [self._arm_a[0] + 2 * self._arm_a[1] * cs1, tmp],
            [tmp, self._arm_a[2] * jnp.ones_like(cs1)],
        ])

        net_torque = torques - c_vec.T - jnp.matmul(d_angle, self.arm_B.T)
        dd_angle = jnp.linalg.solve(inertia_mat, net_torque)

        d_arm = TwoLinkArmState(
            angle=d_angle,
            d_angle=dd_angle,
            torque=jnp.zeros(2),
        )

        return MusculoskeletalState(
            arm=d_arm,
            activations=d_activations,
            fiber_lengths=d_fiber_lengths,
            fiber_velocities=jnp.zeros_like(state.fiber_velocities),  # Not tracked
            forces=jnp.zeros_like(forces),  # Derivative not meaningful
        )

    def __call__(
        self,
        inputs: dict[str, PyTree],
        state: State,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, PyTree], State]:
        """Execute one integration step.

        Args:
            inputs: Dict with 'excitations' [n_muscles].
            state: Current State container.
            key: PRNG key.

        Returns:
            Outputs dict and updated state.
        """
        ms_state: MusculoskeletalState = state.get(self.state_index)
        excitations = inputs.get("excitations", jnp.zeros(self.n_muscles))

        # Integrate one step
        # For simplicity, use Euler step here
        derivatives = self._vector_field(jnp.array(0.0), ms_state, excitations)

        new_arm = TwoLinkArmState(
            angle=ms_state.arm.angle + self.dt * derivatives.arm.angle,
            d_angle=ms_state.arm.d_angle + self.dt * derivatives.arm.d_angle,
            torque=jnp.zeros(2),
        )
        new_activations = ms_state.activations + self.dt * derivatives.activations
        new_activations = jnp.clip(new_activations, 0.0, 1.0)
        new_fiber_lengths = ms_state.fiber_lengths + self.dt * derivatives.fiber_lengths

        # Recompute forces at new state
        mt_lengths = self.geometry.musculotendon_lengths(new_arm.angle)
        mt_velocities = self.geometry.musculotendon_velocities(new_arm.angle, new_arm.d_angle)
        new_forces = self._compute_muscle_forces(
            new_activations,
            new_fiber_lengths,
            mt_velocities,
            mt_lengths,
        )

        new_state = MusculoskeletalState(
            arm=new_arm,
            activations=new_activations,
            fiber_lengths=new_fiber_lengths,
            fiber_velocities=mt_velocities,
            forces=new_forces,
        )

        state = state.set(self.state_index, new_state)

        # Compute outputs
        effector = self._effector(new_arm)
        torques = self.geometry.moment_arms(new_arm.angle).T @ new_forces

        outputs = {
            "effector": effector,
            "state": new_state,
            "forces": new_forces,
            "torques": torques,
        }

        return outputs, state

    def _forward_pos(self, angle: Array) -> tuple[Array, Array]:
        """Forward kinematics position computation."""
        angle_sum = jnp.cumsum(angle)
        length_components = self.arm_l * jnp.array([
            jnp.cos(angle_sum),
            jnp.sin(angle_sum),
        ])
        xy_pos = jnp.cumsum(length_components, axis=1)
        return xy_pos.T, length_components

    def _effector(self, arm_state: TwoLinkArmState) -> CartesianState:
        """Compute effector Cartesian state."""
        xy_pos, length_components = self._forward_pos(arm_state.angle)

        from feedbax.misc import SINCOS_GRAD_SIGNS
        ang_vel_sum = jnp.cumsum(arm_state.d_angle)
        xy_vel = jnp.cumsum(
            SINCOS_GRAD_SIGNS[1] * length_components[:, ::-1] * ang_vel_sum,
            axis=1,
        ).T

        effector = CartesianState(
            pos=xy_pos[-1],
            vel=xy_vel[-1],
            force=jnp.zeros(2),
        )
        return effector

    def state_view(self, state: State) -> MusculoskeletalState:
        """Return the musculoskeletal state."""
        return state.get(self.state_index)


# ============================================================================
# Musculoskeletal Arm with Compliant Tendons (DAE)
# ============================================================================


class CompliantTendonMusculoskeletalArm(DAEComponent[MusculoskeletalState]):
    """Musculoskeletal arm with compliant tendon muscles.

    This is a full DAE model where tendon compliance introduces
    stiff dynamics requiring implicit integration.

    Uses Kvaerno5 by default for handling stiff tendon dynamics.

    Warning:
        This is computationally more expensive than the rigid tendon
        model and requires careful tuning of solver tolerances.
    """

    input_ports = ("excitations",)
    output_ports = ("effector", "state", "forces", "torques")

    # Arm parameters
    arm_l: Float[Array, "2"]
    arm_m: Float[Array, "2"]
    arm_I: Float[Array, "2"]
    arm_s: Float[Array, "2"]
    arm_B: Float[Array, "2 2"]

    # Muscles
    muscle_params: tuple[HillMuscleParams, ...]
    geometry: TwoLinkArmMuscleGeometry

    # Curves
    force_length: ForceLengthCurve = field(static=True)
    passive_force_length: PassiveForceLengthCurve = field(static=True)
    force_velocity: ForceVelocityCurve = field(static=True)
    tendon_force_length: TendonForceLengthCurve = field(static=True)

    def __init__(
        self,
        muscle_params: Optional[tuple[HillMuscleParams, ...]] = None,
        geometry: Optional[TwoLinkArmMuscleGeometry] = None,
        arm_l=(0.30, 0.33),
        arm_m=(1.4, 1.0),
        arm_I=(0.025, 0.045),
        arm_s=(0.11, 0.16),
        arm_B=((0.05, 0.025), (0.025, 0.05)),
        dt: float = 0.001,  # Smaller timestep for stiff dynamics
        solver_type: Type[dfx.AbstractSolver] = dfx.Euler,  # Explicit for now
        root_finder: Optional[optx.AbstractRootFinder] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ):
        """Initialize compliant tendon musculoskeletal arm.

        Args:
            muscle_params: Muscle parameters.
            geometry: Muscle geometry.
            arm_*: Arm physical parameters.
            dt: Integration timestep (smaller than rigid tendon).
            solver_type: Implicit solver (default Kvaerno5).
            root_finder: Root finder with tight tolerances.
            key: PRNG key.
        """
        # Store arm parameters
        self.arm_l = jnp.asarray(arm_l)
        self.arm_m = jnp.asarray(arm_m)
        self.arm_I = jnp.asarray(arm_I)
        self.arm_s = jnp.asarray(arm_s)
        self.arm_B = jnp.asarray(arm_B)

        # Geometry
        if geometry is None:
            geometry = TwoLinkArmMuscleGeometry.default_six_muscle()
        self.geometry = geometry

        # Muscle parameters
        if muscle_params is None:
            muscle_params = tuple(
                HillMuscleParams(
                    max_isometric_force=500.0,
                    optimal_fiber_length=0.08,
                    tendon_slack_length=0.12,
                    pennation_angle=0.0,
                    tau_activation=0.01,
                    tau_deactivation=0.04,
                    vmax=10.0,
                )
                for _ in range(geometry.n_muscles)
            )
        self.muscle_params = muscle_params

        # Curves
        self.force_length = ForceLengthCurve()
        self.passive_force_length = PassiveForceLengthCurve()
        self.force_velocity = ForceVelocityCurve()
        self.tendon_force_length = TendonForceLengthCurve()

        if root_finder is None:
            root_finder = optx.Newton(rtol=1e-8, atol=1e-8)

        super().__init__(
            dt=dt,
            solver_type=solver_type,
            root_finder=root_finder,
            key=key,
        )

    @property
    def n_muscles(self) -> int:
        return self.geometry.n_muscles

    @cached_property
    def _arm_a(self) -> tuple[Array, Array, Array]:
        return (
            self.arm_I[0] + self.arm_I[1] + self.arm_m[1] * self.arm_l[0] ** 2,
            self.arm_m[1] * self.arm_l[0] * self.arm_s[1],
            self.arm_I[1],
        )

    def _compute_tendon_force(self, fiber_length: Array, mt_length: Array, mp: HillMuscleParams) -> Array:
        """Compute tendon force from fiber and MT lengths."""
        # Clamp fiber length to reasonable range
        min_fiber = 0.3 * mp.optimal_fiber_length
        max_fiber = 2.0 * mp.optimal_fiber_length
        clamped_fiber = jnp.clip(fiber_length, min_fiber, max_fiber)

        tendon_length = mt_length - clamped_fiber * jnp.cos(mp.pennation_angle)
        # Ensure tendon length is non-negative
        tendon_length = jnp.maximum(tendon_length, 0.0)

        norm_tendon = tendon_length / mp.tendon_slack_length
        norm_force = self.tendon_force_length(norm_tendon)
        # Clamp force to prevent inf/overflow
        norm_force = jnp.clip(norm_force, 0.0, 10.0)  # Max 10x isometric
        return mp.max_isometric_force * norm_force

    def _compute_fiber_velocity(
        self,
        activation: Array,
        fiber_length: Array,
        mt_length: Array,
        mp: HillMuscleParams,
    ) -> Array:
        """Compute fiber velocity from force equilibrium."""
        # Tendon force
        tendon_force = self._compute_tendon_force(fiber_length, mt_length, mp)

        # Required fiber force
        cos_alpha = jnp.cos(mp.pennation_angle)
        required_fiber_force = tendon_force / cos_alpha

        # Force-length
        norm_length = fiber_length / mp.optimal_fiber_length
        fl = self.force_length(norm_length)
        passive = self.passive_force_length(norm_length)

        # Solve for velocity
        norm_required = required_fiber_force / mp.max_isometric_force
        active_required = jnp.maximum(norm_required - passive, 0.0)

        afl = jnp.maximum(activation * fl, 1e-6)
        fv_required = jnp.clip(active_required / afl, 0.0, 1.4)

        # Invert force-velocity
        a = self.force_velocity.concentric_curvature
        vmax = self.force_velocity.max_shortening_velocity
        norm_velocity = (fv_required - 1.0) / (1.0 / a + fv_required / (a * vmax))
        norm_velocity = jnp.clip(norm_velocity, -vmax, vmax * 0.1)

        return norm_velocity * mp.vmax * mp.optimal_fiber_length

    @jax.named_scope("fbx.CompliantTendonMusculoskeletalArm.vector_field")
    def vector_field(
        self,
        t: Scalar,
        state: MusculoskeletalState,
        input: Array,
    ) -> MusculoskeletalState:
        """Compute derivatives for compliant tendon model."""
        excitations = input  # Rename for clarity within method
        arm = state.arm
        angle, d_angle = arm.angle, arm.d_angle

        # Activation dynamics
        def activation_deriv(exc, act, mp):
            tau = jnp.where(exc > act, mp.tau_activation, mp.tau_deactivation)
            return (exc - act) / tau

        d_activations = jnp.array([
            activation_deriv(excitations[i], state.activations[i], self.muscle_params[i])
            for i in range(self.n_muscles)
        ])

        # Muscle geometry
        mt_lengths = self.geometry.musculotendon_lengths(angle)
        moment_arms = self.geometry.moment_arms(angle)

        # Fiber velocities (from force equilibrium)
        fiber_velocities = jnp.array([
            self._compute_fiber_velocity(
                state.activations[i],
                state.fiber_lengths[i],
                mt_lengths[i],
                self.muscle_params[i],
            )
            for i in range(self.n_muscles)
        ])

        # Muscle forces
        forces = []
        for i, mp in enumerate(self.muscle_params):
            norm_length = state.fiber_lengths[i] / mp.optimal_fiber_length
            norm_velocity = fiber_velocities[i] / (mp.vmax * mp.optimal_fiber_length)
            fl = self.force_length(norm_length)
            fv = self.force_velocity(norm_velocity)
            passive = self.passive_force_length(norm_length)
            force = mp.max_isometric_force * (state.activations[i] * fl * fv + passive)
            forces.append(force)
        forces = jnp.array(forces)

        # Joint torques
        torques = moment_arms.T @ forces

        # Arm dynamics
        c_vec = (
            jnp.array((-d_angle[1] * (2 * d_angle[0] + d_angle[1]), d_angle[0] ** 2))
            * self._arm_a[1]
            * jnp.sin(angle[1])
        )

        cs1 = jnp.cos(angle[1])
        tmp = self._arm_a[2] + self._arm_a[1] * cs1
        inertia_mat = jnp.array([
            [self._arm_a[0] + 2 * self._arm_a[1] * cs1, tmp],
            [tmp, self._arm_a[2] * jnp.ones_like(cs1)],
        ])

        net_torque = torques - c_vec.T - jnp.matmul(d_angle, self.arm_B.T)
        dd_angle = jnp.linalg.solve(inertia_mat, net_torque)

        d_arm = TwoLinkArmState(
            angle=d_angle,
            d_angle=dd_angle,
            torque=jnp.zeros(2),
        )

        return MusculoskeletalState(
            arm=d_arm,
            activations=d_activations,
            fiber_lengths=fiber_velocities,  # d_fiber_length = velocity
            fiber_velocities=jnp.zeros_like(fiber_velocities),
            forces=jnp.zeros_like(forces),
        )

    def init_system_state(self, *, key: PRNGKeyArray) -> MusculoskeletalState:
        """Initialize at rest position."""
        init_arm = TwoLinkArmState()
        mt_lengths = self.geometry.musculotendon_lengths(init_arm.angle)

        init_fiber_lengths = jnp.array([
            mp.optimal_fiber_length for mp in self.muscle_params
        ])

        return MusculoskeletalState(
            arm=init_arm,
            activations=jnp.zeros(self.n_muscles),
            fiber_lengths=init_fiber_lengths,
            fiber_velocities=jnp.zeros(self.n_muscles),
            forces=jnp.zeros(self.n_muscles),
        )

    def extract_outputs(self, state: MusculoskeletalState) -> dict[str, PyTree]:
        """Extract outputs from state."""
        return {}  # Filled in __call__

    def _get_zero_input(self) -> Array:
        return jnp.zeros(self.n_muscles)

    @property
    def input_size(self) -> int:
        return self.n_muscles

    def __call__(
        self,
        inputs: dict[str, PyTree],
        state: State,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, PyTree], State]:
        """Execute integration step."""
        excitations = inputs.get("excitations", jnp.zeros(self.n_muscles))
        modified_inputs = {"input": excitations}
        outputs, state = super().__call__(modified_inputs, state, key=key)

        # Compute additional outputs
        dae_state = state.get(self.state_index)
        ms_state = dae_state.system

        effector = self._effector(ms_state.arm)
        mt_lengths = self.geometry.musculotendon_lengths(ms_state.arm.angle)

        forces = jnp.array([
            self._compute_tendon_force(
                ms_state.fiber_lengths[i],
                mt_lengths[i],
                self.muscle_params[i],
            )
            for i in range(self.n_muscles)
        ])
        torques = self.geometry.moment_arms(ms_state.arm.angle).T @ forces

        outputs.update({
            "effector": effector,
            "forces": forces,
            "torques": torques,
        })

        return outputs, state

    def _forward_pos(self, angle: Array) -> tuple[Array, Array]:
        angle_sum = jnp.cumsum(angle)
        length_components = self.arm_l * jnp.array([
            jnp.cos(angle_sum),
            jnp.sin(angle_sum),
        ])
        xy_pos = jnp.cumsum(length_components, axis=1)
        return xy_pos.T, length_components

    def _effector(self, arm_state: TwoLinkArmState) -> CartesianState:
        xy_pos, length_components = self._forward_pos(arm_state.angle)

        from feedbax.misc import SINCOS_GRAD_SIGNS
        ang_vel_sum = jnp.cumsum(arm_state.d_angle)
        xy_vel = jnp.cumsum(
            SINCOS_GRAD_SIGNS[1] * length_components[:, ::-1] * ang_vel_sum,
            axis=1,
        ).T

        return CartesianState(
            pos=xy_pos[-1],
            vel=xy_vel[-1],
            force=jnp.zeros(2),
        )
