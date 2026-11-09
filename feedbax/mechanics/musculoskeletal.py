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

import logging
from typing import Optional, Type

import diffrax as dfx
from equinox import Module, field
from equinox.nn import State, StateIndex
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray, PyTree, Scalar
import optimistix as optx

from feedbax.runtime.graph import GraphNode
from feedbax.mechanics.dae import DAEComponent
from feedbax.mechanics.geometry import TwoLinkArmMuscleGeometry
from feedbax.mechanics.hill_muscles import (
    HillMuscleParams,
    ActivationDynamics,
    ForceLengthCurve,
    PassiveForceLengthCurve,
    ForceVelocityCurve,
    CompliantTendonHillLaw,
    bounded_rigid_fiber_force,
)
from feedbax.mechanics.skeleton._planar import link_kinematics, two_link_angular_acceleration
from feedbax.mechanics.skeleton.arm import TwoLinkArmState
from feedbax.mechanics.units import (
    DEFAULT_MUSCLE_VMAX,
    require_array_shape,
    require_positive_finite,
    require_positive_array,
)
from feedbax.runtime.state import CartesianState


logger = logging.getLogger(__name__)


def _arm_effector(lengths: Array, arm_state: TwoLinkArmState) -> CartesianState:
    pos, vel = link_kinematics(lengths, arm_state.angle, arm_state.d_angle)
    return CartesianState(pos=pos[-1], vel=vel[-1])


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
    activations: Float[Array, "n_muscles"]  # noqa: F821
    fiber_lengths: Float[Array, "n_muscles"]  # noqa: F821
    fiber_velocities: Float[Array, "n_muscles"]  # noqa: F821
    forces: Float[Array, "n_muscles"]  # noqa: F821


# ============================================================================
# Musculoskeletal Arm with Rigid Tendons
# ============================================================================


class RigidTendonMusculoskeletalArm(GraphNode):
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
    state_index: StateIndex

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
        solver_type: Type[dfx.AbstractSolver] = dfx.Euler,
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
            solver_type: Diffrax solver type. Only ``dfx.Euler`` is supported
                by this explicit rigid-tendon integrator.
            key: PRNG key.
        """
        if solver_type is not dfx.Euler:
            raise ValueError(
                "RigidTendonMusculoskeletalArm uses an explicit Euler update; "
                f"solver_type={solver_type.__name__} is not supported. Use dfx.Euler "
                "or switch to a DAE-based component for implicit integration."
            )

        # Store arm parameters. Units are SI: metres, kg, kg*m^2, seconds.
        self.arm_l = require_positive_array("arm_l", arm_l, (2,))
        self.arm_m = require_positive_array("arm_m", arm_m, (2,))
        self.arm_I = require_positive_array("arm_I", arm_I, (2,))
        self.arm_s = require_positive_array("arm_s", arm_s, (2,))
        self.arm_B = require_array_shape("arm_B", arm_B, (2, 2))

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
                    vmax=DEFAULT_MUSCLE_VMAX,
                )
                for _ in range(actual_n_muscles)
            )
        if len(muscle_params) != actual_n_muscles:
            raise ValueError(
                "muscle_params length must match geometry.n_muscles: "
                f"{len(muscle_params)} != {actual_n_muscles}"
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
        self.dt = require_positive_finite("RigidTendonMusculoskeletalArm.dt", dt)

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

        initial_state = MusculoskeletalState(
            arm=init_arm,
            activations=init_activations,
            fiber_lengths=init_fiber_lengths,
            fiber_velocities=jnp.zeros(actual_n_muscles),
            forces=jnp.zeros(actual_n_muscles),
        )
        self.state_index = StateIndex(initial_state)

    @property
    def n_muscles(self) -> int:
        """Number of muscles."""
        return self.geometry.n_muscles

    def _compute_muscle_forces(
        self,
        activations: Array,
        fiber_lengths: Array,
        fiber_velocities: Array,
    ) -> Array:
        """Compute forces for all muscles.

        Args:
            activations: Muscle activations [n_muscles].
            fiber_lengths: Fiber lengths [n_muscles].
            fiber_velocities: Fiber velocities [n_muscles].

        Returns:
            Muscle forces [n_muscles].
        """
        return jnp.array([
            bounded_rigid_fiber_force(
                activations[i], fiber_lengths[i], fiber_velocities[i],
                optimal_length=params.optimal_fiber_length,
                maximum_speed=params.vmax * params.optimal_fiber_length,
                force_scale=params.max_isometric_force,
                force_length=self.force_length,
                force_velocity=self.force_velocity,
                passive_force_length=self.passive_force_length,
            )
            for i, params in enumerate(self.muscle_params)
        ])

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
        )

        # Muscle forces to joint torques
        torques = moment_arms.T @ forces  # [n_joints]

        dd_angle = two_link_angular_acceleration(
            self.arm_l, self.arm_m, self.arm_I, self.arm_s, self.arm_B,
            angle, d_angle, torques,
        )

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

        # Integrate one explicit Euler step.
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
        mt_velocities = self.geometry.musculotendon_velocities(new_arm.angle, new_arm.d_angle)
        new_forces = self._compute_muscle_forces(
            new_activations,
            new_fiber_lengths,
            mt_velocities,
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

    def _effector(self, arm_state: TwoLinkArmState) -> CartesianState:
        """Compute effector Cartesian state."""
        return _arm_effector(self.arm_l, arm_state)

    def state_view(self, state: State) -> MusculoskeletalState:
        """Return the musculoskeletal state."""
        return state.get(self.state_index)


# ============================================================================
# Musculoskeletal Arm with Compliant Tendons (DAE)
# ============================================================================


class CompliantTendonMusculoskeletalArm(DAEComponent[MusculoskeletalState]):
    """Musculoskeletal arm with compliant tendon muscles.

    This is a full DAE model where tendon compliance introduces
    stiff dynamics. Uses explicit Euler solver by default for simplicity.
    For very stiff dynamics, consider using an implicit solver
    (e.g., ImplicitEuler, Kvaerno3, Kvaerno5).

    Joint torque comes from the tendon spring forces at the current joint
    angles and fiber lengths. If the required force lies outside the range
    the fibers can produce, the inverse force-velocity law limits the speed
    of fiber length evolution. The tendon spring still determines the
    instantaneous transmitted force.

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

    laws: tuple[CompliantTendonHillLaw, ...]
    geometry: TwoLinkArmMuscleGeometry

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
            solver_type: Solver type (default Euler). Use implicit solvers
                (e.g., ImplicitEuler, Kvaerno3) for very stiff dynamics.
            root_finder: Root finder with tight tolerances (only for implicit solvers).
            key: PRNG key.
        """
        # Store arm parameters. Units are SI: metres, kg, kg*m^2, seconds.
        self.arm_l = require_positive_array("arm_l", arm_l, (2,))
        self.arm_m = require_positive_array("arm_m", arm_m, (2,))
        self.arm_I = require_positive_array("arm_I", arm_I, (2,))
        self.arm_s = require_positive_array("arm_s", arm_s, (2,))
        self.arm_B = require_array_shape("arm_B", arm_B, (2, 2))

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
                    vmax=DEFAULT_MUSCLE_VMAX,
                )
                for _ in range(geometry.n_muscles)
            )
        if len(muscle_params) != geometry.n_muscles:
            raise ValueError(
                "muscle_params length must match geometry.n_muscles: "
                f"{len(muscle_params)} != {geometry.n_muscles}"
            )
        self.laws = tuple(CompliantTendonHillLaw(params) for params in muscle_params)

        if root_finder is None:
            root_finder = optx.Newton(rtol=1e-8, atol=1e-8)

        super().__init__(
            dt=require_positive_finite("CompliantTendonMusculoskeletalArm.dt", dt),
            solver_type=solver_type,
            root_finder=root_finder,
            key=key,
        )

    @property
    def n_muscles(self) -> int:
        return self.geometry.n_muscles

    @property
    def muscle_params(self) -> tuple[HillMuscleParams, ...]:
        return tuple(law.params for law in self.laws)

    def _tendon_loads(self, state: MusculoskeletalState) -> tuple[Array, Array]:
        angle = state.arm.angle
        mt_lengths = self.geometry.musculotendon_lengths(angle)
        forces = jnp.stack([
            law.tendon_force(law.tendon_length(fiber, length))
            for law, fiber, length in zip(self.laws, state.fiber_lengths, mt_lengths, strict=True)
        ])
        return forces, self.geometry.moment_arms(angle).T @ forces

    @jax.named_scope("fbx.CompliantTendonMusculoskeletalArm.vector_field")
    def vector_field(
        self,
        t: Scalar,
        state: MusculoskeletalState,
        input: Array,
    ) -> MusculoskeletalState:
        """Compute derivatives for compliant tendon model."""
        arm = state.arm
        angle, d_angle = arm.angle, arm.d_angle
        lengths = self.geometry.musculotendon_lengths(angle)
        moments = self.geometry.moment_arms(angle)
        stages = tuple(
            law.evaluate(jnp.stack((activation, fiber)), excitation, length, velocity)
            for law, activation, fiber, excitation, length, velocity in zip(
                self.laws, state.activations, state.fiber_lengths, input,
                lengths, -moments @ d_angle, strict=True,
            )
        )
        d_activations, fiber_velocities = jnp.stack([stage.derivative for stage in stages]).T
        tendon_forces = jnp.stack([stage.tension for stage in stages])
        torques = moments.T @ tendon_forces

        dd_angle = two_link_angular_acceleration(
            self.arm_l, self.arm_m, self.arm_I, self.arm_s, self.arm_B,
            angle, d_angle, torques,
        )

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
            forces=jnp.zeros_like(tendon_forces),
        )

    def init_system_state(self, *, key: PRNGKeyArray) -> MusculoskeletalState:
        """Initialize at rest position."""
        init_arm = TwoLinkArmState()

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

        # Apply activation bounds clamping after integration
        dae_state = state.get(self.state_index)
        ms_state = dae_state.system
        clamped_activations = jnp.clip(ms_state.activations, 0.0, 1.0)
        ms_state = MusculoskeletalState(
            arm=ms_state.arm,
            activations=clamped_activations,
            fiber_lengths=ms_state.fiber_lengths,
            fiber_velocities=ms_state.fiber_velocities,
            forces=ms_state.forces,
        )
        from feedbax.mechanics.dae import DAEState
        dae_state = DAEState(system=ms_state, solver=dae_state.solver)
        state = state.set(self.state_index, dae_state)

        # Compute additional outputs
        effector = self._effector(ms_state.arm)
        forces, torques = self._tendon_loads(ms_state)

        outputs.update({
            "effector": effector,
            "forces": forces,
            "torques": torques,
        })

        return outputs, state

    def _effector(self, arm_state: TwoLinkArmState) -> CartesianState:
        return _arm_effector(self.arm_l, arm_state)
