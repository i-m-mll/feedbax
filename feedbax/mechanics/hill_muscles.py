"""Hill-type muscle models with rigid and compliant tendons.

This module implements physiologically-based muscle models following the
Hill muscle formulation. It includes:

- Force-length and force-velocity relationships
- Activation dynamics
- Rigid tendon models (algebraic muscle-tendon coupling)
- Compliant tendon models (DAE with tendon dynamics)

Key references:
- Hill (1938): The heat of shortening and dynamic constants of muscle
- Zajac (1989): Muscle and tendon: properties, models, scaling
- Millard et al. (2013): Flexing computational muscle

:copyright: Copyright 2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Optional, Type

import diffrax as dfx
from equinox import Module, field
from equinox.nn import State
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray, PyTree, Scalar
import optimistix as optx

from feedbax._ode import ode_step
from feedbax.mechanics.dae import DAEComponent, DAEParams
from feedbax.mechanics.muscle_law import MuscleStage
from feedbax.mechanics.units import (
    DEFAULT_MUSCLE_VMAX,
    ECCENTRIC_VELOCITY_LIMIT_FRACTION,
)




# ============================================================================
# Force-Length-Velocity Curves
# ============================================================================


class ForceLengthCurve(Module):
    """Active force-length relationship for muscle fibers.

    Models the normalized force a muscle can produce as a function of
    its normalized length. At optimal length (1.0), force is maximal.

    Uses a Gaussian-like curve following Thelen (2003).

    Attributes:
        width: Width parameter controlling curve shape.
        min_norm_length: Minimum normalized length for force production.
        max_norm_length: Maximum normalized length for force production.
    """

    width: float = 0.56
    min_norm_length: float = 0.44
    max_norm_length: float = 1.6

    def __call__(self, norm_length: Array) -> Array:
        """Compute active force-length multiplier.

        Args:
            norm_length: Fiber length / optimal fiber length.

        Returns:
            Force multiplier in [0, 1].
        """
        # Gaussian-like curve centered at 1.0
        return jnp.exp(-((norm_length - 1.0) / self.width) ** 2)


class PassiveForceLengthCurve(Module):
    """Passive force-length relationship for muscle fibers.

    Models the passive elastic force from stretched muscle fibers
    (e.g., titin, connective tissue).

    Attributes:
        strain_at_one_norm_force: Strain at which passive force = max isometric.
        stiffness: Exponential stiffness parameter.
    """

    strain_at_one_norm_force: float = 0.7
    stiffness: float = 4.0

    def __call__(self, norm_length: Array) -> Array:
        """Compute passive force-length multiplier.

        Args:
            norm_length: Fiber length / optimal fiber length.

        Returns:
            Passive force multiplier (>= 0).
        """
        # Clamp strain to be non-negative (no passive force when shorter than optimal)
        strain = jnp.clip(norm_length - 1.0, 0.0, None)
        # Exponential passive force
        passive = jnp.where(
            strain > 0,
            (jnp.exp(self.stiffness * strain / self.strain_at_one_norm_force) - 1.0)
            / (jnp.exp(self.stiffness) - 1.0),
            0.0,
        )
        return passive


class ForceVelocityCurve(Module):
    """Normalized active force for a given muscle fiber velocity.

    The shortening branch is Hill's hyperbola (1938), with zero force at
    maximum shortening speed. A separate lengthening hyperbola rises from
    isometric force to `eccentric_force_max` at the shared lengthening limit.
    Each branch saturates beyond its speed limit. The branches are continuous
    at rest, where autodiff uses the shortening slope.

    Attributes:
        concentric_curvature: Positive Hill shape factor for shortening.
        eccentric_curvature: Positive shape factor for lengthening.
        eccentric_force_max: Lengthening plateau, greater than 1.
    """

    concentric_curvature: float = 0.25
    eccentric_curvature: float = 0.25
    eccentric_force_max: float = 1.4

    def __call__(self, norm_velocity: Array) -> Array:
        """Map fiber velocity / (vmax * optimal_fiber_length) to a force multiplier."""
        limit = ECCENTRIC_VELOCITY_LIMIT_FRACTION
        velocity = jnp.clip(norm_velocity, -1.0, limit)
        shortening = velocity <= 0.0

        # Restrict both rational functions to their own domains, even under vmap.
        v_short = jnp.where(shortening, velocity, 0.0)
        v_long = jnp.where(shortening, 0.0, velocity)
        concentric = (1.0 + v_short) / (1.0 - v_short / self.concentric_curvature)

        b = self.eccentric_curvature
        fmax = jnp.asarray(self.eccentric_force_max, dtype=velocity.dtype)
        progress = (v_long / limit) * ((b + limit) / (b + v_long))
        eccentric = 1.0 + (fmax - 1.0) * progress
        return jnp.where(shortening, concentric, eccentric)

    def inverse(self, force_multiplier: Array) -> Array:
        """Return normalized velocity, choosing the speed limit at either plateau.

        Forces outside the attainable range also map to the corresponding speed
        limit. Isometric force maps to zero with the shortening-side slope.
        """
        limit = ECCENTRIC_VELOCITY_LIMIT_FRACTION
        dtype = jnp.result_type(force_multiplier, 0.0)
        fmax = jnp.asarray(self.eccentric_force_max, dtype=dtype)
        force = jnp.clip(force_multiplier, 0.0, fmax)
        shortening = force <= 1.0

        f_short = jnp.where(shortening, force, 1.0)
        concentric = (f_short - 1.0) / (1.0 + f_short / self.concentric_curvature)

        b = self.eccentric_curvature
        f_long = jnp.where(shortening, 1.0, force)
        progress = (f_long - 1.0) / (fmax - 1.0)
        # This denominator stays at least b, even at the force plateau.
        eccentric = limit * (b * progress / (b + limit * (1.0 - progress)))
        return jnp.where(shortening, concentric, eccentric)

    def _inverse_active_force(self, required: Array, capacity: Array) -> Array:
        """Invert required active force at the given isometric capacity.

        Both forces use the same units. With zero capacity, choose rest if the
        forces already balance and the corresponding speed limit otherwise.
        """
        has_capacity = capacity > 0.0
        bounded_force = jnp.clip(required, 0.0, capacity * self.eccentric_force_max)
        multiplier = bounded_force / jnp.where(has_capacity, capacity, 1.0)
        no_capacity = jnp.where(
            required > 0.0, self.eccentric_force_max, jnp.where(required < 0.0, 0.0, 1.0),
        )
        return self.inverse(jnp.where(has_capacity, multiplier, no_capacity))


def bounded_rigid_fiber_force(
    activation: Array,
    fiber_length: Array,
    fiber_velocity: Array,
    *,
    optimal_length: Array | float,
    maximum_speed: Array | float,
    force_scale: Array | float,
    force_length: ForceLengthCurve,
    force_velocity: ForceVelocityCurve,
    passive_force_length: PassiveForceLengthCurve,
) -> Array:
    """Evaluate the bounded zero-pennation Hill fiber force in newtons."""
    length = jnp.clip(fiber_length, 0.3 * optimal_length, 2.0 * optimal_length)
    normalized_length = length / optimal_length
    velocity = jnp.clip(
        fiber_velocity, -maximum_speed, maximum_speed * ECCENTRIC_VELOCITY_LIMIT_FRACTION,
    )
    active = activation * force_length(normalized_length) * force_velocity(velocity / maximum_speed)
    return jnp.maximum(force_scale * (active + passive_force_length(normalized_length)), 0.0)


class TendonForceLengthCurve(Module):
    """Tendon force-length relationship.

    Models tendon as a nonlinear spring with exponential stiffness
    above slack length.

    Attributes:
        strain_at_one_norm_force: Tendon strain at max isometric force.
        stiffness: Exponential stiffness parameter.
    """

    strain_at_one_norm_force: float = 0.033  # 3.3% strain typical
    stiffness: float = 35.0

    def __call__(self, norm_length: Array) -> Array:
        """Compute tendon force from normalized length.

        Args:
            norm_length: Tendon length / tendon slack length.

        Returns:
            Normalized tendon force.
        """
        # Clamp strain to be non-negative (tendon cannot be compressed)
        strain = jnp.clip(norm_length - 1.0, 0.0, None)
        return jnp.where(
            strain > 0,
            (jnp.exp(self.stiffness * strain / self.strain_at_one_norm_force) - 1.0)
            / (jnp.exp(self.stiffness) - 1.0),
            0.0,
        )

    def inverse(self, norm_force: Array) -> Array:
        """Compute normalized tendon length from force.

        Args:
            norm_force: Normalized tendon force.

        Returns:
            Tendon length / slack length.
        """
        # Invert: strain = (strain_at_1/stiffness) * ln(1 + force * (exp(stiffness) - 1))
        strain = jnp.where(
            norm_force > 0,
            (self.strain_at_one_norm_force / self.stiffness)
            * jnp.log(1.0 + norm_force * (jnp.exp(self.stiffness) - 1.0)),
            0.0,
        )
        return 1.0 + strain


# ============================================================================
# Muscle Parameters
# ============================================================================


class HillMuscleParams(DAEParams):
    """Physical parameters for a Hill-type muscle.

    Attributes:
        max_isometric_force: Peak force at optimal length [N].
        optimal_fiber_length: Length at which peak force is produced [m].
        tendon_slack_length: Unstretched tendon length [m].
        pennation_angle: Fiber angle relative to tendon [rad].
        tau_activation: Activation time constant [s].
        tau_deactivation: Deactivation time constant [s].
        vmax: Maximum shortening velocity [optimal lengths/s].
    """

    max_isometric_force: float
    optimal_fiber_length: float
    tendon_slack_length: float
    pennation_angle: float = 0.0
    tau_activation: float = 0.01
    tau_deactivation: float = 0.04
    vmax: float = DEFAULT_MUSCLE_VMAX


# ============================================================================
# Muscle State
# ============================================================================


class HillMuscleState(Module):
    """State of a Hill-type muscle.

    Attributes:
        activation: Muscle activation level [0, 1].
        fiber_length: Current muscle fiber length [m].
        fiber_velocity: Current fiber velocity [m/s].
        tendon_length: Current tendon length [m] (for compliant tendon).
        force: Current muscle-tendon force [N].
    """

    activation: Float[Array, ""]
    fiber_length: Float[Array, ""]
    fiber_velocity: Float[Array, ""]
    tendon_length: Float[Array, ""]
    force: Float[Array, ""]


# ============================================================================
# Abstract Muscle Base
# ============================================================================


class AbstractHillMuscle(Module):
    """Base class for Hill-type muscle models.

    Defines the interface for muscle force computation given
    activation, fiber length, and fiber velocity.
    """

    params: HillMuscleParams
    force_length: ForceLengthCurve
    passive_force_length: PassiveForceLengthCurve
    force_velocity: ForceVelocityCurve
    tendon_force_length: TendonForceLengthCurve

    @abstractmethod
    def compute_force(
        self,
        activation: Array,
        fiber_length: Array,
        fiber_velocity: Array,
        musculotendon_length: Array,
    ) -> Array:
        """Compute muscle-tendon force.

        Args:
            activation: Muscle activation [0, 1].
            fiber_length: Current fiber length [m].
            fiber_velocity: Current fiber velocity [m/s].
            musculotendon_length: Total muscle-tendon length [m].

        Returns:
            Muscle-tendon force [N].
        """
        ...

    def compute_pennation_angle(self, fiber_length: Array) -> Array:
        """Compute current pennation angle given fiber length.

        Uses constant-thickness assumption: sin(alpha) * L_f = constant.

        Args:
            fiber_length: Current fiber length.

        Returns:
            Current pennation angle [rad].
        """
        h = self.params.optimal_fiber_length * jnp.sin(self.params.pennation_angle)
        sin_alpha = jnp.clip(h / fiber_length, 0.0, 1.0)
        return jnp.arcsin(sin_alpha)

    def compute_fiber_force(
        self,
        activation: Array,
        fiber_length: Array,
        fiber_velocity: Array,
    ) -> Array:
        """Compute fiber force along fiber direction.

        Args:
            activation: Muscle activation [0, 1].
            fiber_length: Current fiber length [m].
            fiber_velocity: Current fiber velocity [m/s].

        Returns:
            Fiber force [N].
        """
        norm_length = fiber_length / self.params.optimal_fiber_length
        norm_velocity = fiber_velocity / (self.params.vmax * self.params.optimal_fiber_length)

        # Active force
        fl = self.force_length(norm_length)
        fv = self.force_velocity(norm_velocity)
        active_force = activation * fl * fv

        # Passive force
        passive_force = self.passive_force_length(norm_length)

        return self.params.max_isometric_force * (active_force + passive_force)


# ============================================================================
# Rigid Tendon Muscle
# ============================================================================


class RigidTendonHillMuscle(AbstractHillMuscle):
    """Hill muscle with rigid (inextensible) tendon.

    The tendon is assumed to be infinitely stiff, so:
        musculotendon_length = tendon_slack_length + fiber_length * cos(pennation)

    This is an algebraic constraint that determines fiber length directly
    from musculotendon length.

    Suitable for muscles with short tendons or when tendon compliance
    is not critical to the dynamics.
    """

    params: HillMuscleParams
    force_length: ForceLengthCurve = field(default_factory=ForceLengthCurve)
    passive_force_length: PassiveForceLengthCurve = field(default_factory=PassiveForceLengthCurve)
    force_velocity: ForceVelocityCurve = field(default_factory=ForceVelocityCurve)
    tendon_force_length: TendonForceLengthCurve = field(default_factory=TendonForceLengthCurve)

    def compute_fiber_length_from_mt_length(
        self,
        musculotendon_length: Array,
    ) -> Array:
        """Compute fiber length from musculotendon length (rigid tendon).

        Args:
            musculotendon_length: Total muscle-tendon length [m].

        Returns:
            Fiber length [m].
        """
        # For rigid tendon: L_mt = L_t_slack + L_f * cos(alpha)
        # With constant thickness: cos(alpha) = sqrt(1 - (h/L_f)^2)
        # This gives a quadratic in L_f^2

        l_mt = musculotendon_length
        l_t_slack = self.params.tendon_slack_length
        h = self.params.optimal_fiber_length * jnp.sin(self.params.pennation_angle)

        # Simplified for small pennation or zero pennation
        if self.params.pennation_angle < 0.01:
            return l_mt - l_t_slack

        # General case: solve (L_mt - L_t_slack)^2 = L_f^2 - h^2
        fiber_projection_sq = (l_mt - l_t_slack) ** 2
        fiber_length = jnp.sqrt(fiber_projection_sq + h ** 2)
        return fiber_length

    def compute_fiber_velocity_from_mt_velocity(
        self,
        fiber_length: Array,
        musculotendon_velocity: Array,
    ) -> Array:
        """Differentiate the rigid tendon fiber length at a taut pose.

        Args:
            fiber_length: Current fiber length [m].
            musculotendon_velocity: Musculotendon lengthening velocity [m/s].

        Returns:
            Fiber velocity [m/s].
        """
        if self.params.pennation_angle < 0.01:
            return musculotendon_velocity
        return musculotendon_velocity * jnp.cos(self.compute_pennation_angle(fiber_length))

    def compute_force(
        self,
        activation: Array,
        fiber_length: Array,
        fiber_velocity: Array,
        musculotendon_length: Array,
    ) -> Array:
        """Compute muscle-tendon force.

        For rigid tendon, fiber length is determined algebraically from
        musculotendon length.

        Args:
            activation: Muscle activation [0, 1].
            fiber_length: Current fiber length [m] (may be ignored).
            fiber_velocity: Current fiber velocity [m/s].
            musculotendon_length: Total muscle-tendon length [m].

        Returns:
            Muscle-tendon force [N].
        """
        # Rigid tendon: compute fiber length from geometry
        l_f = self.compute_fiber_length_from_mt_length(musculotendon_length)

        # Fiber force
        fiber_force = self.compute_fiber_force(activation, l_f, fiber_velocity)

        # Project along tendon
        cos_alpha = jnp.cos(self.compute_pennation_angle(l_f))
        return fiber_force * cos_alpha

    def init_state(
        self,
        musculotendon_length: Array,
        *,
        activation: float = 0.0,
    ) -> HillMuscleState:
        """Initialize muscle state given musculotendon length.

        Args:
            musculotendon_length: Initial MT length [m].
            activation: Initial activation level.

        Returns:
            Initial muscle state.
        """
        l_f = self.compute_fiber_length_from_mt_length(musculotendon_length)
        return HillMuscleState(
            activation=jnp.asarray(activation),
            fiber_length=l_f,
            fiber_velocity=jnp.zeros_like(l_f),
            tendon_length=jnp.asarray(self.params.tendon_slack_length),
            force=jnp.zeros_like(l_f),
        )


# ============================================================================
# Compliant Tendon Muscle
# ============================================================================


class CompliantTendonState(Module):
    """State for compliant tendon muscle (includes tendon dynamics).

    Attributes:
        activation: Muscle activation [0, 1].
        fiber_length: Muscle fiber length [m].
    """

    activation: Float[Array, ""]
    fiber_length: Float[Array, ""]


class CompliantTendonHillMuscle(DAEComponent[CompliantTendonState]):
    """Hill muscle with elastic/compliant tendon.

    The tendon is modeled as a nonlinear spring, introducing an additional
    state variable (fiber length) and creating a DAE:

        fiber_velocity = f(activation, fiber_length, musculotendon_length)

    where the fiber velocity is determined implicitly by force equilibrium
    between fiber and tendon.

    Uses explicit Euler solver by default for simplicity. For very stiff
    dynamics, consider using an implicit solver (e.g., ImplicitEuler, Kvaerno3).

    Attributes:
        muscle_params: Physical muscle parameters.
        force_length: Active force-length curve.
        passive_force_length: Passive force-length curve.
        force_velocity: Force-velocity curve.
        tendon_force_length: Tendon force-length curve.
    """

    input_ports = ("excitation", "musculotendon_length", "musculotendon_velocity")
    output_ports = ("force", "state")

    muscle_params: HillMuscleParams
    force_length: ForceLengthCurve
    passive_force_length: PassiveForceLengthCurve
    force_velocity: ForceVelocityCurve
    tendon_force_length: TendonForceLengthCurve

    def __init__(
        self,
        muscle_params: HillMuscleParams,
        dt: float = 0.001,
        solver_type: Type[dfx.AbstractSolver] = dfx.Euler,  # Explicit for now
        root_finder: Optional[optx.AbstractRootFinder] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ):
        """Initialize compliant tendon muscle.

        Args:
            muscle_params: Physical muscle parameters.
            dt: Integration timestep.
            solver_type: Solver type. Use explicit (Euler) for simplicity,
                or implicit (ImplicitEuler) for very stiff dynamics.
            root_finder: Root finder (only for implicit solvers).
            key: PRNG key.
        """
        self.muscle_params = muscle_params
        self.force_length = ForceLengthCurve()
        self.passive_force_length = PassiveForceLengthCurve()
        self.force_velocity = ForceVelocityCurve()
        self.tendon_force_length = TendonForceLengthCurve()

        super().__init__(
            dt=dt,
            solver_type=solver_type,
            root_finder=root_finder,
            key=key,
        )

    def compute_pennation_angle(self, fiber_length: Array) -> Array:
        """Compute current pennation angle."""
        params = self.muscle_params
        h = params.optimal_fiber_length * jnp.sin(params.pennation_angle)
        sin_alpha = jnp.clip(h / fiber_length, 0.0, 1.0)
        return jnp.arcsin(sin_alpha)

    def compute_tendon_length(
        self,
        fiber_length: Array,
        musculotendon_length: Array,
    ) -> Array:
        """Compute tendon length from fiber length and MT length.

        Args:
            fiber_length: Current fiber length [m].
            musculotendon_length: Total MT length [m].

        Returns:
            Tendon length [m].
        """
        cos_alpha = jnp.cos(self.compute_pennation_angle(fiber_length))
        return musculotendon_length - fiber_length * cos_alpha

    def compute_tendon_force(self, tendon_length: Array) -> Array:
        """Compute tendon force from tendon length.

        Args:
            tendon_length: Current tendon length [m].

        Returns:
            Tendon force [N].
        """
        norm_length = tendon_length / self.muscle_params.tendon_slack_length
        norm_force = self.tendon_force_length(norm_length)
        return self.muscle_params.max_isometric_force * norm_force

    def compute_fiber_force(
        self,
        activation: Array,
        fiber_length: Array,
        fiber_velocity: Array,
    ) -> Array:
        """Compute fiber force along fiber direction.

        Args:
            activation: Muscle activation [0, 1].
            fiber_length: Current fiber length [m].
            fiber_velocity: Current fiber velocity [m/s].

        Returns:
            Fiber force [N].
        """
        norm_length = fiber_length / self.muscle_params.optimal_fiber_length
        norm_velocity = fiber_velocity / (
            self.muscle_params.vmax * self.muscle_params.optimal_fiber_length
        )

        fl = self.force_length(norm_length)
        fv = self.force_velocity(norm_velocity)
        passive = self.passive_force_length(norm_length)

        return self.muscle_params.max_isometric_force * (activation * fl * fv + passive)

    @jax.named_scope("fbx.CompliantTendonHillMuscle.vector_field")
    def vector_field(
        self,
        t: Scalar,
        state: CompliantTendonState,
        input: tuple[Array, Array, Array],
    ) -> CompliantTendonState:
        """Compute time derivatives for muscle state.

        The fiber velocity is determined by force equilibrium:
            F_fiber * cos(alpha) = F_tendon

        Args:
            t: Current time.
            state: Current muscle state (activation, fiber_length).
            input: Tuple of (excitation, mt_length, mt_velocity).

        Returns:
            State derivatives.
        """
        excitation, mt_length, mt_velocity = input

        d_activation = ActivationDynamics(
            self.muscle_params.tau_activation, self.muscle_params.tau_deactivation,
        )(excitation, state.activation)

        # Tendon length and force
        tendon_length = self.compute_tendon_length(state.fiber_length, mt_length)
        tendon_force = self.compute_tendon_force(tendon_length)

        # Force equilibrium to find fiber velocity
        cos_alpha = jnp.cos(self.compute_pennation_angle(state.fiber_length))
        required_fiber_force = tendon_force / cos_alpha

        # Invert force-velocity relationship
        # F_fiber = F0 * (a * fl * fv + passive)
        norm_length = state.fiber_length / self.muscle_params.optimal_fiber_length
        fl = self.force_length(norm_length)
        passive = self.passive_force_length(norm_length)

        norm_required = required_fiber_force / self.muscle_params.max_isometric_force
        active_required = norm_required - passive

        norm_velocity = self.force_velocity._inverse_active_force(
            active_required, state.activation * fl,
        )

        d_fiber_length = norm_velocity * self.muscle_params.vmax * self.muscle_params.optimal_fiber_length

        return CompliantTendonState(
            activation=d_activation,
            fiber_length=d_fiber_length,
        )

    def init_system_state(self, *, key: PRNGKeyArray) -> CompliantTendonState:
        """Initialize muscle state at rest."""
        return CompliantTendonState(
            activation=jnp.array(0.0),
            fiber_length=jnp.array(self.muscle_params.optimal_fiber_length),
        )

    def extract_outputs(self, state: CompliantTendonState) -> dict[str, Array]:
        """Compliant tendon force requires current musculotendon length input."""
        raise NotImplementedError(
            "CompliantTendonHillMuscle cannot extract force from state alone; "
            "call the component with musculotendon_length so tendon force can be computed."
        )

    def _get_zero_input(self) -> tuple[Array, Array, Array]:
        """Zero input tuple."""
        return (jnp.array(0.0), jnp.array(0.0), jnp.array(0.0))

    @property
    def input_size(self) -> int:
        """Input size (excitation scalar)."""
        return 1

    def __call__(
        self,
        inputs: dict[str, Array],
        state,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, PyTree], State]:
        """Execute one integration step.

        Args:
            inputs: Dict with 'excitation', 'musculotendon_length', 'musculotendon_velocity'.
            state: Current State container.
            key: PRNG key.

        Returns:
            Outputs dict and updated state.
        """
        excitation = inputs.get("excitation", jnp.zeros(()))
        mt_length = inputs.get("musculotendon_length", jnp.zeros(()))
        mt_velocity = inputs.get("musculotendon_velocity", jnp.zeros(()))

        input_tuple = (excitation, mt_length, mt_velocity)
        dae_state = state.get(self.state_index)
        muscle_state, new_solver_state = ode_step(
            self.solver, self._term, self.dt, dae_state.system, input_tuple, dae_state.solver,
        )

        clamped_activation = jnp.clip(muscle_state.activation, 0.0, 1.0)
        muscle_state = CompliantTendonState(
            activation=clamped_activation,
            fiber_length=muscle_state.fiber_length,
        )
        from feedbax.mechanics.dae import DAEState
        dae_state = DAEState(system=muscle_state, solver=new_solver_state)
        state = state.set(self.state_index, dae_state)

        tendon_length = self.compute_tendon_length(muscle_state.fiber_length, mt_length)
        force = self.compute_tendon_force(tendon_length)

        outputs = {
            "force": force,
            "state": muscle_state,
        }
        return outputs, state

    def compute_constraint_residual(
        self,
        state: CompliantTendonState,
        mt_length: Array,
    ) -> Array:
        """Compute force equilibrium constraint residual.

        For a well-solved DAE, this should be close to zero.

        Args:
            state: Current muscle state.
            mt_length: Current musculotendon length.

        Returns:
            Residual: F_fiber * cos(alpha) - F_tendon.
        """
        tendon_length = self.compute_tendon_length(state.fiber_length, mt_length)
        tendon_force = self.compute_tendon_force(tendon_length)

        # Need fiber velocity to compute fiber force
        # For constraint check, assume equilibrium (zero residual) means correct velocity
        cos_alpha = jnp.cos(self.compute_pennation_angle(state.fiber_length))
        required_fiber_force = tendon_force / cos_alpha

        # Compute actual fiber force at zero velocity (isometric approximation)
        isometric_fiber_force = self.compute_fiber_force(
            state.activation, state.fiber_length, jnp.zeros(())
        )

        return required_fiber_force - isometric_fiber_force


# ============================================================================
# Activation Filter (for use with rigid tendon muscles)
# ============================================================================


class ActivationDynamics(Module):
    """First-order activation dynamics filter.

    Converts neural excitation to muscle activation with
    separate time constants for activation and deactivation.

    Attributes:
        tau_activation: Activation time constant [s].
        tau_deactivation: Deactivation time constant [s].
    """

    tau_activation: float = 0.01
    tau_deactivation: float = 0.04

    def __call__(
        self,
        excitation: Array,
        activation: Array,
    ) -> Array:
        """Compute activation derivative.

        Args:
            excitation: Neural excitation [0, 1].
            activation: Current activation [0, 1].

        Returns:
            Time derivative of activation.
        """
        tau = jnp.where(
            excitation > activation,
            self.tau_activation,
            self.tau_deactivation,
        )
        return (excitation - activation) / tau



class BoundedRigidTendonHillLaw(Module):
    """Zero-pennation rigid tendon with one integrated activation per path."""

    params: HillMuscleParams
    initial_activation: float = 0.0
    force_length: ForceLengthCurve = field(default_factory=ForceLengthCurve)
    passive_force_length: PassiveForceLengthCurve = field(default_factory=PassiveForceLengthCurve)
    force_velocity: ForceVelocityCurve = field(default_factory=ForceVelocityCurve)

    @property
    def state_size(self) -> int:
        return 1

    def init_state(self, length: Array, velocity: Array) -> Array:
        return jnp.full((1,), self.initial_activation, dtype=length.dtype)

    def evaluate(
        self, state: Array, excitation: Array, length: Array, velocity: Array,
    ) -> MuscleStage:
        params = self.params
        activation = state[0]
        derivative = ActivationDynamics(params.tau_activation, params.tau_deactivation)(
            excitation, activation,
        )
        tension = bounded_rigid_fiber_force(
            activation, length - params.tendon_slack_length, velocity,
            optimal_length=params.optimal_fiber_length,
            maximum_speed=params.vmax * params.optimal_fiber_length,
            force_scale=params.max_isometric_force,
            force_length=self.force_length,
            force_velocity=self.force_velocity,
            passive_force_length=self.passive_force_length,
        )
        return MuscleStage(jnp.expand_dims(derivative, 0), tension)
