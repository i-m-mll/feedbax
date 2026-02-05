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
from functools import cached_property
import logging
from typing import Optional, Type

import diffrax as dfx
import equinox as eqx
from equinox import Module, field
from equinox.nn import State
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray, PyTree, Scalar
import optimistix as optx

from feedbax.mechanics.dae import DAEComponent, DAEParams


logger = logging.getLogger(__name__)


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
        strain = norm_length - 1.0
        # Exponential passive force
        passive = jnp.where(
            strain > 0,
            (jnp.exp(self.stiffness * strain / self.strain_at_one_norm_force) - 1.0)
            / (jnp.exp(self.stiffness) - 1.0),
            0.0,
        )
        return passive


class ForceVelocityCurve(Module):
    """Force-velocity relationship for muscle fibers.

    Models the Hill hyperbola relating contraction velocity to force
    production. Shortening reduces force (concentric), lengthening
    increases force (eccentric).

    Attributes:
        max_shortening_velocity: Max velocity in optimal lengths/second.
        concentric_curvature: Shape factor for shortening side.
        eccentric_curvature: Shape factor for lengthening side.
        eccentric_force_max: Max normalized force during lengthening.
    """

    max_shortening_velocity: float = 10.0  # optimal lengths/second
    concentric_curvature: float = 0.25
    eccentric_curvature: float = 0.25
    eccentric_force_max: float = 1.4

    def __call__(self, norm_velocity: Array) -> Array:
        """Compute force-velocity multiplier.

        Args:
            norm_velocity: Fiber velocity / (vmax * optimal_length).
                Negative = shortening, positive = lengthening.

        Returns:
            Force multiplier.
        """
        # Concentric (shortening): classic Hill hyperbola
        # (a + F)(v + b) = b(F0 + a) where we solve for F
        a = self.concentric_curvature
        concentric = (1.0 + norm_velocity / a) / (1.0 - norm_velocity / (a * self.max_shortening_velocity))

        # Eccentric (lengthening): different hyperbola branch
        b = self.eccentric_curvature
        fmax = self.eccentric_force_max
        eccentric = (fmax - (fmax - 1.0) * (1.0 - norm_velocity / b) /
                    (1.0 + norm_velocity / (b * 0.1 * self.max_shortening_velocity)))

        # Smooth transition using tanh
        blend = 0.5 * (1.0 + jnp.tanh(norm_velocity * 20.0))
        return jnp.where(norm_velocity <= 0, concentric, blend * eccentric + (1 - blend) * concentric)


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
        strain = norm_length - 1.0
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
    vmax: float = 10.0


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
        """Compute fiber velocity from musculotendon velocity (rigid tendon).

        Args:
            fiber_length: Current fiber length [m].
            musculotendon_velocity: Musculotendon shortening velocity [m/s].

        Returns:
            Fiber velocity [m/s].
        """
        cos_alpha = jnp.cos(self.compute_pennation_angle(fiber_length))
        # d/dt(L_mt) = d/dt(L_f * cos(alpha)) ≈ v_f * cos(alpha) for small angle changes
        return musculotendon_velocity / cos_alpha

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

    Uses Kvaerno5 solver by default for stiff tendon dynamics.

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

    # Cached geometry
    _sin_pennation: float = field(static=True)
    _cos_pennation: float = field(static=True)

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

        self._sin_pennation = float(jnp.sin(muscle_params.pennation_angle))
        self._cos_pennation = float(jnp.cos(muscle_params.pennation_angle))

        super().__init__(
            dt=dt,
            solver_type=solver_type,
            root_finder=root_finder,
            key=key,
        )

    def compute_pennation_angle(self, fiber_length: Array) -> Array:
        """Compute current pennation angle."""
        h = self.muscle_params.optimal_fiber_length * self._sin_pennation
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

        # Activation dynamics
        tau = jnp.where(
            excitation > state.activation,
            self.muscle_params.tau_activation,
            self.muscle_params.tau_deactivation,
        )
        d_activation = (excitation - state.activation) / tau

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

        # Clamp to prevent numerical issues
        active_required = jnp.clip(active_required, 0.0, None)

        # Approximate fiber velocity from force-velocity inversion
        # This is a simplified inversion; for full accuracy use root finding
        afl = state.activation * fl
        afl = jnp.maximum(afl, 1e-6)  # Prevent division by zero
        fv_required = active_required / afl
        fv_required = jnp.clip(fv_required, 0.0, self.force_velocity.eccentric_force_max)

        # Simplified inversion of force-velocity curve
        a = self.force_velocity.concentric_curvature
        vmax = self.force_velocity.max_shortening_velocity

        # Solve: fv = (1 + v/a) / (1 - v/(a*vmax)) for v
        # fv * (1 - v/(a*vmax)) = 1 + v/a
        # fv - fv*v/(a*vmax) = 1 + v/a
        # fv - 1 = v/a + fv*v/(a*vmax)
        # fv - 1 = v * (1/a + fv/(a*vmax))
        norm_velocity = (fv_required - 1.0) / (1.0 / a + fv_required / (a * vmax))
        norm_velocity = jnp.clip(norm_velocity, -vmax, vmax * 0.1)

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
        """Extract force output from muscle state."""
        # For outputs, we need to compute force given current state
        # This requires MT length which isn't in the state...
        # Return a placeholder; actual force is computed in __call__
        return {"force": jnp.zeros(())}

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

        # Pack inputs for DAE
        input_tuple = (excitation, mt_length, mt_velocity)
        modified_inputs = {"input": input_tuple}

        outputs, state = super().__call__(modified_inputs, state, key=key)

        # Compute actual force from updated state
        dae_state = state.get(self.state_index)
        muscle_state = dae_state.system

        tendon_length = self.compute_tendon_length(muscle_state.fiber_length, mt_length)
        force = self.compute_tendon_force(tendon_length)

        outputs["force"] = force
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
