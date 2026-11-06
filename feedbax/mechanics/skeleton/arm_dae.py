"""Two-link arm dynamics using DAE/implicit integration.

This module provides an implicit solver-based two-link arm model,
useful for stiff dynamics or when coupled with muscle models.

:copyright: Copyright 2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from __future__ import annotations

from functools import cached_property
from typing import Optional, Type

import diffrax as dfx
from equinox import field
from equinox.nn import State
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray, PyTree, Scalar
import optimistix as optx

from feedbax.mechanics.dae import DAEComponent, DAEParams
from feedbax.mechanics.skeleton._planar import distal_state, effector_jacobian, link_kinematics
from feedbax.mechanics.skeleton.arm import TwoLinkArmState
from feedbax.runtime.state import CartesianState




class TwoLinkArmDAEParams(DAEParams):
    """Physical parameters for the two-link arm.

    Attributes:
        l: Segment lengths [m].
        m: Segment masses [kg].
        I: Moments of inertia [kg*m^2].
        s: Distance from joint to segment center of mass [m].
        B: Joint friction matrix [kg*m^2/s].
    """

    l: Float[Array, "2"] = field(converter=jnp.asarray)
    m: Float[Array, "2"] = field(converter=jnp.asarray)
    I: Float[Array, "2"] = field(converter=jnp.asarray)
    s: Float[Array, "2"] = field(converter=jnp.asarray)
    B: Float[Array, "2 2"] = field(converter=jnp.asarray)

    def __init__(
        self,
        l=(0.30, 0.33),
        m=(1.4, 1.0),
        I=(0.025, 0.045),
        s=(0.11, 0.16),
        B=((0.05, 0.025), (0.025, 0.05)),
    ):
        self.l = l
        self.m = m
        self.I = I
        self.s = s
        self.B = B


class TwoLinkArmDAE(DAEComponent[TwoLinkArmState]):
    """Two-link arm dynamics with implicit integration.

    Uses Kvaerno3 solver by default for moderate stiffness handling.
    The arm is modeled as two rigid segments with rotational joints,
    following standard Lagrangian mechanics.

    Dynamics include:
    - Inertia matrix (configuration-dependent)
    - Coriolis and centripetal terms
    - Joint friction
    - External torques from inputs or effector forces

    Attributes:
        params: Physical parameters for the arm.
        dt: Integration timestep.
    """

    input_ports = ("torque",)
    output_ports = ("effector", "state", "joints")

    params: TwoLinkArmDAEParams

    def __init__(
        self,
        l=(0.30, 0.33),
        m=(1.4, 1.0),
        I=(0.025, 0.045),
        s=(0.11, 0.16),
        B=((0.05, 0.025), (0.025, 0.05)),
        dt: float = 0.01,
        solver_type: Type[dfx.AbstractSolver] = dfx.Euler,  # Use explicit for non-stiff arm
        root_finder: Optional[optx.AbstractRootFinder] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ):
        """Initialize two-link arm DAE component.

        Args:
            l: Segment lengths [m].
            m: Segment masses [kg].
            I: Moments of inertia [kg*m^2].
            s: Distance from joint to center of mass [m].
            B: Joint friction matrix [kg*m^2/s].
            dt: Integration timestep.
            solver_type: Implicit solver type (default: Kvaerno3).
            root_finder: Root finder for implicit steps.
            key: PRNG key for initialization.
        """
        self.params = TwoLinkArmDAEParams(l=l, m=m, I=I, s=s, B=B)
        super().__init__(
            dt=dt,
            solver_type=solver_type,
            root_finder=root_finder,
            key=key,
        )

    @cached_property
    def _a(self) -> tuple[Array, Array, Array]:
        """Precomputed inertia-related constants."""
        return (
            self.params.I[0] + self.params.I[1] + self.params.m[1] * self.params.l[0] ** 2,
            self.params.m[1] * self.params.l[0] * self.params.s[1],
            self.params.I[1],
        )

    @jax.named_scope("fbx.TwoLinkArmDAE.vector_field")
    def vector_field(
        self,
        t: Scalar,
        state: TwoLinkArmState,
        input: Float[Array, "2"],
    ) -> TwoLinkArmState:
        """Compute time derivatives of arm configuration state.

        Implements the equations of motion for a two-link planar arm:
        M(q) * ddq + C(q, dq) + B * dq = tau

        where M is the inertia matrix, C are Coriolis/centripetal terms,
        B is the friction matrix, and tau are the applied torques.

        Args:
            t: Current time (unused).
            state: Current arm configuration state.
            input: Applied torques on the joints.

        Returns:
            Time derivatives: d_angle = d_angle, d_d_angle = acceleration.
        """
        angle, d_angle = state.angle, state.d_angle

        # Centripetal and Coriolis torques
        c_vec = (
            jnp.array((-d_angle[1] * (2 * d_angle[0] + d_angle[1]), d_angle[0] ** 2))
            * self._a[1]
            * jnp.sin(angle[1])
        )

        # Inertia matrix
        cs1 = jnp.cos(angle[1])
        tmp = self._a[2] + self._a[1] * cs1
        inertia_mat = jnp.array([
            [self._a[0] + 2 * self._a[1] * cs1, tmp],
            [tmp, self._a[2] * jnp.ones_like(cs1)],
        ])

        load_torque = self.effector_force_to_torques(angle, state.effector_force)
        net_torque = (
            state.torque + input + load_torque
            - c_vec.T
            - jnp.matmul(d_angle, self.params.B.T)
        )

        # Angular acceleration via inverse dynamics
        dd_angle = jnp.linalg.solve(inertia_mat, net_torque)

        return TwoLinkArmState(
            angle=d_angle,
            d_angle=dd_angle,
            torque=jnp.zeros_like(state.torque),
            effector_force=jnp.zeros_like(state.effector_force),
        )

    def init_system_state(self, *, key: PRNGKeyArray) -> TwoLinkArmState:
        """Initialize arm at rest in default configuration."""
        return TwoLinkArmState(
            angle=jnp.zeros(2),
            d_angle=jnp.zeros(2),
            torque=jnp.zeros(2),
        )

    def extract_outputs(self, state: TwoLinkArmState) -> dict[str, PyTree]:
        """Extract effector and joint states from arm state."""
        joints = self.forward_kinematics(state)
        return {
            "effector": distal_state(joints),
            "joints": joints,
        }

    def _get_zero_input(self) -> Array:
        """Zero torque input."""
        return jnp.zeros(2)

    @property
    def input_size(self) -> int:
        """Number of input dimensions (2 joint torques)."""
        return 2

    def __call__(
        self,
        inputs: dict[str, PyTree],
        state,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, PyTree], State]:
        """Execute integration step with torque input."""
        torque = inputs.get("torque", self._get_zero_input())
        modified_inputs = {"input": torque}
        return super().__call__(modified_inputs, state, key=key)

    # === Kinematics Methods ===

    def forward_kinematics(self, state: TwoLinkArmState) -> CartesianState:
        """Compute Cartesian state of joints from configuration state.

        Args:
            state: Arm configuration state.

        Returns:
            Cartesian state with positions and velocities of both joints.
        """
        xy_pos, xy_vel = link_kinematics(self.params.l, state.angle, state.d_angle)

        return CartesianState(
            pos=xy_pos,
            vel=xy_vel,
            force=jnp.zeros_like(xy_vel),
        )

    def effector(self, state: TwoLinkArmState) -> CartesianState:
        """Compute Cartesian state of the end effector.

        Args:
            state: Arm configuration state.

        Returns:
            Cartesian state of the end effector (distal joint).
        """
        return distal_state(self.forward_kinematics(state))

    def effector_jac(self, angle: Array) -> Array:
        """Compute Jacobian of effector position w.r.t. joint angles.

        Args:
            angle: Joint angles [2].

        Returns:
            Jacobian matrix [2, 2].
        """
        return effector_jacobian(self.params.l, angle)

    def effector_force_to_torques(
        self,
        angle: Array,
        effector_force: Array,
    ) -> Array:
        """Convert effector force to joint torques via Jacobian transpose.

        Args:
            angle: Current joint angles.
            effector_force: Force on end effector [2].

        Returns:
            Joint torques [2].
        """
        return self.effector_jac(angle).T @ effector_force

    def inverse_kinematics(self, effector_state: CartesianState) -> TwoLinkArmState:
        """Compute joint configuration from effector Cartesian state.

        Uses the "elbow down" / "righty" solution.

        Args:
            effector_state: Desired end effector state.

        Returns:
            Arm configuration state.
        """
        import numpy as np

        pos = effector_state.pos
        l = self.params.l

        lsq = l**2
        lsqpm = (lsq[0] - lsq[1], lsq[0] + lsq[1])
        dsq = jnp.sum(pos**2)

        alpha = jnp.arccos((lsqpm[0] + dsq) / (2 * l[0] * jnp.sqrt(dsq)))
        gamma = jnp.arctan2(pos[1], pos[0])
        theta0 = gamma - alpha

        beta = jnp.arccos((lsqpm[1] - dsq) / (2 * l[0] * l[1]))
        theta1 = np.pi - beta

        angle = jnp.stack([theta0, theta1], axis=-1)

        d_angle = jnp.linalg.solve(
            self.effector_jac(angle),
            effector_state.vel,
        )

        force = effector_state.force
        return TwoLinkArmState(
            angle=angle,
            d_angle=d_angle,
            torque=jnp.zeros_like(angle),
            effector_force=jnp.zeros_like(angle) if force is None else force,
        )

    # === Energy Methods ===

    def compute_kinetic_energy(self, state: TwoLinkArmState) -> Array:
        """Compute total kinetic energy of the arm.

        Uses the inertia matrix for accurate energy computation.

        Args:
            state: Arm configuration state.

        Returns:
            Total kinetic energy.
        """
        cs1 = jnp.cos(state.angle[1])
        tmp = self._a[2] + self._a[1] * cs1
        inertia_mat = jnp.array([
            [self._a[0] + 2 * self._a[1] * cs1, tmp],
            [tmp, self._a[2]],
        ])
        return 0.5 * state.d_angle @ inertia_mat @ state.d_angle

    def compute_potential_energy(
        self,
        state: TwoLinkArmState,
        g: float = 9.81,
    ) -> Array:
        """Compute gravitational potential energy (if arm is vertical).

        Assumes y-axis is vertical with positive = up.

        Args:
            state: Arm configuration state.
            g: Gravitational acceleration [m/s^2].

        Returns:
            Total potential energy.
        """
        # Heights of segment centers of mass
        angle_sum = jnp.cumsum(state.angle)
        h1 = self.params.s[0] * jnp.sin(state.angle[0])
        h2 = (
            self.params.l[0] * jnp.sin(state.angle[0])
            + self.params.s[1] * jnp.sin(angle_sum[1])
        )
        return g * (self.params.m[0] * h1 + self.params.m[1] * h2)
