"""


:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0, see LICENSE for details.
"""

from functools import cached_property
from typing import Optional

import equinox as eqx
from equinox import field
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray, Scalar
import numpy as np

from feedbax.mechanics.geometry import corners_2d
from feedbax.mechanics.skeleton import AbstractSkeleton, AbstractSkeletonState
from feedbax.mechanics.skeleton._planar import distal_state, effector_jacobian, link_kinematics
from feedbax.runtime.state import CartesianState, StateBounds




class TwoLinkArmState(AbstractSkeletonState):
    """The configuration state of a 2D arm with two rotational joints.

    Attributes:
        angle: The joint angles.
        d_angle: The angular velocities.
        torque: Joint torque held independently of the controller input.
        effector_force: World-space force held at the distal endpoint.
    """

    angle: Float[Array, "... links=2"] = field(default_factory=lambda: jnp.zeros(2))
    d_angle: Float[Array, "... links=2"] = field(default_factory=lambda: jnp.zeros(2))
    torque: Float[Array, "... links=2"] = field(default_factory=lambda: jnp.zeros(2))
    effector_force: Float[Array, "... 2"] = field(default_factory=lambda: jnp.zeros(2))


class TwoLinkArm(AbstractSkeleton[TwoLinkArmState]):
    """Model of a 2D arm with two straight rigid segments, and two rotational joints.

    Attributes:
        l: The lengths of the arm segments. Units: $[\\mathrm{L}]$.
        m: The masses of the segments. Units: $[\\mathrm{M}]$.
        I: The moments of inertia of the segments.
            Units: $[\\mathrm{M}\\cdot \\mathrm{L}^2]$.
        s: The distance from the joint center to the segment center of mass
            for each segment. Units: $[\\mathrm{L}]$.
        B: The joint friction matrix.
            Units: $[\\mathrm{M}\\cdot \\mathrm{L}^2\\cdot \\mathrm{T}^{-1}]$.
    """

    l: Float[Array, "links=2"] = field(converter=jnp.asarray)
    m: Float[Array, "links=2"] = field(converter=jnp.asarray)
    I: Float[Array, "links=2"] = field(converter=jnp.asarray)
    s: Float[Array, "links=2"] = field(converter=jnp.asarray)
    B: Float[Array, "2 2"] = field(converter=jnp.asarray)
    inertia_gain: float

    def __init__(
        self,
        l=(0.30, 0.33),  # [m]
        m=(1.4, 1.0),  # [kg]
        I=(0.025, 0.045),  # [kg m^2]
        s=(0.11, 0.16),  # [m]
        B=((0.05, 0.025), (0.025, 0.05)),  # [kg m^2 s^-1]
    ):
        self.l = l
        self.m = m
        self.I = I
        self.s = s
        self.B = B
        self.inertia_gain = 1.0

        #! initialize cached properties used by JAX operations
        # otherwise their initialization is a side effect
        self._a

    def to_params(self) -> dict[str, list[float]]:
        """Return the authored GraphSpec parameters represented by this arm."""
        return {"link_lengths": [float(value) for value in self.l]}

    @jax.named_scope("fbx.TwoLinkArm")
    def vector_field(
        self,
        t: Scalar,
        state: TwoLinkArmState,
        input: Array,
    ) -> TwoLinkArmState:
        """Return the time derivatives of the arm's configuration state.

        Arguments:
            t: The time. Not used.
            state: The current state of the arm.
            input: The torques on the joints.
        """
        angle, d_angle = state.angle, state.d_angle

        # centripetal and coriolis torques
        c_vec = (
            jnp.array((-d_angle[1] * (2 * d_angle[0] + d_angle[1]), d_angle[0] ** 2))
            * self._a[1]
            * jnp.sin(angle[1])
        )

        # inertia matrix that maps torques -> angular accelerations
        cs1 = jnp.cos(angle[1])
        tmp = self._a[2] + self._a[1] * cs1
        inertia_mat = jnp.array(
            (
                (self._a[0] + 2 * self._a[1] * cs1, tmp),
                (tmp, self._a[2] * jnp.ones_like(cs1)),
            )
        )

        load_torque = self.effector_force_to_torques(angle, state.effector_force)
        net_torque = state.torque + input + load_torque - c_vec - self.B @ d_angle
        dd_angle = jnp.linalg.solve(inertia_mat, net_torque)

        return TwoLinkArmState(
            angle=d_angle,
            d_angle=dd_angle,
            torque=jnp.zeros_like(state.torque),
            effector_force=jnp.zeros_like(state.effector_force),
        )

    def init(
        self,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> TwoLinkArmState:
        """Return a default state for the arm."""
        return TwoLinkArmState()

    @cached_property
    def _a(self):
        # this is a cached_property to avoid polluting the module's fields with private attributes
        return (
            # + m[1]*s[1]**2 + m[0]*s[0]**2
            self.I[0] + self.I[1] + self.m[1] * self.l[0] ** 2,
            #
            self.m[1] * self.l[0] * self.s[1],
            # + m[1] * s[1] ** 2
            self.I[1],
        )

    @cached_property
    def _lsqpm(self):
        lsq = self.l**2
        return (lsq[0] - lsq[1], lsq[0] + lsq[1])

    @property
    def input_size(self) -> int:
        """Number of input variables—in this case, joint torques."""
        return self.n_links  # per-joint torque

    # @property
    # def state_size(self) -> int:
    #     """Number of configuration state variables."""
    #     return self.n_links * 2  # per-joint angle and angular velocity

    @property
    def n_links(self) -> int:
        """Number of arm segments."""
        return 2

    @jax.named_scope("fbx.TwoLinkArm.inverse_kinematics")
    def inverse_kinematics(self, effector_state: CartesianState) -> TwoLinkArmState:
        """Return the configuration state of the arm, given the Cartesian state
        of the end effector.

        !!! Note ""
            There are two possible solutions to the inverse kinematics problem,
            for a two-link arm. This method returns the "elbow down" or
            "righty" solution.

        Arguments:
            effector_state: The state of the end effector.
        """
        pos = effector_state.pos
        l, lsqpm = self.l, self._lsqpm
        dsq = jnp.sum(pos**2)
        # No solution exists if `dsq` is outside of:
        #   `(l[0] - l[1], l[0] + l[1])`

        alpha = jnp.arccos((lsqpm[0] + dsq) / (2 * l[0] * jnp.sqrt(dsq)))
        gamma = jnp.arctan2(pos[1], pos[0])
        theta0 = gamma - alpha

        beta = jnp.arccos((lsqpm[1] - dsq) / (2 * l[0] * l[1]))
        theta1 = np.pi - beta
        # The "elbow up" or "lefty" solution is given by:
        #   `theta0 = gamma + alpha`
        #   `theta1 = beta - pi`

        angle = jnp.stack([theta0, theta1], axis=-1)

        d_angle = jnp.linalg.inv(self.effector_jac(angle)) @ effector_state.vel
        force = effector_state.force
        return TwoLinkArmState(
            angle=angle,
            d_angle=d_angle,
            torque=jnp.zeros_like(angle),
            effector_force=jnp.zeros_like(angle) if force is None else force,
        )

    def update_state_given_effector_force(
        self,
        effector_force: Array,
        state: TwoLinkArmState,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> TwoLinkArmState:
        """Replace the held world-space endpoint force. Pass zero to clear it."""
        force = jnp.asarray(effector_force, dtype=state.effector_force.dtype)
        return eqx.tree_at(lambda state: state.effector_force, state, force)

    def effector_force_to_torques(
        self,
        angle: Float[Array, "links=2"],
        effector_force: Float[Array, "ndim=2"],
    ) -> Float[Array, "links=2"]:
        """Return the joint torques induced by a force on the end effector.

        Arguments:
            angle: The joint angles.
            effector_force: The force on the end effector.
        """
        torque = self.effector_jac(angle).T @ effector_force
        return torque

    def effector_jac(
        self, angle: Float[Array, "links=2"]
    ) -> Float[Array, "ndim=2 links=2"]:
        """Return the Jacobian of end effector position with respect to joint
        angles.

        Arguments:
            angle: The joint angles.
        """
        return effector_jacobian(self.l, angle)

    @jax.named_scope("fbx.TwoLinkArm.forward_kinematics")
    def forward_kinematics(self, state: TwoLinkArmState) -> CartesianState:
        """Return the Cartesian state of the joints and end effector, given
        the arm's configuration state.

        Arguments:
            state: The configuration state of the arm.
        """
        xy_pos, xy_vel = link_kinematics(self.l, state.angle, state.d_angle)

        # The observed Cartesian state leaves the force channel free for the next step.
        return CartesianState(
            pos=xy_pos,
            vel=xy_vel,
            force=jnp.zeros_like(xy_vel),
        )

    def effector(self, state: TwoLinkArmState) -> CartesianState:
        """Return the Cartesian state of the endpoint of the arm.

        Arguments:
            state: The configuration state of the arm.
        """
        return distal_state(self.forward_kinematics(state))

    @property
    def bounds(self) -> StateBounds[TwoLinkArmState]:
        """Suggested bounds on the arm state.

        !!! ref "Source"
            Joint angle limits adopted from [MotorNet](https://github.com/OlivierCodol/MotorNet/blob/9a56e5670d31b06fd0d81932e9bc6a8b1e46ec4b/motornet/skeleton.py#L336).
        """
        return StateBounds(
            low=TwoLinkArmState(
                angle=jnp.array([0.0, 0.0]),
                d_angle=None,
                torque=None,
                effector_force=None,
            ),
            high=TwoLinkArmState(
                angle=jnp.deg2rad(jnp.array([140.0, 160.0])),
                d_angle=None,
                torque=None,
                effector_force=None,
            ),
        )

    def workspace_test(
        self,
        workspace: Float[Array, "bounds=2 xy=2"],
    ) -> bool:
        """Tests whether a rectangular workspace is reachable by the arm.

        !!! Note ""
            Assumes the arm's joint angles are not bounded.

        Arguments:
            workspace: The bounds of the workspace to test.
        """
        r = sum(self.l)
        lengths = jnp.sum(corners_2d(workspace) ** 2, axis=0) ** 0.5
        if jnp.any(lengths > r):
            return False
        return True
